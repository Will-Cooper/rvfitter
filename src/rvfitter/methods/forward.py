"""
Forward-model MCMC: per window, the posterior of the RV and the atmosphere under a pixel-level model

    m(λ) = P(x) · [G(θ) ⊗ rot(vsini) ⊗ LSF(R)](λ / D(v)) · T(λ)^α

G is the template grid, multilinear in its free axes (Teff, log g, [M/H], ... over the box spanned by the
`_common.candidates` points; holes excluded), LSF-broadened at R, rotated (Gray kernel) and Doppler-shifted, then
averaged over each pixel. P is a polynomial of degree `poly_degree` per window and epoch; T the telluric
transmission at airmass 1 and the default PWV, raised to a free α ∈ [0, 5] (option `telluric`, ignored for
telluric-corrected input). P is marginalised analytically (flat prior), and so is a free error scale s (σ → sσ per
window and epoch; Jeffreys prior), so ln L = −(N − k)/2 · ln χ²_min − ½ ln det(AᵀWA).

Priors are uniform: v within ±500 km/s of the start (template_fit's RV), vsini in [0, vsini_max], the grid axes
within the box. The posterior mode is found (Nelder–Mead), pixels beyond `clip_k` σ of it are clipped once and the
mode refitted, then emcee samples. Converged means n_steps ≥ 50 τ (integrated autocorrelation time) and a mean
acceptance fraction within 0.15–0.8; otherwise the window is flagged not_converged. RV = posterior median, error =
half the 16–84 per cent range. `fit_epochs` shares Teff, log g, [M/H] and vsini over epochs, with one RV each.
"""
import logging
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .. import broadening
from ..calibration import telluric
from ..constants import C_KMS
from ..types import MethodResult, Region, Spectrum
from . import _common as C
from . import template_fit

__all__ = ['measure', 'fit_epochs', 'DEFAULTS']

log = logging.getLogger(__name__)

METHOD = 'forward_mcmc'
DEFAULTS: dict[str, Any] = C.TEMPLATE_DEFAULTS | dict(
    poly_degree=2, epsilon=0.6, vsini_max=100.,
    telluric=False,  # × T(λ)^α with α free, for input that is not telluric-corrected (needs the telluric model)
    n_walkers=32, n_burn=1000, n_steps=4000,
    clip_k=5.,  # one clip-and-refit pass at this many σ (0: off)
)
RV_HALFWIDTH = 500.  # RV prior half-width about the start [km/s]
ALPHA_MAX = 5.
N_TAU = 50
ACCEPTANCE = (0.15, 0.8)


@dataclass
class _Seg:
    """One window of one epoch: its pixels (template_fit's window data), the unclipped ones and T(λ) (or None)"""
    data: Any
    keep: np.ndarray
    tell: np.ndarray | None
    k: int  # polynomial terms

    @property
    def vander(self) -> np.ndarray:
        return np.vander(self.data.x[self.keep], self.k, increasing=True)


class _Window:
    """The posterior of one window over E epochs: θ = (v_1..v_E, vsini, free grid axes..., [α])"""

    def __init__(self, grid, points: np.ndarray, resolution: float, wave_range: tuple[float, float],
                 segs: list[_Seg], v0: Sequence[float], epsilon: float, vsini_max: float):
        self.grid, self.segs, self.epsilon = grid, segs, epsilon
        lo, hi = points.min(axis=0), points.max(axis=0)
        self.free = [i for i in range(len(grid.axes)) if hi[i] > lo[i]]
        self.fixed = lo
        self.flux = {}
        for p in points:
            s = grid.broadened(**dict(zip(grid.axes, p, strict=True)), resolution=resolution, wave_range=wave_range)
            self.flux[s.meta['files'][0]] = np.asarray(s.flux)
        lnw = np.log(s.wave)
        self.lnw0, self.dlnw = float(lnw[0]), float(lnw[1] - lnw[0])
        self.tell = segs[0].tell is not None
        n_ep = len(segs)
        self.names = ([f'rv_{e}' for e in range(n_ep)] + ['vsini'] + [grid.axes[i] for i in self.free]
                      + (['alpha'] if self.tell else []))
        self.lo = np.array([v - RV_HALFWIDTH for v in v0] + [0.] + [lo[i] for i in self.free]
                           + ([0.] if self.tell else []))
        self.hi = np.array([v + RV_HALFWIDTH for v in v0] + [vsini_max] + [hi[i] for i in self.free]
                           + ([ALPHA_MAX] if self.tell else []))
        self._vander = [s.vander for s in segs]

    def params(self, theta: np.ndarray) -> dict[str, float]:
        p = self.fixed.copy()
        p[self.free] = theta[len(self.segs) + 1:len(self.segs) + 1 + len(self.free)]
        return dict(zip(self.grid.axes, map(float, p), strict=True))

    def models(self, theta: np.ndarray) -> list[np.ndarray] | None:
        """The template × telluric at every pixel of each epoch (before P), or None outside the prior or grid"""
        if np.any(theta < self.lo) or np.any(theta > self.hi):
            return None
        corners, _ = self.grid.corners(self.params(theta))
        if corners is None:  # a hole
            return None
        flux = sum(w * self.flux[path.name] for path, w in corners)
        vsini = theta[len(self.segs)]
        if vsini > 0:
            flux = broadening.broaden_loglam(flux, self.dlnw * C_KMS, vsini=vsini, epsilon=self.epsilon)
        tpl = C.Template(self.lnw0, self.dlnw, flux, '')
        out = []
        for e, s in enumerate(self.segs):
            t = tpl.at(s.data.wave, theta[e])
            if self.tell:
                t = t * s.tell ** theta[-1]
            if not np.all(np.isfinite(t[s.keep])):
                return None
            out.append(t)
        return out

    def solve(self, theta: np.ndarray):
        """Per epoch (ln L, χ²_min, full model on every pixel, N − k), or None outside the prior"""
        ts = self.models(theta)
        if ts is None:
            return None
        out = []
        for s, t, vander in zip(self.segs, ts, self._vander, strict=True):
            d, keep = s.data, s.keep
            sw = np.sqrt(d.w[keep])
            a = vander * (t[keep] * sw)[:, None]
            b = d.flux[keep] * sw
            try:
                cf = cho_factor(a.T @ a)
            except np.linalg.LinAlgError:
                return None
            coef = cho_solve(cf, a.T @ b)
            chi2 = float(np.sum((b - a @ coef) ** 2))
            nu = int(keep.sum()) - vander.shape[1]
            lnl = -0.5 * nu * np.log(chi2) - np.sum(np.log(np.diag(cf[0])))
            full = np.vander(d.x, vander.shape[1], increasing=True) @ coef * t
            out.append((lnl, chi2, full, nu))
        return out

    def log_prob(self, theta: np.ndarray) -> float:
        r = self.solve(theta)
        return -np.inf if r is None else float(sum(x[0] for x in r))

    def mode(self, x0: np.ndarray) -> np.ndarray:
        r = minimize(lambda t: -self.log_prob(t), x0, method='Nelder-Mead',
                     bounds=list(zip(self.lo, self.hi, strict=True)),
                     options=dict(maxiter=400 * len(x0), xatol=1e-3, fatol=1e-3))
        return r.x if np.isfinite(r.fun) else x0

    def clip(self, theta: np.ndarray, k: float) -> int:
        """Drop pixels beyond k σ (σ scaled by √(χ²/ν)) of the model at θ; the number clipped"""
        n = 0
        for s, (_, chi2, full, nu) in zip(self.segs, self.solve(theta), strict=True):
            z = np.abs(s.data.flux - full) * np.sqrt(s.data.w) / np.sqrt(chi2 / nu)
            bad = s.keep & (z > k)
            s.keep &= ~bad
            n += int(bad.sum())
        self._vander = [s.vander for s in self.segs]
        return n


def _sample(fit: _Window, x: np.ndarray, rng: np.random.Generator, opts: dict) -> dict:
    try:
        import emcee
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError('forward_mcmc needs emcee: pip install "rvfitter[mcmc]"') from exc
    nw, ndim = opts['n_walkers'], len(x)
    width = fit.hi - fit.lo
    p0 = np.empty((nw, ndim))
    for j in range(nw):
        for _ in range(100):
            p = np.clip(x + 1e-3 * width * rng.standard_normal(ndim), fit.lo + 1e-6 * width, fit.hi - 1e-6 * width)
            if np.isfinite(fit.log_prob(p)):
                break
        else:
            raise ValueError(f'forward_mcmc: no valid walker start near {x}')
        p0[j] = p
    sampler = emcee.EnsembleSampler(nw, ndim, fit.log_prob)
    sampler.random_state = np.random.RandomState(np.random.MT19937(rng.integers(2 ** 63))).get_state()
    sampler.run_mcmc(p0, opts['n_burn'] + opts['n_steps'], progress=False)
    chain = sampler.get_chain(discard=opts['n_burn'])
    tau = emcee.autocorr.integrated_time(chain, tol=0)
    acc = float(np.mean(sampler.acceptance_fraction))
    converged = bool(opts['n_steps'] >= N_TAU * np.max(tau) and ACCEPTANCE[0] <= acc <= ACCEPTANCE[1])
    thin = max(1, int(0.5 * np.min(tau)))
    flat = sampler.get_chain(discard=opts['n_burn'], thin=thin, flat=True)
    return dict(samples=flat, tau=tau, acceptance=acc, converged=converged, thin=thin)


def _run_window(task: tuple) -> dict:
    """One window: build the model, mode, clip and refit, sample; picklable for process pools"""
    args, x0, seed, opts = task
    fit = _Window(*args)
    rng = np.random.default_rng(seed)
    x = fit.mode(np.clip(x0, fit.lo, fit.hi))
    n_clip = fit.clip(x, opts['clip_k']) if opts['clip_k'] else 0
    if n_clip:
        x = fit.mode(x)
    out = _sample(fit, x, rng, opts)
    out.update(names=fit.names, mode=x, n_clipped=n_clip, solved=fit.solve(x), params_mode=fit.params(x))
    return out


def _box(cfg: Any, opts: dict):
    """The grid and its candidate points (the prior box)"""
    cands = C.candidates(cfg, opts)
    grids = {c.grid.key if c.grid is not None else None for c in cands}
    if None in grids or len(grids) != 1:
        raise ValueError('forward_mcmc needs exactly one template grid (templates.grid), not given templates')
    return cands[0].grid, np.array([c.params for c in cands], float)


def fit_epochs(spectra: Sequence[Spectrum], regions: Sequence[Sequence[Region]], cfg: Any, *,
               executor: Executor | None = None, rng: np.random.Generator) -> list[MethodResult]:
    """
    One forward model per window shared by the epochs (one RV each; Teff, log g, [M/H], vsini shared); returns one
    MethodResult per epoch (its windows combined by random effects). Windows are matched by name; every epoch needs
    the same resolving power.
    """
    opts = C.options(cfg, METHOD, DEFAULTS)
    res = {C.resolution_of(s, cfg, opts) for s in spectra}
    if len(res) != 1:
        raise ValueError(f'forward_mcmc: the epochs have different resolving powers {sorted(res)}')
    resolution = res.pop()
    grid, points = _box(cfg, opts)
    starts = [template_fit.measure(s, r, cfg, executor=executor, rng=g)
              for s, r, g in zip(spectra, regions, rng.spawn(len(spectra)), strict=True)]
    if not all(np.isfinite(s.rv) for s in starts):
        raise ValueError('forward_mcmc: template_fit found no starting RV')
    tdiag = starts[0].measurements[0].diagnostics
    use_tell = opts['telluric'] and not cfg.input.telluric_corrected
    if opts['telluric'] and not use_tell:
        log.info('forward_mcmc: telluric-corrected input: no telluric term')

    windows: dict[str, list[tuple[int, Region, Any]]] = {}
    for e, (spec, regs) in enumerate(zip(spectra, regions, strict=True)):
        datas, kept = template_fit._prepare(spec, regs, opts, resolution)
        for d, win in zip(datas, kept, strict=True):
            windows.setdefault(win.name, []).append((e, win, d))
    if not windows:
        raise ValueError(f'no forward_mcmc window has {opts["poly_degree"] + 10} usable pixels')

    tasks = []
    for (name, members), seed in zip(windows.items(), C.child_seeds(rng, len(windows)), strict=True):
        segs = [_Seg(d, np.ones(len(d.wave), bool),
                     telluric.transmission(d.wave, 1., resolution=resolution) if use_tell else None,
                     opts['poly_degree'] + 1) for _, _, d in members]
        lo = min(d.wave[0] for *_, d in members) * broadening.doppler_factor(-opts['pad_kms'])
        hi = max(d.wave[-1] for *_, d in members) * broadening.doppler_factor(opts['pad_kms'])
        v0 = [starts[e].rv for e, *_ in members]
        p0 = tdiag['template_params']
        x0 = np.array(v0 + [min(10., opts['vsini_max'] / 2)]
                      + [p0[grid.axes[i]] for i in range(len(grid.axes)) if np.ptp(points[:, i]) > 0]
                      + ([1.] if use_tell else []))
        tasks.append(((grid, points, resolution, (lo, hi), segs, v0, opts['epsilon'], opts['vsini_max']),
                      x0, seed, opts))
        log.debug('forward_mcmc %s: %d epoch(s), start %s', name, len(members), x0)
    fits = C.pmap(_run_window, tasks, executor)

    per_epoch: list[list] = [[] for _ in spectra]
    for (name, members), fit in zip(windows.items(), fits, strict=True):
        q16, q50, q84 = np.percentile(fit['samples'], [16, 50, 84], axis=0)
        n_ep = len(members)
        shared = {k: (float(q50[i]), float(0.5 * (q84[i] - q16[i]))) for i, k in enumerate(fit['names'])
                  if i >= n_ep}
        flag = 0 if fit['converged'] else C.FLAG_EDGE
        if not fit['converged']:
            log.warning('forward_mcmc %s: not converged (τ_max %.0f steps for %d steps; acceptance %.2f)', name,
                        np.max(fit['tau']), opts['n_steps'], fit['acceptance'])
        for j, (e, win, d) in enumerate(members):
            _, chi2, model, nu = fit['solved'][j]
            nfree = len(fit['names']) - n_ep + 1
            chi2_red = chi2 / max(nu - nfree, 1)
            diag = dict(posterior=shared, vsini=shared['vsini'][0], vsini_err=shared['vsini'][1],
                        **{f'{k}_post': v[0] for k, v in shared.items() if k != 'vsini'},
                        **{f'{k}_post_err': v[1] for k, v in shared.items() if k != 'vsini'},
                        rv_quantiles=(float(q16[j]), float(q50[j]), float(q84[j])),
                        samples=fit['samples'], sample_names=fit['names'], tau=fit['tau'],
                        acceptance=fit['acceptance'], converged=fit['converged'], thin=fit['thin'],
                        n_clipped=fit['n_clipped'], error_scale=float(np.sqrt(chi2 / nu)), chi2=chi2,
                        chi2_red=chi2_red, mode=fit['mode'], template_params=fit['params_mode'],
                        resolution=resolution, n_epochs=n_ep, wave=d.wave, data=d.flux, model=model,
                        weights=d.w)
            per_epoch[e].append(C.window_measurement(METHOD, win, float(q50[j]), float(0.5 * (q84[j] - q16[j])),
                                                     flag, chi2_red, d.sigma_known, diag))
    out = [C.finish(METHOD, meas, opts['outlier_k']) for meas in per_epoch]
    for r in out:
        log.info('forward_mcmc: %.2f ± %.2f km/s (%d windows)', r.rv, r.rv_err, r.n_regions)
    return out


def measure(spectrum: Spectrum, regions: Sequence[Region], cfg: Any, *, executor: Executor | None = None,
            rng: np.random.Generator) -> MethodResult:
    """Forward-model MCMC RVs, one per window, combined by random effects (`fit_epochs` with one epoch)"""
    return fit_epochs([spectrum], [regions], cfg, executor=executor, rng=rng)[0]
