"""
Template fit: per window, σ-weighted least squares of m(λ) = P(x) · T(λ / D(v)) over RV (and optionally vsini).

T is the template broadened to the data resolution, D the relativistic Doppler factor and P a polynomial of
degree `poly_degree` in the scaled wavelength, solved exactly for every trial (variable projection), so only v
(and vsini) are searched: a coarse RV scan, then bounded refinement. Errors come from the χ² curvature
(covariance 2H⁻¹) scaled by √max(1, χ²/ν), or from a pixel bootstrap. `measure_resolution` profiles χ² over R.
"""
import copy
import logging
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .. import broadening
from ..constants import C_KMS
from ..numerics import parabola_vertex
from ..types import MethodResult, Region, Spectrum
from . import _common as C

__all__ = ['measure', 'DEFAULTS', 'measure_resolution', 'estimate_resolution', 'fit_window']

log = logging.getLogger(__name__)

METHOD = 'template_fit'
DEFAULTS: dict[str, Any] = C.TEMPLATE_DEFAULTS | dict(  # rv_step default: half the native pixel step
    poly_degree=3,
    fit_vsini=False, vsini=0., vsini_max=150., epsilon=0.6,
    bootstrap=0,  # number of bootstrap resamples (0: covariance errors)
)


@dataclass
class _Data:
    """Window data on its own pixels (no resampling: pixel noise stays independent)"""
    wave: np.ndarray
    flux: np.ndarray
    w: np.ndarray  # 1/σ²
    x: np.ndarray  # scaled wavelength in [−1, 1]
    sigma_known: bool


def _data(spectrum: Spectrum, window: Region, masks: Sequence[Region]) -> _Data:
    sel = (spectrum.wave >= window.wmin) & (spectrum.wave <= window.wmax) & np.isfinite(spectrum.flux)
    sel &= ~C.in_any(spectrum.wave, [(m.wmin, m.wmax) for m in masks])
    sigma_known = bool(np.any(np.isfinite(spectrum.sigma[sel]) & (spectrum.sigma[sel] > 0)))
    if sigma_known:
        sel &= np.isfinite(spectrum.sigma) & (spectrum.sigma > 0)
    wave, flux = spectrum.wave[sel], spectrum.flux[sel]
    w = 1. / spectrum.sigma[sel] ** 2 if sigma_known else np.ones(sel.sum())
    x = (2. * (wave - window.wmin) / (window.wmax - window.wmin) - 1.) if len(wave) else wave
    return _Data(wave, flux, w, x, sigma_known)


class _Model:
    """Profiled χ² for one window and template, with the vsini broadening cached"""

    def __init__(self, data: _Data, tpl: C.Template, degree: int, epsilon: float):
        self.data, self.tpl, self.degree, self.epsilon = data, tpl, degree, epsilon
        self.vander = np.vander(data.x, degree + 1, increasing=True)
        self.sqw = np.sqrt(data.w)
        self._vsini, self._tpl_rot = 0., tpl

    def template(self, vsini: float) -> C.Template:
        if vsini <= 0:
            return self.tpl
        if vsini != self._vsini:
            flux = broadening.broaden_loglam(self.tpl.flux, self.tpl.dlnw * C_KMS, vsini=vsini, epsilon=self.epsilon)
            self._tpl_rot = C.Template(self.tpl.lnw0, self.tpl.dlnw, flux, self.tpl.label, self.tpl.params)
            self._vsini = vsini
        return self._tpl_rot

    def solve(self, v: float, vsini: float = 0., idx: np.ndarray | None = None):
        """χ², polynomial coefficients and model at (v, vsini), optionally on a pixel subset"""
        d = self.data
        sl = slice(None) if idx is None else idx
        t = self.template(vsini).at(d.wave, v)[sl]  # on all pixels (pixel edges), then the subset
        if not np.all(np.isfinite(t)):
            return np.inf, None, None
        a = self.vander[sl] * t[:, None]
        sw = self.sqw[sl]
        coef, *_ = np.linalg.lstsq(a * sw[:, None], d.flux[sl] * sw, rcond=None)
        model = a @ coef
        return float(np.sum(d.w[sl] * (d.flux[sl] - model) ** 2)), coef, model

    def chi2(self, v: float, vsini: float = 0., idx=None) -> float:
        return self.solve(v, vsini, idx)[0]


def fit_window(model: _Model, rv_grid: np.ndarray, *, fit_vsini: bool, vsini0: float, vsini_max: float,
               idx: np.ndarray | None = None, v_start: float | None = None) -> dict:
    """
    Fit one window: coarse RV scan (unless v_start is given), then bounded refinement of v (and vsini); returns
    'rv', 'vsini', 'chi2', 'edge', 'ok' and 'scan' (the coarse χ²)
    """
    step = rv_grid[1] - rv_grid[0]
    scan = None
    if v_start is None:
        scan = np.array([model.chi2(v, vsini0, idx) for v in rv_grid])
        if not np.isfinite(scan).any():
            return dict(rv=np.nan, vsini=np.nan, chi2=np.nan, edge=False, ok=False, scan=scan)
        i = int(np.nanargmin(scan))
        v_start = float(rv_grid[i])
        edge = i == 0 or i == len(rv_grid) - 1
    else:
        edge = False
    lo, hi = v_start - 2 * step, v_start + 2 * step
    if not fit_vsini:
        r = minimize_scalar(lambda v: model.chi2(v, vsini0, idx), bounds=(lo, hi), method='bounded',
                            options=dict(xatol=1e-4 * step))
        v, s, chi2 = float(r.x), vsini0, float(r.fun)
        edge |= abs(v - lo) < 1e-3 * step or abs(v - hi) < 1e-3 * step
        ok = bool(np.isfinite(chi2))
    else:
        best = None
        for s0 in (max(vsini0, 1.), 10., 30., 60.):  # vsini is multimodal at low R: a few starts
            if s0 > vsini_max:
                continue
            r = minimize(lambda p: model.chi2(p[0], p[1], idx), x0=(v_start, s0), method='L-BFGS-B',
                         bounds=[(lo, hi), (0., vsini_max)])
            if best is None or r.fun < best.fun:
                best = r
        v, s, chi2 = float(best.x[0]), float(best.x[1]), float(best.fun)
        ok = bool(np.isfinite(chi2))
    return dict(rv=v, vsini=s, chi2=chi2, edge=bool(edge), ok=ok, scan=scan)


def _hessian_errors(model: _Model, fit: dict, fit_vsini: bool, resolution: float) -> np.ndarray:
    """1σ errors on (v[, vsini]) from the χ² curvature: covariance = 2 H⁻¹"""
    hv = 0.02 * C_KMS / resolution
    if not fit_vsini:
        f0 = model.chi2(fit['rv'], fit['vsini'])
        h = (model.chi2(fit['rv'] + hv, fit['vsini']) - 2 * f0 + model.chi2(fit['rv'] - hv, fit['vsini'])) / hv ** 2
        return np.array([np.sqrt(2. / h) if h > 0 else np.nan])
    hs = max(0.5, 0.05 * fit['vsini'])
    p0 = np.array([fit['rv'], max(fit['vsini'], hs * 1.01)])
    steps = np.array([hv, hs])

    def f(p):
        return model.chi2(p[0], p[1])

    hess = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            ei, ej = np.eye(2)[i] * steps[i], np.eye(2)[j] * steps[j]
            num = f(p0 + ei + ej) - f(p0 + ei - ej) - f(p0 - ei + ej) + f(p0 - ei - ej)
            hess[i, j] = num / (4 * steps[i] * steps[j])
    try:
        cov = 2. * np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        return np.array([np.nan, np.nan])
    diag = np.diag(cov)
    return np.where(diag > 0, np.sqrt(np.abs(diag)), np.nan)


def _bootstrap_chunk(args) -> list[float]:
    seed, n, model, fit, opts, rv_grid = args
    rng = np.random.default_rng(seed)
    npix = len(model.data.wave)
    out = []
    for _ in range(n):
        idx = np.sort(rng.integers(0, npix, npix))
        f = fit_window(model, rv_grid, fit_vsini=opts['fit_vsini'], vsini0=fit['vsini'], vsini_max=opts['vsini_max'],
                       idx=idx, v_start=fit['rv'])
        out.append(f['rv'] if f['ok'] else np.nan)
    return out


def _score_candidate(cand: C.Candidate, datas, resolution, wave_range, opts, rv_grid) -> float:
    tpl = C.build_template(cand, resolution, wave_range, vsini=opts['vsini'], pad_kms=opts['pad_kms'])
    total = 0.
    for d in datas:
        f = fit_window(_Model(d, tpl, opts['poly_degree'], opts['epsilon']), rv_grid, fit_vsini=False,
                       vsini0=0., vsini_max=opts['vsini_max'])
        if not f['ok']:
            return np.inf
        total += f['chi2']
    return total


def _prepare(spectrum, regions, opts, resolution=None):
    """Window data with enough usable pixels; windows are segmented at `resolution` (None: as given)"""
    windows, masks = C.split_regions(regions)
    if not windows:
        raise ValueError('no template-fit windows among the regions')
    if resolution is not None:
        windows = C.segment(windows, resolution, opts['segment_res'])
    datas, kept = [], []
    for win in windows:
        d = _data(spectrum, win, masks)
        if len(d.wave) < opts['poly_degree'] + 10:
            log.warning('template_fit: skipping %s (%d usable pixels)', win.name, len(d.wave))
            continue
        datas.append(d)
        kept.append(win)
    return datas, kept


def _rv_grid(spectrum, opts):
    dv = float(np.median(np.diff(np.log(spectrum.wave)))) * C_KMS
    step = opts['rv_step'] or dv / 2.
    return np.arange(opts['rv_min'], opts['rv_max'] + 1e-9, step)


def measure(spectrum: Spectrum, regions: Sequence[Region], cfg: Any, *, executor: Executor | None = None,
            rng: np.random.Generator) -> MethodResult:
    """
    Template-fit RVs, one per window (every non-telluric, non-continuum region; telluric ones are masked),
    combined by random effects; options in cfg.methods.options['template_fit'] (`DEFAULTS`). Diagnostics hold
    the χ² scan, best-fit model, polynomial, vsini and errors.
    """
    opts = C.options(cfg, METHOD, DEFAULTS)
    resolution = C.resolution_of(spectrum, cfg, opts)
    cands = C.candidates(cfg, opts)
    datas, kept = _prepare(spectrum, C.covered_regions(METHOD, regions, cands), opts, resolution)
    if not datas:
        raise ValueError(f'no template-fit window has {opts["poly_degree"] + 10} usable pixels')
    rv_grid = _rv_grid(spectrum, opts)
    wave_range = (min(d.wave[0] for d in datas), max(d.wave[-1] for d in datas))
    score = partial(_score_candidate, datas=datas, resolution=resolution, wave_range=wave_range, opts=opts,
                    rv_grid=rv_grid)
    tpl, flag_grid, tdiag = C.choose_template(METHOD, cands, score, cfg, opts, resolution,
                                              wave_range, sum(len(d.wave) for d in datas), executor)
    n_chunks = opts['n_chunks']
    seeds = C.child_seeds(rng, len(datas) * n_chunks)
    measurements = []
    for k, (d, window) in enumerate(zip(datas, kept, strict=True)):
        model = _Model(d, tpl, opts['poly_degree'], opts['epsilon'])
        fit = fit_window(model, rv_grid, fit_vsini=opts['fit_vsini'], vsini0=opts['vsini'],
                         vsini_max=opts['vsini_max'])
        flag = flag_grid | (C.FLAG_EDGE if fit['edge'] else 0) | (0 if fit['ok'] else C.FLAG_FIT_FAILED)
        nfree = opts['poly_degree'] + 1 + 1 + int(opts['fit_vsini'])
        dof = max(len(d.wave) - nfree, 1)
        chi2_red = fit['chi2'] / dof if fit['ok'] else np.nan
        errs = _hessian_errors(model, fit, opts['fit_vsini'], resolution) if fit['ok'] else np.array([np.nan] * 2)
        scale = np.sqrt(max(1., chi2_red)) if np.isfinite(chi2_red) else np.nan  # never deflated (as the CCF)
        cov_err = float(errs[0] * scale)
        boot_std, boot = np.nan, np.array([])
        if opts['bootstrap'] > 0 and fit['ok']:
            boot = C.run_chunks(_bootstrap_chunk, opts['bootstrap'], seeds[k * n_chunks:(k + 1) * n_chunks],
                                (model, fit, opts, rv_grid), executor)
            if np.isfinite(boot).sum() >= 5:
                boot_std = float(np.nanstd(boot, ddof=1))
        err = boot_std if opts['bootstrap'] > 0 else cov_err
        _, coef, best_model = model.solve(fit['rv'], fit['vsini']) if fit['ok'] else (None, None, None)
        diag = dict(tdiag, chi2=fit['chi2'], dof=dof, chi2_red=chi2_red, err_cov=cov_err, err_bootstrap=boot_std,
                    bootstrap_rvs=boot, vsini=fit['vsini'],
                    vsini_err=float(errs[1] * scale) if opts['fit_vsini'] and len(errs) > 1 else np.nan,
                    rv_grid=rv_grid, chi2_scan=fit['scan'], poly_coef=coef, resolution=resolution,
                    wave=d.wave, data=d.flux, model=best_model, weights=d.w)
        measurements.append(C.window_measurement(METHOD, window, fit['rv'], err, flag, chi2_red, d.sigma_known,
                                                 diag))
    result = C.finish(METHOD, measurements, opts['outlier_k'])
    log.info('template_fit: %.2f ± %.2f km/s (template %s, %d windows)', result.rv, result.rv_err, tpl.label,
             result.n_regions)
    return result


def measure_resolution(spectrum: Spectrum, regions: Sequence[Region], cfg: Any,
                       r_grid: Sequence[float] | None = None) -> dict:
    """
    Estimate the resolving power by profiling the template-fit χ² over R (template as for `measure`)

    For each trial R (default 20 log-spaced values from 0.5 to 2 times the known R) the template is broadened and
    refitted (RV free); a parabola through the lowest χ² values in ln R gives the best R and its 1σ range
    (Δχ² = 1 after scaling by the reduced χ²). Returns 'resolution', 'resolution_err', 'r_grid' and 'chi2'.
    """
    opts = C.options(cfg, METHOD, DEFAULTS)
    if r_grid is None:
        r0 = C.resolution_of(spectrum, cfg, opts, estimate=False)
        r_grid = np.geomspace(0.5 * r0, 2. * r0, 20)
    r_grid = np.asarray(r_grid, float)
    datas, _ = _prepare(spectrum, regions, opts)
    rv_grid = _rv_grid(spectrum, opts)
    wave_range = (min(d.wave[0] for d in datas), max(d.wave[-1] for d in datas))
    cand = C.candidates(cfg, opts)[0]
    chi2 = []
    for r in r_grid:
        tpl = C.build_template(cand, float(r), wave_range, pad_kms=opts['pad_kms'])
        chi2.append(sum(fit_window(_Model(d, tpl, opts['poly_degree'], opts['epsilon']), rv_grid, fit_vsini=False,
                                   vsini0=0., vsini_max=0.)['chi2'] for d in datas))
    chi2 = np.array(chi2)
    i = int(np.argmin(chi2))
    sl = slice(max(i - 2, 0), min(i + 3, len(r_grid)))
    x0, _, curv = parabola_vertex(np.log(r_grid[sl]), chi2[sl])
    npix = sum(len(d.wave) for d in datas)
    red = chi2[i] / max(npix - len(datas) * (opts['poly_degree'] + 2), 1)
    if curv > 0:
        best = float(np.exp(x0))
        err = best * np.sqrt(red / (curv / 2))  # Δχ² = χ²/ν about the minimum in ln R
    else:
        best, err = float(r_grid[i]), np.nan
    return dict(resolution=best, resolution_err=float(err), r_grid=r_grid, chi2=chi2)


def estimate_resolution(spectrum: Spectrum, cfg: Any, regions: Sequence[Region] | None = None) -> dict:
    """
    Estimate the resolving power of a spectrum of unknown (or doubtful) resolution from a template-fit LSF

    A coarse χ² profile over R = 300–100 000 (16 log-spaced values) is refined with `measure_resolution` over a
    factor of 1.5 either side of its minimum; the spectrum's own resolution and any configured one are ignored.
    Regions default to `regions.select(..., methods=['template_fit'])`. Returns the refinement's
    `measure_resolution` result plus 'method'.
    """
    from dataclasses import replace as _replace
    spec = _replace(spectrum, resolution=None)
    if regions is None:
        from .. import regions as _regions
        regions = _regions.select(spectrum, teff=cfg.templates.teff_prior, methods=[METHOD]).get(METHOD, [])
    no_r = copy.deepcopy(cfg)
    no_r.input.resolution = None
    no_r.methods.options.get(METHOD, {}).pop('resolution', None)
    coarse = measure_resolution(spec, regions, no_r, r_grid=np.geomspace(300., 1e5, 16))
    r_best = coarse['r_grid'][int(np.argmin(coarse['chi2']))]
    fine = measure_resolution(spec, regions, no_r, r_grid=np.geomspace(r_best / 1.5, r_best * 1.5, 15))
    fine['method'] = METHOD
    return fine
