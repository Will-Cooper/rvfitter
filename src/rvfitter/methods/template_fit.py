"""
Direct template matching: σ-weighted least squares over RV (and optionally vsini) with a continuum polynomial.

In each window the model is m(λ) = P(x) · T(λ / D(v)), where T is the template broadened to the data resolution
(and by v sin i), D the relativistic Doppler factor and P a polynomial of degree `poly_degree` in the scaled
wavelength x ∈ [−1, 1]. P enters linearly and is solved exactly for every trial (v, v sin i) (variable
projection), so only the non-linear parameters are searched: first on a coarse RV grid, then by bounded
refinement.

Errors come from the curvature of the profiled χ² (covariance = 2H⁻¹), scaled by √max(1, χ²/ν) with ν the
degrees of freedom (as for the CCF), or optionally from a pixel bootstrap (seeded, optionally parallel).
Template selection by χ² over the grid candidates is as for the CCF.
`measure_resolution` profiles χ² over R to estimate the data resolution.
"""
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
from ..types import Measurement, MethodResult, Region, Spectrum
from . import _common as C

__all__ = ['measure', 'DEFAULTS', 'measure_resolution', 'estimate_resolution', 'fit_window']

log = logging.getLogger(__name__)

METHOD = 'template_fit'
DEFAULTS: dict[str, Any] = dict(
    resolution=None,
    rv_min=-500., rv_max=500.,
    rv_step=None,  # coarse scan step; default half the native pixel step
    poly_degree=3,
    fit_vsini=False, vsini=0., vsini_max=150., epsilon=0.6,
    bootstrap=0,  # number of bootstrap resamples (0: covariance errors)
    n_chunks=8,
    template=None, templates=None, teff_window=300.,
    region_kinds=None,
    segment_res=150.,  # split windows longer than this many resolution elements (0: never)
    outlier_k=5.,  # gross-outlier rejection among regions (0: off)
    pad_kms=1500.,
    telluric_model=False,  # multiply the model by a telluric transmission (calibration.telluric; unit 5)
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


def _scan(model: _Model, rv: np.ndarray, vsini: float, idx=None) -> np.ndarray:
    return np.array([model.chi2(v, vsini, idx) for v in rv])


def fit_window(model: _Model, rv_grid: np.ndarray, *, fit_vsini: bool, vsini0: float, vsini_max: float,
               idx: np.ndarray | None = None, v_start: float | None = None) -> dict:
    """
    Fit one window: coarse RV scan (unless v_start is given), then bounded refinement of v (and vsini)

    Returns
    -------
    fit
        'rv', 'vsini', 'chi2', 'edge', 'ok', 'scan' (the coarse χ²)
    """
    step = rv_grid[1] - rv_grid[0]
    scan = None
    if v_start is None:
        scan = _scan(model, rv_grid, vsini0, idx)
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


def _score_candidate(cand: C.Candidate, datas, windows, resolution, wave_range, opts, rv_grid) -> float:
    tpl = C.build_template(cand, resolution, wave_range, vsini=opts['vsini'], pad_kms=opts['pad_kms'])
    total = 0.
    for d in datas:
        f = fit_window(_Model(d, tpl, opts['poly_degree'], opts['epsilon']), rv_grid, fit_vsini=False,
                       vsini0=0., vsini_max=opts['vsini_max'])
        if not f['ok']:
            return np.inf
        total += f['chi2']
    return total


def _prepare(spectrum, regions, cfg, opts, resolution=None):
    windows, masks = C.split_regions(regions, opts['region_kinds'])
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


def measure(spectrum: Spectrum, regions: Sequence[Region], cfg: Any = None, *, executor: Executor | None = None,
            rng: np.random.Generator) -> MethodResult:
    """
    Template-fit RVs, one per window, combined with a random-effects model (combine.combine_regions)

    Parameters
    ----------
    spectrum
        Observed spectrum (vacuum)
    regions
        Windows (every non-telluric, non-continuum region by default) and telluric masks
    cfg
        Config or mapping; options under cfg.methods.options['template_fit'] (see `DEFAULTS`)
    executor
        Optional executor for the template scoring and bootstrap
    rng
        Random generator for the bootstrap

    Returns
    -------
    result
        The combined RV; diagnostics hold the χ² scan, best-fit model, polynomial, vsini and errors
    """
    opts = C.options(cfg, METHOD, DEFAULTS)
    if opts['telluric_model']:
        log.warning('template_fit: telluric_model is not implemented yet; fitting without it')
    resolution = C.resolution_of(spectrum, cfg, opts)
    datas, kept = _prepare(spectrum, regions, cfg, opts, resolution)
    if not datas:
        return MethodResult(METHOD, np.nan, np.nan, 0, [])
    rv_grid = _rv_grid(spectrum, opts)
    wave_range = (min(d.wave[0] for d in datas), max(d.wave[-1] for d in datas))
    cands = C.candidates(cfg, opts)
    if len(cands) > 1:
        scores = C.pmap(partial(_score_candidate, datas=datas, windows=kept, resolution=resolution,
                                wave_range=wave_range, opts=opts, rv_grid=rv_grid), cands, executor)
        best = cands[int(np.argmin(scores))]
        log.info('template_fit: template %s chosen by χ² from %d candidates', best.label, len(cands))
    else:
        best, scores = cands[0], [np.nan]
    tpl = C.build_template(best, resolution, wave_range, pad_kms=opts['pad_kms'])
    flag_grid = C.FLAG_OUTSIDE_GRID if tpl.params.get('outside_grid') else 0
    summary = C.template_summary(METHOD, cands, scores, best, sum(len(d.wave) for d in datas),
                                 C.section(cfg, 'templates'))
    flag_grid |= C.FLAG_PARAM_EDGE if summary['template_at_edge'] else 0
    summary_scalars = {k: v for k, v in summary.items() if k.startswith('template_')}
    seeds = C.child_seeds(rng, len(datas) * opts['n_chunks'])
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
            n_each = np.diff(np.linspace(0, opts['bootstrap'], opts['n_chunks'] + 1).astype(int))
            tasks = [(seeds[k * opts['n_chunks'] + j], int(n), model, fit, opts, rv_grid)
                     for j, n in enumerate(n_each) if n > 0]
            boot = np.concatenate([np.asarray(x, float) for x in C.pmap(_bootstrap_chunk, tasks, executor)])
            if np.isfinite(boot).sum() >= 5:
                boot_std = float(np.nanstd(boot, ddof=1))
        err = boot_std if opts['bootstrap'] > 0 else cov_err
        if not np.isfinite(err):
            flag |= C.FLAG_FIT_FAILED
        if np.isfinite(chi2_red) and chi2_red > 3. and d.sigma_known:
            flag |= C.FLAG_POOR_FIT
        _, coef, best_model = model.solve(fit['rv'], fit['vsini']) if fit['ok'] else (None, None, None)
        diag = dict(template_summary=summary, **summary_scalars,
                    chi2=fit['chi2'], dof=dof, chi2_red=chi2_red, err_cov=cov_err, err_bootstrap=boot_std,
                    bootstrap_rvs=boot, vsini=fit['vsini'],
                    vsini_err=float(errs[1] * scale) if opts['fit_vsini'] and len(errs) > 1 else np.nan,
                    rv_grid=rv_grid, chi2_scan=fit['scan'], poly_coef=coef, resolution=resolution,
                    template=tpl.label, template_params=tpl.params,
                    template_scores=dict(zip([c.label for c in cands], map(float, scores), strict=False)),
                    wave=d.wave, data=d.flux, model=best_model, weights=d.w)
        m = Measurement(METHOD, window.name, float(fit['rv']), float(err), int(flag), diag)
        log.debug('template_fit %s: %.2f ± %.2f km/s (χ²/ν %.2f, flag %d)', window.name, m.rv, m.rv_err, chi2_red,
                  flag)
        measurements.append(m)
    result = C.finish(METHOD, measurements, opts['outlier_k'])
    log.info('template_fit: %.2f ± %.2f km/s (template %s, %d windows)', result.rv, result.rv_err, tpl.label,
             result.n_regions)
    return result


def measure_resolution(spectrum: Spectrum, regions: Sequence[Region], cfg: Any = None,
                       r_grid: Sequence[float] | None = None) -> dict:
    """
    Estimate the resolving power by profiling the template-fit χ² over R

    For each R the template is broadened to R and refitted (RV free); a parabola through the lowest χ² values
    in ln R gives the best R and its 1σ range (Δχ² = 1 after scaling by the reduced χ²).

    Parameters
    ----------
    spectrum, regions, cfg
        As for `measure` (the template comes from the same options/config)
    r_grid
        Trial resolving powers (default: 20 values log-spaced between 0.5 and 2 times the nominal R, or
        1000–10000 if none is known)

    Returns
    -------
    result
        'resolution', 'resolution_err', 'r_grid', 'chi2'
    """
    opts = C.options(cfg, METHOD, DEFAULTS)
    try:
        r0 = C.resolution_of(spectrum, cfg, opts, estimate=False)
        r_grid = np.geomspace(0.5 * r0, 2. * r0, 20) if r_grid is None else np.asarray(r_grid, float)
    except ValueError:
        r_grid = np.geomspace(1000., 10000., 25) if r_grid is None else np.asarray(r_grid, float)
    datas, _ = _prepare(spectrum, regions, cfg, opts)
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
    x = np.log(r_grid[sl])
    a, b, _ = np.polyfit(x, chi2[sl], 2)
    npix = sum(len(d.wave) for d in datas)
    red = chi2[i] / max(npix - len(datas) * (opts['poly_degree'] + 2), 1)
    if a > 0:
        best = float(np.exp(-b / (2 * a)))
        err = best * np.sqrt(red / a)  # Δχ² = χ²/ν about the minimum in ln R
    else:
        best, err = float(r_grid[i]), np.nan
    return dict(resolution=best, resolution_err=float(err), r_grid=r_grid, chi2=chi2)


def estimate_resolution(spectrum: Spectrum, regions: Sequence[Region] | None = None, cfg: Any = None) -> dict:
    """
    Estimate the resolving power of a spectrum of unknown (or doubtful) resolution from a template-fit LSF

    A coarse χ² profile over R = 300–100 000 (16 log-spaced values) is refined with `measure_resolution` over
    a factor of 1.5 either side of its minimum. Templates come from the config (or the 'template' option of
    template_fit). Regions default to `regions.select(..., methods=['template_fit'])`.

    Parameters
    ----------
    spectrum
        The spectrum (its own `resolution` is ignored)
    regions
        Fitting windows and telluric masks
    cfg
        Config or mapping

    Returns
    -------
    result
        'resolution', 'resolution_err', 'r_grid', 'chi2' (of the refinement), 'method' ('template_fit')
    """
    from dataclasses import replace as _replace
    spec = _replace(spectrum, resolution=None)
    tcfg = C.section(cfg, 'templates')
    if regions is None:
        from .. import regions as _regions
        regions = _regions.select(spectrum, teff=tcfg.get('teff_prior'), methods=[METHOD]).get(METHOD, [])
    no_r = _strip_resolution(cfg)
    coarse = measure_resolution(spec, regions, no_r, r_grid=np.geomspace(300., 1e5, 16))
    i = int(np.argmin(coarse['chi2']))
    r_best = coarse['r_grid'][i]
    fine = measure_resolution(spec, regions, no_r, r_grid=np.geomspace(r_best / 1.5, r_best * 1.5, 15))
    fine['method'] = METHOD
    return fine


def _strip_resolution(cfg: Any) -> dict:
    """A mapping copy of the parts of cfg the template fit reads, without any resolution override"""
    opts = dict(C.section(cfg, 'methods').get('options', {}) or {})
    tf = dict(opts.get(METHOD, {}) or {})
    tf.pop('resolution', None)
    opts[METHOD] = tf
    return {'templates': C.section(cfg, 'templates'), 'methods': {'options': opts}, 'input': {}}
