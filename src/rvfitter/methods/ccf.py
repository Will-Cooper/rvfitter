"""
Cross-correlation against a template in log-wavelength, with peak fitting and Monte Carlo errors.

For each window the data are resampled onto a uniform ln λ grid at their native sampling, divided by a smooth
continuum (normalised Gaussian convolution of FWHM `highpass_kms`, so the broad molecular-band shape is
removed) and the same high-pass is applied to the template, which has been broadened to the data resolution.
The noise-weighted, mean-subtracted correlation coefficient

    C(v) = Σ w d̃ t̃_v / √(Σ w d̃² · Σ w t̃_v²)

is computed on a grid of radial velocities (the template is shifted relativistically and interpolated, so the
data are never shifted), and its peak is located with a parabola through the top of the peak (or, optionally, a
Gaussian fit, which is biased for asymmetric peaks).

The statistical error is the standard deviation of the peak over Monte Carlo trials in which Gaussian noise of
the quoted σ is added to the observed pixels (seeded, optionally parallel), multiplied by √max(1, χ²/ν) of the
scaled template against the normalised data, so residuals beyond the noise (template mismatch) inflate it as
they do for the template fit. The Zucker (2003) maximum-likelihood
error, σ_v² = −[N C''/C · C²/(1 − C²)]⁻¹, is reported as a diagnostic.
When several templates are candidates, the one with the lowest χ² of the scaled, normalised template at its
CCF peak (summed over windows) is used.
"""
import logging
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from ..constants import C_KMS
from ..types import Measurement, MethodResult, Region, Spectrum
from . import _common as C

__all__ = ['measure', 'DEFAULTS', 'ccf_curve', 'fit_peak', 'zucker_error']

log = logging.getLogger(__name__)

METHOD = 'ccf'
DEFAULTS: dict[str, Any] = dict(
    resolution=None,  # override the data's R
    rv_min=-500., rv_max=500.,  # search range [km/s]
    rv_step=None,  # default min(native step / 5, 5 km/s)
    highpass_kms=None,  # FWHM of the continuum smoothing; default 25 resolution elements
    peak='parabola',  # 'parabola' (default) or 'gaussian' (see fit_peak)
    n_mc=100,  # Monte Carlo noise trials (0 disables; then the Zucker error is used)
    chi2_scale=True,  # multiply the error by sqrt(max(1, chi2/dof)) when flux errors are given
    n_chunks=8,  # Monte Carlo tasks (fixed, so results do not depend on the executor)
    vsini=0.,  # rotational broadening applied to the template [km/s] (Gray kernel, epsilon 0.6)
    template=None, templates=None, teff_window=300.,
    region_kinds=None,  # region kinds used as windows (default: all but telluric/continuum)
    segment_res=150.,  # split windows longer than this many resolution elements (0: never)
    outlier_k=5.,  # gross-outlier rejection among regions (0: off)
    pad_kms=1500.,
)


@dataclass
class _Window:
    """A prepared window: normalised data, weights and the template's normalised flux"""
    prep: C.Prepared
    d: np.ndarray  # normalised data residual on mask pixels
    w: np.ndarray  # weights on mask pixels
    wave: np.ndarray  # mask pixel wavelengths
    cont: np.ndarray  # continuum on the full grid


def _normalise(prep: C.Prepared, flux: np.ndarray, highpass_kms: float) -> tuple[np.ndarray, np.ndarray]:
    cont = C.highpass(prep.wave, flux, prep.mask, highpass_kms)
    with np.errstate(invalid='ignore', divide='ignore'):
        return flux / cont - 1., cont


def _window(prep: C.Prepared, highpass_kms: float, flux: np.ndarray | None = None) -> _Window:
    flux = prep.flux if flux is None else flux
    d, cont = _normalise(prep, flux, highpass_kms)
    m = prep.mask & np.isfinite(d) & np.isfinite(cont) & (cont > 0)
    if prep.sigma_known:
        w = (cont[m] / prep.sigma[m]) ** 2
    else:
        w = np.ones(m.sum())
    prep.mask = m
    return _Window(prep, d[m], w, prep.wave[m], cont)


def _template_highpassed(tpl: C.Template, highpass_kms: float) -> C.Template:
    """The template divided by its own smooth continuum, minus one"""
    wave = tpl.wave
    cont = C.highpass(wave, tpl.flux, np.isfinite(tpl.flux), highpass_kms)
    with np.errstate(invalid='ignore', divide='ignore'):
        flux = tpl.flux / cont - 1.
    return C.Template(tpl.lnw0, tpl.dlnw, flux, tpl.label, tpl.params)


def ccf_curve(d: np.ndarray, w: np.ndarray, tmat: np.ndarray) -> np.ndarray:
    """
    Weighted, mean-subtracted normalised cross-correlation

    Parameters
    ----------
    d
        Normalised data (n_pix)
    w
        Weights (n_pix)
    tmat
        Shifted normalised template, one row per trial velocity (n_v, n_pix); NaN marks no coverage

    Returns
    -------
    c
        C(v) for each row (NaN if a row has no coverage)
    """
    good = np.isfinite(tmat)
    tm = np.where(good, tmat, 0.)
    ww = w[None, :] * good
    sw = ww.sum(axis=1)
    dm = (ww * d[None, :]).sum(axis=1) / sw
    tmean = (ww * tm).sum(axis=1) / sw
    dd = d[None, :] - dm[:, None]
    tt = tm - tmean[:, None]
    num = (ww * dd * tt).sum(axis=1)
    den = np.sqrt((ww * dd ** 2).sum(axis=1) * (ww * tt ** 2).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    out[good.sum(axis=1) < 0.9 * len(d)] = np.nan  # require the template to cover 90 per cent of the pixels
    return out


def _half_max_extent(cg: np.ndarray, i: int) -> tuple[int, int, float]:
    """Indices of the contiguous region above half the peak height (relative to the median) and the floor"""
    floor = float(np.median(cg))
    half = cg[i] - 0.5 * (cg[i] - floor)
    lo = i
    while lo > 0 and cg[lo - 1] > half:
        lo -= 1
    hi = i
    while hi < len(cg) - 1 and cg[hi + 1] > half:
        hi += 1
    return lo, hi, floor


def fit_peak(v: np.ndarray, c: np.ndarray, kind: str = 'parabola') -> dict:
    """
    Fit the CCF peak

    Parameters
    ----------
    v, c
        Velocity grid [km/s] and CCF
    kind
        'parabola' (default): a parabola through the top of the peak, within ±max(2 steps, 0.2 HWHM) of the
        maximum. 'gaussian': a Gaussian plus constant over the upper half of the peak; this is biased when the
        peak is asymmetric and the noise moves the half-maximum boundary (≈0.2 km/s at R = 20 000 and SNR 30 in
        the injection tests), so it is offered for comparison only.

    Returns
    -------
    peak
        'rv', 'height' C(v̂), 'curvature' C''(v̂) [(km/s)⁻²], 'width' (Gaussian-equivalent σ from the half-maximum
        width), 'edge' (maximum at the grid edge) and 'ok'
    """
    good = np.isfinite(c)
    if good.sum() < 5:
        return dict(rv=np.nan, height=np.nan, curvature=np.nan, width=np.nan, edge=False, ok=False)
    vg, cg = v[good], c[good]
    i = int(np.argmax(cg))
    edge = i < 2 or i > len(cg) - 3
    lo, hi, floor = _half_max_extent(cg, i)
    step = float(vg[1] - vg[0])
    hwhm = max(0.5 * (vg[hi] - vg[lo]), step)
    width = hwhm / np.sqrt(2. * np.log(2.))
    if kind == 'parabola':
        k = max(2, int(round(0.2 * hwhm / step)))
        sl = slice(max(i - k, 0), min(i + k + 1, len(cg)))
        if sl.stop - sl.start < 3:
            return dict(rv=float(vg[i]), height=float(cg[i]), curvature=np.nan, width=width, edge=edge, ok=False)
        a, b, cc = np.polyfit(vg[sl] - vg[i], cg[sl], 2)
        if a >= 0:
            return dict(rv=float(vg[i]), height=float(cg[i]), curvature=np.nan, width=width, edge=edge, ok=False)
        x0 = -b / (2 * a)
        return dict(rv=float(vg[i] + x0), height=float(cc - b ** 2 / (4 * a)), curvature=float(2 * a),
                    width=float(width), edge=edge, ok=bool(abs(x0) <= (k + 0.5) * step))
    if kind != 'gaussian':
        raise ValueError(f"peak must be 'gaussian' or 'parabola', not {kind!r}")
    lo, hi = max(0, min(lo, i - 2)), min(len(cg) - 1, max(hi, i + 2))
    x, y = vg[lo:hi + 1], cg[lo:hi + 1]
    if len(x) < 5:  # truncated by the grid edge
        return fit_peak(v, c, 'parabola')
    s0 = max((x[-1] - x[0]) / 2.355, vg[1] - vg[0])
    try:
        popt, _ = curve_fit(lambda xx, a, mu, s, b: a * np.exp(-0.5 * ((xx - mu) / s) ** 2) + b, x, y,
                            p0=(cg[i] - floor, vg[i], s0, floor), maxfev=5000)
    except (RuntimeError, ValueError):
        return fit_peak(v, c, 'parabola')
    a, mu, s, b = popt
    ok = bool(a > 0 and x[0] <= mu <= x[-1])
    return dict(rv=float(mu), height=float(a + b), curvature=float(-a / s ** 2), width=float(abs(s)), edge=edge,
                ok=ok)


def zucker_error(n_pix: int, height: float, curvature: float) -> float:
    """
    Zucker (2003) maximum-likelihood RV error from the CCF peak

    σ_v² = −[N · C''(v̂)/C(v̂) · C(v̂)²/(1 − C(v̂)²)]⁻¹, with N the number of pixels

    Returns
    -------
    sigma
        [km/s], NaN if undefined
    """
    if not (np.isfinite(height) and np.isfinite(curvature)) or curvature >= 0 or not 0 < height < 1:
        return np.nan
    return float(np.sqrt(-1. / (n_pix * curvature / height * height ** 2 / (1. - height ** 2))))


def _ccf_on_grid(win: _Window, tpl: C.Template, rv: np.ndarray) -> np.ndarray:
    return ccf_curve(win.d, win.w, tpl.at(win.wave, rv))


def _chi2_at(win: _Window, tpl: C.Template, v: float) -> tuple[float, float, np.ndarray]:
    """χ² of the best-scaled normalised template at v, the scale and the model"""
    t = tpl.at(win.wave, v)
    ok = np.isfinite(t)
    if ok.sum() < 3:
        return np.inf, np.nan, t
    a = np.sum(win.w[ok] * win.d[ok] * t[ok]) / np.sum(win.w[ok] * t[ok] ** 2)
    return float(np.sum(win.w[ok] * (win.d[ok] - a * t[ok]) ** 2)), float(a), a * t


def _score_candidate(cand: C.Candidate, preps: list[C.Prepared], resolution: float, wave_range, opts) -> float:
    """Total χ² of one candidate template at its own CCF peak in every window"""
    raw = C.build_template(cand, resolution, wave_range, vsini=opts['vsini'], pad_kms=opts['pad_kms'])
    tpl = _template_highpassed(raw, opts['highpass_kms'])
    rv = np.arange(opts['rv_min'], opts['rv_max'] + 1e-9, opts['_coarse_step'])
    total = 0.
    for prep in preps:
        win = _window(_copy_prep(prep), opts['highpass_kms'])
        c = _ccf_on_grid(win, tpl, rv)
        if not np.isfinite(c).any():
            return np.inf
        total += _chi2_at(win, tpl, float(rv[np.nanargmax(c)]))[0]
    return total


def _copy_prep(prep: C.Prepared) -> C.Prepared:
    return C.Prepared(prep.wave, prep.flux, prep.sigma, prep.mask.copy(), prep.dv, prep.sigma_known, prep.region,
                      prep.source_pixels)


def _mc_chunk(args) -> list[float]:
    """Monte Carlo trials: add noise to the observed pixels, redo the CCF on a local grid and fit the peak"""
    seed, n, spectrum, sigma_pix, window, masks, dv, tpl, rv_local, highpass_kms, kind = args
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        flux = spectrum.flux + rng.normal(0., 1., len(spectrum.flux)) * np.nan_to_num(sigma_pix)
        prep = C.prepare(spectrum, window, masks, dv=dv, flux=flux)
        win = _window(prep, highpass_kms)
        peak = fit_peak(rv_local, _ccf_on_grid(win, tpl, rv_local), kind)
        out.append(peak['rv'] if peak['ok'] else np.nan)
    return out


def measure(spectrum: Spectrum, regions: Sequence[Region], cfg: Any = None, *, executor: Executor | None = None,
            rng: np.random.Generator) -> MethodResult:
    """
    Cross-correlation RVs, one per window, combined with the Birge-inflated weighted mean

    Parameters
    ----------
    spectrum
        Observed spectrum (vacuum, in the frame the RV is wanted in)
    regions
        Regions for this method: windows (by default every non-telluric, non-continuum region) and telluric masks
    cfg
        Config (or mapping); options under cfg.methods.options['ccf'] (see `DEFAULTS`); templates from
        cfg.templates unless the option 'template'/'templates' is given
    executor
        Optional executor for the template scoring and the Monte Carlo trials
    rng
        Random generator for the Monte Carlo trials

    Returns
    -------
    result
        The combined RV and the per-window measurements (diagnostics hold the CCF curve, peak fit, errors,
        template and data/model arrays)
    """
    opts = C.options(cfg, METHOD, DEFAULTS)
    resolution = C.resolution_of(spectrum, cfg, opts)
    windows, masks = C.split_regions(regions, opts['region_kinds'])
    if not windows:
        raise ValueError('no CCF windows among the regions')
    windows = C.segment(windows, resolution, opts['segment_res'])
    if opts['highpass_kms'] is None:
        opts['highpass_kms'] = 25. * C_KMS / resolution
    preps, kept = [], []
    for win in windows:
        try:
            p = C.prepare(spectrum, win, masks)
        except ValueError as err:
            log.warning('ccf: skipping %s (%s)', win.name, err)
            continue
        if p.n < 10:
            log.warning('ccf: skipping %s (%d usable pixels)', win.name, p.n)
            continue
        preps.append(p)
        kept.append(win)
    if not preps:
        return MethodResult(METHOD, np.nan, np.nan, 0, [])
    dv = min(p.dv for p in preps)
    step = opts['rv_step'] or min(dv / 5., 5.)
    opts['_coarse_step'] = max(step, dv / 2.)
    wave_range = (min(p.wave[0] for p in preps), max(p.wave[-1] for p in preps))
    cands = C.candidates(cfg, opts)
    if len(cands) > 1:
        scores = C.pmap(partial(_score_candidate, preps=preps, resolution=resolution, wave_range=wave_range,
                                opts=opts), cands, executor)
        best = cands[int(np.argmin(scores))]
        log.info('ccf: template %s chosen by χ² from %d candidates', best.label, len(cands))
    else:
        best, scores = cands[0], [np.nan]
    tpl_raw = C.build_template(best, resolution, wave_range, vsini=opts['vsini'], pad_kms=opts['pad_kms'])
    tpl = _template_highpassed(tpl_raw, opts['highpass_kms'])
    flag_grid = C.FLAG_OUTSIDE_GRID if tpl_raw.params.get('outside_grid') else 0
    rv = np.arange(opts['rv_min'], opts['rv_max'] + 1e-9, step)
    seeds = C.child_seeds(rng, len(preps) * opts['n_chunks'])
    measurements = []
    for k, (prep, window) in enumerate(zip(preps, kept, strict=True)):
        win = _window(prep, opts['highpass_kms'])
        c = _ccf_on_grid(win, tpl, rv)
        peak = fit_peak(rv, c, opts['peak'])
        flag = flag_grid | (C.FLAG_EDGE if peak['edge'] else 0) | (0 if peak['ok'] else C.FLAG_FIT_FAILED)
        chi2, scale, model = _chi2_at(win, tpl, peak['rv']) if peak['ok'] else (np.nan, np.nan, None)
        dof = max(win.prep.n - 2, 1)
        zerr = zucker_error(win.prep.n, peak['height'], peak['curvature'])
        # σ on the observed pixels for the Monte Carlo trials: as given, or estimated from the residual scatter
        sel = (spectrum.wave >= window.wmin) & (spectrum.wave <= window.wmax)
        if prep.sigma_known:
            sigma_pix = np.where(sel, spectrum.sigma, 0.)
        else:
            resid = np.std(win.d - model) if model is not None else np.nan
            cont_pix = np.interp(spectrum.wave, prep.wave, np.nan_to_num(win.cont))
            sigma_pix = np.where(sel, resid * cont_pix, 0.)
            log.warning('ccf: %s has no flux errors; σ estimated from the residual scatter', window.name)
        mc_std, mc = np.nan, []
        if opts['n_mc'] > 0 and peak['ok'] and not peak['edge']:
            half = 3. * max(peak['width'], C_KMS / resolution / 2.355)
            rv_local = np.arange(peak['rv'] - half, peak['rv'] + half + 1e-9, step)
            n_each = np.diff(np.linspace(0, opts['n_mc'], opts['n_chunks'] + 1).astype(int))
            tasks = [(seeds[k * opts['n_chunks'] + j], int(n), spectrum, sigma_pix, window, masks, prep.dv, tpl,
                      rv_local, opts['highpass_kms'], opts['peak']) for j, n in enumerate(n_each) if n > 0]
            mc = np.concatenate([np.asarray(x, float) for x in C.pmap(_mc_chunk, tasks, executor)])
            finite = np.isfinite(mc)
            if finite.sum() >= max(5, 0.8 * len(mc)):
                mc_std = float(np.std(mc[finite], ddof=1))
            else:
                flag |= C.FLAG_FIT_FAILED
        err = mc_std if np.isfinite(mc_std) else zerr
        # residuals beyond the quoted noise (template mismatch) inflate the error, as for the template fit
        chi2_scale = 1.
        if opts['chi2_scale'] and prep.sigma_known and np.isfinite(chi2):
            chi2_scale = float(np.sqrt(max(1., chi2 / dof)))
        err *= chi2_scale
        if not np.isfinite(err):
            flag |= C.FLAG_FIT_FAILED
        if np.isfinite(chi2) and chi2 / dof > 3. and prep.sigma_known:
            flag |= C.FLAG_POOR_FIT
        diag = dict(rv_grid=rv, ccf=c, peak=peak, zucker_err=zerr, mc_std=mc_std, chi2_scale=chi2_scale,
                    n_mc=int(len(mc)), mc_rvs=mc,
                    chi2=chi2, dof=dof, scale=scale, n_pix=win.prep.n, resolution=resolution,
                    highpass_kms=opts['highpass_kms'], template=tpl.label, template_params=tpl_raw.params,
                    template_scores=dict(zip([c_.label for c_ in cands], map(float, scores), strict=False)),
                    wave=win.wave, data=win.d, model=model, weights=win.w)
        m = Measurement(METHOD, window.name, float(peak['rv']), float(err), int(flag), diag)
        log.debug('ccf %s: %.2f ± %.2f km/s (Zucker %.2f, flag %d)', window.name, m.rv, m.rv_err, zerr, flag)
        measurements.append(m)
    result = C.finish(METHOD, measurements, opts['outlier_k'])
    log.info('ccf: %.2f ± %.2f km/s (template %s, %d windows)', result.rv, result.rv_err, tpl.label,
             result.n_regions)
    return result
