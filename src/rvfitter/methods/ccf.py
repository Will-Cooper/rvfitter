"""
Cross-correlation in log-wavelength, one RV per window, with Monte Carlo errors.

Each window's data (on their own pixels) are divided by their continuum: a linear fit to the catalogue continuum
windows for feature windows of ≤100 Å, else a normalised Gaussian smoothing (FWHM `highpass_kms`, at most a third of
the window), after which a polynomial fitted to data/template at the first peak (`flatten_degree`) is divided out
and the CCF redone. Every shifted template row is normalised identically. The peak of the noise-weighted,
mean-subtracted correlation coefficient C(v) = Σ w d̃ t̃_v / √(Σ w d̃² · Σ w t̃_v²) is located with a parabola. The
error is the standard deviation of the peak over Monte Carlo noise trials, scaled by √max(1, χ²/ν) of the scaled
template against the normalised data (template mismatch); the Zucker (2003) error is a diagnostic.
"""
import logging
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import numpy as np

from .. import broadening
from ..constants import C_KMS
from ..numerics import parabola_vertex
from ..types import MethodResult, Region, Spectrum
from . import _common as C

__all__ = ['measure', 'DEFAULTS', 'ccf_curve', 'fit_peak', 'zucker_error']

log = logging.getLogger(__name__)

METHOD = 'ccf'
DEFAULTS: dict[str, Any] = C.TEMPLATE_DEFAULTS | dict(  # rv_step default: min(native step / 5, 5 km/s)
    highpass_kms=None,  # continuum-smoothing FWHM; default 25 resolution elements, capped at 1/3 of each window
    n_mc=100,  # Monte Carlo noise trials (0 disables; then the Zucker error is used)
    flatten_degree=1,  # polynomial dividing out the data/template continuum slope before the CCF (-1: off)
    chi2_scale=True,  # multiply the error by sqrt(max(1, chi2/dof)) when flux errors are given
    vsini=0.,  # rotational broadening applied to the template [km/s] (Gray kernel, epsilon 0.6)
)


@dataclass
class _Window:
    """A prepared window: its continuum model and the normalised data and weights on the used pixels"""
    prep: C.Prepared
    d: np.ndarray  # normalised data residual on the used pixels
    w: np.ndarray  # weights on the used pixels
    wave: np.ndarray  # used pixel wavelengths
    cont: np.ndarray  # data continuum on the full grid
    used: np.ndarray  # used pixels (boolean on the full grid)
    mask0: np.ndarray  # pixels entering the continuum (boolean on the full grid)
    kernel: np.ndarray | None  # high-pass smoothing kernel ('highpass' mode)
    den: np.ndarray | None  # filter normalisation: mask0 convolved with the kernel ('highpass' mode)
    mode: str = 'highpass'  # 'highpass' or 'linear'
    filter_fwhm: float = np.nan  # [km/s], 'highpass' mode
    design: np.ndarray | None = None  # [1, x] on the full grid ('linear' mode)
    proj: np.ndarray | None = None  # (2, n_cont) weighted least-squares projector onto the continuum pixels
    cont_idx: np.ndarray | None = None  # continuum pixel indices ('linear' mode)


def _linear_setup(prep: C.Prepared, mask0: np.ndarray, max_width: float):
    """Design and projector of a linear continuum from the region's continuum windows, or None if not applicable"""
    region = prep.region
    if not region.continuum or prep.wave[-1] - prep.wave[0] > max_width:
        return None
    incont = C.in_any(prep.wave, region.continuum) & mask0
    if incont.sum() < 4:
        return None
    x = _xscale(prep.wave)
    design = np.vstack([np.ones_like(x), x]).T
    idx = np.flatnonzero(incont)
    wts = 1. / prep.sigma[idx] ** 2 if prep.sigma_known else np.ones(len(idx))
    a = design[idx]
    proj = np.linalg.solve(a.T @ (a * wts[:, None]), (a * wts[:, None]).T)
    return design, proj, idx


def _window(prep: C.Prepared, highpass_kms: float, flux: np.ndarray | None = None,
            linear_max_width: float = 100.) -> _Window:
    """
    Normalise the data by their continuum

    Windows of at most `linear_max_width` Å with continuum windows (e.g. catalogue feature windows) use an
    explicit linear continuum fitted (σ-weighted) to the continuum pixels; others a normalised Gaussian
    convolution over the unmasked pixels (FWHM `highpass_kms`, capped at a third of the window). Either way the
    same operation (same pixels, weights or kernel) is later applied to every shifted template row, so data and
    template are normalised identically.
    """
    from scipy.signal import fftconvolve
    flux = prep.flux if flux is None else flux
    mask0 = prep.mask & np.isfinite(flux)
    lin = _linear_setup(prep, mask0, linear_max_width)
    if lin is not None:
        design, proj, idx = lin
        cont = design @ (proj @ flux[idx])
        with np.errstate(invalid='ignore', divide='ignore'):
            d = flux / cont - 1.
        used = mask0 & np.isfinite(d) & (cont > 0)
        w = (cont[used] / prep.sigma[used]) ** 2 if prep.sigma_known else np.ones(used.sum())
        prep.mask = used
        return _Window(prep, d[used], w, prep.wave[used], cont, used, mask0, None, None, 'linear', np.nan, design,
                       proj, idx)
    # the filter must be narrower than the window, or it cannot remove the continuum slope within it
    span_kms = float(np.log(prep.wave[-1] / prep.wave[0])) * C_KMS  # pixels: near-uniform in velocity
    fwhm = min(highpass_kms, span_kms / 3.)
    kernel = broadening.gaussian_kernel(fwhm * broadening.FWHM_TO_SIGMA / prep.dv, nsigma=3.)
    den = fftconvolve(mask0.astype(float), kernel, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        cont = fftconvolve(np.where(mask0, flux, 0.), kernel, mode='same') / den
        d = flux / cont - 1.
    cont[den < 1e-3] = np.nan
    used = mask0 & np.isfinite(d) & np.isfinite(cont) & (cont > 0) & (den >= 1e-3)
    w = (cont[used] / prep.sigma[used]) ** 2 if prep.sigma_known else np.ones(used.sum())
    prep.mask = used
    return _Window(prep, d[used], w, prep.wave[used], cont, used, mask0, kernel, den, 'highpass', float(fwhm))


def _template_rows(win: _Window, tpl: C.Template, rv: np.ndarray) -> np.ndarray:
    """Shifted template rows, normalised exactly as the data, on the used pixels: shape (len(rv), n_used)"""
    from scipy.signal import fftconvolve
    raw = np.atleast_2d(tpl.at(win.prep.wave, np.atleast_1d(rv)))
    finite = np.isfinite(raw)
    if win.mode == 'linear':
        coef = np.where(np.isfinite(raw[:, win.cont_idx]), raw[:, win.cont_idx], 0.) @ win.proj.T  # (n_v, 2)
        cont = coef @ win.design.T
    else:
        num = fftconvolve(np.where(win.mask0[None, :] & finite, raw, 0.), win.kernel[None, :], mode='same', axes=1)
        cont = num / win.den[None, :]
    with np.errstate(invalid='ignore', divide='ignore'):
        t = raw / cont - 1.
    t[~finite] = np.nan
    return t[:, win.used]


def ccf_curve(d: np.ndarray, w: np.ndarray, tmat: np.ndarray) -> np.ndarray:
    """
    Weighted, mean-subtracted normalised cross-correlation of data `d` (weights `w`) with each row of `tmat`
    (shifted normalised template, one row per trial velocity; NaN marks no coverage); NaN for a row covering
    less than 90 per cent of the pixels
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


def fit_peak(v: np.ndarray, c: np.ndarray) -> dict:
    """
    The CCF peak from a parabola through its top (±max(2 steps, 0.2 HWHM) of the maximum): 'rv', 'height'
    C(v̂), 'curvature' C''(v̂) [(km/s)⁻²], 'width' (Gaussian-equivalent σ from the half-maximum width), 'edge'
    (maximum at the grid edge) and 'ok'
    """
    good = np.isfinite(c)
    if good.sum() < 5:
        return dict(rv=np.nan, height=np.nan, curvature=np.nan, width=np.nan, edge=False, ok=False)
    vg, cg = v[good], c[good]
    i = int(np.argmax(cg))
    edge = i < 2 or i > len(cg) - 3
    lo, hi, _ = _half_max_extent(cg, i)
    step = float(vg[1] - vg[0])
    hwhm = max(0.5 * (vg[hi] - vg[lo]), step)
    width = hwhm / np.sqrt(2. * np.log(2.))
    k = max(2, int(round(0.2 * hwhm / step)))
    sl = slice(max(i - k, 0), min(i + k + 1, len(cg)))
    x0, height, curv = parabola_vertex(vg[sl] - vg[i], cg[sl]) if sl.stop - sl.start >= 3 else (0., 0., 0.)
    if curv >= 0:  # too few points, or not a peak
        return dict(rv=float(vg[i]), height=float(cg[i]), curvature=np.nan, width=width, edge=edge, ok=False)
    return dict(rv=float(vg[i] + x0), height=height, curvature=curv, width=float(width), edge=edge,
                ok=bool(abs(x0) <= (k + 0.5) * step))


def zucker_error(n_pix: int, height: float, curvature: float) -> float:
    """Zucker (2003) RV error [km/s]: σ_v² = −[N · C''(v̂)/C(v̂) · C(v̂)²/(1 − C(v̂)²)]⁻¹; NaN if undefined"""
    if not (np.isfinite(height) and np.isfinite(curvature)) or curvature >= 0 or not 0 < height < 1:
        return np.nan
    return float(np.sqrt(-1. / (n_pix * curvature / height * height ** 2 / (1. - height ** 2))))


def _ccf_on_grid(win: _Window, tpl: C.Template, rv: np.ndarray) -> np.ndarray:
    return ccf_curve(win.d, win.w, _template_rows(win, tpl, rv))


def _chi2_at(win: _Window, tpl: C.Template, v: float) -> tuple[float, float, np.ndarray]:
    """χ² of the best-scaled normalised template at v, the scale and the model"""
    t = _template_rows(win, tpl, np.array([v]))[0]
    ok = np.isfinite(t)
    if ok.sum() < 3:
        return np.inf, np.nan, t
    a = np.sum(win.w[ok] * win.d[ok] * t[ok]) / np.sum(win.w[ok] * t[ok] ** 2)
    return float(np.sum(win.w[ok] * (win.d[ok] - a * t[ok]) ** 2)), float(a), a * t


def _score_candidate(cand: C.Candidate, preps: list[C.Prepared], resolution: float, wave_range, opts,
                     coarse_step: float) -> float:
    """Total χ² of one candidate template at its own CCF peak in every window"""
    tpl = C.build_template(cand, resolution, wave_range, vsini=opts['vsini'], pad_kms=opts['pad_kms'])
    rv = np.arange(opts['rv_min'], opts['rv_max'] + 1e-9, coarse_step)
    total = 0.
    for prep in preps:
        win = _window(replace(prep, mask=prep.mask.copy()), opts['highpass_kms'])
        c = _ccf_on_grid(win, tpl, rv)
        if not np.isfinite(c).any():
            return np.inf
        total += _chi2_at(win, tpl, float(rv[np.nanargmax(c)]))[0]
    return total


def _flatten_fit(prep: C.Prepared, tpl: C.Template, v: float, degree: int) -> np.ndarray | None:
    """
    Polynomial (in scaled wavelength) fitted to data / template at velocity v, σ-weighted; None if not possible

    Dividing the data by it removes the continuum slope of the data relative to the template, which the
    high-pass filter alone leaves as a slope × line term (a 0.1–0.5 km/s bias per feature window on J1745-like
    BT-Settl injections).
    """
    if degree < 0:
        return None
    t = tpl.at(prep.wave, v)
    m = prep.mask & np.isfinite(t) & (t > 0) & np.isfinite(prep.flux)
    if m.sum() < degree + 3:
        return None
    x = _xscale(prep.wave)
    ratio = prep.flux[m] / t[m]
    w = (t[m] / prep.sigma[m]) ** 2 if prep.sigma_known else np.ones(m.sum())
    return np.polynomial.polynomial.polyfit(x[m], ratio, degree, w=np.sqrt(w))


def _xscale(wave: np.ndarray) -> np.ndarray:
    return 2. * (wave - wave[0]) / (wave[-1] - wave[0]) - 1.


def _flatten(prep: C.Prepared, coef: np.ndarray | None) -> C.Prepared:
    """A copy of prep with the flux and σ divided by the polynomial"""
    if coef is None:
        return prep
    p = np.polynomial.polynomial.polyval(_xscale(prep.wave), coef)
    with np.errstate(invalid='ignore', divide='ignore'):
        return replace(prep, flux=prep.flux / p, sigma=prep.sigma / np.abs(p), mask=prep.mask & (p > 0))


def _mc_chunk(args) -> list[float]:
    """Monte Carlo trials: add noise to the observed pixels, redo the CCF on a local grid and fit the peak"""
    seed, n, spectrum, sigma_pix, window, masks, tpl, rv_local, highpass_kms, coef = args
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        flux = spectrum.flux + rng.normal(0., 1., len(spectrum.flux)) * np.nan_to_num(sigma_pix)
        prep = _flatten(C.prepare(spectrum, window, masks, flux=flux), coef)
        win = _window(prep, highpass_kms)
        peak = fit_peak(rv_local, _ccf_on_grid(win, tpl, rv_local))
        out.append(peak['rv'] if peak['ok'] else np.nan)
    return out


def measure(spectrum: Spectrum, regions: Sequence[Region], cfg: Any, *, executor: Executor | None = None,
            rng: np.random.Generator) -> MethodResult:
    """
    Cross-correlation RVs, one per window (every non-telluric, non-continuum region; telluric ones are masked),
    combined by random effects; options in cfg.methods.options['ccf'] (`DEFAULTS`). Diagnostics hold the CCF,
    peak fit, errors, template and data/model arrays.
    """
    opts = C.options(cfg, METHOD, DEFAULTS)
    resolution = C.resolution_of(spectrum, cfg, opts)
    cands = C.candidates(cfg, opts)
    windows, masks = C.split_regions(C.covered_regions(METHOD, regions, cands))
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
        raise ValueError('no CCF window has 10 usable pixels')
    dv = min(p.dv for p in preps)
    step = opts['rv_step'] or min(dv / 5., 5.)
    wave_range = (min(p.wave[0] for p in preps), max(p.wave[-1] for p in preps))
    score = partial(_score_candidate, preps=preps, resolution=resolution, wave_range=wave_range, opts=opts,
                    coarse_step=max(step, dv / 2.))
    tpl, flag_grid, tdiag = C.choose_template(METHOD, cands, score, cfg, opts, resolution,
                                              wave_range, sum(p.n for p in preps), executor, vsini=opts['vsini'])
    rv = np.arange(opts['rv_min'], opts['rv_max'] + 1e-9, step)
    n_chunks = opts['n_chunks']
    seeds = C.child_seeds(rng, len(preps) * n_chunks)
    measurements = []
    for k, (prep, window) in enumerate(zip(preps, kept, strict=True)):
        win = _window(replace(prep, mask=prep.mask.copy()), opts['highpass_kms'])
        c = _ccf_on_grid(win, tpl, rv)
        peak = fit_peak(rv, c)
        coef = None
        flatten = opts['flatten_degree'] if win.mode == 'highpass' else -1  # a linear continuum already does it
        if peak['ok'] and flatten >= 0:  # remove the relative continuum slope, then redo the CCF
            coef = _flatten_fit(prep, tpl, peak['rv'], flatten)
            if coef is not None:
                prep = _flatten(prep, coef)
                win = _window(prep, opts['highpass_kms'])
                c = _ccf_on_grid(win, tpl, rv)
                peak = fit_peak(rv, c)
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
            mc = C.run_chunks(_mc_chunk, opts['n_mc'], seeds[k * n_chunks:(k + 1) * n_chunks],
                              (spectrum, sigma_pix, window, masks, tpl, rv_local, opts['highpass_kms'], coef),
                              executor)
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
        diag = dict(tdiag, rv_grid=rv, ccf=c, peak=peak, zucker_err=zerr, mc_std=mc_std, chi2_scale=chi2_scale,
                    n_mc=int(len(mc)), mc_rvs=mc, chi2=chi2, dof=dof, scale=scale, n_pix=win.prep.n,
                    resolution=resolution, highpass_kms=opts['highpass_kms'], continuum_mode=win.mode,
                    filter_fwhm=win.filter_fwhm, flatten_degree=flatten if coef is not None else -1,
                    wave=win.wave, data=win.d, model=model, weights=win.w)
        measurements.append(C.window_measurement(METHOD, window, peak['rv'], err * chi2_scale, flag, chi2 / dof,
                                                 prep.sigma_known, diag))
    result = C.finish(METHOD, measurements, opts['outlier_k'])
    log.info('ccf: %.2f ± %.2f km/s (template %s, %d windows)', result.rv, result.rv_err, tpl.label,
             result.n_regions)
    return result
