"""
Least-squares deconvolution (LSD; Donati et al. 1997): the mean line profile Z(v) of the spectrum against a line mask.

The data are modelled as F = P(x) · (1 − M Z), with M[j, k] = Σ_l w_l Λ(u_jl − s_k) the mask lines (weights w_l)
seen by pixel j at velocity bin k (Λ: linear interpolation between bins; u, s in ln λ) and P a polynomial per window.
Z and the polynomials are solved by Gauss–Newton weighted least squares (the model is bilinear), with the covariance
(JᵀWJ)⁻¹ scaled by √max(1, χ²/ν). The mask comes from the minima of the best template (chosen by the template-fit
χ²) at `mask_resolution`, weighted by depth, or from the catalogue's line features (weight 1).
RV = centre of a Gaussian + constant fitted to Z with its full covariance, minus the mask zero point: the centre
that the same LSD gives for the noiseless template shifted to the first estimate (blends, asymmetric lines and pixel
phase), less that shift. vsini comes from the profile width beyond the template's own (zero-point) profile width,
when resolved (> 2σ).
"""
import logging
from collections.abc import Sequence
from concurrent.futures import Executor
from functools import partial
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .. import broadening
from .. import features as F
from ..constants import C_KMS
from ..types import MethodResult, Region, Spectrum
from . import _common as C
from . import template_fit

__all__ = ['measure', 'DEFAULTS', 'line_mask', 'deconvolve']

log = logging.getLogger(__name__)

METHOD = 'lsd'
FWHM = 2. * np.sqrt(2. * np.log(2.))
DEFAULTS: dict[str, Any] = C.TEMPLATE_DEFAULTS | dict(  # rv_min/rv_max: the profile's velocity span
    rv_min=-400., rv_max=400.,  # rv_step: profile bin [km/s]; default the median pixel step
    mask='template',  # 'template' (minima of the best template) | 'catalogue' (line features, weight 1)
    mask_resolution=50000.,  # R of the template the mask lines are read from
    min_depth=0.05,  # mask lines shallower than this (relative to the template's local maximum) are dropped
    continuum_resel=5.,  # local-maximum width for the template's pseudo-continuum [data resolution elements]
    poly_degree=2,
    joint=True,  # one profile shared by every segment (each with its own polynomial); False: one per segment
    epsilon=0.6,  # limb darkening for the vsini conversion
)


def line_mask(tpl: C.Template, resolution: float, min_depth: float, continuum_resel: float
              ) -> tuple[np.ndarray, np.ndarray]:
    """(ln λ, optical depth −ln(1 − depth)) of the template's minima deeper than min_depth below its local maximum"""
    size = max(int(round(continuum_resel / (resolution * tpl.dlnw))), 3)
    flux = np.nan_to_num(tpl.flux, nan=0.)
    cont = maximum_filter1d(flux, size, mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        depth = np.where(cont > 0, 1. - flux / cont, 0.)
    idx, _ = find_peaks(depth, height=min_depth)
    return tpl.lnw0 + tpl.dlnw * idx, -np.log1p(-np.minimum(depth[idx], 0.999))


def _catalogue_mask(teff: float | None) -> tuple[np.ndarray, np.ndarray]:
    lines = [f for f in F.features() if f.kind == 'line' and f.teff_ok(teff)]
    return np.log([f.wave_vac for f in lines]), np.ones(len(lines))


def _mask_matrix(lnw: np.ndarray, lnl: np.ndarray, wts: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, int]:
    """M[j, k], the mask weight that pixel j sees at shift s_k (linear interpolation between bins), and the number
    of mask lines seen by any pixel"""
    ds = s[1] - s[0]
    order = np.argsort(lnl)
    lnl, wts = lnl[order], wts[order]
    lo = np.searchsorted(lnw, lnl + s[0], side='left')
    hi = np.searchsorted(lnw, lnl + s[-1], side='right')
    counts = hi - lo
    ll = np.repeat(np.arange(len(lnl)), counts)
    jj = np.repeat(lo, counts) + np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    x = (lnw[jj] - lnl[ll] - s[0]) / ds
    k = np.clip(np.floor(x).astype(int), 0, len(s) - 2)
    f = x - k
    m = np.zeros((len(lnw), len(s)))
    np.add.at(m, (jj, k), wts[ll] * (1. - f))
    np.add.at(m, (jj, k + 1), wts[ll] * f)
    return m, int(np.count_nonzero(counts))


def deconvolve(datas: Sequence[C.Prepared], flux: Sequence[np.ndarray], mask: tuple[np.ndarray, np.ndarray],
               s: np.ndarray, degree: int, ref: dict | None = None) -> dict:
    """
    The mean profile Z on the shift grid s from the windows' pixels (`flux` replaces each window's data), with its
    covariance (scaled by max(1, χ²/ν); by χ²/ν when σ is unknown), χ², ν and the best-fit model (flux)

    The model is ln F = ln P − M Z (optical depths add, so deep and blended lines stay linear), weighted by
    (F/σ)². `ref` (an earlier result) imposes its pixels and weights, so a noiseless model is deconvolved exactly
    as the data were.
    """
    keep = ref['keep'] if ref else [d.mask & (np.asarray(fl, float) > 0) for d, fl in zip(datas, flux, strict=True)]
    lnw = np.concatenate([np.log(d.wave[k]) for d, k in zip(datas, keep, strict=True)])
    f = np.concatenate([np.asarray(fl, float)[k] for fl, k in zip(flux, keep, strict=True)])
    known = all(d.sigma_known for d in datas)
    if ref:
        w = ref['w']
    elif known:
        w = (f / np.concatenate([d.sigma[k] for d, k in zip(datas, keep, strict=True)])) ** 2  # σ(ln F) = σ/F
    else:
        w = np.ones(len(f))
    vander = np.zeros((len(f), len(datas) * (degree + 1)))
    i0 = 0
    for i, (d, k) in enumerate(zip(datas, keep, strict=True)):
        x = 2. * (d.wave[k] - d.wave[0]) / (d.wave[-1] - d.wave[0]) - 1.
        vander[i0:i0 + len(x), i * (degree + 1):(i + 1) * (degree + 1)] = np.vander(x, degree + 1, increasing=True)
        i0 += len(x)
    m, n_lines = _mask_matrix(lnw, *mask, s)
    if not np.any(m):
        raise ValueError('no mask line falls on the data within the velocity span')
    a = np.hstack([-m, vander])
    sw = np.sqrt(w)
    sol, *_ = np.linalg.lstsq(a * sw[:, None], np.log(f) * sw, rcond=None)
    chi2 = float(np.sum(w * (np.log(f) - a @ sol) ** 2))
    dof = max(len(f) - a.shape[1], 1)
    scale = chi2 / dof if not known else max(1., chi2 / dof)
    cov = np.linalg.pinv(a.T @ (a * w[:, None]))[:len(s), :len(s)] * scale
    return dict(z=sol[:len(s)], cov=cov, chi2=chi2, dof=dof, sigma_known=known, model=np.exp(a @ sol), lnw=lnw,
                n_lines=n_lines, keep=keep, w=w)


def _gauss(v, b, a, mu, sig):
    return b + a * np.exp(-0.5 * ((v - mu) / sig) ** 2)


def _fit_profile(v: np.ndarray, z: np.ndarray, cov: np.ndarray, sig0: float) -> dict:
    """
    Gaussian + constant fitted to the profile with the diagonal errors; parameter errors by the sandwich
    H Jᵀ W C W J H with the full covariance C (the bins are correlated, and C is near-singular for dense masks, so
    it is not inverted)
    """
    err_z = np.sqrt(np.diag(cov))
    i = int(np.argmax(z))
    b0 = float(np.median(z))
    p0 = (b0, max(float(z[i]) - b0, 1e-6), float(v[i]), sig0)
    lo, hi = float(v[0]), float(v[-1])
    bounds = ([-np.inf, 0., lo, 0.25 * (v[1] - v[0])], [np.inf, np.inf, hi, hi - lo])
    try:
        p, h = curve_fit(_gauss, v, z, p0=p0, sigma=err_z, absolute_sigma=True, bounds=bounds, maxfev=20000)
    except RuntimeError as exc:  # curve_fit's "optimal parameters not found"
        log.debug('lsd: profile fit failed: %s', exc)
        return dict(ok=False, mu=np.nan, mu_err=np.nan, sig=np.nan, sig_err=np.nan, chi2_red=np.nan, params=None)
    steps = np.maximum(1e-6 * np.abs(p), 1e-8)
    jac = np.array([(_gauss(v, *(p + e)) - _gauss(v, *(p - e))) / (2 * e[k])
                    for k, e in enumerate(np.diag(steps))]).T
    wj = jac / err_z[:, None] ** 2
    pcov = h @ wj.T @ cov @ wj @ h
    red = float(np.sum(((z - _gauss(v, *p)) / err_z) ** 2)) / max(len(v) - 4, 1)
    err = np.sqrt(np.diag(pcov))
    edge = p[2] - 2 * p[3] < lo or p[2] + 2 * p[3] > hi
    return dict(ok=bool(np.all(np.isfinite(err))), edge=bool(edge), mu=float(p[2]), mu_err=float(err[2]),
                sig=float(p[3]), sig_err=float(err[3]), chi2_red=red, params=p)


def _rotation_sigma(epsilon: float) -> float:
    """Standard deviation of the Gray rotation kernel per km/s of vsini"""
    dv = 1e-3
    k = broadening.rotation_kernel(dv, 1., epsilon)
    x = (np.arange(len(k)) - len(k) // 2) * dv
    return float(np.sqrt(np.sum(k * x * x)))


def _vsini(fit: dict, sig0: float, epsilon: float) -> tuple[float, float, float]:
    """(vsini, error, 2σ upper limit) from the profile width beyond sig0; vsini NaN when not resolved"""
    q = fit['sig'] ** 2 - sig0 ** 2
    q_err = 2. * fit['sig'] * fit['sig_err']
    k = _rotation_sigma(epsilon)
    upper = float(np.sqrt(max(q, 0.) + 2. * q_err) / k)
    if not (q > 2. * q_err):
        return np.nan, np.nan, upper
    return float(np.sqrt(q) / k), float(q_err / (2. * k * np.sqrt(q))), upper


def _measure_group(datas: list[C.Prepared], name: str, mask, s, v, tpl_data: C.Template, opts, resolution,
                   flag_grid: int, tdiag: dict):
    sig0 = np.sqrt((C_KMS / resolution / FWHM) ** 2 + (C_KMS / opts['mask_resolution'] / FWHM) ** 2
                   + float(np.median([d.dv for d in datas])) ** 2 / 12.)
    sol = deconvolve(datas, [d.flux for d in datas], mask, s, opts['poly_degree'])
    fit = _fit_profile(v, sol['z'], sol['cov'], sig0)
    # mask zero point: the same LSD of the noiseless template, shifted to the first estimate (same pixel phase)
    v_tpl, zp = 0., np.nan
    for _ in range(2):
        rest = deconvolve(datas, [tpl_data.at(d.wave, v_tpl) for d in datas], mask, s, opts['poly_degree'], sol)
        zero = _fit_profile(v, rest['z'], sol['cov'], sig0)  # the data's weights, so the same estimator
        zp = zero['mu'] - v_tpl
        if not (fit['ok'] and zero['ok']):
            break
        v_tpl = fit['mu'] - zp
    ok = fit['ok'] and zero['ok']
    rv = fit['mu'] - zp if ok else np.nan
    flag = flag_grid | (0 if ok else C.FLAG_FIT_FAILED) | (C.FLAG_EDGE if fit.get('edge') else 0)
    vsini, vsini_err, vsini_upper = _vsini(fit, zero['sig'] if zero['ok'] else sig0, opts['epsilon']) \
        if fit['ok'] else (np.nan, np.nan, np.nan)
    region = Region(name, 'band', min(d.wave[0] for d in datas), max(d.wave[-1] for d in datas))
    diag = dict(tdiag, profile_v=v, profile=sol['z'], profile_err=np.sqrt(np.diag(sol['cov'])),
                profile_model=_gauss(v, *fit['params']) if fit['ok'] else None, profile_zero=rest['z'],
                mask_zero_point=zp, profile_sigma=fit['sig'], profile_sigma_err=fit['sig_err'],
                profile_chi2_red=fit['chi2_red'], vsini=vsini, vsini_err=vsini_err, vsini_upper=vsini_upper,
                n_mask_lines=sol['n_lines'], chi2=sol['chi2'], dof=sol['dof'], resolution=resolution,
                wave=np.exp(sol['lnw']), model=sol['model'], windows=[d.region.name for d in datas])
    return C.window_measurement(METHOD, region, rv, fit['mu_err'], flag, sol['chi2'] / sol['dof'],
                                sol['sigma_known'], diag)


def measure(spectrum: Spectrum, regions: Sequence[Region], cfg: Any, *, executor: Executor | None = None,
            rng: np.random.Generator) -> MethodResult:
    """
    LSD RV from the mean line profile of every non-telluric, non-continuum region (telluric ones masked); options in
    cfg.methods.options['lsd'] (`DEFAULTS`). Diagnostics hold the profile, its error and Gaussian model, the
    mask zero point and vsini.
    """
    opts = C.options(cfg, METHOD, DEFAULTS)
    if opts['mask'] not in ('template', 'catalogue'):
        raise ValueError(f'lsd option mask must be "template" or "catalogue", not {opts["mask"]!r}')
    resolution = C.resolution_of(spectrum, cfg, opts)
    cands = C.candidates(cfg, opts)
    regions = C.covered_regions(METHOD, regions, cands)
    windows, masks = C.split_regions(regions)
    windows = C.segment(windows, resolution, opts['segment_res'])  # each segment has its own polynomial
    datas = []
    for win in windows:
        d = C.prepare(spectrum, win, masks)
        if d.n < opts['poly_degree'] + 10:
            log.warning('lsd: skipping %s (%d usable pixels)', win.name, d.n)
            continue
        datas.append(d)
    if not datas:
        raise ValueError(f'no lsd window has {opts["poly_degree"] + 10} usable pixels')
    wave_range = (min(d.wave[0] for d in datas), max(d.wave[-1] for d in datas))

    tf_opts = template_fit.DEFAULTS | {k: opts[k] for k in ('pad_kms', 'teff_window')}
    tf_datas, _ = template_fit._prepare(spectrum, regions, tf_opts, resolution)
    score = partial(template_fit._score_candidate, datas=tf_datas, resolution=resolution, wave_range=wave_range,
                    opts=tf_opts, rv_grid=template_fit._rv_grid(spectrum, tf_opts))
    tpl_mask, flag_grid, tdiag = C.choose_template(METHOD, cands, score, cfg, opts,
                                                   opts['mask_resolution'], wave_range,
                                                   sum(d.n for d in datas), executor)
    tpl_data = C.Template(tpl_mask.lnw0, tpl_mask.dlnw, broadening.broaden_loglam(
        tpl_mask.flux, tpl_mask.dlnw * C_KMS, resolution=resolution, resolution_native=opts['mask_resolution']),
        tpl_mask.label, tpl_mask.params)
    if opts['mask'] == 'template':
        mask = line_mask(tpl_mask, resolution, opts['min_depth'], opts['continuum_resel'])
    else:
        mask = _catalogue_mask(cfg.templates.teff_prior)
    if not len(mask[0]):
        raise ValueError('the lsd line mask is empty')

    step = opts['rv_step'] or float(np.median([d.dv for d in datas]))
    s = np.arange(np.log(broadening.doppler_factor(opts['rv_min'])),
                  np.log(broadening.doppler_factor(opts['rv_max'])) + 1e-12, step / C_KMS)
    v = C_KMS * np.tanh(s)  # exp(s) = √((1+β)/(1−β))
    groups = [(datas, 'all')] if opts['joint'] else [([d], d.region.name) for d in datas]
    measurements = [_measure_group(g, name, mask, s, v, tpl_data, opts, resolution, flag_grid, tdiag)
                    for g, name in groups]
    result = C.finish(METHOD, measurements, opts['outlier_k'])
    log.info('lsd: %.2f ± %.2f km/s (template %s, %d profile(s), %s mask lines)', result.rv, result.rv_err,
             tpl_mask.label, len(measurements), [m.diagnostics['n_mask_lines'] for m in measurements])
    return result
