"""
Line centring: per line window, a σ-weighted least-squares fit of m(λ) = (c₀ + c₁ x) · (1 − W φ(λ − λ_c)), with
x = λ − λ_rest, λ_c = λ_rest D(v) (D the relativistic Doppler factor) and φ an area-normalised Gaussian,
Lorentzian or Voigt profile (`scipy.special.voigt_profile`). The fit uses the line window (the core the catalogue
gives; pressure-broadened alkali lines get core-only windows) and the region's continuum windows. The error is
the covariance of v scaled by √max(1, χ²/ν). With `joint`, every line shares one v (the multi-feature fit).

By default the method is model-free: the RVs come from the NIST rest wavelengths alone, with no template
involvement, so it is the template-independent cross-check of the template methods. With `shape_correction`
(opt-in) each line's RV is corrected for the bias of the profile fit on the line's real, asymmetric or blended
shape: the template nearest the priors, at the spectrum's R on its pixels, noise-free and unshifted, is fitted
exactly like the data, its RV (truth 0) is subtracted, and the standard deviation of that offset over the
template and its ±1 Teff-step grid neighbours is added in quadrature to the error. The method then depends on
the template models (logged at INFO).
"""
import logging
from collections.abc import Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

from .. import broadening
from ..constants import C_KMS
from ..types import MethodResult, Region, Spectrum
from . import _common as C

__all__ = ['measure', 'DEFAULTS', 'PROFILES']

log = logging.getLogger(__name__)

METHOD = 'linecentre'
PROFILES = {'gaussian': ('sigma',), 'lorentzian': ('gamma',), 'voigt': ('sigma', 'gamma')}  # free widths [Å]
DEFAULTS: dict[str, Any] = dict(
    profile='voigt',  # 'gaussian' | 'lorentzian' | 'voigt'
    joint=False,  # one RV shared by every line (per-line depth, widths and continuum)
    outlier_k=5.,  # gross-outlier rejection among lines (0: off)
    shape_correction=False,  # subtract the template-calibrated per-line fit bias (makes the method model-dependent)
)
_EDGE_FRAC = 0.01  # v within this fraction of its range from a bound: at the edge of the line window


def _velocity(ratio: float | np.ndarray) -> float | np.ndarray:
    """Radial velocity [km/s] for λ_obs / λ_rest (the inverse of `broadening.doppler_factor`)"""
    r2 = np.asarray(ratio, float) ** 2
    return C_KMS * (r2 - 1.) / (r2 + 1.)


@dataclass
class _Line:
    """A line's used pixels (line window and continuum windows), flux and σ divided by the continuum level"""
    region: Region
    wave: np.ndarray
    flux: np.ndarray
    sigma: np.ndarray
    core: np.ndarray  # pixels in the line window
    p0: list[float]  # initial [W, *widths, c0, c1]
    v0: float
    v_bounds: tuple[float, float]


def _line(spectrum: Spectrum, region: Region, masks: Sequence[Region], widths: tuple[str, ...]) -> _Line:
    """The pixels of a line and its initial parameters; ValueError if it cannot be fitted"""
    if region.centre is None:
        raise ValueError('no rest wavelength')
    if not region.continuum:
        raise ValueError('no continuum windows')
    windows = [(region.wmin, region.wmax), *region.continuum]
    span = replace(region, wmin=min(w[0] for w in windows), wmax=max(w[1] for w in windows))
    p = C.prepare(spectrum, span, masks)
    core = (p.wave >= region.wmin) & (p.wave <= region.wmax)
    cont = C.in_any(p.wave, region.continuum) & ~core
    use = p.mask & (core | cont)
    n_core, n_cont = int((use & core).sum()), int((use & cont).sum())
    if n_core < len(widths) + 2 or n_cont < 2:  # the line parameters (v, W, widths) need the core pixels
        raise ValueError(f'{n_core} usable line pixels and {n_cont} continuum pixels')
    wave, flux = p.wave[use], p.flux[use]
    sigma = p.sigma[use] if p.sigma_known else np.ones(len(wave))
    core, cont = core[use], cont[use]
    x = wave - region.centre
    c1, c0 = np.polyfit(x[cont], flux[cont], 1, w=1. / sigma[cont])
    level = c0 + c1 * x
    if not np.all(level > 0):
        raise ValueError('non-positive continuum')
    scale = float(np.median(level))
    depth = np.clip(1. - flux[core] / level[core], 0., None)
    ew = float(np.sum(depth * np.gradient(wave)[core]))
    width0 = (region.wmax - region.wmin) / 6.
    v_bounds = (float(_velocity(region.wmin / region.centre)), float(_velocity(region.wmax / region.centre)))
    v0 = float(_velocity(wave[core][np.argmin(flux[core] / level[core])] / region.centre))
    return _Line(region, wave, flux / scale, sigma / scale, core,
                 [max(ew, 1e-3 * width0), *[width0] * len(widths), c0 / scale, c1 / scale], v0, v_bounds)


def _model(lines: list[_Line], profile: str, params: np.ndarray) -> np.ndarray:
    """The concatenated model of every line for params [v, block₁, block₂, ...], block = [W, *widths, c0, c1]"""
    names = PROFILES[profile]
    nb = len(names) + 3
    factor = broadening.doppler_factor(params[0])
    out = []
    for k, ln in enumerate(lines):
        w, *widths, c0, c1 = params[1 + k * nb:1 + (k + 1) * nb]
        width = dict(zip(names, widths, strict=True))
        phi = voigt_profile(ln.wave - ln.region.centre * factor, width.get('sigma', 0.), width.get('gamma', 0.))
        out.append((c0 + c1 * (ln.wave - ln.region.centre)) * (1. - w * phi))
    return np.concatenate(out)


def _fit(lines: list[_Line], profile: str, sigma_known: bool, name: str) -> tuple[float, float, int, float, dict]:
    """Fit lines sharing one v: (v, error, flag, χ²/ν, diagnostics)"""
    names = PROFILES[profile]
    v_lo, v_hi = max(ln.v_bounds[0] for ln in lines), min(ln.v_bounds[1] for ln in lines)
    if v_lo >= v_hi:
        raise ValueError(f'{name}: the line windows have no velocity in common')
    pad = 1e-6 * (v_hi - v_lo)
    v0 = float(np.clip(np.median([ln.v0 for ln in lines]), v_lo + pad, v_hi - pad))
    lo, hi, p0 = [v_lo], [v_hi], [v0]
    # an unresolved width (< half a pixel) lets a huge, narrow profile fit one pixel with a spurious tiny error
    half_pix = np.array([0.5 * float(np.median(np.diff(ln.wave))) for ln in lines])
    for ln, hp in zip(lines, half_pix, strict=True):
        span = ln.region.wmax - ln.region.wmin
        lo += [0., hp, *[0.] * (len(names) - 1), -np.inf, -np.inf]
        hi += [np.inf, *[2. * span] * len(names), np.inf, np.inf]
        p0 += ln.p0
    flux = np.concatenate([ln.flux for ln in lines])
    sigma = np.concatenate([ln.sigma for ln in lines])
    nb = len(names) + 3
    try:
        params, cov = curve_fit(lambda _, *p: _model(lines, profile, np.array(p)), np.zeros(len(flux)), flux,
                                p0=p0, sigma=sigma, absolute_sigma=sigma_known, bounds=(lo, hi), method='trf',
                                x_scale='jac')
    except RuntimeError as err:  # no convergence
        log.warning('%s %s: fit failed (%s)', METHOD, name, err)
        return np.nan, np.nan, C.FLAG_FIT_FAILED, np.nan, dict(profile=profile, error=str(err))
    fit = _model(lines, profile, params)
    chi2 = float(np.sum(((flux - fit) / sigma) ** 2))
    dof = max(len(flux) - len(params), 1)
    chi2_scale = float(np.sqrt(max(1., chi2 / dof))) if sigma_known else 1.  # unknown σ: curve_fit already scales
    v = float(params[0])
    err = float(np.sqrt(cov[0, 0])) * chi2_scale
    flag = C.FLAG_EDGE if min(v - v_lo, v_hi - v) < _EDGE_FRAC * (v_hi - v_lo) else 0
    absent = params[1::nb] <= 1e-6 * np.array([ln.p0[0] for ln in lines])  # no absorption
    widths = sum(params[2 + j::nb] for j in range(len(names)))  # σ + γ for a Voigt profile
    unresolved = widths <= half_pix * (1. + 1e-3)  # a width at its bound: a noise spike, not a line
    if absent.all() or unresolved.any():
        flag |= C.FLAG_FIT_FAILED
        log.warning('%s %s: no line fitted (%s)', METHOD, name, 'no absorption' if absent.all() else 'unresolved')
    per_line = {ln.region.name: dict(zip(['ew', *names, 'c0', 'c1'], map(float, params[1 + k * nb:1 + (k + 1) * nb]),
                                         strict=True)) for k, ln in enumerate(lines)}
    diag = dict(profile=profile, lines=per_line, rest_wave=[ln.region.centre for ln in lines],
                fit_wave=[ln.region.centre * broadening.doppler_factor(v) for ln in lines],
                chi2=chi2, dof=dof, chi2_scale=chi2_scale, cov_err=float(np.sqrt(cov[0, 0])), v_bounds=(v_lo, v_hi),
                n_pix=len(flux), wave=np.concatenate([ln.wave for ln in lines]), data=flux, model=fit, sigma=sigma)
    return v, err, flag, chi2 / dof, diag


def _shape_offsets(spectrum: Spectrum, template: Any, groups: list[list[_Line]], names: list[str],
                   masks: Sequence[Region], profile: str) -> np.ndarray:
    """The RV [km/s] the fit measures on a noise-free, unshifted template on the data's pixels, per group"""
    flux = template.at(spectrum.wave, 0.)
    synth = replace(spectrum, flux=flux, sigma=np.abs(flux) * 1e-3)
    out = []
    for g, n in zip(groups, names, strict=True):
        try:
            sg = [_line(synth, ln.region, masks, PROFILES[profile]) for ln in g]
            out.append(_fit(sg, profile, True, f'{n} (template {template.label})')[0])
        except ValueError as err:
            log.warning('%s %s: no shape offset from template %s (%s)', METHOD, n, template.label, err)
            out.append(np.nan)
    return np.array(out)


def _shape_corrected(spectrum: Spectrum, cfg: Any, groups: list[list[_Line]], names: list[str], fits: list,
                     masks: Sequence[Region], profile: str) -> list:
    """`fits` with each group's template-calibrated offset subtracted and its grid-neighbour spread in the error"""
    from .telluric_ref import neighbour_templates, prior_template
    res = spectrum.resolution or cfg.input.resolution
    if not res:
        raise ValueError('linecentre shape_correction needs the resolving power (input.resolution)')
    wr = (float(spectrum.wave[0]), float(spectrum.wave[-1]))
    nominal = prior_template(cfg, res, wr)
    alts = neighbour_templates(cfg, res, wr, nominal)
    offs = np.array([_shape_offsets(spectrum, t, groups, names, masks, profile) for t in [nominal, *alts]])
    log.info('%s: shape_correction on: per-line offsets from template %s (spread over %s) are subtracted; the '
             'method is now template-dependent and no longer a model-free cross-check', METHOD, nominal.label,
             ', '.join(t.label for t in alts) or 'no neighbours')
    out = []
    for k, (v, err, flag, chi2_red, diag) in enumerate(fits):
        off = float(offs[0, k])
        fin = offs[:, k][np.isfinite(offs[:, k])]
        spread = float(np.std(fin, ddof=1)) if len(fin) > 1 else 0.
        if not np.isfinite(off):
            flag |= C.FLAG_FIT_FAILED
        diag = {**diag, 'shape_correction': True, 'shape_offset': off, 'shape_spread': spread,
                'shape_template': nominal.label, 'rv_uncorrected_shape': v}
        log.info('%s %s: shape offset %+.1f km/s (spread %.1f) subtracted', METHOD, names[k], off, spread)
        out.append((v - off, float(np.hypot(err, spread)), flag, chi2_red, diag))
    return out


def measure(spectrum: Spectrum, regions: Sequence[Region], cfg: Any, *, executor: Executor | None = None,
            rng: np.random.Generator) -> MethodResult:
    """
    Line-centre RVs, one per 'line' region (telluric regions are masked), combined by random effects, or one joint
    RV with `joint`; options in cfg.methods.options['linecentre'] (`DEFAULTS`). `executor` and `rng` are unused
    (a few fast, deterministic fits); they keep the method signature.
    """
    opts = C.options(cfg, METHOD, DEFAULTS)
    profile = opts['profile']
    if profile not in PROFILES:
        raise ValueError(f'linecentre profile must be one of {sorted(PROFILES)}, not {profile!r}')
    masks = [r for r in regions if r.kind in C.MASK_KINDS]
    lines = []
    for region in (r for r in regions if r.kind == 'line'):
        try:
            lines.append(_line(spectrum, region, masks, PROFILES[profile]))
        except ValueError as err:
            log.warning('%s: skipping %s (%s)', METHOD, region.name, err)
    if not lines:
        raise ValueError('no fittable line regions (each needs a rest wavelength, line pixels and continuum windows)')
    sigma_known = bool(np.any(np.isfinite(spectrum.sigma) & (spectrum.sigma > 0)))
    if not sigma_known:
        log.warning('%s: no flux errors; errors scaled by the fit residuals', METHOD)
    if opts['joint']:
        groups = [lines]
        names = ['joint: ' + ', '.join(ln.region.name for ln in lines)]
        windows = [Region(names[0], 'line', min(ln.region.wmin for ln in lines),
                          max(ln.region.wmax for ln in lines), reason=f'joint {profile} fit of {len(lines)} lines')]
    else:
        groups = [[ln] for ln in lines]
        names = [ln.region.name for ln in lines]
        windows = [ln.region for ln in lines]
    fits = [_fit(g, profile, sigma_known, n) for g, n in zip(groups, names, strict=True)]
    if opts['shape_correction']:
        fits = _shape_corrected(spectrum, cfg, groups, names, fits, masks, profile)
    else:  # model-free: NIST rest wavelengths only
        fits = [(v, e, f, c, {**d, 'shape_correction': False}) for v, e, f, c, d in fits]
    measurements = [C.window_measurement(METHOD, win, v, err, flag, chi2_red, sigma_known, diag)
                    for win, (v, err, flag, chi2_red, diag) in zip(windows, fits, strict=True)]
    result = C.finish(METHOD, measurements, opts['outlier_k'])
    log.info('%s: %.2f ± %.2f km/s (%s profile, %d line(s)%s)', METHOD, result.rv, result.rv_err, profile, len(lines),
             ', joint' if opts['joint'] else '')
    return result
