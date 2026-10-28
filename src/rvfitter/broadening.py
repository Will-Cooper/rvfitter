"""
Doppler shifts, instrumental line-spread function and rotational broadening.

All kernels are applied on grids that are uniform in a velocity-like coordinate, so that the kernel has a
fixed width in pixels:

- the rotational kernel (Gray 2005) and a constant-R Gaussian line-spread function (LSF) on a uniform
  ln λ grid;
- a wavelength-dependent R(λ) on a uniform grid in u(λ) = ∫ R(λ)/λ dλ, on which a Gaussian of
  FWHM λ/R(λ) always has FWHM 1.

Internal grids sample the kernel FWHM with at least ten pixels. Spectra are moved onto them with `rebin`
(the flux-conserving mean of the piecewise-linear interpolant over each pixel) when the input is at least as
finely sampled, e.g. a model, or by cubic-spline interpolation when the internal grid is finer, e.g. observed
data at 2–5 pixels per resolution element. The result is point-sampled back onto the input grid, so a
coarsely sampled input is not broadened by the round trip.

Velocities are relativistic radial velocities (λ_obs/λ_rest = √((1 + β)/(1 − β))), not cz.
"""
import logging
from collections.abc import Callable
from dataclasses import replace

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve
from scipy.special import ndtr

from .constants import C_KMS
from .types import Spectrum

__all__ = ['doppler_factor', 'shift_wave', 'doppler_shift', 'log_wave_grid', 'rebin', 'resample',
           'gaussian_kernel', 'rotation_kernel', 'convolve', 'broaden_loglam', 'instrumental', 'rotational',
           'FWHM_TO_SIGMA']

log = logging.getLogger(__name__)

FWHM_TO_SIGMA = 1. / (2. * np.sqrt(2. * np.log(2.)))
_MIN_PIX_PER_FWHM = 10.  # minimum sampling of the internal grids per kernel FWHM
_UPSAMPLE_RATIO = 1.5  # internal grid this much finer than the input: cubic interpolation instead of rebin

ResolutionLike = float | Callable[[np.ndarray], np.ndarray] | tuple[np.ndarray, np.ndarray]


# ---------------------------------------------------------------------------------------------------------------
# Doppler shifts
# ---------------------------------------------------------------------------------------------------------------
def doppler_factor(v: float | np.ndarray) -> float | np.ndarray:
    """
    The relativistic Doppler factor λ_obs / λ_rest = √((1 + β) / (1 − β)), with β = v / c

    Parameters
    ----------
    v
        Radial velocity [km/s], positive away from the observer

    Returns
    -------
    factor
        The ratio of observed to rest wavelength
    """
    beta = np.asarray(v, dtype=float) / C_KMS
    out = np.sqrt((1. + beta) / (1. - beta))
    return float(out) if out.ndim == 0 else out


def shift_wave(wave: np.ndarray, v: float) -> np.ndarray:
    """
    Doppler shift rest-frame wavelengths to the observed frame of a source moving at velocity v

    Parameters
    ----------
    wave
        Rest-frame vacuum wavelengths [Å]
    v
        Radial velocity [km/s]

    Returns
    -------
    wave_obs
        Observed vacuum wavelengths [Å]
    """
    return np.asarray(wave, dtype=float) * doppler_factor(v)


def doppler_shift(spec: Spectrum, v: float, *, onto_original_grid: bool = False) -> Spectrum:
    """
    Doppler shift a spectrum by velocity v (relativistic)

    Parameters
    ----------
    spec
        The spectrum (vacuum wavelengths)
    v
        Radial velocity [km/s]; positive moves features redwards
    onto_original_grid
        If False (default), only the wavelength axis is scaled. If True, the shifted spectrum is resampled
        (flux-conserving) back onto the input wavelength grid, with NaN where it no longer covers a pixel

    Returns
    -------
    shifted
        The shifted spectrum; meta['doppler_shift_kms'] records the cumulative shift
    """
    meta = dict(spec.meta)
    # successive shifts compose relativistically; the record adds them linearly (exact to ≲1 m/s for |v| < 100 km/s)
    meta['doppler_shift_kms'] = float(meta.get('doppler_shift_kms', 0.)) + float(v)
    wave_new = shift_wave(spec.wave, v)
    if not onto_original_grid:
        return replace(spec, wave=wave_new, meta=meta)
    flux = rebin(wave_new, spec.flux, spec.wave)
    sigma = _rebin_sigma(wave_new, spec.sigma, spec.wave)
    return replace(spec, flux=flux, sigma=sigma, meta=meta)


# ---------------------------------------------------------------------------------------------------------------
# Grids and flux-conserving resampling
# ---------------------------------------------------------------------------------------------------------------
def log_wave_grid(wmin: float, wmax: float, dv: float) -> np.ndarray:
    """
    A wavelength grid uniform in ln λ

    Parameters
    ----------
    wmin, wmax
        Range [Å] (wmax is included if it falls on the grid)
    dv
        Velocity step per pixel [km/s]

    Returns
    -------
    wave
        The wavelength grid [Å]
    """
    if dv <= 0 or wmax <= wmin:
        raise ValueError('need dv > 0 and wmax > wmin')
    dlnw = dv / C_KMS
    n = int(np.floor(np.log(wmax / wmin) / dlnw + 1e-9)) + 1
    return wmin * np.exp(dlnw * np.arange(n))


def _edges(x: np.ndarray) -> np.ndarray:
    """Pixel edges half-way between centres, extrapolated by half a pixel at each end"""
    mid = 0.5 * (x[1:] + x[:-1])
    return np.concatenate([[x[0] - (mid[0] - x[0])], mid, [x[-1] + (x[-1] - mid[-1])]])


def _cumulative_integral(x: np.ndarray, y: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Exact integral from x[0] to `at` (clipped to [x[0], x[-1]]) of the piecewise-linear interpolant of y"""
    cum = np.concatenate([[0.], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])
    at = np.clip(at, x[0], x[-1])
    i = np.clip(np.searchsorted(x, at, side='right') - 1, 0, len(x) - 2)
    dx = at - x[i]
    slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    return cum[i] + y[i] * dx + 0.5 * slope * dx ** 2


def rebin(wave: np.ndarray, flux: np.ndarray, new_wave: np.ndarray) -> np.ndarray:
    """
    Flux-conserving resampling: the mean of the linear interpolant of `flux` over each output pixel

    Output pixels span half-way to their neighbours. Where an output pixel is only partly covered by the input,
    the mean is taken over the covered part; uncovered pixels, and pixels that touch a non-finite input value,
    are NaN. Upsampling reduces to (smoothed) linear interpolation and downsampling to pixel integration.

    Parameters
    ----------
    wave
        Input wavelengths (strictly increasing)
    flux
        Input flux (NaN allowed)
    new_wave
        Output wavelengths (strictly increasing)

    Returns
    -------
    new_flux
        The resampled flux
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    new_wave = np.asarray(new_wave, dtype=float)
    bad = ~np.isfinite(flux)
    if bad.all():
        return np.full(len(new_wave), np.nan)
    if bad.any():
        flux = flux.copy()
        flux[bad] = np.interp(wave[bad], wave[~bad], flux[~bad])
    edges = _edges(new_wave)
    clipped = np.clip(edges, wave[0], wave[-1])
    integral = np.diff(_cumulative_integral(wave, flux, edges))
    width = np.diff(clipped)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = integral / width
    out[width <= 0] = np.nan
    if bad.any():
        badfrac = np.diff(_cumulative_integral(wave, bad.astype(float), edges))
        out[badfrac > 0] = np.nan
    return out


def _rebin_sigma(wave: np.ndarray, sigma: np.ndarray, new_wave: np.ndarray) -> np.ndarray:
    """
    Approximate propagation of independent per-pixel errors through `rebin`

    The variance is rebinned and divided by the number of input pixels per output pixel when downsampling
    (never multiplied when upsampling, where neighbouring output pixels become correlated).
    """
    sigma = np.asarray(sigma, dtype=float)
    if not np.any(np.isfinite(sigma)):
        return np.full(len(new_wave), np.nan)
    var = rebin(wave, sigma ** 2, new_wave)
    dw_in = np.interp(new_wave, wave, np.gradient(wave))
    dw_out = np.gradient(np.asarray(new_wave, dtype=float))
    return np.sqrt(var * np.minimum(1., dw_in / dw_out))


def resample(spec: Spectrum, new_wave: np.ndarray) -> Spectrum:
    """
    Flux-conserving resampling of a spectrum onto a new wavelength grid (see `rebin`)

    Parameters
    ----------
    spec
        The spectrum
    new_wave
        The new vacuum wavelength grid [Å]

    Returns
    -------
    resampled
        The spectrum on the new grid; σ is propagated approximately (see `_rebin_sigma`)
    """
    new_wave = np.asarray(new_wave, dtype=float)
    return replace(spec, wave=new_wave, flux=rebin(spec.wave, spec.flux, new_wave),
                   sigma=_rebin_sigma(spec.wave, spec.sigma, new_wave),
                   sky=None if spec.sky is None else rebin(spec.wave, spec.sky, new_wave))


# ---------------------------------------------------------------------------------------------------------------
# Kernels and convolution on uniform grids
# ---------------------------------------------------------------------------------------------------------------
def gaussian_kernel(sigma_pix: float, nsigma: float = 5.) -> np.ndarray:
    """
    A normalised Gaussian kernel

    For σ ≥ 1 pixel the Gaussian is point-sampled (so the kernel variance is σ²); for narrower kernels it is
    integrated over each pixel (erf differences), so it still smooths rather than collapsing to a delta.

    Parameters
    ----------
    sigma_pix
        Standard deviation [pixels]
    nsigma
        Half-width of the kernel in standard deviations

    Returns
    -------
    kernel
        Odd-length kernel summing to one
    """
    if sigma_pix <= 0:
        return np.ones(1)
    n = max(1, int(np.ceil(nsigma * sigma_pix)))
    x = np.arange(-n, n + 1)
    if sigma_pix >= 1:
        k = np.exp(-0.5 * (x / sigma_pix) ** 2)
    else:
        k = ndtr((x + 0.5) / sigma_pix) - ndtr((x - 0.5) / sigma_pix)
    return k / k.sum()


def rotation_kernel(dv: float, vsini: float, epsilon: float = 0.6, subsample: int = 21) -> np.ndarray:
    """
    The Gray (2005) rotational broadening kernel, averaged over pixels of width dv

    G(x) ∝ 2(1 − ε)√(1 − x²) + (πε/2)(1 − x²) for |x| = |Δv / v sin i| ≤ 1, with linear limb darkening ε.

    Parameters
    ----------
    dv
        Pixel step [km/s]
    vsini
        Projected rotational velocity [km/s]
    epsilon
        Linear limb-darkening coefficient (0–1)
    subsample
        Number of sub-pixel samples used to average the kernel over each pixel

    Returns
    -------
    kernel
        Odd-length kernel summing to one (a single 1 for vsini ≤ 0)
    """
    if not 0. <= epsilon <= 1.:
        raise ValueError(f'epsilon must lie in [0, 1], not {epsilon}')
    if vsini <= 0:
        return np.ones(1)
    n = int(np.ceil(vsini / dv - 0.5))
    if n == 0:  # the whole kernel lies within one pixel
        return np.ones(1)
    sub = (np.arange(subsample) + 0.5) / subsample - 0.5
    x = (np.arange(-n, n + 1)[:, None] + sub[None, :]) * dv / vsini
    inside = np.abs(x) < 1
    one_minus = np.where(inside, 1. - x ** 2, 0.)
    g = 2. * (1. - epsilon) * np.sqrt(one_minus) + 0.5 * np.pi * epsilon * one_minus
    k = g.mean(axis=1)
    return k / k.sum()


def convolve(flux: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve with an odd-length, normalised kernel, padding with the edge values so the ends do not droop

    Non-finite flux is linearly interpolated over before convolving; afterwards every pixel within the kernel
    half-width of a non-finite input is set to NaN, so the filled values never reach a returned pixel.
    """
    flux = np.asarray(flux, dtype=float)
    if len(kernel) == 1:
        return flux.copy()
    bad = ~np.isfinite(flux)
    if bad.all():
        return np.full(len(flux), np.nan)
    half = len(kernel) // 2
    if bad.any():
        idx = np.arange(len(flux))
        flux = flux.copy()
        flux[bad] = np.interp(idx[bad], idx[~bad], flux[~bad])
        bad = binary_dilation(bad, structure=np.ones(2 * half + 1, dtype=bool))
    padded = np.pad(flux, half, mode='edge')
    out = fftconvolve(padded, kernel, mode='valid')
    out[bad] = np.nan
    return out


def broaden_loglam(flux: np.ndarray, dv: float, resolution: float | None = None, vsini: float = 0.,
                   epsilon: float = 0.6, resolution_native: float | None = None) -> np.ndarray:
    """
    Rotational and Gaussian instrumental broadening of a flux array on a uniform ln λ grid

    This is the fast path used inside fitting loops.

    Parameters
    ----------
    flux
        Flux on a grid uniform in ln λ
    dv
        Velocity step per pixel [km/s]
    resolution
        Target resolving power R = λ/FWHM (None for no instrumental broadening)
    vsini
        Projected rotational velocity [km/s]
    epsilon
        Linear limb-darkening coefficient
    resolution_native
        Resolving power already present in the flux; the kernel is the quadrature difference

    Returns
    -------
    broadened
        The broadened flux on the same grid
    """
    out = convolve(flux, rotation_kernel(dv, vsini, epsilon)) if vsini > 0 else np.asarray(flux, dtype=float)
    if resolution is not None and np.isfinite(resolution) and resolution > 0:
        inv2 = 1. / resolution ** 2 - (0. if resolution_native is None else 1. / resolution_native ** 2)
        if inv2 > 0:
            sigma_pix = C_KMS * np.sqrt(inv2) * FWHM_TO_SIGMA / dv
            out = convolve(out, gaussian_kernel(sigma_pix))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Spectrum-level broadening
# ---------------------------------------------------------------------------------------------------------------
def _resolution_callable(resolution: ResolutionLike) -> Callable[[np.ndarray], np.ndarray] | None:
    """None for a constant R, otherwise a function λ → R(λ)"""
    if callable(resolution):
        return resolution
    if isinstance(resolution, tuple):
        rwave, rval = (np.asarray(a, dtype=float) for a in resolution)
        order = np.argsort(rwave)
        rwave, rval = rwave[order], rval[order]
        return lambda w: np.interp(w, rwave, rval)
    return None


def _velocity_step(wave: np.ndarray) -> float:
    """Median velocity step of a wavelength grid [km/s]"""
    return float(np.median(np.diff(np.log(wave)))) * C_KMS


def _onto_internal(wave: np.ndarray, flux: np.ndarray, fine: np.ndarray, upsampling: bool) -> np.ndarray:
    """
    Move flux onto an internal grid: cubic spline when upsampling, flux-conserving rebin otherwise

    The spline is meant for observed (resolved) data; for unresolved features it does not conserve flux exactly
    (+0.3 per cent for a delta function at 6 pixels per FWHM).
    """
    if not upsampling:
        return rebin(wave, flux, fine)
    good = np.isfinite(flux)
    if good.sum() < 4:
        return np.full(len(fine), np.nan)
    out = CubicSpline(wave[good], flux[good])(fine)
    if not good.all():  # any internal pixel next to a bad input pixel is bad
        badfrac = np.interp(fine, wave, (~good).astype(float))
        out[badfrac > 0] = np.nan
    out[(fine < wave[0]) | (fine > wave[-1])] = np.nan
    return out


def _check_native(spec: Spectrum, resolution_native: float | None) -> float | None:
    """Native resolution: the argument, else spec.resolution; np.inf means unresolved"""
    native = spec.resolution if resolution_native is None else resolution_native
    if native is None or not np.isfinite(native):
        return None
    return float(native)


def instrumental(spec: Spectrum, resolution: ResolutionLike, *, resolution_native: float | None = None) -> Spectrum:
    """
    Convolve a spectrum with a Gaussian line-spread function so that its resolution becomes R

    The kernel FWHM is λ/R_eff with R_eff⁻² = R⁻² − R_native⁻², so a spectrum that already has a finite
    resolution is only broadened by the quadrature difference.

    Parameters
    ----------
    spec
        The spectrum
    resolution
        The target: a constant R; a callable λ → R(λ); or a (wave, R) table, linearly interpolated
    resolution_native
        Constant resolving power already present in the spectrum. Default: `spec.resolution`; pass np.inf to
        treat the input as infinitely resolved

    Returns
    -------
    broadened
        The broadened spectrum on the input grid (resolution set to R for a constant R, else to the median R)

    Raises
    ------
    ValueError
        If the target R is not positive and finite, or exceeds the native resolution anywhere
    """
    wave = spec.wave
    rfunc = _resolution_callable(resolution)
    if rfunc is None:
        r_at = np.full(len(wave), float(resolution))
    else:
        r_at = np.asarray(rfunc(wave), dtype=float)
    if np.any(~np.isfinite(r_at)) or np.any(r_at <= 0):
        raise ValueError('R(λ) must be positive and finite over the spectrum')
    r_out = float(np.median(r_at))
    native = _check_native(spec, resolution_native)
    meta = dict(spec.meta)
    meta['lsf_resolution'] = r_out
    if native is not None:
        if np.any(r_at > native * (1 + 1e-12)):
            raise ValueError(f'target resolution (up to {r_at.max():.0f}) is sharper than the native {native:.0f}')
        inv2 = 1. / r_at ** 2 - 1. / native ** 2
        if np.all(inv2 <= 1e-12 / r_at ** 2):  # already at the target
            return replace(spec, resolution=r_out, meta=meta)
        r_at = 1. / np.sqrt(np.maximum(inv2, 1e-24))
    # u(λ) = ∫ R_eff/λ dλ, on which the kernel has FWHM = 1 everywhere
    u_nodes = np.concatenate([[0.], np.cumsum(0.5 * (r_at[1:] / wave[1:] + r_at[:-1] / wave[:-1]) * np.diff(wave))])
    du_native = float(np.median(np.diff(u_nodes)))
    du_target = min(1. / _MIN_PIX_PER_FWHM, du_native)
    u_grid = np.linspace(0., u_nodes[-1], int(np.ceil(u_nodes[-1] / du_target)) + 1)  # both ends included
    du = u_grid[1] - u_grid[0]
    wave_u = np.interp(u_grid, u_nodes, wave)
    flux_u = _onto_internal(wave, spec.flux, wave_u, du_native > _UPSAMPLE_RATIO * du)
    flux_u = convolve(flux_u, gaussian_kernel(FWHM_TO_SIGMA / du))
    flux = np.interp(wave, wave_u, flux_u)
    return replace(spec, flux=flux, resolution=r_out, meta=meta)


def rotational(spec: Spectrum, vsini: float, epsilon: float = 0.6) -> Spectrum:
    """
    Rotationally broaden a spectrum with the Gray (2005) kernel

    Parameters
    ----------
    spec
        The spectrum
    vsini
        Projected rotational velocity [km/s]
    epsilon
        Linear limb-darkening coefficient (0–1)

    Returns
    -------
    broadened
        The broadened spectrum on the input grid
    """
    if vsini < 0:
        raise ValueError(f'vsini must be non-negative, not {vsini}')
    if vsini == 0:
        return spec
    dv_native = _velocity_step(spec.wave)
    dv0 = min(dv_native, 2. * vsini / _MIN_PIX_PER_FWHM)  # the kernel's full width is 2 vsini
    n = int(np.ceil(np.log(spec.wmax / spec.wmin) * C_KMS / dv0)) + 1
    grid = np.geomspace(spec.wmin, spec.wmax, n)  # includes both ends
    dv = float(np.log(grid[1] / grid[0])) * C_KMS
    flux = _onto_internal(spec.wave, spec.flux, grid, dv_native > _UPSAMPLE_RATIO * dv)
    flux = convolve(flux, rotation_kernel(dv, vsini, epsilon))
    meta = dict(spec.meta)
    meta['vsini'] = float(vsini)
    return replace(spec, flux=np.interp(spec.wave, grid, flux), meta=meta)
