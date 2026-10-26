"""
Doppler shifts, instrumental line-spread function and rotational broadening.

All kernels are applied on grids that are uniform in a velocity-like coordinate, so that the kernel has a
fixed width in pixels:

- the rotational kernel (Gray 2005) and a constant-R Gaussian line-spread function (LSF) on a uniform
  ln λ grid;
- a wavelength-dependent R(λ) on a uniform grid in u(λ) = ∫ R(λ)/λ dλ, on which a Gaussian of
  FWHM λ/R(λ) always has FWHM 1.

Spectra are moved between grids with `rebin`, which returns the mean of the piecewise-linear interpolant
over each output pixel, so the integral of the flux is conserved (to the extent the input grid covers the
output pixel).
The broadened flux is returned on the input wavelength grid.
"""
import logging
from collections.abc import Callable
from dataclasses import replace

import numpy as np
from scipy.signal import fftconvolve

from .constants import C_KMS
from .types import Spectrum

__all__ = ['doppler_factor', 'shift_wave', 'doppler_shift', 'log_wave_grid', 'rebin', 'resample',
           'gaussian_kernel', 'rotation_kernel', 'convolve', 'broaden_loglam', 'instrumental', 'rotational',
           'FWHM_TO_SIGMA']

log = logging.getLogger(__name__)

FWHM_TO_SIGMA = 1. / (2. * np.sqrt(2. * np.log(2.)))
_MIN_PIX_PER_FWHM = 5.  # sampling of the internal grids per resolution element

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
                   sigma=_rebin_sigma(spec.wave, spec.sigma, new_wave))


# ---------------------------------------------------------------------------------------------------------------
# Kernels and convolution on uniform grids
# ---------------------------------------------------------------------------------------------------------------
def gaussian_kernel(sigma_pix: float, nsigma: float = 5.) -> np.ndarray:
    """
    A normalised, sampled Gaussian kernel

    Parameters
    ----------
    sigma_pix
        Standard deviation [pixels]; should be ≳1 for the sampling to be accurate
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
    k = np.exp(-0.5 * (x / sigma_pix) ** 2)
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

    Non-finite flux is linearly interpolated over before convolving and set back to NaN afterwards.
    """
    flux = np.asarray(flux, dtype=float)
    if len(kernel) == 1:
        return flux.copy()
    bad = ~np.isfinite(flux)
    if bad.any():
        idx = np.arange(len(flux))
        flux = flux.copy()
        flux[bad] = np.interp(idx[bad], idx[~bad], flux[~bad])
    half = len(kernel) // 2
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
            if sigma_pix < 1:
                log.debug('Gaussian LSF of %.2f pixels is undersampled', sigma_pix)
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


def instrumental(spec: Spectrum, resolution: ResolutionLike, *, resolution_native: float | None = None) -> Spectrum:
    """
    Convolve a spectrum with a Gaussian line-spread function of FWHM λ / R

    Parameters
    ----------
    spec
        The spectrum
    resolution
        A constant R; a callable λ → R(λ); or a (wave, R) table, linearly interpolated
    resolution_native
        Constant resolving power already present in the spectrum (e.g. a model's), removed in quadrature;
        None treats the input as infinitely resolved

    Returns
    -------
    broadened
        The broadened spectrum on the input grid (resolution set to R for a constant R, else to the median R)
    """
    wave = spec.wave
    rfunc = _resolution_callable(resolution)
    if rfunc is None:
        r_at = np.full(len(wave), float(resolution))
    else:
        r_at = np.asarray(rfunc(wave), dtype=float)
    r_out = float(np.median(r_at))
    if np.any(~np.isfinite(r_at)) or np.any(r_at <= 0):
        raise ValueError('R(λ) must be positive and finite over the spectrum')
    if resolution_native is not None:
        inv2 = 1. / r_at ** 2 - 1. / resolution_native ** 2
        if np.any(inv2 <= 0):
            raise ValueError('target resolution must be lower than the native resolution everywhere')
        r_at = 1. / np.sqrt(inv2)
    # u(λ) = ∫ R/λ dλ, on which the LSF has FWHM = 1 everywhere
    u_nodes = np.concatenate([[0.], np.cumsum(0.5 * (r_at[1:] / wave[1:] + r_at[:-1] / wave[:-1]) * np.diff(wave))])
    du_native = float(np.median(np.diff(u_nodes)))
    nu = int(np.ceil(u_nodes[-1] / min(1. / _MIN_PIX_PER_FWHM, du_native))) + 1
    u_grid = np.linspace(0., u_nodes[-1], nu)  # includes both ends, so every input pixel stays covered
    du = u_grid[1] - u_grid[0]
    wave_u = np.interp(u_grid, u_nodes, wave)
    flux_u = rebin(wave, spec.flux, wave_u)
    flux_u = convolve(flux_u, gaussian_kernel(FWHM_TO_SIGMA / du))
    flux = rebin(wave_u, flux_u, wave)
    meta = dict(spec.meta)
    meta['lsf_resolution'] = r_out
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
    dv0 = min(_velocity_step(spec.wave), vsini / 10.)
    n = int(np.ceil(np.log(spec.wmax / spec.wmin) * C_KMS / dv0)) + 1
    grid = np.geomspace(spec.wmin, spec.wmax, n)  # includes both ends
    dv = float(np.log(grid[1] / grid[0])) * C_KMS
    flux = rebin(spec.wave, spec.flux, grid)
    flux = convolve(flux, rotation_kernel(dv, vsini, epsilon))
    meta = dict(spec.meta)
    meta['vsini'] = float(vsini)
    return replace(spec, flux=rebin(grid, flux, spec.wave), meta=meta)
