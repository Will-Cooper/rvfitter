"""
Doppler factors, flux-conserving rebinning, and the rotational and instrumental kernels on uniform ln λ grids.

Velocities are relativistic radial velocities (λ_obs/λ_rest = √((1 + β)/(1 − β))), not cz.
"""
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve
from scipy.special import ndtr

from .constants import C_KMS

__all__ = ['doppler_factor', 'log_wave_grid', 'rebin', 'gaussian_kernel', 'rotation_kernel', 'convolve',
           'broaden_loglam', 'FWHM_TO_SIGMA']

FWHM_TO_SIGMA = 1. / (2. * np.sqrt(2. * np.log(2.)))
_ROT_SUBSAMPLE = 21  # sub-pixel samples averaging the rotation kernel over each pixel


def doppler_factor(v: float | np.ndarray) -> float | np.ndarray:
    """λ_obs / λ_rest for radial velocity v [km/s], positive away from the observer."""
    beta = np.asarray(v, dtype=float) / C_KMS
    out = np.sqrt((1. + beta) / (1. - beta))
    return float(out) if out.ndim == 0 else out


def log_wave_grid(wmin: float, wmax: float, dv: float) -> np.ndarray:
    """Wavelength grid [Å] uniform in ln λ with step dv [km/s]; wmax is included only if it falls on the grid."""
    if dv <= 0 or wmax <= wmin:
        raise ValueError('need dv > 0 and wmax > wmin')
    dlnw = dv / C_KMS
    n = int(np.floor(np.log(wmax / wmin) / dlnw + 1e-9)) + 1
    return wmin * np.exp(dlnw * np.arange(n))


def _edges(x: np.ndarray) -> np.ndarray:
    """Pixel edges half-way between centres, extrapolated by half a pixel at each end."""
    mid = 0.5 * (x[1:] + x[:-1])
    return np.concatenate([[x[0] - (mid[0] - x[0])], mid, [x[-1] + (x[-1] - mid[-1])]])


def _cumulative_integral(x: np.ndarray, y: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Exact integral from x[0] to `at` (clipped to [x[0], x[-1]]) of the piecewise-linear interpolant of y."""
    cum = np.concatenate([[0.], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])
    at = np.clip(at, x[0], x[-1])
    i = np.clip(np.searchsorted(x, at, side='right') - 1, 0, len(x) - 2)
    dx = at - x[i]
    slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    return cum[i] + y[i] * dx + 0.5 * slope * dx ** 2


def rebin(wave: np.ndarray, flux: np.ndarray, new_wave: np.ndarray) -> np.ndarray:
    """
    Flux-conserving resampling: the mean of the linear interpolant of `flux` over each output pixel.

    Output pixels span half-way to their neighbours. A partly covered pixel is averaged over its covered part;
    uncovered pixels, and pixels that touch a non-finite input value, are NaN.
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
    Approximate propagation of independent per-pixel errors through `rebin`.

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


def gaussian_kernel(sigma_pix: float, nsigma: float = 5.) -> np.ndarray:
    """
    Normalised, odd-length Gaussian kernel, `nsigma` standard deviations wide each side.

    For σ ≥ 1 pixel it is point-sampled (kernel variance σ²); narrower kernels are integrated over each pixel
    (erf differences), so they still smooth rather than collapsing to a delta.
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


def rotation_kernel(dv: float, vsini: float, epsilon: float = 0.6) -> np.ndarray:
    """
    Gray (2005) rotational kernel averaged over pixels of dv [km/s]; normalised, odd length.

    G(x) ∝ 2(1 − ε)√(1 − x²) + (πε/2)(1 − x²) for |x| = |Δv / v sin i| ≤ 1, with linear limb darkening ε.
    """
    if not 0. <= epsilon <= 1.:
        raise ValueError(f'epsilon must lie in [0, 1], not {epsilon}')
    if vsini <= 0:
        return np.ones(1)
    n = int(np.ceil(vsini / dv - 0.5))
    if n == 0:  # the whole kernel lies within one pixel
        return np.ones(1)
    sub = (np.arange(_ROT_SUBSAMPLE) + 0.5) / _ROT_SUBSAMPLE - 0.5
    x = (np.arange(-n, n + 1)[:, None] + sub[None, :]) * dv / vsini
    inside = np.abs(x) < 1
    one_minus = np.where(inside, 1. - x ** 2, 0.)
    g = 2. * (1. - epsilon) * np.sqrt(one_minus) + 0.5 * np.pi * epsilon * one_minus
    k = g.mean(axis=1)
    return k / k.sum()


def convolve(flux: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolve with an odd-length, normalised kernel, padding with the edge values so the ends do not droop.

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
    Rotational, then Gaussian instrumental, broadening of flux on a uniform ln λ grid of step dv [km/s].

    The instrumental kernel is the quadrature difference between `resolution` and `resolution_native` (the
    resolving power already in the flux); None or a non-positive `resolution` skips it.
    """
    out = convolve(flux, rotation_kernel(dv, vsini, epsilon)) if vsini > 0 else np.asarray(flux, dtype=float)
    if resolution is not None and np.isfinite(resolution) and resolution > 0:
        inv2 = 1. / resolution ** 2 - (0. if resolution_native is None else 1. / resolution_native ** 2)
        if inv2 > 0:
            sigma_pix = C_KMS * np.sqrt(inv2) * FWHM_TO_SIGMA / dv
            out = convolve(out, gaussian_kernel(sigma_pix))
    return out
