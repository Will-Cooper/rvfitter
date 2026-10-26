"""
Synthetic spectra with known radial velocity, for injection–recovery tests.

The spectrum is a toy forest of Gaussian absorption lines on a sloped continuum, built on a fine
log-wavelength grid, Doppler shifted (relativistically), rotationally and instrumentally broadened,
resampled at ~2.5 pixels per resolution element and given Gaussian noise.
"""

import numpy as np

from rvfitter.constants import C_KMS
from rvfitter.medium import vac_to_air
from rvfitter.types import Spectrum

# a few real vacuum alkali lines so feature-catalogue tests find something, plus random lines
ANCHOR_LINES = (7667.009, 7701.084, 7802.411, 7949.789, 8185.505, 8197.077, 8523.473, 8945.894,
                11693.4, 11776.2, 12435.7, 12525.6)


def _rotation_kernel(dv: float, vsini: float, epsilon: float = 0.6) -> np.ndarray:
    """Gray (2005) rotational broadening kernel sampled at velocity step dv [km/s]"""
    n = int(np.ceil(vsini / dv))
    x = np.arange(-n, n + 1) * dv / vsini
    x = np.clip(x, -1, 1)
    k = (2 * (1 - epsilon) * np.sqrt(1 - x ** 2) + 0.5 * np.pi * epsilon * (1 - x ** 2))
    return k / k.sum()


def _gaussian_kernel(sigma_pix: float) -> np.ndarray:
    n = int(np.ceil(5 * sigma_pix))
    x = np.arange(-n, n + 1)
    k = np.exp(-0.5 * (x / sigma_pix) ** 2)
    return k / k.sum()


def make_template(wave_range: tuple[float, float] = (7500., 9000.), n_random: int = 60,
                  rng: np.random.Generator | None = None, r_model: float = 200_000.) -> tuple[np.ndarray, np.ndarray]:
    """
    Make a noiseless, unbroadened rest-frame model on a fine log-λ grid

    Returns
    -------
    wave, flux
        Vacuum wavelengths [Å] and normalised flux
    """
    rng = np.random.default_rng(0) if rng is None else rng
    wmin, wmax = wave_range
    dlnw = 1. / (3 * r_model)
    wave = np.exp(np.arange(np.log(wmin * 0.98), np.log(wmax * 1.02), dlnw))
    flux = 1. + 0.1 * (wave - wave.mean()) / (wmax - wmin)
    centres = [c for c in ANCHOR_LINES if wmin * 0.98 < c < wmax * 1.02]
    centres += list(rng.uniform(wmin * 0.98, wmax * 1.02, n_random))
    for c in centres:
        depth = rng.uniform(0.1, 0.6)
        width = c * rng.uniform(5., 25.) / C_KMS  # intrinsic width 5–25 km/s
        flux *= 1. - depth * np.exp(-0.5 * ((wave - c) / width) ** 2)
    return wave, flux


def make_spectrum(rv: float = 0., resolution: float = 2500., snr: float = 50., vsini: float = 0.,
                  wave_range: tuple[float, float] = (7500., 9000.), medium: str = 'vacuum',
                  rng: np.random.Generator | None = None, template: tuple[np.ndarray, np.ndarray] | None = None,
                  pix_per_res: float = 2.5) -> tuple[Spectrum, dict]:
    """
    Make an observed spectrum with known RV

    Parameters
    ----------
    rv
        True radial velocity [km/s]
    resolution
        Instrumental resolving power (Gaussian LSF)
    snr
        Signal-to-noise ratio per pixel (np.inf for no noise)
    vsini
        Projected rotational velocity [km/s]
    wave_range
        Output vacuum wavelength range [Å]
    medium
        Medium of the returned wavelength array ('vacuum' or 'air'); Spectrum.meta['medium_in'] records it
    rng
        Random generator (noise and random lines)
    template
        Optional (wave, flux) rest-frame model; made with `make_template` if None
    pix_per_res
        Pixels per resolution element

    Returns
    -------
    spectrum, truth
        The synthetic spectrum and a dict of the true parameters (including the rest-frame template)
    """
    rng = np.random.default_rng(1) if rng is None else rng
    if template is None:
        template = make_template(wave_range, rng=np.random.default_rng(0))
    wave_t, flux_t = template
    dv = C_KMS * np.diff(np.log(wave_t)).mean()
    flux = flux_t.copy()
    if vsini > 0:
        flux = np.convolve(flux, _rotation_kernel(dv, vsini), mode='same')
    sigma_v = C_KMS / resolution / (2 * np.sqrt(2 * np.log(2)))
    flux = np.convolve(flux, _gaussian_kernel(sigma_v / dv), mode='same')
    beta = rv / C_KMS
    wave_shifted = wave_t * np.sqrt((1 + beta) / (1 - beta))
    wmin, wmax = wave_range
    dlnw_out = 1. / (resolution * pix_per_res)
    wave_out = np.exp(np.arange(np.log(wmin), np.log(wmax), dlnw_out))
    flux_out = np.interp(wave_out, wave_shifted, flux)
    if np.isfinite(snr):
        sigma = flux_out / snr
        flux_out = flux_out + rng.normal(0., sigma)
    else:
        sigma = np.full_like(flux_out, np.nan)
    wave_report = vac_to_air(wave_out) if medium == 'air' else wave_out
    spec = Spectrum(wave_report, flux_out, sigma, resolution,
                    {'object': 'SYNTH', 'medium_in': medium}, 'synthetic')
    truth = dict(rv=rv, resolution=resolution, snr=snr, vsini=vsini, template=template)
    return spec, truth
