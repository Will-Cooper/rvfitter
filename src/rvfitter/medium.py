"""
Conversions between air and vacuum wavelengths.

All of rvfitter works internally in vacuum Ångström; these are the only functions that convert.
Following the IAU convention, wavelengths below 2000 Å are always vacuum and are returned unchanged.
"""
from collections.abc import Callable

import numpy as np

from .constants import AIR_VACUUM_THRESHOLD

__all__ = ['METHODS', 'refractive_index', 'vac_to_air', 'air_to_vac', 'to_vacuum']


def _ciddor(sigma2: np.ndarray) -> np.ndarray:
    """Ciddor (1996) standard dry air (15 °C, 101 325 Pa, 450 ppm CO2); sigma2 = (1/λ_vac[µm])²."""
    return 1. + 0.05792105 / (238.0185 - sigma2) + 0.00167917 / (57.362 - sigma2)


def _edlen(sigma2: np.ndarray) -> np.ndarray:
    """Edlén (1953) standard air; sigma2 = (1/λ_vac[µm])²."""
    return 1. + 6.4328e-5 + 2.94981e-2 / (146. - sigma2) + 2.5540e-4 / (41. - sigma2)


def _morton(sigma2: np.ndarray) -> np.ndarray:
    """Morton (2000), the IAU standard used by VALD; sigma2 = (1/λ_vac[µm])²."""
    return 1. + 8.34254e-5 + 2.406147e-2 / (130. - sigma2) + 1.5998e-4 / (38.9 - sigma2)


METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {'ciddor': _ciddor, 'edlen': _edlen, 'morton': _morton}


def refractive_index(wave_vac: np.ndarray, method: str = 'ciddor') -> np.ndarray:
    """
    Refractive index of standard air at the given vacuum wavelengths

    Parameters
    ----------
    wave_vac
        Vacuum wavelengths [Å]
    method
        One of 'ciddor' (default), 'edlen' or 'morton'

    Returns
    -------
    n
        The refractive index
    """
    try:
        func = METHODS[method.lower()]
    except KeyError:
        raise ValueError(f'Unknown air/vacuum method {method!r}; choose from {sorted(METHODS)}') from None
    sigma2 = (1e4 / np.asarray(wave_vac, dtype=float)) ** 2
    return func(sigma2)


def vac_to_air(wave_vac: np.ndarray, method: str = 'ciddor') -> np.ndarray:
    """
    Convert vacuum to air wavelengths

    Parameters
    ----------
    wave_vac
        Vacuum wavelengths [Å]
    method
        One of 'ciddor' (default), 'edlen' or 'morton'

    Returns
    -------
    wave_air
        Air wavelengths [Å]; unchanged below 2000 Å
    """
    wave_vac = np.asarray(wave_vac, dtype=float)
    out = wave_vac / refractive_index(wave_vac, method)
    return np.where(wave_vac < AIR_VACUUM_THRESHOLD, wave_vac, out)


def air_to_vac(wave_air: np.ndarray, method: str = 'ciddor', precision: float = 1e-12, maxiter: int = 30) -> np.ndarray:
    """
    Convert air to vacuum wavelengths by iteratively inverting `vac_to_air`

    Parameters
    ----------
    wave_air
        Air wavelengths [Å]
    method
        One of 'ciddor' (default), 'edlen' or 'morton'
    precision
        Relative convergence tolerance
    maxiter
        Maximum number of iterations

    Returns
    -------
    wave_vac
        Vacuum wavelengths [Å]; unchanged below 2000 Å
    """
    wave_air = np.asarray(wave_air, dtype=float)
    wave_vac = wave_air * refractive_index(wave_air, method)  # first guess
    for _ in range(maxiter):
        new = wave_air * refractive_index(wave_vac, method)
        converged = np.all(np.abs(new - wave_vac) <= precision * np.abs(wave_vac))
        wave_vac = new
        if converged:
            break
    return np.where(wave_air < AIR_VACUUM_THRESHOLD, wave_air, wave_vac)


def to_vacuum(wave: np.ndarray, medium: str, method: str = 'ciddor') -> np.ndarray:
    """
    Return vacuum wavelengths given the medium of the input

    Parameters
    ----------
    wave
        Wavelengths [Å]
    medium
        'air' or 'vacuum'
    method
        Conversion formula for air input

    Returns
    -------
    wave_vac
        Vacuum wavelengths [Å]
    """
    medium = medium.lower()
    if medium == 'vacuum':
        return np.asarray(wave, dtype=float)
    if medium == 'air':
        return air_to_vac(wave, method)
    raise ValueError(f"medium must be 'air' or 'vacuum', not {medium!r}")
