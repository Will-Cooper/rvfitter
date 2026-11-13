"""
Conversions between air and vacuum wavelengths.

All of rvfitter works internally in vacuum Ångström; these are the only functions that convert.
Following the IAU convention, wavelengths below 2000 Å are always vacuum and are returned unchanged.
"""
from collections.abc import Callable

import numpy as np

from .constants import AIR_VACUUM_THRESHOLD

__all__ = ['METHODS', 'vac_to_air', 'air_to_vac', 'to_vacuum']


def _ciddor(sigma2: np.ndarray) -> np.ndarray:
    """Ciddor (1996) standard dry air (15 °C, 101 325 Pa, 450 ppm CO2); sigma2 = (1/λ_vac[µm])²."""
    return 1. + 0.05792105 / (238.0185 - sigma2) + 0.00167917 / (57.362 - sigma2)


def _edlen(sigma2: np.ndarray) -> np.ndarray:
    """Edlén (1953) standard air; sigma2 = (1/λ_vac[µm])²."""
    return 1. + 6.4328e-5 + 2.94981e-2 / (146. - sigma2) + 2.5540e-4 / (41. - sigma2)


def _morton(sigma2: np.ndarray) -> np.ndarray:
    """Morton (2000), the IAU standard used by VALD; sigma2 = (1/λ_vac[µm])²."""
    return 1. + 8.34254e-5 + 2.406147e-2 / (130. - sigma2) + 1.5998e-4 / (38.9 - sigma2)


_PRECISION = 1e-12  # relative tolerance of the air_to_vac iteration
_MAXITER = 30
METHODS: dict[str, Callable[[np.ndarray], np.ndarray]] = {'ciddor': _ciddor, 'edlen': _edlen, 'morton': _morton}


def refractive_index(wave_vac: np.ndarray, method: str = 'ciddor') -> np.ndarray:
    """Refractive index of standard air at vacuum wavelengths [Å]; ``method`` is a key of `METHODS`."""
    try:
        func = METHODS[method.lower()]
    except KeyError:
        raise ValueError(f'Unknown air/vacuum method {method!r}; choose from {sorted(METHODS)}') from None
    sigma2 = (1e4 / np.asarray(wave_vac, dtype=float)) ** 2
    return func(sigma2)


def vac_to_air(wave_vac: np.ndarray, method: str = 'ciddor') -> np.ndarray:
    """Convert vacuum to air wavelengths [Å]; unchanged below 2000 Å."""
    wave_vac = np.asarray(wave_vac, dtype=float)
    out = wave_vac / refractive_index(wave_vac, method)
    return np.where(wave_vac < AIR_VACUUM_THRESHOLD, wave_vac, out)


def air_to_vac(wave_air: np.ndarray, method: str = 'ciddor') -> np.ndarray:
    """
    Convert air to vacuum wavelengths [Å] by iteratively inverting `vac_to_air`; unchanged below 2000 Å.

    Raises ``RuntimeError`` if the iteration has not converged after `_MAXITER` steps.
    """
    wave_air = np.asarray(wave_air, dtype=float)
    wave_vac = wave_air * refractive_index(wave_air, method)  # first guess
    for _ in range(_MAXITER):
        new = wave_air * refractive_index(wave_vac, method)
        converged = not np.any(np.abs(new - wave_vac) > _PRECISION * np.abs(wave_vac))
        wave_vac = new
        if converged:
            break
    else:
        raise RuntimeError(f'air_to_vac did not converge in {_MAXITER} iterations')
    return np.where(wave_air < AIR_VACUUM_THRESHOLD, wave_air, wave_vac)


def to_vacuum(wave: np.ndarray, medium: str, method: str = 'ciddor') -> np.ndarray:
    """Vacuum wavelengths [Å] from ``wave`` in ``medium`` ('air' or 'vacuum'); ``method`` applies to air input."""
    medium = medium.lower()
    if medium == 'vacuum':
        return np.asarray(wave, dtype=float)
    if medium == 'air':
        return air_to_vac(wave, method)
    raise ValueError(f"medium must be 'air' or 'vacuum', not {medium!r}")
