"""
Literature-backed catalogue of spectral features (lines, bands, bandheads, telluric bands).

The catalogue is the packaged file ``data/features.ecsv`` and the only place where wavelength windows are written
down.
All wavelengths are vacuum Å; NaN marks a column that does not apply (e.g. ``wave_vac`` for a band).
"""
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

from astropy.table import Table

__all__ = ['Feature', 'load_catalogue', 'features', 'KINDS']

log = logging.getLogger(__name__)

KINDS = ('line', 'band', 'bandhead', 'telluric')
_CORE_ONLY = 'core-only window'


@dataclass(frozen=True)
class Feature:
    """
    One catalogue entry

    Attributes
    ----------
    name, species
        Label (e.g. 'K I 7701') and species (e.g. 'K I')
    kind
        'line' | 'band' | 'bandhead' | 'telluric'
    wave_vac
        Vacuum rest wavelength [Å] of a line or bandhead (NaN otherwise)
    wmin, wmax
        Vacuum band limits [Å] (NaN for lines)
    teff_min, teff_max
        Teff range [K] where the feature is useful (NaN = unrestricted)
    r_min
        Minimum resolving power for RV use (for telluric bands: for use as a wavelength reference)
    rest_ref, caveat
        Reference for the wavelength, Teff range and R, and known caveats
    depth
        Telluric bands only: approximate fractional absorption depth at R ~ 2000 and airmass 1
    model_fidelity
        How well synthetic grids reproduce the feature in late-M/L/T dwarfs: 'good' (atomic lines, CO bandheads),
        'poor' (TiO, VO, CrH, FeH, H2O, CH4 molecular bands; see the catalogue description) or 'n/a' (telluric)
    """
    name: str
    species: str
    kind: str
    wave_vac: float
    wmin: float
    wmax: float
    teff_min: float
    teff_max: float
    r_min: float
    rest_ref: str
    caveat: str
    depth: float
    model_fidelity: str = 'n/a'

    @property
    def core_only(self) -> bool:
        """True if only the line core should be used (pressure-broadened wings)"""
        return _CORE_ONLY in self.caveat

    @property
    def teff_range(self) -> tuple[float, float]:
        return self.teff_min, self.teff_max

    @property
    def span(self) -> tuple[float, float]:
        """(low, high) wavelength limits; a line has zero width at its rest wavelength"""
        if self.kind == 'line':
            return self.wave_vac, self.wave_vac
        return self.wmin, self.wmax

    def teff_ok(self, teff: float | None) -> bool:
        """Whether the feature is useful at ``teff`` (always True if teff is None or the range is unrestricted)"""
        if teff is None or not math.isfinite(teff):
            return True
        lo = -math.inf if math.isnan(self.teff_min) else self.teff_min
        hi = math.inf if math.isnan(self.teff_max) else self.teff_max
        return lo <= teff <= hi


@lru_cache(maxsize=1)
def _read() -> Table:
    path = resources.files('rvfitter') / 'data' / 'features.ecsv'
    with resources.as_file(path) as p:
        table = Table.read(p, format='ascii.ecsv')
    bad = set(table['kind']) - set(KINDS)
    if bad:
        raise ValueError(f'features.ecsv has unknown kinds {sorted(bad)}')
    log.debug('Loaded %d catalogue features', len(table))
    return table


def load_catalogue() -> Table:
    """
    Load the feature catalogue (cached)

    Returns
    -------
    table
        A copy of the cached `astropy.table.Table`, so callers may modify it freely
    """
    return _read().copy()


@lru_cache(maxsize=1)
def features() -> tuple[Feature, ...]:
    """The catalogue as a tuple of `Feature` records (cached)"""
    table = _read()
    return tuple(Feature(str(r['name']), str(r['species']), str(r['kind']), float(r['wave_vac']), float(r['wmin']),
                         float(r['wmax']), float(r['teff_min']), float(r['teff_max']), float(r['r_min']),
                         str(r['rest_ref']), str(r['caveat']), float(r['depth']),
                         str(r['model_fidelity']) if 'model_fidelity' in table.colnames else 'n/a') for r in table)
