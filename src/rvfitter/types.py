"""
Core data types shared by every rvfitter module (see docs/architecture.md).

Wavelengths are vacuum Ångström and velocities km/s throughout.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ['Spectrum', 'Region', 'Measurement', 'MethodResult', 'ObjectResult', 'FLAG_OK']

FLAG_OK = 0


@dataclass(frozen=True)
class Spectrum:
    """
    A one-dimensional spectrum in vacuum wavelengths

    Attributes
    ----------
    wave
        Vacuum wavelengths [Å], strictly increasing and finite
    flux
        Flux in arbitrary units
    sigma
        1σ flux uncertainty; NaN where unknown
    resolution
        Resolving power R = λ/Δλ(FWHM); None if unknown
    meta
        Header-derived metadata (object, ra, dec, mjd_mid, site, exptime, airmass, medium_in, frame_in, vel_corr…)
    source
        File path or archive member the spectrum was read from
    sky
        Optional sky spectrum on the same pixels (e.g. PypeIt OPT_COUNTS_SKY), used for the sky-line wavelength
        zero point; None if absent
    """
    wave: np.ndarray
    flux: np.ndarray
    sigma: np.ndarray
    resolution: float | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)
    source: str = ''
    sky: np.ndarray | None = None

    def __post_init__(self):
        wave = np.asarray(self.wave, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        sigma = np.asarray(self.sigma, dtype=float)
        if not (wave.ndim == flux.ndim == sigma.ndim == 1) or not (len(wave) == len(flux) == len(sigma)):
            raise ValueError('wave, flux and sigma must be one-dimensional arrays of equal length')
        if len(wave) < 2 or not np.all(np.isfinite(wave)) or np.any(np.diff(wave) <= 0):
            raise ValueError('wave must be finite and strictly increasing with at least two pixels')
        object.__setattr__(self, 'wave', wave)
        object.__setattr__(self, 'flux', flux)
        object.__setattr__(self, 'sigma', sigma)
        if self.sky is not None:
            sky = np.asarray(self.sky, dtype=float)
            if sky.shape != wave.shape:
                raise ValueError('sky must have the same length as wave')
            object.__setattr__(self, 'sky', sky)

    @property
    def wmin(self) -> float:
        return float(self.wave[0])

    @property
    def wmax(self) -> float:
        return float(self.wave[-1])

    def cut(self, wmin: float, wmax: float) -> 'Spectrum':
        """Return the part of the spectrum within [wmin, wmax]"""
        keep = (self.wave >= wmin) & (self.wave <= wmax)
        return Spectrum(self.wave[keep], self.flux[keep], self.sigma[keep], self.resolution, self.meta, self.source,
                        None if self.sky is None else self.sky[keep])


@dataclass(frozen=True)
class Region:
    """A wavelength region used by a method, with the reason it was chosen"""
    name: str
    kind: str  # 'line' | 'band' | 'ccf' | 'telluric' | 'continuum'
    wmin: float
    wmax: float
    centre: float | None = None
    continuum: tuple[tuple[float, float], ...] = ()
    reason: str = ''


@dataclass
class Measurement:
    """One method applied to one region or order"""
    method: str
    region: str
    rv: float
    rv_err: float
    flag: int = FLAG_OK
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class MethodResult:
    """A method's measurements, combined over regions"""
    method: str
    rv: float
    rv_err: float
    n_regions: int
    measurements: list[Measurement] = field(default_factory=list)


@dataclass
class ObjectResult:
    """Everything measured for one object; one row of the results table"""
    object: str
    files: list[str]
    methods: dict[str, MethodResult] = field(default_factory=dict)
    adopted: dict[str, Any] = field(default_factory=dict)
    systematics: dict[str, float] = field(default_factory=dict)
    barycentric: dict[str, Any] = field(default_factory=dict)
    binary: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ''
    version: str = ''
    refitted: bool = False  # measured with a per-object [[refits]] patch (see `pipeline.refit_object`)
