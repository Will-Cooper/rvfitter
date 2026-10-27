"""
Barycentric velocity correction from header information (auto, apply or skip).

Sign convention
---------------
The correction ``v_corr`` is the value returned by ``SkyCoord.radial_velocity_correction``: the velocity that must
be *added* to a measured radial velocity to refer it to the barycentre (or the Sun), i.e.
``RV_bary = RV_obs + v_corr`` (to first order; the shift is applied relativistically).
`apply` therefore multiplies the wavelengths by ``sqrt((1 + v_corr/c) / (1 - v_corr/c))``, so that a line seen at
``RV_obs`` is afterwards seen at ``RV_obs + v_corr``.
This is also PypeIt's convention: its ``VEL_CORR`` header value is the multiplicative factor
``sqrt((1 + v/c) / (1 - v/c))`` with the same ``v``, so ``v = c (VEL_CORR**2 - 1) / (VEL_CORR**2 + 1)``.
"""
import dataclasses
import logging
import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from .constants import C_KMS
from .types import Spectrum

__all__ = ['SITES', 'lookup_site', 'correction', 'apply', 'vel_corr_to_kms']

log = logging.getLogger(__name__)

# Built-in offline observatory table: key -> (geodetic latitude [deg], east longitude [deg], altitude [m]).
# Values are rounded public coordinates (good to ~0.01 deg, i.e. << 1 m/s in the correction).
SITES: dict[str, tuple[float, float, float]] = {
    'gtc': (28.7622, -17.8920, 2267.),  # Gran Telescopio Canarias, Roque de los Muchachos
    'irtf': (19.8263, -155.4720, 4168.),  # NASA IRTF, Mauna Kea
    'keck': (19.82833, -155.47833, 4160.),
    'gemini_north': (19.8238, -155.4690, 4213.),
    'gemini_south': (-30.2407, -70.7367, 2722.),
    'vlt': (-24.6272, -70.4042, 2635.),  # Paranal
    'magellan': (-29.0146, -70.6926, 2282.),  # Las Campanas
    'calar_alto': (37.2236, -2.5463, 2168.),
    'mcdonald': (30.6714, -104.0225, 2075.),
    'palomar': (33.3564, -116.8650, 1712.),
    'kitt_peak': (31.9583, -111.5967, 2096.),
    'la_silla': (-29.2563, -70.7380, 2400.),
    'salt': (-32.3760, 20.8120, 1798.),
    'subaru': (19.8255, -155.4761, 4163.),
    'lbt': (32.7013, -109.8891, 3221.),
}
# Lower-case substrings found in TELESCOP/OBSERVAT strings -> key of SITES (first match wins)
_ALIASES: tuple[tuple[str, str], ...] = (
    ('gtc', 'gtc'), ('gran telescopio', 'gtc'), ('roque de los muchachos', 'gtc'),
    ('irtf', 'irtf'), ('infrared telescope facility', 'irtf'), ('mauna kea', 'irtf'), ('maunakea', 'irtf'),
    ('keck', 'keck'),
    ('gemini-n', 'gemini_north'), ('gemini_n', 'gemini_north'), ('gemini north', 'gemini_north'),
    ('gemini-s', 'gemini_south'), ('gemini_s', 'gemini_south'), ('gemini south', 'gemini_south'),
    ('vlt', 'vlt'), ('paranal', 'vlt'),
    ('magellan', 'magellan'), ('las campanas', 'magellan'), ('lco', 'magellan'),
    ('calar alto', 'calar_alto'), ('calar_alto', 'calar_alto'),
    ('mcdonald', 'mcdonald'), ('palomar', 'palomar'), ('kitt peak', 'kitt_peak'), ('kpno', 'kitt_peak'),
    ('la silla', 'la_silla'), ('lasilla', 'la_silla'), ('salt', 'salt'), ('southern african large', 'salt'),
    ('subaru', 'subaru'), ('lbt', 'lbt'), ('large binocular', 'lbt'),
)

# Keywords (compared case-insensitively; meta may hold raw header keys or the lower-case names of
# `Spectrum.meta`).
# Evidence that the spectrum is ALREADY referred to the Sun or barycentre:
#  - PypeIt: VEL_TYPE ('heliocentric' | 'barycentric' | 'None'); VEL_CORR is the multiplicative factor.
#  - `frame_in` in Spectrum.meta; IRAF VHELIO (heliocentric RV of the object, km/s).
# Correction VALUES (computed by a pipeline but not applied to the wavelengths), used only as a FALLBACK when the
# correction cannot be computed, and logged at WARNING:
#  - barycentric: ESO DRS BERV (HIERARCH ESO DRS BERV, or QC BERV), the barycentric Earth radial velocity in km/s,
#    which is subtracted from a barycentric RV to obtain the observer-frame one, i.e. RV_bary = RV_obs + BERV
#    (HARPS DRS products: arXiv:1601.05646 states the header BERV is subtracted from the barycentric velocity to
#    give the observer-frame one; the ESO QC1 database lists the keyword in km/s. Indirect evidence only:
#    check your DRS manual).
#  - NOT used: IRAF/RVSAO BCV/HCV and ESO QC VRAD BARYCOR, because we could not cite their sign convention
#    (IRAF rvcorrect documents only that the correction is applied to an observed velocity).
# Spextool: we could not verify any velocity keyword in Spextool v4 headers (they hold XUNITS, YUNITS, AIRMASS,
# HA, MJD_OBS...), so SpeX spectra are treated as uncorrected. UNVERIFIED: use mode='skip' if your reduction
# already corrected them.
_CORRECTED_FRAMES = {'barycentric': 'barycentric', 'bary': 'barycentric', 'heliocentric': 'heliocentric',
                     'helio': 'heliocentric'}
_CORRECTED_NUMERIC = ('VHELIO',)
_VALUE_KEYS = {
    'barycentric': ('BERV', 'QC BERV', 'HIERARCH ESO QC BERV', 'ESO QC BERV', 'HIERARCH ESO DRS BERV',
                    'ESO DRS BERV'),
    'heliocentric': (),
}


def vel_corr_to_kms(vel_corr: float) -> float:
    """Convert a PypeIt ``VEL_CORR`` factor, sqrt((1+β)/(1-β)), to the velocity v [km/s] it applies."""
    f2 = float(vel_corr) ** 2
    return C_KMS * (f2 - 1.) / (f2 + 1.)


def _alias_match(alias: str, text: str) -> bool:
    """Alias at a word start (and a word end too for short aliases such as 'salt')"""
    return re.search(r'\b' + re.escape(alias) + (r'\b' if len(alias) <= 4 else ''), text) is not None


def lookup_site(name: str):
    """
    Find a site in the built-in offline table

    Parameters
    ----------
    name
        Telescope or observatory name (case-insensitive substring match, e.g. 'GTC', 'IRTF', 'Mauna Kea')

    Returns
    -------
    location
        `astropy.coordinates.EarthLocation`, or None if the site is unknown
    """
    from astropy.coordinates import EarthLocation
    low = str(name).strip().lower()
    key = low if low in SITES else next((k for a, k in _ALIASES if _alias_match(a, low)), None)
    if key is None:
        return None
    lat, lon, alt = SITES[key]
    return EarthLocation.from_geodetic(lon, lat, alt)


def _meta_get(meta: Mapping[str, Any], *keys: str) -> Any:
    """Case-insensitive lookup of the first non-empty key"""
    lower = {str(k).lower(): v for k, v in meta.items()}
    for k in keys:
        v = lower.get(k.lower())
        if v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, float) and math.isnan(v)):
            continue
        return v
    return None


def _already_corrected(meta: Mapping[str, Any]) -> tuple[str, str | None] | None:
    """Return (description, frame or None) if the metadata show the spectrum has been corrected, else None"""
    for key in ('frame_in', 'vel_type'):
        v = _meta_get(meta, key)
        if v is not None and str(v).strip().lower() in _CORRECTED_FRAMES:
            return f'{key}={v}', _CORRECTED_FRAMES[str(v).strip().lower()]
    for key in _CORRECTED_NUMERIC:
        v = _meta_get(meta, key)
        try:
            if v is not None and np.isfinite(float(v)):
                return f'{key}={v}', 'heliocentric'
        except (TypeError, ValueError):
            continue
    return None


def _header_value(meta: Mapping[str, Any], frame: str) -> tuple[float, str] | None:
    """A correction value [km/s] written by the reduction pipeline for ``frame``, with its keyword"""
    for key in _VALUE_KEYS[frame]:
        v = _meta_get(meta, key)
        try:
            if v is not None and np.isfinite(float(v)):
                return float(v), key
        except (TypeError, ValueError):
            continue
    return None


def _location(meta: Mapping[str, Any]):
    from astropy.coordinates import EarthLocation
    lat = _meta_get(meta, 'lat', 'lat-obs', 'latitude', 'sitelat')
    lon = _meta_get(meta, 'lon', 'long', 'lon-obs', 'long-obs', 'longitude', 'sitelong')
    alt = _meta_get(meta, 'alt', 'alt-obs', 'altitude', 'sitealt')
    if lat is not None and lon is not None:
        # Longitude is taken as degrees east (some headers, e.g. GTC's, are west-positive: use `site` there).
        return EarthLocation.from_geodetic(float(lon), float(lat), float(alt or 0.))
    for key in ('site', 'telescop', 'observat'):
        name = _meta_get(meta, key)
        if name is not None:
            loc = lookup_site(name)
            if loc is not None:
                return loc
            log.debug('site %r (%s) is not in the built-in table', name, key)
    return None


def _coord(meta: Mapping[str, Any]):
    from astropy.coordinates import SkyCoord
    ra, dec = _meta_get(meta, 'ra'), _meta_get(meta, 'dec')
    if ra is None or dec is None:
        return None
    if isinstance(ra, str):
        try:
            ra = float(ra)
            dec = float(dec)
        except ValueError:
            return SkyCoord(ra, dec, unit=('hourangle', 'deg'))
    return SkyCoord(float(ra), float(dec), unit='deg')


def _compute(meta: Mapping[str, Any], frame: str) -> tuple[float | None, list[str]]:
    """Compute the correction [km/s]; return (None, missing items) if inputs are incomplete"""
    from astropy.time import Time
    coord, loc, mjd = _coord(meta), _location(meta), _meta_get(meta, 'mjd_mid')
    missing = [n for n, v in (('ra/dec', coord), ('mjd_mid', mjd), ('site (or lat/lon/alt)', loc)) if v is None]
    if missing:
        return None, missing
    obstime = Time(float(mjd), format='mjd', scale='utc')
    return float(coord.radial_velocity_correction(kind=frame, obstime=obstime, location=loc).to('km/s').value), []


def correction(spectrum: Spectrum, mode: str = 'auto', frame: str = 'barycentric',
               overrides: Mapping[str, Any] | None = None) -> tuple[float, str]:
    """
    Decide on and compute the velocity correction for a spectrum

    Parameters
    ----------
    spectrum
        The spectrum; ``spectrum.meta`` may hold ra, dec [deg], mjd_mid, site, lat, lon (east), alt [m],
        frame_in and the raw keywords listed in the module source
    mode
        'auto' (skip if already corrected; else compute if ra, dec, mjd_mid and a site are available; else fall
        back to an ESO BERV header value; else warn and skip), 'apply' (compute; if inputs are missing use a BERV
        header value, else raise `ValueError`) or 'skip' (never correct)
    frame
        'barycentric' or 'heliocentric'
    overrides
        Values replacing the metadata (ra, dec, mjd_mid, site or lat/lon/alt)

    Returns
    -------
    kms, source
        Velocity to add to measured RVs (RV_frame = RV_obs + kms) and its source:
        'header' (already corrected, so 0, or a pipeline correction value), 'computed' or 'skipped'
    """
    if frame not in ('barycentric', 'heliocentric'):
        raise ValueError(f"frame must be 'barycentric' or 'heliocentric', not {frame!r}")
    mode = mode.lower()
    if mode not in ('auto', 'apply', 'skip'):
        raise ValueError(f"mode must be 'auto', 'apply' or 'skip', not {mode!r}")
    if mode == 'skip':
        log.info('Velocity correction skipped by request')
        return 0., 'skipped'
    meta = {**spectrum.meta, **(overrides or {})}
    if mode == 'auto':
        done = _already_corrected(meta)
        if done:
            if done[1] != frame:
                log.warning('Spectrum is already corrected to the %s frame (%s) but %s was requested; the '
                            'difference is up to ~0.02 km/s and is not corrected', done[1], done[0], frame)
            else:
                log.info('Spectrum is already corrected (%s); no further correction applied', done[0])
            return 0., 'header'
    kms, missing = _compute(meta, frame)
    if kms is None:
        hv = _header_value(meta, frame)
        if hv:
            log.warning('Cannot compute the %s correction (missing %s); using %+.3f km/s from header keyword %s '
                        '(RV = RV_obs + value)', frame, ', '.join(missing), hv[0], hv[1])
            return hv[0], 'header'
    if kms is None:
        if mode == 'apply':
            raise ValueError('Cannot compute the velocity correction; missing ' + ', '.join(missing)
                             + ' (supply them in the overrides)')
        log.warning('Cannot compute the %s correction for %s (missing %s); skipping it. '
                    'Measured RVs are then in the observed frame', frame, spectrum.source or 'spectrum',
                    ', '.join(missing))
        return 0., 'skipped'
    log.info('Computed %s correction %+.3f km/s', frame, kms)
    return kms, 'computed'


def apply(spectrum: Spectrum, kms: float) -> Spectrum:
    """
    Shift a spectrum so that its radial velocities increase by ``kms`` (relativistic)

    Wavelengths are multiplied by sqrt((1+β)/(1-β)), β = kms/c, so RV_new = RV_old + kms.
    """
    beta = kms / C_KMS
    return dataclasses.replace(spectrum, wave=spectrum.wave * math.sqrt((1. + beta) / (1. - beta)))
