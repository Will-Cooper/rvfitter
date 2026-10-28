"""
Reading spectra (FITS, text, PypeIt) from files, directories and archives, and grouping them into objects.

`read_spectrum` is the single entry point.
It tries a registry of readers in order (PypeIt spec1d, PypeIt coadd/tellcorr tables, generic 1D FITS, ASCII/CSV/TSV)
and finally falls back to ``specutils``.
Further readers can be added with `register_reader`.

Conventions applied to every spectrum, whatever the reader:

* σ = 1/√ivar, with NaN where ivar ≤ 0.
* PypeIt pixels masked in ``OPT_MASK`` (``BOX_MASK`` for box extraction) are dropped.
* Non-finite (or non-positive) wavelengths and non-finite fluxes are dropped, the rest is sorted by wavelength and
  duplicate wavelengths are removed (the first is kept).
* The wavelength medium is never guessed silently.
  With ``medium='auto'`` it is read from the header
  (``AIRORVAC``, ``VACUUM``, ``WAVE_MED`` or ``CTYPE1`` 'AWAV'/'WAVE').
  PypeIt products carry no such keyword and are documented to be in vacuum, so PypeIt readers fall back to
  vacuum (recorded in ``meta['medium_source']``).
  Anything else raises a `ValueError` asking for ``medium='air'`` or ``medium='vacuum'``.
  The wavelengths are then converted to vacuum with `rvfitter.medium`.

Header keywords used for the metadata (`Spectrum.meta`):

========== =============================================================================================
object      ``OBJECT``, ``TARGNAME``, ``TARGET``, ``OBJNAME`` (first non-empty)
ra, dec     ``RA``/``DEC`` (or ``OBJRA``/``OBJDEC``, ``TELRA``/``TELDEC``); floats are degrees,
            sexagesimal strings are hours (RA) and degrees (Dec)
mjd_mid     ``MJD_MID``/``MJD-MID`` if present (already mid-exposure).
            Otherwise ``MJD``, ``MJD-OBS`` or ``DATE-OBS`` (+ ``TIME-OBS``), which all mean the *start* of the
            exposure, plus ``EXPTIME``/2.
            ``meta['mjd_source']`` records which was used.
site        ``OBSERVAT`` if present, else ``TELESCOP`` (``meta['telescope']`` is always ``TELESCOP``)
exptime     ``EXPTIME``, ``EXPOSURE`` or ``ITIME`` [s]
airmass     ``AIRMASS``, else the mean of ``AMSTART`` and ``AMEND``
medium_in   'air' or 'vacuum': the medium of the file as read (before conversion)
frame_in    ``VEL_TYPE`` (e.g. 'barycentric', 'heliocentric') if present, else None
vel_corr    ``VEL_CORR`` if present (PypeIt: the multiplicative factor already applied to the wavelengths);
            ``vel_corr_kms`` is the same as a velocity, c(f²-1)/(f²+1)
arc_range   (PypeIt only) [min, max] vacuum Å of the fitted arc lines of a ``wvcalib.fits`` / ``WaveCalib*.fits`` found
            next to the file or in ``Calibrations/`` (`find_wvcalib`); ``arc_file`` is its path.
            PypeIt stores vacuum wavelengths; they are converted like the spectrum if it was read as air.
========== =============================================================================================

``meta['velocity_keywords']`` additionally holds any of the velocity-correction keywords found in the headers
(``VHELIO``, ``BCV``, ``BERV``, ``VELOSYS``, ``QC VRAD BARYCOR`` …), so that the barycentric module can tell whether a
spectrum is already corrected.
"""
import gzip
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from .constants import C_KMS
from .medium import to_vacuum
from .types import Spectrum

__all__ = ['read_spectrum', 'register_reader', 'RawSpectrum', 'Reader', 'readers']

log = logging.getLogger(__name__)

# PypeIt wavelengths are vacuum by default (documented PypeIt behaviour); used only when the header is silent
PYPEIT_DEFAULT_MEDIUM = 'vacuum'

_WAVE_NAMES = ('wave', 'wavelength', 'lambda', 'lam', 'wl', 'wavel', 'wavelengths')
_FLUX_NAMES = ('flux', 'flam', 'flux_density', 'fluxdensity', 'fluxes')
_ERR_NAMES = ('err', 'error', 'sigma', 'flux_err', 'fluxerr', 'flux_error', 'e_flux', 'uncertainty', 'unc', 'noise',
              'errors', 'sig')
_IVAR_NAMES = ('ivar', 'invvar', 'inverse_variance', 'flux_ivar')
_VAR_NAMES = ('var', 'variance')
_SKY_NAMES = ('sky', 'sky_flux', 'skyflux', 'counts_sky', 'sky_counts')
_ROLE_NAMES = {'wave': _WAVE_NAMES, 'flux': _FLUX_NAMES, 'err': _ERR_NAMES, 'ivar': _IVAR_NAMES, 'var': _VAR_NAMES,
               'sky': _SKY_NAMES}

_RES_KEYS = ('RESOLUTION', 'SPECRES', 'SPEC_RES', 'RESOLVING', 'RESPOWER', 'RESOLPOW')
_VEL_KEYS = ('VHELIO', 'VBARY', 'VLSR', 'BCV', 'BERV', 'BARYCORR', 'HELIOCOR', 'BARY_CORR', 'VELOSYS', 'HELCORR',
             'QC VRAD BARYCOR', 'QC VRAD HELICOR', 'VEL_TYPE', 'VEL_CORR', 'RADVEL', 'VELFRAME')

_UNIT_SCALES = {  # to Å
    'a': 1., 'å': 1., 'aa': 1., 'ang': 1., 'angs': 1., 'angstrom': 1., 'angstroms': 1., 'ångström': 1.,
    'angstroem': 1.,
    'nm': 10., 'nanometer': 10., 'nanometers': 10., 'nanometre': 10., 'nanometres': 10.,
    'um': 1e4, 'µm': 1e4, 'μm': 1e4, 'micron': 1e4, 'microns': 1e4, 'micrometer': 1e4, 'micrometre': 1e4,
    'micrometers': 1e4, 'micrometres': 1e4,
    'mm': 1e7, 'm': 1e10, 'cm': 1e8,
}


@dataclass
class RawSpectrum:
    """
    What a reader returns, before the common clean-up.

    Attributes
    ----------
    wave
        Wavelengths [Å] in the medium of the file
    flux
        Flux in arbitrary units
    sigma
        1σ uncertainties (NaN where unknown)
    header
        Flat, upper-case keyword → value mapping (primary header overridden by the extension header)
    medium
        'air' or 'vacuum' if the file says so, else None
    medium_source
        Where ``medium`` came from (e.g. 'header:AIRORVAC')
    default_medium
        Documented default for this kind of file when the header is silent (PypeIt: 'vacuum'), else None
    resolution
        R if the reader found one
    reader
        Name of the reader that produced it
    sky
        Optional sky spectrum on the same pixels (PypeIt ``*_COUNTS_SKY``, or a ``sky`` column), else None
    """
    wave: np.ndarray
    flux: np.ndarray
    sigma: np.ndarray
    header: dict[str, Any] = field(default_factory=dict)
    medium: str | None = None
    medium_source: str = ''
    default_medium: str | None = None
    resolution: float | None = None
    reader: str = ''
    sky: np.ndarray | None = None


@dataclass(frozen=True)
class Reader:
    """A registered reader: ``sniff(path, hdu, columns) -> bool`` and ``read(path, hdu, columns) -> RawSpectrum``"""
    name: str
    sniff: Callable[..., bool]
    read: Callable[..., RawSpectrum]


_READERS: list[Reader] = []


def register_reader(name: str, sniff: Callable[..., bool], read: Callable[..., RawSpectrum], *,
                    first: bool = False) -> None:
    """
    Register a spectrum reader.

    Parameters
    ----------
    name
        Unique name; registering an existing name replaces it
    sniff
        ``sniff(path, hdu=None, columns=None) -> bool``; whether the reader understands the file.
        Exceptions are treated as False.
    read
        ``read(path, hdu=None, columns=None) -> RawSpectrum`` with wavelengths in Å
    first
        Put the reader at the front of the queue (so that it takes priority over the built-in ones)
        rather than at the end
    """
    reader = Reader(name, sniff, read)
    for i, old in enumerate(_READERS):
        if old.name == name:
            _READERS[i] = reader
            return
    if first:
        _READERS.insert(0, reader)
    else:
        _READERS.append(reader)


def readers() -> list[str]:
    """Names of the registered readers, in the order they are tried"""
    return [r.name for r in _READERS]


# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------

def _clean_value(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, bytes):
        return _clean_value(value.decode(errors='replace'))
    return value


def _header_dict(*headers: fits.Header) -> dict[str, Any]:
    """Flatten FITS headers into one upper-case dict; later headers override earlier ones"""
    out: dict[str, Any] = {}
    for hdr in headers:
        for card in hdr.cards:
            key = card.keyword.upper().strip()
            if key in ('', 'COMMENT', 'HISTORY', 'END'):
                continue
            try:
                value = _clean_value(card.value)
            except Exception:  # undefined card value
                continue
            if value is not None and not isinstance(value, fits.card.Undefined):
                out[key] = value
    return out


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _first(header: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[Any, str | None]:
    for key in keys:
        value = _clean_value(header.get(key))
        if value is not None:
            return value, key
    return None, None


def _ivar_to_sigma(ivar: np.ndarray) -> np.ndarray:
    """σ = 1/√ivar, NaN where ivar ≤ 0 or non-finite"""
    ivar = np.asarray(ivar, dtype=float)
    sigma = np.full(ivar.shape, np.nan)
    good = np.isfinite(ivar) & (ivar > 0)
    sigma[good] = 1. / np.sqrt(ivar[good])
    return sigma


def _clean_sigma(sigma: np.ndarray) -> np.ndarray:
    sigma = np.array(sigma, dtype=float)
    sigma[~np.isfinite(sigma) | (sigma <= 0)] = np.nan
    return sigma


def _unit_scale(unit: Any) -> float | None:
    """Factor to convert a wavelength unit string to Å, or None if it is not a length unit"""
    if unit is None:
        return None
    text = str(unit).strip().lower().replace(' ', '')
    if not text:
        return None
    if text in _UNIT_SCALES:
        return _UNIT_SCALES[text]
    try:
        from astropy import units as u
        return float(u.Unit(str(unit).strip()).to(u.AA))
    except Exception:
        return None


def _wave_to_angstrom(wave: np.ndarray, unit: Any, source: str) -> np.ndarray:
    """
    Convert to Å using the declared unit.
    Without one, guess from the magnitude (µm below 100, nm below 3000, else Å) and warn.
    """
    scale = _unit_scale(unit)
    if scale is not None:
        return wave * scale
    finite = wave[np.isfinite(wave) & (wave > 0)]
    median = float(np.median(finite)) if finite.size else 0.
    if median < 100.:
        scale, guess = 1e4, 'µm'
    elif median < 3000.:
        scale, guess = 10., 'nm'
    else:
        scale, guess = 1., 'Å'
    log.warning('%s: no wavelength unit given; assuming %s from the median wavelength (%.4g)', source, guess, median)
    return wave * scale


def _medium_from_header(header: Mapping[str, Any]) -> tuple[str | None, str]:
    """Return ('air'|'vacuum'|None, source) from the header"""
    def from_text(text: Any) -> str | None:
        t = str(text).strip().lower()
        if t.startswith('vac'):
            return 'vacuum'
        if t.startswith('air') or t == 'a':
            return 'air'
        return None

    for key in ('AIRORVAC', 'MEDIUM', 'WAVE_MED', 'WAVEMED', 'WAVE_MEDIUM', 'WAVEMEDIUM', 'WAVMED'):
        if key in header:
            medium = from_text(header[key])
            if medium:
                return medium, f'header:{key}'
    if 'VACUUM' in header:
        value = header['VACUUM']
        if isinstance(value, str):
            value = value.strip().lower() in ('t', 'true', 'yes', '1')
        return ('vacuum' if value else 'air'), 'header:VACUUM'
    for key in ('CTYPE1', 'CTYPE'):
        if key in header:
            code = str(header[key]).strip().upper().split('-')[0]
            if code == 'AWAV':
                return 'air', f'header:{key}'
            if code == 'WAVE':
                return 'vacuum', f'header:{key}'
    return None, ''


def _parse_coord(value: Any, is_ra: bool) -> float | None:
    if value is None:
        return None
    number = _as_float(value)
    if number is None and isinstance(value, str):
        try:
            from astropy import units as u
            from astropy.coordinates import Angle
            number = float(Angle(value.strip(), unit=u.hourangle if is_ra else u.deg).deg)
        except Exception:
            return None
    if number is None:
        return None
    lim = 360. if is_ra else 90.
    if (is_ra and not 0. <= number <= lim) or (not is_ra and abs(number) > lim):
        return None
    return number


def _date_to_mjd(header: Mapping[str, Any]) -> float | None:
    date, _ = _first(header, ('DATE-OBS', 'DATE_OBS', 'DATEOBS'))
    if date is None:
        return None
    date = str(date)
    if 'T' not in date and ':' not in date:
        clock, _ = _first(header, ('TIME-OBS', 'TIME_OBS', 'UT', 'UTSTART'))
        if clock is not None:
            date = f'{date}T{clock}'
    try:
        from astropy.time import Time
        return float(Time(date, format='isot', scale='utc').mjd)
    except Exception:
        return None


def _meta_from_header(header: Mapping[str, Any]) -> dict[str, Any]:
    """Build the metadata dictionary of a `Spectrum` from a flat header"""
    meta: dict[str, Any] = {}
    obj, _ = _first(header, ('OBJECT', 'TARGNAME', 'TARGET', 'OBJNAME'))
    if obj is not None and str(obj).strip().lower() not in ('none', 'unknown', 'n/a', ''):
        meta['object'] = str(obj)
    ra, _ = _first(header, ('RA', 'OBJRA', 'TELRA', 'RA_DEG'))
    dec, _ = _first(header, ('DEC', 'OBJDEC', 'TELDEC', 'DEC_DEG'))
    meta['ra'] = _parse_coord(ra, True)
    meta['dec'] = _parse_coord(dec, False)
    exptime = _as_float(_first(header, ('EXPTIME', 'EXPOSURE', 'ITIME'))[0])
    meta['exptime'] = exptime
    # mid-exposure time
    mjd_mid, source = None, None
    mid, key = _first(header, ('MJD_MID', 'MJD-MID', 'MJDMID'))
    if _as_float(mid) is not None:
        mjd_mid, source = _as_float(mid), key
    else:
        start, key = _first(header, ('MJD-OBS', 'MJD_OBS', 'MJD'))
        start = _as_float(start)
        if start is None:
            start, key = _date_to_mjd(header), 'DATE-OBS'
        if start is not None:
            half = (exptime or 0.) / 2. / 86400.
            mjd_mid = start + half
            source = f'{key}(start)' + ('+EXPTIME/2' if exptime else ' (EXPTIME unknown; start time used)')
    meta['mjd_mid'] = mjd_mid
    meta['mjd_source'] = source
    site, _ = _first(header, ('OBSERVAT', 'TELESCOP', 'SITENAME'))
    meta['site'] = None if site is None else str(site)
    telescope, _ = _first(header, ('TELESCOP',))
    meta['telescope'] = None if telescope is None else str(telescope)
    lat = _as_float(_first(header, ('SITELAT', 'OBSLAT', 'LATITUDE', 'LAT-OBS', 'OBS_LAT'))[0])
    lon = _as_float(_first(header, ('SITELONG', 'OBSLON', 'OBSLONG', 'LONGITUD', 'LONG-OBS', 'OBS_LON'))[0])
    if lat is not None and lon is not None:
        meta['site_lat'], meta['site_lon'] = lat, lon
    airmass = _as_float(_first(header, ('AIRMASS',))[0])
    if airmass is None:
        a0, a1 = _as_float(header.get('AMSTART')), _as_float(header.get('AMEND'))
        vals = [a for a in (a0, a1) if a is not None]
        airmass = float(np.mean(vals)) if vals else None
    meta['airmass'] = airmass
    frame = _first(header, ('VEL_TYPE',))[0]
    meta['frame_in'] = None if frame is None else str(frame).strip().lower()
    f = _as_float(header.get('VEL_CORR'))
    meta['vel_corr'] = f
    # PypeIt's factor f = sqrt((1+β)/(1-β)) expressed as a velocity: c (f²-1)/(f²+1)
    meta['vel_corr_kms'] = None if f is None else C_KMS * (f * f - 1.) / (f * f + 1.)
    meta['velocity_keywords'] = {k: header[k] for k in _VEL_KEYS if k in header}
    return meta


def _header_resolution(header: Mapping[str, Any]) -> float | None:
    value = _as_float(_first(header, _RES_KEYS)[0])
    return value if value is not None and value > 1. else None


def _role_lookup(names: list[str], columns: Mapping[str, str | int] | None) -> dict[str, int]:
    """
    Map the roles wave, flux and err/ivar/var to column indices.

    ``columns`` entries (role → name or index) take priority; case-insensitive aliases fill the rest.
    """
    lower = [str(n).strip().lower() for n in names]
    out: dict[str, int] = {}
    for role, spec in (columns or {}).items():
        role = {'sigma': 'err', 'error': 'err', 'wavelength': 'wave'}.get(role.lower(), role.lower())
        if role not in _ROLE_NAMES:
            raise ValueError(f"columns key {role!r} must be one of wave, flux, err, ivar, var, sky")
        if isinstance(spec, int):
            if not 0 <= spec < len(names):
                raise ValueError(f'columns[{role!r}]={spec} is out of range for {len(names)} columns')
            out[role] = spec
        else:
            try:
                out[role] = lower.index(str(spec).strip().lower())
            except ValueError:
                raise ValueError(f'columns[{role!r}]={spec!r} not found among {names}') from None
    for role, aliases in _ROLE_NAMES.items():
        if role in out:
            continue
        for alias in aliases:
            if alias in lower:
                out[role] = lower.index(alias)
                break
    return out


def _sigma_from_roles(get: Callable[[int], np.ndarray], roles: dict[str, int], n: int) -> np.ndarray:
    if 'err' in roles:
        return _clean_sigma(get(roles['err']))
    if 'ivar' in roles:
        return _ivar_to_sigma(get(roles['ivar']))
    if 'var' in roles:
        var = get(roles['var'])
        return _ivar_to_sigma(np.where(var > 0, 1. / np.where(var > 0, var, 1.), 0.))
    return np.full(n, np.nan)


def _is_fits(path: Path) -> bool:
    try:
        opener = gzip.open if path.suffix.lower() == '.gz' else open
        with opener(path, 'rb') as fh:
            return fh.read(6) == b'SIMPLE'
    except OSError:
        return False


def _open_fits(path: Path) -> fits.HDUList:
    return fits.open(path, memmap=False)


def _hdu_index(hdul: fits.HDUList, hdu: int | str) -> int:
    try:
        return hdul.index_of(hdu)
    except (KeyError, IndexError) as exc:
        raise ValueError(f'HDU {hdu!r} not found in file with {len(hdul)} HDUs') from exc


def _table_columns(hdu_obj: Any) -> list[str]:
    cols = getattr(hdu_obj, 'columns', None)
    return [] if cols is None else list(cols.names)


def _table_wave_unit(hdu_obj: Any, name: str) -> str | None:
    try:
        return hdu_obj.columns[name].unit
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------------------------------
# readers 1 and 2: PypeIt
# ----------------------------------------------------------------------------------------------------------------------

def _pypeit_tables(hdul: fits.HDUList) -> list[int]:
    """Indices of the PypeIt object tables (extensions with OPT_WAVE or BOX_WAVE)"""
    out = []
    for idx in range(1, len(hdul)):
        names = _table_columns(hdul[idx])
        if 'OPT_WAVE' in names or 'BOX_WAVE' in names:
            out.append(idx)
    return out


def _norm_name(x: Any) -> str:
    return ''.join(ch for ch in str(x).lower() if ch.isalnum())


def _median_snr(ext: Any) -> float:
    """Median OPT (else BOX) flux × √ivar of a PypeIt object table (NaN if neither exists)"""
    names = [n.upper() for n in _table_columns(ext)]
    for prefix in ('OPT', 'BOX'):
        for fcol, icol in ((f'{prefix}_FLAM', f'{prefix}_FLAM_IVAR'), (f'{prefix}_COUNTS', f'{prefix}_COUNTS_IVAR')):
            if fcol in names and icol in names:
                f = np.asarray(ext.data[fcol], dtype=float)
                iv = np.asarray(ext.data[icol], dtype=float)
                ok = np.isfinite(f) & np.isfinite(iv) & (iv > 0)
                if ok.any():
                    return float(np.median(f[ok] * np.sqrt(iv[ok])))
    return float('nan')


def _choose_pypeit_object(hdul: fits.HDUList, tables: list[int], path: Any = '') -> int:
    """
    The target among several PypeIt object tables

    1. The table whose MASKDEF_OBJNAME, NAME or EXTNAME matches the primary header's OBJECT/TARGNAME/TARGET (a
       unique match only; long-slit spec1d tables are named by trace position, so this rarely applies to them).
    2. Otherwise the highest median OPT S/N among the ``OBJTYPE = 'science'`` tables (all tables if none is
       labelled science), with a warning when it is less than twice the next one's.

    The choice and the alternatives are logged; ``input.hdu`` (``hdu=``) overrides it.
    """
    name = getattr(path, 'name', str(path))
    target, _ = _first(_header_dict(hdul[0].header), ('OBJECT', 'TARGNAME', 'TARGET'))

    def label(i: int) -> str:
        return str(hdul[i].header.get('NAME', hdul[i].name))

    if target is not None and len(_norm_name(target)) > 3:
        t = _norm_name(target)
        hits = []
        for idx in tables:
            h = hdul[idx].header
            labels = [_norm_name(h.get(k)) for k in ('MASKDEF_OBJNAME', 'NAME', 'EXTNAME') if h.get(k)]
            if any(lab == t or t in lab for lab in labels if lab):
                hits.append(idx)
        if len(hits) == 1:
            log.info('%s: PypeIt spec1d with %d objects; reading HDU %d (%s), which matches the target name %r', name,
                     len(tables), hits[0], label(hits[0]), target)
            return hits[0]
    science = [i for i in tables if str(hdul[i].header.get('OBJTYPE', '')).strip().lower() == 'science']
    pool = science or tables
    snr = {i: _median_snr(hdul[i]) for i in tables}
    ranked = sorted(pool, key=lambda i: -snr[i] if np.isfinite(snr[i]) else np.inf)
    best = ranked[0]
    log.info('%s: PypeIt spec1d with %d objects, none named after the target %r; reading HDU %d (%s, median S/N '
             '%.1f), the highest-S/N %sobject (others: %s). Set input.hdu to choose another', name, len(tables),
             target, best, label(best), snr[best], 'science ' if science else '',
             ', '.join(f'HDU {i} {label(i)} S/N {snr[i]:.1f}' for i in tables if i != best))
    if len(ranked) > 1 and np.isfinite(snr[ranked[1]]) and snr[ranked[1]] > 0 and snr[best] < 2. * snr[ranked[1]]:
        log.warning('%s: the chosen object (S/N %.1f) is less than twice as bright as HDU %d (S/N %.1f); check that it '
                    'is the target, or set input.hdu', name, snr[best], ranked[1], snr[ranked[1]])
    return best


def _pypeit_hdu(hdul: fits.HDUList, hdu: int | str | None, path: Any = '') -> int | None:
    if hdu is not None:
        idx = _hdu_index(hdul, hdu)
        names = _table_columns(hdul[idx])
        return idx if ('OPT_WAVE' in names or 'BOX_WAVE' in names) else None
    tables = _pypeit_tables(hdul)
    if len(tables) > 1:
        return _choose_pypeit_object(hdul, tables, path)
    return tables[0] if tables else None


def _sniff_pypeit_spec1d(path: Path, hdu: int | str | None = None, columns: Any = None) -> bool:
    if not _is_fits(path):
        return False
    with _open_fits(path) as hdul:
        if hdu is None:
            return bool(_pypeit_tables(hdul))
        return _pypeit_hdu(hdul, hdu) is not None


def _read_pypeit_spec1d(path: Path, hdu: int | str | None = None, columns: Any = None) -> RawSpectrum:
    with _open_fits(path) as hdul:
        idx = _pypeit_hdu(hdul, hdu, path)
        if idx is None:
            raise ValueError('no PypeIt OPT_/BOX_ extraction columns found')
        ext = hdul[idx]
        header = _header_dict(hdul[0].header, ext.header)
        header.update(RVF_HDU=idx, RVF_EXTNAME=str(ext.header.get('NAME', ext.name)),
                      RVF_NOBJ=len(_pypeit_tables(hdul)))
        data = ext.data
        names = [n.upper() for n in _table_columns(ext)]
        chosen = None
        for prefix in ('OPT', 'BOX'):
            wcol, fcol, icol = f'{prefix}_WAVE', f'{prefix}_FLAM', f'{prefix}_FLAM_IVAR'
            if not all(c in names for c in (wcol, fcol, icol)):
                continue
            wave = np.asarray(data[wcol], dtype=float)
            flux = np.asarray(data[fcol], dtype=float)
            ivar = np.asarray(data[icol], dtype=float)
            mcol = f'{prefix}_MASK'
            if mcol in names:  # PypeIt: True = good pixel; masked pixels can hold wild fluxes with finite ivar
                bad = ~np.asarray(data[mcol], dtype=bool)
                if bad.any():
                    log.debug('%s: dropping %d pixels masked by %s', path.name, int(bad.sum()), mcol)
                    flux = np.where(bad, np.nan, flux)  # dropped with the other non-finite pixels
                    ivar = np.where(bad, 0., ivar)
            if np.any(np.isfinite(wave) & (wave > 0) & np.isfinite(flux)):
                skycol = f'{prefix}_COUNTS_SKY'
                sky = np.asarray(data[skycol], dtype=float) if skycol in names else None
                chosen = (prefix, wave, flux, ivar, sky)
                break
        if chosen is None:
            raise ValueError('PypeIt table has no usable OPT_ or BOX_ wave/flam/flam_ivar columns')
    prefix, wave, flux, ivar, sky = chosen
    log.debug('%s: PypeIt spec1d, HDU %d, %s extraction%s', path.name, idx, prefix, ' with sky' if sky is not None
              else '')
    medium, source = _medium_from_header(header)
    return RawSpectrum(wave, flux, _ivar_to_sigma(ivar), header, medium, source, PYPEIT_DEFAULT_MEDIUM,
                       _header_resolution(header), 'pypeit_spec1d', sky)


def find_wvcalib(path: str | Path) -> Path | None:
    """
    The PypeIt wavelength-calibration file belonging to a spectrum, or None

    Looked for in the spectrum's directory and up to three parents, each also as ``Calibrations/``, as
    ``wvcalib*.fits`` or ``WaveCalib*.fits`` (the first directory with exactly one match wins; with several, the
    lexically first is used and the choice is logged).
    """
    here = Path(path).resolve().parent
    for d in [here, *here.parents[:3]]:
        for sub in (d, d / 'Calibrations'):
            if not sub.is_dir():
                continue
            hits = sorted(p for p in sub.iterdir()
                          if p.is_file() and p.suffix == '.fits'
                          and p.name.lower().startswith(('wvcalib', 'wavecalib')))
            if hits:
                if len(hits) > 1:
                    log.info('%s: %d wavelength-calibration files in %s; using %s', Path(path).name, len(hits), sub,
                             hits[0].name)
                return hits[0]
    return None


def arc_line_range(wvcalib: str | Path, extname: str | None = None) -> tuple[float, float] | None:
    """
    Span [Å, as stored by PypeIt: vacuum] of the arc lines fitted in a PypeIt ``wvcalib.fits``

    Reads ``wave_fit`` of the ``SPAT_ID-*_WAVEFIT`` tables.
    If ``extname`` is a spec1d table name such as ``SPAT0240-SLIT0457-DET02`` and a table with that spatial ID
    exists, only that slit is used; otherwise (e.g. a long slit whose calibration has a different ID) all tables are.
    """
    spat = re.match(r'SPAT0*(\d+)', str(extname or ''))
    lines: list[np.ndarray] = []
    matched: list[np.ndarray] = []
    with fits.open(wvcalib) as hdul:
        for ext in hdul[1:]:
            m = re.fullmatch(r'SPAT_ID-(\d+)_WAVEFIT', str(ext.name))
            if not m or 'wave_fit' not in [c.lower() for c in (ext.columns.names or [])]:
                continue
            col = next(c for c in ext.columns.names if c.lower() == 'wave_fit')
            w = np.asarray(ext.data[col], dtype=float).ravel()
            w = w[np.isfinite(w) & (w > 0)]
            lines.append(w)
            if spat and int(m.group(1)) == int(spat.group(1)):
                matched.append(w)
    pool = np.concatenate(matched or lines) if (matched or lines) else np.array([])
    if len(pool) < 2:
        return None
    return float(pool.min()), float(pool.max())


def _coadd_hdu(hdul: fits.HDUList, hdu: int | str | None) -> int | None:
    candidates = [_hdu_index(hdul, hdu)] if hdu is not None else range(1, len(hdul))
    for idx in candidates:
        names = _table_columns(hdul[idx])
        if all(c in names for c in ('wave', 'flux', 'ivar')):  # PypeIt's lower-case convention
            return idx
    return None


def _sniff_pypeit_coadd(path: Path, hdu: int | str | None = None, columns: Any = None) -> bool:
    if columns is not None or not _is_fits(path):
        return False
    with _open_fits(path) as hdul:
        return _coadd_hdu(hdul, hdu) is not None


def _read_pypeit_coadd(path: Path, hdu: int | str | None = None, columns: Any = None) -> RawSpectrum:
    with _open_fits(path) as hdul:
        idx = _coadd_hdu(hdul, hdu)
        if idx is None:
            raise ValueError('no wave/flux/ivar table found')
        ext = hdul[idx]
        header = _header_dict(hdul[0].header, ext.header)
        data = ext.data
        wave = np.asarray(data['wave'], dtype=float).ravel()
        flux = np.asarray(data['flux'], dtype=float).ravel()
        ivar = np.asarray(data['ivar'], dtype=float).ravel()
        unit = _table_wave_unit(ext, 'wave')
        if 'mask' in _table_columns(ext):
            good = np.asarray(data['mask']).ravel().astype(bool)  # PypeIt: True/1 = good pixel
            wave, flux, ivar = wave[good], flux[good], ivar[good]
    wave = _wave_to_angstrom(wave, unit or 'Angstrom', path.name)  # PypeIt wavelengths are in Å
    medium, source = _medium_from_header(header)
    return RawSpectrum(wave, flux, _ivar_to_sigma(ivar), header, medium, source, PYPEIT_DEFAULT_MEDIUM,
                       _header_resolution(header), 'pypeit_coadd')


# ----------------------------------------------------------------------------------------------------------------------
# reader 3: generic 1D FITS
# ----------------------------------------------------------------------------------------------------------------------

def _generic_kind(hdu_obj: Any, columns: Any) -> str | None:
    """Classify an HDU as 'table', 'wcs' (1D image with CRVAL1) or 'rows' (2×N or 3×N image), else None"""
    names = _table_columns(hdu_obj)
    if names:
        roles = _role_lookup(names, columns) if columns else _role_lookup(names, None)
        return 'table' if 'wave' in roles and 'flux' in roles else None
    data = getattr(hdu_obj, 'data', None)
    if data is None or getattr(data, 'dtype', None) is None or data.dtype.names:
        return None
    shape = tuple(s for s in np.shape(data) if s != 1)
    if len(shape) == 1 and shape[0] > 1 and 'CRVAL1' in hdu_obj.header:
        return 'wcs'
    if len(shape) == 2 and ((shape[0] in (2, 3) and shape[1] > 3) or (shape[1] in (2, 3) and shape[0] > 3)):
        return 'rows'
    return None


def _generic_hdu(hdul: fits.HDUList, hdu: int | str | None, columns: Any) -> tuple[int, str] | None:
    candidates = [_hdu_index(hdul, hdu)] if hdu is not None else range(len(hdul))
    for idx in candidates:
        kind = _generic_kind(hdul[idx], columns)
        if kind:
            return idx, kind
    return None


def _sniff_fits_generic(path: Path, hdu: int | str | None = None, columns: Any = None) -> bool:
    if not _is_fits(path):
        return False
    with _open_fits(path) as hdul:
        return _generic_hdu(hdul, hdu, columns) is not None


def _wcs_wave(header: Mapping[str, Any], n: int) -> np.ndarray:
    """
    Wavelength axis from CRVAL1/CDELT1 (or CD1_1)/CRPIX1.

    Log-linear if ``DC-FLAG`` = 1 (IRAF: CRVAL1 and CDELT1 in log10 λ) or ``CTYPE1`` ends in '-LOG'
    (FITS WCS paper III: λ = λ_r exp[(w − w_r)/λ_r]).
    """
    crval = float(header['CRVAL1'])
    cdelt = header.get('CDELT1', header.get('CD1_1'))
    if cdelt is None:
        raise ValueError('1D image has CRVAL1 but neither CDELT1 nor CD1_1')
    cdelt = float(cdelt)
    crpix = float(header.get('CRPIX1', 1.))
    x = np.arange(n) + 1.
    lin = crval + (x - crpix) * cdelt
    ctype = str(header.get('CTYPE1', '')).upper()
    dc = _as_float(header.get('DC-FLAG'))
    if dc is not None and int(dc) == 1:
        return 10. ** lin
    if ctype.endswith('-LOG'):
        return crval * np.exp((lin - crval) / crval)
    return lin


def _image_unit(header: Mapping[str, Any]) -> str | None:
    for key in ('XUNITS', 'CUNIT1', 'WAVEUNIT', 'WAVE_UNIT', 'XUNIT', 'BUNIT'):
        value = header.get(key)
        if value is not None and _unit_scale(value) is not None:
            return str(value)
    return None


def _read_fits_generic(path: Path, hdu: int | str | None = None, columns: Any = None) -> RawSpectrum:
    with _open_fits(path) as hdul:
        found = _generic_hdu(hdul, hdu, columns)
        if found is None:
            raise ValueError('no table with wave/flux columns, 1D image with CRVAL1 or 2×N/3×N image found')
        idx, kind = found
        ext = hdul[idx]
        header = _header_dict(hdul[0].header, ext.header)
        if kind == 'table':
            names = _table_columns(ext)
            roles = _role_lookup(names, columns)
            data = ext.data

            def get(i: int) -> np.ndarray:
                return np.asarray(data[names[i]], dtype=float).ravel()

            wave, flux = get(roles['wave']), get(roles['flux'])
            sigma = _sigma_from_roles(get, roles, len(wave))
            sky = get(roles['sky']) if 'sky' in roles else None
            unit = _table_wave_unit(ext, names[roles['wave']])
        elif kind == 'wcs':
            sky = None
            flux = np.asarray(np.squeeze(ext.data), dtype=float)
            wave = _wcs_wave(header, len(flux))
            sigma = np.full(len(flux), np.nan)
            for j, other in enumerate(hdul):  # optional error image
                if j == idx or getattr(other, 'data', None) is None or _table_columns(other):
                    continue
                if str(other.name).upper() in ('ERR', 'ERROR', 'SIGMA', 'UNCERT', 'NOISE', 'STDDEV') \
                        and np.size(other.data) == len(flux):
                    sigma = _clean_sigma(np.asarray(other.data, dtype=float).ravel())
                    break
            unit = _image_unit(header)
        else:  # rows
            sky = None
            arr = np.asarray(np.squeeze(ext.data), dtype=float)
            if arr.shape[0] > 3:
                arr = arr.T
            wave, flux = arr[0], arr[1]
            sigma = _clean_sigma(arr[2]) if arr.shape[0] > 2 else np.full(len(wave), np.nan)
            unit = _image_unit(header)
    wave = _wave_to_angstrom(wave, unit, path.name)
    medium, source = _medium_from_header(header)
    return RawSpectrum(wave, flux, sigma, header, medium, source, None, _header_resolution(header), 'fits_generic',
                       sky)


# ----------------------------------------------------------------------------------------------------------------------
# reader 4: ASCII / CSV / TSV
# ----------------------------------------------------------------------------------------------------------------------

_SPLIT = re.compile(r'\s*[,;\t]\s*|\s+')
_KV = re.compile(r'^([A-Za-z_][\w\-. ]{0,40}?)\s*[=:]\s*(.+?)\s*$')
_UNIT_RE = re.compile(r'(?<![A-Za-z])(µm|μm|um|micron[s]?|micrometer[s]?|micrometre[s]?|nm|nanometer[s]?|nanometre[s]?|'
                      r'angstrom[s]?|ångström|å|aa|ang)(?![A-Za-z])', re.IGNORECASE)


def _tokens(line: str) -> list[str]:
    return [t for t in _SPLIT.split(line.strip()) if t != '']


def _to_floats(tokens: list[str]) -> list[float] | None:
    try:
        return [float(t) for t in tokens]
    except ValueError:
        return None


def _parse_ascii(path: Path) -> tuple[np.ndarray, list[str], list[str], dict[str, Any], list[str] | None]:
    """Return (data[n, ncol], comment lines, header-row text, pseudo-header from 'KEY = value' comments, names)"""
    with open(path, 'rb') as fh:
        head = fh.read(4096)
    if b'\x00' in head:
        raise ValueError('binary file')
    rows: list[list[float]] = []
    comments: list[str] = []
    header_rows: list[str] = []
    pseudo: dict[str, Any] = {}
    names: list[str] | None = None
    ncol = None
    with open(path, encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line[0] in '#%!' or line.startswith('//'):
                text = line.lstrip('#%!/ \t')
                comments.append(text)
                m = _KV.match(text)
                if m:
                    value: Any = m.group(2).strip().strip('\'"')
                    num = _as_float(value)
                    pseudo[m.group(1).strip().upper().replace(' ', '_')] = value if num is None else num
                elif not rows and text:
                    toks = _tokens(text)
                    if len(toks) >= 2:
                        names = toks  # candidate column names (checked against ncol later)
                continue
            line = line.split('#', 1)[0]
            vals = _to_floats(_tokens(line))
            if vals is None:
                if not rows:
                    header_rows.append(line)
                    names = _tokens(line)
                continue
            if ncol is None:
                ncol = len(vals)
            if len(vals) == ncol:
                rows.append(vals)
    if ncol is None or ncol < 2 or len(rows) < 2:
        raise ValueError('fewer than two numeric columns or rows')
    if names is not None and len(names) != ncol:
        names = None
    return np.asarray(rows, dtype=float), comments, header_rows, pseudo, names


def _sniff_ascii(path: Path, hdu: int | str | None = None, columns: Any = None) -> bool:
    if _is_fits(path):
        return False
    try:
        _parse_ascii(path)
    except (ValueError, OSError):
        return False
    return True


def _read_ascii(path: Path, hdu: int | str | None = None, columns: Any = None) -> RawSpectrum:
    data, comments, header_rows, pseudo, names = _parse_ascii(path)
    ncol = data.shape[1]
    text = '\n'.join(comments + header_rows)
    roles: dict[str, int] = {}
    if names is not None:  # strip units from the names: 'wave[nm]', 'flux (Jy)', 'wave_um'
        names = [re.sub(r'(_(um|nm|a|aa|ang|angstrom|micron|microns))$', '', re.sub(r'[\[(].*?[\])]', '', n).strip(),
                        flags=re.IGNORECASE) for n in names]
    if names is not None or columns:
        roles = _role_lookup(names or [f'col{i}' for i in range(ncol)], columns)
    if 'wave' not in roles or 'flux' not in roles:
        roles = {'wave': 0, 'flux': 1, **({'err': 2} if ncol > 2 else {})}
        if ncol > 3:
            log.warning('%s: %d columns and no usable names; using the first three as wave, flux, err',
                        path.name, ncol)
    elif names is not None:
        log.debug('%s: columns recognised from header: %s', path.name, names)
    wave, flux = data[:, roles['wave']], data[:, roles['flux']]
    sigma = _sigma_from_roles(lambda i: data[:, i], roles, len(wave))
    sky = data[:, roles['sky']] if 'sky' in roles else None
    m = _UNIT_RE.search(text)
    unit = m.group(1) if m else None
    header = dict(pseudo)
    medium, source = _medium_from_header(header)
    wave = _wave_to_angstrom(wave, unit, path.name)
    return RawSpectrum(wave, flux, sigma, header, medium, source, None, _header_resolution(header), 'ascii', sky)


# ----------------------------------------------------------------------------------------------------------------------
# fallback: specutils
# ----------------------------------------------------------------------------------------------------------------------

def _read_specutils(path: Path, hdu: int | str | None = None, columns: Any = None) -> RawSpectrum:
    from astropy import units as u
    try:
        from specutils import Spectrum as SpectrumCls
    except ImportError:  # specutils < 2
        from specutils import Spectrum1D as SpectrumCls
    kwargs = {} if hdu is None else {'hdu': hdu}
    spec = SpectrumCls.read(str(path), **kwargs)
    wave = np.asarray(spec.spectral_axis.to(u.AA, equivalencies=u.spectral()).value, dtype=float)
    flux = np.asarray(spec.flux.value, dtype=float).ravel()
    sigma = np.full(len(flux), np.nan)
    unc = getattr(spec, 'uncertainty', None)
    if unc is not None:
        arr = np.asarray(unc.array, dtype=float).ravel()
        kind = type(unc).__name__.lower()
        sigma = _ivar_to_sigma(arr) if 'inverse' in kind else _clean_sigma(np.sqrt(arr) if 'variance' in kind else arr)
    meta = getattr(spec, 'meta', {}) or {}
    header = {}
    hdr = meta.get('header')
    if hdr is not None:
        header = _header_dict(hdr) if isinstance(hdr, fits.Header) else {str(k).upper(): v for k, v in hdr.items()}
    medium, source = _medium_from_header(header)
    return RawSpectrum(wave, flux, sigma, header, medium, source, None, _header_resolution(header), 'specutils')


# ----------------------------------------------------------------------------------------------------------------------
# public entry point
# ----------------------------------------------------------------------------------------------------------------------

def _clean_arrays(wave: np.ndarray, flux: np.ndarray, sigma: np.ndarray, sky: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    wave = np.asarray(wave, dtype=float).ravel()
    flux = np.asarray(flux, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    sky = None if sky is None else np.asarray(sky, dtype=float).ravel()
    if not len(wave) == len(flux) == len(sigma) or (sky is not None and len(sky) != len(wave)):
        raise ValueError('wave, flux, sigma (and sky) have different lengths')
    good = np.isfinite(wave) & (wave > 0) & np.isfinite(flux)
    order = np.argsort(wave[good], kind='stable')
    wave, flux, sigma = wave[good][order], flux[good][order], sigma[good][order]
    sky = None if sky is None else sky[good][order]
    keep = np.concatenate([[True], np.diff(wave) > 0])  # drop duplicate wavelengths, keeping the first
    return wave[keep], flux[keep], sigma[keep], None if sky is None else sky[keep]


def _attach_arc_range(meta: dict[str, Any], path: Path, extname: str | None, medium_in: str,
                      medium_method: str) -> None:
    """Set ``meta['arc_range']`` (vacuum Å) and ``meta['arc_file']`` from a neighbouring PypeIt wavecal file"""
    try:
        calib = find_wvcalib(path)
        span = None if calib is None else arc_line_range(calib, extname)
    except Exception as exc:  # an unreadable calibration file is never fatal
        log.debug('%s: could not read the arc-line range: %s', path.name, exc)
        return
    if calib is None or span is None:
        return
    # PypeIt stores vacuum wavelengths; they are in the file's medium, so convert them as the spectrum was
    lo, hi = (float(x) for x in to_vacuum(np.array(span), medium_in, medium_method))
    meta.update(arc_range=(lo, hi), arc_file=str(calib))
    log.debug('%s: arc lines span %.1f-%.1f Å (vacuum) in %s', path.name, lo, hi, calib)


def read_spectrum(path: str | Path, *, medium: str = 'auto', medium_method: str = 'ciddor',
                  hdu: int | str | None = None,
                  columns: Mapping[str, str | int] | None = None, resolution: float | None = None) -> Spectrum:
    """
    Read a one-dimensional spectrum and return it in vacuum Å.

    Parameters
    ----------
    path
        FITS (PypeIt spec1d or coadd, generic 1D image, 2×N/3×N image, binary table), or ASCII/CSV/TSV file;
        anything else is passed to ``specutils``
    medium
        'auto' (read the header; raise `ValueError` if it cannot be determined; PypeIt files default to vacuum),
        'air' or 'vacuum'.
        An explicit value overrides the header (a conflict is logged as a warning).
    medium_method
        Air to vacuum formula: 'ciddor' (default), 'edlen' or 'morton'
    hdu
        HDU index or name to read (default: the first suitable one; for a PypeIt spec1d holding several
        objects, the table matching the target name, else the highest-S/N science object, which is logged)
    columns
        Column overrides, role → column name or (ASCII, FITS) index, with roles 'wave', 'flux' and one of
        'err', 'ivar', 'var'
    resolution
        Resolving power R = λ/Δλ to attach; overrides any value found in the header

    Returns
    -------
    spectrum
        Cleaned `Spectrum` with ``meta`` filled (see the module docstring)

    Raises
    ------
    ValueError
        If the file cannot be read or the medium cannot be determined with ``medium='auto'``
    """
    path = Path(path)
    medium = str(medium).strip().lower()
    medium = {'vac': 'vacuum'}.get(medium, medium)
    if medium not in ('auto', 'air', 'vacuum'):
        raise ValueError(f"medium must be 'auto', 'air' or 'vacuum', not {medium!r}")
    if not path.is_file():
        raise FileNotFoundError(f'Spectrum file not found: {path}')
    errors: list[str] = []
    raw: RawSpectrum | None = None
    for reader in _READERS:
        try:
            if not reader.sniff(path, hdu, columns):
                continue
        except Exception as exc:
            log.debug('%s: sniff by %s failed: %s', path.name, reader.name, exc)
            continue
        try:
            raw = reader.read(path, hdu, columns)
            break
        except Exception as exc:
            log.debug('%s: reader %s failed: %s', path.name, reader.name, exc)
            errors.append(f'{reader.name}: {exc}')
    if raw is None:
        try:
            raw = _read_specutils(path, hdu, columns)
        except Exception as exc:
            errors.append(f'specutils: {exc}')
            why = '; '.join(errors) or 'no reader recognised it'
            raise ValueError(f'Could not read {path} as a spectrum ({why})') from exc
    log.debug('%s: read with %s', path.name, raw.reader)
    try:
        wave, flux, sigma, sky = _clean_arrays(raw.wave, raw.flux, raw.sigma, raw.sky)
    except ValueError as exc:
        raise ValueError(f'{path}: {exc}') from exc
    if len(wave) < 2:
        raise ValueError(f'{path}: fewer than two valid pixels after removing non-finite values')

    # medium: explicit > header > documented reader default > error
    if medium != 'auto':
        medium_in, source = medium, 'user'
        if raw.medium and raw.medium != medium:
            log.warning('%s: medium=%r given but the header (%s) says %r; using %r',
                        path.name, medium, raw.medium_source, raw.medium, medium)
    elif raw.medium:
        medium_in, source = raw.medium, raw.medium_source
    elif raw.default_medium:
        medium_in, source = raw.default_medium, f'default:{raw.reader}'
        log.info('%s: no medium keyword in the header; assuming %s, the documented default for %s files',
                 path.name, medium_in, raw.reader)
    else:
        raise ValueError(f"Cannot determine whether the wavelengths in {path} are in air or vacuum "
                         "(no AIRORVAC, VACUUM, WAVE_MED or CTYPE1 'AWAV'/'WAVE' in the header). "
                         "Pass medium='air' or medium='vacuum' (CLI: --medium).")
    wave = to_vacuum(wave, medium_in, medium_method)
    wave, flux, sigma, sky = _clean_arrays(wave, flux, sigma, sky)

    meta = _meta_from_header(raw.header)
    meta.update(medium_in=medium_in, medium_source=source, reader=raw.reader)
    for key, out in (('RVF_HDU', 'hdu'), ('RVF_EXTNAME', 'extname'), ('RVF_NOBJ', 'n_objects')):
        if key in raw.header:  # which table of a multi-object file was read
            meta[out] = raw.header[key]
    r = resolution if resolution is not None else raw.resolution
    if r is not None and not (np.isfinite(r) and r > 0):
        raise ValueError(f'resolution must be positive, not {r}')
    if raw.reader in ('pypeit_spec1d', 'pypeit_coadd'):
        _attach_arc_range(meta, path, raw.header.get('RVF_EXTNAME'), medium_in, medium_method)
    log.debug('%s: %d pixels, %.1f-%.1f Å (vacuum), medium_in=%s (%s)', path.name, len(wave), wave[0], wave[-1],
              medium_in, source)
    return Spectrum(wave, flux, sigma, None if r is None else float(r), meta, str(path), sky)


register_reader('pypeit_spec1d', _sniff_pypeit_spec1d, _read_pypeit_spec1d)
register_reader('pypeit_coadd', _sniff_pypeit_coadd, _read_pypeit_coadd)
register_reader('fits_generic', _sniff_fits_generic, _read_fits_generic)
register_reader('ascii', _sniff_ascii, _read_ascii)
