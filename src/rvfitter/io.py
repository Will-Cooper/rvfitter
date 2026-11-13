"""
Reading spectra (FITS, text, PypeIt) into `Spectrum` objects.

`read_spectrum` is the single entry point: it tries PypeIt spec1d, PypeIt coadd/tellcorr tables, Spextool images
(SpeX, ARCoIRIS), generic 1D FITS and ASCII/CSV/TSV readers in turn, and finally ``specutils``.

Conventions applied whatever the reader:

* σ = 1/√ivar, NaN where ivar ≤ 0; PypeIt pixels masked in ``OPT_MASK`` (``BOX_MASK`` for box extraction) are dropped;
  sky pixels that are non-finite or above 10³ × the median |sky| are set to NaN.
* Non-finite or non-positive wavelengths and non-finite fluxes are dropped, the rest is sorted and duplicate
  wavelengths removed (first kept).
* The wavelength medium is never guessed silently: ``medium='auto'`` reads it from the header (``AIRORVAC``,
  ``VACUUM``, ``WAVE_MED``, ``WAVETYPE`` or ``CTYPE1``); PypeIt products are documented to be vacuum
  (``meta['medium_source']``); anything else raises `ValueError`. Wavelengths are then converted to vacuum with
  `rvfitter.medium`.

The header keywords behind each `Spectrum.meta` entry are listed in `_meta_from_header` and `_slit_meta`.
``meta['velocity_keywords']`` holds any velocity-correction keywords found (``VHELIO``, ``BCV``, ``BERV``,
``VELOSYS``…), so that `rvfitter.barycentric` can tell whether a spectrum is already corrected.
"""
import glob
import gzip
import logging
import re
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.io.registry import IORegistryError
from astropy.utils.exceptions import AstropyUserWarning

from .barycentric import vel_corr_to_kms
from .medium import to_vacuum
from .types import Spectrum

__all__ = ['read_spectrum', 'RawSpectrum', 'raw_headers']

log = logging.getLogger(__name__)

# PypeIt wavelengths are vacuum by default (documented PypeIt behaviour); used only when the header is silent
PYPEIT_DEFAULT_MEDIUM = 'vacuum'
# a coadd without its spec1d files: PypeIt corrects each spec1d's wavelengths to WavelengthSolutionPar.refframe,
# 'heliocentric' by default, and coadd1d keeps them
PYPEIT_FRAME_NOTE = ("PypeIt's default (refframe 'heliocentric') makes its wavelengths probably heliocentric; if so, "
                     "set velocity_frame.correction = 'skip' and frame = 'heliocentric'")

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
            value = _clean_value(card.value)
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
    except ValueError:
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

    for key in ('AIRORVAC', 'MEDIUM', 'WAVE_MED', 'WAVEMED', 'WAVE_MEDIUM', 'WAVEMEDIUM', 'WAVMED', 'WAVETYPE'):
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
    if number is None:
        try:
            from astropy import units as u
            from astropy.coordinates import Angle
            number = float(Angle(str(value).strip(), unit=u.hourangle if is_ra else u.deg).deg)
        except ValueError:
            log.warning('Ignoring unparseable %s %r', 'RA' if is_ra else 'Dec', value)
            return None
    if not 0. <= number <= 360. if is_ra else not -90. <= number <= 90.:
        log.warning('Ignoring out-of-range %s %r', 'RA' if is_ra else 'Dec', value)
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
    except ValueError:
        log.warning('Ignoring unparseable DATE-OBS %r', date)
        return None


def _meta_from_header(header: Mapping[str, Any]) -> dict[str, Any]:
    """
    Build the `Spectrum.meta` dictionary from a flat header.

    - object: ``OBJECT``, ``TARGNAME``, ``TARGET``, ``OBJNAME`` (first non-empty).
    - ra, dec: ``RA``/``DEC`` (or ``OBJRA``/``OBJDEC``, ``TELRA``/``TELDEC``, ``TCS_RA``/``TCS_DEC``); floats are
      degrees, sexagesimal strings hours (RA) and degrees (Dec).
    - mjd_mid: ``MJD_MID``/``MJD-MID`` (already mid-exposure), else ``MJD``, ``MJD-OBS`` or ``DATE-OBS`` (+
      ``TIME-OBS``), which mean the exposure start, plus ``EXPTIME``/2; ``mjd_source`` records which (a reader
      that sets ``MJD_MID`` names its origin in ``RVF_MJD_SOURCE``).
    - site: ``OBSERVAT``, else ``TELESCOP`` (``telescope`` is always ``TELESCOP``); site_lat/site_lon: ``SITELAT``...
    - exptime: ``EXPTIME``, ``EXPOSURE`` or ``ITIME`` [s]; airmass: ``AIRMASS``, else the mean of ``AMSTART``/``AMEND``.
    - frame_in: ``VEL_TYPE``; vel_corr: ``VEL_CORR`` (PypeIt's factor already applied to the wavelengths) and
      ``vel_corr_kms``, the same as a velocity.
    """
    meta: dict[str, Any] = {}
    obj, _ = _first(header, ('OBJECT', 'TARGNAME', 'TARGET', 'OBJNAME'))
    if obj is not None and str(obj).strip().lower() not in ('none', 'unknown', 'n/a', ''):
        meta['object'] = str(obj)
    ra, _ = _first(header, ('RA', 'OBJRA', 'TELRA', 'RA_DEG', 'TCS_RA'))
    dec, _ = _first(header, ('DEC', 'OBJDEC', 'TELDEC', 'DEC_DEG', 'TCS_DEC'))
    meta['ra'] = _parse_coord(ra, True)
    meta['dec'] = _parse_coord(dec, False)
    exptime = _as_float(_first(header, ('EXPTIME', 'EXPOSURE', 'ITIME'))[0])
    meta['exptime'] = exptime
    # mid-exposure time
    mjd_mid, source = None, None
    mid, key = _first(header, ('MJD_MID', 'MJD-MID', 'MJDMID'))
    if _as_float(mid) is not None:
        mjd_mid, source = _as_float(mid), header.get('RVF_MJD_SOURCE', key)
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
    meta['vel_corr_kms'] = None if f is None else vel_corr_to_kms(f)
    meta['velocity_keywords'] = {k: header[k] for k in _VEL_KEYS if k in header}
    meta['frame_note'] = header.get('RVF_FRAME_NOTE')  # why the input frame is unknown, when a reader knows
    raw = header.get('RVF_RAW_FILES') or header.get('FILENAME')  # PypeIt: the raw frame(s) behind the spectrum
    meta['raw_files'] = [s for s in str(raw).split(',') if s] if raw else []
    meta.update(_slit_meta(header))
    return meta


_SLIT_WIDTH_RE = re.compile(r'(\d+(?:\.\d+)?)\s*(?:arcsec|asec|")', re.IGNORECASE)


def _slit_meta(header: Mapping[str, Any]) -> dict[str, Any]:
    """
    Slit width, binning, pixel scale, instrument and trace FWHM, for the slit-centring prior

    - ``slit_width_arcsec``: ``SLITW`` (a number, arcsec; OSIRIS raw frames) or ``SLTW_ARC`` (Spextool), else the
      first of ``MASKNAME``, ``DECKER`` (PypeIt), ``SLIT``, ``SLITNAME`` and ``APERTURE`` whose text holds a number
      followed by 'arcsec' (e.g. 'LongSlits1.23Arcsec'); ``slit_width_key`` records which.
    - ``binning_spec``, ``binning_spat``: PypeIt's ``BINNING`` 'spectral,spatial', else ``CCDSUM`` when both
      factors are equal (its axis order is instrument-dependent).
    - ``pixscale_arcsec``: ``PIXSCALE``, ``PLATESCL`` or ``SECPIX`` [arcsec per (binned) pixel] if present.
    - ``instrument``: ``INSTRUME``, else ``INSTR`` (Spextool).
    - ``trace_fwhm_pix``: PypeIt's object FWHM along the slit [binned pixels].
    """
    out: dict[str, Any] = {}
    inst, _ = _first(header, ('INSTRUME', 'INSTRUMENT', 'INSTR'))
    out['instrument'] = None if inst is None else str(inst).strip()
    pyp = header.get('PYP_SPEC')
    out['pypeit_spec'] = None if pyp is None else str(pyp).strip()
    width, key = None, None
    w, wkey = _first(header, ('SLITW', 'SLTW_ARC'))
    w = _as_float(w)
    if w is not None and w > 0:
        width, key = w, wkey
    else:
        for k in ('MASKNAME', 'DECKER', 'SLIT', 'SLITNAME', 'APERTURE'):
            v = header.get(k)
            m = _SLIT_WIDTH_RE.search(str(v)) if v is not None else None
            if m and float(m.group(1)) > 0:
                width, key = float(m.group(1)), k
                break
    out['slit_width_arcsec'], out['slit_width_key'] = width, key
    spec_bin = spat_bin = None
    b = header.get('BINNING')
    if b is not None:
        parts = re.split(r'[,x\s]+', str(b).strip())
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            spec_bin, spat_bin = int(parts[0]), int(parts[1])
    if spec_bin is None and header.get('CCDSUM') is not None:
        parts = str(header['CCDSUM']).split()
        if len(parts) == 2 and all(p.isdigit() for p in parts) and parts[0] == parts[1]:
            spec_bin = spat_bin = int(parts[0])
    out['binning_spec'], out['binning_spat'] = spec_bin, spat_bin
    out['pixscale_arcsec'] = _as_float(_first(header, ('PIXSCALE', 'PLATESCL', 'SECPIX'))[0])
    out['trace_fwhm_pix'] = _as_float(header.get('RVF_TRACE_FWHM'))
    return out


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
    except KeyError:
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


def _pypeit_trace_fwhm(data: Any, names: list[str], ext_header: Any) -> float | None:
    """The object's spatial FWHM [binned pixels]: the median of PypeIt's ``FWHMFIT`` column, else the ``FWHM`` key"""
    if 'FWHMFIT' in names:
        x = np.asarray(data['FWHMFIT'], dtype=float)
        x = x[np.isfinite(x) & (x > 0)]
        if len(x):
            return float(np.median(x))
    v = _as_float(ext_header.get('FWHM'))
    return v if v is not None and v > 0 else None


def _sniff_pypeit_spec1d(path: Path, hdu: int | str | None = None, columns: Any = None) -> bool:
    if not _is_fits(path):
        return False
    with fits.open(path, memmap=False) as hdul:
        if hdu is None:
            return bool(_pypeit_tables(hdul))
        return _pypeit_hdu(hdul, hdu) is not None


def _read_pypeit_spec1d(path: Path, hdu: int | str | None = None, columns: Any = None) -> RawSpectrum:
    with fits.open(path, memmap=False) as hdul:
        idx = _pypeit_hdu(hdul, hdu, path)
        if idx is None:
            raise ValueError('no PypeIt OPT_/BOX_ extraction columns found')
        ext = hdul[idx]
        header = _header_dict(hdul[0].header, ext.header)
        header.update(RVF_HDU=idx, RVF_EXTNAME=str(ext.header.get('NAME', ext.name)),
                      RVF_NOBJ=len(_pypeit_tables(hdul)))
        data = ext.data
        names = [n.upper() for n in _table_columns(ext)]
        fwhm = _pypeit_trace_fwhm(data, names, ext.header)
        if fwhm is not None:
            header['RVF_TRACE_FWHM'] = fwhm
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


def _arc_lines(wvcalib: str | Path, extname: str | None) -> tuple[np.ndarray, np.ndarray]:
    """
    The fitted arc-line wavelengths [Å] of a PypeIt ``wvcalib.fits`` (``wave_fit`` of the ``SPAT_ID-*_WAVEFIT``
    tables), and the solution's residuals at them [Å] (``wave_soln`` at ``pixel_fit`` − ``wave_fit``; NaN where a
    table lacks those columns or the ``SPAT_ID-*_PYPEITFIT`` gpm rejected the line)

    If ``extname`` is a spec1d table name such as ``SPAT0240-SLIT0457-DET02`` and a table with that spatial ID
    exists, only that slit is used; otherwise (e.g. a long slit whose calibration has a different ID) all tables are.
    """
    spat = re.match(r'SPAT0*(\d+)', str(extname or ''))
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    matched: list[tuple[np.ndarray, np.ndarray]] = []
    with fits.open(wvcalib) as hdul:
        for ext in hdul[1:]:
            m = re.fullmatch(r'SPAT_ID-(\d+)_WAVEFIT', str(ext.name))
            cols = {c.lower(): c for c in (ext.columns.names or [])} if m else {}
            if 'wave_fit' not in cols:
                continue
            w = np.asarray(ext.data[cols['wave_fit']], dtype=float).ravel()
            res = np.full(len(w), np.nan)
            if 'pixel_fit' in cols and 'wave_soln' in cols:
                soln = np.asarray(ext.data[cols['wave_soln']], dtype=float).ravel()
                pix = np.asarray(ext.data[cols['pixel_fit']], dtype=float).ravel()
                res = np.interp(pix, np.arange(len(soln)), soln) - w
                fit = f'SPAT_ID-{m.group(1)}_PYPEITFIT'
                if fit in hdul and 'gpm' in [c.lower() for c in hdul[fit].columns.names]:
                    gpm = np.asarray(hdul[fit].data[0]['gpm']).astype(bool).ravel()
                    res[~gpm[:len(res)]] = np.nan
            ok = np.isfinite(w) & (w > 0)
            lines.append((w[ok], res[ok]))
            if spat and int(m.group(1)) == int(spat.group(1)):
                matched.append((w[ok], res[ok]))
    use = matched or lines
    if not use:
        return np.array([]), np.array([])
    return np.concatenate([u[0] for u in use]), np.concatenate([u[1] for u in use])


def arc_line_range(wvcalib: str | Path, extname: str | None = None) -> tuple[float, float] | None:
    """Span [Å, as stored by PypeIt: vacuum] of the arc lines fitted in a PypeIt ``wvcalib.fits`` (`_arc_lines`)"""
    pool = _arc_lines(wvcalib, extname)[0]
    if len(pool) < 2:
        return None
    return float(pool.min()), float(pool.max())


def arc_rms(wvcalib: str | Path, extname: str | None = None) -> float | None:
    """RMS [Å] of the wavelength solution's residuals at its fitted arc lines (`_arc_lines`); None if unknown"""
    res = _arc_lines(wvcalib, extname)[1]
    res = res[np.isfinite(res)]
    return float(np.sqrt(np.mean(res * res))) if len(res) >= 2 else None


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
    with fits.open(path, memmap=False) as hdul:
        return _coadd_hdu(hdul, hdu) is not None


_COADD_FROM_RE = re.compile(r'From "(spec1d_[^"]+?\.fits)"')


def _coadd_provenance(path: Path, primary: fits.Header) -> dict[str, Any]:
    """
    The frame cards (VEL_CORR, VEL_TYPE) and raw FILENAMEs a PypeIt coadd or its ``*_tellcorr.fits`` inherit from
    its spec1d files

    coadd1d works on the spec1d wavelengths, which already carry each exposure's ``VEL_CORR``, but writes no
    VEL_CORR itself; without it the zero-point curves would be evaluated in the wrong frame. The spec1d names come
    from the coadd's HISTORY ('From "spec1d_…"'); a tellcorr file (often with an empty header) takes them from the
    coadd of the same name next to it. When the spec1d files are not found the frame is unknown: a WARNING, and
    ``RVF_FRAME_NOTE`` says so (`rvfitter.barycentric` repeats it instead of calling the frame 'observed').
    """
    def sources(head: fits.Header) -> list[str]:
        return _COADD_FROM_RE.findall(''.join(str(v) for k, v in head.items() if k == 'HISTORY'))

    out: dict[str, Any] = {}
    names = sources(primary)
    coadd = path.with_name(path.stem.removesuffix('_tellcorr') + path.suffix)
    if not names and coadd != path and coadd.is_file():
        primary = fits.getheader(coadd, 0)
        names = sources(primary)
    found = [path.parent / n for n in names if (path.parent / n).is_file()]
    if not names or len(found) < len(names):
        log.warning('%s: the spec1d file(s) it was coadded from (%s) are not next to it: the velocity frame '
                    '(VEL_CORR) is unknown. %s', path.name, ', '.join(names) or 'none named in its HISTORY or a coadd',
                    PYPEIT_FRAME_NOTE)
        return {'RVF_FRAME_NOTE': PYPEIT_FRAME_NOTE}
    heads = []
    for p in found:  # PypeIt writes VEL_CORR in the object tables, the raw FILENAME in the primary header
        with fits.open(p) as hl:
            ext = next((h.header for h in hl[1:] if 'VEL_CORR' in h.header), fits.Header())
            heads.append((hl[0].header, ext))
    corr = [_as_float(e.get('VEL_CORR')) for _, e in heads]
    out['RVF_RAW_FILES'] = ','.join(str(h.get('FILENAME', '')) for h, _ in heads)
    if any(c is None for c in corr):
        return out  # the spec1d files were not velocity-corrected either
    out['VEL_CORR'] = float(np.mean(corr))
    if heads[0][1].get('VEL_TYPE') is not None:
        out['VEL_TYPE'] = heads[0][1]['VEL_TYPE']
    log.info('%s: VEL_CORR %.9f (%s) from %s%s', path.name, out['VEL_CORR'], out.get('VEL_TYPE', 'frame unnamed'),
             ', '.join(names), f' (mean of {len(corr)}; spread {np.ptp(corr):.2g})' if len(corr) > 1 else '')
    return out


def raw_headers(meta: Mapping[str, Any], pattern: str | None = None, directory: str | None = None
                ) -> list[dict[str, Any]]:
    """
    The primary headers of the raw frames behind a reduced spectrum (reduction pipelines drop the slit PA)

    `pattern` (a path or glob, ``input.raw_header``) names them directly: no match raises. Otherwise each raw
    frame in meta['raw_files'] (PypeIt's FILENAME, e.g. '0000809229-20150803-OSIRIS-….fits') is matched by its
    leading frame number (0000809229) against ``<frame>*.fits`` anywhere under `directory`
    (``input.raw_header_dir``): no match logs a WARNING and returns [], several raise. A spectrum that names no
    raw frame (e.g. a coadd without its spec1d files) takes every frame under `directory` whose ``OBJECT`` is
    meta['object'] (case and spaces ignored) and that has a slit width (spectra, not acquisition images); this may
    include other epochs of the object. Each dict holds the header cards and 'RVF_FILE'.
    """
    if pattern:
        files = sorted(Path(p) for p in glob.glob(str(Path(pattern).expanduser())))
        if not files:
            raise FileNotFoundError(f'input.raw_header {pattern!r} matches no file')
    elif directory and not meta.get('raw_files') and meta.get('object'):
        key = re.sub(r'\s+', '', str(meta['object'])).lower()
        files = []
        for f in sorted(Path(directory).expanduser().rglob('*.fits')):
            head = _header_dict(fits.getheader(f, 0))
            if re.sub(r'\s+', '', str(head.get('OBJECT', ''))).lower() == key and _slit_meta(head)['slit_width_arcsec']:
                files.append(f)
        if not files:
            log.warning('%s: no raw spectrum header with OBJECT %r under input.raw_header_dir %s: the slit PA is '
                        'unknown', meta['object'], meta['object'], directory)
            return []
    elif directory:
        files = []
        for name in meta.get('raw_files') or []:
            m = re.match(r'\d+', Path(name).name)
            hits = sorted(Path(directory).expanduser().rglob(f'{m.group(0)}*.fits')) if m else []
            if len(hits) > 1:
                raise ValueError(f'raw frame {name}: {len(hits)} raw headers under {directory} ({hits[0].name}, ...)')
            files += hits
        if not files:
            log.warning('%s: no raw header for raw frame(s) %s under input.raw_header_dir %s: the slit PA is unknown',
                        meta.get('object', ''), meta.get('raw_files') or 'unknown', directory)
            return []
    else:
        return []
    out = [{**_header_dict(fits.getheader(f, 0)), 'RVF_FILE': str(f)} for f in files]
    log.info('%s: raw header(s) %s', meta.get('object', ''), ', '.join(f.name for f in files))
    return out


def _read_pypeit_coadd(path: Path, hdu: int | str | None = None, columns: Any = None) -> RawSpectrum:
    with fits.open(path, memmap=False) as hdul:
        idx = _coadd_hdu(hdul, hdu)
        if idx is None:
            raise ValueError('no wave/flux/ivar table found')
        ext = hdul[idx]
        header = _header_dict(hdul[0].header, ext.header)
        if 'VEL_CORR' not in header:
            header = {**_coadd_provenance(path, hdul[0].header), **header}
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
# reader 3: Spextool (SpeX, uSpeX, ARCoIRIS)
# ----------------------------------------------------------------------------------------------------------------------

#: Spextool ``INSTR`` -> observatory, for files without TELESCOP/OBSERVAT (ARCoIRIS is on the CTIO Blanco 4-m)
_SPEXTOOL_SITES = {'spex': 'irtf', 'uspex': 'irtf', 'arcoiris': 'ctio'}


def _primary_image(path: Path) -> tuple[dict[str, Any], np.ndarray | None]:
    """The primary header (flat dict) and data"""
    with fits.open(path, memmap=False) as hdul:
        return _header_dict(hdul[0].header), None if hdul[0].data is None else np.asarray(hdul[0].data, dtype=float)


def _sniff_spextool(path: Path, hdu: int | str | None = None, columns: Any = None) -> bool:
    if hdu not in (None, 0) or columns is not None or not _is_fits(path):
        return False
    head, data = _primary_image(path)
    shape = np.shape(data) if data is not None else ()
    return 'NORDERS' in head and 'XUNITS' in head and len(shape) in (2, 3) and shape[-2] in (3, 4)


def _read_spextool(path: Path, hdu: int | str | None = None, columns: Any = None) -> RawSpectrum:
    """
    A Spextool image: rows wavelength, flux, error (and a quality flag, not used), one block per order (orders × rows
    × N, concatenated; where orders overlap, the bluer order's pixels are kept, so differently calibrated orders never
    interleave)

    Spextool writes the time averaged over the combined frames (``AVE_MJD``; ``MJD_OBS`` from the 'Average'
    ``TIME_OBS`` of SpeX; ARCoIRIS ``DATE`` + 'Average' ``UT``), so it is taken as the mid-exposure time as it stands.
    ``RP`` is the resolving power. The site comes from ``INSTR`` when TELESCOP/OBSERVAT are absent (ARCoIRIS).
    """
    header, arr = _primary_image(path)
    blocks, top = [], -np.inf
    for b in sorted(arr.reshape(-1, *arr.shape[-2:]), key=lambda b: np.nanmin(b[0])):
        blocks.append(b[:3, b[0] > top])
        top = max(top, np.nanmax(b[0]))
    wave, flux, sigma = np.concatenate(blocks, axis=1)
    mid, key = _as_float(header.get('AVE_MJD')), 'AVE_MJD'
    if mid is None:
        mid, key = _as_float(header.get('MJD_OBS')), 'MJD_OBS (average)'
    if mid is None and header.get('DATE') and header.get('UT'):
        from astropy.time import Time
        mid, key = float(Time(f"{header['DATE']}T{header['UT']}", format='isot', scale='utc').mjd), 'DATE+UT (average)'
    if mid is not None:
        header.update(MJD_MID=mid, RVF_MJD_SOURCE=key)
    inst = str(header.get('INSTR', '')).strip().lower()
    if not any(k in header for k in ('OBSERVAT', 'TELESCOP')) and inst in _SPEXTOOL_SITES:
        header['OBSERVAT'] = _SPEXTOOL_SITES[inst]
    rp = _as_float(header.get('RP'))
    medium, source = _medium_from_header(header)
    return RawSpectrum(_wave_to_angstrom(wave, _image_unit(header), path.name), flux, _clean_sigma(sigma), header,
                       medium, source, None, rp if rp is not None and rp > 1. else None, 'spextool')


# ----------------------------------------------------------------------------------------------------------------------
# reader 4: generic 1D FITS
# ----------------------------------------------------------------------------------------------------------------------

def _generic_kind(hdu_obj: Any, columns: Any) -> str | None:
    """Classify an HDU as 'table', 'wcs' (1D image with CRVAL1) or 'rows' (2×N or 3×N image), else None"""
    names = _table_columns(hdu_obj)
    if names:
        roles = _role_lookup(names, columns)
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
    with fits.open(path, memmap=False) as hdul:
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
    with fits.open(path, memmap=False) as hdul:
        found = _generic_hdu(hdul, hdu, columns)
        if found is None:
            raise ValueError('no table with wave/flux columns, 1D image with CRVAL1 or 2×N/3×N image found')
        idx, kind = found
        ext = hdul[idx]
        header = _header_dict(hdul[0].header, ext.header)
        sky = None
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
# reader 5: ASCII / CSV / TSV
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
    ncol, n_dropped = None, 0
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
            else:
                n_dropped += 1
    if n_dropped:
        log.warning('%s: dropped %d row(s) with a column count other than %s', path.name, n_dropped, ncol)
    if ncol is None or ncol < 2 or len(rows) < 2:
        raise ValueError('fewer than two numeric columns or rows')
    if names is not None and len(names) != ncol:
        names = None
    return np.asarray(rows, dtype=float), comments, header_rows, pseudo, names


def _read_ascii(path: Path, hdu: int | str | None = None, columns: Any = None) -> RawSpectrum:
    if _is_fits(path):
        raise ValueError('a FITS file, not text')
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
    from specutils import Spectrum as SpectrumCls

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


#: sky pixels further than this factor above the median |sky| are treated as bad (e.g. PypeIt's 1e10-level fills)
SKY_ABSURD_FACTOR = 1e3


def _clean_sky(sky: np.ndarray | None, name: str = '') -> np.ndarray | None:
    """
    The sky with absurd pixels set to NaN: non-finite, or |sky| > `SKY_ABSURD_FACTOR` × the median |sky|

    Pixels masked in the extraction are already dropped with their flux. NaN sky pixels are skipped by the sky
    zero point (interpolated over for the baseline, never correlated).
    """
    if sky is None:
        return None
    sky = np.asarray(sky, dtype=float).copy()
    finite = np.isfinite(sky)
    if not finite.any():
        log.warning('%s: the sky spectrum has no finite pixel; ignored', name)
        return None
    med = float(np.median(np.abs(sky[finite])))
    bad = ~finite | ((np.abs(sky) > SKY_ABSURD_FACTOR * med) if med > 0 else np.zeros(len(sky), dtype=bool))
    if bad.any():
        log.info('%s: %d sky pixel(s) are non-finite or above %g × the median |sky| (%.3g); set to NaN', name,
                 int(bad.sum()), SKY_ABSURD_FACTOR, med)
        sky[bad] = np.nan
    return sky


def _attach_arc_range(meta: dict[str, Any], path: Path, extname: str | None) -> None:
    """
    Set ``meta['arc_range']``, ``arc_file`` (PypeIt arc solutions are vacuum) and ``wave_rms`` (the arc-fit RMS [Å],
    the lower bound of sys_wavezp) from a neighbouring wavecal file
    """
    try:
        calib = find_wvcalib(path)
        span = None if calib is None else arc_line_range(calib, extname)
        rms = None if span is None else arc_rms(calib, extname)
    except (OSError, ValueError) as exc:
        log.warning('%s: could not read the arc-line range: %s', path.name, exc)
        return
    if span is None:
        return
    meta.update(arc_range=span, arc_file=str(calib), wave_rms=rms)
    log.info('%s: arc lines span %.1f-%.1f Å (vacuum), fit RMS %s Å, in %s', path.name, *span,
             'unknown' if rms is None else f'{rms:.3f}', calib)


def read_spectrum(path: str | Path, *, medium: str = 'auto', medium_method: str = 'ciddor',
                  hdu: int | str | None = None,
                  columns: Mapping[str, str | int] | None = None, resolution: float | None = None) -> Spectrum:
    """
    Read a one-dimensional spectrum (FITS, ASCII/CSV/TSV, else ``specutils``) and return it in vacuum Å.

    ``medium`` is 'auto' (read the header; PypeIt files default to vacuum; otherwise `ValueError`), 'air' or
    'vacuum' (overrides the header, a conflict is logged); ``medium_method`` is the air formula of `rvfitter.medium`.
    ``hdu`` selects an HDU (default: the first suitable; for a multi-object PypeIt spec1d the table matching the
    target name, else the highest-S/N science object, which is logged). ``columns`` maps the roles 'wave', 'flux' and
    one of 'err', 'ivar', 'var' to a column name or index. ``resolution`` (R = λ/Δλ) overrides the header's.

    Raises `ValueError` if the file cannot be read or the medium cannot be determined.
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
    with warnings.catch_warnings():  # Spextool's '#### Xspextool History ####' header dividers are not FITS cards
        warnings.filterwarnings('ignore', 'The following header keyword is invalid', AstropyUserWarning)
        for name, sniff, read in _READERS:
            try:
                if sniff is not None and not sniff(path, hdu, columns):
                    continue
                raw = read(path, hdu, columns)
                break
            except (OSError, ValueError, KeyError, IndexError) as exc:
                log.debug('%s: reader %s failed: %s', path.name, name, exc)
                errors.append(f'{name}: {exc}')
    if raw is None:
        try:
            raw = _read_specutils(path, hdu, columns)
        except (OSError, ValueError, KeyError, IndexError, IORegistryError) as exc:
            errors.append(f'specutils: {exc}')
            raise ValueError(f"Could not read {path} as a spectrum ({'; '.join(errors)})") from exc
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
    sky = _clean_sky(sky, path.name)

    meta = _meta_from_header(raw.header)
    meta.update(medium_in=medium_in, medium_source=source, reader=raw.reader)
    for key, out in (('RVF_HDU', 'hdu'), ('RVF_EXTNAME', 'extname'), ('RVF_NOBJ', 'n_objects')):
        if key in raw.header:  # which table of a multi-object file was read
            meta[out] = raw.header[key]
    r = resolution if resolution is not None else raw.resolution
    if r is not None and not (np.isfinite(r) and r > 0):
        raise ValueError(f'resolution must be positive, not {r}')
    if raw.reader in ('pypeit_spec1d', 'pypeit_coadd'):
        _attach_arc_range(meta, path, raw.header.get('RVF_EXTNAME'))
    log.debug('%s: %d pixels, %.1f-%.1f Å (vacuum), medium_in=%s (%s)', path.name, len(wave), wave[0], wave[-1],
              medium_in, source)
    return Spectrum(wave, flux, sigma, None if r is None else float(r), meta, str(path), sky)


# (name, sniff, read) in the order tried; ASCII has no sniffer (its reader raises for anything else) and `specutils`
# is the last resort in `read_spectrum`.
_READERS = (
    ('pypeit_spec1d', _sniff_pypeit_spec1d, _read_pypeit_spec1d),
    ('pypeit_coadd', _sniff_pypeit_coadd, _read_pypeit_coadd),
    ('spextool', _sniff_spextool, _read_spextool),
    ('fits_generic', _sniff_fits_generic, _read_fits_generic),
    ('ascii', None, _read_ascii),
)
