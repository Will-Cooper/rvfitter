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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.io.registry import IORegistryError
from astropy.utils.exceptions import AstropyUserWarning

from .barycentric import vel_corr_to_kms
from .calibration import slit
from .config import REGIME_EDGES
from .medium import to_vacuum
from .types import Spectrum

__all__ = ['read_spectrum', 'read_spectra', 'list_spectra', 'SpectrumEntry', 'split_source', 'cube_image',
           'brightest_spaxel', 'RawSpectrum', 'raw_headers', 'estimate_resolution', 'implausible_resolution',
           'SNR_USABLE_MIN', 'CUBE_RADIUS_DEFAULT']

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

# R from the pixel spacing, used only when neither the header nor the caller gives R
SAMPLING_PX = 2.5  # pixels per resolution element assumed (Nyquist sampling, as most reductions deliver)
# the estimate is trusted to this factor: real sampling is about 2-3.75 px per resolution element, and R varies by
# about 20 per cent along an order (IRTF spectral library, Rayner et al. 2009), so a documented range is widened by it
R_ESTIMATE_TOLERANCE = 1.5
# unknown instrument: an estimate above the low-resolution regime edge would change the regime defaults
# (`config.regime_for`) on a guess; rebinned or coadded spectra on a finer grid give such values
R_ESTIMATE_CEILING = REGIME_EDGES[0]

_RES_KEYS = ('RESOLUTION', 'SPECRES', 'SPEC_RES', 'RESOLVING', 'RESPOWER', 'RESOLPOW')
_VEL_KEYS = ('VHELIO', 'VBARY', 'VLSR', 'BCV', 'BERV', 'BARYCORR', 'HELIOCOR', 'BARY_CORR', 'VELOSYS', 'HELCORR',
             'QC VRAD BARYCOR', 'QC VRAD HELICOR', 'VEL_TYPE', 'VEL_CORR', 'RADVEL', 'VELFRAME')

#: `list_spectra`: an entry with a median S/N per pixel below this is listed but unusable ('auto' and 'all' skip it,
#: with a WARNING): at S/N < 3 per pixel even the strongest bands are barely above the noise
SNR_USABLE_MIN = 3.
#: JWST pipeline products: vacuum wavelengths, corrected to the barycentre by assign_wcs (a relativistic Doppler
#: correction from VELOSYS applied to all spectroscopic wavelengths; jwst-pipeline.readthedocs.io, assign_wcs), whether
#: or not the file records SPECSYS (x1d files do not)
JWST_MEDIUM = 'vacuum'
JWST_FRAME_SOURCE = 'JWST pipeline: assign_wcs applies the barycentric correction to all spectroscopic wavelengths'
#: JWST data-quality bit DO_NOT_USE; in other DQ arrays any non-zero value marks a bad pixel
JWST_DO_NOT_USE = 1
#: IFU cubes: aperture radius [spaxels] when ``input.cube.radius`` is unset. Three spaxels (0.3 arcsec on the NIRSpec
#: IFU) hold 95 per cent of the flux of the JWST pipeline's point-source extraction of VHS 1256-1257 b
#: (examples/jwst_ifu)
CUBE_RADIUS_DEFAULT = 3.
#: the collapsed cube ignores spaxels valid in fewer than this fraction of the wavelength planes that hold data
#: anywhere: edge spaxels with a handful of valid planes give meaningless medians (and would be the 'brightest')
CUBE_MIN_COVERAGE = 0.5
#: EXTNAME words of extensions that are never spectra (planes belonging to another extension) → their kind
_AUX_WORDS = {**dict.fromkeys(('ERR', 'ERROR', 'ERRS', 'SIGMA', 'UNCERT', 'NOISE', 'STDDEV'), 'error'),
              **dict.fromkeys(('VAR', 'VARIANCE', 'IVAR', 'INVVAR'), 'variance'),
              **dict.fromkeys(('DQ', 'QUALITY'), 'dq'),
              **dict.fromkeys(('MASK', 'BPM'), 'mask'), **dict.fromkeys(('WAVE', 'WAVELENGTH', 'LAMBDA'), 'wavelength'),
              'SKY': 'sky', **dict.fromkeys(('BKG', 'BKGD', 'BACKGROUND'), 'background'),
              **dict.fromkeys(('WMAP', 'WHT', 'WEIGHT'), 'weight map')}
_SPECTRUM_KINDS = ('object', 'cube')
_SOURCE_RE = re.compile(r'(.+)\[(\d+)\]')

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
    meta
        Entries added to `Spectrum.meta` as they are (e.g. how a cube was extracted)
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
    meta: dict[str, Any] = field(default_factory=dict)


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
    for key in ('CTYPE1', 'CTYPE3', 'CTYPE'):
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

    - object: the source of one entry of a multi-object file (``RVF_OBJECT``: PypeIt MASKDEF_OBJNAME, JWST SRCNAME),
      else ``OBJECT``, ``TARGNAME``, ``TARGET``, ``OBJNAME`` (first non-empty).
    - ra, dec: ``RA``/``DEC`` (or ``OBJRA``/``OBJDEC``, ``TELRA``/``TELDEC``, ``TCS_RA``/``TCS_DEC``, JWST
      ``TARG_RA``/``TARG_DEC``); floats are degrees, sexagesimal strings hours (RA) and degrees (Dec).
    - mjd_mid: ``MJD_MID``/``MJD-MID``/JWST ``EXPMID`` (already mid-exposure), else ``MJD``, ``MJD-OBS`` or
      ``DATE-OBS`` (+ ``TIME-OBS``), which mean the exposure start, plus ``EXPTIME``/2; ``mjd_source`` records
      which (a reader that sets ``MJD_MID`` names its origin in ``RVF_MJD_SOURCE``).
    - site: ``OBSERVAT``, else ``TELESCOP`` (``telescope`` is always ``TELESCOP``); site_lat/site_lon: ``SITELAT``...
    - exptime: ``EXPTIME``, ``EXPOSURE`` or ``ITIME`` [s]; airmass: ``AIRMASS``, else the mean of ``AMSTART``/``AMEND``.
    - frame_in: ``VEL_TYPE``, else ``SPECSYS`` 'BARYCENT'/'HELIOCEN' (FITS WCS paper III), else 'barycentric' for
      JWST (`JWST_FRAME_SOURCE`); ``frame_source`` records which. vel_corr: ``VEL_CORR`` (PypeIt's factor already
      applied to the wavelengths) and ``vel_corr_kms``, the same as a velocity.
    """
    meta: dict[str, Any] = {}
    obj, _ = _first(header, ('RVF_OBJECT', 'OBJECT', 'TARGNAME', 'TARGET', 'OBJNAME'))
    if obj is not None and str(obj).strip().lower() not in ('none', 'unknown', 'n/a', ''):
        meta['object'] = str(obj)
    ra, _ = _first(header, ('RA', 'OBJRA', 'TELRA', 'RA_DEG', 'TCS_RA', 'TARG_RA'))
    dec, _ = _first(header, ('DEC', 'OBJDEC', 'TELDEC', 'DEC_DEG', 'TCS_DEC', 'TARG_DEC'))
    meta['ra'] = _parse_coord(ra, True)
    meta['dec'] = _parse_coord(dec, False)
    exptime = _as_float(_first(header, ('EXPTIME', 'EXPOSURE', 'ITIME'))[0])
    meta['exptime'] = exptime
    # mid-exposure time
    mjd_mid, source = None, None
    mid, key = _first(header, ('MJD_MID', 'MJD-MID', 'MJDMID', 'EXPMID'))
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
    frame, key = _first(header, ('VEL_TYPE',))
    if frame is None:
        frame, key = {'BARYCENT': 'barycentric', 'HELIOCEN': 'heliocentric'}.get(
            str(header.get('SPECSYS', '')).strip().upper()), 'SPECSYS'
    if frame is None and _is_jwst(header):
        frame, key = 'barycentric', None
    meta['frame_in'] = None if frame is None else str(frame).strip().lower()
    meta['frame_source'] = None if frame is None else f'header:{key}' if key else JWST_FRAME_SOURCE
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
    - ``instrument``: ``INSTRUME``, else ``INSTR`` (Spextool); ``instrument_names``: every INSTRUME/INSTR value, then
      '<TELESCOP>_<INSTRUME>_<GRATING>' when all three exist (the per-mode rows of ``data/instruments.csv``).
    - ``trace_fwhm_pix``: PypeIt's object FWHM along the slit [binned pixels].
    """
    out: dict[str, Any] = {}
    inst, _ = _first(header, ('INSTRUME', 'INSTRUMENT', 'INSTR'))
    out['instrument'] = None if inst is None else str(inst).strip()
    # every INSTRUME/INSTR value: Spextool files carry both ('SpeX Spectrograph', 'uSpeX'), and the table keys on INSTR
    out['instrument_names'] = tuple(str(header[k]).strip() for k in ('INSTRUME', 'INSTRUMENT', 'INSTR')
                                    if header.get(k) is not None)
    if all(header.get(k) for k in ('TELESCOP', 'INSTRUME', 'GRATING')):  # a mode row, e.g. 'JWST_NIRSPEC_G395H'
        out['instrument_names'] += ('_'.join(str(header[k]).strip() for k in ('TELESCOP', 'INSTRUME', 'GRATING')),)
    pyp =header.get('PYP_SPEC')
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


def estimate_resolution(wave: np.ndarray) -> float:
    """R ≈ median(λ / (`SAMPLING_PX` Δλ)) over the pixels, assuming Nyquist sampling; `ValueError` if no spacing"""
    wave = np.asarray(wave, dtype=float)
    d = np.diff(wave)
    good = np.isfinite(d) & (d > 0)
    if not good.any():
        raise ValueError('cannot estimate the resolving power: no increasing wavelength steps')
    return float(np.median(wave[1:][good] / (SAMPLING_PX * d[good])))


def implausible_resolution(r: float, meta: Mapping[str, Any]) -> str | None:
    """
    Why an R estimated from the pixel spacing is implausible for the instrument of ``meta``, else None

    Known instrument (``data/instruments.csv`` row by PYP_SPEC, INSTRUME, INSTR, with an R range): outside the
    documented range widened by `R_ESTIMATE_TOLERANCE`. Otherwise: above `R_ESTIMATE_CEILING`.
    """
    row = slit.instrument(meta.get('pypeit_spec'), *meta.get('instrument_names', ()))
    if row is not None and row.resolution_min is not None:
        lo, hi = row.resolution_min / R_ESTIMATE_TOLERANCE, row.resolution_max * R_ESTIMATE_TOLERANCE
        if lo <= r <= hi:
            return None
        return (f'outside {lo:.0f}-{hi:.0f}, the documented range of {row.key} ({row.resolution_min:g}-'
                f'{row.resolution_max:g}, {row.resolution_source}) widened by a factor {R_ESTIMATE_TOLERANCE:g}')
    if r <= R_ESTIMATE_CEILING:
        return None
    return f'above {R_ESTIMATE_CEILING:g} for an instrument without a documented R range'


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
        idx = hdul.index_of(hdu)
        hdul[idx]  # index_of passes any integer through
        return idx
    except (KeyError, IndexError) as exc:
        raise ValueError(f'HDU {hdu!r} not found in file with {len(hdul)} HDUs') from exc


def _is_jwst(header: Mapping[str, Any]) -> bool:
    return str(header.get('TELESCOP', '')).strip().upper() == 'JWST'


def _aux_kind(extname: Any) -> str | None:
    """What an extension holds if its EXTNAME marks it as never a spectrum (error, variance, dq, mask, wavelength,
    sky, background, weight map: any word of the name in `_AUX_WORDS`), else None"""
    return next((_AUX_WORDS[w] for w in re.split(r'[^A-Z]+', str(extname).upper()) if w in _AUX_WORDS), None)


def _dq_bad(dq: np.ndarray, header: Mapping[str, Any]) -> np.ndarray:
    """Bad pixels of a DQ array: the DO_NOT_USE bit for JWST, any non-zero value otherwise"""
    dq = np.asarray(dq).astype(np.int64)
    return (dq & JWST_DO_NOT_USE) != 0 if _is_jwst(header) else dq != 0


def split_source(source: str | Path) -> tuple[Path, int | None]:
    """
    ``'file.fits[3]'`` (the `Spectrum.source` of one entry of a multi-object file) → (Path('file.fits'), 3); any
    other path (or an existing file whose name ends in '[n]') → (path, None)
    """
    m = _SOURCE_RE.fullmatch(str(source))
    if m is None or Path(source).exists():
        return Path(source), None
    return Path(m.group(1)), int(m.group(2))


def _to_flambda(wave: np.ndarray, flux: np.ndarray, sigma: np.ndarray, unit: Any, header: Mapping[str, Any],
                name: str) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """
    The one fν → fλ conversion: flux and σ per frequency (e.g. Jy) → erg s⁻¹ cm⁻² Å⁻¹ at ``wave`` [Å]; surface
    brightness (e.g. MJy/sr) is first multiplied by the pixel solid angle ``PIXAR_SR`` (`ValueError` without it). Any
    other unit (fλ, counts, unknown) is returned unchanged. The third item is ``{'flux_conversion': what was done}``
    (empty if nothing), for `Spectrum.meta`; `read_spectrum` logs it.
    """
    from astropy import units as u
    q = None if unit is None else u.Unit(str(unit).strip(), parse_strict='silent')
    if q is None or isinstance(q, u.UnrecognizedUnit):
        return flux, sigma, {}
    scale, what = 1., str(unit)
    if q.is_equivalent(u.Jy / u.sr):
        pixar = _as_float(header.get('PIXAR_SR'))
        if pixar is None or pixar <= 0:
            raise ValueError(f'{name}: the flux is a surface brightness ({unit}); converting it to a flux needs the '
                             'pixel solid angle PIXAR_SR [sr], which the header lacks')
        q, scale, what = q * u.sr, pixar, f'{unit} × PIXAR_SR {pixar:.4g} sr'
    if not q.is_equivalent(u.Jy):
        return flux, sigma, {}
    factor = scale * (1. * q).to(u.erg / u.s / u.cm ** 2 / u.AA, equivalencies=u.spectral_density(wave * u.AA)).value
    return flux * factor, sigma * factor, {'flux_conversion': f'{what} (per unit frequency) to fλ [erg/s/cm2/Å]'}


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
        if _clean_value(ext.header.get('MASKDEF_OBJNAME')):  # a slit-mask object: its own name, not the mask's
            header['RVF_OBJECT'] = str(ext.header['MASKDEF_OBJNAME']).strip()
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
    candidates = [_hdu_index(hdul, hdu)] if hdu is not None else [i for i in range(len(hdul))
                                                                    if not _aux_kind(hdul[i].name)]
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


def _wcs_wave(header: Mapping[str, Any], n: int, axis: int = 1) -> np.ndarray:
    """
    Wavelength axis from CRVALi/CDELTi (× PCi_i; or CDi_i)/CRPIXi of WCS axis ``axis`` (1: 1D image, 3: cube).

    Log-linear if ``DC-FLAG`` = 1 (IRAF: CRVAL1 and CDELT1 in log10 λ) or ``CTYPEi`` ends in '-LOG'
    (FITS WCS paper III: λ = λ_r exp[(w − w_r)/λ_r]); a tabulated axis ('-TAB') raises `ValueError`.
    """
    a = axis
    crval = float(header[f'CRVAL{a}'])
    cdelt = header.get(f'CDELT{a}', header.get(f'CD{a}_{a}'))
    if cdelt is None:
        raise ValueError(f'the image has CRVAL{a} but neither CDELT{a} nor CD{a}_{a}')
    cdelt = float(cdelt) * float(header.get(f'PC{a}_{a}', 1.)) if f'CDELT{a}' in header else float(cdelt)
    crpix = float(header.get(f'CRPIX{a}', 1.))
    x = np.arange(n) + 1.
    lin = crval + (x - crpix) * cdelt
    ctype = str(header.get(f'CTYPE{a}', '')).upper()
    if ctype.endswith('-TAB'):
        raise ValueError(f'CTYPE{a} = {ctype!r}: a tabulated wavelength axis is not supported')
    dc = _as_float(header.get('DC-FLAG'))
    if a == 1 and dc is not None and int(dc) == 1:
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
        header.update(RVF_HDU=idx, RVF_EXTNAME=str(ext.name))
        sky, flux_unit = None, None
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
            flux_unit = _table_wave_unit(ext, names[roles['flux']])
            if 'DQ' in names:  # JWST x1d
                flux = np.where(_dq_bad(data['DQ'], header).ravel(), np.nan, flux)
            if _clean_value(ext.header.get('SRCNAME')):  # JWST: one extension per source
                header['RVF_OBJECT'] = str(ext.header['SRCNAME']).strip()
        elif kind == 'wcs':
            flux = np.asarray(np.squeeze(ext.data), dtype=float)
            wave = _wcs_wave(header, len(flux))
            # optional error image: an ERR... image of the same length and EXTVER (FITS default 1), as in STScI files
            ver = ext.header.get('EXTVER', 1)
            errs = [other for j, other in enumerate(hdul) if j != idx and getattr(other, 'data', None) is not None
                    and not _table_columns(other) and _aux_kind(other.name) == 'error'
                    and np.size(other.data) == len(flux) and other.header.get('EXTVER', 1) == ver]
            if len(errs) > 1:  # a warning, not an error: other readers would retry the file silently
                log.warning('%s: HDU %d: %d error images (%s) match its length and EXTVER %s, so none is used; give '
                            'each its own EXTVER', path.name, idx, len(errs), ', '.join(e.name for e in errs), ver)
            one = len(errs) == 1
            sigma = _clean_sigma(np.asarray(errs[0].data, dtype=float).ravel()) if one else np.full(len(flux), np.nan)
            unit = _image_unit(header)
        else:  # rows
            arr = np.asarray(np.squeeze(ext.data), dtype=float)
            if arr.shape[0] > 3:
                arr = arr.T
            wave, flux = arr[0], arr[1]
            sigma = _clean_sigma(arr[2]) if arr.shape[0] > 2 else np.full(len(wave), np.nan)
            unit = _image_unit(header)
    wave = _wave_to_angstrom(wave, unit, path.name)
    flux, sigma, note = _to_flambda(wave, flux, sigma, flux_unit, header, path.name)
    medium, source = _medium_from_header(header)
    jwst = _is_jwst(header)
    return RawSpectrum(wave, flux, sigma, header, medium, source, JWST_MEDIUM if jwst else None,
                       _header_resolution(header), 'jwst' if jwst else 'fits_generic', sky, note)


# ----------------------------------------------------------------------------------------------------------------------
# IFU cubes
# ----------------------------------------------------------------------------------------------------------------------

def _cube_index(hdul: fits.HDUList, hdu: int | str | None = None) -> int | None:
    """The HDU of an IFU cube: a 3D image whose third WCS axis is a wavelength (CTYPE3 'WAVE' or 'AWAV')"""
    for idx in [_hdu_index(hdul, hdu)] if hdu is not None else range(len(hdul)):
        h = hdul[idx].header
        if (h.get('NAXIS') == 3 and str(h.get('CTYPE3', '')).upper().split('-')[0] in ('WAVE', 'AWAV')
                and not _aux_kind(hdul[idx].name)):
            return idx
    return None


def _load_cube(path: Path, hdu: int | str | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int]:
    """
    (SCI [λ, y, x] with DQ-flagged and non-finite voxels NaN, its σ (NaN without an ERR plane), flat header, HDU)

    The ERR and DQ planes are the extensions of the same shape whose EXTNAME says so (JWST s3d layout).
    """
    with fits.open(path, memmap=False) as hdul:
        idx = _cube_index(hdul, hdu)
        if idx is None:
            raise ValueError('no 3D image with a wavelength axis (CTYPE3 WAVE or AWAV)')
        header = _header_dict(hdul[0].header, hdul[idx].header)
        sci = np.array(hdul[idx].data, dtype=float)
        planes = {_aux_kind(h.name): h.data for h in hdul if h.is_image and h.header.get('NAXIS') == 3
                  and h.data is not None and h.data.shape == sci.shape}
        err, dq = planes.get('error'), planes.get('dq')
        err = np.full(sci.shape, np.nan) if err is None else _clean_sigma(err)
    bad = ~np.isfinite(sci) | (False if dq is None else _dq_bad(dq, header))
    sci[bad] = np.nan
    if not np.isfinite(err).any():
        log.warning('%s: the cube has no ERR plane: the extracted spectrum has no uncertainties', path.name)
    return sci, err, header, idx


def _collapse(sci: np.ndarray) -> np.ndarray:
    """Median over wavelength of each spaxel; NaN where valid in < `CUBE_MIN_COVERAGE` of the planes with data"""
    valid = np.isfinite(sci)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN spaxels
        image = np.nanmedian(sci, axis=0)
    image[valid.sum(axis=0) < CUBE_MIN_COVERAGE * valid.any(axis=(1, 2)).sum()] = np.nan
    return image


def cube_image(path: str | Path, hdu: int | str | None = None) -> np.ndarray:
    """
    The median-collapsed image [y, x] of an IFU cube on which the extraction centres (DQ-flagged voxels excluded;
    spaxels valid in fewer than `CUBE_MIN_COVERAGE` of the wavelength planes are NaN)
    """
    return _collapse(_load_cube(Path(path), hdu)[0])


def brightest_spaxel(image: np.ndarray) -> tuple[int, int]:
    """(x, y) [0-based pixels] of the brightest finite spaxel of a collapsed image; `ValueError` if none"""
    if not np.isfinite(image).any():
        raise ValueError('the collapsed cube has no valid spaxel')
    y, x = np.unravel_index(np.nanargmax(image), image.shape)
    return int(x), int(y)


def _read_cube(path: Path, hdu: int | str | None = None, cube: Any = None) -> RawSpectrum:
    """
    A spectrum extracted from an IFU cube in a circular aperture

    Centre ``cube.position`` (x, y) [0-based pixels of the collapsed image], else the brightest spaxel of the
    median-collapsed cube (`cube_image`); radius ``cube.radius`` [spaxels], else `CUBE_RADIUS_DEFAULT`. The spaxels
    whose centres lie within the radius are summed, σ in quadrature. A wavelength plane with a DQ-flagged or missing
    spaxel in the aperture is dropped (a partial sum would fake an absorption feature); the count is logged. Surface
    brightness is converted with PIXAR_SR (`_to_flambda`). ``meta['extraction']`` records all this, and
    ``meta['no_slit']`` makes the slit-centring term not apply.
    """
    sci, err, header, idx = _load_cube(path, hdu)
    nw, ny, nx = sci.shape
    position, radius = (None, None) if cube is None else (cube.position, cube.radius)  # cube: config.CubeConfig
    if position is None:
        (x, y), centre_source = brightest_spaxel(_collapse(sci)), 'brightest spaxel of the median-collapsed cube'
    else:
        x, y = (float(v) for v in position)
        if not (0 <= x <= nx - 1 and 0 <= y <= ny - 1):
            raise ValueError(f'input.cube.position ({x:g}, {y:g}) is outside the {nx} × {ny} spaxel image')
        centre_source = 'input.cube.position'
    r, radius_source = ((CUBE_RADIUS_DEFAULT, 'default') if radius is None else (float(radius), 'input.cube.radius'))
    if not r > 0:
        raise ValueError(f'input.cube.radius = {r:g} must be positive')
    yy, xx = np.mgrid[:ny, :nx]
    aperture = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    if not aperture.any():
        raise ValueError(f'the aperture of radius {r:g} spaxels at ({x:g}, {y:g}) contains no spaxel centre')
    flux_ap, err_ap = sci[:, aperture], err[:, aperture]
    incomplete = ~np.isfinite(flux_ap).all(axis=1)
    held = np.isfinite(sci).any(axis=(1, 2))
    flux = np.where(incomplete, np.nan, flux_ap.sum(axis=1))
    sigma = np.sqrt((err_ap ** 2).sum(axis=1))
    if not np.isfinite(flux).any():
        raise ValueError(f'{path.name}: no wavelength plane has every spaxel of the aperture valid')
    dropped = int((incomplete & held).sum())
    text = (f'IFU cube HDU {idx} extracted in a circular aperture at x={x:g}, y={y:g} ({centre_source}), radius {r:g} '
            f'spaxels ({radius_source}), {int(aperture.sum())} spaxels; {dropped} of the {int(held.sum())} wavelength '
            'planes with data dropped (a flagged or missing spaxel in the aperture)')
    if dropped > 0.5 * held.sum():
        log.warning('%s: the aperture loses %d of %d wavelength planes: it probably reaches the edge of the field; '
                    'move it (input.cube.position) or shrink it (input.cube.radius)', path.name, dropped,
                    int(held.sum()))
    wave = _wave_to_angstrom(_wcs_wave(header, nw, axis=3), header.get('CUNIT3'), path.name)
    flux, sigma, note = _to_flambda(wave, flux, sigma, header.get('BUNIT'), header, path.name)
    header.update(RVF_HDU=idx, RVF_EXTNAME=str(header.get('EXTNAME', idx)))
    medium, source = _medium_from_header(header)
    meta = {'extraction': {'method': 'circular aperture', 'x': float(x), 'y': float(y), 'centre_source': centre_source,
                           'radius': r, 'radius_source': radius_source, 'n_spaxels': int(aperture.sum()),
                           'planes_dropped': dropped, 'text': text},
            'no_slit': 'IFU cube extraction', **note}
    return RawSpectrum(wave, flux, sigma, header, medium, source, JWST_MEDIUM if _is_jwst(header) else None,
                       _header_resolution(header), 'ifu_cube', meta=meta)


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
    keep = np.diff(wave, prepend=-np.inf) > 0  # drop duplicate wavelengths, keeping the first
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
                  columns: Mapping[str, str | int] | None = None, resolution: float | None = None,
                  cube: Any = None) -> Spectrum:
    """
    Read a one-dimensional spectrum (FITS, ASCII/CSV/TSV, else ``specutils``) and return it in vacuum Å.

    ``medium`` is 'auto' (read the header; PypeIt files default to vacuum; otherwise `ValueError`), 'air' or
    'vacuum' (overrides the header, a conflict is logged); ``medium_method`` is the air formula of `rvfitter.medium`.
    ``path`` may be ``'file.fits[3]'`` for HDU 3 (the source `read_spectra` gives one entry of a multi-object file).
    ``hdu`` selects an HDU (default: the first suitable one that is not an error, DQ, sky... extension (`_aux_kind`;
    naming one raises); for a multi-object PypeIt spec1d the table matching the target name, else the highest-S/N
    science object, which is logged). An IFU cube is extracted in an aperture set by ``cube`` (``input.cube``: its
    ``position`` and ``radius``, see `_read_cube`). Fluxes per unit frequency are converted to fλ (`_to_flambda`).
    ``columns`` maps the roles 'wave', 'flux' and one of 'err', 'ivar', 'var' to a column name or index.
    ``resolution`` (R = λ/Δλ) overrides the header's; without either, R is estimated from the pixel spacing
    (`estimate_resolution`, WARNING), and checked against the instrument (`implausible_resolution`).
    ``meta['resolution_source']`` is 'config', 'header' or 'estimated'; ``meta['resolution_implausible']`` is True
    for an estimate that fails the check.

    Raises `ValueError` if the file cannot be read or the medium cannot be determined.
    """
    given = str(path)
    path, at = split_source(path)
    if at is not None:
        if hdu is not None and hdu != at:
            raise ValueError(f'{given}: HDU {at} in the source but hdu={hdu!r} given')
        hdu = at
    medium = str(medium).strip().lower()
    medium = {'vac': 'vacuum'}.get(medium, medium)
    if medium not in ('auto', 'air', 'vacuum'):
        raise ValueError(f"medium must be 'auto', 'air' or 'vacuum', not {medium!r}")
    if not path.is_file():
        raise FileNotFoundError(f'Spectrum file not found: {path}')
    errors: list[str] = []
    raw: RawSpectrum | None = None
    if _is_fits(path):
        with fits.open(path, memmap=False) as hdul:
            aux = None if hdu is None else _aux_kind(hdul[_hdu_index(hdul, hdu)].name)
            is_cube = _cube_index(hdul, hdu) is not None
        if aux:
            raise ValueError(f'{path}: HDU {hdu!r} is a {aux} extension, not a spectrum')
        if is_cube:  # a cube that cannot be extracted raises: no other reader would make sense of it
            raw = _read_cube(path, hdu, cube)
    with warnings.catch_warnings():  # Spextool's '#### Xspextool History ####' header dividers are not FITS cards
        warnings.filterwarnings('ignore', 'The following header keyword is invalid', AstropyUserWarning)
        for name, sniff, read in () if raw is not None else _READERS:
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
    meta.update(medium_in=medium_in, medium_source=source, reader=raw.reader, **raw.meta)
    for key in ('flux_conversion', 'extraction'):
        if key in raw.meta:
            log.info('%s: %s', path.name, raw.meta[key]['text'] if key == 'extraction' else raw.meta[key])
    for key, out in (('RVF_HDU', 'hdu'), ('RVF_EXTNAME', 'extname'), ('RVF_NOBJ', 'n_objects')):
        if key in raw.header:  # which table of a multi-object file was read
            meta[out] = raw.header[key]
    r, meta['resolution_source'] = ((resolution, 'config') if resolution is not None else
                                    (raw.resolution, 'header') if raw.resolution is not None else (None, 'estimated'))
    if r is not None and not (np.isfinite(r) and r > 0):
        raise ValueError(f'resolution must be positive, not {r}')
    meta['resolution_implausible'] = False
    if r is None:
        r = estimate_resolution(wave)
        why = implausible_resolution(r, meta)
        meta['resolution_implausible'] = why is not None
        if why:
            log.warning('%s: no resolving power in the header; the R ≈ %.0f estimated from the pixel spacing is %s; '
                        'the spectrum may be rebinned: type the resolving power in the Setup tab (input.resolution).',
                        path.name, r, why)
        else:
            log.warning('%s: no resolving power in the header; estimated R ≈ %.0f from the pixel spacing (assuming '
                        '%g pixels per resolution element). Set input.resolution if you know R.', path.name, r,
                        SAMPLING_PX)
    if raw.reader in ('pypeit_spec1d', 'pypeit_coadd'):
        _attach_arc_range(meta, path, raw.header.get('RVF_EXTNAME'))
    log.debug('%s: %d pixels, %.1f-%.1f Å (vacuum), medium_in=%s (%s)', path.name, len(wave), wave[0], wave[-1],
              medium_in, source)
    return Spectrum(wave, flux, sigma, float(r), meta, given, sky)


# ----------------------------------------------------------------------------------------------------------------------
# multi-object files
# ----------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SpectrumEntry:
    """
    One candidate spectrum of a file (`list_spectra`)

    Attributes
    ----------
    hdu
        HDU index, for ``read_spectrum(hdu=...)``; None for a file read as a whole (text, a single spectrum)
    name
        PypeIt MASKDEF_OBJNAME else NAME (the trace), JWST SRCNAME, else EXTNAME (else the file name)
    kind
        'object' or 'cube' (spectra), or what else the extension holds: 'background' (a JWST background slit),
        'error', 'variance', 'dq', 'mask', 'wavelength', 'sky', 'weight map' (by EXTNAME, `_AUX_WORDS`)
    snr
        Median S/N per pixel (NaN if unknown)
    usable
        Whether 'auto' and 'all' select it
    reason
        Why it is not usable ('' if it is)
    named
        ``name`` is the source's own (MASKDEF_OBJNAME, SRCNAME), not a trace or extension label
    """
    hdu: int | None
    name: str
    kind: str
    snr: float
    usable: bool
    reason: str = ''
    named: bool = False


def _entry(hdu: int, name: str, kind: str, read: Callable[[], RawSpectrum | float], named: bool = False,
           why: str = '') -> SpectrumEntry:
    """An entry; ``read`` gives its S/N (or the raw spectrum to take it from): a failure makes it unusable"""
    if why:
        return SpectrumEntry(hdu, name, kind, float('nan'), False, why, named)
    try:
        got = read()
    except (OSError, ValueError, KeyError, IndexError) as exc:
        return SpectrumEntry(hdu, name, kind, float('nan'), False, f'unreadable: {exc}', named)
    if isinstance(got, RawSpectrum):
        if (np.isfinite(got.flux) & np.isfinite(got.wave)).sum() < 2:
            return SpectrumEntry(hdu, name, kind, float('nan'), False, 'fewer than two valid pixels', named)
        ok = np.isfinite(got.flux) & np.isfinite(got.sigma) & (got.sigma > 0)
        got = float(np.median(got.flux[ok] / got.sigma[ok])) if ok.any() else float('nan')
    low = bool(got < SNR_USABLE_MIN)
    return SpectrumEntry(hdu, name, kind, got, not low,
                         f'median S/N {got:.1f} per pixel is below {SNR_USABLE_MIN:g}' if low else '', named)


def list_spectra(path: str | Path, cube: Any = None) -> list[SpectrumEntry]:
    """
    The candidate spectra of a file, each with its kind, median S/N and whether it is usable

    - PypeIt spec1d: one entry per object table (S/N of OPT, else BOX).
    - FITS: one entry per extension that holds a spectrum (a table with wave/flux columns, e.g. one JWST x1d
      EXTRACT1D per source; a 1D or 2/3×N image) or an IFU cube (S/N of its extraction with ``cube``, see
      `read_spectrum`); JWST background slits (SRCNAME 'background…') and extensions whose EXTNAME marks error,
      variance, DQ, mask, wavelength, sky, background or weight planes are listed as unusable, with the reason.
    - Anything else (text, a Spextool image, ...): one entry read as a whole (hdu None, S/N not computed).

    Entries below `SNR_USABLE_MIN` are unusable, with the reason; nothing is dropped silently.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Spectrum file not found: {path}')
    whole = [SpectrumEntry(None, path.name, 'object', float('nan'), True)]
    if not _is_fits(path):
        return whole
    with fits.open(path, memmap=False) as hdul:
        tables = _pypeit_tables(hdul)
        if tables:
            out = []
            for i in tables:
                h, own = hdul[i].header, _clean_value(hdul[i].header.get('MASKDEF_OBJNAME'))
                out.append(_entry(i, str(own or h.get('NAME', hdul[i].name)), 'object',
                                  lambda i=i: _median_snr(hdul[i]), named=bool(own)))
            return out
        cube_idx = _cube_index(hdul)
        found = []
        for i, ext in enumerate(hdul):
            aux = _aux_kind(ext.name)
            if aux and ext.header.get('NAXIS', 0):
                found.append((i, ext.name, aux, False, f'EXTNAME {ext.name}: {aux} plane, not a spectrum'))
            elif i == cube_idx:
                found.append((i, ext.name, 'cube', False, ''))
            elif not aux and _generic_kind(ext, None):
                src = _clean_value(ext.header.get('SRCNAME'))
                bkg = str(src or '').lower().startswith('background')
                found.append((i, str(src or ext.name), 'background' if bkg else 'object', bool(src),
                              f'JWST background slit (SRCNAME {src})' if bkg else ''))
    if not any(kind in _SPECTRUM_KINDS for _, _, kind, _, _ in found):
        return whole
    return [_entry(i, name, kind, lambda i=i, kind=kind: _read_cube(path, i, cube) if kind == 'cube' else
                   _read_fits_generic(path, i), named, why) for i, name, kind, named, why in found]


def _target_entry(path: Path, entries: list[SpectrumEntry]) -> SpectrumEntry | None:
    """The one entry named after the primary header's OBJECT/TARGNAME/TARGET (equal or containing it), else None"""
    target, _ = _first(_header_dict(fits.getheader(path, 0)), ('OBJECT', 'TARGNAME', 'TARGET'))
    t = _norm_name(target or '')
    hits = [e for e in entries if len(t) > 3 and (t == _norm_name(e.name) or t in _norm_name(e.name))]
    return hits[0] if len(hits) == 1 else None


def _select(path: Path, entries: list[SpectrumEntry], objects: str | list[str | int]) -> list[int | None]:
    """
    The HDUs to read for ``input.objects`` (None: the reader's own choice of the target)

    'auto': the entry named after the header target, if exactly one is; otherwise every usable entry when the
    entries carry their own source names (a MOS file), else the target ('target'). 'target': the reader's choice
    (`read_spectrum` with no ``hdu``). 'all': every usable entry. A list: the entries whose HDU index or name is in
    it (none raises). Unusable entries are skipped with a WARNING, or read with one when listed explicitly.
    """
    name = path.name
    label = ', '.join(f'HDU {e.hdu} {e.name} (S/N {e.snr:.1f}{"" if e.usable else "; unusable"})' for e in entries)
    if isinstance(objects, str):
        mode, why = objects, ''
        if mode == 'auto':
            hit = _target_entry(path, entries)
            if hit is not None:
                log.info("%s: %d spectra (%s); input.objects 'auto': reading HDU %d (%s), named after the header "
                         'target', name, len(entries), label, hit.hdu, hit.name)
                return [hit.hdu]
            mode = 'all' if any(e.named for e in entries) else 'target'
            why = (" ('auto': the spectra carry their own source names)" if mode == 'all' else
                   " ('auto': the spectra are not named after their sources, so they are taken as traces of one "
                   "target; set input.objects = 'all' to measure each)")
        if mode == 'target':
            log.info("%s: %d spectra (%s); input.objects 'target'%s: reading the target", name, len(entries), label,
                     why)
            return [None]
        if mode != 'all':
            raise ValueError(f"input.objects must be 'auto', 'target', 'all' or a list, not {objects!r}")
        picks = [e for e in entries if e.usable]
        for e in entries:
            if not e.usable:
                log.warning('%s: skipping HDU %s (%s): %s', name, e.hdu, e.name, e.reason)
        log.info("%s: %d spectra (%s); input.objects 'all'%s: reading %d", name, len(entries), label, why, len(picks))
    else:
        wanted = {str(x) for x in objects}
        picks = [e for e in entries if str(e.hdu) in wanted or e.name in wanted]
        for e in picks:
            if not e.usable:
                log.warning('%s: HDU %s (%s) is read because input.objects names it, although %s', name, e.hdu,
                            e.name, e.reason)
        log.info('%s: %d spectra (%s); input.objects %s: reading %s', name, len(entries), label, list(objects),
                 ', '.join(f'HDU {e.hdu}' for e in picks) or 'none')
    if not picks:
        raise ValueError(f'input.objects = {objects!r} selects none of the spectra of {name} ({label})')
    return [e.hdu for e in picks]


def read_spectra(source: str | Path, *, objects: str | list[str | int] = 'auto', hdu: int | str | None = None,
                 **read_kwargs: Any) -> list[Spectrum]:
    """
    Read the spectra of a file that ``objects`` (``input.objects``, see `_select`) selects, each as its own object

    ``source`` is a path, or ``'file.fits[3]'`` for one HDU. With ``hdu`` (``input.hdu``), a single HDU, or a file with
    one candidate spectrum (`list_spectra`), this is `read_spectrum`. Otherwise the source of each spectrum is
    ``'file.fits[<hdu>]'`` and ``meta['entry']`` its entry name; an entry without its own source name, other than the
    target (`read_spectrum` with no ``hdu``), is named '<object> <entry>' (e.g. 'J1745-1640 SPAT0168-SLIT0457-DET02')
    however many are read at once, so that grouping and rereading a source never merge them. ``read_kwargs`` go to
    `read_spectrum`.
    """
    path, at = split_source(source)
    if at is not None:
        if hdu is not None and hdu != at:
            raise ValueError(f'{source}: HDU {at} in the source but hdu={hdu!r} given')
        hdu = at
    entries = [e for e in list_spectra(path, cube=read_kwargs.get('cube')) if e.kind in _SPECTRUM_KINDS]
    if len(entries) < 2:  # the candidate's own HDU: the reader's default may be a background slit
        return [read_spectrum(path, hdu=hdu if hdu is not None else entries[0].hdu, **read_kwargs)]
    picks = [hdu] if hdu is not None else _select(path, entries, objects)
    read = []
    for h in picks:
        spec = read_spectrum(path, hdu=h, **read_kwargs)
        entry = next((e for e in entries if e.hdu == spec.meta.get('hdu')), None)
        if entry is None:
            raise ValueError(f'{path.name}: HDU {h!r} is not one of the spectra ({", ".join(e.name for e in entries)})')
        read.append((h, spec, entry))
    # the target (the reader's choice with no HDU) keeps the plain object name, whichever call reads it
    target = next((spec.meta['hdu'] for h, spec, _ in read if h is None), None)
    if target is None and not all(entry.named for _, _, entry in read):
        target = read_spectrum(path, **read_kwargs).meta['hdu']
    out = []
    for _, spec, entry in read:
        meta = {**spec.meta, 'entry': entry.name}
        if not entry.named and entry.hdu != target:
            meta['object'] = f"{spec.meta.get('object') or path.stem} {entry.name}"
        out.append(replace(spec, source=f'{path}[{entry.hdu}]', meta=meta))
    return out


# (name, sniff, read) in the order tried; ASCII has no sniffer (its reader raises for anything else) and `specutils`
# is the last resort in `read_spectrum`.
_READERS = (
    ('pypeit_spec1d', _sniff_pypeit_spec1d, _read_pypeit_spec1d),
    ('pypeit_coadd', _sniff_pypeit_coadd, _read_pypeit_coadd),
    ('spextool', _sniff_spextool, _read_spextool),
    ('fits_generic', _sniff_fits_generic, _read_fits_generic),
    ('ascii', None, _read_ascii),
)
