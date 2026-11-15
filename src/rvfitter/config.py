"""
Reading, validating and writing the TOML configuration that fully describes an analysis.

`Config` is a tree of dataclasses with one section per concern: ``[input]``, ``[grouping]``, ``[velocity_frame]``,
``[templates]``, ``[methods]``, ``[regions]``, ``[combine]``, ``[calibration]``, ``[parallel]`` and ``[output]``,
plus the top-level ``seed`` and ``schema_version``.
`load` reads and validates a TOML file, `dump` writes one, `default_for` builds the regime defaults for a spectrum
and `Config.hash` is a SHA-256 of the canonical content, stored with every result.
Unset optional values (``None``) are omitted from the TOML file, which has no null.
"""
import copy
import hashlib
import json
import logging
import math
import tomllib
import types
import typing
from collections.abc import Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any, NamedTuple, Union, get_args, get_origin, get_type_hints

import tomli_w

from .constants import C_KMS
from .medium import METHODS as AIR_METHODS
from .types import Spectrum

__all__ = ['Config', 'ConfigError', 'InputConfig', 'CubeConfig', 'OBJECT_MODES', 'R_MIN_RV', 'resolution_too_low',
           'VelocityFrameConfig', 'TemplatesConfig', 'MethodsConfig',
           'GroupingConfig', 'RegionsConfig', 'CombineConfig', 'ParallelConfig', 'OutputConfig', 'METHOD_NAMES',
           'load', 'loads', 'dump', 'dumps', 'default_for', 'regime_for', 'method_advice', 'MethodAdvice',
           'setting_advice', 'SettingAdvice', 'grid_advice', 'LOCATION_FIELDS',
           'SCHEMA_VERSION']

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
WAVELENGTH_CORRECTIONS = ('sky-if-needed', 'sky', 'auto', 'telluric', 'none')
METHOD_NAMES = ('ccf', 'template_fit', 'telluric_ref', 'linecentre', 'forward_mcmc', 'lsd')  # methods.REGISTRY keys
METHOD_LABELS = {'ccf': 'CCF', 'template_fit': 'Template fit', 'linecentre': 'Line centring',  # display names
                 'forward_mcmc': 'Forward model', 'lsd': 'LSD', 'telluric_ref': 'Telluric reference'}
OUTPUT_FORMATS = ('csv', 'json', 'tsv', 'html')
MEDIA = ('air', 'vacuum')
INPUT_MEDIA = ('auto', *MEDIA)
VELOCITY_CORRECTIONS = ('auto', 'apply', 'skip')
VELOCITY_FRAMES = ('barycentric', 'heliocentric')
TEMPLATE_WINDOWS = ('features', 'full')
PLOT_STYLES = ('quick', 'paper')

# Regime boundaries in resolving power (low: R<~4500, e.g. GTC R2500, MagE, SpeX; mid: R~6000; high: R>20000)
REGIME_EDGES = (4500., 20000.)
# Band classification of the spectral coverage [Å]. These only choose between sets of default methods;
# the spectral regions themselves always come from the feature catalogue via `rvfitter.regions.select`.
BAND_OPTICAL_MAX = 11000.  # coverage ending below this is 'optical'
BAND_NIR_MIN = 9000.  # coverage starting above this (and ending above BAND_OPTICAL_MAX) is 'nir'; else 'mixed'
# Below this resolving power no RV is constrained: one resolution element, c/R, exceeds 600 km/s, so even a centroid
# good to a twentieth of it (30 km/s at high S/N) is as large as the RVs of nearby stars and brown dwarfs (tens of km/s)
R_MIN_RV = 500.
OBJECT_MODES = ('auto', 'target', 'all')  # input.objects, else a list of entry names / HDU indices


class ConfigError(ValueError):
    """The configuration file is malformed or has invalid values; the message lists every problem found"""


@dataclass
class CubeConfig:
    """IFU cube extraction: a circular aperture (`rvfitter.io.read_spectrum`)"""
    # aperture centre (x, y) [0-based pixels of the median-collapsed cube]; unset: its brightest spaxel
    position: tuple[float, float] | None = None
    radius: float | None = None  # aperture radius [spaxels]; unset: rvfitter.io.CUBE_RADIUS_DEFAULT (3)


@dataclass
class InputConfig:
    """Which spectra to read and how to interpret them"""
    paths: list[str] = field(default_factory=list)
    medium: str = field(default='auto', metadata={'choices': INPUT_MEDIA})  # medium of the input wavelengths
    medium_method: str = field(default='ciddor', metadata={'choices': tuple(AIR_METHODS)})  # air↔vacuum formula
    hdu: int | str | None = None
    resolution: float | None = None  # R override (applied to every spectrum)
    telluric_corrected: bool = False  # spectra are already telluric-corrected: relax the telluric masking
    # raw frame headers (the reduction drops the slit PA, needed for the DAR across the slit): a path or glob, or
    # a directory searched for '<frame number>*.fits' of each raw frame named in the reduced file (PypeIt FILENAME)
    raw_header: str | None = None
    raw_header_dir: str | None = None
    # which spectra of a multi-object file (PypeIt spec1d, JWST x1d, multi-extension FITS) to measure, each as its own
    # object: 'auto', 'target', 'all', or a list of entry names / HDU indices (`rvfitter.io.read_spectra`)
    objects: str | list[int | str] = 'auto'
    cube: CubeConfig = field(default_factory=CubeConfig)


@dataclass
class VelocityFrameConfig:
    """Barycentric (or heliocentric) correction"""
    correction: str = field(default='auto', metadata={'choices': VELOCITY_CORRECTIONS})
    frame: str = field(default='barycentric', metadata={'choices': VELOCITY_FRAMES})
    # overrides of the header values; allowed keys: ra, dec [deg], mjd (mid-exposure), site (an astropy site name),
    # site_lat, site_lon [deg] (used instead of ``site``)
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupingConfig:
    """Grouping of spectra into objects"""
    override: dict[str, str] | None = None  # file (path, name or stem) or header object name -> object


@dataclass
class TemplatesConfig:
    """Model-atmosphere grid and the parameter space searched"""
    grid: str | list[str] = 'bt-settl-cifist'  # one grid name, or several (searched together; their spread is a
    # systematic)
    local_dir: str | dict[str, str] | None = None  # local template directory (first grid), or {grid: directory}
    # declared medium of the grid; None = take it from the grid's metadata
    medium: str | None = field(default=None, metadata={'choices': MEDIA})
    teff_range: tuple[float, float] | None = None  # [K]
    logg_range: tuple[float, float] | None = None  # [dex, cgs]
    met_range: tuple[float, float] | None = None  # [dex]
    alpha_range: tuple[float, float] | None = None  # [dex]
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)  # extra grid axes, e.g. {co = [0.5, 1]}
    teff_prior: float | None = None  # expected Teff [K] (e.g. from the spectral type), used to select features
    logg_prior: float | None = None
    met_prior: float | None = None  # met, alpha and extra axes without a range are held at their prior/default
    alpha_prior: float | None = None


@dataclass
class MethodsConfig:
    """Which RV methods run, and their options"""
    enabled: list[str] = field(default_factory=lambda: ['template_fit', 'ccf', 'linecentre'])
    options: dict[str, dict[str, Any]] = field(default_factory=dict)  # method name → keyword options
    binary: bool = False  # run the single- versus two-template test (binary_dbic, binary_flag)


@dataclass
class RegionsConfig:
    """
    Region selection thresholds and overrides.

    ``overrides`` is keyed by method name.
    Each entry is either ``{name, kind, wmin, wmax[, centre, continuum]}`` (add or replace the region of that name)
    or ``{name, remove = true}`` (drop it); wavelengths are vacuum Å.
    """
    overrides: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    min_snr: float = 5.  # drop features whose local S/N is below this
    telluric_strictness: float = 1.  # scales the airmass-dependent telluric threshold (larger masks more)
    max_regions: int = 12  # keep at most this many, ranked by forecast information content
    continuum_fwhm: float = 5.  # continuum windows extend this many resolution elements around a feature
    # windows for ccf/template_fit at R <= 5000: 'features' (around well-modelled atomic/CO features; the
    # default, because synthetic grids reproduce molecular bands poorly) or 'full' (every non-telluric span)
    template_windows: str = field(default='features', metadata={'choices': TEMPLATE_WINDOWS})


@dataclass
class CombineConfig:
    """Combination of regions, methods and epochs"""
    tau_scale: float | None = None  # scale of the hierarchical prior on the scatter [km/s]; None = automatic
    systematic_floor: float = 0.  # added in quadrature to the adopted error [km/s]


@dataclass
class ParallelConfig:
    """Parallelism within an object"""
    n_procs: int | None = None  # None = automatic: serial, or cpu_count() - 1 when forward_mcmc runs; 1 = serial


@dataclass
class OutputConfig:
    """Where and how results are written"""
    path: str = 'results.csv'
    formats: list[str] = field(default_factory=lambda: ['csv'])
    plots: bool = True
    # 'quick': bundled Computer Modern, no LaTeX needed; 'paper': text typeset by LaTeX (usetex; needs latex+dvipng)
    plot_style: str = field(default='quick', metadata={'choices': PLOT_STYLES})
    medium: str = field(default='vacuum', metadata={'choices': MEDIA})  # medium of the wavelengths in reports and plots


@dataclass
class CalibrationConfig:
    """Wavelength-solution corrections applied to the spectra before any method runs"""
    # 'sky-if-needed': apply the OH sky zero point only when it is safe and significant; 'auto' (opt-in): also
    # apply the star - sky constant; 'sky': sky curve alone; 'telluric': the star's telluric curve; 'none': report
    # only. See docs/methods.md. What was applied is logged and recorded (wavelength_correction_applied, zp_flags).
    wavelength_correction: str = field(default='sky-if-needed', metadata={'choices': WAVELENGTH_CORRECTIONS})
    # slit-centring term when the star - sky constant is not applied: 'auto' = (W - F)/√12 from the slit width W and
    # trace FWHM F; a number [km/s]; or 'none' (unconstrained, warned)
    slit_prior: str | float = 'auto'
    # [arcsec] seeing FWHM assumed for the 'auto' slit prior when the trace FWHM is unknown (WARNING, seeing_assumed)
    seeing_fwhm: float = 0.8
    # [km/s] wavelength-calibration accuracy assumed when the file records no arc RMS (WARNING, wave_rms_assumed); wins
    # over the instrument-typical value of data/instruments.csv
    wave_rms_kms: float | None = None
    zero_point_known: bool = False  # fibre/IFU data or absolute calibration: no slit-centring term
    sky_source: str | None = None  # file whose sky spectrum shares the input's wavelength solution (e.g. a spec1d)
    sky_segment_width: float = 250.  # [Å]; Δv(λ) is interpolated between segments of the telluric-free spans
    telluric_source: str | None = None  # uncorrected file with the input's wavelength solution (star's own zero point)
    telluric_segment_width: float = 250.  # [Å]; maximum width of the telluric-band segments
    telluric_template: bool = True  # model the star with the template nearest the priors in the telluric fit
    # [min, max] vacuum Å (observatory frame) where the wavelength solution is constrained, e.g. the arc-line range
    wave_valid_range: tuple[float, float] | None = None

@dataclass
class Config:
    """The complete description of an analysis"""
    schema_version: int = SCHEMA_VERSION
    seed: int = 0
    input: InputConfig = field(default_factory=InputConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    velocity_frame: VelocityFrameConfig = field(default_factory=VelocityFrameConfig)
    templates: TemplatesConfig = field(default_factory=TemplatesConfig)
    methods: MethodsConfig = field(default_factory=MethodsConfig)
    regions: RegionsConfig = field(default_factory=RegionsConfig)
    combine: CombineConfig = field(default_factory=CombineConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    # per-object configuration patches from pipeline.refit_object, in order ({'object': ..., 'patch': {...}});
    # pipeline.run applies each only to its object (the values above are the defaults for every object)
    refits: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, drop_none: bool = True) -> dict[str, Any]:
        """Plain nested dictionary (tuples as lists); ``None`` values are dropped unless ``drop_none`` is False"""
        return _to_plain(self, drop_none)

    def hash(self) -> str:
        """
        SHA-256 (hex) of the canonical JSON dump (sorted keys) of the analysis.

        The file locations (`LOCATION_FIELDS`) and the whole ``output`` section are excluded:
        they say where data come from and where results go, not how the analysis is done (the results record
        input-file checksums separately).
        """
        plain = self.to_dict(drop_none=False)
        for section, name in LOCATION_FIELDS:
            plain[section].pop(name)
        plain.pop('output')
        plain.pop('refits', None)  # the per-object configuration (patches applied) is what is hashed
        canonical = json.dumps(plain, sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def validate(self) -> None:
        """Raise `ConfigError` listing every invalid value"""
        problems = _validate(self)
        if problems:
            raise ConfigError('Invalid configuration:\n  - ' + '\n  - '.join(problems))


# ----------------------------------------------------------------------------------------------------------------------
# dict <-> dataclass
# ----------------------------------------------------------------------------------------------------------------------

def _to_plain(obj: Any, drop_none: bool) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        out = {f.name: _to_plain(getattr(obj, f.name), drop_none) for f in fields(obj)}
        return {k: v for k, v in out.items() if v is not None} if drop_none else out
    if isinstance(obj, dict):
        out = {str(k): _to_plain(v, drop_none) for k, v in obj.items()}
        return {k: v for k, v in out.items() if v is not None} if drop_none else out
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v, drop_none) for v in obj]
    return obj


def _suggest(key: Any, choices: Sequence[str]) -> str:
    close = get_close_matches(str(key), choices, n=1)
    return f" (did you mean '{close[0]}'?)" if close else ''


def _type_name(hint: Any) -> str:
    if hint is Any:
        return 'any value'
    origin, args = get_origin(hint), get_args(hint)
    if origin in (Union, types.UnionType):
        return ' or '.join('nothing' if a is type(None) else _type_name(a) for a in args)
    if origin is list:
        return f'a list of {_type_name(args[0])}'
    if origin is tuple:
        return f'a list of {len(args)} {_type_name(args[0])}'
    if origin is dict or is_dataclass(hint):
        return 'a table'
    return {str: 'a string', int: 'an integer', float: 'a number', bool: 'true or false'}.get(hint, str(hint))


def _coerce(value: Any, hint: Any, where: str) -> Any:
    origin, args = get_origin(hint), get_args(hint)
    bad = ConfigError(f'{where}: expected {_type_name(hint)}, got {value!r}')
    if hint is Any:
        return value
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(value, arg, where)
            except ConfigError:
                continue
        raise bad
    if hint is bool:
        if isinstance(value, bool):
            return value
        raise bad
    if hint is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise bad
    if hint is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
        raise bad
    if hint is str:
        if isinstance(value, str):
            return value
        raise bad
    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise bad
        return [_coerce(v, args[0], f'{where}[{i}]') for i, v in enumerate(value)]
    if origin is tuple:
        if not isinstance(value, (list, tuple)) or len(value) != len(args):
            raise bad
        return tuple(_coerce(v, a, f'{where}[{i}]') for i, (v, a) in enumerate(zip(value, args, strict=True)))
    if origin is dict:
        if not isinstance(value, dict):
            raise bad
        return {str(k): _coerce(v, args[1], f'{where}.{k}') for k, v in value.items()}
    if is_dataclass(hint):
        return _from_dict(hint, value, where)
    raise TypeError(f'unsupported type {hint}')  # programming error


def _from_dict(cls: type, data: Any, where: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f'{where or "config"}: expected a table, got {data!r}')
    hints = get_type_hints(cls, localns=vars(typing))
    known = {f.name: f for f in fields(cls)}
    unknown = [k for k in data if k not in known]
    if unknown:
        msgs = []
        for key in unknown:
            msgs.append(f"'{key}'" + (_suggest(key, list(known)) or f' (valid keys: {", ".join(known)})'))
        section = f'[{where}]' if where else 'the top level'
        raise ConfigError(f'Unknown key in {section}: ' + '; '.join(msgs))
    kwargs = {}
    for name in known:
        if name in data:
            kwargs[name] = _coerce(data[name], hints[name], f'{where}.{name}' if where else name)
    return cls(**kwargs)


# ----------------------------------------------------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------------------------------------------------

def _choice(problems: list[str], where: str, value: Any, choices: tuple[str, ...]) -> None:
    if value not in choices:
        problems.append(f'{where} = {value!r} is not one of {", ".join(choices)}{_suggest(value, choices)}')


def _range(problems: list[str], where: str, value: tuple[float, float] | None, positive: bool = False) -> None:
    if value is None:
        return
    lo, hi = value
    if not lo < hi:
        problems.append(f'{where} = {list(value)} must be [min, max] with min < max')
    elif positive and lo <= 0:
        problems.append(f'{where} = {list(value)} must be positive')


def _validate(cfg: Config) -> list[str]:
    p: list[str] = []
    if cfg.schema_version != SCHEMA_VERSION:
        p.append(f'schema_version = {cfg.schema_version}; this rvfitter reads version {SCHEMA_VERSION}')
    if cfg.seed < 0:
        p.append(f'seed = {cfg.seed} must be non-negative')
    i = cfg.input
    _choice(p, 'input.medium', i.medium, INPUT_MEDIA)
    _choice(p, 'input.medium_method', i.medium_method, tuple(AIR_METHODS))
    if i.resolution is not None and not i.resolution > 1:
        p.append(f'input.resolution = {i.resolution} must be a resolving power R > 1')
    if isinstance(i.objects, str):
        _choice(p, 'input.objects', i.objects, OBJECT_MODES)
    elif not i.objects:
        p.append("input.objects = [] selects nothing; use 'auto', 'target', 'all' or a list of names / HDU indices")
    if i.hdu is not None and i.objects != 'auto':
        p.append(f'input.hdu = {i.hdu!r} and input.objects = {i.objects!r} both choose the spectra; set only one')
    if i.cube.position is not None and min(i.cube.position) < 0:
        p.append(f'input.cube.position = {list(i.cube.position)} must be (x, y) pixel indices >= 0')
    if i.cube.radius is not None and not i.cube.radius > 0:
        p.append(f'input.cube.radius = {i.cube.radius} must be positive [spaxels]')
    v = cfg.velocity_frame
    _choice(p, 'velocity_frame.correction', v.correction, VELOCITY_CORRECTIONS)
    _choice(p, 'velocity_frame.frame', v.frame, VELOCITY_FRAMES)
    allowed = ('ra', 'dec', 'mjd', 'site', 'site_lat', 'site_lon')
    for key, val in v.overrides.items():
        if key not in allowed:
            p.append(f'velocity_frame.overrides has unknown key {key!r}; allowed: {", ".join(allowed)}'
                     + _suggest(key, allowed))
        elif key == 'site':
            if not isinstance(val, str):
                p.append(f'velocity_frame.overrides.site = {val!r} must be a string')
        elif isinstance(val, bool) or not isinstance(val, (int, float)):
            p.append(f'velocity_frame.overrides.{key} = {val!r} must be a number')
    num = {k: v.overrides[k] for k in allowed if isinstance(v.overrides.get(k), (int, float))}
    if 'ra' in num and not 0 <= num['ra'] < 360:
        p.append(f"velocity_frame.overrides.ra = {num['ra']} must be in degrees, 0 <= ra < 360")
    if 'dec' in num and not -90 <= num['dec'] <= 90:
        p.append(f"velocity_frame.overrides.dec = {num['dec']} must be in degrees, -90 <= dec <= 90")
    if ('site_lat' in v.overrides) != ('site_lon' in v.overrides):
        p.append('velocity_frame.overrides: site_lat and site_lon must be given together')
    if 'site_lat' in num and not -90 <= num['site_lat'] <= 90:
        p.append(f"velocity_frame.overrides.site_lat = {num['site_lat']} must be in degrees, -90 to 90")
    t = cfg.templates
    if t.medium is not None:
        _choice(p, 'templates.medium', t.medium, MEDIA)
    _range(p, 'templates.teff_range', t.teff_range, positive=True)
    _range(p, 'templates.logg_range', t.logg_range)
    _range(p, 'templates.met_range', t.met_range)
    _range(p, 'templates.alpha_range', t.alpha_range)
    for axis, r_ in t.ranges.items():
        _range(p, f'templates.ranges.{axis}', r_)
    grids = [t.grid] if isinstance(t.grid, str) else list(t.grid)
    if not grids or len(set(grids)) != len(grids):
        p.append(f'templates.grid = {t.grid!r} must be a name or a list of distinct names')
    if t.teff_prior is not None and not t.teff_prior > 0:
        p.append(f'templates.teff_prior = {t.teff_prior} must be a positive temperature [K]')
    m = cfg.methods
    if not m.enabled:
        p.append('methods.enabled is empty; enable at least one of ' + ', '.join(METHOD_NAMES))
    for k, name in enumerate(m.enabled):
        _choice(p, f'methods.enabled[{k}]', name, METHOD_NAMES)
    if len(set(m.enabled)) != len(m.enabled):
        p.append('methods.enabled contains duplicates')
    for name in m.options:
        _choice(p, 'methods.options key', name, METHOD_NAMES)
    r = cfg.regions
    for method, entries in r.overrides.items():
        _choice(p, 'regions.overrides method key', method, METHOD_NAMES)
        for k, e in enumerate(entries):
            w = f'regions.overrides.{method}[{k}]'
            if not isinstance(e.get('name'), str) or not e.get('name'):
                p.append(f'{w}: needs a non-empty "name"')
                continue
            if e.get('remove') is True:
                continue
            missing = [key for key in ('kind', 'wmin', 'wmax') if key not in e]
            if missing:
                p.append(f'{w}: needs {", ".join(missing)} (or remove = true)')
                continue
            _choice(p, f'{w}.kind', e['kind'], ('line', 'band', 'ccf', 'telluric', 'continuum'))
            lo, hi = e['wmin'], e['wmax']
            if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and 0 < lo < hi):
                p.append(f'{w}: need numbers 0 < wmin < wmax, got wmin={lo!r}, wmax={hi!r}')
                continue
            c = e.get('centre')
            if c is not None and not (isinstance(c, (int, float)) and lo <= c <= hi):
                p.append(f'{w}.centre = {c!r} lies outside [{lo}, {hi}]')
            for j, win in enumerate(e.get('continuum', [])):
                if not (isinstance(win, (list, tuple)) and len(win) == 2 and 0 < win[0] < win[1]):
                    p.append(f'{w}.continuum[{j}] = {win!r} must be [min, max] with 0 < min < max')
    if r.min_snr < 0:
        p.append(f'regions.min_snr = {r.min_snr} must be >= 0')
    if not r.telluric_strictness > 0:
        p.append(f'regions.telluric_strictness = {r.telluric_strictness} must be > 0')
    if r.max_regions < 1:
        p.append(f'regions.max_regions = {r.max_regions} must be >= 1')
    if not r.continuum_fwhm > 0:
        p.append(f'regions.continuum_fwhm = {r.continuum_fwhm} must be > 0')
    _choice(p, 'regions.template_windows', r.template_windows, TEMPLATE_WINDOWS)
    _choice(p, 'calibration.wavelength_correction', cfg.calibration.wavelength_correction,
            WAVELENGTH_CORRECTIONS)
    _range(p, 'calibration.wave_valid_range', cfg.calibration.wave_valid_range, positive=True)
    sp = cfg.calibration.slit_prior
    if isinstance(sp, str):
        _choice(p, 'calibration.slit_prior', sp, ('auto', 'none'))
    elif not (isinstance(sp, (int, float)) and not isinstance(sp, bool) and math.isfinite(sp) and sp >= 0):
        p.append(f'calibration.slit_prior = {sp!r} must be "auto", "none" or a velocity >= 0 [km/s]')
    if not cfg.calibration.seeing_fwhm > 0:
        p.append(f'calibration.seeing_fwhm = {cfg.calibration.seeing_fwhm} must be > 0 [arcsec]')
    if cfg.calibration.wave_rms_kms is not None and not cfg.calibration.wave_rms_kms > 0:
        p.append(f'calibration.wave_rms_kms = {cfg.calibration.wave_rms_kms} must be > 0 [km/s]')
    if not cfg.calibration.sky_segment_width > 0:
        p.append(f'calibration.sky_segment_width = {cfg.calibration.sky_segment_width} must be > 0 [Å]')
    if not cfg.calibration.telluric_segment_width > 0:
        p.append(f'calibration.telluric_segment_width = {cfg.calibration.telluric_segment_width} must be > 0 [Å]')
    c = cfg.combine
    if c.tau_scale is not None and not c.tau_scale > 0:
        p.append(f'combine.tau_scale = {c.tau_scale} must be > 0')
    if c.systematic_floor < 0:
        p.append(f'combine.systematic_floor = {c.systematic_floor} must be >= 0')
    if cfg.parallel.n_procs is not None and cfg.parallel.n_procs < 1:
        p.append(f'parallel.n_procs = {cfg.parallel.n_procs} must be >= 1 (or omitted for automatic)')
    o = cfg.output
    if not o.formats:
        p.append('output.formats is empty')
    for k, fmt in enumerate(o.formats):
        _choice(p, f'output.formats[{k}]', fmt, OUTPUT_FORMATS)
    _choice(p, 'output.medium', o.medium, MEDIA)
    _choice(p, 'output.plot_style', o.plot_style, PLOT_STYLES)
    return p


# ----------------------------------------------------------------------------------------------------------------------
# public I/O
# ----------------------------------------------------------------------------------------------------------------------

def from_dict(data: dict[str, Any]) -> Config:
    """Build and validate a `Config` from a nested dictionary (unknown keys are errors)"""
    cfg = _from_dict(Config, data, '')
    cfg.validate()
    return cfg


def loads(text: str) -> Config:
    """Parse and validate a TOML string"""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f'Invalid TOML: {exc}') from exc
    return from_dict(data)


# every file location, (section, name) -> what it names (the GUI's Browse dialog): 'files' (a list of files, folders
# or globs), 'file', 'folder' or 'save' (a file written); relative ones in a TOML file are relative to that file
LOCATION_FIELDS = {('input', 'paths'): 'files', ('input', 'raw_header'): 'file', ('input', 'raw_header_dir'): 'folder',
                   ('templates', 'local_dir'): 'folder', ('calibration', 'sky_source'): 'file',
                   ('calibration', 'telluric_source'): 'file', ('output', 'path'): 'save'}


def resolve_locations(cfg: Config, base: str | Path) -> Config:
    """A copy of ``cfg`` with its relative file locations (`LOCATION_FIELDS`, also inside ``[[refits]]`` patches,
    ``grouping.override`` keys that contain a folder, and ``input.objects`` items ``file.fits[hdu]``) made
    absolute against ``base``; absolute ones, and override keys and objects that are a file name, stem, object name
    or HDU number, are kept."""
    cfg, base = copy.deepcopy(cfg), Path(base).resolve()

    def fix(p):
        if isinstance(p, dict):
            return {k: fix(v) for k, v in p.items()}
        if isinstance(p, list):
            return [fix(v) for v in p]
        return p if p is None or Path(p).is_absolute() else str(base / p)
    for section, name in LOCATION_FIELDS:
        setattr(getattr(cfg, section), name, fix(getattr(getattr(cfg, section), name)))
        for r in cfg.refits:
            sub = r.get('patch', {}).get(section, {})
            if name in sub:
                sub[name] = fix(sub[name])
    if cfg.grouping.override:
        cfg.grouping.override = {fix(k) if Path(k).parent != Path('.') else k: v
                                 for k, v in cfg.grouping.override.items()}
    if isinstance(cfg.input.objects, list):  # 'file.fits[hdu]' selectors follow input.paths; names and HDUs are kept
        from .io import split_source  # io imports config
        cfg.input.objects = [x if not isinstance(x, str) or (sp := split_source(x))[1] is None
                             else f'{fix(str(sp[0]))}[{sp[1]}]' for x in cfg.input.objects]
    return cfg


def load(path: str | Path) -> Config:
    """
    Read and validate a TOML configuration file; relative file locations in it are taken relative to its folder.

    Raises
    ------
    FileNotFoundError
        If the file does not exist
    ConfigError
        If the TOML is malformed, has unknown keys or wrongly typed values, or fails validation;
        the message names the file and the offending keys
    """
    path = Path(path)
    try:
        with open(path, 'rb') as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f'{path}: invalid TOML: {exc}') from exc
    try:
        return resolve_locations(from_dict(data), path.parent)
    except ConfigError as exc:
        raise ConfigError(f'{path}: {exc}') from exc


def dumps(cfg: Config) -> str:
    """Serialise to a TOML string (after validating)"""
    cfg.validate()
    return tomli_w.dumps(cfg.to_dict())


def dump(cfg: Config, path: str | Path) -> None:
    """Validate and write the configuration to ``path`` as TOML"""
    Path(path).write_text(dumps(cfg), encoding='utf-8')


# ----------------------------------------------------------------------------------------------------------------------
# regime defaults
# ----------------------------------------------------------------------------------------------------------------------

def regime_for(resolution: float | None, wave_range: tuple[float, float]) -> tuple[str, str]:
    """
    Classify a spectrum for the default settings.

    Parameters
    ----------
    resolution
        Resolving power R (None = unknown, treated as low resolution)
    wave_range
        (minimum, maximum) vacuum wavelength [Å]

    Returns
    -------
    regime, band
        regime is 'low' (R below about 4500), 'mid' (R~6000) or 'high' (R above 20000);
        band is 'optical' (ends below 11000 Å), 'nir' (starts above 9000 Å) or 'mixed'
    """
    if resolution is None or resolution < REGIME_EDGES[0]:
        regime = 'low'
    elif resolution <= REGIME_EDGES[1]:
        regime = 'mid'
    else:
        regime = 'high'
    wmin, wmax = wave_range
    if wmax <= BAND_OPTICAL_MAX:
        band = 'optical'
    elif wmin >= BAND_NIR_MIN:
        band = 'nir'
    else:
        band = 'mixed'
    return regime, band


class MethodAdvice(NamedTuple):
    """Whether a method is on by default in a regime, why, and its default options there"""
    recommended: bool
    reason: str
    options: dict[str, Any]


def resolution_too_low(resolution: float | None) -> str | None:
    """Why a spectrum of this R cannot constrain an RV (R below `R_MIN_RV`), else None (also for an unknown R)"""
    if resolution is None or resolution >= R_MIN_RV:
        return None
    return (f'R {resolution:.0f} is below {R_MIN_RV:.0f}: one resolution element is {C_KMS / resolution:.0f} km/s, so '
            'the spectrum cannot constrain an RV (any value measured is flagged resolution_too_low)')


def method_advice(regime: str, resolution: float | None = None) -> dict[str, MethodAdvice]:
    """
    Every method's `MethodAdvice` in ``regime`` ('low', 'mid' or 'high', from `regime_for`); the recommended methods,
    in this order, are the defaults of `default_for`. The advice does not depend on the band. With a ``resolution``
    below `R_MIN_RV`, every reason starts with `resolution_too_low`'s statement (the recommendations are unchanged).
    """
    if regime not in ('low', 'mid', 'high'):
        raise ValueError(f"regime must be 'low', 'mid' or 'high', not {regime!r}")
    low, high = regime == 'low', regime == 'high'
    telluric_model = 'Needs the telluric model (rvfitter templates telluric --build)'
    advice = {
        'template_fit': (True, 'Recommended: the most precise RV at low R, a fit of a broadened model template (vsini '
                               'held at 0, as it is unresolved).' if low else
                         'Recommended: a fit of a broadened model template, with vsini free, as it is resolved above '
                         'R 4500.'),
        'ccf': (False, 'Optional: above R 20000 the template fit (vsini free) and the telluric reference cover it; add '
                       'it as a cross-check.') if high else
               (True, 'Recommended: cross-correlation with the model template, one RV per window; a check of the '
                      'template fit.'),
        'telluric_ref': (True, 'Recommended: the star against the telluric lines of the same band, free of wavelength-'
                               f'calibration and slit-centring offsets. {telluric_model} and spectra that are not '
                               'telluric-corrected.') if high else
                        (False, f'Not recommended below R 20000: it needs resolved telluric lines. {telluric_model}.'),
        'linecentre': (True, 'Recommended: the model-free check, line centres against laboratory (NIST) wavelengths; '
                             'it needs no templates.') if low else
                      (False, 'Optional: model-free and needs no templates, but above R 4500 the template methods are '
                              'more precise; add it as a template-independent check.'),
        'forward_mcmc': (False, 'Optional: slow (MCMC, on all but one CPU core); worth it at high R, especially for '
                                'several epochs fitted together.' if high else
                         'Not recommended below R 20000: slow (MCMC), and at low R its vsini and Teff are not '
                         'physical; it is for high R and several epochs fitted together.'),
        'lsd': (False, 'Optional: least-squares deconvolution needs the many lines of a high-R spectrum; it gives the '
                       'RV and vsini from the mean line profile.' if high else
                'Not recommended below R 20000: the mean line profile needs many resolved lines.'),
    }
    too_low = resolution_too_low(resolution)
    return {m: MethodAdvice(ok, f'{too_low}. {why}' if too_low else why,
                            {'fit_vsini': True} if m == 'template_fit' and not low else {})
            for m, (ok, why) in advice.items()}


def default_for(spectrum: Spectrum) -> Config:
    """
    Regime defaults for a spectrum: the methods that `setting_advice` (from `method_advice`) recommends at its
    resolving power, i.e.
    template_fit, ccf and linecentre at low R (R < 4500); template_fit (free vsini) and ccf at R 4500-20000;
    template_fit (free vsini) and telluric_ref above.

    The spectrum's measured medium and resolution are written into the config, so that re-running it does not
    depend on header sniffing.
    An unknown resolution is treated as low (with a warning).
    """
    cfg = Config()
    R = spectrum.resolution
    regime, band = regime_for(R, (spectrum.wmin, spectrum.wmax))
    if R is None:
        log.warning('Resolution of %s is unknown; using the low-resolution defaults. '
                    'Set input.resolution to change this.', spectrum.source or 'the spectrum')
    log.info('Default settings for the %s-resolution regime, %s band', regime, band)
    cfg.input.resolution = None if R is None else float(R)
    cfg.input.medium = spectrum.meta['medium_in']
    advice = setting_advice(spectrum, cfg)  # one rule set: the methods the Advanced dialog recommends
    cfg.methods.enabled, cfg.methods.options = advice['methods.enabled'].value, advice['methods.options'].value
    cfg.validate()
    return cfg


# ----------------------------------------------------------------------------------------------------------------------
# per-setting advice (the Advanced dialog; default_for takes its methods from it)
# ----------------------------------------------------------------------------------------------------------------------

SPACE_SITES = ('JWST',)  # meta['site'] of observatories without telluric absorption or sky lines
# what each known grid models (from its KNOWN_GRIDS reference), for the grid advice
GRID_PHYSICS = {
    'bt-settl-cifist': 'BT-Settl CIFIST (Allard et al. 2012): condensate settling (clouds) with the CIFIST solar '
                       'abundances, the reference grid for M and L dwarfs, validated on the GTC/OSIRIS benchmark; '
                       'solar metallicity only',
    'bt-settl-agss2009': 'BT-Settl AGSS2009 (Allard et al. 2012): [M/H] and [α/M] vary, and it extends to hot stars',
    'sonora-diamondback': 'Sonora Diamondback (Morley et al. 2024): silicate and iron clouds (fsed), which shape the '
                          'spectra of L and L/T-transition dwarfs',
    'sonora-elfowl': 'Sonora Elf Owl (Mukherjee et al. 2024): disequilibrium chemistry (Kzz) and C/O, cloud-free, '
                     'as T and Y dwarfs need',
    'sonora-bobcat': 'Sonora Bobcat (Marley et al. 2021): cloud-free, equilibrium chemistry, C/O; T and Y dwarfs',
    'sand': 'SAND (Alvarado et al. 2024): metal-poor, α-enhanced cool dwarfs',
    'lowz': 'LOWZ (Meisner et al. 2021): metal-poor, cloud-free cool atmospheres with disequilibrium chemistry',
}
# grid preference by Teff band [K] (the first lower edge the Teff reaches); `grid_advice` keeps the grids covering it
GRID_ORDER = ((4000., ('bt-settl-agss2009', 'bt-settl-cifist')),
              (2000., ('bt-settl-cifist', 'bt-settl-agss2009', 'sonora-diamondback', 'sand')),
              (1300., ('bt-settl-cifist', 'sonora-diamondback', 'sonora-elfowl')),
              (0., ('sonora-elfowl', 'sonora-bobcat', 'sonora-diamondback')))
METAL_POOR = -0.5  # [dex]; a templates.met_prior at or below this ranks the metal-poor grids first
METAL_POOR_GRIDS = ('sand', 'lowz', 'bt-settl-agss2009')
REGIME_GRID = {'low': 'bt-settl-cifist', 'mid': 'bt-settl-cifist', 'high': 'bt-settl-agss2009'}  # no Teff known
REGIME_TARGETS = {'low': 'M and L dwarfs', 'mid': 'M and L dwarfs', 'high': 'FGK and M stars'}


class SettingAdvice(NamedTuple):
    """A recommended value of one setting (None: no value to apply; the reason says what to do) and why"""
    value: Any
    reason: str


def grid_advice(teff: float, met: float | None = None) -> list[tuple[str, str]]:
    """
    The known grids whose documented coverage contains ``teff`` [K], best first, each with what it models: by Teff
    band (`GRID_ORDER`), the metal-poor grids first when ``met`` [dex] is at most `METAL_POOR`. Empty if none covers it.
    """
    from .templates import KNOWN_GRIDS  # templates is heavy; config is imported everywhere
    order = next(names for edge, names in GRID_ORDER if teff >= edge)
    if met is not None and met <= METAL_POOR:
        order = (*METAL_POOR_GRIDS, *order)
    return [(g, GRID_PHYSICS[g]) for g in dict.fromkeys(order)
            if KNOWN_GRIDS[g].coverage['teff'][0] <= teff <= KNOWN_GRIDS[g].coverage['teff'][1]]


def _template_advice(cfg: Config, regime: str, teff: float | None, source: str) -> dict[str, SettingAdvice]:
    from .methods._common import TEMPLATE_DEFAULTS
    from .templates import KNOWN_GRIDS
    t = cfg.templates
    if teff is None:
        g = REGIME_GRID[regime]
        return {'templates.grid': SettingAdvice(g, f'No Teff known (no Teff prior, no run yet): at {regime} R the '
                                                   f'usual targets are {REGIME_TARGETS[regime]}, so {GRID_PHYSICS[g]}. '
                                                   'Set Templates > Teff prior for a better choice.'),
                'templates.teff_range': SettingAdvice(None, 'No Teff known: search the whole grid.'),
                'templates.logg_range': SettingAdvice(None, 'No Teff known: search the whole grid.')}
    ranked = grid_advice(teff, t.met_prior)
    if not ranked:
        none = SettingAdvice(None, f'No known grid covers {teff:.0f} K ({source}): use a local grid that does '
                                   '(Templates > Local folder).')
        return dict.fromkeys(('templates.grid', 'templates.teff_range', 'templates.logg_range'), none)
    best, why = ranked[0]
    others = ', '.join(g for g, _ in ranked[1:])
    lo, hi = KNOWN_GRIDS[best].coverage['teff']
    glo, ghi = KNOWN_GRIDS[best].coverage['logg']
    half = TEMPLATE_DEFAULTS['teff_window']
    return {
        'templates.grid': SettingAdvice(best, f'From {source} ({teff:.0f} K): {why}.' +
                                        f' Also covering it: {others}.' * bool(others)),
        'templates.teff_range': SettingAdvice((max(lo, teff - half), min(hi, teff + half)),
                                              f'±{half:.0f} K around {source} ({teff:.0f} K), within {best}; widen it '
                                              'if the fit reaches an edge.'),
        'templates.logg_range': SettingAdvice((max(glo, 4.5), min(ghi, 5.5)),
                                              f'Below 4000 K ({source}): field dwarfs and brown dwarfs; lower the '
                                              'minimum for young, low-gravity objects.') if teff < 4000. else
        SettingAdvice(None, f'At {teff:.0f} K ({source}) giants are possible: search the whole grid.'),
    }


def setting_advice(spectrum: Spectrum, cfg: Config, fit_teff: float | None = None) -> dict[str, SettingAdvice]:
    """
    Advice on the data-dependent settings for ``spectrum`` (its R, band, medium, site, sky and velocity frame) under
    ``cfg``, by dotted config path (always the same paths); ``fit_teff`` [K] is the best-fit Teff of the last run of
    its object. The methods and their options are `method_advice`'s, and `default_for` takes them from here. The grid
    follows the Teff (the Teff prior, else ``fit_teff``, else the regime), ranked by `grid_advice`. Nothing is changed.
    """
    from .features import features
    from .regions import SelectOptions  # regions imports io, which imports config
    meta, R = spectrum.meta, spectrum.resolution
    regime, band = regime_for(R, (spectrum.wmin, spectrum.wmax))
    space = meta.get('site') in SPACE_SITES
    r_src = meta.get('resolution_source', 'header')
    too_low = resolution_too_low(R)
    r_why = ('Unknown R: the low-R defaults are used; type the instrument\'s R.' if R is None else
             f'R {R:.0f} is outside the plausible range: type the instrument\'s R.'
             if meta.get('resolution_implausible') else
             f'R {R:.0f} was estimated from the pixel spacing (2.5 pixels per resolution element): type the '
             'instrument\'s R if you know it.' if r_src == 'estimated' else
             f'R {R:.0f} from the {r_src}: leave it empty, so that each spectrum keeps its own.')
    out = {'input.resolution': SettingAdvice(None, f'{too_low}. {r_why}' if too_low else r_why)}
    medium, m_src = meta.get('medium_in'), str(meta.get('medium_source', ''))
    out['input.medium'] = SettingAdvice(None, f'The header gives {medium} ({m_src.removeprefix("header:")}): auto '
                                              'reads it.') if m_src.startswith('header') else \
        SettingAdvice(medium, f'No header keyword gives the medium; {medium} is the reader\'s assumption '
                              f'({m_src or "none"}): change it if you know otherwise.')
    tellcorr = 'tellcorr' in Path(spectrum.source.split('[')[0]).name.lower()
    out['input.telluric_corrected'] = (
        SettingAdvice(True, 'Space data: there is no telluric absorption.') if space else
        SettingAdvice(True, 'A telluric-corrected (_tellcorr) product.') if tellcorr else
        SettingAdvice(None, 'Ground-based: tick it only if the reduction removed the telluric absorption.'))
    frame, frame_src, note = meta.get('frame_in'), meta.get('frame_source'), meta.get('frame_note')
    if frame:
        out['velocity_frame.correction'] = SettingAdvice('skip', f'Already {frame} ({frame_src}); auto also skips it.')
        out['velocity_frame.frame'] = SettingAdvice(frame, f'The frame of the wavelengths ({frame_src}).')
    else:
        out['velocity_frame.correction'] = SettingAdvice(None, note) if note else \
            SettingAdvice('auto', 'Computed from the header\'s coordinates, time and site.')
        out['velocity_frame.frame'] = SettingAdvice(None, 'The frame the RVs are reported in: barycentric is usual.')
    if spectrum.sky is not None or cfg.calibration.sky_source:
        where = 'the file\'s sky spectrum' if spectrum.sky is not None else 'Sky source'
        out['calibration.wavelength_correction'] = SettingAdvice(
            'sky-if-needed', f'OH sky lines are available ({where}): their zero point is applied only when it is safe '
                             'and significant.')
    else:
        out['calibration.wavelength_correction'] = SettingAdvice(
            'none', 'No sky spectrum, and space data have no sky lines.' if space else
            'No sky spectrum (none in the file, no Sky source): nothing to measure the zero point against; on a '
            'spectrum that is not telluric-corrected, "telluric" uses the star\'s own telluric lines.')
    out['calibration.zero_point_known'] = SettingAdvice(True, f'{meta["no_slit"]}: no slit, so no slit-centring '
                                                              'term.') if meta.get('no_slit') else \
        SettingAdvice(None, 'Slit data (or unknown): keep the slit-centring term; tick it for fibre data.')
    teff, source = (cfg.templates.teff_prior, 'the Teff prior') if cfg.templates.teff_prior is not None else \
        (fit_teff, 'the best-fit Teff of the last run') if fit_teff is not None else (None, '')
    r_max = SelectOptions.template_windows_r_max
    feature_teff = max(f.teff_max for f in features() if f.model_fidelity == 'good')  # the catalogue's limit
    out['regions.template_windows'] = (
        SettingAdvice(None, f'Not used above R {r_max:.0f}.') if R is not None and R > r_max else
        SettingAdvice('full', f'{teff:.0f} K ({source}) is above the {feature_teff:.0f} K limit of the feature '
                              'catalogue: there are no feature windows at this Teff.')
        if teff is not None and teff > feature_teff else
        SettingAdvice('features', f'At R ≤ {r_max:.0f} the synthetic grids reproduce the molecular bands of late-M '
                                  'and L dwarfs poorly, so the CCF and template fit use windows around well-modelled '
                                  'atomic and CO features.'))
    advice = method_advice(regime, R)
    rec = [m for m, a in advice.items() if a.recommended]
    out['methods.enabled'] = SettingAdvice(rec, f'{too_low}. ' * bool(too_low) + f'For {regime} R ({band}): '
                                           f'{", ".join(METHOD_LABELS[m] for m in rec)}; the Setup tab says why '
                                           'for each method.')
    out['methods.options'] = SettingAdvice({m: advice[m].options for m in rec if advice[m].options},
                                           f'The default options of the recommended methods at {regime} R.')
    out['parallel.n_procs'] = SettingAdvice(None, 'auto: all but one core, as the forward model runs.'
                                            if 'forward_mcmc' in cfg.methods.enabled else
                                            'auto: serial, the fastest without the forward model.')
    out.update(_template_advice(cfg, regime, teff, source))
    return out
