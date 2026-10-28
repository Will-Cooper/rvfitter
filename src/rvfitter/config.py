"""
Reading, validating and writing the TOML configuration that fully describes an analysis.

The configuration is a tree of dataclasses (`Config`) with one section per concern:

====================  ===========================================================================================
``[input]``            paths, medium (and air→vacuum formula), HDU, resolution override, telluric-corrected flag
``[grouping]``         optional file -> object overrides
``[velocity_frame]``   barycentric correction mode, target frame and overrides of the header values
``[templates]``        grid name or local directory, its medium, Teff/log g/[M/H] ranges and priors
``[methods]``          enabled methods and per-method options
``[regions]``          region overrides and selection thresholds
``[combine]``          hierarchical prior scale and systematic floor
``[parallel]``         number of processes
``[output]``           path, formats, plots and output medium
====================  ===========================================================================================

plus the top-level ``seed`` and ``schema_version``.
`load` reads and validates a TOML file, `dump` writes one, `default_for` builds the regime defaults for a spectrum
(by resolving power and band) and `Config.hash` is a SHA-256 of the canonical content, stored with every result.
Unset optional values (``None``) are omitted from the TOML file, which has no null.
"""
import hashlib
import json
import logging
import math
import tomllib
import types
import typing
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

import tomli_w

from .types import Spectrum

__all__ = ['Config', 'ConfigError', 'InputConfig', 'VelocityFrameConfig', 'TemplatesConfig', 'MethodsConfig',
           'GroupingConfig', 'RegionsConfig', 'CombineConfig', 'ParallelConfig', 'OutputConfig', 'METHOD_NAMES',
           'load', 'loads', 'dump', 'dumps', 'default_for', 'regime_for', 'SCHEMA_VERSION']

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
METHOD_NAMES = ('linecentre', 'ccf', 'template_fit', 'forward_mcmc', 'lsd', 'telluric_ref')
OUTPUT_FORMATS = ('csv', 'json', 'tsv', 'html')

# Regime boundaries in resolving power (low: R<~4500, e.g. GTC R2500, MagE, SpeX; mid: R~6000; high: R>20000)
REGIME_EDGES = (4500., 20000.)
# Band classification of the spectral coverage [Å]. These only choose between sets of default methods;
# the spectral regions themselves always come from the feature catalogue via `rvfitter.regions.select`.
BAND_OPTICAL_MAX = 11000.  # coverage ending below this is 'optical'
BAND_NIR_MIN = 9000.  # coverage starting above this (and ending above BAND_OPTICAL_MAX) is 'nir'; else 'mixed'


class ConfigError(ValueError):
    """The configuration file is malformed or has invalid values; the message lists every problem found"""


@dataclass
class InputConfig:
    """Which spectra to read and how to interpret them"""
    paths: list[str] = field(default_factory=list)
    medium: str = 'auto'  # 'auto' | 'air' | 'vacuum': medium of the input wavelengths
    medium_method: str = 'ciddor'  # air↔vacuum formula: 'ciddor' | 'edlen' | 'morton'
    hdu: int | str | None = None
    resolution: float | None = None  # R override (applied to every spectrum)
    telluric_corrected: bool = False  # spectra are already telluric-corrected: relax the telluric masking


@dataclass
class VelocityFrameConfig:
    """Barycentric (or heliocentric) correction"""
    correction: str = 'auto'  # 'auto' | 'apply' | 'skip'
    frame: str = 'barycentric'  # 'barycentric' | 'heliocentric'
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
    medium: str | None = None  # declared medium of the grid; None = take it from the grid's metadata
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
    # windows for ccf/template_fit/forward at R <= 5000: 'features' (around well-modelled atomic/CO features; the
    # default, because synthetic grids reproduce molecular bands poorly) or 'full' (every non-telluric span)
    template_windows: str = 'features'


@dataclass
class CombineConfig:
    """Combination of regions, methods and epochs"""
    tau_scale: float | None = None  # scale of the hierarchical prior on the scatter [km/s]; None = automatic
    systematic_floor: float = 0.  # added in quadrature to the adopted error [km/s]


@dataclass
class ParallelConfig:
    """Parallelism within an object"""
    n_procs: int | None = None  # None = os.cpu_count() - 1 (at least 1); 1 = serial


@dataclass
class OutputConfig:
    """Where and how results are written"""
    path: str = 'results.csv'
    formats: list[str] = field(default_factory=lambda: ['csv'])
    plots: bool = True
    medium: str = 'vacuum'  # medium of the wavelengths in reports and plots


@dataclass
class CalibrationConfig:
    """Wavelength-solution corrections applied to the spectra before any method runs"""
    # 'none' (default until validated): measure and report the zero point (zp_sky_* columns, counted in the
    # zero-point systematic) but do not apply it; 'auto': sky shape + telluric constant when available, else
    # whichever exists; 'sky' or 'telluric' force one. What was applied is always logged and recorded.
    wavelength_correction: str = 'none'
    sky_source: str | None = None  # a file whose sky spectrum shares the input's wavelength solution (e.g. a spec1d)
    sky_segment_width: float = 250.  # [Å]; segments of the telluric-free spans; Δv(λ) is interpolated between them
    # a non-telluric-corrected file sharing the input's wavelength solution (e.g. the PypeIt spec1d of a corrected
    # coadd), whose telluric lines give the star's own zero point; the object best matching the input is used
    telluric_source: str | None = None
    telluric_segment_width: float = 250.  # [Å]; maximum width of the telluric-band segments
    telluric_template: bool = True  # model the star with the template nearest the priors in the telluric fit


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

        ``input.paths`` and the whole ``output`` section are excluded: they say where data come from and where
        results go, not how the analysis is done (the results record input-file checksums separately).
        """
        plain = self.to_dict(drop_none=False)
        plain['input'].pop('paths')
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
    if origin is dict:
        return 'a table'
    if is_dataclass(hint):
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
            close = get_close_matches(key, list(known), n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else f' (valid keys: {", ".join(known)})'
            msgs.append(f"'{key}'{hint}")
        section = f'[{where}]' if where else 'the top level'
        raise ConfigError(f'Unknown key in {section}: ' + '; '.join(msgs))
    kwargs = {}
    for name, f in known.items():
        if name not in data:
            if f.default is MISSING and f.default_factory is MISSING:
                raise ConfigError(f'{where or "config"}: missing required key {name!r}')
            continue
        kwargs[name] = _coerce(data[name], hints[name], f'{where}.{name}' if where else name)
    return cls(**kwargs)


# ----------------------------------------------------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------------------------------------------------

def _choice(problems: list[str], where: str, value: Any, choices: tuple[str, ...]) -> None:
    if value not in choices:
        close = get_close_matches(str(value), choices, n=1)
        hint = f" (did you mean '{close[0]}'?)" if close else ''
        problems.append(f'{where} = {value!r} is not one of {", ".join(choices)}{hint}')


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
    _choice(p, 'input.medium', i.medium, ('auto', 'air', 'vacuum'))
    _choice(p, 'input.medium_method', i.medium_method, ('ciddor', 'edlen', 'morton'))
    if i.resolution is not None and not i.resolution > 1:
        p.append(f'input.resolution = {i.resolution} must be a resolving power R > 1')
    v = cfg.velocity_frame
    _choice(p, 'velocity_frame.correction', v.correction, ('auto', 'apply', 'skip'))
    _choice(p, 'velocity_frame.frame', v.frame, ('barycentric', 'heliocentric'))
    allowed = ('ra', 'dec', 'mjd', 'site', 'site_lat', 'site_lon')
    for key, val in v.overrides.items():
        if key not in allowed:
            close = get_close_matches(key, allowed, n=1)
            hint = f" (did you mean '{close[0]}'?)" if close else ''
            p.append(f'velocity_frame.overrides has unknown key {key!r}; allowed: {", ".join(allowed)}{hint}')
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
        _choice(p, 'templates.medium', t.medium, ('air', 'vacuum'))
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
    _choice(p, 'regions.template_windows', r.template_windows, ('features', 'full'))
    _choice(p, 'calibration.wavelength_correction', cfg.calibration.wavelength_correction,
            ('auto', 'sky', 'telluric', 'none'))
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
    _choice(p, 'output.medium', o.medium, ('vacuum', 'air'))
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


def load(path: str | Path) -> Config:
    """
    Read and validate a TOML configuration file.

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
        return from_dict(data)
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


def default_for(spectrum: Spectrum) -> Config:
    """
    Regime defaults for a spectrum, by resolving power and band (the plan's "default method per regime" table).

    ========================  ==========================================================  ==========================
    Regime                    Methods                                                     Other settings
    ========================  ==========================================================  ==========================
    low R, optical            template_fit, ccf, linecentre (secondary)                   O2-A telluric mask
    low R, NIR                template_fit, ccf                                           telluric model
    low R, optical + NIR      template_fit, ccf, linecentre                               telluric model
    R~6000 (4500 to 20000)    forward_mcmc (free RV and vsini), template_fit
    R>20000                   forward_mcmc (telluric and wavelength-offset terms),
                              telluric_ref                                                per-order combination
    ========================  ==========================================================  ==========================

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
    medium = spectrum.meta.get('medium_in')
    if medium in ('air', 'vacuum'):
        cfg.input.medium = medium
    if regime == 'low':
        if band == 'optical':
            cfg.methods.enabled = ['template_fit', 'ccf', 'linecentre']
        elif band == 'nir':
            cfg.methods.enabled = ['template_fit', 'ccf']
            cfg.methods.options = {'template_fit': {'telluric_model': True}}
        else:
            cfg.methods.enabled = ['template_fit', 'ccf', 'linecentre']
            cfg.methods.options = {'template_fit': {'telluric_model': True}}
    elif regime == 'mid':
        cfg.methods.enabled = ['forward_mcmc', 'template_fit']
        cfg.methods.options = {'forward_mcmc': {'fit_vsini': True}}
    else:
        cfg.methods.enabled = ['forward_mcmc', 'telluric_ref']
        cfg.methods.options = {'forward_mcmc': {'fit_vsini': True, 'telluric_model': True, 'fit_wave_offset': True,
                                                'per_order': True}}
    cfg.validate()
    return cfg
