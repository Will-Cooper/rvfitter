"""
Template grids (BT-Settl, SAND, Sonora, LOWZ, user-supplied): metadata, medium check, multilinear interpolation
over any number of parameter axes, broadening cache and download from the SVO theory server.

A grid is a local directory of two-column (wavelength [Å], flux) text files, plus a metadata file
`rvfitter_grid.toml` declaring:

- `name`: the grid name;
- `medium`: "air", "vacuum" or "auto" (measured from the alkali line minima on first use);
- `flux_units`: free text;
- `pattern`: a regular expression on the file name with named groups for the axes: `teff`, `logg` and
  optionally `met`, `alpha` and any `extra_axes` (e.g. `co`, `kzz`, `fsed`);
- `teff_scale`: the factor turning the `teff` group into kelvin (100 for Lyon `lte020.0` names);
- `met_default`, `alpha_default`, `extra_defaults`: values of axes the pattern does not encode;
- `resolution_native`: the model's resolving power, if finite;
- `coverage`: the documented extent of the full grid per axis (used for guidance when a fit hits an edge).

The parameter axes are always teff, logg, met and alpha, plus the grid's `extra_axes`. Axes other than teff and
log g are held at their prior (or default) unless a range is configured, so a solar-only grid behaves exactly
as before. Directories without a metadata file can be opened with a known grid name (`KNOWN_GRIDS`).
Wavelengths are converted to vacuum on reading, with `rvfitter.medium`.
"""
import functools
import hashlib
import json
import logging
import re
import tomllib
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import broadening
from .constants import C_KMS
from .medium import to_vacuum, vac_to_air
from .types import Spectrum

__all__ = ['GridMeta', 'TemplateGrid', 'KNOWN_GRIDS', 'STANDARD_AXES', 'AXIS_SCALE', 'medium_check_lines',
           'MediumMismatchError', 'METADATA_FILE', 'read_two_column', 'write_metadata', 'verify_medium',
           'measure_medium', 'download', 'available', 'cache_dir', 'cli_main', 'SVO_SSAP_URL', 'GENERIC_PATTERN']

log = logging.getLogger(__name__)

METADATA_FILE = 'rvfitter_grid.toml'
_USER_AGENT = 'rvfitter (github.com/Will-Cooper/rvfitter)'
SVO_SSAP_URL = 'http://svo2.cab.inta-csic.es/theory/newov2/ssap.php'
_LYON_PATTERN = r'^lte(?P<teff>\d+(?:\.\d+)?)-(?P<logg>\d+(?:\.\d+)?)(?P<met>[-+]\d+(?:\.\d+)?)?.*\.(?:txt|dat)$'
#: file names written by `download`: t2000_g5.0_m-0.5_a+0.2[_co0.5][_kzz7].txt (teff in K)
GENERIC_PATTERN = (r'^t(?P<teff>[\d.]+)_g(?P<logg>[-\d.]+)_m(?P<met>[-+\d.]+)_a(?P<alpha>[-+\d.]+)'
                   r'(?:_co(?P<co>[\d.]+))?(?:_kzz(?P<kzz>[-\d.]+))?(?:_fsed(?P<fsed>[\d.]+))?\.txt$')
STANDARD_AXES = ('teff', 'logg', 'met', 'alpha')
#: distance scales for the nearest grid point and the guidance (100 K ≡ 0.5 dex in log g ≡ 0.5 dex in [M/H] ...)
AXIS_SCALE = {'teff': 100., 'logg': 0.5, 'met': 0.5, 'alpha': 0.2, 'co': 0.2, 'kzz': 2., 'fsed': 2.}

_MEDIUM_CHECK_SPECIES = ('Na I', 'K I', 'Rb I', 'Cs I')
_MEDIUM_CHECK_MIN_AKI = 2.5e7  # s^-1: only the strongest alkali transitions, which have deep, isolated cores


@functools.lru_cache(maxsize=1)
def medium_check_lines() -> tuple[tuple[str, float], ...]:
    """
    The strong alkali lines used to identify the wavelength medium of a model

    They are read from the NIST ASD reference `data/nist_lines.csv` (vacuum wavelengths): the Na I, K I, Rb I
    and Cs I transitions with A_ki ≥ 2.5e7 s⁻¹ that have an air wavelength.

    Returns
    -------
    lines
        (species, vacuum wavelength [Å]) pairs, sorted by wavelength
    """
    from importlib import resources
    with resources.files('rvfitter').joinpath('data', 'nist_lines.csv').open() as fh:
        df = pd.read_csv(fh)
    aki = pd.to_numeric(df['aki'], errors='coerce')
    sel = df['species'].isin(_MEDIUM_CHECK_SPECIES) & (aki >= _MEDIUM_CHECK_MIN_AKI) & df['wave_air'].notna()
    df = df[sel].sort_values('wave_vac')
    return tuple((str(sp), float(w)) for sp, w in zip(df['species'], df['wave_vac'], strict=True))


class MediumMismatchError(ValueError):
    """The declared wavelength medium of a grid disagrees with the measured one"""


@dataclass(frozen=True)
class GridMeta:
    """The metadata of a template grid (see the module docstring)"""
    name: str
    medium: str
    flux_units: str = 'erg/s/cm2/A'
    pattern: str = _LYON_PATTERN
    teff_scale: float = 100.
    met_default: float = 0.
    resolution_native: float | None = None
    svo_model: str | None = None
    alpha_default: float = 0.
    extra_axes: tuple[str, ...] = ()
    extra_defaults: Mapping[str, float] = field(default_factory=dict)
    coverage: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    verified: bool = True  # False: the SVO model id, medium or coverage still need a live check
    reference: str = ''
    upstream: str = ''  # where the grid is published (DOI or URL), for manual downloads

    def __post_init__(self):
        if self.medium not in ('air', 'vacuum', 'auto'):
            raise ValueError(f"grid medium must be 'air', 'vacuum' or 'auto', not {self.medium!r}")
        re.compile(self.pattern)
        object.__setattr__(self, 'extra_axes', tuple(self.extra_axes))
        object.__setattr__(self, 'extra_defaults', dict(self.extra_defaults))
        object.__setattr__(self, 'coverage', {k: tuple(v) for k, v in dict(self.coverage).items()})
        bad = set(self.extra_axes) & set(STANDARD_AXES)
        if bad:
            raise ValueError(f'extra axes {sorted(bad)} duplicate the standard axes')

    @property
    def axes(self) -> tuple[str, ...]:
        """All parameter axes: teff, logg, met, alpha, then the extra axes"""
        return STANDARD_AXES + self.extra_axes

    def defaults(self) -> dict[str, float]:
        """Values of the axes that a file name does not encode"""
        return {'met': self.met_default, 'alpha': self.alpha_default, **self.extra_defaults}

    def parse(self, filename: str) -> tuple[float, ...] | None:
        """Parameter values (in `axes` order) from a file name, or None if it does not match"""
        m = re.match(self.pattern, filename)
        if m is None:
            return None
        groups = m.groupdict()
        defaults = self.defaults()
        out = []
        for axis in self.axes:
            val = groups.get(axis)
            if val is None:
                if axis not in defaults:
                    raise ValueError(f'grid {self.name}: no value or default for axis {axis!r} in {filename}')
                out.append(float(defaults[axis]))
            else:
                out.append(float(val) * (self.teff_scale if axis == 'teff' else 1.))
        return tuple(out)


# Known grids. `coverage` is the documented extent of the full grid (from the model papers and the SVO pages);
# it is only used to advise which grid to download when a fit reaches the edge of the current one.
# verified=False: the SVO model id, the medium and/or the coverage have not yet been checked against a live
# SVO query (SVO was unreachable during development). `rvfitter templates list --available` and the
# network-marked test check the ids; the medium is always measured on download ('auto' grids) or verified.
KNOWN_GRIDS: dict[str, GridMeta] = {
    # Lyon BT-Settl-CIFIST: air above 2000 Å (verified on the local grid: alkali minima within 0.03 Å of NIST air)
    'bt-settl-cifist': GridMeta('bt-settl-cifist', 'air', svo_model='bt-settl-cifist',
                                coverage={'teff': (1200., 7000.), 'logg': (2.5, 5.5), 'met': (0., 0.),
                                          'alpha': (0., 0.)},
                                reference='Allard et al. 2012; Caffau et al. 2011 (CIFIST solar abundances)',
                                upstream='https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011_2015/'),
    # BT-Settl with Asplund et al. (2009) abundances: [M/H] and [alpha/M] vary
    'bt-settl-agss2009': GridMeta('bt-settl-agss2009', 'auto', svo_model='bt-settl-agss',
                                  coverage={'teff': (400., 70000.), 'logg': (-0.5, 6.), 'met': (-4., 0.5),
                                            'alpha': (0., 0.4)}, verified=False,
                                  reference='Allard et al. 2012 (BT-Settl, AGSS2009 abundances)',
                                  upstream='https://phoenix.ens-lyon.fr/Grids/BT-Settl/AGSS2009/'),
    # SAND: metal-poor, alpha-enhanced PHOENIX-based grid for cool dwarfs
    # SAND chemistries are specific ([M/H], [alpha/M]) pairs, not a regular met x alpha grid; the defaults are the
    # near-solar chemistry with the most "recommended" models (coverage from the Zenodo file list, 2026-10-27)
    'sand': GridMeta('sand', 'auto', svo_model='sand', met_default=0.1, alpha_default=0.15,
                     coverage={'teff': (700., 4000.), 'logg': (2., 6.), 'met': (-2.4, 0.3), 'alpha': (-0.05, 0.4)},
                     verified=False, reference='Alvarado et al. 2024',
                     upstream='https://doi.org/10.5281/zenodo.11582126'),
    'sonora-bobcat': GridMeta('sonora-bobcat', 'auto', svo_model='sonora_bobcat', extra_axes=('co',),
                              extra_defaults={'co': 1.},
                              coverage={'teff': (200., 2400.), 'logg': (3., 5.5), 'met': (-0.5, 0.5),
                                        'alpha': (0., 0.), 'co': (0.5, 1.5)}, verified=False,
                              reference='Marley et al. 2021', upstream='https://doi.org/10.5281/zenodo.5063476'),
    'sonora-diamondback': GridMeta('sonora-diamondback', 'auto', svo_model='sonora_diamondback',
                                   extra_axes=('fsed',), extra_defaults={'fsed': 2.},
                                   coverage={'teff': (900., 2400.), 'logg': (3.5, 5.5), 'met': (-0.5, 0.5),
                                             'alpha': (0., 0.), 'fsed': (0., 8.)}, verified=False,
                                   reference='Morley et al. 2024 (fsed = 0: cloud-free)',
                                   upstream='https://doi.org/10.5281/zenodo.12735103'),
    'sonora-elfowl': GridMeta('sonora-elfowl', 'auto', svo_model='sonora_elf_owl', extra_axes=('co', 'kzz'),
                              extra_defaults={'co': 1., 'kzz': 4.},
                              coverage={'teff': (275., 2400.), 'logg': (3.25, 5.5), 'met': (-1., 1.),
                                        'alpha': (0., 0.), 'co': (0.22, 1.14), 'kzz': (2., 9.)}, verified=False,
                              reference='Mukherjee et al. 2024; Wogan et al. 2025 (Elf Owl, Paper IV)',
                              upstream='https://doi.org/10.5281/zenodo.15150881'),
    # LOWZ: metal-poor cool atmospheres; may not be on SVO: use a local directory with its own GridMeta
    'lowz': GridMeta('lowz', 'auto', svo_model='lowz', extra_axes=('co', 'kzz'),
                     extra_defaults={'co': 0.55, 'kzz': -1.},
                     coverage={'teff': (500., 1600.), 'logg': (3.5, 5.25), 'met': (-2.5, 1.), 'alpha': (0., 0.),
                               'co': (0.1, 0.85), 'kzz': (-1., 10.)}, verified=False,
                     reference='Meisner et al. 2021', upstream='https://dataverse.harvard.edu/dataverse/lowz'),
}


def write_metadata(directory: str | Path, meta: GridMeta) -> Path:
    """Write `rvfitter_grid.toml` into a grid directory and return its path"""
    import tomli_w
    path = Path(directory) / METADATA_FILE
    data = {k: (list(v) if isinstance(v, tuple) else {kk: list(vv) if isinstance(vv, tuple) else vv
                                                      for kk, vv in v.items()} if isinstance(v, dict) else v)
            for k, v in asdict(meta).items() if v is not None}
    path.write_text(tomli_w.dumps(data))
    return path


def _read_metadata(directory: Path) -> GridMeta | None:
    path = directory / METADATA_FILE
    if not path.exists():
        return None
    data = tomllib.loads(path.read_text())
    return GridMeta(**data)


# ---------------------------------------------------------------------------------------------------------------
# reading and caching
# ---------------------------------------------------------------------------------------------------------------
def read_two_column(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a whitespace-separated two-column (wavelength, flux) text file, skipping '#' comments

    Returns
    -------
    wave, flux
        Strictly increasing positive wavelengths (duplicates removed) and their fluxes, in the file's medium
    """
    df = pd.read_csv(path, sep=r'\s+', comment='#', header=None, usecols=[0, 1], dtype=float, engine='c')
    wave, flux = df[0].to_numpy(), df[1].to_numpy()
    good = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    wave, flux = wave[good], flux[good]
    order = np.argsort(wave, kind='stable')
    wave, flux = wave[order], flux[order]
    keep = np.concatenate([[True], np.diff(wave) > 0])
    return wave[keep], flux[keep]


@functools.lru_cache(maxsize=32)
def _load_vacuum(path: str, mtime_ns: int, size: int, medium: str) -> tuple[np.ndarray, np.ndarray]:
    """Cached read of one template file, converted to vacuum (the key includes mtime and size)"""
    wave, flux = read_two_column(path)
    wave = to_vacuum(wave, medium)
    keep = np.concatenate([[True], np.diff(wave) > 0])  # the conversion is monotonic, but guard anyway
    wave, flux = wave[keep], flux[keep]
    wave.setflags(write=False)
    flux.setflags(write=False)
    return wave, flux


def _file_key(path: Path) -> tuple[str, int, int]:
    st = path.stat()
    return str(path.resolve()), st.st_mtime_ns, st.st_size


@functools.lru_cache(maxsize=64)
def _broadened_file(path: str, mtime_ns: int, size: int, medium: str, grid: str, resolution: float | None,
                    resolution_native: float | None, wmin: float, wmax: float, dv: float) -> np.ndarray:
    """
    Cached LSF-broadened template on the uniform ln λ grid log_wave_grid(wmin, wmax, dv)

    Keyed on the file (path, mtime, size), the grid name, R and the output grid.
    """
    wave, flux = _load_vacuum(path, mtime_ns, size, medium)
    out_grid = broadening.log_wave_grid(wmin, wmax, dv)
    pad = 10. * (C_KMS / resolution if resolution else dv)
    lo, hi = wmin * broadening.doppler_factor(-pad), wmax * broadening.doppler_factor(pad)
    sel = slice(max(np.searchsorted(wave, lo) - 1, 0), np.searchsorted(wave, hi) + 1)
    w, f = wave[sel], flux[sel]
    if len(w) < 2:
        raise ValueError(f'template {path} does not cover {wmin:.1f}–{wmax:.1f} Å')
    dv_fine = min(float(np.median(np.diff(np.log(w)))) * C_KMS, dv)
    fine = broadening.log_wave_grid(w[0], w[-1], dv_fine)
    f_fine = broadening.rebin(w, f, fine)
    if resolution:
        f_fine = broadening.broaden_loglam(f_fine, dv_fine, resolution=resolution,
                                           resolution_native=resolution_native)
    out = broadening.rebin(fine, f_fine, out_grid)
    out.setflags(write=False)
    log.debug('broadened %s (%s) to R=%s on %d pixels', Path(path).name, grid, resolution, len(out))
    return out


def clear_cache() -> None:
    """Empty the in-memory template and broadening caches"""
    _load_vacuum.cache_clear()
    _broadened_file.cache_clear()


# ---------------------------------------------------------------------------------------------------------------
# the grid
# ---------------------------------------------------------------------------------------------------------------
class TemplateGrid:
    """
    A grid of model spectra in a local directory, over the axes of its GridMeta

    Parameters
    ----------
    directory
        The directory holding the template files (searched non-recursively)
    grid
        A name from `KNOWN_GRIDS`, used when the directory has no metadata file
    meta
        Explicit metadata, overriding both the file and `grid`
    verify
        Run `verify_medium` before the first use and refuse the grid if the declared medium is wrong (an 'auto'
        medium is measured instead)
    """

    def __init__(self, directory: str | Path, grid: str | None = None, meta: GridMeta | None = None,
                 verify: bool = True):
        self.directory = Path(directory).expanduser()
        if not self.directory.is_dir():
            raise FileNotFoundError(f'template directory {self.directory} does not exist')
        if meta is None:
            meta = _read_metadata(self.directory)
        if meta is None:
            if grid is None or grid not in KNOWN_GRIDS:
                raise FileNotFoundError(f'{self.directory} has no {METADATA_FILE}; pass a known grid name '
                                        f'({sorted(KNOWN_GRIDS)}) or a GridMeta')
            meta = KNOWN_GRIDS[grid]
            log.warning('%s has no %s: assuming the %s defaults (medium %s)', self.directory, METADATA_FILE,
                        grid, meta.medium)
        files, points = [], []
        for path in sorted(self.directory.iterdir()):
            p = meta.parse(path.name) if path.is_file() else None
            if p is not None:
                files.append(path)
                points.append(p)
        if not files:
            raise FileNotFoundError(f'no files in {self.directory} match the pattern {meta.pattern!r}')
        self.files = files
        self.points = np.array(points, dtype=float)
        self._index = {tuple(p): f for p, f in zip(points, files, strict=True)}
        self.meta = meta
        self._verify = verify
        self._verified = not verify and meta.medium != 'auto'
        if meta.medium == 'auto':
            self._resolve_auto_medium()
        log.debug('grid %s: %d templates, %s', meta.name, len(files),
                  ', '.join(f'{a} {lo:g}-{hi:g}' for a, (lo, hi) in self.bounds().items()))

    def __repr__(self) -> str:
        return f'TemplateGrid({self.meta.name!r}, {len(self.files)} files, medium={self.meta.medium})'

    def _resolve_auto_medium(self) -> None:
        mid = {a: 0.5 * (lo + hi) for a, (lo, hi) in self.bounds().items()}
        path = self._index[self.nearest(**mid)]
        res = measure_medium(*read_two_column(path))
        if res['medium'] is None:
            raise MediumMismatchError(f'grid {self.name}: the wavelength medium could not be measured from '
                                      f'{path.name}; declare it in {METADATA_FILE}')
        log.info('Grid %s: medium measured as %s from %s (%d air vs %d vacuum alkali lines)', self.name,
                 res['medium'], path.name, res['n_air'], res['n_vacuum'])
        self.meta = replace(self.meta, medium=res['medium'])
        self._verified = True

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def axes(self) -> tuple[str, ...]:
        return self.meta.axes

    @property
    def key(self) -> str:
        """A string identifying the grid (name and directory) for caches and provenance"""
        return f'{self.meta.name}:{self.directory.resolve()}'

    def axis_values(self, axis: str) -> np.ndarray:
        """The distinct values of one axis"""
        return np.unique(self.points[:, self.axes.index(axis)])

    def bounds(self) -> dict[str, tuple[float, float]]:
        """{axis: (min, max)} over the files present"""
        return {a: (float(self.points[:, i].min()), float(self.points[:, i].max())) for i, a in enumerate(self.axes)}

    def _ensure_verified(self) -> None:
        if not self._verified:
            verify_medium(self)
            self._verified = True

    def params(self, teff: float, logg: float = 5., met: float | None = None, **extra: float | None) -> dict:
        """A full parameter dict: missing or None axes take the grid's defaults"""
        unknown = set(extra) - set(self.axes)
        if unknown:
            raise ValueError(f'grid {self.name} has no axes {sorted(unknown)} (axes: {self.axes})')
        p = dict(self.meta.defaults())
        p.update(teff=float(teff), logg=float(logg))
        if met is not None:
            p['met'] = float(met)
        p.update({k: float(v) for k, v in extra.items() if v is not None})
        return {a: p[a] for a in self.axes}

    def nearest(self, teff: float, logg: float = 5., met: float | None = None, **extra) -> tuple[float, ...]:
        """The grid point nearest to the request (distance scaled by `AXIS_SCALE`)"""
        p = self.params(teff, logg, met, **extra)
        d = np.zeros(len(self.points))
        for i, a in enumerate(self.axes):
            d += ((self.points[:, i] - p[a]) / AXIS_SCALE.get(a, 1.)) ** 2
        return tuple(float(x) for x in self.points[np.argmin(d)])

    def _corners(self, p: dict):
        """
        Multilinear corners and weights over every axis

        Returns
        -------
        corners, missing
            corners: [(path, weight)], or None if the request lies outside the grid's range on some axis or a
            corner file is missing; missing: the parameter tuples of the missing corners ([] when outside)
        """
        per_axis = []
        for a in self.axes:
            axis, value = self.axis_values(a), p[a]
            if value < axis[0] - 1e-9 or value > axis[-1] + 1e-9:
                return None, []
            j = int(np.clip(np.searchsorted(axis, value), 1, max(len(axis) - 1, 1)))
            if len(axis) == 1 or np.isclose(value, axis[j - 1]) or np.isclose(value, axis[min(j, len(axis) - 1)]):
                per_axis.append(((float(axis[np.argmin(np.abs(axis - value))]), 1.),))
            else:
                lo, hi = axis[j - 1], axis[j]
                t = (value - lo) / (hi - lo)
                per_axis.append(((float(lo), 1. - t), (float(hi), t)))
        corners, missing = [], []
        for combo in np.ndindex(*[len(x) for x in per_axis]):
            key = tuple(per_axis[i][k][0] for i, k in enumerate(combo))
            w = float(np.prod([per_axis[i][k][1] for i, k in enumerate(combo)]))
            if w <= 0:
                continue
            if key not in self._index:
                missing.append(key)
            else:
                corners.append((self._index[key], w))
        return (None, missing) if missing else (corners, [])

    def _resolve(self, p: dict, method: str):
        """
        Corners for a request, with fallbacks: (corners, outside, missing)

        Outside the grid's range → nearest point, outside=True (flag template_grid_edge). Inside the range but
        with a missing corner file (a hole) → nearest point, missing lists the absent corners (flag
        template_grid_hole). Renormalising over the corners that exist is not done: it would bias the template.
        """
        corners, missing = (None, []) if method == 'nearest' else self._corners(p)
        outside = False
        if corners is None:
            point = self.nearest(**p)
            near = _fmt_params(dict(zip(self.axes, point, strict=True)))
            if method != 'nearest' and missing:
                names = [_generic_name(dict(zip(self.axes, m, strict=True))) for m in missing]
                log.warning('%s lies in a hole of grid %s (missing corner%s: %s): using the nearest point %s',
                            _fmt_params(p), self.name, 's' if len(missing) > 1 else '', ', '.join(names), near)
            elif method != 'nearest':
                outside = True
                log.warning('%s is outside grid %s: using the nearest point %s', _fmt_params(p), self.name, near)
            corners = [(self._index[point], 1.)]
        return corners, outside, missing

    def _meta(self, p, corners, outside, missing=(), **extra) -> dict:
        return dict(object=f'{self.name} {_fmt_params(p, short=True)}', grid=self.name, params=dict(p), **p,
                    medium_in=self.meta.medium, files=[c[0].name for c in corners],
                    weights=[float(c[1]) for c in corners], interpolated=len(corners) > 1, outside_grid=outside,
                    grid_hole=bool(missing), missing_corners=[list(m) for m in missing], **extra)

    def get(self, teff: float, logg: float = 5., met: float | None = None, *,
            wave_range: tuple[float, float] | None = None, method: str = 'linear', **extra: float) -> Spectrum:
        """
        A template at the requested parameters, multilinearly interpolated in flux, or the nearest grid point

        Parameters
        ----------
        teff, logg, met
            The standard parameters; met defaults to the grid's default
        wave_range
            Vacuum wavelength range [Å] to return (default: everything)
        method
            'linear' (default), falling back to the nearest point with meta['outside_grid'] = True outside the
            grid or where a corner is missing; or 'nearest'
        **extra
            Other axes (alpha and the grid's extra axes); missing ones take their defaults

        Returns
        -------
        template
            Vacuum wavelengths, flux and NaN σ; meta records the parameters, files, weights and flags
        """
        if method not in ('linear', 'nearest'):
            raise ValueError(f"method must be 'linear' or 'nearest', not {method!r}")
        self._ensure_verified()
        p = self.params(teff, logg, met, **extra)
        corners, outside, missing = self._resolve(p, method)
        wave_full, flux = _load_vacuum(*_file_key(corners[0][0]), self.meta.medium)
        if wave_range is not None:
            sel = (wave_full >= wave_range[0]) & (wave_full <= wave_range[1])
        else:
            sel = np.ones(len(wave_full), dtype=bool)
        wave = wave_full[sel]
        total = corners[0][1] * flux[sel]
        for path, w in corners[1:]:
            w2, f2 = _load_vacuum(*_file_key(path), self.meta.medium)
            total = total + w * (f2[sel] if np.array_equal(w2, wave_full) else np.interp(wave, w2, f2))
        return Spectrum(wave, total, np.full(len(wave), np.nan), self.meta.resolution_native,
                        self._meta(p, corners, outside, missing), str(corners[0][0]))

    def broadened(self, teff: float, logg: float = 5., met: float | None = None, *, resolution: float | None,
                  wave_range: tuple[float, float], dv: float | None = None, vsini: float = 0., epsilon: float = 0.6,
                  method: str = 'linear', **extra: float) -> Spectrum:
        """
        A template broadened to resolving power R (and optionally vsini) on a uniform ln λ grid

        The LSF-broadened grid points are cached (keyed on the file, R, the grid and the output grid), and
        interpolation is done after broadening, which is equivalent because both operations are linear.

        Parameters
        ----------
        teff, logg, met, method, **extra
            As for `get`
        resolution
            Target resolving power (None: no instrumental broadening)
        wave_range
            Vacuum wavelength range [Å] of the output grid
        dv
            Output pixel step [km/s]; default c/(10 R), or 1 km/s without R
        vsini, epsilon
            Rotational broadening (applied after the cache)

        Returns
        -------
        template
            The broadened template; resolution is set to R
        """
        self._ensure_verified()
        p = self.params(teff, logg, met, **extra)
        corners, outside, missing = self._resolve(p, method)
        if dv is None:
            dv = C_KMS / (10. * resolution) if resolution else 1.
        wmin, wmax = (float(x) for x in wave_range)
        flux = None
        for path, w in corners:
            f = _broadened_file(*_file_key(path), self.meta.medium, self.key,
                                None if resolution is None else float(resolution),
                                self.meta.resolution_native, wmin, wmax, float(dv))
            flux = w * f if flux is None else flux + w * f
        wave = broadening.log_wave_grid(wmin, wmax, dv)
        if vsini > 0:
            flux = broadening.broaden_loglam(flux, dv, vsini=vsini, epsilon=epsilon)
        return Spectrum(wave, flux, np.full(len(wave), np.nan), resolution,
                        self._meta(p, corners, outside, missing, vsini=vsini, dv=dv), str(corners[0][0]))

    def select(self, teff: float | None = None, teff_range: tuple[float, float] | None = None,
               logg_range: tuple[float, float] | None = None, met_range: tuple[float, float] | None = None,
               alpha_range: tuple[float, float] | None = None,
               ranges: Mapping[str, tuple[float, float]] | None = None,
               priors: Mapping[str, float] | None = None) -> list[tuple[float, ...]]:
        """
        Grid points within the given ranges (inclusive)

        teff and log g without a range take every value (teff: ±300 K around `teff` if given). Every other
        axis (met, alpha, extra axes) without a range is held at the grid value nearest its prior (if given in
        `priors`) or its default, so the candidate count does not multiply over axes the user did not ask for.

        Returns
        -------
        points
            Sorted tuples in `axes` order
        """
        if teff_range is None and teff is not None:
            teff_range = (teff - 300., teff + 300.)
        rng = dict(ranges or {})
        rng.update({k: v for k, v in (('teff', teff_range), ('logg', logg_range), ('met', met_range),
                                      ('alpha', alpha_range)) if v is not None})
        unknown = set(rng) - set(self.axes)
        if unknown:
            raise ValueError(f'grid {self.name} has no axes {sorted(unknown)} (axes: {self.axes})')
        priors = dict(priors or {})
        defaults = self.meta.defaults()
        keep = np.ones(len(self.points), dtype=bool)
        for i, a in enumerate(self.axes):
            if a in rng:
                lo, hi = rng[a]
                keep &= (self.points[:, i] >= lo - 1e-9) & (self.points[:, i] <= hi + 1e-9)
            elif a not in ('teff', 'logg'):
                vals = self.axis_values(a)
                target = priors.get(a, defaults.get(a, vals[len(vals) // 2]))
                keep &= np.isclose(self.points[:, i], vals[np.argmin(np.abs(vals - target))])
        return sorted(tuple(float(x) for x in p) for p in self.points[keep])


def _fmt_params(p: Mapping[str, float], short: bool = False) -> str:
    if short:
        return '/'.join(f'{v:g}' for v in p.values())
    return ', '.join(f'{k} {v:g}' for k, v in p.items())


# ---------------------------------------------------------------------------------------------------------------
# medium verification
# ---------------------------------------------------------------------------------------------------------------
def _line_minimum(wave: np.ndarray, flux: np.ndarray, centre: float, half: float) -> float | None:
    """Wavelength of the flux minimum within ±half of centre, refined by a parabola; None if not a clear dip"""
    sel = np.flatnonzero(np.abs(wave - centre) <= half)
    if len(sel) < 5:
        return None
    i = sel[np.argmin(flux[sel])]
    if i in (sel[0], sel[-1]) or i == 0 or i == len(wave) - 1:
        return None
    x, y = wave[i - 1:i + 2], flux[i - 1:i + 2]
    denom = y[0] - 2 * y[1] + y[2]
    if denom <= 0:
        return float(x[1])
    h = 0.5 * (x[2] - x[0])
    return float(x[1] + 0.5 * h * (y[0] - y[2]) / denom)


def measure_medium(wave: np.ndarray, flux: np.ndarray) -> dict:
    """
    Decide whether a model's wavelengths are in air or vacuum from the minima of the alkali resonance lines

    Each line minimum is located within ±(|λ_vac − λ_air| + 1 Å) of the mid-point and assigned to the closer
    of the NIST vacuum and air wavelengths.

    Parameters
    ----------
    wave, flux
        The model, in its own (unconverted) wavelengths [Å]

    Returns
    -------
    result
        'medium' ('air', 'vacuum' or None if no line could be measured), 'n_air', 'n_vacuum', and per-line
        'offsets_air' and 'offsets_vacuum' [Å]
    """
    n_air = n_vac = 0
    off_air, off_vac = [], []
    for _, lam_vac in medium_check_lines():
        lam_air = float(vac_to_air(lam_vac))
        mid, half = 0.5 * (lam_vac + lam_air), 0.5 * abs(lam_vac - lam_air) + 1.
        lam_min = _line_minimum(wave, flux, mid, half)
        if lam_min is None:
            continue
        d_air, d_vac = lam_min - lam_air, lam_min - lam_vac
        off_air.append(d_air)
        off_vac.append(d_vac)
        if abs(d_air) < abs(d_vac):
            n_air += 1
        else:
            n_vac += 1
    medium = None if n_air + n_vac == 0 else ('air' if n_air > n_vac else 'vacuum' if n_vac > n_air else None)
    return dict(medium=medium, n_air=n_air, n_vacuum=n_vac, offsets_air=off_air, offsets_vacuum=off_vac)


def verify_medium(grid: TemplateGrid, teff: float | None = None, logg: float | None = None) -> dict:
    """
    Check a grid's declared wavelength medium against its alkali line minima

    Parameters
    ----------
    grid
        The grid
    teff, logg
        The template to test (default: the grid point nearest the middle of the grid)

    Returns
    -------
    result
        As `measure_medium`, plus 'declared' and 'file'

    Raises
    ------
    MediumMismatchError
        If the measured medium disagrees with the declared one
    """
    b = grid.bounds()
    mid = {a: 0.5 * (lo + hi) for a, (lo, hi) in b.items()}
    if teff is not None:
        mid['teff'] = teff
    if logg is not None:
        mid['logg'] = logg
    path = grid._index[grid.nearest(**mid)]
    wave, flux = read_two_column(path)
    res = measure_medium(wave, flux)
    res.update(declared=grid.meta.medium, file=path.name)
    if res['medium'] is None:
        log.warning('Could not measure the wavelength medium of %s from %s; trusting the declared %s',
                    grid.name, path.name, grid.meta.medium)
    elif res['medium'] != grid.meta.medium:
        raise MediumMismatchError(f'grid {grid.name} declares {grid.meta.medium} wavelengths but the alkali minima in '
                                  f'{path.name} match {res["medium"]} ({res["n_air"]} air vs {res["n_vacuum"]} '
                                  f'vacuum lines)')
    else:
        offs = res['offsets_air'] if res['medium'] == 'air' else res['offsets_vacuum']
        log.info('Grid %s: declared medium %s confirmed (%d lines, median offset %.3f Å)', grid.name,
                 grid.meta.medium, len(offs), float(np.median(offs)))
    return res


# ---------------------------------------------------------------------------------------------------------------
# download from SVO
# ---------------------------------------------------------------------------------------------------------------
def cache_dir() -> Path:
    """The user cache directory for downloaded templates (platformdirs; overridable with $RVFITTER_CACHE)"""
    import os

    import platformdirs
    env = os.environ.get('RVFITTER_CACHE')
    return Path(env) if env else Path(platformdirs.user_cache_dir('rvfitter')) / 'templates'


def _fetch(url: str, timeout: float = 60.) -> bytes:
    """GET a URL (the single network entry point; patched in tests)"""
    req = urllib.request.Request(url, headers={'User-Agent': _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https/http URLs)
        return resp.read()


def _parse_ssap(votable_bytes: bytes) -> list[dict]:
    """Rows of an SSAP VOTable as dicts with 'url' and every numeric parameter column (lower-case names)"""
    import io

    from astropy.io.votable import parse_single_table
    table = parse_single_table(io.BytesIO(votable_bytes)).to_table()
    names = {n.lower(): n for n in table.colnames}

    def find(*cands):
        for c in cands:
            for low, orig in names.items():
                if low == c or low.endswith(c):
                    return orig
        return None

    url_col = find('access_url', 'acref', 'spectrum', 'url')
    teff_col, logg_col = find('teff'), find('logg')
    if url_col is None or teff_col is None or logg_col is None:
        raise ValueError(f'unrecognised SSAP response columns: {table.colnames}')
    aliases = {'met': ('meta', 'met', 'feh', 'mh', 'z'), 'alpha': ('alpha', 'afe'), 'co': ('co', 'c_o', 'c/o'),
               'kzz': ('kzz', 'logkzz'), 'fsed': ('fsed',)}
    cols = {'teff': teff_col, 'logg': logg_col}
    for axis, cands in aliases.items():
        col = find(*cands)
        if col is not None and col not in cols.values():
            cols[axis] = col
    rows = []
    for row in table:
        d = {'url': str(row[url_col])}
        d.update({axis: float(row[col]) for axis, col in cols.items()})
        rows.append(d)
    return rows


def _generic_name(p: Mapping[str, float]) -> str:
    name = f't{p["teff"]:g}_g{p["logg"]:.2f}_m{p.get("met", 0.):+.2f}_a{p.get("alpha", 0.):+.2f}'
    for axis in ('co', 'kzz', 'fsed'):
        if axis in p:
            name += f'_{axis}{p[axis]:g}'
    return name + '.txt'


def download(grid: str = 'bt-settl-cifist', teff_range: tuple[float, float] = (1200., 3000.),
             logg_range: tuple[float, float] = (4.5, 5.5), met_range: tuple[float, float] | None = None,
             alpha_range: tuple[float, float] | None = None, *, ranges: Mapping[str, tuple[float, float]] | None = None,
             directory: str | Path | None = None, fetch: Callable[[str], bytes] | None = None, verify: bool = True,
             source: str = 'auto') -> TemplateGrid:
    """
    Download a template grid into the cache, from SVO or from its upstream Zenodo record

    Parameters
    ----------
    grid, teff_range, logg_range, met_range, alpha_range, ranges, directory, verify
        As for `_download_svo` (met and alpha default to the grid default only)
    fetch
        URL → bytes for the SVO queries (tests, proxies)
    source
        'svo', 'zenodo', or 'auto' (default): SVO first, then the upstream Zenodo record if SVO fails. Grids with
        neither (LOWZ, BT-Settl AGSS2009 when SVO is down) must be downloaded manually from `upstream`

    Returns
    -------
    grid
        The downloaded TemplateGrid
    """
    from . import grid_sources
    if source not in ('auto', 'svo', 'zenodo'):
        raise ValueError(f"source must be 'auto', 'svo' or 'zenodo', not {source!r}")
    meta = KNOWN_GRIDS.get(grid)
    if meta is None:
        raise ValueError(f'unknown grid {grid!r}; choose from {sorted(KNOWN_GRIDS)}')
    if source in ('auto', 'svo') and meta.svo_model is not None:
        try:
            return _download_svo(grid, teff_range, logg_range, met_range, alpha_range, ranges=ranges,
                                 directory=directory, fetch=fetch, verify=verify)
        except Exception as exc:  # noqa: BLE001 - network failures fall back to the upstream source
            if source == 'svo' or grid not in grid_sources.ZENODO_GRIDS:
                raise
            log.warning('SVO download of %s failed (%s); using the upstream Zenodo record', grid, exc)
    if grid not in grid_sources.ZENODO_GRIDS:
        raise ValueError(f'grid {grid} has no automatic upstream source: download it manually from {meta.upstream} '
                         f'and open the directory with a GridMeta (templates.local_dir)')
    return _download_zenodo(grid, teff_range, logg_range, met_range, alpha_range, ranges=ranges,
                            directory=directory, verify=verify)


def _download_zenodo(grid, teff_range, logg_range, met_range, alpha_range, *, ranges=None, directory=None,
                     verify=True) -> TemplateGrid:
    """Download from the grid's Zenodo record (see `grid_sources`); files are written in the generic format"""
    from . import grid_sources
    meta = KNOWN_GRIDS[grid]
    directory = Path(directory) if directory is not None else cache_dir() / grid
    directory.mkdir(parents=True, exist_ok=True)
    defaults = meta.defaults()
    want = {'teff': tuple(teff_range), 'logg': tuple(logg_range),
            'met': tuple(met_range) if met_range is not None else (defaults['met'],) * 2,
            'alpha': tuple(alpha_range) if alpha_range is not None else (defaults['alpha'],) * 2}
    for axis in meta.extra_axes:
        want[axis] = tuple((ranges or {}).get(axis, (defaults[axis],) * 2))
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    media = set()

    def write(params, wave, flam, medium, prov):
        p = {a: params.get(a, defaults.get(a, 0.)) for a in meta.axes}
        path = directory / _generic_name(p)
        np.savetxt(path, np.c_[wave, flam], fmt='%.6f %.6e')
        manifest[path.name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **p, **prov)
        if medium:
            media.add(medium)
        return path

    paths = grid_sources.download_zenodo(grid, want, directory, axes=meta.axes, write=write)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    log.info('%s: %d models from Zenodo (%s) in %s', grid, len(paths), grid_sources.ZENODO_GRIDS[grid].doi,
             directory)
    medium = media.pop() if len(media) == 1 else meta.medium  # a header-declared medium is still verified
    out_meta = replace(meta, pattern=GENERIC_PATTERN, teff_scale=1., medium=medium)
    tg = TemplateGrid(directory, meta=out_meta, verify=verify)
    if verify:
        tg._ensure_verified()
    write_metadata(directory, tg.meta)
    return tg


def _download_svo(grid: str = 'bt-settl-cifist', teff_range: tuple[float, float] = (1200., 3000.),
             logg_range: tuple[float, float] = (4.5, 5.5), met_range: tuple[float, float] | None = None,
             alpha_range: tuple[float, float] | None = None, *, ranges: Mapping[str, tuple[float, float]] | None = None,
             directory: str | Path | None = None, fetch: Callable[[str], bytes] | None = None,
             verify: bool = True) -> TemplateGrid:
    """
    Download a template grid from the SVO theory server into the cache, with SHA-256 checksums

    Files already in the cache whose checksum matches the manifest are not downloaded again; a mismatch is
    re-downloaded. The directory gets a metadata file (generic file-name pattern, the grid's axes and its
    medium, measured if the grid's is 'auto') and a `manifest.json` (file → sha256, source URL and parameters).

    Parameters
    ----------
    grid
        A name in `KNOWN_GRIDS` with an SVO model
    teff_range, logg_range, met_range, alpha_range
        Inclusive parameter ranges (met and alpha: None = the grid default only, e.g. solar)
    ranges
        Ranges of extra axes (e.g. {'co': (0.5, 1.0)})
    directory
        Where to store the grid (default: `cache_dir() / grid`)
    fetch
        Function URL → bytes (default: urllib); for tests or proxies
    verify
        Verify (or, for 'auto', measure) the medium

    Returns
    -------
    grid
        The downloaded TemplateGrid
    """
    meta = KNOWN_GRIDS.get(grid)
    if meta is None or meta.svo_model is None:
        raise ValueError(f'unknown downloadable grid {grid!r}; choose from {sorted(KNOWN_GRIDS)}')
    if not meta.verified:
        log.warning('Grid %s: the SVO model id %r is not yet verified; if the query fails, check '
                    '"rvfitter templates list --available"', grid, meta.svo_model)
    fetch = fetch or _fetch
    directory = Path(directory) if directory is not None else cache_dir() / grid
    directory.mkdir(parents=True, exist_ok=True)
    defaults = meta.defaults()
    # an unranged met/alpha is held at the grid default (solar, unenhanced), not the whole axis
    met_range = met_range if met_range is not None else (defaults['met'], defaults['met'])
    alpha_range = alpha_range if alpha_range is not None else (defaults['alpha'], defaults['alpha'])
    want = {'teff': teff_range, 'logg': logg_range, 'met': met_range, 'alpha': alpha_range, **dict(ranges or {})}
    params = dict(model=meta.svo_model, REQUEST='queryData')
    svo_names = {'teff': 'teff', 'logg': 'logg', 'met': 'meta', 'alpha': 'alpha'}
    for axis, (lo, hi) in want.items():
        params[f'{svo_names.get(axis, axis)}_min'] = lo
        params[f'{svo_names.get(axis, axis)}_max'] = hi
    log.info('Querying SVO for %s (%s)', grid, ', '.join(f'{a} {lo:g} to {hi:g}' for a, (lo, hi) in want.items()))
    rows = _parse_ssap(fetch(f'{SVO_SSAP_URL}?{urllib.parse.urlencode(params)}'))
    rows = [r for r in rows if all(lo - 1e-9 <= r.get(a, lo) <= hi + 1e-9 for a, (lo, hi) in want.items())]
    if not rows:
        raise FileNotFoundError(f'SVO returned no {grid} models in the requested ranges')
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for i, r in enumerate(rows, 1):
        name = _generic_name(r)
        path = directory / name
        known = manifest.get(name)
        if path.exists() and known and hashlib.sha256(path.read_bytes()).hexdigest() == known['sha256']:
            log.debug('[%d/%d] %s cached', i, len(rows), name)
            continue
        log.info('[%d/%d] downloading %s', i, len(rows), name)
        data = fetch(r['url'])
        path.write_bytes(data)
        manifest[name] = dict(sha256=hashlib.sha256(data).hexdigest(), **r)
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    out_meta = replace(meta, pattern=GENERIC_PATTERN, teff_scale=1.)
    tg = TemplateGrid(directory, meta=out_meta, verify=verify)  # an 'auto' medium is measured here
    if verify:
        tg._ensure_verified()
    write_metadata(directory, tg.meta)
    return tg


def available(fetch: Callable[[str], bytes] | None = None, grids: Sequence[str] | None = None) -> dict[str, dict]:
    """
    Check which known grids the SVO theory server serves, with a small SSAP query per grid

    Returns
    -------
    status
        {grid: {'svo_model', 'ok', 'n' (rows returned for 1900–2100 K), 'error'}}
    """
    fetch = fetch or _fetch
    out = {}
    for name in grids or sorted(KNOWN_GRIDS):
        meta = KNOWN_GRIDS[name]
        if meta.svo_model is None:
            continue
        q = urllib.parse.urlencode(dict(model=meta.svo_model, REQUEST='queryData', teff_min=1900, teff_max=2100))
        try:
            rows = _parse_ssap(fetch(f'{SVO_SSAP_URL}?{q}'))
            out[name] = dict(svo_model=meta.svo_model, ok=bool(rows), n=len(rows), error='')
        except Exception as exc:  # noqa: BLE001 - reported per grid
            out[name] = dict(svo_model=meta.svo_model, ok=False, n=0, error=str(exc)[:200])
    return out


def _cli_telluric(args) -> int:
    """`rvfitter templates telluric [--build]`: the model's status and provenance, or build it"""
    import os

    from .calibration import telluric, telluric_build
    if args.out:
        os.environ['RVFITTER_TELLURIC_MODEL'] = str(Path(args.out).resolve())
    if args.build:
        try:
            telluric_build.build(args.out, line_cache=args.line_cache, force=args.force)
        except Exception as exc:  # noqa: BLE001 - reported to the user
            log.error('Building the telluric model failed: %s', exc)
            return 1
    if not telluric.available():
        log.info('Telluric model: not built (%s); build it with `%s`', telluric.cache_path(), telluric.BUILD_COMMAND)
        return 0
    m = telluric.model()
    log.info('Telluric model: %s (%.0f-%.0f Å)', telluric.cache_path(), m.wmin, m.wmax)
    for key in ('source', 'builder', 'built', 'citation', 'terms'):
        if m.meta.get(key):
            log.info('  %s: %s', key, m.meta[key])
    return 0


def cli_main(argv: Sequence[str] | None = None) -> int:
    """
    `rvfitter templates download|list|verify|telluric`

    Returns
    -------
    code
        Process exit code
    """
    import argparse
    parser = argparse.ArgumentParser(prog='rvfitter templates', description='Manage template grids.')
    sub = parser.add_subparsers(dest='action', required=True)
    d = sub.add_parser('download', help='download a grid from the SVO theory server')
    d.add_argument('--grid', default='bt-settl-cifist', choices=sorted(KNOWN_GRIDS))
    d.add_argument('--teff', nargs=2, type=float, default=(1200., 3000.), metavar=('MIN', 'MAX'), help='K')
    d.add_argument('--logg', nargs=2, type=float, default=(4.5, 5.5), metavar=('MIN', 'MAX'))
    d.add_argument('--met', nargs=2, type=float, default=None, metavar=('MIN', 'MAX'), help='[M/H]')
    d.add_argument('--alpha', nargs=2, type=float, default=None, metavar=('MIN', 'MAX'), help='[alpha/M]')
    d.add_argument('--range', nargs=3, action='append', default=[], metavar=('AXIS', 'MIN', 'MAX'),
                   help='range of an extra axis (e.g. --range co 0.5 1.0)')
    d.add_argument('--dir', default=None)
    d.add_argument('--source', choices=('auto', 'svo', 'zenodo'), default='auto',
                   help='where to download from (auto: SVO, then the upstream Zenodo record)')
    ls = sub.add_parser('list', help='list the grids in the cache or a directory, or those available on SVO')
    ls.add_argument('--dir', default=None)
    ls.add_argument('--available', action='store_true', help='query SVO for the known grids')
    v = sub.add_parser('verify', help='check the declared wavelength medium of a grid')
    v.add_argument('directory')
    v.add_argument('--grid', default=None)
    t = sub.add_parser('telluric', help='show, or build (--build), the HITRAN telluric transmission model')
    t.add_argument('--build', action='store_true', help='build it from HITRAN line lists (fetched once, ~90 MB)')
    t.add_argument('--force', action='store_true', help='rebuild even if it exists')
    t.add_argument('--out', default=None, help='model file (default: the user cache)')
    t.add_argument('--line-cache', default=None, help='directory of the HITRAN line files (default: next to --out)')
    args = parser.parse_args(argv)

    if args.action == 'telluric':
        return _cli_telluric(args)

    if args.action == 'download':
        extra = {a: (float(lo), float(hi)) for a, lo, hi in args.range}
        tg = download(args.grid, tuple(args.teff), tuple(args.logg), None if args.met is None else tuple(args.met),
                      None if args.alpha is None else tuple(args.alpha), ranges=extra, directory=args.dir,
                      source=args.source)
        log.info('%s: %d templates in %s (medium %s)', tg.name, len(tg.files), tg.directory, tg.meta.medium)
    elif args.action == 'list':
        if args.available:
            for name, st in available().items():
                note = '' if KNOWN_GRIDS[name].verified else ' (id unverified)'
                log.info('%-20s SVO model %-22s %s%s', name, st['svo_model'],
                         f'available ({st["n"]} models at 1900-2100 K)' if st['ok'] else
                         f'NOT available: {st["error"] or "no rows"}', note)
            return 0
        root = Path(args.dir) if args.dir else cache_dir()
        dirs = [root] + sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        for d_ in dirs:
            if (d_ / METADATA_FILE).exists():
                tg = TemplateGrid(d_, verify=False)
                log.info('%s: %d templates, %s (%s)', tg.name, len(tg.files),
                         ', '.join(f'{a} {lo:g}-{hi:g}' for a, (lo, hi) in tg.bounds().items()), d_)
    else:
        tg = TemplateGrid(args.directory, grid=args.grid, verify=False)
        try:
            verify_medium(tg)
        except MediumMismatchError as err:
            log.error('%s', err)
            return 1
    return 0
