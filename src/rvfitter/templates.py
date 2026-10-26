"""
Template grids (BT-Settl-CIFIST, Sonora, user-supplied): metadata, medium check, interpolation, broadening
cache and download from the SVO theory server.

A grid is a local directory of two-column (wavelength [Å], flux) text files, plus a metadata file
`rvfitter_grid.toml` declaring:

- `name`: the grid name;
- `medium`: "air" or "vacuum" (of the wavelengths in the files);
- `flux_units`: free text;
- `pattern`: a regular expression on the file name with named groups `teff`, `logg` and optionally `met`;
- `teff_scale`: the factor turning the `teff` group into kelvin (100 for Lyon `lte020.0` names);
- `met_default`: [M/H] when the pattern has no `met` group;
- `resolution_native`: the model's resolving power, if finite.

Directories without a metadata file can be opened with a known grid name (`KNOWN_GRIDS`), which supplies the
same defaults with a logged warning.
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
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import broadening
from .constants import C_KMS
from .medium import to_vacuum, vac_to_air
from .types import Spectrum

__all__ = ['GridMeta', 'TemplateGrid', 'KNOWN_GRIDS', 'medium_check_lines', 'MediumMismatchError', 'METADATA_FILE',
           'read_two_column', 'write_metadata', 'verify_medium', 'measure_medium', 'download', 'cache_dir',
           'cli_main', 'SVO_SSAP_URL']

log = logging.getLogger(__name__)

METADATA_FILE = 'rvfitter_grid.toml'
SVO_SSAP_URL = 'http://svo2.cab.inta-csic.es/theory/newov2/ssap.php'
_LYON_PATTERN = r'^lte(?P<teff>\d+(?:\.\d+)?)-(?P<logg>\d+(?:\.\d+)?)(?P<met>[-+]\d+(?:\.\d+)?)?.*\.(?:txt|dat)$'

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

    def __post_init__(self):
        if self.medium not in ('air', 'vacuum'):
            raise ValueError(f"grid medium must be 'air' or 'vacuum', not {self.medium!r}")
        re.compile(self.pattern)

    def parse(self, filename: str) -> tuple[float, float, float] | None:
        """(teff [K], logg, [M/H]) from a file name, or None if it does not match"""
        m = re.match(self.pattern, filename)
        if m is None:
            return None
        met = m.groupdict().get('met')
        return (float(m.group('teff')) * self.teff_scale, float(m.group('logg')),
                float(met) if met is not None else self.met_default)


KNOWN_GRIDS: dict[str, GridMeta] = {
    # the Lyon BT-Settl files are in air above 2000 Å (verified: line minima within 0.013 Å of NIST air)
    'bt-settl-cifist': GridMeta('bt-settl-cifist', 'air', svo_model='bt-settl-cifist'),
    'sonora-bobcat': GridMeta('sonora-bobcat', 'vacuum', svo_model='sonora_bobcat'),
    'sonora-elfowl': GridMeta('sonora-elfowl', 'vacuum', svo_model='sonora_elf_owl'),
}


def write_metadata(directory: str | Path, meta: GridMeta) -> Path:
    """Write `rvfitter_grid.toml` into a grid directory and return its path"""
    import tomli_w
    path = Path(directory) / METADATA_FILE
    data = {k: v for k, v in asdict(meta).items() if v is not None}
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
    A grid of model spectra in a local directory

    Parameters
    ----------
    directory
        The directory holding the template files (searched non-recursively)
    grid
        A name from `KNOWN_GRIDS`, used when the directory has no metadata file
    meta
        Explicit metadata, overriding both the file and `grid`
    verify
        Run `verify_medium` before the first use and refuse the grid if the declared medium is wrong
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
        self.meta = meta
        self._verify = verify
        self._verified = not verify
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
        self.axes = tuple(np.unique(self.points[:, i]) for i in range(3))
        log.debug('grid %s: %d templates, teff %s, logg %s, met %s', meta.name, len(files), *self.axes)

    def __repr__(self) -> str:
        return f'TemplateGrid({self.meta.name!r}, {len(self.files)} files, medium={self.meta.medium})'

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def key(self) -> str:
        """A string identifying the grid (name and directory) for caches and provenance"""
        return f'{self.meta.name}:{self.directory.resolve()}'

    def bounds(self) -> tuple[tuple[float, float], ...]:
        """(min, max) of teff, logg and met"""
        return tuple((float(a[0]), float(a[-1])) for a in self.axes)

    def _ensure_verified(self) -> None:
        if not self._verified:
            verify_medium(self)
            self._verified = True

    def nearest(self, teff: float, logg: float = 5., met: float | None = None) -> tuple[float, float, float]:
        """The grid point nearest to the request (scaled distance: 100 K ≡ 0.5 dex in log g ≡ 0.5 dex in [M/H])"""
        met = self.meta.met_default if met is None else met
        d = (((self.points[:, 0] - teff) / 100.) ** 2 + ((self.points[:, 1] - logg) / 0.5) ** 2
             + ((self.points[:, 2] - met) / 0.5) ** 2)
        return tuple(float(x) for x in self.points[np.argmin(d)])

    def _corners(self, teff: float, logg: float, met: float):
        """Trilinear corners and weights, or None if the point is outside the grid or a corner is missing"""
        per_axis = []
        for value, axis in zip((teff, logg, met), self.axes, strict=True):
            if value < axis[0] - 1e-9 or value > axis[-1] + 1e-9:
                return None
            j = int(np.clip(np.searchsorted(axis, value), 1, max(len(axis) - 1, 1)))
            if len(axis) == 1 or np.isclose(value, axis[j - 1]) or np.isclose(value, axis[min(j, len(axis) - 1)]):
                exact = axis[np.argmin(np.abs(axis - value))]
                per_axis.append(((exact, 1.),))
            else:
                lo, hi = axis[j - 1], axis[j]
                t = (value - lo) / (hi - lo)
                per_axis.append(((lo, 1. - t), (hi, t)))
        corners = []
        for a in per_axis[0]:
            for b in per_axis[1]:
                for c in per_axis[2]:
                    w = a[1] * b[1] * c[1]
                    if w <= 0:
                        continue
                    key = (a[0], b[0], c[0])
                    if key not in self._index:
                        return None
                    corners.append((self._index[key], w))
        return corners

    def _resolve(self, teff: float, logg: float, met: float | None, method: str):
        met = self.meta.met_default if met is None else float(met)
        corners = None if method == 'nearest' else self._corners(teff, logg, met)
        outside = False
        if corners is None:
            point = self.nearest(teff, logg, met)
            outside = method != 'nearest'
            if outside:
                log.warning('(%g K, %g, %g) is outside grid %s or not enclosed: using the nearest point %s',
                            teff, logg, met, self.name, point)
            corners = [(self._index[point], 1.)]
        return met, corners, outside

    def _meta(self, teff, logg, met, corners, outside, **extra) -> dict:
        return dict(object=f'{self.name} {teff:g}/{logg:g}/{met:g}', teff=teff, logg=logg, met=met, grid=self.name,
                    medium_in=self.meta.medium, files=[c[0].name for c in corners],
                    weights=[c[1] for c in corners], interpolated=len(corners) > 1, outside_grid=outside, **extra)

    def get(self, teff: float, logg: float = 5., met: float | None = None, *,
            wave_range: tuple[float, float] | None = None, method: str = 'linear') -> Spectrum:
        """
        A template at (teff, logg, met), trilinearly interpolated in flux, or the nearest grid point

        Parameters
        ----------
        teff, logg, met
            The requested parameters; met defaults to the grid's `met_default`
        wave_range
            Vacuum wavelength range [Å] to return (default: everything)
        method
            'linear' (default), falling back to the nearest point with meta['outside_grid'] = True outside the
            grid or where a corner is missing; or 'nearest'

        Returns
        -------
        template
            Vacuum wavelengths, flux and NaN σ; meta records the parameters, files, weights and flags
        """
        if method not in ('linear', 'nearest'):
            raise ValueError(f"method must be 'linear' or 'nearest', not {method!r}")
        self._ensure_verified()
        met, corners, outside = self._resolve(teff, logg, met, method)
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
                        self._meta(teff, logg, met, corners, outside), str(corners[0][0]))

    def broadened(self, teff: float, logg: float = 5., met: float | None = None, *, resolution: float | None,
                  wave_range: tuple[float, float], dv: float | None = None, vsini: float = 0., epsilon: float = 0.6,
                  method: str = 'linear') -> Spectrum:
        """
        A template broadened to resolving power R (and optionally vsini) on a uniform ln λ grid

        The LSF-broadened grid points are cached (keyed on the file, R, the grid and the output grid), and
        interpolation is done after broadening, which is equivalent because both operations are linear.

        Parameters
        ----------
        teff, logg, met, method
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
        met, corners, outside = self._resolve(teff, logg, met, method)
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
                        self._meta(teff, logg, met, corners, outside, vsini=vsini, dv=dv), str(corners[0][0]))

    def select(self, teff: float | None = None, teff_range: tuple[float, float] | None = None,
               logg_range: tuple[float, float] | None = None,
               met_range: tuple[float, float] | None = None) -> list[tuple[float, float, float]]:
        """
        Grid points within the given ranges (inclusive); with only `teff`, the points within ±300 K

        Returns
        -------
        points
            Sorted (teff, logg, met) tuples
        """
        if teff_range is None and teff is not None:
            teff_range = (teff - 300., teff + 300.)
        keep = np.ones(len(self.points), dtype=bool)
        for i, rng in enumerate((teff_range, logg_range, met_range)):
            if rng is not None:
                keep &= (self.points[:, i] >= rng[0]) & (self.points[:, i] <= rng[1])
        return sorted(tuple(float(x) for x in p) for p in self.points[keep])


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
    (t0, t1), (g0, g1), _ = grid.bounds()
    point = grid.nearest(0.5 * (t0 + t1) if teff is None else teff, 0.5 * (g0 + g1) if logg is None else logg)
    path = grid._index[point]
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
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed https/http SVO URLs)
        return resp.read()


def _parse_ssap(votable_bytes: bytes) -> list[dict]:
    """Rows of an SSAP VOTable as dicts with 'url', 'teff', 'logg', 'met'"""
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

    url_col = None
    for col in table.columns.values():
        utype = (col.meta or {}).get('utype', '') if hasattr(col, 'meta') else ''
        if 'access.reference' in str(utype).lower():
            url_col = col.name
    url_col = url_col or find('access_url', 'acref', 'spectrum', 'url')
    teff_col, logg_col, met_col = find('teff'), find('logg'), find('meta', 'met', 'feh', 'z')
    if url_col is None or teff_col is None or logg_col is None:
        raise ValueError(f'unrecognised SSAP response columns: {table.colnames}')
    rows = []
    for row in table:
        rows.append(dict(url=str(row[url_col]), teff=float(row[teff_col]), logg=float(row[logg_col]),
                         met=float(row[met_col]) if met_col is not None else 0.))
    return rows


def download(grid: str = 'bt-settl-cifist', teff_range: tuple[float, float] = (1200., 3000.),
             logg_range: tuple[float, float] = (4.5, 5.5), met_range: tuple[float, float] = (0., 0.), *,
             directory: str | Path | None = None, fetch: Callable[[str], bytes] | None = None,
             verify: bool = True) -> TemplateGrid:
    """
    Download a template grid from the SVO theory server into the cache, with SHA-256 checksums

    Files already in the cache whose checksum matches the manifest are not downloaded again; a mismatch is
    re-downloaded. The directory gets a metadata file and a `manifest.json` (file → sha256 and source URL).

    Parameters
    ----------
    grid
        A name in `KNOWN_GRIDS` with an SVO model ('bt-settl-cifist', 'sonora-bobcat', 'sonora-elfowl')
    teff_range, logg_range, met_range
        Inclusive parameter ranges
    directory
        Where to store the grid (default: `cache_dir() / grid`)
    fetch
        Function URL → bytes (default: urllib); for tests or proxies
    verify
        Run `verify_medium` on the result

    Returns
    -------
    grid
        The downloaded TemplateGrid
    """
    meta = KNOWN_GRIDS.get(grid)
    if meta is None or meta.svo_model is None:
        raise ValueError(f'unknown downloadable grid {grid!r}; choose from {sorted(KNOWN_GRIDS)}')
    fetch = fetch or _fetch
    directory = Path(directory) if directory is not None else cache_dir() / grid
    directory.mkdir(parents=True, exist_ok=True)
    query = urllib.parse.urlencode(dict(model=meta.svo_model, REQUEST='queryData', teff_min=teff_range[0],
                                        teff_max=teff_range[1], logg_min=logg_range[0], logg_max=logg_range[1],
                                        meta_min=met_range[0], meta_max=met_range[1]))
    log.info('Querying SVO for %s (teff %g–%g K, logg %g–%g)', grid, *teff_range, *logg_range)
    rows = _parse_ssap(fetch(f'{SVO_SSAP_URL}?{query}'))
    rows = [r for r in rows if teff_range[0] <= r['teff'] <= teff_range[1]
            and logg_range[0] <= r['logg'] <= logg_range[1] and met_range[0] <= r['met'] <= met_range[1]]
    if not rows:
        raise FileNotFoundError(f'SVO returned no {grid} models in the requested ranges')
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for i, r in enumerate(rows, 1):
        name = f'lte{r["teff"] / 100:05.1f}-{r["logg"]:.1f}{r["met"]:+.1f}.txt'
        path = directory / name
        known = manifest.get(name)
        if path.exists() and known and hashlib.sha256(path.read_bytes()).hexdigest() == known['sha256']:
            log.debug('[%d/%d] %s cached', i, len(rows), name)
            continue
        log.info('[%d/%d] downloading %s', i, len(rows), name)
        data = fetch(r['url'])
        path.write_bytes(data)
        manifest[name] = dict(sha256=hashlib.sha256(data).hexdigest(), url=r['url'], teff=r['teff'],
                              logg=r['logg'], met=r['met'])
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    out_meta = GridMeta(meta.name, meta.medium, meta.flux_units, _LYON_PATTERN, 100., meta.met_default,
                        meta.resolution_native, meta.svo_model)
    write_metadata(directory, out_meta)
    tg = TemplateGrid(directory, verify=verify)
    if verify:
        tg._ensure_verified()
    return tg


def cli_main(argv: Sequence[str] | None = None) -> int:
    """
    `rvfitter templates download|list|verify`

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
    d.add_argument('--teff', default='1200-3000', help='range in K, e.g. 1500-2500')
    d.add_argument('--logg', default='4.5-5.5')
    d.add_argument('--dir', default=None)
    ls = sub.add_parser('list', help='list the grids in the cache or a directory')
    ls.add_argument('--dir', default=None)
    v = sub.add_parser('verify', help='check the declared wavelength medium of a grid')
    v.add_argument('directory')
    v.add_argument('--grid', default=None)
    args = parser.parse_args(argv)

    def rng(s):
        lo, _, hi = s.partition('-')
        return float(lo), float(hi or lo)

    if args.action == 'download':
        tg = download(args.grid, rng(args.teff), rng(args.logg), directory=args.dir)
        log.info('%s: %d templates in %s', tg.name, len(tg.files), tg.directory)
    elif args.action == 'list':
        root = Path(args.dir) if args.dir else cache_dir()
        dirs = [root] + sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        for d_ in dirs:
            if (d_ / METADATA_FILE).exists():
                tg = TemplateGrid(d_, verify=False)
                log.info('%s: %d templates, teff %s, logg %s (%s)', tg.name, len(tg.files), *tg.bounds()[:2],
                         d_)
    else:
        tg = TemplateGrid(args.directory, grid=args.grid, verify=False)
        try:
            verify_medium(tg)
        except MediumMismatchError as err:
            log.error('%s', err)
            return 1
    return 0
