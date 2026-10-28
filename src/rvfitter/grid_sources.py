"""
Upstream sources of template grids other than SVO: the Zenodo records of the model papers.

Each supported grid maps to its Zenodo record(s) (verified live, see `ZENODO_GRIDS`), a rule choosing the record
files that cover the requested parameters, a file-name parser and a reader for the native format. Downloads:

- zip archives (SAND, Diamondback) are read remotely through HTTP range requests, so only the selected members
  are transferred; each member is checked by its zip CRC-32;
- tar.gz archives (Bobcat, Elf Owl) are streamed once; the selected members are converted on the fly, and the
  whole stream is checked against the MD5 checksum that Zenodo publishes (a mismatch deletes the converted
  files and raises).

Every file is first written under a temporary name ('.part') in its final directory and renamed into place only
after its checksum has passed, so the cache never holds an unverified model; on any error the files written by
the call are removed. Transient network errors (timeouts, connection errors, HTTP 429 and 5xx) are retried
three times with exponential backoff; other HTTP errors are not.

Members are never extracted to their archive paths: only regular files whose names parse as models are read,
names are checked with the same rules as `archives` (no absolute paths, no '..'), a member above
`MAX_MEMBER_BYTES` is refused, and the output names are generated from the parsed parameters. The spectra keep
their native sampling and resolution; wavelengths are converted to Å and fluxes to F_λ (arbitrary scale), and the
medium is measured from the alkali lines when the grid is opened ('auto').

All four records are licensed CC-BY-4.0; cite the model papers (`templates.KNOWN_GRIDS[...].reference`). The
licence, record id(s), DOI and a citation string are returned by `provenance` and stored with the downloaded grid.
"""
import hashlib
import http.client
import io
import json
import logging
import os
import re
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

__all__ = ['ZENODO_API', 'ZenodoGrid', 'ZENODO_GRIDS', 'download_zenodo', 'record_files', 'provenance',
           'MAX_MEMBER_BYTES', 'LICENCE', 'RETRIES']

log = logging.getLogger(__name__)

ZENODO_API = 'https://zenodo.org/api/records/'
MAX_MEMBER_BYTES = 500 * 1024 ** 2
LICENCE = 'CC-BY-4.0'
RETRIES = 3  # attempts per network operation
_BACKOFF = 2.  # s; the wait before attempt n + 1 is _BACKOFF * 2 ** (n - 1) seconds
_C_ANGSTROM = 2.99792458e18  # Å/s
_USER_AGENT = 'rvfitter (github.com/Will-Cooper/rvfitter)'


# ---------------------------------------------------------------------------------------------------------------
# network entry points (patched in tests)
# ---------------------------------------------------------------------------------------------------------------
class RangeError(OSError):
    """A range request was not honoured by the server (not retried)"""


def _sleep(seconds: float) -> None:
    time.sleep(seconds)  # patched in tests


def _transient(exc: BaseException) -> bool:
    """Whether a network error may succeed on retry: timeouts, connection errors, HTTP 429 and 5xx"""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, (TimeoutError, ConnectionError, urllib.error.URLError, http.client.HTTPException))


def _retry(func: Callable, what: str, *, attempts: int = RETRIES):
    """Call `func()`, retrying transient errors with exponential backoff (1, 2, ... times `_BACKOFF` seconds)"""
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:
            if attempt == attempts or not _transient(exc):
                raise
            wait = _BACKOFF * 2 ** (attempt - 1)
            log.warning('%s failed (%s); retrying in %.0f s (attempt %d of %d)', what, exc, wait, attempt + 1,
                        attempts)
            _sleep(wait)


def _fetch_json(url: str, timeout: float = 60.) -> dict:
    def get():
        req = urllib.request.Request(url, headers={'User-Agent': _USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https Zenodo URLs)
            return json.load(resp)
    return _retry(get, f'Zenodo query {url}')


def _open_stream(url: str, timeout: float = 120.):
    """A readable binary stream of a whole file"""
    def get():
        req = urllib.request.Request(url, headers={'User-Agent': _USER_AGENT})
        return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310
    return _retry(get, f'Opening {url}')


class _RangeFile(io.RawIOBase):
    """
    A seekable read-only view of a remote file through HTTP range requests (for zipfile)

    Each request must be answered with HTTP 206 and, where the server sends it, a Content-Range that matches the
    request and the file size; otherwise `RangeError` is raised (a server that ignores 'Range' would send the
    whole file). Transient errors are retried.
    """

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, int(size), 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self) -> int:
        return self.pos

    def _get(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}', 'User-Agent': _USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
            status = getattr(r, 'status', None)
            if status != 206:
                raise RangeError(f'{self.url}: range request bytes={start}-{end} answered with HTTP {status}, '
                                 'not 206 (the server does not support range requests)')
            crange = r.headers.get('Content-Range') if getattr(r, 'headers', None) is not None else None
            if crange:
                m = re.match(r'^bytes (\d+)-(\d+)/(\d+|\*)$', crange.strip())
                if m is None or int(m[1]) != start or int(m[2]) > end or (m[3] != '*' and int(m[3]) != self.size):
                    raise RangeError(f'{self.url}: Content-Range {crange!r} does not match the request '
                                     f'bytes={start}-{end} of a {self.size}-byte file')
            data = r.read()
        if len(data) > end - start + 1:
            raise RangeError(f'{self.url}: {len(data)} bytes returned for the {end - start + 1}-byte range '
                             f'{start}-{end}')
        return data

    def readinto(self, b) -> int:
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        data = _retry(lambda: self._get(self.pos, end), f'Range read of {self.url}')
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def _open_zip(url: str, size: int) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(_RangeFile(url, size), buffer_size=1 << 20))


# ---------------------------------------------------------------------------------------------------------------
# native formats → (wavelength [Å], F_λ, medium or None)
# ---------------------------------------------------------------------------------------------------------------
def _text_columns(data: bytes, skip: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(io.BytesIO(data), skiprows=skip, usecols=(0, 1))
    return arr[:, 0], arr[:, 1]


def _sorted(w: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(w)
    w, f = w[order], f[order]
    keep = np.concatenate([[True], np.diff(w) > 0])
    return w[keep], f[keep]


def read_bobcat(data: bytes) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Sonora Bobcat: two header lines; wavelength [μm] (descending), F_ν [erg/cm²/s/Hz]"""
    um, fnu = _text_columns(data, 2)
    w = um * 1e4
    return (*_sorted(w, fnu * _C_ANGSTROM / w ** 2), None)


def read_diamondback(data: bytes) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Sonora Diamondback: three header lines; wavelength [μm], F_λ [W/m²/m]"""
    um, flam = _text_columns(data, 3)
    return (*_sorted(um * 1e4, flam * 1e-7), None)


def read_elfowl(data: bytes) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Sonora Elf Owl: netCDF-4 (HDF5) with 'wavelength' [μm] and 'flux' [erg/s/cm²/cm]"""
    import h5py
    with h5py.File(io.BytesIO(data), 'r') as h:
        um = np.asarray(h['wavelength'][()], dtype=float)
        flux = np.asarray(h['flux'][()], dtype=float)
    return (*_sorted(um * 1e4, flux * 1e-8), None)


def read_sand(data: bytes) -> tuple[np.ndarray, np.ndarray, str | None]:
    """SAND: FITS, extension 1 'Wavelength' [Å] and 'Flux density' [erg/s/cm²/Å]; AIRORVAC in the primary header"""
    from astropy.io import fits
    with fits.open(io.BytesIO(data)) as h:
        med = str(h[0].header.get('AIRORVAC', '')).strip().lower()
        d = h[1].data
        w = np.asarray(d['Wavelength'], dtype=float)
        f = np.asarray(d['Flux density'], dtype=float)
    medium = 'vacuum' if med.startswith('vac') else 'air' if med.startswith('air') else None
    return (*_sorted(w, f), medium)


# ---------------------------------------------------------------------------------------------------------------
# member-name parsers → parameters
# ---------------------------------------------------------------------------------------------------------------
def _logg_from_mks(g: str) -> float:
    return float(np.round(np.log10(float(g)) + 2., 2))


def parse_bobcat(name: str) -> dict | None:
    m = re.match(r'^sp_t(?P<teff>\d+)g(?P<g>\d+)nc_m(?P<met>[-+]?\d+\.\d+)(?:_co(?P<co>[\d.]+))?$', name)
    if m is None:
        return None
    return dict(teff=float(m['teff']), logg=_logg_from_mks(m['g']), met=float(m['met']), alpha=0.,
                co=float(m['co']) if m['co'] else 1.0)


def parse_diamondback(name: str) -> dict | None:
    m = re.match(r'^t(?P<teff>\d+)g(?P<g>\d+)(?:f(?P<fsed>\d+)|nc)_m(?P<met>[-+]?\d+\.\d+)_co(?P<co>[\d.]+)\.spec$',
                 name)
    if m is None:
        return None
    # fsed = 0 encodes the cloud-free ("nc") models
    return dict(teff=float(m['teff']), logg=_logg_from_mks(m['g']), met=float(m['met']), alpha=0.,
                fsed=float(m['fsed']) if m['fsed'] else 0.)


def parse_elfowl(name: str) -> dict | None:
    m = re.match(r'^spectra_logzz_(?P<kzz>[\d.]+)_teff_(?P<teff>[\d.]+)_grav_(?P<g>[\d.]+)_mh_(?P<met>[-+\d.]+)'
                 r'_co_(?P<co>[\d.]+)\.nc$', name)
    if m is None:
        return None
    return dict(teff=float(m['teff']), logg=_logg_from_mks(m['g']), met=float(m['met']), alpha=0.,
                co=float(m['co']), kzz=float(m['kzz']))


def parse_sand(name: str) -> dict | None:
    m = re.match(r'^SAND_z(?P<met>[-+]?[\d.]+)_a(?P<alpha>[-+]?[\d.]+)_t(?P<teff>\d+)_g(?P<logg>[\d.]+)_(?P<conv>\w+)'
                 r'\.fits$', name)
    if m is None:
        return None
    return dict(teff=float(m['teff']), logg=float(m['logg']), met=float(m['met']), alpha=float(m['alpha']))


# ---------------------------------------------------------------------------------------------------------------
# record-file selection
# ---------------------------------------------------------------------------------------------------------------
def _within(v: float, rng: tuple[float, float] | None, tol: float = 1e-6) -> bool:
    return rng is None or rng[0] - tol <= v <= rng[1] + tol


def files_bobcat(key: str, want: Mapping[str, tuple[float, float]]) -> bool:
    m = re.match(r'^spectra_m(?P<met>[-+]\d\.\d)(?:_co(?P<co>[\d.]+)_g1000nc)?\.tar\.gz$', key)
    if m is None:
        return False
    co = float(m['co']) if m['co'] else 1.0
    return _within(float(m['met']), want.get('met')) and _within(co, want.get('co', (1.0, 1.0)))


def files_elfowl(key: str, want: Mapping[str, tuple[float, float]]) -> bool:
    m = re.match(r'^teff_(?P<lo>\d+)_(?P<hi>\d+)\.tar\.gz$', key)
    if m is None:
        return False
    lo, hi = want.get('teff', (0., np.inf))
    return float(m['lo']) <= hi and float(m['hi']) >= lo


def _single(name: str) -> Callable[[str, Mapping], bool]:
    return lambda key, want: key == name


@dataclass(frozen=True)
class ZenodoGrid:
    """How to fetch one grid from Zenodo"""
    records: tuple[str, ...]  # Zenodo record ids (verified live)
    select_file: Callable[[str, Mapping[str, tuple[float, float]]], bool]
    parse: Callable[[str], dict | None]
    read: Callable[[bytes], tuple[np.ndarray, np.ndarray, str | None]]
    member_filter: Callable[[str], bool] = staticmethod(lambda path: True)
    doi: str = ''
    citation: str = ''  # what to cite for the models (the record is CC-BY-4.0)


# record ids verified against the live Zenodo API on 2026-10-27 (titles, CC-BY-4.0 licences and file lists)
ZENODO_GRIDS: dict[str, ZenodoGrid] = {
    'sonora-bobcat': ZenodoGrid(('5063476',), files_bobcat, parse_bobcat, read_bobcat,
                                doi='10.5281/zenodo.5063476',
                                citation='Marley et al. 2021, ApJ 920, 85 (Sonora Bobcat); Zenodo record 5063476'),
    'sonora-diamondback': ZenodoGrid(('12735103',), _single('spectra.zip'), parse_diamondback, read_diamondback,
                                     doi='10.5281/zenodo.12735103',
                                     citation='Morley et al. 2024, ApJ 975, 59 (Sonora Diamondback); '
                                              'Zenodo record 12735103'),
    'sonora-elfowl': ZenodoGrid(('15150865', '15150874', '15150881'), files_elfowl, parse_elfowl, read_elfowl,
                                doi='10.5281/zenodo.15150865 / 15150874 / 15150881',
                                citation='Mukherjee et al. 2024, ApJ 963, 73 (Sonora Elf Owl); Zenodo records '
                                         '15150865, 15150874, 15150881'),
    # SAND: only the "recommended" models (continuous synthetic photometry; see the record's README)
    'sand': ZenodoGrid(('11582126',), _single('SAND.zip'), parse_sand, read_sand,
                       member_filter=staticmethod(lambda path: '/recommended/' in f'/{path}'),
                       doi='10.5281/zenodo.11582126',
                       citation='Alvarado et al. 2024, RNAAS 8, 134 (SAND; doi:10.3847/2515-5172/ad4bd7); '
                                'Zenodo record 11582126'),
}


def provenance(grid: str) -> dict[str, str]:
    """The licence, Zenodo record id(s), DOI and citation string of a grid, for its cache metadata and manifest"""
    spec = ZENODO_GRIDS[grid]
    return dict(license=LICENCE, record=' / '.join(spec.records), doi=spec.doi, citation=spec.citation)


def record_files(record: str) -> list[dict]:
    """The files of a Zenodo record: [{'key', 'size', 'checksum' ('md5:...'), 'url'}]"""
    d = _fetch_json(f'{ZENODO_API}{record}')
    return [dict(key=f['key'], size=int(f['size']), checksum=f['checksum'], url=f['links']['self'])
            for f in d['files']]


# ---------------------------------------------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------------------------------------------
class _HashingReader(io.RawIOBase):
    """Wraps a stream and accumulates its MD5"""

    def __init__(self, raw):
        self.raw, self.md5, self.n = raw, hashlib.md5(), 0  # noqa: S324 (Zenodo publishes MD5 checksums)

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        data = self.raw.read(len(b))
        b[:len(data)] = data
        self.md5.update(data)
        self.n += len(data)
        return len(data)

    def drain(self) -> None:
        while self.raw.read(1 << 20):
            pass


def _safe_name(path: str) -> str | None:
    """The member's base name if its path is safe (relative, no '..'), else None"""
    from .archives import _check_name
    try:
        _check_name(path)
    except Exception:  # noqa: BLE001 - unsafe names are skipped, with a warning
        log.warning('Skipping unsafe archive member name %r', path)
        return None
    return PurePosixPath(path).name


def _wanted(params: dict, want: Mapping[str, tuple[float, float]], axes: tuple[str, ...]) -> bool:
    return all(_within(params.get(a, np.nan), rng) for a, rng in want.items() if a in axes and a in params)


class _Staging:
    """
    The files written by one download call. Models are written to '<name>.part' and renamed into place by
    `commit` once verified; `rollback` removes every staged file and every committed file that did not exist
    before this call (a cached model that was overwritten by a verified copy is kept).
    """

    def __init__(self):
        self.staged: list[Path] = []
        self.committed: list[Path] = []

    @staticmethod
    def part(path: Path) -> Path:
        return path.with_name(path.name + '.part')

    def add(self, path: Path) -> None:
        self.staged.append(path)

    def commit(self, paths: Sequence[Path]) -> None:
        for p in paths:
            existed = p.exists()
            os.replace(self.part(p), p)
            if not existed:
                self.committed.append(p)
        self.staged = [p for p in self.staged if p not in paths]

    def rollback(self) -> None:
        for p in self.staged:
            self.part(p).unlink(missing_ok=True)
        for p in self.committed:
            p.unlink(missing_ok=True)
        self.staged, self.committed = [], []


def download_zenodo(grid: str, want: Mapping[str, tuple[float, float]], directory: Path, *, axes: tuple[str, ...],
                    write: Callable[[dict, np.ndarray, np.ndarray, str | None, dict], Path]) -> list[Path]:
    """
    Fetch the models of `grid` within `want` from Zenodo and write them with `write`

    Each model is written by `write` to the temporary name ``<path>.part`` (see `_Staging.part`) and renamed to
    ``<path>`` only once its checksum has passed (MD5 of the whole tar.gz stream, CRC-32 of each zip member). If
    anything fails (after the retries of transient errors), every file written by this call is removed and the
    exception is re-raised.

    Parameters
    ----------
    grid
        A key of `ZENODO_GRIDS`
    want
        {axis: (min, max)} ranges (axes absent from a model's parameters are ignored)
    directory
        Output directory (used by `write`)
    axes
        The grid's axes (for range checks)
    write
        Callback (params, wave [Å], F_λ, medium or None, provenance) → final path; it must write the data to
        ``<final path>.part`` and not to the final path itself

    Returns
    -------
    paths
        The written files
    """
    spec = ZENODO_GRIDS[grid]
    stage = _Staging()
    written: list[Path] = []
    try:
        for record in spec.records:
            for f in record_files(record):
                if not spec.select_file(f['key'], want):
                    continue
                log.info('Zenodo %s: %s (%.0f MB)', record, f['key'], f['size'] / 1e6)
                prov = dict(source='zenodo', record=record, file=f['key'], checksum=f['checksum'],
                            license=LICENCE, doi=f'10.5281/zenodo.{record}')
                if f['key'].endswith('.zip'):
                    # range reads retry individually
                    written += _from_zip(spec, f, want, axes, write, prov, stage)
                elif f['key'].endswith(('.tar.gz', '.tgz')):
                    # a dropped stream restarts the file
                    written += _retry(lambda f=f, prov=prov: _from_targz(spec, f, want, axes, write, prov, stage),
                                      f'Streaming {f["key"]}')
                else:
                    log.warning('Zenodo %s: unsupported file type %s', record, f['key'])
        if not written:
            raise FileNotFoundError(f'no {grid} models on Zenodo within {dict(want)}')
    except BaseException:
        stage.rollback()
        raise
    return written


def _from_zip(spec, f, want, axes, write, prov, stage) -> list[Path]:
    out = []
    with _open_zip(f['url'], f['size']) as z:
        for info in z.infolist():
            if info.is_dir() or not spec.member_filter(info.filename):
                continue
            name = _safe_name(info.filename)
            params = spec.parse(name) if name else None
            if params is None or not _wanted(params, want, axes):
                continue
            if info.file_size > MAX_MEMBER_BYTES:
                log.warning('Skipping %s: %d bytes exceeds the member limit', info.filename, info.file_size)
                continue
            data = z.read(info)  # zipfile checks the member's CRC-32; repeated here so that it is explicit
            if zlib.crc32(data) != info.CRC:
                raise OSError(f'CRC-32 mismatch for {info.filename} in {f["key"]}')
            w, flam, medium = spec.read(data)
            path = write(params, w, flam, medium, {**prov, 'member': info.filename, 'crc32': f'{info.CRC:08x}'})
            stage.add(path)
            stage.commit([path])  # the member has been verified
            out.append(path)
    return out


def _from_targz(spec, f, want, axes, write, prov, stage) -> list[Path]:
    out = []
    mark = len(stage.staged)
    raw = _open_stream(f['url'])
    reader = _HashingReader(raw)
    try:
        with tarfile.open(fileobj=io.BufferedReader(reader, buffer_size=1 << 20), mode='r|gz') as tar:
            for member in tar:
                if not member.isfile() or not spec.member_filter(member.name):
                    continue
                name = _safe_name(member.name)
                params = spec.parse(name) if name else None
                if params is None or not _wanted(params, want, axes):
                    continue
                if member.size > MAX_MEMBER_BYTES:
                    log.warning('Skipping %s: %d bytes exceeds the member limit', member.name, member.size)
                    continue
                data = tar.extractfile(member).read()
                w, flam, medium = spec.read(data)
                path = write(params, w, flam, medium, {**prov, 'member': member.name})
                stage.add(path)
                out.append(path)
        reader.drain()
    except BaseException:
        _discard(stage, mark)
        raise
    finally:
        raw.close()
    expected = f['checksum'].split(':', 1)[-1]
    if reader.md5.hexdigest() != expected:
        _discard(stage, mark)
        raise OSError(f'MD5 mismatch for {f["key"]} (Zenodo {expected}, got {reader.md5.hexdigest()}); the '
                      f'{len(out)} converted file(s) were deleted')
    stage.commit(out)  # only now do the models appear under their final names
    log.info('%s: MD5 verified (%s), %d model(s) converted', f['key'], expected, len(out))
    return out


def _discard(stage: _Staging, mark: int) -> None:
    """Remove the files staged since `mark` (one failed attempt at one archive)"""
    for p in stage.staged[mark:]:
        stage.part(p).unlink(missing_ok=True)
    del stage.staged[mark:]
