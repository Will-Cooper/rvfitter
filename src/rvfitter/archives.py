"""
Safe extraction of zip and tar archives into private temporary directories, and grouping of spectra into objects.

Inputs may be a single file, a directory (scanned recursively) or a ``.zip`` / ``.tar`` / ``.tar.gz`` / ``.tgz`` /
``.tar.bz2`` / ``.tar.xz`` archive.
Archives are extracted with the standard library into a `tempfile.TemporaryDirectory` that lives as long as the
`open_inputs` context.
Extraction rejects absolute paths, ``..`` components, symbolic and hard links and device/FIFO members, and caps the
total uncompressed size and the number of members (so a decompression bomb cannot fill the disk).
"""
import contextlib
import logging
import re
import stat
import tarfile
import tempfile
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath

from .types import Spectrum

__all__ = ['UnsafeArchiveError', 'ArchiveLimitError', 'SPECTRUM_SUFFIXES', 'DEFAULT_MAX_BYTES', 'DEFAULT_MAX_MEMBERS',
           'is_archive', 'open_inputs', 'collect_inputs', 'load_spectra', 'group_by_object', 'normalise_name']

log = logging.getLogger(__name__)

SPECTRUM_SUFFIXES = ('.fits', '.fit', '.fits.gz', '.txt', '.dat', '.csv', '.tsv', '.ascii')
DEFAULT_MAX_BYTES = 2 * 1024 ** 3  # 2 GB uncompressed
DEFAULT_MAX_MEMBERS = 10_000
_CHUNK = 1024 * 1024
_ARCHIVE_SUFFIXES = ('.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tbz', '.tar.xz', '.txz')

# kept alive until interpreter exit when collect_inputs() has to extract an archive
_PERSISTENT: list[tempfile.TemporaryDirectory] = []


class UnsafeArchiveError(ValueError):
    """The archive contains a member that is not safe to extract"""


class ArchiveLimitError(UnsafeArchiveError):
    """The archive exceeds the size or member-count limit"""


def _has_suffix(name: str, suffixes: tuple[str, ...]) -> bool:
    return name.lower().endswith(suffixes)


def is_archive(path: str | Path) -> bool:
    """Whether ``path`` is a zip or tar archive (by name, falling back to the content for unknown suffixes)"""
    path = Path(path)
    if not path.is_file():
        return False
    if _has_suffix(path.name, _ARCHIVE_SUFFIXES):
        return True
    if _has_suffix(path.name, SPECTRUM_SUFFIXES):
        return False
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


def _check_name(name: str) -> PurePosixPath:
    """Validate an archive member name and return it as a relative POSIX path"""
    if not name or '\x00' in name:
        raise UnsafeArchiveError(f'unsafe member name {name!r}')
    norm = name.replace('\\', '/')
    if norm.startswith('/') or re.match(r'^[A-Za-z]:', norm):
        raise UnsafeArchiveError(f'absolute path in archive: {name!r}')
    parts = PurePosixPath(norm).parts
    if '..' in parts:
        raise UnsafeArchiveError(f"'..' in archive member path: {name!r}")
    return PurePosixPath(*parts)


def _dest_path(dest: Path, rel: PurePosixPath) -> Path:
    target = (dest / rel).resolve()
    if not target.is_relative_to(dest.resolve()):
        raise UnsafeArchiveError(f'member escapes the extraction directory: {rel}')
    return target


class _Budget:
    def __init__(self, max_bytes: int, max_members: int):
        self.max_bytes, self.max_members = max_bytes, max_members
        self.bytes = 0
        self.members = 0

    def member(self, name: str) -> None:
        self.members += 1
        if self.members > self.max_members:
            raise ArchiveLimitError(f'archive has more than {self.max_members} members (at {name!r})')

    def add(self, n: int) -> None:
        self.bytes += n
        if self.bytes > self.max_bytes:
            raise ArchiveLimitError(f'archive expands to more than {self.max_bytes / 1024 ** 3:.2f} GB')


def _extract_zip(path: Path, dest: Path, budget: _Budget) -> None:
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            budget.member(info.filename)
            rel = _check_name(info.filename)
            mode = info.external_attr >> 16
            if mode:
                kind = stat.S_IFMT(mode)
                if kind == stat.S_IFLNK:
                    raise UnsafeArchiveError(f'symbolic link in archive: {info.filename!r}')
                if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise UnsafeArchiveError(f'special file in archive: {info.filename!r}')
            if info.file_size > budget.max_bytes - budget.bytes:  # cheap check before reading anything
                raise ArchiveLimitError(f'archive expands to more than {budget.max_bytes / 1024 ** 3:.2f} GB')
            target = _dest_path(dest, rel)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, 'wb') as out:
                while chunk := src.read(_CHUNK):  # count real bytes: the declared size can lie
                    budget.add(len(chunk))
                    out.write(chunk)


def _extract_tar(path: Path, dest: Path, budget: _Budget) -> None:
    # Python >= 3.11.4 has extraction filters; older versions rely on the explicit per-member checks below
    use_filter = hasattr(tarfile, 'data_filter')
    with tarfile.open(path, 'r:*') as tf:
        for member in tf:
            budget.member(member.name)
            rel = _check_name(member.name)
            if member.issym() or member.islnk():
                raise UnsafeArchiveError(f'link in archive: {member.name!r}')
            if not (member.isreg() or member.isdir()):
                raise UnsafeArchiveError(f'special file in archive: {member.name!r}')
            _dest_path(dest, rel)  # resolved path must stay inside dest
            if member.isreg():
                budget.add(member.size)  # tar sizes are what is actually read from the stream
            if use_filter:
                tf.extract(member, dest, filter='data')
            else:
                member.mode = 0o755 if member.isdir() else 0o644
                member.uid = member.gid = 0
                tf.extract(member, dest)


def _extract(path: Path, dest: Path, max_bytes: int, max_members: int) -> None:
    budget = _Budget(max_bytes, max_members)
    if zipfile.is_zipfile(path):
        _extract_zip(path, dest, budget)
    elif tarfile.is_tarfile(path):
        _extract_tar(path, dest, budget)
    else:
        raise UnsafeArchiveError(f'{path} is not a readable zip or tar archive')
    log.debug('extracted %d members (%.1f MB) from %s', budget.members, budget.bytes / 1e6, path.name)


def _scan(root: Path, suffixes: tuple[str, ...] = SPECTRUM_SUFFIXES) -> list[Path]:
    """Spectrum-like files under ``root`` (recursive), skipping anything hidden or in ``__MACOSX``"""
    files = []
    for p in sorted(root.rglob('*')):
        rel_parts = p.relative_to(root).parts
        if any(part.startswith('.') or part == '__MACOSX' for part in rel_parts):
            continue
        if p.is_file() and _has_suffix(p.name, suffixes):
            files.append(p)
    return files


@contextlib.contextmanager
def open_inputs(path: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES,
                max_members: int = DEFAULT_MAX_MEMBERS) -> Iterator[list[Path]]:
    """
    Context manager yielding the spectrum files in ``path``.

    Parameters
    ----------
    path
        A single file (returned as is, whatever its extension), a directory (scanned recursively), or a
        zip/tar(.gz/.bz2/.xz) archive (extracted to a temporary directory that is deleted on exit)
    max_bytes
        Cap on the total uncompressed size of an archive [bytes]
    max_members
        Cap on the number of archive members

    Yields
    ------
    files
        Sorted paths of the spectrum files (``.fits``, ``.fit``, ``.fits.gz``, ``.txt``, ``.dat``, ``.csv``, ``.tsv``,
        ``.ascii``; hidden files are skipped)

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist
    UnsafeArchiveError
        If the archive holds absolute or ``..`` paths, links or special files, or exceeds the limits
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Input not found: {path}')
    if path.is_dir():
        yield _scan(path)
    elif is_archive(path):
        with tempfile.TemporaryDirectory(prefix='rvfitter-') as tmp:
            dest = Path(tmp)
            _extract(path, dest, max_bytes, max_members)
            files = _scan(dest)
            log.info('%s: %d spectrum file(s) in archive', path.name, len(files))
            yield files
    else:
        yield [path]


def collect_inputs(path: str | Path, *, max_bytes: int = DEFAULT_MAX_BYTES,
                   max_members: int = DEFAULT_MAX_MEMBERS) -> list[Path]:
    """
    List the spectrum files in ``path`` (a file, a directory or an archive).

    For an archive the files live in a temporary directory that persists until the interpreter exits;
    prefer `open_inputs` when the files should be removed as soon as they have been read.
    See `open_inputs` for the parameters and exceptions.
    """
    path = Path(path)
    if path.exists() and not path.is_dir() and is_archive(path):
        tmp = tempfile.TemporaryDirectory(prefix='rvfitter-')
        try:
            _extract(path, Path(tmp.name), max_bytes, max_members)
        except BaseException:
            tmp.cleanup()
            raise
        _PERSISTENT.append(tmp)  # its finalizer deletes the directory at exit
        return _scan(Path(tmp.name))
    with open_inputs(path, max_bytes=max_bytes, max_members=max_members) as files:
        return files


def load_spectra(files: Sequence[Path], **read_kwargs) -> list[Spectrum]:
    """
    Read every file with `rvfitter.io.read_spectrum`; unreadable files are logged and skipped.

    Parameters
    ----------
    files
        Paths to read
    **read_kwargs
        Passed to `read_spectrum` (``medium``, ``hdu``, ``columns``, ``resolution``)

    Raises
    ------
    ValueError
        If ``files`` is non-empty but none could be read (the first error is quoted)
    """
    from .io import read_spectrum  # local import keeps archives importable without astropy-heavy io at import time
    spectra, first_error = [], None
    for f in files:
        try:
            spectra.append(read_spectrum(f, **read_kwargs))
        except Exception as exc:
            first_error = first_error or f'{f}: {exc}'
            log.warning('Skipping unreadable file %s: %s', f, exc)
    if files and not spectra:
        raise ValueError(f'None of the {len(files)} input file(s) could be read; first error: {first_error}')
    return spectra


def normalise_name(name: object) -> str:
    """Strip an object name and collapse runs of whitespace to single spaces"""
    return re.sub(r'\s+', ' ', str(name)).strip()


def _stem(source: str) -> str:
    name = Path(source).name
    for suffix in sorted(SPECTRUM_SUFFIXES, key=len, reverse=True):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return Path(name).stem


def group_by_object(spectra: Sequence[Spectrum], override: Mapping[str, str] | None = None
                    ) -> dict[str, list[Spectrum]]:
    """
    Group spectra into objects.

    The object name is ``meta['object']`` (from OBJECT/TARGNAME/TARGET), else the file stem of ``source``.
    Names are normalised (stripped, whitespace collapsed).

    Parameters
    ----------
    spectra
        Spectra to group
    override
        Optional mapping to a group name, looked up by the spectrum's source path, file name, file stem and then
        its header object name; the first hit wins

    Returns
    -------
    groups
        Object name → spectra, in order of first appearance
    """
    override = {str(k): normalise_name(v) for k, v in (override or {}).items()}
    groups: dict[str, list[Spectrum]] = {}
    for spec in spectra:
        obj = normalise_name(spec.meta.get('object') or '')
        stem = _stem(spec.source) if spec.source else ''
        keys = (spec.source, Path(spec.source).name if spec.source else '', stem, obj)
        name = next((override[k] for k in keys if k and k in override), None)
        name = name or obj or stem or 'unnamed'
        groups.setdefault(name, []).append(spec)
    return groups


