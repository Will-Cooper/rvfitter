"""Tests for rvfitter.archives: inputs from files, directories and archives, hostile archives, grouping."""
import io as _io
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from rvfitter import archives
from rvfitter.archives import ArchiveLimitError, UnsafeArchiveError, group_by_object, open_inputs
from rvfitter.types import Spectrum

TXT = '# AIRORVAC = vac\n# OBJECT = Star  A\n' + '\n'.join(f'{7000 + i} {1 + 0.01 * i}' for i in range(20)) + '\n'


def _tree(root: Path) -> Path:
    (root / 'sub').mkdir(parents=True)
    (root / 'a.txt').write_text(TXT)
    (root / 'sub' / 'b.csv').write_text(TXT)
    (root / 'sub' / 'notes.pdf').write_text('no')
    (root / '.hidden.txt').write_text(TXT)
    (root / '__MACOSX').mkdir()
    (root / '__MACOSX' / 'c.txt').write_text(TXT)
    return root


def _names(files):
    return sorted(f.name for f in files)


def test_single_file_any_extension(tmp_path):
    f = tmp_path / 'spec.weird'
    f.write_text(TXT)
    with open_inputs(f) as files:
        assert files == [f]
    assert archives.collect_inputs(f) == [f]


def test_directory_recursive_skips_hidden_and_other(tmp_path):
    root = _tree(tmp_path / 'd')
    with open_inputs(root) as files:
        assert _names(files) == ['a.txt', 'b.csv']
    assert _names(archives.collect_inputs(root)) == ['a.txt', 'b.csv']


def test_zip(tmp_path):
    root = _tree(tmp_path / 'd')
    zpath = tmp_path / 'x.zip'
    with zipfile.ZipFile(zpath, 'w') as zf:
        for p in root.rglob('*'):
            zf.write(p, p.relative_to(root))
    with open_inputs(zpath) as files:
        assert _names(files) == ['a.txt', 'b.csv']
        kept = files[0].parent
        assert files[0].read_text() == TXT
    assert not kept.exists()  # temporary directory removed on exit


def test_tar_gz_and_variants(tmp_path):
    root = _tree(tmp_path / 'd')
    for suffix, mode in (('.tar.gz', 'w:gz'), ('.tar.bz2', 'w:bz2'), ('.tar.xz', 'w:xz'), ('.tar', 'w')):
        tpath = tmp_path / f'x{suffix}'
        with tarfile.open(tpath, mode) as tf:
            tf.add(root, arcname='.')
        with open_inputs(tpath) as files:
            assert _names(files) == ['a.txt', 'b.csv']


def test_collect_inputs_archive_persists(tmp_path):
    root = _tree(tmp_path / 'd')
    tpath = tmp_path / 'x.tgz'
    with tarfile.open(tpath, 'w:gz') as tf:
        tf.add(root, arcname='d')
    files = archives.collect_inputs(tpath)
    assert _names(files) == ['a.txt', 'b.csv'] and all(f.exists() for f in files)


def test_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError), open_inputs(tmp_path / 'nope.zip'):
        pass


def _zip_with(path, name, data=b'x', symlink=False):
    with zipfile.ZipFile(path, 'w') as zf:
        info = zipfile.ZipInfo(name)
        if symlink:
            info.external_attr = (0o120777) << 16
        zf.writestr(info, data)
    return path


@pytest.mark.parametrize('name', ['../evil.txt', 'a/../../evil.txt', '/etc/evil.txt', 'C:/evil.txt', '..\\evil.txt'])
def test_zip_rejects_bad_names(tmp_path, name):
    z = _zip_with(tmp_path / 'bad.zip', name)
    with pytest.raises(UnsafeArchiveError), open_inputs(z):
        pass
    assert not (tmp_path / 'evil.txt').exists()


def test_zip_rejects_symlink(tmp_path):
    z = _zip_with(tmp_path / 'bad.zip', 'link.txt', b'/etc/passwd', symlink=True)
    with pytest.raises(UnsafeArchiveError, match='symbolic'), open_inputs(z):
        pass


def _tar_with(path, member: tarfile.TarInfo, data: bytes | None = None):
    with tarfile.open(path, 'w:gz') as tf:
        tf.addfile(member, None if data is None else _io.BytesIO(data))
    return path


@pytest.mark.parametrize('name', ['../evil.txt', '/tmp/evil.txt', 'x/../../evil.txt'])
def test_tar_rejects_bad_names(tmp_path, name):
    m = tarfile.TarInfo(name)
    m.size = 1
    t = _tar_with(tmp_path / 'bad.tar.gz', m, b'x')
    with pytest.raises(UnsafeArchiveError), open_inputs(t):
        pass
    assert not (tmp_path / 'evil.txt').exists()


@pytest.mark.parametrize('kind', [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE])
def test_tar_rejects_links_and_devices(tmp_path, kind):
    m = tarfile.TarInfo('thing.txt')
    m.type = kind
    m.linkname = '/etc/passwd'
    t = _tar_with(tmp_path / 'bad.tar.gz', m)
    with pytest.raises(UnsafeArchiveError), open_inputs(t):
        pass


def test_oversize_total(tmp_path):
    z = tmp_path / 'big.zip'
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('a.txt', b'0' * 100_000)
        zf.writestr('b.txt', b'0' * 100_000)
    with pytest.raises(ArchiveLimitError), open_inputs(z, max_bytes=150_000):
        pass
    with open_inputs(z, max_bytes=250_000) as files:
        assert len(files) == 2
    t = tmp_path / 'big.tar.gz'
    with tarfile.open(t, 'w:gz') as tf:
        for n in 'ab':
            m = tarfile.TarInfo(f'{n}.txt')
            m.size = 100_000
            tf.addfile(m, _io.BytesIO(b'0' * 100_000))
    with pytest.raises(ArchiveLimitError), open_inputs(t, max_bytes=150_000):
        pass


def test_too_many_members(tmp_path):
    z = tmp_path / 'many.zip'
    with zipfile.ZipFile(z, 'w') as zf:
        for i in range(6):
            zf.writestr(f'f{i}.txt', 'x')
    with pytest.raises(ArchiveLimitError, match='members'), open_inputs(z, max_members=5):
        pass


def test_load_spectra_skips_bad(tmp_path, caplog):
    (tmp_path / 'good.txt').write_text(TXT)
    (tmp_path / 'bad.txt').write_text('nothing useful\n')
    with open_inputs(tmp_path) as files:
        specs = archives.load_spectra(files)
    assert [Path(s.source).name for s in specs] == ['good.txt']
    assert 'Skipping unreadable' in caplog.text
    with pytest.raises(ValueError, match='None of the 1'):
        archives.load_spectra([tmp_path / 'bad.txt'])


def _spec(source, obj=None):
    w = np.linspace(7000, 7100, 5)
    return Spectrum(w, np.ones(5), np.ones(5), None, {'object': obj} if obj else {}, source)


def test_group_by_object():
    specs = [_spec('/x/a.fits', ' Star   A '), _spec('/x/b.fits', 'Star A'), _spec('/x/c.fits.gz'),
             _spec('/x/d.txt', 'Other')]
    groups = group_by_object(specs)
    assert list(groups) == ['Star A', 'c', 'Other']
    assert len(groups['Star A']) == 2
    over = group_by_object(specs, override={'c': 'Star A', 'Other': 'Star A'})
    assert list(over) == ['Star A'] and len(over['Star A']) == 4
    over2 = group_by_object(specs, override={'/x/a.fits': 'solo  one'})
    assert 'solo one' in over2 and len(over2['Star A']) == 1
