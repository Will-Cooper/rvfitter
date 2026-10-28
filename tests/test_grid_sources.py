"""Tests of the Zenodo grid source (mocked archives; one live test marked network + slow)."""
import hashlib
import io
import json
import logging
import socket
import tarfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from dataclasses import replace

import numpy as np
import pytest

from rvfitter import grid_sources as G
from rvfitter import templates as T
from rvfitter.medium import vac_to_air


def _bobcat_text(teff, g, met='+0.0', co=None):
    um = np.linspace(1.2, 0.6, 20000)  # descending, as in the files
    w = um * 1e4
    flam = np.ones_like(w) * (teff / 1000.)
    for _, lam in T.medium_check_lines():
        flam *= 1. - 0.5 * np.exp(-0.5 * ((w - lam) / 0.8) ** 2)  # vacuum lines
    fnu = flam * w ** 2 / 2.99792458e18
    head = (f' {teff:.0f}.  {g}.  0.28 0.00 1.0E+05 0.00 0.50 1.00  Teff, grav (MKS), ...\n'
            '    microns  Flux (erg/cm^2/s/Hz)\n')
    body = '\n'.join(f'{a:.7f} {b:.5E}' for a, b in zip(um, fnu, strict=True))
    name = f'sp_t{teff:.0f}g{g}nc_m{met}' + (f'_co{co}' if co else '')
    return name, (head + body + '\n').encode()


def _sand_fits(teff, logg, z, a, medium='VAC'):
    from astropy.io import fits
    w = np.linspace(6000., 12000., 30000)
    f = np.ones_like(w)
    for _, lam in T.medium_check_lines():
        c = lam if medium == 'VAC' else float(vac_to_air(lam))
        f *= 1. - 0.5 * np.exp(-0.5 * ((w - c) / 0.8) ** 2)
    prim = fits.PrimaryHDU()
    prim.header['AIRORVAC'] = medium
    tab = fits.BinTableHDU.from_columns([fits.Column('Wavelength', 'D', 'AA', array=w),
                                         fits.Column('Flux density', 'D', 'ERG/S/CM2/AA', array=f)])
    buf = io.BytesIO()
    fits.HDUList([prim, tab]).writeto(buf)
    return f'SAND_z{z}_a{a}_t{teff}_g{logg}_gold.fits', buf.getvalue()


@pytest.fixture
def fake_zenodo(monkeypatch, tmp_path):
    """Serve fake Bobcat (tar.gz) and SAND (zip) records from local files"""
    store = {}
    tar_path = tmp_path / 'spectra_m+0.0.tar.gz'
    with tarfile.open(tar_path, 'w:gz') as tar:
        for teff in (1800, 1900, 2000):
            for g in (316, 1000):
                name, data = _bobcat_text(teff, g)
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        evil = tarfile.TarInfo('../sp_t1900g1000nc_m+0.0')  # an unsafe member name: must be skipped
        evil.size = 5
        tar.addfile(evil, io.BytesIO(b'xxxxx'))
    zip_path = tmp_path / 'SAND.zip'
    with zipfile.ZipFile(zip_path, 'w') as z:
        for teff in (1900, 2000):
            name, data = _sand_fits(teff, 5.0, 0.1, 0.15)
            z.writestr(f'SAND/recommended/{name}', data)
        name, data = _sand_fits(2000, 5.0, -0.6, 0.2)
        z.writestr(f'SAND/rest/{name}', data)  # not "recommended": skipped
    for path, rec in ((tar_path, '5063476'), (zip_path, '11582126')):
        data = path.read_bytes()
        store[rec] = [dict(key=path.name, size=len(data), checksum='md5:' + hashlib.md5(data).hexdigest(),
                           url=str(path))]
    monkeypatch.setattr(G, 'record_files', lambda record: store.get(record, []))
    monkeypatch.setattr(G, '_open_stream', lambda url: open(url, 'rb'))
    monkeypatch.setattr(G, '_open_zip', lambda url, size: zipfile.ZipFile(url))
    return store


def test_bobcat_from_targz(fake_zenodo, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger='rvfitter')
    tg = T.download('sonora-bobcat', (1800., 1900.), (5.0, 5.0), directory=tmp_path / 'bob', source='zenodo')
    assert len(tg.files) == 2 and tg.bounds()['logg'] == (5.0, 5.0) and tg.meta.medium == 'vacuum'
    assert 'MD5 verified' in caplog.text and 'unsafe' in caplog.text
    spec = tg.get(1850., 5.0, wave_range=(7600., 7620.))
    assert np.all(spec.flux > 0)


def test_md5_mismatch_deletes(fake_zenodo, tmp_path):
    fake_zenodo['5063476'][0]['checksum'] = 'md5:' + '0' * 32
    with pytest.raises(OSError, match='MD5 mismatch'):
        T.download('sonora-bobcat', (1800., 1900.), (5.0, 5.0), directory=tmp_path / 'bob', source='zenodo')
    assert not list((tmp_path / 'bob').glob('t*.txt'))


def test_sand_from_zip_members(fake_zenodo, tmp_path):
    tg = T.download('sand', (1900., 2000.), (5.0, 5.0), directory=tmp_path / 'sand', source='zenodo')
    assert len(tg.files) == 2 and tg.meta.medium == 'vacuum'
    assert tg.bounds()['met'] == (0.1, 0.1) and tg.bounds()['alpha'] == (0.15, 0.15)
    man = json.loads((tmp_path / 'sand' / 'manifest.json').read_text())
    assert all(v['source'] == 'zenodo' and v['record'] == '11582126' and 'crc32' in v for v in man.values())


def test_auto_falls_back_from_svo(fake_zenodo, tmp_path, caplog):
    def fail(url):
        raise OSError('HTTP Error 403: Forbidden')
    tg = T.download('sand', (1900., 2000.), (5.0, 5.0), directory=tmp_path / 'sand', fetch=fail)
    assert len(tg.files) == 2 and 'SVO download of sand failed' in caplog.text
    with pytest.raises(OSError):
        T.download('sand', (1900., 2000.), (5.0, 5.0), directory=tmp_path / 's2', fetch=fail, source='svo')


def test_no_upstream_source_message(tmp_path):
    with pytest.raises(ValueError, match='dataverse'):
        T.download('lowz', (1000., 1100.), (5.0, 5.0), directory=tmp_path / 'l', source='zenodo')


def test_parsers():
    assert G.parse_bobcat('sp_t1900g316nc_m-0.5_co1.5') == dict(teff=1900., logg=4.5, met=-0.5, alpha=0., co=1.5)
    assert G.parse_diamondback('t1500g31f3_m-0.5_co1.0.spec')['fsed'] == 3.
    assert G.parse_diamondback('t1100g31nc_m0.0_co1.0.spec')['fsed'] == 0.
    p = G.parse_elfowl('spectra_logzz_9.0_teff_450.0_grav_31.0_mh_-1.0_co_0.5.nc')
    assert p['kzz'] == 9. and p['logg'] == pytest.approx(3.49) and p['met'] == -1.
    assert G.parse_sand('SAND_z-0.15_a-0.05_t700_g4.0_inconclusive.fits')['alpha'] == -0.05


def _zenodo_online() -> bool:
    try:
        socket.create_connection(('zenodo.org', 443), timeout=5).close()
        return True
    except OSError:
        return False


@pytest.mark.slow
@pytest.mark.network
def test_live_sand_from_zenodo(tmp_path):
    if not _zenodo_online():
        pytest.skip('Zenodo not reachable')
    tg = T.download('sand', (2000., 2000.), (5.0, 5.0), directory=tmp_path / 'sand', source='zenodo')
    assert len(tg.files) >= 1 and tg.meta.medium == 'vacuum'


# ---------------------------------------------------------------------------------------------------------------
# atomic writes, retries, range requests and provenance (all mocked)
# ---------------------------------------------------------------------------------------------------------------
class _Resp:
    """A minimal urlopen response"""

    def __init__(self, body=b'', status=206, headers=None):
        self.body, self.status, self.headers = body, status, headers or {}

    def read(self, n=-1):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def sleeps(monkeypatch):
    waits = []
    monkeypatch.setattr(G, '_sleep', waits.append)
    return waits


def _no_files(directory):
    return not [p for p in directory.iterdir() if p.suffix in ('.txt', '.part')]


def test_targz_files_appear_only_after_md5(fake_zenodo, tmp_path, monkeypatch):
    seen = []
    commit = G._Staging.commit

    def spy(self, paths):
        seen.append((sorted(p.suffix for p in (tmp_path / 'bob').iterdir()), len(paths)))
        return commit(self, paths)
    monkeypatch.setattr(G._Staging, 'commit', spy)
    T.download('sonora-bobcat', (1800., 1900.), (5.0, 5.0), directory=tmp_path / 'bob', source='zenodo')
    assert seen == [(['.part', '.part'], 2)]  # staged under temporary names until the checksum had passed
    assert not list((tmp_path / 'bob').glob('*.part'))


def test_md5_mismatch_leaves_no_part_files(fake_zenodo, tmp_path):
    fake_zenodo['5063476'][0]['checksum'] = 'md5:' + '0' * 32
    (tmp_path / 'bob').mkdir()
    with pytest.raises(OSError, match='MD5 mismatch'):
        T.download('sonora-bobcat', (1800., 1900.), (5.0, 5.0), directory=tmp_path / 'bob', source='zenodo')
    assert _no_files(tmp_path / 'bob') and not (tmp_path / 'bob' / 'manifest.json').exists()


def test_dropped_stream_retried_then_cleaned(fake_zenodo, tmp_path, sleeps, monkeypatch):
    path = fake_zenodo['5063476'][0]['url']
    calls = []

    class Dropping(io.RawIOBase):
        def __init__(self):
            self.f = open(path, 'rb')

        def readable(self):
            return True

        def readinto(self, b):
            data = self.f.read(min(len(b), 100000))
            if self.f.tell() > 200000:
                raise ConnectionResetError('connection dropped')
            b[:len(data)] = data
            return len(data)

        def read(self, n=-1):
            data = self.f.read(min(n, 100000))
            if self.f.tell() > 200000:
                raise ConnectionResetError('connection dropped')
            return data
    monkeypatch.setattr(G, '_open_stream', lambda url: calls.append(1) or Dropping())
    (tmp_path / 'bob').mkdir()
    with pytest.raises(ConnectionResetError):
        T.download('sonora-bobcat', (1800., 1900.), (5.0, 5.0), directory=tmp_path / 'bob', source='zenodo')
    assert len(calls) == 3 and sleeps == [2., 4.]
    assert _no_files(tmp_path / 'bob')


def test_zip_failure_removes_verified_members(fake_zenodo, tmp_path, monkeypatch):
    spec = G.ZENODO_GRIDS['sand']
    n = []

    def flaky(data):
        n.append(1)
        if len(n) == 2:
            raise ValueError('bad member')
        return spec.read(data)
    monkeypatch.setitem(G.ZENODO_GRIDS, 'sand', replace(spec, read=flaky))
    (tmp_path / 'sand').mkdir()
    with pytest.raises(ValueError, match='bad member'):
        T.download('sand', (1900., 2000.), (5.0, 5.0), directory=tmp_path / 'sand', source='zenodo')
    assert len(n) == 2 and _no_files(tmp_path / 'sand')  # the first (already renamed) model was removed too


def test_rollback_keeps_previously_cached_models(tmp_path):
    old, new = tmp_path / 'old.txt', tmp_path / 'new.txt'
    old.write_text('cached')
    stage = G._Staging()
    for p in (old, new):
        stage.part(p).write_text('verified')
        stage.add(p)
    stage.commit([old, new])
    stage.rollback()
    assert old.read_text() == 'verified' and not new.exists()  # overwritten cache kept, new file removed


def _http(code):
    return urllib.error.HTTPError('u', code, 'x', {}, None)


def _sequence(errors):
    it = iter(errors)

    def f():
        e = next(it)
        if e is not None:
            raise e
        return 'ok'
    return f


def test_retry_transient_only(sleeps):
    assert G._retry(_sequence([_http(503), TimeoutError(), None]), 'x') == 'ok' and sleeps == [2., 4.]
    assert G._retry(_sequence([_http(429), urllib.error.URLError('refused'), None]), 'x') == 'ok'
    with pytest.raises(urllib.error.HTTPError):  # three attempts, then the last error is raised
        G._retry(_sequence([_http(500)] * 3), 'x')
    calls = []

    def not_found():
        calls.append(1)
        raise _http(404)
    with pytest.raises(urllib.error.HTTPError):
        G._retry(not_found, 'x')
    assert len(calls) == 1  # a 4xx other than 429 is final


def test_fetch_json_retries(monkeypatch, sleeps):
    n = []

    def urlopen(req, timeout):
        n.append(1)
        if len(n) < 3:
            raise urllib.error.HTTPError(req.full_url, 502, 'bad gateway', {}, None)
        return _Resp(b'{"files": []}', status=200)
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    assert G._fetch_json('https://zenodo.org/api/records/1') == {'files': []} and len(n) == 3


def test_rangefile_requires_206(monkeypatch, sleeps):
    monkeypatch.setattr(urllib.request, 'urlopen', lambda req, timeout: _Resp(b'x' * 100, status=200))
    with pytest.raises(G.RangeError, match='HTTP 200, not 206'):
        G._RangeFile('https://z/f', 100).readinto(bytearray(10))
    assert not sleeps  # not a transient error


def test_rangefile_checks_content_range(monkeypatch, sleeps):
    def serve(content_range):
        monkeypatch.setattr(urllib.request, 'urlopen', lambda req, timeout: _Resp(
            b'a' * 10, headers={'Content-Range': content_range} if content_range else {}))
        return G._RangeFile('https://z/f', 100).readinto(bytearray(10))
    assert serve('bytes 0-9/100') == 10 and serve(None) == 10
    for bad in ('bytes 5-14/100', 'bytes 0-9/999', 'garbage'):
        with pytest.raises(G.RangeError, match='Content-Range'):
            serve(bad)


def test_rangefile_retries_timeouts(monkeypatch, sleeps):
    n = []

    def urlopen(req, timeout):
        n.append(1)
        if len(n) == 1:
            raise TimeoutError('timed out')
        return _Resp(b'b' * 10, headers={'Content-Range': 'bytes 0-9/100'})
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    buf = bytearray(10)
    assert G._RangeFile('https://z/f', 100).readinto(buf) == 10 and bytes(buf) == b'b' * 10 and len(sleeps) == 1


def test_provenance_recorded(fake_zenodo, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger='rvfitter')
    tg = T.download('sand', (1900., 2000.), (5.0, 5.0), directory=tmp_path / 'sand', source='zenodo')
    man = json.loads((tmp_path / 'sand' / 'manifest.json').read_text())
    assert all(v['license'] == 'CC-BY-4.0' and v['doi'] == '10.5281/zenodo.11582126' for v in man.values())
    meta = tomllib.loads((tmp_path / 'sand' / T.METADATA_FILE).read_text())
    assert meta['license'] == 'CC-BY-4.0' and meta['record'] == '11582126'
    assert meta['doi'] == '10.5281/zenodo.11582126' and 'Alvarado' in meta['citation']
    assert tg.meta.license == 'CC-BY-4.0'
    assert 'licence CC-BY-4.0' in caplog.text and '11582126' in caplog.text and 'please cite' in caplog.text
    assert T.TemplateGrid(tmp_path / 'sand').meta.doi == '10.5281/zenodo.11582126'
