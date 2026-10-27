"""Tests of the Zenodo grid source (mocked archives; one live test marked network + slow)."""
import hashlib
import io
import logging
import socket
import tarfile
import zipfile

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
    import json
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
