"""Tests of rvfitter.templates: metadata, medium verification, interpolation, broadening cache and download."""
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from rvfitter import templates as T
from rvfitter.constants import C_KMS
from rvfitter.medium import vac_to_air

TEFFS, LOGGS = (1800., 1900., 2000.), (4.5, 5.0)


def _model(teff, logg, medium='air'):
    """A toy model whose continuum level is linear in teff and logg, with the alkali lines in the given medium"""
    wave = np.arange(7500., 9100., 0.05)
    flux = np.full_like(wave, teff / 1000. + logg)
    for _, lam in T.medium_check_lines():
        c = float(vac_to_air(lam)) if medium == 'air' else lam
        flux *= 1. - 0.6 * np.exp(-0.5 * ((wave - c) / 0.4) ** 2)
    return wave, flux


def _make_grid(directory: Path, medium='air', declare=None, with_meta=True):
    directory.mkdir(parents=True, exist_ok=True)
    for t in TEFFS:
        for g in LOGGS:
            wave, flux = _model(t, g, medium)
            np.savetxt(directory / f'lte{t / 100:05.1f}-{g:.1f}.txt', np.c_[wave, flux])
    if with_meta:
        T.write_metadata(directory, T.GridMeta('toy', declare or medium))
    return directory


@pytest.fixture(autouse=True)
def _clear():
    T.clear_cache()
    yield
    T.clear_cache()


def test_metadata_parse_and_roundtrip(tmp_path):
    meta = T.GridMeta('toy', 'air')
    assert meta.parse('lte020.0-5.0.txt') == (2000., 5.0, 0.)
    assert meta.parse('lte015.0-4.5-0.5a+0.0.BT-Settl.spec.7.dat.txt') == (1500., 4.5, -0.5)
    assert meta.parse('readme.md') is None
    T.write_metadata(tmp_path, meta)
    assert T._read_metadata(tmp_path) == meta
    with pytest.raises(ValueError):
        T.GridMeta('toy', 'water')


def test_grid_scans_and_bounds(tmp_path):
    grid = T.TemplateGrid(_make_grid(tmp_path / 'g'))
    assert len(grid.files) == 6
    assert grid.bounds() == ((1800., 2000.), (4.5, 5.0), (0., 0.))
    assert grid.select(teff=1850.) == [(1800., 4.5, 0.), (1800., 5.0, 0.), (1900., 4.5, 0.), (1900., 5.0, 0.),
                                       (2000., 4.5, 0.), (2000., 5.0, 0.)]
    with pytest.raises(FileNotFoundError):
        T.TemplateGrid(_make_grid(tmp_path / 'nometa', with_meta=False))
    assert T.TemplateGrid(tmp_path / 'nometa', grid='bt-settl-cifist').meta.medium == 'air'


def test_medium_detected_and_mismatch_refused(tmp_path):
    air = T.TemplateGrid(_make_grid(tmp_path / 'air'))
    assert T.verify_medium(air)['medium'] == 'air'
    vac = T.TemplateGrid(_make_grid(tmp_path / 'vac', medium='vacuum'))
    assert T.verify_medium(vac)['medium'] == 'vacuum'
    wrong = T.TemplateGrid(_make_grid(tmp_path / 'wrong', medium='vacuum', declare='air'))
    with pytest.raises(T.MediumMismatchError):
        wrong.get(1900., 5.0)


def _a_line():
    return T.medium_check_lines()[len(T.medium_check_lines()) // 2][1]


def test_medium_check_lines_from_nist():
    lines = T.medium_check_lines()
    assert {sp for sp, _ in lines} == {'Na I', 'K I', 'Rb I', 'Cs I'} and len(lines) >= 8


def test_get_converts_to_vacuum(tmp_path):
    grid = T.TemplateGrid(_make_grid(tmp_path / 'g'))
    lam0 = _a_line()
    spec = grid.get(1900., 5.0, wave_range=(lam0 - 5, lam0 + 5))
    lam = T._line_minimum(spec.wave, spec.flux, lam0, 1.)
    np.testing.assert_allclose(lam, lam0, atol=0.01)  # back in vacuum
    assert spec.meta['medium_in'] == 'air' and not spec.meta['interpolated']


def test_trilinear_interpolation_exact_for_linear_model(tmp_path):
    grid = T.TemplateGrid(_make_grid(tmp_path / 'g'))
    spec = grid.get(1930., 4.8, wave_range=(7600., 7620.))  # line-free: flux = teff/1000 + logg
    np.testing.assert_allclose(spec.flux, 1.93 + 4.8, rtol=1e-12)
    assert spec.meta['interpolated'] and len(spec.meta['files']) == 4
    np.testing.assert_allclose(sum(spec.meta['weights']), 1.)


def test_outside_grid_uses_nearest_with_flag(tmp_path, caplog):
    grid = T.TemplateGrid(_make_grid(tmp_path / 'g'))
    spec = grid.get(2400., 5.2, wave_range=(7600., 7620.))
    assert spec.meta['outside_grid']
    np.testing.assert_allclose(spec.flux, 2.0 + 5.0)
    assert 'outside grid' in caplog.text
    near = grid.get(1930., 4.8, method='nearest', wave_range=(7600., 7620.))
    np.testing.assert_allclose(near.flux, 1.9 + 5.0)
    assert not near.meta['outside_grid']


def test_broadened_is_cached_and_correct(tmp_path):
    grid = T.TemplateGrid(_make_grid(tmp_path / 'g'))
    lam0 = _a_line()
    rng = (lam0 - 100., lam0 + 100.)
    b1 = grid.broadened(1900., 5.0, resolution=3000., wave_range=rng)
    info = T._broadened_file.cache_info()
    b2 = grid.broadened(1900., 5.0, resolution=3000., wave_range=rng)
    assert T._broadened_file.cache_info().hits == info.hits + 1
    np.testing.assert_array_equal(b1.flux, b2.flux)
    np.testing.assert_allclose(np.diff(np.log(b1.wave)) * C_KMS, C_KMS / 30000., rtol=1e-6)
    # the broadened line width matches the quadrature of the intrinsic (0.4 Å) and LSF widths
    sel = np.abs(b1.wave - lam0) < 6
    depth = 1 - b1.flux[sel] / b1.flux[sel].max()
    mu = np.sum(depth * b1.wave[sel]) / depth.sum()
    sd = np.sqrt(np.sum(depth * (b1.wave[sel] - mu) ** 2) / depth.sum())
    assert abs(mu - lam0) < 0.2
    expected = np.hypot(0.4, lam0 / 3000. / 2.3548)
    assert abs(sd - expected) / expected < 0.15  # the line is not Gaussian in depth, so loosely


def _fake_ssap(rows):
    t = Table(rows=[(r['url'], r['teff'], r['logg'], r['met']) for r in rows],
              names=('Spectrum', 'teff', 'logg', 'meta'), dtype=(str, float, float, float))
    buf = io.BytesIO()
    t.write(buf, format='votable')
    return buf.getvalue()


def test_download_mocked_with_checksums(tmp_path):
    calls = []
    rows = [dict(url=f'http://svo/x?fid={i}', teff=t, logg=g, met=0.) for i, (t, g) in
            enumerate([(1800., 5.0), (1900., 5.0), (2000., 5.0), (2100., 5.0)])]
    bodies = {r['url']: '\n'.join(f'{w} {f}' for w, f in zip(*_model(r['teff'], r['logg']), strict=True)).encode()
              for r in rows}

    def fetch(url):
        calls.append(url)
        return _fake_ssap(rows) if 'ssap.php' in url else bodies[url]

    grid = T.download('bt-settl-cifist', (1800., 2000.), (5.0, 5.0), directory=tmp_path / 'dl', fetch=fetch)
    assert len(grid.files) == 3 and grid.meta.medium == 'air'
    manifest = json.loads((tmp_path / 'dl' / 'manifest.json').read_text())
    assert set(manifest) == {'lte018.0-5.0+0.0.txt', 'lte019.0-5.0+0.0.txt', 'lte020.0-5.0+0.0.txt'}
    n = len(calls)
    T.download('bt-settl-cifist', (1800., 2000.), (5.0, 5.0), directory=tmp_path / 'dl', fetch=fetch)
    assert len(calls) == n + 1  # only the query: every file's checksum matched
    (tmp_path / 'dl' / 'lte019.0-5.0+0.0.txt').write_text('corrupt')
    T.download('bt-settl-cifist', (1800., 2000.), (5.0, 5.0), directory=tmp_path / 'dl', fetch=fetch)
    assert len(calls) == n + 3  # query + the corrupted file


def test_cli_verify(tmp_path):
    assert T.cli_main(['verify', str(_make_grid(tmp_path / 'g'))]) == 0
    assert T.cli_main(['verify', str(_make_grid(tmp_path / 'w', medium='vacuum', declare='air'))]) == 1


@pytest.mark.data
def test_bt_settl_useful_grid_is_air():
    root = os.environ.get('RVFITTER_TESTDATA')
    if not root:
        pytest.skip('RVFITTER_TESTDATA not set')
    grid = T.TemplateGrid(Path(root) / 'templates' / 'bt-settl-cifist' / 'useful', grid='bt-settl-cifist')
    res = T.verify_medium(grid)
    assert res['medium'] == 'air' and res['n_air'] >= 6
    assert np.max(np.abs(res['offsets_air'])) < 0.05
    assert grid.bounds()[:2] == ((1500., 2500.), (4.5, 5.5))
    spec = grid.broadened(1950., 5.2, resolution=2500., wave_range=(7600., 8000.))
    assert spec.meta['interpolated'] and np.all(np.isfinite(spec.flux))
