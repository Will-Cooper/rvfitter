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


pytestmark = pytest.mark.usefixtures('clear_template_cache')


def test_metadata_parse_and_roundtrip(tmp_path):
    meta = T.GridMeta('toy', 'air')
    assert meta.parse('lte020.0-5.0.txt') == (2000., 5.0, 0., 0.)
    assert meta.parse('lte015.0-4.5-0.5a+0.0.BT-Settl.spec.7.dat.txt') == (1500., 4.5, -0.5, 0.)
    assert meta.parse('readme.md') is None
    T.write_metadata(tmp_path, meta)
    assert T._read_metadata(tmp_path) == meta
    with pytest.raises(ValueError):
        T.GridMeta('toy', 'water')


def test_grid_scans_and_bounds(tmp_path):
    grid = T.TemplateGrid(_make_grid(tmp_path / 'g'))
    assert len(grid.files) == 6
    assert grid.bounds() == {'teff': (1800., 2000.), 'logg': (4.5, 5.0), 'met': (0., 0.), 'alpha': (0., 0.)}
    assert grid.select(teff=1850.) == [(t, g, 0., 0.) for t in (1800., 1900., 2000.) for g in (4.5, 5.0)]
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
    """Within one grid step of the edge: the nearest point, flagged; further out: GridCoverageError"""
    grid = T.TemplateGrid(_make_grid(tmp_path / 'g'))
    spec = grid.get(2100., 5.2, wave_range=(7600., 7620.))
    assert spec.meta['outside_grid']
    np.testing.assert_allclose(spec.flux, 2.0 + 5.0)
    assert 'outside grid' in caplog.text
    for teff, logg in ((2101., 5.0), (1600., 5.0), (2000., 5.6)):
        for method in ('linear', 'nearest'):
            with pytest.raises(T.GridCoverageError, match='grid toy covers .* requested, more than one grid step'):
                grid.get(teff, logg, method=method, wave_range=(7600., 7620.))
    with pytest.raises(T.GridCoverageError, match=r'teff 1800-2000 \(step 100 at that edge\), but teff = 1600'):
        grid.check_covers({'teff': 1600.})
    grid.check_covers({'teff': 1700., 'logg': 4.0})  # exactly one step out on each axis
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


def test_wave_range_reads_only_the_file_ends(tmp_path):
    """wave_range never loads a whole spectrum; it follows a rewritten file; a file out of λ order fails on load"""
    d = _make_grid(tmp_path / 'g')
    grid = T.TemplateGrid(d)
    misses = T._load_vacuum.cache_info().misses
    lo, hi = grid.wave_range(1900., 5.0)
    assert T._load_vacuum.cache_info().misses == misses
    wave = grid.get(1900., 5.0).wave
    np.testing.assert_allclose((lo, hi), (wave[0], wave[-1]), rtol=1e-12)
    path = d / 'lte019.0-5.0.txt'
    w, f = _model(1900., 5.0)
    np.savetxt(path, np.c_[w[:-200], f[:-200]], header='comment', footer='trailing comment')
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10**9))
    assert grid.wave_range(1900., 5.0)[1] < hi
    np.savetxt(path, np.c_[w[::-1], f[::-1]])
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 2 * 10**9))
    with pytest.raises(ValueError, match='no valid ascending wavelength range'):
        grid.wave_range(1900., 5.0)
    np.savetxt(path, np.c_[np.r_[w[:10], w[-1] + 1., w[10:]], np.r_[f[:10], 1., f[10:]]])
    with pytest.raises(ValueError, match='not in ascending wavelength order'):
        grid.get(1900., 5.0)


def _fake_ssap(rows):
    t = Table(rows=[(r['url'], r['teff'], r['logg'], r['met']) for r in rows],
              names=('Access.Reference', 'teff', 'logg', 'meta'), dtype=(str, float, float, float))
    return _votable_bytes(t)


def _fake_svo_spectrum(teff, logg):
    """A spectrum as SVO serves it: a VOTable with WAVELENGTH [Å] and FLUX"""
    w, f = _model(teff, logg)
    t = Table({'WAVELENGTH': w, 'FLUX': f})
    t['WAVELENGTH'].unit = 'Angstrom'
    return _votable_bytes(t)


def test_download_mocked_with_checksums(tmp_path):
    calls = []
    rows = [dict(url=f'http://svo/x?fid={i}', teff=t, logg=g, met=0.) for i, (t, g) in
            enumerate([(1800., 5.0), (1900., 5.0), (2000., 5.0), (2100., 5.0)])]
    bodies = {r['url']: _fake_svo_spectrum(r['teff'], r['logg']) for r in rows}

    def fetch(url):
        calls.append(url)
        return _fake_ssap(rows) if 'ssap.php' in url else bodies[url]

    grid = T.download('bt-settl-cifist', (1800., 2000.), (5.0, 5.0), directory=tmp_path / 'dl', fetch=fetch)
    assert len(grid.files) == 3 and grid.meta.medium == 'air'
    manifest = json.loads((tmp_path / 'dl' / 'manifest.json').read_text())
    assert set(manifest) == {f't{t}_g5.00_m+0.00_a+0.00.txt' for t in (1800, 1900, 2000)}
    n = len(calls)
    T.download('bt-settl-cifist', (1800., 2000.), (5.0, 5.0), directory=tmp_path / 'dl', fetch=fetch)
    assert len(calls) == n + 1  # only the query: every file's checksum matched
    (tmp_path / 'dl' / 't1900_g5.00_m+0.00_a+0.00.txt').write_text('corrupt')
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
    assert grid.bounds()['teff'] == (1500., 2500.) and grid.bounds()['logg'] == (4.5, 5.5)
    spec = grid.broadened(1950., 5.2, resolution=2500., wave_range=(7600., 8000.))
    assert spec.meta['interpolated'] and np.all(np.isfinite(spec.flux))


def _votable_bytes(table: Table) -> bytes:
    from astropy.io.votable import from_table
    buf = io.BytesIO()
    from_table(table).to_xml(buf)
    return buf.getvalue()


def test_svo_ssap_and_spectrum_formats(tmp_path):
    """SVO's live formats (2026-11): `Access.Reference` URLs, and VOTable spectra (WAVELENGTH [Å], FLUX) padded with
    zero rows, unsorted; both become what `_download_svo` and `read_two_column` expect"""
    q = Table({'teff': [2000.], 'logg': [5.], 'meta': [0.], 'Access.Reference': ['https://x/ssap.php?fid=1']})
    assert T._parse_ssap(_votable_bytes(q)) == [{'url': 'https://x/ssap.php?fid=1', 'teff': 2000., 'logg': 5.,
                                                  'met': 0.}]
    spec = Table({'WAVELENGTH': [0., 8501., 8500., 0.001], 'FLUX': [0., 2., 1., 5.]})
    spec['WAVELENGTH'].unit = 'Angstrom'
    path = tmp_path / 'svo.txt'
    path.write_bytes(T._svo_two_column(_votable_bytes(spec)))
    wave, flux = T.read_two_column(path)
    assert list(wave) == [0.001, 8500., 8501.] and list(flux) == [5., 1., 2.]
    spec['WAVELENGTH'].unit = 'nm'
    with pytest.raises(ValueError, match='unit'):
        T._svo_two_column(_votable_bytes(spec))


def _toy_cfg(directory: Path, teff_prior: float | None):
    from rvfitter import config
    cfg = config.Config()
    cfg.templates.grid = 'bt-settl-cifist'
    cfg.templates.local_dir = str(directory)
    cfg.templates.teff_prior = teff_prior
    return cfg


def test_prior_beyond_grid_fails_before_any_fit(tmp_path, monkeypatch):
    """A teff_prior more than one step outside the grid: the candidates and the coverage check raise (a cache of
    2600-4500 K used to fit every L dwarf with the 2600 K template); within one step the edge point is used"""
    from rvfitter import pipeline
    from rvfitter.methods import _common as C
    d = _make_grid(tmp_path / 'g')
    ok = _toy_cfg(d, 1700.)
    C.check_grid_coverage(ok.templates)
    edge = C.candidates(ok, {'teff_window': 50.})
    assert [c.params[:2] for c in edge] == [(1800., 5.0)]  # none in the window: the nearest point (WARNING)
    far = _toy_cfg(d, 1500.)
    with pytest.raises(T.GridCoverageError, match='teff 1800-2000 .* teff = 1500'):
        C.check_grid_coverage(far.templates)
    with pytest.raises(T.GridCoverageError):
        C.candidates(far, {'teff_window': 50.})
    C.check_grid_coverage(_toy_cfg(d, None).templates)  # no prior: nothing to check
    C.check_grid_coverage(_toy_cfg(tmp_path / 'absent', 1500.).templates)  # no grid there: reported where used
    read = []
    monkeypatch.setattr(pipeline, '_read_all', lambda *a, **k: read.append(1) or [])
    with pytest.raises(T.GridCoverageError):
        pipeline.run(far, [])
    assert not read  # nothing was read
    far.methods.enabled = ['linecentre']  # model-free, no telluric template: the grid is never read, so no check
    far.calibration.wavelength_correction = 'none'
    assert pipeline.run(far, []) == [] and read == [1]
