"""Unit 3b: N-dimensional grids, best-template parameters, edge guidance, results columns and refit_object."""
import socket

import numpy as np
import pytest

from rvfitter import pipeline, results
from rvfitter import templates as T
from rvfitter.medium import vac_to_air
from rvfitter.methods import _common as C
from rvfitter.types import Measurement, MethodResult, ObjectResult, Spectrum

TEFFS, LOGGS, METS, ALPHAS = (1800., 1900., 2000.), (4.5, 5.0, 5.5), (-1.0, -0.5, 0.0), (0.0, 0.2)


def _model(p, medium='air'):
    """Continuum linear in every parameter; the NIST alkali lines in the given medium, depth set by teff"""
    wave = np.arange(7500., 9100., 0.05)
    flux = np.full_like(wave, p['teff'] / 1000. + p['logg'] + p['met'] + p['alpha'] + p.get('co', 0.))
    for _, lam in T.medium_check_lines():
        c = float(vac_to_air(lam)) if medium == 'air' else lam
        flux *= 1. - (0.3 + 0.2 * (p['teff'] - 1800.) / 200.) * np.exp(-0.5 * ((wave - c) / 0.6) ** 2)
    return wave, flux


def _grid4(directory, medium='air', extra=None):
    directory.mkdir(parents=True, exist_ok=True)
    extra = extra or {}
    for t in TEFFS:
        for g in LOGGS:
            for m in METS:
                for a in ALPHAS:
                    for co in extra.get('co', [None]):
                        p = dict(teff=t, logg=g, met=m, alpha=a, **({} if co is None else {'co': co}))
                        np.savetxt(directory / T._generic_name(p), np.c_[_model(p, medium)])
    meta = T.GridMeta('toy4', medium, pattern=T.GENERIC_PATTERN, teff_scale=1.,
                      extra_axes=('co',) if extra else (), extra_defaults={'co': 0.5} if extra else {},
                      coverage={'teff': (1800., 2000.), 'logg': (4.5, 5.5), 'met': (-1., 0.), 'alpha': (0., 0.2)})
    T.write_metadata(directory, meta)
    return directory


@pytest.fixture(autouse=True)
def _clear():
    T.clear_cache()
    yield
    T.clear_cache()


def test_metadata_roundtrip_with_extra_axes(tmp_path):
    grid = T.TemplateGrid(_grid4(tmp_path / 'g', extra={'co': [0.5, 1.0]}))
    assert grid.axes == ('teff', 'logg', 'met', 'alpha', 'co')
    assert T._read_metadata(tmp_path / 'g').coverage['met'] == (-1., 0.)
    assert grid.bounds()['co'] == (0.5, 1.0)


def test_multilinear_interpolation_4d_and_extra(tmp_path):
    grid = T.TemplateGrid(_grid4(tmp_path / 'g', extra={'co': [0.5, 1.0]}))
    p = dict(teff=1930., logg=4.8, met=-0.7, alpha=0.05, co=0.6)  # 2^5 = 32 corners
    spec = grid.get(p['teff'], p['logg'], p['met'], alpha=p['alpha'], co=p['co'], wave_range=(7600., 7620.))
    np.testing.assert_allclose(spec.flux, sum(v / (1000. if k == 'teff' else 1.) for k, v in p.items()), rtol=1e-12)
    assert len(spec.meta['files']) == 32 and spec.meta['interpolated']
    assert spec.meta['params'] == p
    out = grid.get(1930., 4.8, -1.5, wave_range=(7600., 7620.))  # met below the grid: nearest, flagged
    assert out.meta['outside_grid']


def test_select_holds_unranged_axes(tmp_path):
    grid = T.TemplateGrid(_grid4(tmp_path / 'g'))
    pts = grid.select(teff=1900.)
    assert {p[2] for p in pts} == {0.0} and {p[3] for p in pts} == {0.0}  # met, alpha at defaults
    assert len(pts) == len(TEFFS) * len(LOGGS)
    pts = grid.select(teff=1900., met_range=(-1., 0.), alpha_range=(0., 0.2))
    assert len(pts) == len(TEFFS) * len(LOGGS) * len(METS) * len(ALPHAS)
    assert {p[2] for p in grid.select(teff=1900., priors={'met': -0.6})} == {-0.5}
    with pytest.raises(ValueError):
        grid.select(ranges={'kzz': (1., 2.)})


def test_auto_medium_is_measured(tmp_path):
    d = _grid4(tmp_path / 'v', medium='vacuum')
    T.write_metadata(d, T.GridMeta('toy4', 'auto', pattern=T.GENERIC_PATTERN, teff_scale=1.))
    assert T.TemplateGrid(d).meta.medium == 'vacuum'


def _cands(grid, **ranges):
    return [C.Candidate(f'{grid.name} {p}', grid=grid, params=p) for p in grid.select(**ranges)]


def test_edge_guidance_widen(tmp_path, caplog):
    grid = T.TemplateGrid(_grid4(tmp_path / 'g'))
    cands = _cands(grid, teff_range=(1800., 2000.), met_range=(-0.5, 0.))
    best = next(c for c in cands if c.param_dict == dict(teff=1900., logg=5.0, met=-0.5, alpha=0.))
    scores = [10. if c is best else 50. for c in cands]
    tcfg = {'grid': 'toy4', 'met_range': (-0.5, 0.)}
    s = C.template_summary('template_fit', cands, scores, best, 1000, tcfg)
    assert s['template_at_edge'] and s['template_met'] == -0.5 and s['template_teff'] == 1900.
    assert not np.isnan(s['template_alpha'])  # alpha is encoded in the generic names
    (edge,) = s['edges']
    assert edge['axis'] == 'met' and edge['direction'] == 'low' and edge['action'] == 'widen'
    assert edge['cfg_patch'] == {'templates': {'met_range': [-1.0, 0.0]}}
    assert 'widen templates.met_range to [-1, 0]' in edge['message']
    assert 'met' in caplog.text and 'edge' in caplog.text
    assert s['estimate']['met'][0] == pytest.approx(-0.5, abs=1e-6)


def test_edge_guidance_download(tmp_path):
    grid = T.TemplateGrid(_grid4(tmp_path / 'g'))
    cands = _cands(grid, teff_range=(1800., 2000.), met_range=(-1., 0.))
    best = next(c for c in cands if c.param_dict == dict(teff=1900., logg=5.0, met=-1., alpha=0.))
    scores = [10. if c is best else 50. for c in cands]
    s = C.template_summary('ccf', cands, scores, best, 1000, {'grid': 'toy4'})
    (edge,) = s['edges']
    assert edge['action'] == 'download'
    grid_name = edge['download']['grid']
    meta = T.KNOWN_GRIDS[grid_name]
    assert meta.coverage['met'][0] < -1.
    assert edge['command'].startswith(f'rvfitter templates download --grid {grid_name} ')
    assert '--met' in edge['command']
    assert edge['cfg_patch']['templates']['grid'] == ['toy4', grid_name]
    lo, hi = edge['cfg_patch']['templates']['met_range']
    assert lo < -1. and hi == 0.
    assert f'templates.grid = {["toy4", grid_name]}' in edge['message']


def test_no_edge_inside(tmp_path):
    grid = T.TemplateGrid(_grid4(tmp_path / 'g'))
    cands = _cands(grid, teff_range=(1800., 2000.))
    best = next(c for c in cands if c.param_dict['teff'] == 1900. and c.param_dict['logg'] == 5.0)
    scores = [10. if c is best else 50. for c in cands]
    s = C.template_summary('ccf', cands, scores, best, 100, {})
    assert s['edges'] == [] and not s['template_at_edge']


def test_template_fit_flags_edge_and_results_columns(tmp_path):
    from rvfitter.methods import template_fit
    from rvfitter.types import Region
    grid_dir = _grid4(tmp_path / 'g')
    grid = T.TemplateGrid(grid_dir)
    tpl = C.build_template(C.Candidate('truth', grid=grid, params=(1800., 5.0, 0., 0.)), 3000., (7600., 9000.))
    wave = np.exp(np.arange(np.log(7650.), np.log(8950.), 1. / 3000. / 2.5))
    flux = tpl.at(wave, 12.)
    sigma = flux / 200.
    spec = Spectrum(wave, flux + np.random.default_rng(0).normal(0., 1., len(wave)) * sigma, sigma, 3000., {}, 's')
    cfg = {'templates': {'local_dir': str(grid_dir), 'grid': 'toy4', 'teff_range': (1800., 2000.)},
           'methods': {'options': {'template_fit': {'segment_res': 0}}}}
    res = template_fit.measure(spec, [Region('all', 'band', 7700., 8900.)], cfg, rng=np.random.default_rng(1))
    m = res.measurements[0]
    assert m.diagnostics['template_teff'] == 1800. and m.flag & results.FLAGS['template_param_edge']
    assert m.diagnostics['template_summary']['edges'][0]['axis'] == 'teff'
    obj = ObjectResult('X', ['s'], methods={'template_fit': res})
    row = results.to_table([obj]).iloc[0]
    assert row['template_method'] == 'template_fit' and row['template_grid'] == 'toy4'
    assert row['template_teff'] == 1800. and bool(row['template_at_edge'])
    assert np.isfinite(row['template_chi2'])
    det = results.details_table([obj])
    assert 'diag_template_teff' in det.columns
    acts = pipeline.suggested_refits(obj)
    assert acts and acts[0]['axis'] == 'teff' and acts[0]['action'] in ('download', 'none')


def test_refit_object_applies_and_records_patch(monkeypatch):
    from rvfitter import config
    cfg = config.Config()
    seen = {}
    spec = Spectrum(np.linspace(8000., 8100., 50), np.ones(50), np.full(50, 0.01), 2000., {}, 'f.fits')
    monkeypatch.setattr(pipeline, '_read_all', lambda c, inputs, medium: [spec])

    def fake_measure(spectra, c, ex, rng, *, name=None, methods=None, bary=None, cancel=None):
        seen['cfg'], seen['rng'] = c, rng.integers(1 << 30)
        return ObjectResult(name, [s.source for s in spectra])

    monkeypatch.setattr(pipeline, 'measure_object', fake_measure)
    old = ObjectResult('J1745', ['f.fits'], methods={'ccf': MethodResult('ccf', 1., 1., 1, [Measurement('ccf', 'a',
                                                                                                         1., 1.)])})
    patch = {'templates': {'met_range': [-1.0, 0.0], 'grid': ['bt-settl-cifist', 'sand']}}
    res, new = pipeline.refit_object(old, cfg, patch)
    assert res.object == 'J1745' and seen['cfg'] is new
    assert new.templates.met_range == (-1.0, 0.0) and new.templates.grid == ['bt-settl-cifist', 'sand']
    assert cfg.templates.met_range is None and cfg.refits == []  # the original is untouched
    assert new.refits == [{'object': 'J1745', 'patch': patch}]
    text = config.dumps(new)
    assert '[[refits]]' in text and config.loads(text).refits == new.refits
    first = seen['rng']
    pipeline.refit_object(old, cfg, patch)
    assert seen['rng'] == first  # deterministic


def _online() -> bool:
    try:
        socket.create_connection(('svo2.cab.inta-csic.es', 80), timeout=5).close()
        return True
    except OSError:
        return False


@pytest.mark.slow
@pytest.mark.network
def test_svo_known_models_live():
    """Live check of the SVO model ids (unverified ids are reported, not asserted, except CIFIST)"""
    if not _online():
        pytest.skip('SVO not reachable')
    status = T.available(grids=list(T.KNOWN_GRIDS))
    assert status['bt-settl-cifist']['ok'], status
    for name, st in status.items():
        print(f'{name}: {st}')
