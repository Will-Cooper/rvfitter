"""Tests of rvfitter.systematics"""
import logging

import numpy as np

from rvfitter import config, regions, systematics
from rvfitter import templates as T
from rvfitter.methods import REGISTRY
from rvfitter.methods import _common as C
from rvfitter.methods.guidance import TEMPLATE_COLUMNS as TEMPLATE_AXES
from rvfitter.types import Measurement, MethodResult, Spectrum
from test_templates import _make_grid


def _toy(tmp_path):
    """A synthetic spectrum at 10 km/s from a toy grid, and a config using that grid"""
    grid_dir = _make_grid(tmp_path / 'g')
    grid = T.TemplateGrid(grid_dir)
    tpl = C.build_template(C.Candidate('t', grid=grid, params=(1900., 5.0, 0.)), 3000., (7600., 9000.))
    wave = np.exp(np.arange(np.log(7600.), np.log(9000.), 1. / 3000. / 2.5))
    flux = tpl.at(wave, 10.)
    rng = np.random.default_rng(0)
    sigma = flux / 100.
    spec = Spectrum(wave, flux + rng.normal(0., 1., len(wave)) * sigma, sigma, 3000., {}, 'synthetic')
    cfg = config.Config()
    cfg.templates.local_dir, cfg.templates.teff_prior = str(grid_dir), 1900.
    return spec, cfg


def test_template_spread_components(tmp_path):
    spec, cfg = _toy(tmp_path)
    sel = regions.select(spec, teff=1900., methods=['template_fit'])
    out = systematics.template_spread_details(spec, sel, cfg, rng=np.random.default_rng(1))
    assert len(out['templates']) == 6 and all(np.isfinite(v) for v in out['templates'].values())
    assert set(out['window_modes']) == {'features', 'full'}
    assert np.isfinite(out['total']) and out['total'] < 5. and np.isfinite(out['delta_modes'])


def test_template_spread_with_ccf_on_its_own_regions(tmp_path):
    spec, cfg = _toy(tmp_path)
    sel = regions.select(spec, teff=1900., methods=['ccf'])
    out = systematics.template_spread_details(spec, sel, cfg, rng=np.random.default_rng(1), method='ccf')
    assert len(out['teff_steps']) >= 2 and np.isfinite(out['total'])


def test_grid_opened_once_and_reruns_at_debug(tmp_path, monkeypatch, caplog):
    spec, cfg = _toy(tmp_path)
    cfg.templates.grid = 'toy'  # as the variants pin it: one grid name, one cache entry
    opened = []

    class Counting(T.TemplateGrid):
        def __init__(self, *args, **kwargs):
            opened.append(args[0])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(T, 'TemplateGrid', Counting)
    sel = regions.select(spec, teff=1900., methods=['template_fit', 'ccf'])
    with caplog.at_level(logging.DEBUG, logger='rvfitter'):
        for method in ('template_fit', 'ccf'):
            REGISTRY[method](spec, sel[method], cfg, rng=np.random.default_rng(0))
        systematics.template_spread_details(spec, sel, cfg, rng=np.random.default_rng(1))
    assert len(opened) == 1  # two methods and every systematics variant share one grid
    summaries = [r.levelno for r in caplog.records
                 if r.name.endswith('template_fit') and r.getMessage().startswith('template_fit: ')
                 and 'km/s (template' in r.getMessage()]
    assert len(summaries) > 2 and summaries == [logging.INFO] + [logging.DEBUG] * (len(summaries) - 1)


def test_template_spread_one_teff_step_about_the_best_fit(tmp_path, monkeypatch):
    """A 1500-4500 K grid, best fit at 2000 K/5.0: the term is the RV change one Teff step either side, at log g
    5.0, measured when outside the candidates (prior 1700 K); catastrophic templates (4400 K, and 2100 K/4.5)
    at +437 km/s never enter, with or without a prior, and nor does a catastrophic full-span fit"""
    spec, cfg = _toy(tmp_path)
    sel = regions.select(spec, teff=1900., methods=['template_fit'])
    pts = np.array([(t, g, 0., 0.) for t in np.arange(1500., 4501., 100.) for g in (4.5, 5.)])

    class Grid:
        name, axes, points = 'fake', TEMPLATE_AXES, pts

        def params(self, **p):
            return dict(dict(logg=5., met=0., alpha=0.), **p)

    def cands(cfg, opts):
        t0 = cfg.templates.teff_prior
        return [C.Candidate(f'fake {t:g}/{g:g}/0/0', grid=Grid(), params=(t, g)) for t, g, _, _ in pts
                if t0 is None or abs(t - t0) <= opts['teff_window']]

    def fake(spectrum, regs, c, *, executor=None, rng):
        t, g = (c.templates.teff_range[0], c.templates.logg_range[0]) if c.templates.teff_range else (2000., 5.)
        rv = 437. if t == 4400. or (t, g) == (2100., 4.5) or len(regs) != len(sel['template_fit']) \
            else 10. + (t - 2000.) / 100.
        chi2 = 1000. + 50. * ((t - 2000.) / 100.) ** 2 + (0. if g == 5. else 10.)
        return MethodResult('template_fit', rv, 1., 1, [Measurement('template_fit', 'w', rv, 1., 0,
                                                                    dict(chi2=chi2, dof=1000))])

    monkeypatch.setattr(C, 'candidates', cands)
    monkeypatch.setitem(REGISTRY, 'template_fit', fake)
    for prior in (None, 1700.):
        cfg.templates.teff_prior = prior
        out = systematics.template_spread_details(spec, sel, cfg, rng=np.random.default_rng(1))
        assert out['teff_steps'] == {'fake 2000/5/0/0': 10., 'fake 1900/5/0/0': 9., 'fake 2100/5/0/0': 11.}
        assert out['delta_modes'] == 427. and out['sd_templates'] == 1. and out['total'] == 1.


def test_grid_reopened_when_a_template_file_is_overwritten(tmp_path):
    """An in-place overwrite leaves the directory mtime alone; the cached grid and its medium check must not survive"""
    _, cfg = _toy(tmp_path)
    cfg.templates.grid = 'toy'
    first = C.open_grid(cfg.templates)
    assert C.open_grid(cfg.templates) is first  # cached
    path = sorted(p for p in (tmp_path / 'g').iterdir() if p.name.startswith('lte'))[0]
    path.write_text(path.read_text() + '\n')  # same file, new contents
    assert C.open_grid(cfg.templates) is not first


def test_wavelength_zero_point():
    spec = Spectrum(np.linspace(8000., 9000., 10), np.ones(10), np.ones(10), 2000., {'wave_rms': 0.085}, '')
    assert abs(systematics.wavelength_zero_point(spec) - 299792.458 * 0.085 / 8500.) < 1e-6
    assert np.isnan(systematics.wavelength_zero_point(Spectrum(spec.wave, spec.flux, spec.sigma)))
