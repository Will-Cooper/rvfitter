"""Tests of rvfitter.systematics"""
import numpy as np

from rvfitter import regions, systematics
from rvfitter import templates as T
from rvfitter.methods import _common as C
from rvfitter.types import Spectrum
from test_templates import _make_grid


def test_template_spread_components(tmp_path):
    grid_dir = _make_grid(tmp_path / 'g')
    grid = T.TemplateGrid(grid_dir)
    tpl = C.build_template(C.Candidate('t', grid=grid, params=(1900., 5.0, 0.)), 3000., (7600., 9000.))
    wave = np.exp(np.arange(np.log(7600.), np.log(9000.), 1. / 3000. / 2.5))
    flux = tpl.at(wave, 10.)
    rng = np.random.default_rng(0)
    sigma = flux / 100.
    spec = Spectrum(wave, flux + rng.normal(0., 1., len(wave)) * sigma, sigma, 3000., {}, 'synthetic')
    cfg = {'templates': {'local_dir': str(grid_dir), 'teff_prior': 1900.}, 'regions': {'template_windows': 'features'},
           'input': {'telluric_corrected': False}}
    sel = regions.select(spec, teff=1900., methods=['template_fit'])
    out = systematics.template_spread_details(spec, sel, cfg, rng=np.random.default_rng(1))
    assert len(out['templates']) == 6 and all(np.isfinite(v) for v in out['templates'].values())
    assert set(out['window_modes']) == {'features', 'full'}
    assert np.isfinite(out['total']) and out['total'] < 5.
    assert systematics.template_spread(spec, sel, cfg, rng=np.random.default_rng(1)) == out['total']


def test_wavelength_zero_point():
    spec = Spectrum(np.linspace(8000., 9000., 10), np.ones(10), np.ones(10), 2000., {'wave_rms': 0.085}, '')
    assert abs(systematics.wavelength_zero_point(spec, {}) - 299792.458 * 0.085 / 8500.) < 1e-6
    assert np.isnan(systematics.wavelength_zero_point(Spectrum(spec.wave, spec.flux, spec.sigma), {}))
