"""Tests of rvfitter.methods.ccf: peak fitting, injection–recovery, template choice and executor independence."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import injection
import synth
from rvfitter.methods import REGISTRY, ccf
from rvfitter.methods import _common as C


def test_fit_peak_gaussian_and_parabola():
    v = np.arange(-300., 300., 5.)
    c = 0.1 + 0.7 * np.exp(-0.5 * ((v - 12.3) / 40.) ** 2)
    g = ccf.fit_peak(v, c, 'gaussian')
    np.testing.assert_allclose([g['rv'], g['height'], g['width']], [12.3, 0.8, 40.], rtol=1e-6)
    np.testing.assert_allclose(g['curvature'], -0.7 / 40. ** 2, rtol=1e-6)
    p = ccf.fit_peak(v, c, 'parabola')
    assert abs(p['rv'] - 12.3) < 0.5 and p['ok']
    edge = ccf.fit_peak(v, np.exp(-0.5 * ((v - 400.) / 40.) ** 2))
    assert edge['edge']


def test_zucker_error_formula():
    n, c, c2 = 1000, 0.6, -1e-3
    np.testing.assert_allclose(ccf.zucker_error(n, c, c2), np.sqrt(-1 / (n * c2 / c * c ** 2 / (1 - c ** 2))))
    assert np.isnan(ccf.zucker_error(n, 1.2, c2)) and np.isnan(ccf.zucker_error(n, c, 1e-3))


def test_ccf_curve_is_one_for_identical():
    d = np.sin(np.linspace(0, 20, 300))
    c = ccf.ccf_curve(d, np.ones(300), np.vstack([d, -d, 2 * d + 3]))
    np.testing.assert_allclose(c, [1., -1., 1.])


def test_registry():
    assert REGISTRY['ccf'].__call__  # lazy wrapper
    spec, truth = synth.make_spectrum(rv=-40., snr=50., rng=np.random.default_rng(3))
    cfg = {'methods': {'options': {'ccf': {'template': truth['template'], 'n_mc': 10}}}}
    res = REGISTRY['ccf'](spec, injection.full_region(spec), cfg, rng=np.random.default_rng(0))
    assert abs(res.rv + 40.) < 5. and res.n_regions == len(res.measurements) == 3  # 1500 Å = 3 segments at R 2500
    m = res.measurements[0]
    for key in ('rv_grid', 'ccf', 'zucker_err', 'mc_std', 'template', 'wave', 'data', 'model'):
        assert key in m.diagnostics


def test_unknown_option():
    spec, truth = synth.make_spectrum(rng=np.random.default_rng(3))
    with pytest.raises(ValueError, match='unknown ccf options'):
        ccf.measure(spec, injection.full_region(spec), {'methods': {'options': {'ccf': {'n_mcc': 3}}}},
                    rng=np.random.default_rng(0))


def test_unknown_resolution_is_estimated(caplog):
    from dataclasses import replace
    spec, truth = synth.make_spectrum(rv=10., resolution=3000., snr=60., rng=np.random.default_rng(3))
    cfg = {'methods': {'options': {'ccf': {'template': truth['template'], 'n_mc': 10}}}}
    res = ccf.measure(replace(spec, resolution=None), injection.full_region(spec), cfg, rng=np.random.default_rng(0))
    assert 'estimated R' in caplog.text
    r = res.measurements[0].diagnostics['resolution']
    assert abs(r - 3000.) / 3000. < 0.1 and abs(res.rv - 10.) < 3 * res.rv_err


def test_injection_recovery_fast():
    _, _, s = injection.run(ccf.measure, lambda t: {'template': t, 'n_mc': 40}, n=40, method='ccf', seed=11)
    injection.check(s)


SLOW_CASES = [(2500., 15., 0., (7500., 9000.)), (6000., 40., 20., (7500., 9000.)), (20000., 30., 30., (8000., 8500.))]


@pytest.mark.slow
@pytest.mark.parametrize('resolution,snr,vsini,wave_range', SLOW_CASES)
def test_injection_recovery_slow(resolution, snr, vsini, wave_range):
    opts = lambda t: {'template': t, 'n_mc': 60, 'vsini': vsini}  # noqa: E731
    _, _, s = injection.run(ccf.measure, opts, n=120, resolution=resolution, snr=snr, vsini=vsini,
                            wave_range=wave_range, method='ccf',
                            seed=21)
    injection.check(s)


def test_template_choice_by_chi2():
    spec, truth = synth.make_spectrum(rv=10., snr=40., rng=np.random.default_rng(5))
    wrong = [synth.make_template((7500., 9000.), rng=np.random.default_rng(k)) for k in (7, 8)]
    from rvfitter.types import Spectrum
    templates = [Spectrum(w, f, np.full(len(w), np.nan), None, {'object': f'T{k}'})
                 for k, (w, f) in enumerate([wrong[0], truth['template'], wrong[1]])]
    cfg = {'methods': {'options': {'ccf': {'templates': templates, 'n_mc': 10}}}}
    res = ccf.measure(spec, injection.full_region(spec), cfg, rng=np.random.default_rng(0))
    assert res.measurements[0].diagnostics['template'] == 'T1'


def test_executor_independent_and_masks():
    spec, truth = synth.make_spectrum(rv=5., snr=30., rng=np.random.default_rng(9))
    regions = injection.full_region(spec)
    mid = 0.5 * (spec.wmin + spec.wmax)
    regions.append(C.Region('tell', 'telluric', mid - 30., mid + 30.))
    cfg = {'methods': {'options': {'ccf': {'template': truth['template'], 'n_mc': 24}}}}
    serial = ccf.measure(spec, regions, cfg, rng=np.random.default_rng(1))
    with ThreadPoolExecutor(4) as ex:
        par = ccf.measure(spec, regions, cfg, executor=ex, rng=np.random.default_rng(1))
    assert serial.rv == par.rv and serial.rv_err == par.rv_err
    wave = serial.measurements[0].diagnostics['wave']
    assert not np.any((wave > mid - 30.) & (wave < mid + 30.))


def test_no_sigma_estimates_noise(caplog):
    from dataclasses import replace
    spec, truth = synth.make_spectrum(rv=5., snr=30., rng=np.random.default_rng(9))
    spec = replace(spec, sigma=np.full(len(spec.wave), np.nan))
    cfg = {'methods': {'options': {'ccf': {'template': truth['template'], 'n_mc': 20}}}}
    res = ccf.measure(spec, injection.full_region(spec), cfg, rng=np.random.default_rng(1))
    assert 'no flux errors' in caplog.text
    assert abs(res.rv - 5.) < 4 * res.rv_err and 0.5 < res.rv_err < 10.
