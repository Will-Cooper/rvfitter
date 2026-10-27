"""Tests of rvfitter.methods.template_fit: injection–recovery, vsini, bootstrap, resolution and executor use."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import injection
import synth
from rvfitter.methods import REGISTRY, template_fit
from rvfitter.methods import _common as C


def _cfg(tpl, **kw):
    return {'methods': {'options': {'template_fit': {'template': tpl, **kw}}}}


def test_registry_and_diagnostics():
    spec, truth = synth.make_spectrum(rv=-40., snr=50., rng=np.random.default_rng(3))
    res = REGISTRY['template_fit'](spec, injection.full_region(spec), _cfg(truth['template']),
                                   rng=np.random.default_rng(0))
    assert abs(res.rv + 40.) < 5.
    d = res.measurements[0].diagnostics
    assert 0.7 < d['chi2_red'] < 1.3
    for key in ('chi2_scan', 'rv_grid', 'model', 'data', 'wave', 'poly_coef', 'err_cov'):
        assert key in d


def test_injection_recovery_fast():
    _, _, s = injection.run(template_fit.measure, lambda t: {'template': t}, n=60, method='template_fit', seed=12)
    injection.check(s)


SLOW_CASES = [(2500., 15., 0., (7500., 9000.)), (6000., 40., 20., (7500., 9000.)), (20000., 30., 30., (8000., 8500.))]


@pytest.mark.slow
@pytest.mark.parametrize('resolution,snr,vsini,wave_range', SLOW_CASES)
def test_injection_recovery_slow(resolution, snr, vsini, wave_range):
    opts = lambda t: {'template': t, 'fit_vsini': vsini > 0, 'vsini': 10. if vsini else 0.}  # noqa: E731
    _, _, s = injection.run(template_fit.measure, opts, n=120, resolution=resolution, snr=snr, vsini=vsini,
                            wave_range=wave_range,
                            method='template_fit', seed=22)
    injection.check(s)


def test_vsini_recovered_at_high_resolution():
    spec, truth = synth.make_spectrum(rv=12., resolution=20000., snr=60., vsini=25., wave_range=(8000., 8600.),
                                      rng=np.random.default_rng(4))
    res = template_fit.measure(spec, injection.full_region(spec), _cfg(truth['template'], fit_vsini=True, vsini=10.),
                               rng=np.random.default_rng(0))
    d = res.measurements[0].diagnostics
    assert abs(res.rv - 12.) < 3 * res.rv_err
    assert abs(d['vsini'] - 25.) < max(3 * d['vsini_err'], 2.)


def test_bootstrap_matches_covariance_and_is_executor_independent():
    spec, truth = synth.make_spectrum(rv=3., snr=30., rng=np.random.default_rng(8))
    cfg = _cfg(truth['template'], bootstrap=80)
    serial = template_fit.measure(spec, injection.full_region(spec), cfg, rng=np.random.default_rng(2))
    with ThreadPoolExecutor(4) as ex:
        par = template_fit.measure(spec, injection.full_region(spec), cfg, executor=ex, rng=np.random.default_rng(2))
    assert serial.rv_err == par.rv_err
    d = serial.measurements[0].diagnostics
    assert 0.7 < d['err_bootstrap'] / d['err_cov'] < 1.4


def test_measure_resolution():
    spec, truth = synth.make_spectrum(rv=3., resolution=3000., snr=80., rng=np.random.default_rng(8))
    out = template_fit.measure_resolution(spec, injection.full_region(spec), _cfg(truth['template']))
    assert abs(out['resolution'] - 3000.) < max(3 * out['resolution_err'], 150.)


def test_edge_flag_when_outside_search_range():
    spec, truth = synth.make_spectrum(rv=180., snr=50., rng=np.random.default_rng(3))
    res = template_fit.measure(spec, injection.full_region(spec), _cfg(truth['template'], rv_min=-100., rv_max=100.),
                               rng=np.random.default_rng(0))
    assert all(m.flag & C.FLAG_EDGE for m in res.measurements) and res.n_regions == 0


def test_segments_and_outlier_rejection():
    from rvfitter.types import Measurement
    spec, truth = synth.make_spectrum(rv=3., snr=40., rng=np.random.default_rng(8))
    segs = C.segment(injection.full_region(spec), 2500., 150.)
    assert len(segs) == 3 and segs[0].wmin == injection.full_region(spec)[0].wmin
    np.testing.assert_allclose([s.wmax for s in segs[:-1]], [s.wmin for s in segs[1:]])
    assert C.segment(injection.full_region(spec), 2500., 0) == injection.full_region(spec)
    ms = [Measurement('x', str(i), v, 2.) for i, v in enumerate([20., 22., 25., 21., 400.])]
    res = C.finish('x', ms)
    assert ms[-1].flag & C.FLAG_OUTLIER and res.n_regions == 4 and abs(res.rv - 22.) < 2.
