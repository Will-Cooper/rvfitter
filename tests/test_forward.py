"""Tests of rvfitter.methods.forward (forward-model MCMC): flags, executor invariance, epochs and injection–recovery."""
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import injection
import synth
from rvfitter import templates as T
from rvfitter.methods import REGISTRY, forward
from rvfitter.types import Region

pytestmark = pytest.mark.usefixtures('clear_template_cache')
log = logging.getLogger(__name__)
TEFFS = (1900., 2000., 2100.)
FAST = dict(n_walkers=12, n_burn=30, n_steps=60, segment_res=0.)


def _toy_grid(directory: Path) -> Path:
    """Three vacuum models (synthetic lines plus the alkali lines) whose line depths scale with teff"""
    directory.mkdir(parents=True, exist_ok=True)
    wave, base = synth.make_template((7800., 8900.), rng=np.random.default_rng(0))
    for teff in TEFFS:
        flux = base ** (1. + (teff - 2000.) / 400.)
        for _, lam in T.medium_check_lines():
            flux = flux * (1. - 0.4 * np.exp(-0.5 * ((wave - lam) / 0.6) ** 2))
        np.savetxt(directory / T._generic_name(dict(teff=teff, logg=5., met=0., alpha=0.)), np.c_[wave, flux])
    T.write_metadata(directory, T.GridMeta('toy', 'vacuum', pattern=T.GENERIC_PATTERN, teff_scale=1.))
    return directory


@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    d = _toy_grid(tmp_path_factory.mktemp('toygrid'))
    return d, T.TemplateGrid(d).get(2000., 5.)


def _cfg(directory, **opts):
    return injection.cfg_of({'templates': {'grid': 'toy', 'local_dir': str(directory), 'teff_prior': 2000.},
                             'methods': {'options': {'forward_mcmc': {**FAST, **opts},
                                                     'template_fit': {'segment_res': 0.}}}})


def _spec(tpl, rv, seed, snr=50.):
    spec, _ = synth.make_spectrum(rv=rv, resolution=2500., snr=snr, wave_range=(8000., 8700.),
                                  template=_uniform(tpl), rng=np.random.default_rng(seed))
    return spec


def _uniform(tpl):
    """A template Spectrum on a uniform ln λ grid (as synth.make_spectrum needs)"""
    lnw = np.arange(np.log(tpl.wave[0]), np.log(tpl.wave[-1]), 1. / 600_000)
    return np.exp(lnw), np.interp(np.exp(lnw), tpl.wave, tpl.flux)


WINDOWS = [Region('a', 'band', 8020., 8340.), Region('b', 'band', 8360., 8680.)]


def test_registry_flags_short_chains_not_converged(toy):
    d, tpl = toy
    res = REGISTRY['forward_mcmc'](_spec(tpl, 15., 1), WINDOWS[:1], _cfg(d), rng=np.random.default_rng(0))
    m = res.measurements[0]
    assert not m.diagnostics['converged'] and m.flag & forward.C.FLAG_EDGE  # not_converged
    assert np.isnan(res.rv)  # excluded from the combination
    assert m.diagnostics['sample_names'] == ['rv_0', 'vsini', 'teff']
    assert abs(m.rv - 15.) < 4 * m.rv_err and 1900. <= m.diagnostics['teff_post'] <= 2100.


def test_executor_invariance(toy):
    d, tpl = toy
    spec, cfg = _spec(tpl, 15., 2), _cfg(d)
    serial = forward.measure(spec, WINDOWS, cfg, rng=np.random.default_rng(3))
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as ex:
        par = forward.measure(spec, WINDOWS, cfg, executor=ex, rng=np.random.default_rng(3))
    for a, b in zip(serial.measurements, par.measurements, strict=True):
        assert a.rv == b.rv and a.rv_err == b.rv_err
        np.testing.assert_array_equal(a.diagnostics['samples'], b.diagnostics['samples'])


def test_fit_epochs_recovers_two_rvs(toy):
    d, tpl = toy
    spectra = [_spec(tpl, 10., 4), _spec(tpl, 60., 5)]
    out = forward.fit_epochs(spectra, [WINDOWS[:1]] * 2, _cfg(d, n_steps=200), rng=np.random.default_rng(6))
    for res, truth in zip(out, (10., 60.), strict=True):
        m = res.measurements[0]
        assert m.diagnostics['n_epochs'] == 2 and m.diagnostics['sample_names'][:2] == ['rv_0', 'rv_1']
        assert abs(m.rv - truth) < 4 * m.rv_err


def test_telluric_term_only_for_uncorrected_input(toy):
    d, tpl = toy
    spec = _spec(tpl, 15., 7)
    cfg = _cfg(d, telluric=True)
    m = forward.measure(spec, WINDOWS[:1], cfg, rng=np.random.default_rng(0)).measurements[0]
    assert m.diagnostics['sample_names'][-1] == 'alpha'
    cfg.input.telluric_corrected = True
    m = forward.measure(spec, WINDOWS[:1], cfg, rng=np.random.default_rng(0)).measurements[0]
    assert 'alpha' not in m.diagnostics['sample_names']


def _bt_settl_trial(seed: int) -> tuple[float, float, bool]:
    root = Path(os.environ['RVFITTER_TESTDATA']) / 'templates' / 'bt-settl-cifist' / 'useful'
    tpl = T.TemplateGrid(root, grid='bt-settl-cifist').broadened(2000., 5., resolution=None,
                                                                 wave_range=(7700., 9100.), dv=1.)
    spec, _ = synth.make_spectrum(rv=23.4, resolution=2500., snr=30., wave_range=(8000., 8800.),
                                  template=(tpl.wave, tpl.flux), rng=np.random.default_rng(seed))
    cfg = injection.cfg_of({'templates': {'local_dir': str(root), 'teff_prior': 2000.},
                            'methods': {'options': {'forward_mcmc': {'segment_res': 0.},
                                                    'template_fit': {'segment_res': 0.}}}})
    m = forward.measure(spec, [Region('all', 'band', 8020., 8780.)], cfg,
                        rng=np.random.default_rng(seed + 1000)).measurements[0]
    return m.rv, m.rv_err, m.diagnostics['converged']


@pytest.mark.data
@pytest.mark.slow
def test_injection_recovery_bt_settl():
    """BT-Settl 2000/5.0 at R 2500, SNR 30, default chains: the posterior covers the truth (40 seeds)"""
    if not os.environ.get('RVFITTER_TESTDATA'):
        pytest.skip('RVFITTER_TESTDATA not set')
    with ProcessPoolExecutor(min(os.cpu_count() or 1, 20), mp_context=multiprocessing.get_context('spawn')) as ex:
        rvs, errs, conv = map(np.array, zip(*ex.map(_bt_settl_trial, range(40)), strict=True))
    pull = (rvs - 23.4) / errs
    log.info('pull width %.2f, mean %.2f; converged %d/40', np.std(pull, ddof=1), np.mean(pull), conv.sum())
    assert conv.mean() >= 0.9, conv
    assert 0.8 < np.std(pull, ddof=1) < 1.2 and abs(np.mean(pull)) < 0.5, pull


@pytest.mark.data
@pytest.mark.slow
def test_j1745(caplog):
    """J1745−1640 with and without the default sky correction: RV, vsini and Teff are reported, not tuned"""
    from rvfitter import pipeline
    root = os.environ.get('RVFITTER_TESTDATA')
    if not root:
        pytest.skip('RVFITTER_TESTDATA not set')
    caplog.set_level(logging.INFO, logger='rvfitter')
    sci = Path(root) / 'pypeit/R2500I/GTC54-15A0/OB0055/Science'
    for mode in ('none', 'sky-if-needed'):
        c = injection.cfg_of({'templates': {'local_dir': str(Path(root) / 'templates/bt-settl-cifist/useful'),
                                            'teff_prior': 2100.}})
        c.input.telluric_corrected = True
        c.input.resolution = 2602.
        c.velocity_frame.correction = 'skip'
        c.velocity_frame.frame = 'heliocentric'
        c.methods.enabled = ['forward_mcmc']
        c.calibration.wavelength_correction = mode
        c.calibration.sky_source = str(next(sci.glob('spec1d_*.fits')))
        obj = pipeline.run(c, [str(sci / 'DENISJ174534.6-164053_tellcorr.fits')], n_procs=min(os.cpu_count(), 20))[0]
        res = obj.methods['forward_mcmc']
        for m in res.measurements:
            d = m.diagnostics
            log.info('%s %s: %.2f ± %.2f km/s, vsini %.1f ± %.1f, teff %.0f ± %.0f, logg %.2f (flag %d, τ %.0f)',
                     mode, m.region, m.rv, m.rv_err, d['vsini'], d['vsini_err'], d['teff_post'], d['teff_post_err'],
                     d['logg_post'], m.flag, np.max(d['tau']))
        log.info('%s: forward_mcmc %.2f ± %.2f km/s (%d windows; z %.2f)', mode, res.rv, res.rv_err, res.n_regions,
                 abs(res.rv - 26.2) / np.hypot(2.3, res.rv_err))
        assert res.n_regions >= 3 and np.isfinite(res.rv)
