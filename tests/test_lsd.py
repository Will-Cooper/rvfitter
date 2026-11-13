"""Least-squares deconvolution: recovery, vsini, the catalogue mask, injection pulls and J1745−1640."""
import logging
import os
from pathlib import Path

import numpy as np
import pytest

import injection
import synth
from rvfitter import config
from rvfitter.methods import REGISTRY, lsd

log = logging.getLogger(__name__)


def _measure(spec, **opts):
    cfg = injection.cfg_of({'methods': {'options': {'lsd': opts}}})
    return lsd.measure(spec, injection.full_region(spec), cfg, rng=np.random.default_rng(1))


def test_registered_not_default():
    assert REGISTRY['lsd'] is lsd.measure
    assert 'lsd' in config.METHOD_NAMES and 'lsd' not in config.Config().methods.enabled


@pytest.mark.parametrize('vsini', [0., 60.])
def test_recovers_rv_and_vsini(vsini):
    tpl = synth.make_template(rng=np.random.default_rng(0))
    spec, _ = synth.make_spectrum(rv=23.4, snr=100., vsini=vsini, template=tpl, rng=np.random.default_rng(3))
    res = _measure(spec, template=tpl)
    d = res.measurements[0].diagnostics
    assert abs(res.rv - 23.4) < 3 * res.rv_err and res.rv_err < 2.
    assert len(d['profile']) == len(d['profile_v']) == len(d['profile_err'])
    if vsini:
        assert abs(d['vsini'] - vsini) < 3 * d['vsini_err']
    else:
        assert np.isnan(d['vsini']) and d['vsini_upper'] < 50.


def test_catalogue_mask_and_errors():
    tpl = synth.make_template(rng=np.random.default_rng(0))
    spec, _ = synth.make_spectrum(rv=-40., snr=100., template=tpl, rng=np.random.default_rng(4))
    res = _measure(spec, template=tpl, mask='catalogue')
    assert abs(res.rv + 40.) < 3 * res.rv_err and res.measurements[0].diagnostics['n_mask_lines'] >= 4
    with pytest.raises(ValueError, match='mask'):
        _measure(spec, template=tpl, mask='vald')


@pytest.mark.slow
def test_pulls():
    """R 2500, SNR 30: pull width 0.8–1.2 and |bias| < 0.2σ (200 seeded spectra)"""
    _, _, summary = injection.run(lsd.measure, lambda tpl: {'template': tpl}, n=200, method='lsd', seed=21)
    log.info('lsd pulls: %s', summary)
    injection.check(summary)


J1745 = Path('pypeit/R2500I/GTC54-15A0/OB0055/Science')


@pytest.mark.data
@pytest.mark.slow
def test_j1745():
    """J1745−1640: LSD RV without and with the default sky correction (reported, not tuned)"""
    from rvfitter import pipeline, results
    root = os.environ.get('RVFITTER_TESTDATA')
    if not root:
        pytest.skip('RVFITTER_TESTDATA not set')
    sci = Path(root) / J1745
    for mode in ('none', 'sky-if-needed'):
        c = config.Config()
        c.templates.local_dir = str(Path(root) / 'templates' / 'bt-settl-cifist' / 'useful')
        c.templates.teff_prior = 2100.
        c.input.telluric_corrected = True
        c.input.resolution = 2602.
        c.velocity_frame.correction = 'skip'
        c.velocity_frame.frame = 'heliocentric'
        c.methods.enabled = ['lsd']
        c.calibration.wavelength_correction = mode
        c.calibration.sky_source = str(next(sci.glob('spec1d_*.fits')))
        obj = pipeline.run(c, [str(sci / 'DENISJ174534.6-164053_tellcorr.fits')], n_procs=1)[0]
        m = obj.methods['lsd']
        d = m.measurements[0].diagnostics
        total = float(results.to_table([obj]).iloc[0]['rv_total_err'])
        log.info('%s: lsd %.2f ± %.2f km/s (total %.2f; z %.2f stat, %.2f total; zero point %.2f; vsini %s, < %.0f; '
                 '%d mask lines; template %s)', mode, m.rv, m.rv_err, total, abs(m.rv - 26.2) / np.hypot(2.3, m.rv_err),
                 abs(m.rv - 26.2) / np.hypot(2.3, total), d['mask_zero_point'], d['vsini'], d['vsini_upper'],
                 d['n_mask_lines'], d['template'])
        assert np.isfinite(m.rv) and np.isfinite(total)
