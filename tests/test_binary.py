"""Binary test: an injected SB2 is flagged and a single star is not; J1745−1640's ΔBIC (reported)."""
import logging
import os
from pathlib import Path

import numpy as np
import pytest

import injection
import synth
from rvfitter.methods import binary


def _spec(sb2: bool, seed: int):
    tpl = synth.make_template(rng=np.random.default_rng(0))
    a, _ = synth.make_spectrum(rv=-60., snr=np.inf, template=tpl, rng=np.random.default_rng(seed))
    flux = a.flux.copy()
    if sb2:
        b, _ = synth.make_spectrum(rv=90., snr=np.inf, template=tpl, rng=np.random.default_rng(seed))
        flux = (flux + 0.6 * b.flux) / 1.6
    rng = np.random.default_rng(seed + 100)
    sigma = flux / 50.
    return a.__class__(a.wave, flux + rng.normal(0., sigma), sigma, a.resolution, a.meta, a.source), tpl


@pytest.mark.parametrize('seed', [1, 2])
def test_sb2_flagged_single_not(seed):
    for sb2 in (False, True):
        spec, tpl = _spec(sb2, seed)
        cfg = injection.cfg_of({'methods': {'options': {'template_fit': {'template': tpl}}}})
        out = binary.test(spec, injection.full_region(spec), cfg)
        assert out['flag'] is sb2, out
        if sb2:
            assert sorted([out['rv1'], out['rv2']]) == pytest.approx([-60., 90.], abs=10.)
            assert out['q'] == pytest.approx(0.6, abs=0.15)


@pytest.mark.data
@pytest.mark.slow
def test_j1745():
    """J1745−1640 through the pipeline with the binary test on (reported, not tuned)"""
    from rvfitter import config, pipeline, results
    root = os.environ.get('RVFITTER_TESTDATA')
    if not root:
        pytest.skip('RVFITTER_TESTDATA not set')
    sci = Path(root) / 'pypeit/R2500I/GTC54-15A0/OB0055/Science'
    c = config.Config()
    c.templates.local_dir = str(Path(root) / 'templates' / 'bt-settl-cifist' / 'useful')
    c.templates.teff_prior = 2100.
    c.input.telluric_corrected = True
    c.input.resolution = 2602.
    c.velocity_frame.correction = 'skip'
    c.velocity_frame.frame = 'heliocentric'
    c.methods.enabled = ['template_fit']
    c.methods.binary = True
    c.calibration.wavelength_correction = 'none'
    obj = pipeline.run(c, [str(sci / 'DENISJ174534.6-164053_tellcorr.fits')], n_procs=1)[0]
    row = results.to_table([obj]).iloc[0]
    e = obj.binary['epochs'][0]
    logging.getLogger(__name__).info('J1745 binary: ΔBIC %.1f, flag %s (single %.1f; double %.1f, %.1f, q %.2f; '
                                     'χ²/ν %.1f)', row['binary_dbic'], row['binary_flag'], e['rv'], e['rv1'],
                                     e['rv2'], e['q'], e['chi2_red'])
    assert np.isfinite(row['binary_dbic'])
