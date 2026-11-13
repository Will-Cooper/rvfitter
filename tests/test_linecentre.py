"""Line centring: injection–recovery pulls per profile, flags, the joint fit and J1745−1640."""
import logging
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import voigt_profile

import injection
import synth
from rvfitter import config, regions
from rvfitter.constants import C_KMS
from rvfitter.methods import REGISTRY, linecentre
from rvfitter.methods import _common as C

log = logging.getLogger(__name__)
RV = 23.4
# intrinsic widths [km/s]: the Lorentzian is wide, so that it dominates the Gaussian LSF (c/R/2.355 ≈ 51 km/s)
WIDTH = {'gaussian': 15., 'voigt': 15., 'lorentzian': 100.}


def _template(profile: str, wave_range=(7500., 9000.)):
    """Rest-frame anchor lines with the given profile (depth 0.5) on a sloped continuum"""
    wave = np.exp(np.arange(np.log(wave_range[0] * 0.98), np.log(wave_range[1] * 1.02), 1. / 600_000))
    flux = 1. + 0.1 * (wave - wave.mean()) / (wave_range[1] - wave_range[0])
    s, g = profile != 'lorentzian', profile != 'gaussian'
    for c in synth.ANCHOR_LINES:
        if wave_range[0] < c < wave_range[1]:
            w = c * WIDTH[profile] / C_KMS
            flux *= 1. - 0.5 * voigt_profile(wave - c, s * w, g * w) / voigt_profile(0., s * w, g * w)
    return wave, flux


def _line_regions(spec):
    return regions.select(spec, methods=['linecentre'])['linecentre']


def _measure(spec, regs, **opts):
    cfg = injection.cfg_of({'methods': {'options': {'linecentre': opts}}})
    return linecentre.measure(spec, regs, cfg, rng=np.random.default_rng(0))


def test_registered_and_default():
    assert REGISTRY['linecentre'] is linecentre.measure
    assert 'linecentre' in config.Config().methods.enabled


@pytest.mark.parametrize('profile', sorted(linecentre.PROFILES))
def test_recovers_rv(profile):
    spec, _ = synth.make_spectrum(rv=RV, snr=300., template=_template(profile), rng=np.random.default_rng(1))
    res = _measure(spec, _line_regions(spec), profile=profile)
    assert res.n_regions >= 4
    assert abs(res.rv - RV) < 3. * res.rv_err and res.rv_err < 3.
    for m in res.measurements:
        assert m.diagnostics['profile'] == profile and len(m.diagnostics['rest_wave']) == 1


def test_joint():
    spec, _ = synth.make_spectrum(rv=RV, snr=30., template=_template('voigt'), rng=np.random.default_rng(2))
    regs = _line_regions(spec)
    per_line = _measure(spec, regs)
    joint = _measure(spec, regs, joint=True)
    assert len(joint.measurements) == 1 and joint.measurements[0].region.startswith('joint: ')
    assert set(joint.measurements[0].diagnostics['lines']) == {m.region for m in per_line.measurements}
    assert abs(joint.rv - RV) < 3. * joint.rv_err
    # the joint error matches the inverse-variance combination of the per-line errors
    ivar = 1. / np.sqrt(sum(m.rv_err ** -2 for m in per_line.measurements if not m.flag))
    assert joint.rv_err == pytest.approx(ivar, rel=0.25)


def test_failed_and_edge_lines_flagged():
    # high SNR, so that the edge window cannot settle on a noise dip instead
    spec, _ = synth.make_spectrum(rv=RV, snr=300., template=_template('gaussian'), rng=np.random.default_rng(3))
    regs = _line_regions(spec)
    line = regs[0]
    shift = 40.  # Å: a line-free copy of the window
    empty = replace(line, name='empty', wmin=line.wmin + shift, wmax=line.wmax + shift, centre=line.centre + shift,
                    continuum=tuple((a + shift, b + shift) for a, b in line.continuum))
    # the window ends 1 Å blueward of the true (redshifted) centre: the fit runs into the window edge
    edge = replace(line, name='edge', wmin=line.centre - 8., wmax=line.centre - 1.,
                   continuum=tuple(w for w in line.continuum if w[1] < line.centre))
    res = _measure(spec, [*regs, empty, edge], profile='gaussian')
    flags = {m.region: m.flag for m in res.measurements}
    assert flags['empty'] & (C.FLAG_FIT_FAILED | C.FLAG_EDGE)
    assert flags['edge'] & C.FLAG_EDGE
    assert abs(res.rv - RV) < 3. * res.rv_err  # the flagged lines are excluded


def test_unusable_inputs_raise():
    spec, _ = synth.make_spectrum(rv=RV, snr=30., template=_template('gaussian'), rng=np.random.default_rng(4))
    regs = _line_regions(spec)
    with pytest.raises(ValueError, match='profile'):
        _measure(spec, regs, profile='moffat')
    with pytest.raises(ValueError, match='no fittable line'):
        _measure(spec, [replace(r, continuum=()) for r in regs])


def _pulls(profile: str, n: int, seed: int = 100, **opts):
    """Pulls (RV − truth)/σ of the unflagged measurements (per line, or the joint one) over n synthetic spectra"""
    tpl = _template(profile)
    out = []
    for i in range(n):
        spec, _ = synth.make_spectrum(rv=RV, snr=30., template=tpl, rng=np.random.default_rng(seed + i))
        res = _measure(spec, _line_regions(spec), profile=profile, **opts)
        out += [(m.rv - RV) / m.rv_err for m in res.measurements if not m.flag]
    return np.array(out)


@pytest.mark.slow
@pytest.mark.parametrize(('profile', 'joint'), [(p, False) for p in sorted(linecentre.PROFILES)] + [('voigt', True)])
def test_pulls(profile, joint):
    """R 2500, SNR 30: the errors (fit covariance, √(χ²/ν) when > 1) are calibrated, per line and jointly"""
    pull = _pulls(profile, 200 if joint else 60, joint=joint)  # one pull per spectrum when joint
    log.info('%s%s: %d pulls, mean %.3f, std %.3f', profile, ' joint' if joint else '', len(pull), pull.mean(),
             pull.std(ddof=1))
    assert len(pull) > 190 if joint else len(pull) > 300
    assert abs(pull.mean()) < 0.2
    assert 0.8 < pull.std(ddof=1) < 1.2


J1745 = Path('pypeit/R2500I/GTC54-15A0/OB0055/Science')


@pytest.mark.data
@pytest.mark.slow
def test_j1745(caplog):
    """J1745−1640: per-line and method RVs without and with the default sky correction (reported, not tuned)"""
    from rvfitter import pipeline, results
    root = os.environ.get('RVFITTER_TESTDATA')
    if not root:
        pytest.skip('RVFITTER_TESTDATA not set')
    sci = Path(root) / J1745
    out = {}
    for mode in ('none', 'sky-if-needed'):
        c = config.Config()
        c.input.telluric_corrected = True
        c.velocity_frame.correction = 'skip'
        c.velocity_frame.frame = 'heliocentric'
        c.input.resolution = 2602.
        c.methods.enabled = ['linecentre']
        c.calibration.wavelength_correction = mode
        c.calibration.sky_source = str(next(sci.glob('spec1d_*.fits')))
        obj = pipeline.run(c, [str(sci / 'DENISJ174534.6-164053_tellcorr.fits')], n_procs=1)[0]
        m = obj.methods['linecentre']
        total = float(results.to_table([obj]).iloc[0]['rv_total_err'])
        for x in m.measurements:
            log.info('%s %s: %.2f ± %.2f km/s (flag %d)', mode, x.region, x.rv, x.rv_err, x.flag)
        log.info('%s: %.2f ± %.2f km/s (total %.2f; z %.2f)', mode, m.rv, m.rv_err, total,
                 abs(m.rv - 26.2) / np.hypot(2.3, total))
        assert m.n_regions >= 3 and np.isfinite(m.rv) and np.isfinite(total)
        out[mode] = m.rv
    # sky-if-needed refuses the sky curve here (the K I lines lie in its unmeasured O2 A gap): no change
    assert out['sky-if-needed'] == pytest.approx(out['none'])
