"""
Tests of the telluric transmission model, the telluric (star's own) wavelength zero point, the star − sky constant,
the 'telluric'/'auto' wavelength corrections and the telluric_ref method.

The synthetic spectra are built on a 100 000-resolution grid in the observed frame: tellurics at λ_t·D(Δv(λ)),
stellar lines at λ_rest·D(rv)·D(Δv(λ)) (the same distortion, as for light sharing the slit position), multiplied,
convolved with the LSF and averaged over the pixels.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from rvfitter import broadening, config, medium, pipeline
from rvfitter.calibration import sky_zeropoint, telluric
from rvfitter.calibration import telluric_zeropoint as TZ
from rvfitter.types import Spectrum
from synth import make_template

R = 2500.
AIRMASS, PWV = 1.6, 4.


def _dv_true(w):
    return 10. + 20. * (np.asarray(w) - 8500.) / 1000.  # km/s: a zero point plus a distortion


def _observed(dv_func=_dv_true, rv_star=None, snr=80., rng=None, frame_kms=0., with_sky=False,
              star=None) -> Spectrum:
    """A synthetic observed spectrum (7300–10 150 Å, 1.39 Å pixels) with tellurics and optionally a star"""
    rng = rng or np.random.default_rng(1)
    wave = 7300. * np.exp(np.arange(2050) * 1.39 / 8600.)
    step = 299792.458 / 100_000.
    hr = broadening.log_wave_grid(wave[0] - 20., wave[-1] + 20., step)
    f_dv = broadening.doppler_factor(dv_func(hr))
    flux_hr = telluric.transmission(hr / f_dv, AIRMASS, PWV) * (1. + 0.3 * (hr - 8500.) / 1500.)
    if rv_star is not None:
        sw, sf = star
        flux_hr = flux_hr * np.interp(hr / f_dv / broadening.doppler_factor(rv_star), sw, sf)
    flux_hr = broadening.broaden_loglam(flux_hr, step, resolution=R)
    flux = broadening.rebin(hr, flux_hr, wave)
    sigma = flux / snr
    flux = flux + rng.normal(0., 1., len(wave)) * sigma
    meta = {'airmass': AIRMASS}
    sky = None
    if with_sky:
        sky = sky_zeropoint.model_sky(wave / broadening.doppler_factor(dv_func(wave)), 5.5) * 100. + 5.
    if frame_kms:
        f = broadening.doppler_factor(frame_kms)
        wave = wave * f
        meta.update(vel_corr=f, vel_corr_kms=frame_kms, frame_in='heliocentric')
    return Spectrum(wave, flux, sigma, None, meta, 'synthetic', sky)


# ----------------------------------------------------------------------------------------------------- model
def test_model_is_vacuum_and_cited():
    m = telluric.model()
    assert m.wmin == pytest.approx(6000., abs=1.) and m.wmax == pytest.approx(25000., abs=1.)
    assert 'HITRAN' in m.meta['citation'] and 'citepolicy' in m.meta['licence']
    assert {'H2O', 'O2', 'CO2', 'CH4'} <= set(m.meta['molecules'])
    # the blue edges of the O2 A and B bands sit just redward of the Fraunhofer A and B lines in vacuum, and far
    # from their air values (an air-wavelength model would put them ~2 Å bluer)
    w = broadening.log_wave_grid(6800., 7700., 299792.458 / 400_000.)
    t = telluric.transmission(w, 1., 2.5, resolution=20_000.)
    for lo, air in ((7580., 7593.70), (6855., 6867.19)):
        s = w > lo
        edge = w[s][np.argmax(t[s] < 0.5)]
        vac = float(medium.air_to_vac(np.array([air]))[0])
        assert 0. < edge - vac < 1.6, (edge, vac)
        assert edge - air > 2.5


def test_transmission_scaling_shift_and_mask():
    w = broadening.log_wave_grid(7550., 7750., 299792.458 / 200_000.)
    t1 = telluric.transmission(w, 1., 2.5, resolution=R)
    t2 = telluric.transmission(w, 2., 2.5, resolution=R)
    assert t2.min() < t1.min() < 0.5 and np.all(t1 <= 1. + 1e-9)
    # a positive dv moves the band to the red
    ts = telluric.transmission(w, 1., 2.5, resolution=R, dv=50.)
    edge = lambda tt: w[np.argmax(tt < 0.6)]  # noqa: E731
    assert edge(ts) - edge(t1) == pytest.approx(7600. * 50. / 299792.458, abs=0.3)
    assert telluric.mask(w, 0.3, resolution=R).any() and not telluric.mask(np.array([8000.5, 8001.]), 0.3).any()
    assert np.all(telluric.transmission(np.array([5000., 30000.]), 1., 2.5) == 1.)  # outside the coverage


# ---------------------------------------------------------------------------------------- telluric zero point
@pytest.mark.parametrize('frame_kms', [0., -21.])
def test_recovers_zero_point_and_distortion(frame_kms):
    spec = _observed(frame_kms=frame_kms)
    zp = TZ.measure(spec, resolution=R)
    assert zp is not None and zp.n_used >= 5
    for s in zp.segments:
        if s['used']:
            assert abs(s['dv'] - _dv_true(s['centre'])) < 3.5 * s['err'] + 0.5, s
    assert abs(zp.dv(9300.) - _dv_true(9300.)) < 4.
    dv, err = zp  # unpacks as the architecture's (dv, err)
    assert np.isfinite(dv) and err > 0


def test_estimates_resolution_from_tellurics():
    zp = TZ.measure(_observed())
    assert zp.resolution_source == 'telluric lines' and abs(zp.resolution / R - 1.) < 0.1


def test_star_minus_sky_constant():
    spec = _observed(with_sky=True)
    sky = sky_zeropoint.measure(spec)
    # the star sits 12 km/s further to the red than the sky (e.g. off-centre in the slit)
    star = _observed(dv_func=lambda w: _dv_true(w) + 12.)
    tell = TZ.measure(star, resolution=R)
    ss = TZ.star_minus_sky(tell, sky)
    assert abs(ss['zp_star_sky'] - 12.) < 3. * ss['zp_star_sky_err'] + 1., ss
    assert len(ss['nodes']) == tell.n_used


def test_pipeline_telluric_and_auto_modes():
    raw = _observed(with_sky=True)
    cfg = config.Config()
    cfg.input.resolution = R
    cfg.calibration.telluric_template = False
    spec0, info0 = pipeline.wavelength_correction(raw, raw, cfg, 'X')  # 'none': the telluric term is not measured
    assert info0['applied'] == 'none' and np.isnan(info0['zp_tell_kms'])
    cfg.calibration.wavelength_correction = 'telluric'
    spec, info = pipeline.wavelength_correction(raw, raw, cfg, 'X')
    assert info['applied'] == 'telluric' and np.isfinite(info['zp_tell_kms'])
    assert np.isfinite(info['zp_star_sky']) and abs(info['zp_star_sky']) < 3. * info['zp_star_sky_err'] + 2.
    lam = 9300.
    corrected = float(np.interp(lam, raw.wave, spec.wave))
    assert (lam / corrected - 1.) * 299792.458 == pytest.approx(_dv_true(lam), abs=5.)
    cfg.calibration.wavelength_correction = 'auto'
    spec_a, info_a = pipeline.wavelength_correction(raw, raw, cfg, 'X')
    assert info_a['applied'] == 'sky+telluric'
    wc = spec_a.meta['wavelength_correction']
    assert wc['constant_kms'] == pytest.approx(info_a['zp_star_sky']) and wc['sky'] is not None
    from rvfitter import systematics
    assert systematics.wavelength_zero_point(spec_a, {}) >= info_a['zp_star_sky_err'] - 1e-9
    nosky = Spectrum(raw.wave, raw.flux, raw.sigma, None, raw.meta, 'x')
    _, info_t = pipeline.wavelength_correction(nosky, nosky, cfg, 'X')
    assert info_t['applied'] == 'telluric'


# ------------------------------------------------------------------------------------------------ telluric_ref
@pytest.fixture(scope='module')
def star():
    return make_template((7000., 10500.), n_random=400, rng=np.random.default_rng(3))


def test_telluric_ref_recovers_rv_despite_distortion(star):
    from rvfitter.methods import REGISTRY
    rv_true, frame = 30., -21.
    spec = _observed(rv_star=rv_true, star=star, frame_kms=frame, snr=100.)
    tmpl = Spectrum(star[0], star[1], np.full(len(star[0]), np.nan), None, {'object': 'toy'}, 'toy')
    cfg = {'input': {'resolution': R}, 'methods': {'options': {'telluric_ref': {'template': tmpl}}}}
    res = REGISTRY['telluric_ref'].measure(spec, [], cfg, rng=np.random.default_rng(0))
    assert res.n_regions >= 4
    expected = float(299792.458 * np.tanh(np.arctanh(rv_true / 299792.458) + np.arctanh(frame / 299792.458)))
    assert abs(res.rv - expected) < 3. * res.rv_err + 0.5, (res.rv, res.rv_err, expected)
    m = [x for x in res.measurements if np.isfinite(x.rv)]
    # each segment's telluric zero point is the injected distortion (observatory frame)
    assert all(abs(x.diagnostics['dv'] - _dv_true(x.diagnostics['wave_centre'])) < 4. * x.diagnostics['err'] + 1.
               for x in m)


# --------------------------------------------------------------------------------------------------- pull test
def _effective_truth(seg: dict) -> float:
    """
    The Δv a segment measures when Δv varies across it: the mean over its pixels weighted by the shift
    information (dT/dλ)² of the telluric lines, not the value at the segment centre
    """
    pad = 15. * 0.5 * (seg['wmin'] + seg['wmax']) / R
    w = 7300. * np.exp(np.arange(2050) * 1.39 / 8600.)
    w = w[(w >= seg['wmin'] - pad) & (w <= seg['wmax'] + pad)]
    info = np.gradient(telluric.transmission(w, AIRMASS, PWV, resolution=R), w) ** 2
    return float(np.sum(info * _dv_true(w)) / np.sum(info))


@pytest.mark.slow
@pytest.mark.parametrize('distorted', [False, True])
def test_injection_pulls(distorted):
    """A constant zero point, or the zero point plus distortion, in 30 noise realisations: bias and pull width"""
    pulls = []
    for k in range(30):
        func = _dv_true if distorted else (lambda w: np.full(np.shape(w), 15.))
        zp = TZ.measure(_observed(dv_func=func, rng=np.random.default_rng(100 + k), snr=60.), resolution=R)
        for s in zp.segments:
            if s['used'] and s['err'] < 20.:
                truth = _effective_truth(s) if distorted else 15.
                pulls.append((s['dv'] - truth) / s['err'])
    pulls = np.asarray(pulls)
    assert len(pulls) > 150
    assert abs(np.mean(pulls)) < 0.25 and 0.7 < np.std(pulls) < 1.3, (np.mean(pulls), np.std(pulls))


# --------------------------------------------------------------------------------------------------- J1745 data
REL = Path('pypeit/R2500I/GTC54-15A0/OB0055/sciencebk/nopipecorr/'
           'spec1d_0000809229-20150803-OSIRIS-OsirisLongSlitSpectroscopy-DENISJ174534.6-164053_OSIRIS_20150804T003233'
           '.951.fits')
TELLCORR = Path('pypeit/R2500I/GTC54-15A0/OB0055/Science/DENISJ174534.6-164053_tellcorr.fits')


@pytest.mark.data
def test_j1745_telluric_zero_point():
    root = os.environ.get('RVFITTER_TESTDATA')
    if not root:
        pytest.skip('RVFITTER_TESTDATA not set')
    from rvfitter import io
    target = io.read_spectrum(Path(root) / TELLCORR, medium='vacuum')
    cfg = config.Config()
    cfg.input.medium = 'vacuum'
    # the spec1d holds two objects; the one matching the (target) coadd is HDU 2, not the first
    src = pipeline._matching_object(str(Path(root) / REL), target, cfg, 'J1745')
    assert src.source.endswith('.fits') and np.nanmedian(src.flux / src.sigma) > 50.
    zp = TZ.measure(src, resolution=2602.)
    assert zp is not None and zp.n_used >= 5
    assert all(abs(s['dv']) < 100. for s in zp.segments if s['used'])
