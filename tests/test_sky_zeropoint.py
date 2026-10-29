"""Tests of the OH sky-line wavelength zero point, its application in the pipeline and the io sky column."""
from dataclasses import replace

import numpy as np
import pytest

from rvfitter import broadening, config, io, pipeline
from rvfitter.calibration import sky_zeropoint as Z
from rvfitter.types import Region, Spectrum


def _dv_true(w):
    return 15. + 12. * (w - 8400.) / 1000.  # km/s: offset plus a linear trend


def _sky_spectrum(dv_func=_dv_true, fwhm=5.5, snr=200., frame_kms=0., rng=None):
    """A spectrum whose sky lines appear at λ_true × D(Δv(λ)) (and × D(frame_kms) if a correction was applied)"""
    rng = rng or np.random.default_rng(0)
    wave_obs = broadening.log_wave_grid(7300., 10100., 299792.458 / 1600. / 2.5)
    wave_true = wave_obs / broadening.doppler_factor(dv_func(wave_obs))
    sky = Z.model_sky(wave_true, fwhm) * 100. + 5.
    sky = sky + rng.normal(0., 1., len(sky)) * np.sqrt(sky) * 100. / snr / 10.
    meta = {}
    if frame_kms:
        f = broadening.doppler_factor(frame_kms)
        wave_obs = wave_obs * f
        meta = {'vel_corr': f, 'vel_corr_kms': frame_kms, 'frame_in': 'heliocentric'}
    return Spectrum(wave_obs, np.ones_like(wave_obs), np.full_like(wave_obs, 0.01), 1600., meta, 'synthetic', sky)


def test_oh_list_is_vacuum_and_cited():
    w, a = Z.oh_lines()
    assert len(w) > 150 and w.min() < 6500 and w.max() > 10500 and a.max() == 1.
    from importlib import resources
    text = resources.files('rvfitter').joinpath('data', 'oh_sky_vacuum.csv').read_text()
    assert 'Rousselot' in text and 'BSD' in text
    assert resources.files('rvfitter').joinpath('data', 'PYPEIT_LICENSE.txt').is_file()


@pytest.mark.parametrize('frame_kms', [0., -21.])
def test_recovers_zero_point_curve(frame_kms):
    spec = _sky_spectrum(frame_kms=frame_kms)
    zp = Z.measure(spec)
    assert zp is not None and zp.n_used >= 3
    assert abs(zp.fwhm - 5.5) < 0.6
    for lam in np.linspace(*zp.main_span, 5):
        err = float(zp.dv_err(lam)[0])
        assert abs(float(zp.dv(lam)) - _dv_true(lam)) < 3. * err, (lam, float(zp.dv(lam)), _dv_true(lam), err)
    # the strong telluric bands are cut out before segmenting: no segment overlaps O2 A or the 9300 Å H2O band
    from rvfitter.features import features
    tell = [(f.wmin, f.wmax) for f in features() if f.kind == 'telluric' and f.depth >= 0.3]
    assert not any(s['wmin'] < b and s['wmax'] > a for s in zp.segments for a, b in tell)
    assert len(set(zp.node_span)) >= 2


def test_curve_is_local_per_span():
    from rvfitter.calibration.sky_zeropoint import _curve
    nw, nd, ne, ns = [7500., 7800., 8000., 8200.], [-100., 10., 12., 14.], [1., 1., 1., 1.], [0, 1, 1, 1]
    bounds = {0: (7400., 7600.), 1: (7700., 8300.)}
    dv, err = _curve([7750., 7710., 7650., 7550., 8250.], nw, nd, ne, ns, 0.1, bounds)
    assert dv[0] == 10. and dv[1] == 10.  # held at span 1's first node out to its boundary: not bridged
    assert dv[2] == pytest.approx(-45.)  # inside the unmeasured gap: a ramp keeps the curve continuous
    assert dv[3] == -100. and dv[4] == 14.
    assert err[1] > err[0] > 1.  # grows with the distance from the nearest node


def test_no_sky_returns_none():
    spec = _sky_spectrum()
    assert Z.measure(Spectrum(spec.wave, spec.flux, spec.sigma)) is None


def test_apply_and_applied_dv():
    spec = _sky_spectrum()
    zp = Z.measure(spec)
    out = Z.apply(spec, zp, constant_kms=3.)
    lam = 8500.
    i = np.argmin(np.abs(spec.wave - lam))
    np.testing.assert_allclose(out.wave[i], spec.wave[i] / broadening.doppler_factor(float(zp.dv(spec.wave[i])) + 3.))
    np.testing.assert_allclose(Z.applied_dv(out.meta, spec.wave[i]), float(zp.dv(spec.wave[i])) + 3.)
    assert out.meta['wavelength_correction']['source'] == 'sky'
    assert np.all(np.diff(out.wave) > 0)


def test_pipeline_wavelength_correction_modes(caplog):
    import logging
    caplog.set_level(logging.INFO, logger='rvfitter')
    raw = _sky_spectrum()
    cfg = config.Config()
    assert cfg.calibration.wavelength_correction == 'sky-if-needed'  # the default
    cfg.calibration.wavelength_correction = 'none'
    spec0, info0 = pipeline.wavelength_correction(raw, raw, cfg, 'X')
    assert info0['applied'] == 'none' and np.isfinite(info0['zp_sky_kms'])  # measured and reported anyway
    assert np.array_equal(spec0.wave, raw.wave) and 'sky_zeropoint' in spec0.meta
    assert 'no correction is applied (mode none) and that offset is not in the errors' in caplog.text
    from rvfitter import systematics
    # a measured offset's value is never an error: only the unapplied curve's uncertainty is counted
    assert systematics.wavelength_zero_point(spec0, {}) == pytest.approx(info0['zp_sky_err'])
    assert systematics.wavelength_zero_point(spec0, {}) < abs(info0['zp_sky_kms'])
    cfg.calibration.wavelength_correction = 'sky'
    spec, info = pipeline.wavelength_correction(raw, raw, cfg, 'X')
    assert info['applied'] == 'sky' and abs(info['zp_sky_kms'] - np.mean(_dv_true(np.linspace(7900., 8800.)))) < 4.
    assert 'wavelength_correction' in spec.meta and not np.array_equal(spec.wave, raw.wave)
    assert np.isnan(info['zp_star_sky'])
    cfg.calibration.wavelength_correction = 'sky'
    nosky = Spectrum(raw.wave, raw.flux, raw.sigma, raw.resolution, {}, 'x')
    _, info3 = pipeline.wavelength_correction(nosky, nosky, cfg, 'X')
    assert info3['applied'] == 'none' and info3['zp_flags'] & pipeline.ZP_FLAGS['no_sky']


ALKALI = [Region('Na I', 'line', 8175., 8205.), Region('Rb I', 'line', 7940., 7960.),
          Region('Cs I', 'line', 8515., 8530.)]


def test_sky_if_needed_applies_a_significant_offset(caplog):
    import logging
    caplog.set_level(logging.INFO, logger='rvfitter')
    raw = _sky_spectrum()
    cfg = config.Config()
    spec, info = pipeline.wavelength_correction(raw, raw, cfg, 'X', regions=ALKALI)
    t = info['sky_tests']
    assert info['applied'] == 'sky' and info['zp_flags'] & pipeline.ZP_FLAGS['sky_applied']
    assert t['coverage_ok'] and t['significant'] and t['above_floor'] and t['n_segments'] >= 3
    assert t['n_sigma'] > 3. and t['max_correction_kms'] > t['floor_kms']
    assert 'applying' in caplog.text and not np.array_equal(spec.wave, raw.wave)


def test_sky_if_needed_skips_a_negligible_offset():
    raw = _sky_spectrum(dv_func=lambda w: np.zeros_like(w))
    cfg = config.Config()
    spec, info = pipeline.wavelength_correction(raw, raw, cfg, 'X', regions=ALKALI)
    assert info['applied'] == 'none' and info['zp_flags'] & pipeline.ZP_FLAGS['sky_not_needed']
    assert np.array_equal(spec.wave, raw.wave) and np.isfinite(info['zp_sky_kms'])
    # the floor alone: a 1 km/s offset at R 1600 (0.05 resolution elements = 9.4 km/s) is never applied
    raw1 = _sky_spectrum(dv_func=lambda w: np.ones_like(w), snr=5000.)
    _, info1 = pipeline.wavelength_correction(raw1, raw1, cfg, 'X', regions=ALKALI)
    assert info1['applied'] == 'none' and not info1['sky_tests']['above_floor']


def test_sky_if_needed_refuses_unmeasured_regions():
    """A region in an unmeasured part of the curve (here inside the O2 A gap) blocks the correction"""
    raw = _sky_spectrum()
    cfg = config.Config()
    spec, info = pipeline.wavelength_correction(raw, raw, cfg, 'X', regions=[*ALKALI,
                                                                             Region('K I', 'line', 7660., 7675.)])
    assert info['applied'] == 'none' and info['zp_flags'] & pipeline.ZP_FLAGS['sky_unsafe']
    assert info['sky_tests']['region_measured']['K I'] == 0. and not info['sky_tests']['coverage_ok']
    _, none = pipeline.wavelength_correction(raw, raw, cfg, 'X')  # no regions: nothing to vouch for
    assert none['applied'] == 'none'


def test_valid_range_drops_segments_beyond_the_arcs():
    raw = _sky_spectrum()
    full = Z.measure(raw)
    cut = Z.measure(raw, valid_range=(7400., 9500.))
    assert cut.n_used < full.n_used
    assert all(s['reason'].startswith('outside the calibrated range') for s in cut.segments
               if not 7400. <= s['centre'] <= 9500.)


def test_pipeline_sky_source_file(tmp_path):
    raw = _sky_spectrum()
    path = tmp_path / 'sky.csv'
    np.savetxt(path, np.c_[raw.wave, raw.flux, raw.sigma, raw.sky], delimiter=',', header='wave,flux,err,sky',
               comments='')
    read = io.read_spectrum(path, medium='vacuum')
    assert read.sky is not None and np.allclose(read.sky, raw.sky)
    cfg = config.Config()
    cfg.input.medium = 'vacuum'
    cfg.calibration.wavelength_correction = 'sky'
    cfg.calibration.sky_source = str(path)
    nosky = Spectrum(raw.wave, raw.flux, raw.sigma, raw.resolution, {}, 'x')
    _, info = pipeline.wavelength_correction(nosky, nosky, cfg, 'X')
    assert info['applied'] == 'sky'


def test_wavelength_zero_point_systematic():
    from rvfitter import systematics
    raw = _sky_spectrum()
    spec = Z.apply(raw, Z.measure(raw))
    sys_kms = systematics.wavelength_zero_point(spec, {})
    assert sys_kms == pytest.approx(spec.meta['wavelength_correction']['sky']['zp_sky_err'])
    # at the measurements' wavelengths: the curve's own error there, with the curve-model spread
    lam = np.array([8190., 8520.])
    c = spec.meta['wavelength_correction']['sky']
    at = systematics.wavelength_zero_point(spec, {}, wave=lam)
    expect = np.hypot(np.sqrt(np.mean(Z.curve_at(c, lam)[1] ** 2)), c['zp_sky_model_spread'])
    assert at == pytest.approx(expect)
    # not applied: only the curve's uncertainty there, never the offset's value
    c2 = Z.measure(raw).as_dict()
    raw2 = replace(raw, meta={**raw.meta, 'sky_zeropoint': c2})
    unapplied = systematics.wavelength_zero_point(raw2, {}, wave=lam)
    assert unapplied == pytest.approx(np.hypot(np.sqrt(np.mean(Z.curve_at(c2, lam)[1] ** 2)),
                                               c2['zp_sky_model_spread']))
    assert unapplied < np.min(np.abs(_dv_true(lam)))
    big = Spectrum(spec.wave, spec.flux, spec.sigma, 1600., {**spec.meta, 'wave_rms': 2.}, 'x')
    assert systematics.wavelength_zero_point(big, {}) == pytest.approx(299792.458 * 2. / np.median(spec.wave))
