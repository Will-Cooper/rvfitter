"""Tests of the OH sky-line wavelength zero point, its application in the pipeline and the io sky column."""
import numpy as np
import pytest

from rvfitter import broadening, config, io, pipeline
from rvfitter.calibration import sky_zeropoint as Z
from rvfitter.types import Spectrum


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
    for lam in np.linspace(zp.wmin_fit, zp.wmax_fit, 5):
        assert abs(float(zp.dv(lam)) - _dv_true(lam)) < 2., (lam, float(zp.dv(lam)), _dv_true(lam))
    assert any(s['reason'].startswith('telluric') for s in zp.segments)  # O2 A / H2O segments rejected


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


def test_pipeline_wavelength_correction_modes():
    raw = _sky_spectrum()
    cfg = config.Config()
    spec, info = pipeline.wavelength_correction(raw, raw, cfg, 'X')
    assert info['applied'] == 'sky' and abs(info['zp_sky_kms'] - _dv_true(8400.)) < 3.
    assert 'wavelength_correction' in spec.meta
    cfg.calibration.wavelength_correction = 'none'
    spec2, info2 = pipeline.wavelength_correction(raw, raw, cfg, 'X')
    assert spec2 is raw and info2['applied'] == 'none'
    cfg.calibration.wavelength_correction = 'sky'
    nosky = Spectrum(raw.wave, raw.flux, raw.sigma, raw.resolution, {}, 'x')
    _, info3 = pipeline.wavelength_correction(nosky, nosky, cfg, 'X')
    assert info3['applied'] == 'none'


def test_pipeline_sky_source_file(tmp_path):
    raw = _sky_spectrum()
    path = tmp_path / 'sky.csv'
    np.savetxt(path, np.c_[raw.wave, raw.flux, raw.sigma, raw.sky], delimiter=',', header='wave,flux,err,sky',
               comments='')
    read = io.read_spectrum(path, medium='vacuum')
    assert read.sky is not None and np.allclose(read.sky, raw.sky)
    cfg = config.Config()
    cfg.input.medium = 'vacuum'
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
    big = Spectrum(spec.wave, spec.flux, spec.sigma, 1600., {**spec.meta, 'wave_rms': 2.}, 'x')
    assert systematics.wavelength_zero_point(big, {}) == pytest.approx(299792.458 * 2. / np.median(spec.wave))
