"""
The inputs of the absolute zero-point budget: io's sky cleaning and slit metadata, the slit-centring prior, the sky
segments' synthetic bias test, the configuration, and (marked data) J1745−1640's slit prior and sky segments.
The pipeline behaviour (sky+star, slit prior, never a measured value as an error) is tested in test_telluric.py.
"""
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from rvfitter import broadening, config, io
from rvfitter.calibration import sky_zeropoint as Z
from rvfitter.calibration import slit
from rvfitter.types import Spectrum

C = 299792.458


def _pypeit_file(path: Path, sky: np.ndarray, decker='LongSlits1.23Arcsec', binning='2,2', fwhm=3.06, **extra):
    n = len(sky)
    wave = 7300. + 1.39 * np.arange(n)
    cols = [fits.Column(name='OPT_WAVE', format='D', array=wave),
            fits.Column(name='OPT_FLAM', format='D', array=np.ones(n)),
            fits.Column(name='OPT_FLAM_IVAR', format='D', array=np.full(n, 100.)),
            fits.Column(name='OPT_MASK', format='L', array=np.ones(n, dtype=bool)),
            fits.Column(name='OPT_COUNTS_SKY', format='D', array=sky),
            fits.Column(name='FWHMFIT', format='D', array=np.full(n, fwhm))]
    tab = fits.BinTableHDU.from_columns(cols, name='SPAT0240-SLIT0457-DET02')
    tab.header['FWHM'] = fwhm
    prim = fits.PrimaryHDU()
    prim.header.update(DECKER=decker, BINNING=binning, INSTRUME='OSIRIS', PYP_SPEC='gtc_osiris', **extra)
    fits.HDUList([prim, tab]).writeto(path)
    return wave


def test_io_drops_absurd_sky_and_reads_slit(tmp_path):
    sky = np.full(400, 50.) + np.random.default_rng(0).normal(0., 1., 400)
    sky[10], sky[20] = 1e10, np.inf
    _pypeit_file(tmp_path / 's.fits', sky)
    s = io.read_spectrum(tmp_path / 's.fits')
    assert np.isnan(s.sky[10]) and np.isnan(s.sky[20]) and np.isfinite(s.sky).sum() == 398
    m = s.meta
    assert m['slit_width_arcsec'] == 1.23 and m['slit_width_key'] == 'DECKER'
    assert (m['binning_spec'], m['binning_spat']) == (2, 2) and m['instrument'] == 'OSIRIS'
    assert m['trace_fwhm_pix'] == pytest.approx(3.06) and m['pypeit_spec'] == 'gtc_osiris'
    # SLITW (OSIRIS raw frames) wins over a mask name
    _pypeit_file(tmp_path / 't.fits', sky, SLITW=0.8, MASKNAME='slit 1.0 arcsec')
    t = io.read_spectrum(tmp_path / 't.fits')
    assert t.meta['slit_width_arcsec'] == 0.8 and t.meta['slit_width_key'] == 'SLITW'
    _pypeit_file(tmp_path / 'u.fits', sky, decker='Custom', MASKNAME='Long slit 1.0 arcsec')
    assert io.read_spectrum(tmp_path / 'u.fits').meta['slit_width_key'] == 'MASKNAME'


def test_slit_prior_formula_and_missing_inputs():
    wave = 7300. + 1.39 * np.arange(2000)
    meta = dict(slit_width_arcsec=1.23, slit_width_key='DECKER', binning_spec=2, binning_spat=2, instrument='OSIRIS',
                trace_fwhm_pix=3.06)
    spec = Spectrum(wave, np.ones(len(wave)), np.ones(len(wave)), None, meta, 'x')
    inp = slit.inputs([None, spec])
    assert inp.scale_spec == pytest.approx(0.254) and inp.fwhm_arcsec == pytest.approx(3.06 * 0.254)
    sig = (1.23 - 3.06 * 0.254) / np.sqrt(12.)
    assert inp.sigma_arcsec == pytest.approx(sig)
    assert slit.prior_kms(inp, 8500.) == pytest.approx(C * sig / 0.254 * 1.39 / 8500.)
    assert slit.instrument_pixscale('gtc_osiris_plus', 'OSIRIS')[0] == 0.125
    # the seeing overfills the slit, or an input is missing: NaN (the pipeline warns)
    wide = slit.inputs([Spectrum(wave, wave, wave, None, {**meta, 'trace_fwhm_pix': 6.}, 'x')])
    assert np.isnan(slit.prior_kms(wide, 8500.))
    bare = slit.inputs([Spectrum(wave, wave, wave, None, {'instrument': 'UNKNOWN'}, 'x')])
    assert np.isnan(slit.prior_kms(bare, 8500.)) and len(bare.missing) == 3
    # a header plate scale is used as is
    own = slit.inputs([Spectrum(wave, wave, wave, None, {**meta, 'instrument': None, 'pixscale_arcsec': 0.2}, 'x')])
    assert own.scale_spec == 0.2 and own.fwhm_arcsec == pytest.approx(3.06 * 0.2)


def _sky(wave, fwhm=5.5):
    return Z.model_sky(wave, fwhm) * 100. + 5.


def test_sky_bias_test_downweights_and_excludes(monkeypatch):
    wave = broadening.log_wave_grid(7300., 10100., C / 1600. / 2.5)
    spec = Spectrum(wave, np.ones(len(wave)), np.ones(len(wave)), None, {}, 'x', _sky(wave))
    zp = Z.measure(spec)
    used = [s for s in zp.segments if s['used']]
    assert used and all(np.isfinite(s['bias']) and s['err'] >= s['err_stat'] for s in used)
    for s in used:  # the bias is added in quadrature
        assert s['err'] == pytest.approx(np.hypot(s['err_stat'], s['bias']))
    assert all(abs(s['dv']) < s['err'] for s in used)  # noise-free and unshifted: truth 0
    # the short segment ending at the O2 A band is the most model-sensitive (as on J1745's sky)
    assert max(used, key=lambda s: s['bias'])['wmax'] == pytest.approx(7592., abs=5.)
    monkeypatch.setattr(Z, 'BIAS_FAIL_RESEL', 1e-6)  # every segment fails
    assert Z.measure(spec) is None
    monkeypatch.undo()
    off = Z.measure(spec, bias_test=False)
    assert all(np.isnan(s['bias']) and s['err'] == s['err_stat'] for s in off.segments if s['used'])


def test_measure_windows_inside_telluric_bands():
    """The sky in any window, e.g. a telluric band the curve excludes, with the same correlation and errors"""
    wave = broadening.log_wave_grid(7300., 10100., C / 1600. / 2.5)
    true = 18.
    spec = Spectrum(wave, np.ones(len(wave)), np.ones(len(wave)), None, {}, 'x',
                    _sky(wave / broadening.doppler_factor(true)))
    zp = Z.measure(spec)
    res = Z.measure_windows(spec, [(9180., 9415.), (8122., 8352.), (7000., 7010.)], zp.fwhm)
    assert res[0]['used'] and abs(res[0]['dv'] - true) < 3. * res[0]['err'] + 1.
    assert res[1]['used'] and abs(res[1]['dv'] - true) < 3. * res[1]['err'] + 1.
    assert not res[2]['used'] and 'sky pixels' in res[2]['reason']
    # the curve itself never contains a segment inside the strong bands
    assert not any(s['wmin'] < 9300. < s['wmax'] for s in zp.segments)


def test_config_slit_prior_validation():
    cfg = config.Config()
    assert cfg.calibration.slit_prior == 'auto' and cfg.calibration.zero_point_known is False
    cfg.calibration.slit_prior = 12.
    cfg.validate()
    cfg.calibration.slit_prior = 'wide'
    with pytest.raises(config.ConfigError, match='slit_prior'):
        cfg.validate()
    cfg.calibration.slit_prior = -1.
    with pytest.raises(config.ConfigError, match='slit_prior'):
        cfg.validate()
    loaded = config.from_dict({'calibration': {'slit_prior': 20, 'zero_point_known': True}})
    assert loaded.calibration.slit_prior == 20. and loaded.calibration.zero_point_known


# --------------------------------------------------------------------------------------------------------- J1745
J1745 = ('pypeit/R2500I/GTC54-15A0/OB0055/Science/spec1d_0000809229-20150803-OSIRIS-OsirisLongSlitSpectroscopy-'
         'DENISJ174534.6-164053_OSIRIS_20150804T003233.951.fits')


@pytest.fixture
def testdata():
    root = os.environ.get('RVFITTER_TESTDATA')
    if not root or not (Path(root) / J1745).is_file():
        pytest.skip('RVFITTER_TESTDATA not set or J1745 missing')
    return Path(root)


@pytest.mark.data
def test_j1745_slit_prior(testdata):
    """J1745 (HDU 2, the target): 1.23 arcsec slit (DECKER), FWHM 3.06 binned pixels, 2x2 binning, 0.254 arcsec/pix"""
    s = io.read_spectrum(testdata / J1745, hdu=2)
    inp = slit.inputs([s])
    assert inp.width_arcsec == 1.23 and 'DECKER' in inp.width_source
    assert inp.fwhm_arcsec == pytest.approx(3.06 * 0.254, abs=0.01) and inp.scale_spec == pytest.approx(0.254)
    prior = slit.prior_kms(inp, np.array([7802., 7949., 8190., 8523.]))
    assert 20. < prior < 30.  # (1.23 - 0.78)/√12 = 0.13 arcsec = 0.51 pixel ≈ 0.71 Å


@pytest.mark.data
def test_j1745_sky_segments(testdata):
    """J1745's sky: the short blue-edge segment (next to O2 A) is the most model-sensitive and is down-weighted"""
    s = io.read_spectrum(testdata / J1745, hdu=1)
    zp = Z.measure(s, valid_range=(7437.4, 9925.9))
    used = [g for g in zp.segments if g['used']]
    assert len(used) >= 5
    blue = min(used, key=lambda g: g['wmin'])
    assert blue['wmax'] < 7600. and blue['bias'] > blue['err_stat'] and blue['err'] > 2. * blue['err_stat']
    assert blue['bias'] == max(g['bias'] for g in used)
