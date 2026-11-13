"""DAR across the slit: the OSIRIS PA convention, the parallactic angle, and a + s·D(λ) recovered from star − sky."""
import os
from pathlib import Path

import numpy as np
import pytest

from rvfitter import io
from rvfitter.calibration import dar, slit
from rvfitter.calibration import telluric_zeropoint as TZ

TESTDATA = os.environ.get('RVFITTER_TESTDATA')
RAW = Path(TESTDATA or '.') / 'raw_headers/R2500I/GTC54-15A0/OB0055'


def test_parallactic_angle_and_refraction():
    lat = np.radians(28.76)
    assert dar.parallactic_deg(np.array([0.]), np.radians(-16.6), lat)[0] == pytest.approx(0.)  # south of zenith
    assert dar.parallactic_deg(np.array([0.]), np.radians(60.), lat)[0] == pytest.approx(180.)  # north of zenith
    assert dar.parallactic_deg(np.array([0.5]), np.radians(-16.6), lat)[0] > 0.  # west of the meridian: q > 0
    r = dar.refraction_arcsec(np.array([7000., 8000., 10000.]), 60., 780., 10., 0.3)
    assert np.all(np.diff(r) < 0) and 0.1 < r[0] - r[-1] < 0.5  # the blue is refracted more


def _inputs():
    w = np.linspace(7000., 10000., 2000)
    return slit.SlitInputs(width_arcsec=1.23, fwhm_arcsec=0.78, scale_spat=0.254, scale_spec=0.254, disp_wave=w,
                           disp=np.gradient(w))


def _geo(s=0.3, s_err=0.):
    return dict(s=s, s_err=s_err, pa_known=True, zenith_deg=58., pressure_hpa=780., temp_c=12., rh=0.3)


def test_dar_velocity_scale():
    """D(λ) = c g ΔR / s_spec · d / λ: zero at the reference, positive in the blue, damped by the slit (g < 1)"""
    inp = _inputs()
    d = dar.dar_kms(_geo(), inp, [7700., 9067., 9800.], 9067.)
    assert d[1] == 0. and d[0] > 15. and d[2] < 0.
    g = slit.centroid_gain(1.23, 0.78)
    assert 0.5 < g < 0.9 and slit.centroid_gain(10., 0.5) == pytest.approx(1., abs=1e-6)
    r = dar.refraction_arcsec(np.array([7700., 9067.]), 58., 780., 12., 0.3)
    assert d[0] == pytest.approx(299792.458 * g * (r[0] - r[1]) / 0.254 * 1.5 / 7700., rel=1e-2)


def _tell(waves, dv, err):
    segs = [dict(band='H2O', wmin=w - 100., wmax=w + 100., used=True) for w in waves]
    return TZ.TelluricZeroPoint(np.asarray(waves, float), np.asarray(dv, float), np.full(len(waves), err), segs)


@pytest.mark.parametrize('s_true', [-0.8, 0.05, 0.6])
def test_injected_dar_recovered(s_true):
    """Synthetic nodes a + s·D(λ) + noise: a and s recovered with the PA given (s fixed) and free (uniform prior)"""
    rng = np.random.default_rng(3)
    a_true, err = -23., 1.5
    waves = np.array([7600., 8200., 8950., 9300., 9800.])
    ref = float(np.mean(waves))
    dd = dar.dar_kms(_geo(), _inputs(), waves, ref)
    pulls_a, pulls_s = [], []
    for _ in range(200):
        sky = [dict(dv=0., err=0.5, used=True) for _ in waves]
        tell = _tell(waves, a_true + s_true * dd + rng.normal(0., np.hypot(err, 0.5), len(waves)), err)
        known = TZ.star_minus_sky_direct(tell, sky, dd, (s_true, 0.))
        free = TZ.star_minus_sky_direct(tell, sky, dd, None)
        assert known['dar_mode'] == 'known' and known['s'] == s_true and free['dar_mode'] == 'fitted'
        pulls_a.append((known['zp_star_sky'] - a_true) / known['zp_star_sky_err'])
        pulls_s.append((free['s'] - s_true) / free['s_err'])
    assert abs(np.mean(pulls_a)) < 0.25 and 0.8 < np.std(pulls_a) < 1.2
    assert abs(np.median(pulls_s)) < 0.3 and np.std(pulls_s) < 1.3  # the [-1, 1] prior truncates near the edges
    # without D(λ) (no zenith distance or plate scale) the model is a constant and s is not reported
    none = TZ.star_minus_sky_direct(_tell(waves, a_true + s_true * dd, err), sky, None)
    assert none['dar_mode'] == 'none' and np.isnan(none['s'])


@pytest.mark.data
@pytest.mark.skipif(not TESTDATA, reason='RVFITTER_TESTDATA not set')
def test_osiris_pa_convention_against_header_wcs():
    """
    The documented example: the raw header's WCS puts the detector +Y axis (the dispersion, wavelength increasing:
    PypeIt specflip = False) at PA = header PA; the OSIRIS manual's IPA0 ≈ 150.5 deg gives PA 0 (North up), and the
    slit (detector X) at PA + 90 deg was set at the parallactic angle at acquisition (J1745: 36.1 vs 37.9 deg)
    """
    from astropy.io import fits
    for f in sorted(RAW.glob('*/*_hdr.fits')):
        h0, h1 = fits.getheader(f, 0), fits.getheader(f, 1)
        pa_y = np.degrees(np.arctan2(h1['CD1_2'], h1['CD2_2']))  # east (RA) and north components of +Y
        assert (dar.DISPERSION_PA['osiris'](dict(h0)) - pa_y + 180.) % 360. - 180. == pytest.approx(0., abs=0.01)
        assert h0['IPA'] == pytest.approx(h0['IPA0'] + h0['PA'], abs=1e-6) or h0['IPA'] == 0.
    meta = {'raw_files': ['0000809229-20150803-OSIRIS-OsirisLongSlitSpectroscopy.fits'], 'site': 'GTC',
            'object': 'J1745'}
    geo = dar.geometry(io.raw_headers(meta, directory=str(RAW)), meta)
    assert geo['pa_known'] and geo['pa_disp_deg'][0] + 90. == pytest.approx(36.08, abs=0.01)
    assert geo['q_deg'][0][0] == pytest.approx(37.9, abs=0.3)
    assert -0.1 < geo['s'] < 0. and 0. < geo['s_err'] < 0.05 and geo['zenith_deg'] == pytest.approx(58., abs=0.5)
    assert geo['pressure_hpa'] == pytest.approx(782.5, abs=0.1)


@pytest.mark.data
@pytest.mark.slow
@pytest.mark.skipif(not TESTDATA, reason='RVFITTER_TESTDATA not set')
def test_j1745_star_minus_sky_with_dar(real_telluric_model):
    """
    J1745−1640 (auto): with the raw header the slit PA is known and s ≈ −0.05 is fixed; without it s is fitted
    with a uniform prior, which widens the constant's error. Reported, not tuned to the benchmark.
    """
    from rvfitter import config
    from rvfitter.calibration import wavelength_correction as wcorr
    sci = Path(TESTDATA) / 'pypeit/R2500I/GTC54-15A0/OB0055/Science'
    spec1d = str(next(sci.glob('spec1d_*.fits')))
    raw = io.read_spectrum(sci / 'DENISJ174534.6-164053_tellcorr.fits', medium='vacuum', resolution=2602.)
    out = {}
    for known in (True, False):
        c = config.Config()
        c.input.telluric_corrected, c.input.resolution = True, 2602.
        c.templates.local_dir = str(Path(TESTDATA) / 'templates/bt-settl-cifist/useful')
        c.templates.teff_prior = 2100.
        c.calibration.wavelength_correction = 'auto'
        c.calibration.sky_source = c.calibration.telluric_source = spec1d
        c.input.raw_header_dir = str(Path(TESTDATA) / 'raw_headers') if known else None
        _, info = wcorr.correct(raw, raw, c, 'J1745')
        out[known] = info
        assert np.isfinite(info['zp_star_sky']) and info['star_sky']['n_nodes'] >= 2
        assert bool(info['zp_flags'] & wcorr.ZP_FLAGS['dar_pa_unknown']) is not known
    k, f = out[True], out[False]
    assert k['star_sky']['dar_mode'] == 'known' and k['zp_dar_s'] == pytest.approx(k['dar']['s'], abs=0.003)
    assert -0.1 < k['zp_dar_s'] < 0. and k['zp_dar_s_err'] < 0.05
    assert f['star_sky']['dar_mode'] == 'fitted' and f['zp_dar_s_err'] > 0.1
    assert f['zp_star_sky_err'] > k['zp_star_sky_err']


def test_dar_geometry_numpy_meta_values():
    """Numpy meta values are merged without an elementwise `!= []`; None and empty lists keep the source's items"""
    from types import SimpleNamespace

    from rvfitter import config
    from rvfitter.calibration import wavelength_correction as wcorr
    from rvfitter.types import Spectrum
    w = np.linspace(8000., 8100., 50)
    raw = Spectrum(w, np.ones(50), np.full(50, 0.1), 2500., {'object': 'X', 'airmass': np.float64(1.3),
                                                             'raw_files': [], 'exptime': None,
                                                             'extra': np.array([1., 2.])})
    tsrc = Spectrum(w, np.ones(50), np.full(50, 0.1), 2500., {'airmass': 2., 'exptime': 600.})
    info = {'zp_flags': 0, 'slit': SimpleNamespace(scale_spec=None)}
    geo = wcorr._dar_geometry(raw, tsrc, config.Config(), 'X', info)
    assert geo['zenith_deg'] == pytest.approx(np.degrees(np.arccos(1. / 1.3)), abs=0.1)
