"""Tests for rvfitter.barycentric"""
import glob
import logging
import os

import numpy as np
import pytest
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from rvfitter.barycentric import SITES, apply, correction, lookup_site, vel_corr_to_kms
from rvfitter.constants import C_KMS
from rvfitter.types import Spectrum


def _spec(**meta):
    wave = np.linspace(7000., 9000., 50)
    return Spectrum(wave, np.ones(50), np.ones(50), 2500., meta, 'test')


FULL = dict(ra=10.6847083, dec=41.26875, mjd_mid=57798.0, site='Keck')


def test_astropy_documented_example():
    # astropy docs: M31, Keck, 2017-02-14 -> -22.359784554780255 km/s (doc value uses the of_site Keck, SIMBAD M31)
    kms, src = correction(_spec(**FULL), 'auto')
    assert src == 'computed'
    loc = EarthLocation.from_geodetic(-155.47833, 19.82833, 4160.)
    ref = SkyCoord(10.6847083, 41.26875, unit='deg').radial_velocity_correction(
        obstime=Time('2017-02-14'), location=loc).to('km/s').value
    assert abs(ref - -22.359784554780255) < 1e-3  # our environment reproduces the documented value to 1 m/s
    assert abs(kms - ref) < 1e-3


def test_offline_table():
    assert len(SITES) >= 15
    for name in ('GTC', 'IRTF', 'Keck', 'Gemini-North', 'VLT', 'Magellan', 'Calar Alto', 'McDonald', 'Palomar',
                 'Kitt Peak', 'La Silla', 'SALT', 'Subaru', 'LBT', 'Gemini South'):
        assert lookup_site(name) is not None, name
    assert lookup_site('nowhere') is None


@pytest.mark.parametrize('key', ['frame_in', 'vel_type', 'VHELIO'])
def test_auto_skips_when_corrected(key):
    value = 'barycentric' if key in ('frame_in', 'vel_type') else -12.3
    assert correction(_spec(**{key: value}, **FULL), 'auto') == (0., 'header')


def test_frame_mismatch_warns(caplog):
    with caplog.at_level(logging.WARNING):
        assert correction(_spec(frame_in='heliocentric', **FULL), 'auto', 'barycentric') == (0., 'header')
    assert any('requested' in r.message for r in caplog.records)


@pytest.mark.parametrize('key', ['BERV', 'HIERARCH ESO DRS BERV', 'QC BERV'])
def test_berv_is_fallback_only(key, caplog):
    # no coordinates: the BERV header value (RV_bary = RV_obs + BERV) is the fallback, with a WARNING
    with caplog.at_level(logging.WARNING):
        assert correction(_spec(**{key: -12.3}), 'auto') == (-12.3, 'header')
        assert correction(_spec(**{key: -12.3}), 'apply') == (-12.3, 'header')
    assert any(r.levelno == logging.WARNING and key in r.message for r in caplog.records)
    # with coordinates and a site the correction is always computed, never taken from the header
    assert correction(_spec(**{key: -12.3}, **FULL), 'auto')[1] == 'computed'
    assert correction(_spec(**{key: -12.3}, **FULL), 'apply')[1] == 'computed'
    # a heliocentric request does not use a barycentric value
    assert correction(_spec(**{key: -12.3}), 'auto', 'heliocentric') == (0., 'skipped')


@pytest.mark.parametrize('key', ['BCV', 'HCV', 'HIERARCH ESO QC VRAD BARYCOR'])
def test_uncited_sign_keywords_ignored(key):
    assert correction(_spec(**{key: -12.3}), 'auto') == (0., 'skipped')


def test_site_alias_is_whole_word():
    assert lookup_site('SALT') is not None
    assert lookup_site('basalt hill') is None


def test_auto_warns_and_skips(caplog):
    with caplog.at_level(logging.WARNING):
        kms, src = correction(_spec(), 'auto')
    assert (kms, src) == (0., 'skipped')
    assert any(r.levelno == logging.WARNING and 'missing' in r.message for r in caplog.records)


def test_skip_never_corrects():
    assert correction(_spec(**FULL), 'skip') == (0., 'skipped')


def test_apply_mode_errors_and_overrides():
    with pytest.raises(ValueError, match='missing'):
        correction(_spec(), 'apply')
    kms, src = correction(_spec(), 'apply', overrides=FULL)
    assert src == 'computed' and abs(kms) > 1.
    # forced computation even when the header says corrected
    kms2, src2 = correction(_spec(frame_in='barycentric', **FULL), 'apply')
    assert src2 == 'computed' and kms2 == kms
    # lat/lon/alt instead of a site name
    lat, lon, alt = SITES['keck']
    kms3, _ = correction(_spec(), 'apply', overrides=dict(FULL, site=None, lat=lat, lon=lon, alt=alt))
    assert abs(kms3 - kms) < 1e-6
    # sexagesimal coordinates
    kms4, _ = correction(_spec(), 'apply', overrides=dict(FULL, ra='00:42:44.33', dec='+41:16:07.5'))
    assert abs(kms4 - kms) < 0.01


def test_heliocentric_differs_from_barycentric():
    kb, _ = correction(_spec(**FULL), 'auto', 'barycentric')
    kh, _ = correction(_spec(**FULL), 'auto', 'heliocentric')
    assert 0. < abs(kb - kh) < 0.02


def test_vel_corr_factor_roundtrip():
    beta = -21.0 / C_KMS
    factor = np.sqrt((1 + beta) / (1 - beta))
    assert abs(vel_corr_to_kms(factor) - -21.0) < 1e-9


def test_sign_convention():
    # Observed line at RV_obs = 30 km/s, observer moving at v_corr = -21 km/s => RV_bary = RV_obs + v_corr = 9 km/s
    rest, rv_obs, v_corr = 8000., 30., -21.
    wave = np.linspace(7990., 8010., 4001)
    flux = 1. - 0.5 * np.exp(-0.5 * ((wave - rest * np.sqrt((1 + rv_obs / C_KMS) / (1 - rv_obs / C_KMS))) / 0.3) ** 2)
    spec = Spectrum(wave, flux, np.full_like(wave, 0.01), 5000., {}, 'x')
    new = apply(spec, v_corr)
    depth = 1. - new.flux
    core = np.sum(new.wave * depth) / np.sum(depth)
    rv_new = C_KMS * (core ** 2 - rest ** 2) / (core ** 2 + rest ** 2)
    assert abs(rv_new - (rv_obs + v_corr)) < 0.01
    assert np.array_equal(new.flux, spec.flux)
    assert apply(spec, 0.).wave.tolist() == spec.wave.tolist()


# ---------------------------------------------------------------- data test
DATA = os.environ.get('RVFITTER_TESTDATA')


@pytest.mark.data
@pytest.mark.skipif(DATA is None, reason='RVFITTER_TESTDATA not set')
def test_j1745_matches_pypeit():
    """
    PypeIt writes VEL_CORR = sqrt((1+v/c)/(1-v/c)) = 0.99992993... => v = -21.01 km/s (heliocentric), i.e. kms to
    add is NEGATIVE (RV_helio = RV_obs - 21 km/s) for this target and date.
    PypeIt's header has LON-OBS = +17.877 (GTC lies at 17.89 deg WEST), so PypeIt used a wrong longitude; to
    reproduce its value we therefore pass its lat/lon/alt keys verbatim (the barycentric file matches to <1 m/s).
    """
    from astropy.io import fits
    base = os.path.join(DATA, 'pypeit/R2500I/GTC54-15A0/OB0055/sciencebk')
    pat = 'spec1d_*DENIS*.fits'
    path = glob.glob(os.path.join(base, 'nopipecorr', pat))[0]
    with fits.open(path) as hdul:
        h = hdul[0].header
        meta = dict(ra=h['RA'], dec=h['DEC'], mjd_mid=h['MJD'] + h['EXPTIME'] / 2. / 86400.,
                    lat=h['LAT-OBS'], lon=h['LON-OBS'], alt=h['ALT-OBS'])
        assert 'VEL_TYPE' not in h
    spec = _spec(**meta)
    for sub, frame, tol in (('yespipecorr', 'barycentric', 2e-3), ('yespipecorrhelio', 'heliocentric', 0.03)):
        with fits.open(glob.glob(os.path.join(base, sub, pat))[0]) as hdul:
            expected = vel_corr_to_kms(hdul[1].header['VEL_CORR'])
        kms, src = correction(spec, 'auto', frame)
        assert src == 'computed'
        assert abs(kms - expected) < tol, (frame, kms, expected)
    assert abs(vel_corr_to_kms(0.9999299355651026) - -21.0055) < 1e-3
    # with a correct (west) longitude from the built-in table the answer moves by ~0.15 km/s
    kms_site, _ = correction(_spec(**{k: v for k, v in meta.items() if k not in ('lat', 'lon', 'alt')}, site='GTC'),
                             'auto', 'heliocentric')
    assert 0.1 < abs(kms_site - expected) < 0.2
    # the headers of PypeIt-corrected files are recognised
    with fits.open(glob.glob(os.path.join(base, 'yespipecorrhelio', pat))[0]) as hdul:
        corrected = _spec(vel_type=hdul[1].header['VEL_TYPE'], **meta)
    assert correction(corrected, 'auto') == (0., 'header')
