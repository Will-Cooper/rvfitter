"""
Tests of spectra already tied to the sky in the reduction: ``calibration.zero_point_tied`` (sys_wavezp), the
'header:KEY' form of it and of ``calibration.slit_prior``, and the ``calibration.star_sky_applied`` guard
"""
from dataclasses import replace

import numpy as np
import pytest
from astropy.io import fits

from rvfitter import config, io, pipeline, systematics
from rvfitter.calibration import wavelength_correction as wcorr
from rvfitter.types import Spectrum
from test_telluric import REGS, SLIT_META, _budget_cfg, _observed

TIES = {'WSKYERR': 2.9, 'WSTARERR': 6.3, 'WSTARCOR': True}


def _tied(header: dict | None = None) -> Spectrum:
    """A spectrum tied to the sky (no distortion), the star 25 km/s off it, with the reduction's cards"""
    sky = _observed(dv_func=lambda w: 0. * w, with_sky=True).sky
    star = _observed(dv_func=lambda w: 25. + 0. * w)
    meta = {**star.meta, **SLIT_META, 'header': TIES if header is None else header}
    return Spectrum(star.wave, star.flux, star.sigma, None, meta, 'tied.fits', sky)


def test_config_accepts_numbers_and_header_cards():
    cfg = config.Config()
    for key, good, bad in (('zero_point_tied', [None, 3., 'header:WSKYERR'], [-1., 'WSKYERR', 'header:']),
                           ('slit_prior', ['auto', 'none', 2., 'header:WSTARERR'], ['wide', -1.]),
                           ('star_sky_applied', [True, False, 'header:WSTARCOR'], ['yes'])):
        for v in good:
            setattr(cfg.calibration, key, v)
            cfg.validate()
        for v in bad:
            setattr(cfg.calibration, key, v)
            with pytest.raises(config.ConfigError, match=key):
                cfg.validate()
        setattr(cfg.calibration, key, getattr(config.Config().calibration, key))
    cfg.calibration.zero_point_tied, cfg.calibration.wavelength_correction = 3., 'sky'
    with pytest.raises(config.ConfigError, match='replaces the tie'):
        cfg.validate()


def test_tied_replaces_sys_wavezp_but_the_curve_is_still_measured():
    raw = _tied()
    cfg = _budget_cfg('none')
    spec, info = wcorr.correct(raw, raw, cfg, 'X', regions=REGS)
    assert np.isfinite(info['zp_sky_kms']) and np.isfinite(info['zp_sky_err'])  # the check is still reported
    w = np.array([8190., 8520.])
    measured = float(systematics.wavelength_zero_point(spec, wave=w))
    assert pipeline._wavelength_term(spec, info, w, cfg, 'X') == (measured, 'measured')
    cfg.calibration.zero_point_tied = 3.5
    assert pipeline._wavelength_term(spec, info, w, cfg, 'X') == (3.5, 'tied (calibration.zero_point_tied = 3.5)')
    cfg.calibration.zero_point_tied = 'header:WSKYERR'
    assert pipeline._wavelength_term(spec, info, w, cfg, 'X')[0] == 2.9
    cfg.calibration.zero_point_tied = 'header:NOPE'
    with pytest.raises(ValueError, match='tied.fits: no header card NOPE'):
        pipeline._wavelength_term(spec, info, w, cfg, 'X')
    bad = replace(spec, meta={**spec.meta, 'header': {'WSKYERR': float('nan')}})
    cfg.calibration.zero_point_tied = 'header:WSKYERR'
    with pytest.raises(ValueError, match='not a velocity'):
        pipeline._wavelength_term(bad, info, w, cfg, 'X')


def test_slit_prior_header_and_the_double_correction_guard(caplog):
    raw = _tied()
    cfg = _budget_cfg('auto')
    _, info = wcorr.correct(raw, raw, cfg, 'X', regions=REGS)
    assert info['applied'] == 'star' and info['star_sky_applied']  # unguarded: auto applies the constant
    cfg.calibration.star_sky_applied = 'header:WSTARCOR'
    cfg.calibration.slit_prior = 'header:WSTARERR'
    caplog.set_level('INFO')
    spec, info = wcorr.correct(raw, raw, cfg, 'X', regions=REGS)
    assert info['applied'] == 'none' and not info['star_sky_applied'] and np.isfinite(info['zp_star_sky'])
    assert 'already applied in the reduction' in caplog.text
    term = wcorr.star_sky_term(info, spec, np.array([8190.]), cfg, 'X')
    assert term['value'] == 6.3 and term['flags'] & wcorr.ZP_FLAGS['slit_prior']
    with pytest.raises(ValueError, match='no header card WSTARCOR'):
        wcorr.correct(_tied({}), _tied({}), cfg, 'X', regions=REGS)


def test_coadd_cards_come_from_its_spec1d_tables(tmp_path):
    """A coadd without the card takes it from the object tables named in its HISTORY: common value, else RMS"""
    for name, err in (('a', 3.), ('b', 4.)):
        tab = fits.BinTableHDU.from_columns([fits.Column('x', 'D', array=[0.])], name='SPAT0241-SLIT0467-DET02')
        tab.header.update(WSKYERR=err, WSTARCOR=True, VEL_CORR=1.)
        other = fits.BinTableHDU.from_columns([fits.Column('x', 'D', array=[0.])], name='SPAT0578-SLIT0467-DET02')
        other.header.update(VEL_CORR=1.)
        fits.HDUList([fits.PrimaryHDU(), other, tab]).writeto(tmp_path / f'spec1d_{name}.fits')
    primary = fits.Header()
    for name in 'ab':
        primary.add_history(f'From "spec1d_{name}.fits"')
        primary.add_history('SPAT0241-SLIT0467-DET02')
    head = io._coadd_provenance(tmp_path / 'coadd.fits', primary)
    spec = Spectrum(np.arange(3.), np.ones(3), np.ones(3), 1000., {'header': head}, 'coadd.fits')
    assert io.header_value(spec, 'WSKYERR') == pytest.approx(np.sqrt(12.5))
    assert io.header_value(spec, 'WSTARCOR') is True
    with pytest.raises(ValueError, match='no header card WSTARC'):
        io.header_value(spec, 'WSTARC')
