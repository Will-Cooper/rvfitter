"""Instrument-typical plate scales and arc RMS (data/instruments.csv) for Spextool files that record neither"""
import logging
import os
from pathlib import Path

import numpy as np
import pytest

from rvfitter import config, io, results, systematics
from rvfitter.calibration import slit
from rvfitter.calibration import wavelength_correction as wcorr
from rvfitter.types import Spectrum

FLAGS = wcorr.ZP_FLAGS
WAVE = np.linspace(8500., 24000., 3000)
FAKE_ROW = slit.Instrument('uSpeX', 0.10, 0.10, 1, 4.0, 'IRTF SpeX page', 'test value')


def _speex(instrument_names=('SpeX Spectrograph', 'uSpeX')):
    meta = dict(instrument=instrument_names[0], instrument_names=instrument_names, slit_width_arcsec=0.3,
                slit_width_key='SLTW_ARC')
    return Spectrum(WAVE, np.ones(len(WAVE)), np.ones(len(WAVE)), 2000., meta, 'sxd_fake.fits')


def _budget(spec, cfg):
    """(rv_total_err, zp flags) of a 5 km/s measurement through the zero-point terms, as the pipeline forms them"""
    spec, info = wcorr.correct(spec, spec, cfg, 'X')
    term = wcorr.star_sky_term(info, spec, WAVE[::100], cfg, 'X')
    wavezp = systematics.wavelength_zero_point(spec, wave=WAVE[::100])
    return results._quad(5., 1., wavezp, term['value']), info['zp_flags'] | term['flags']


@pytest.fixture
def typical(monkeypatch):
    monkeypatch.setattr(slit, '_table', lambda: {'uspex': FAKE_ROW})


def test_shipped_table_has_no_unsourced_arc_value():
    for row in ('uSpeX', 'ARCoIRIS', 'OSIRIS'):
        assert slit.instrument(None, row).wave_rms_kms is None
    assert slit.instrument(None, 'SpeX Spectrograph', 'uSpeX').pixscale_spat == 0.10
    assert slit.instrument(None, 'ARCoIRIS').pixscale_spat == 0.417


def test_spextool_scale_matches_instr_not_only_instrume():
    inp = slit.inputs([_speex()])
    assert inp.scale_spec == pytest.approx(0.10) and 'uSpeX' in inp.scale_source


def test_typical_value_gives_finite_total_with_flags(typical, caplog):
    caplog.set_level(logging.WARNING)
    total, flags = _budget(_speex(), config.Config())
    assert np.isfinite(total)
    assert flags & FLAGS['wave_rms_assumed'] and flags & FLAGS['seeing_assumed']
    assert 'assuming a wavelength-calibration accuracy of 4.00 km/s (data/instruments.csv[uSpeX]: test value)' \
        in caplog.text


def test_config_override_wins(typical):
    cfg = config.Config()
    cfg.calibration.wave_rms_kms = 9.
    spec, info = wcorr.correct(_speex(), _speex(), cfg, 'X')
    assert wcorr.arc_kms(spec) == 9. and 'calibration.wave_rms_kms' in spec.meta['wave_rms_source']
    assert info['zp_flags'] & FLAGS['wave_rms_assumed']


def test_config_override_works_for_unknown_instrument():
    cfg = config.Config()
    cfg.calibration.wave_rms_kms = 3.
    spec, info = wcorr.correct(_speex(('Mystery',)), _speex(('Mystery',)), cfg, 'X')
    assert wcorr.arc_kms(spec) == 3. and info['zp_flags'] & FLAGS['wave_rms_assumed']


def test_unknown_instrument_is_nan(typical):
    total, flags = _budget(_speex(('Mystery',)), config.Config())
    assert np.isnan(total) and not flags & FLAGS['wave_rms_assumed']


def test_file_arc_rms_is_not_replaced(typical):
    spec = _speex()
    spec.meta['wave_rms'] = 0.5
    out, info = wcorr.correct(spec, spec, config.Config(), 'X')
    assert 'wave_rms_kms' not in out.meta and not info['zp_flags'] & FLAGS['wave_rms_assumed']


def test_config_validates_wave_rms():
    cfg = config.Config()
    cfg.calibration.wave_rms_kms = -1.
    assert any('wave_rms_kms' in p for p in config._validate(cfg))


@pytest.mark.data
@pytest.mark.parametrize('name', ['sxd_J0508+3319.fits', 'prism_J0832+3538.fits', 'xtc_J0526-5026.fits'])
def test_spextool_spectra_get_scale_and_seeing(name):
    root = os.environ.get('RVFITTER_TESTDATA')
    if not root:
        pytest.skip('RVFITTER_TESTDATA not set')
    spec = io.read_spectrum(Path(root) / 'gemma_spectra' / 'spectra' / name)
    cfg = config.Config()
    cfg.calibration.wave_rms_kms = 5.
    spec, info = wcorr.correct(spec, spec, cfg, 'X')
    term = wcorr.star_sky_term(info, spec, spec.wave[::200], cfg, 'X')
    assert np.isfinite(term['value']) and info['zp_flags'] & FLAGS['seeing_assumed']
    assert np.isfinite(systematics.wavelength_zero_point(spec, wave=spec.wave[::200]))
