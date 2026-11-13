"""Forecast: the Bouchy σ_RV of a Gaussian line against its analytic value, and the table layout."""
import numpy as np
import pytest

from rvfitter import broadening, forecast
from rvfitter.constants import C_KMS
from rvfitter.types import Spectrum

LAM0, DEPTH, SIG0_KMS, R, SNR = 8650., 0.01, 20., 5000., 100.


def _gaussian_line_template():
    wave = broadening.log_wave_grid(8300., 9000., 0.5)
    sig = LAM0 * SIG0_KMS / C_KMS
    flux = 1. - DEPTH * np.exp(-0.5 * ((wave - LAM0) / sig) ** 2)
    return Spectrum(wave, flux, np.full(len(wave), np.nan), None, {'object': 'gauss'}, 'gauss')


def test_gaussian_line_matches_analytic():
    """Shallow Gaussian line: Σ W = SNR² λ₀² d² √π / (2 σ_λ Δλ), with d and σ after the LSF and pixel"""
    dv_pix = C_KMS / (10. * R)
    wave = broadening.log_wave_grid(8550., 8750., dv_pix)
    out = forecast.table(_gaussian_line_template(), wave, R, SNR)
    sig_kms = np.sqrt(SIG0_KMS ** 2 + (C_KMS / R / (2 * np.sqrt(2 * np.log(2)))) ** 2 + dv_pix ** 2 / 12.)
    depth = DEPTH * SIG0_KMS / sig_kms  # equivalent width conserved
    sig_lam, dlam = LAM0 * sig_kms / C_KMS, LAM0 * dv_pix / C_KMS
    analytic = C_KMS / np.sqrt(SNR ** 2 * LAM0 ** 2 * depth ** 2 * np.sqrt(np.pi) / (2 * sig_lam * dlam))
    total = out[out['kind'] == 'total']['sigma_rv'][0]
    assert total == pytest.approx(analytic, rel=0.03)
    assert set(out['kind']) >= {'band', 'total'} and out['sigma_rv'].unit == 'km/s'


def test_cli(tmp_path, monkeypatch):
    """Options before any positional reach the module (the lazy dispatcher does not parse them)"""
    from astropy.table import Table

    from rvfitter import cli, config
    monkeypatch.setattr(forecast, 'setup', lambda args: (config.Config(), _gaussian_line_template(),
                                                         forecast.wave_grid(*args.wave_range, args.resolution), R))
    out = tmp_path / 'fc.csv'
    assert cli.main(['forecast', '-q', '--resolution', '5000', '--snr', '50', '--wave-range', '8550', '8750',
                     '--out', str(out)]) == 0
    assert 'total' in list(Table.read(out, format='ascii.csv')['kind'])


def test_recommended_and_scaling():
    """σ_RV ∝ 1/SNR; the recommended feature windows are the most informative"""
    tpl = _gaussian_line_template()
    wave = forecast.wave_grid(8550., 8750., R)
    a = forecast.table(tpl, wave, R, 50.)
    b = forecast.table(tpl, wave, R, 100.)
    assert np.allclose(a['sigma_rv'], 2. * b['sigma_rv'])
    feat = a[a['kind'] == 'feature']
    if len(feat):
        assert feat['recommended'].any() and feat[feat['recommended']]['sigma_rv'].max() <= feat['sigma_rv'].max()
