"""Tests of rvfitter.broadening: Doppler factors, flux-conserving resampling and the LSF/rotation kernels."""
import numpy as np
import pytest

from rvfitter import broadening as br
from rvfitter.constants import C_KMS
from rvfitter.types import Spectrum


def _line_spectrum(width_kms=2., centres=(8000.,), depth=0.5, dv=0.5, wmin=7900., wmax=8100.):
    wave = br.log_wave_grid(wmin, wmax, dv)
    flux = np.ones_like(wave)
    for c in centres:
        flux -= depth * np.exp(-0.5 * ((wave - c) / (c * width_kms / C_KMS)) ** 2)
    return Spectrum(wave, flux, np.full_like(wave, 0.01), None, {}, 'test')


def _moments(wave, flux, centre, half=15.):
    sel = np.abs(wave - centre) < half
    depth = 1. - flux[sel]
    ew = np.trapezoid(depth, wave[sel])
    mu = np.trapezoid(depth * wave[sel], wave[sel]) / ew
    var = np.trapezoid(depth * (wave[sel] - mu) ** 2, wave[sel]) / ew
    return ew, mu, np.sqrt(var)


def test_doppler_factor_relativistic_and_inverse():
    v = 1234.5
    beta = v / C_KMS
    np.testing.assert_allclose(br.doppler_factor(v), np.sqrt((1 + beta) / (1 - beta)), rtol=1e-15)
    np.testing.assert_allclose(br.doppler_factor(v) * br.doppler_factor(-v), 1., rtol=1e-15)


def test_log_wave_grid_uniform():
    w = br.log_wave_grid(7000., 9000., 3.)
    np.testing.assert_allclose(np.diff(np.log(w)) * C_KMS, 3., rtol=1e-8)
    assert w[0] == 7000. and w[-1] <= 9000.


def test_rebin_conserves_flux_integral():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 100, 500))
    y = 1 + np.sin(x / 3.) + rng.normal(0, 0.1, len(x))
    for new in (np.linspace(10, 90, 37), np.linspace(10, 90, 3001)):
        out = br.rebin(x, y, new)
        edges = br._edges(new)
        integral = np.sum(out * np.diff(edges))
        ref = br._cumulative_integral(x, y, edges[-1:]) - br._cumulative_integral(x, y, edges[:1])
        np.testing.assert_allclose(integral, ref[0], rtol=1e-12)


def test_rebin_linear_is_exact_and_nan_handled():
    x = np.linspace(0, 10, 101)
    new = np.linspace(1, 9, 33)
    np.testing.assert_allclose(br.rebin(x, 3 * x + 2, new), 3 * new + 2, rtol=1e-12)
    y = 3 * x + 2
    y[50] = np.nan
    out = br.rebin(x, y, new)
    assert np.isnan(out[np.argmin(np.abs(new - 5.))])
    assert np.isfinite(out[0]) and np.isfinite(out[-1])
    assert np.isnan(br.rebin(x, y, np.array([20., 21.]))).all()


def test_rotation_kernel_normalised_and_shape():
    k = br.rotation_kernel(0.1, 30., 0.6)
    assert np.isclose(k.sum(), 1.) and len(k) % 2 == 1
    x = (np.arange(len(k)) - len(k) // 2) * 0.1 / 30.
    var = np.sum(k * (x * 30.) ** 2)
    # second moment of the continuous Gray kernel, by quadrature
    xx = np.linspace(-1, 1, 200001)
    g = 2 * 0.4 * np.sqrt(1 - xx ** 2) + 0.5 * np.pi * 0.6 * (1 - xx ** 2)
    var_ref = np.trapezoid(g * (30. * xx) ** 2, xx) / np.trapezoid(g, xx)
    np.testing.assert_allclose(var, var_ref, rtol=2e-3)
    assert br.rotation_kernel(5., 1.).tolist() == [1.]
    with pytest.raises(ValueError):
        br.rotation_kernel(1., 10., 1.5)


def test_convolve_keeps_edges_flat():
    flux = np.ones(100)
    out = br.convolve(flux, br.gaussian_kernel(5.))
    np.testing.assert_allclose(out, 1., rtol=1e-12)


def test_convolve_nan_mask_dilated():
    flux = np.ones(200)
    flux[100] = np.nan
    kernel = br.gaussian_kernel(3.)
    half = len(kernel) // 2
    out = br.convolve(flux, kernel)
    assert np.isnan(out[100 - half:100 + half + 1]).all()
    assert np.isfinite(out[:100 - half]).all() and np.isfinite(out[100 + half + 1:]).all()
    assert np.isnan(br.convolve(np.full(10, np.nan), kernel)).all()


def test_gaussian_kernel_narrow_still_smooths():
    k = br.gaussian_kernel(0.4)
    assert k.max() < 0.9 and np.isclose(k.sum(), 1.)



def _gray_sigma(vsini, epsilon=0.6):
    xx = np.linspace(-1, 1, 200001)
    g = 2 * (1 - epsilon) * np.sqrt(1 - xx ** 2) + 0.5 * np.pi * epsilon * (1 - xx ** 2)
    return vsini * np.sqrt(np.trapezoid(g * xx ** 2, xx) / np.trapezoid(g, xx))


@pytest.mark.parametrize('resolution, vsini, native', [(2000., 0., None), (8000., 0., None), (None, 40., None),
                                                       (4000., 20., None), (3000., 0., 10000.)])
def test_broaden_loglam_width_and_ew(resolution, vsini, native):
    spec = _line_spectrum(width_kms=2., dv=0.5)
    out = br.broaden_loglam(spec.flux, 0.5, resolution=resolution, vsini=vsini, resolution_native=native)
    ew0, mu0, s0 = _moments(spec.wave, spec.flux, 8000.)
    ew1, mu1, s1 = _moments(spec.wave, out, 8000.)
    inv2 = 0. if resolution is None else 1 / resolution ** 2 - (0. if native is None else 1 / native ** 2)
    s_lsf = 8000. * np.sqrt(inv2) * br.FWHM_TO_SIGMA
    s_rot = 8000. / C_KMS * _gray_sigma(vsini)
    np.testing.assert_allclose(ew1, ew0, rtol=2e-3)
    np.testing.assert_allclose(mu1, mu0, atol=1e-3)
    np.testing.assert_allclose(s1, np.sqrt(s0 ** 2 + s_lsf ** 2 + s_rot ** 2), rtol=1e-2)
