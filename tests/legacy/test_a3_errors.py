"""
A3: error propagation (wavelength rms in quadrature, the SNR-based RV floor, the xcorr per-line error,
the per-method combination and the adopted RV).
"""
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pandas as pd
from matplotlib import rcParams
from specutils import Spectrum1D, SpectralRegion

import utils
from splot import Splot
from xcorr import Xcorr

C_KMS = 299792.458


def test_getshift_adds_waverms_in_quadrature():
    fake = SimpleNamespace(x_0=7666. * u.AA, labline=7664.9 * u.AA, wunit=u.AA, waverms=0.4 * u.AA,
                           fitter=SimpleNamespace(fit_info={'param_cov': np.diag([0.09, 0.3 ** 2, 1.])}))
    Splot.getshift(fake, 1)
    np.testing.assert_allclose(fake.shifterr.to(u.AA).value, 0.5)  # sqrt(0.3^2 + 0.4^2), not 0.7


def test_working_fwhm_follows_profile():
    fake = SimpleNamespace(fwhm_G=0 * u.AA, fwhm_L=3. * u.AA, fwhm_V=4. * u.AA, working_profile='lorentzian')
    assert Splot.working_fwhm(fake) == 3. * u.AA
    fake.working_profile = 'voigt'
    assert Splot.working_fwhm(fake) == 4. * u.AA


def test_get_snr_and_rvunc_formula():
    wave = np.arange(7900., 8100., 0.5)
    flux = np.where(np.arange(len(wave)) % 2, 1.1, 0.9)  # mean 1, std 0.1, so SNR = 10
    spec = Spectrum1D(flux * u.Jy, wave * u.AA)
    cont = SpectralRegion([[7900. * u.AA, 7950. * u.AA], [8050. * u.AA, 8099. * u.AA]])
    line = SpectralRegion(7995. * u.AA, 8005. * u.AA)
    snr, rvunc = utils.get_snr_and_rvunc(spec, cont, line, 2. * u.AA)
    np.testing.assert_allclose(snr, 10., rtol=1e-3)
    npix = np.sum((wave >= 7995.) & (wave <= 8005.))
    lam = np.mean(wave[(wave >= 7995.) & (wave <= 8005.)])
    expected = C_KMS / ((lam / 2.) * snr * np.sqrt(npix))
    np.testing.assert_allclose(rvunc, expected, rtol=1e-3)
    _, rvunc0 = utils.get_snr_and_rvunc(spec, cont, line, 0. * u.AA)  # e.g. fwhm_G of a Lorentzian fit
    assert np.isnan(rvunc0)


def test_xcorr_rverr_converts_waverms_to_velocity():
    fake = SimpleNamespace(rvstep=10. * u.km / u.s, waverms=0.077 * u.AA, wunit=u.AA,
                           labline=7664.9 * u.AA, rvunit=u.km / u.s)
    Xcorr.getrverr(fake)
    expected = np.hypot(10. / np.sqrt(12.), C_KMS * 0.077 / 7664.9)  # uniform quantisation; ~4.18 km/s
    np.testing.assert_allclose(fake.rverr.to(u.km / u.s).value, expected)


def test_combine_rvs():
    rv, err = utils.combine_rvs([12.], [3.])
    assert (rv, err) == (12., 3.)
    # identical values: zero scatter, so the formal error 1/sqrt(sum w) wins
    rv, err = utils.combine_rvs([10., 10., 10., 10.], [2., 2., 2., 2.])
    np.testing.assert_allclose([rv, err], [10., 1.])
    # widely scattered values: the formal error is inflated by sqrt(chi^2 / (N - 1)) = sqrt(200)
    rv, err = utils.combine_rvs([0., 20.], [1., 1.])
    np.testing.assert_allclose([rv, err], [10., 10.])
    # NaNs are ignored
    rv, err = utils.combine_rvs([np.nan, 5.], [1., 2.])
    assert (rv, err) == (5., 2.)


LINES = ['k1-a', 'k1-b', 'rb1-a', 'rb1-b', 'na1-a', 'na1-b']
LINE_OFFSETS = np.array([8., -6., 3., -9., 5., -1.])
NOISE = np.array([0.3, -0.2, 0.1, -0.4, 0.2, 0.0])


def _two_methods(line_offsets, method_offset=0., err=1.):
    values = np.concatenate([20. + line_offsets + NOISE, 20. + line_offsets - NOISE + method_offset])
    return (values, np.full(len(values), err), LINES + LINES,
            ['lc'] * len(LINES) + ['xcorr'] * len(LINES))


def test_hierarchical_tau_zero_recovers_inverse_variance():
    values = np.array([10., 12., 15., 11.])
    errors = np.array([1., 2., 3., 1.5])
    res = utils.hierarchical_combine(values, errors, ['a', 'b', 'a', 'b'], ['m', 'm', 'n', 'n'],
                                     tau_line=0., tau_meth=0.)
    w = 1 / errors ** 2
    np.testing.assert_allclose(res['mean'], np.sum(w * values) / np.sum(w))
    np.testing.assert_allclose(res['sd'], 1 / np.sqrt(np.sum(w)))
    np.testing.assert_allclose(res['median'], res['mean'])


def test_hierarchical_single_measurement_returns_itself():
    res = utils.hierarchical_combine([7.5], [2.], ['a'], ['m'])
    np.testing.assert_allclose([res['mean'], res['median']], [7.5, 7.5])
    assert res['sd'] >= 2.  # broadened by the unidentified taus, never narrowed
    res0 = utils.hierarchical_combine([7.5], [2.], ['a'], ['m'], tau_line=0., tau_meth=0.)
    np.testing.assert_allclose([res0['mean'], res0['sd']], [7.5, 2.])


def test_hierarchical_shared_line_offsets_inflate_error():
    quiet = utils.hierarchical_combine(*_two_methods(np.zeros(len(LINES))))
    shared = utils.hierarchical_combine(*_two_methods(LINE_OFFSETS))
    formal = 1 / np.sqrt(2 * len(LINES))
    assert shared['sd'] > 2.5 > quiet['sd']
    assert shared['sd'] > 5 * formal
    assert shared['tau_line'] > 3. > quiet['tau_line']
    np.testing.assert_allclose(shared['mean'], 20. + LINE_OFFSETS.mean(), atol=0.5)


def test_hierarchical_method_offset_shows_in_tau_meth():
    same = utils.hierarchical_combine(*_two_methods(np.zeros(len(LINES))))
    offset = utils.hierarchical_combine(*_two_methods(np.zeros(len(LINES)), method_offset=15.))
    assert offset['tau_meth'] > 5. > same['tau_meth']
    np.testing.assert_allclose(offset['mean'], 27.5, atol=0.5)  # halfway between the two methods
    assert offset['sd'] > 3. > same['sd']  # two methods 15 km/s apart leave mu poorly constrained


def test_hierarchical_invariant_to_input_order():
    values, errors, lines, methods = _two_methods(LINE_OFFSETS, method_offset=4.)
    errors = errors * np.linspace(0.8, 1.6, len(errors))
    order = np.random.default_rng(1).permutation(len(values))
    res = utils.hierarchical_combine(values, errors, lines, methods)
    res_perm = utils.hierarchical_combine(values[order], errors[order], np.array(lines)[order],
                                          np.array(methods)[order])
    for key in ('mean', 'sd', 'median', 'q16', 'q84', 'tau_line', 'tau_meth'):
        np.testing.assert_allclose(res_perm[key], res[key], rtol=1e-9)


def test_mixture_pdf_integrates_to_one():
    res = utils.hierarchical_combine(*_two_methods(LINE_OFFSETS))
    x = np.linspace(res['mean'] - 15 * res['sd'], res['mean'] + 15 * res['sd'], 4001)
    pdf = utils.mixture_pdf(x, res['means'], res['sds'], res['weights'])
    np.testing.assert_allclose(np.trapz(pdf, x), 1., rtol=1e-3)


def _run_adoptedrv(tmp_path, monkeypatch, lcvals, lcerr, xcorr, xerr, systematic=0.):
    import fullrv
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'adoptedrvplots').mkdir(exist_ok=True)
    monkeypatch.setitem(rcParams, 'text.usetex', False)
    df = pd.DataFrame({'shortname': ['X'], 'thisrv': [np.nan], 'thisrverr': [np.nan]})
    spec_indices = {name: 7600. + 100 * i for i, name in enumerate(LINES[:len(lcvals)])}
    df = fullrv.adoptedrv(df, 'shortname', 'X', True, np.array(lcvals), np.array(lcerr),
                          np.array(xcorr), np.array(xerr), spec_indices, systematic=systematic)
    return df.thisrv.iloc[0], df.thisrverr.iloc[0]


def test_adoptedrv_uses_hierarchical_model(tmp_path, monkeypatch):
    lcvals, lcerr = [30., 28., 35., 20., np.nan], [5., 3., 4., np.nan, 2.]  # last two have no usable pair
    xcorr, xerr = [31., 25., 36., 22., 24.], [5., 5., 5., 5., np.nan]
    rv, err = _run_adoptedrv(tmp_path, monkeypatch, lcvals, lcerr, xcorr, xerr, systematic=2.)
    names = np.array(LINES[:5])
    res = utils.hierarchical_combine(np.array(lcvals + xcorr), np.array(lcerr + xerr),
                                     np.concatenate([names, names]), ['lc'] * 5 + ['xcorr'] * 5)
    assert res['n'] == 7  # joint masking: values without errors (and vice versa) are dropped
    np.testing.assert_allclose(rv, res['median'])
    halfwidth = (res['q84'] - res['q16']) / 2
    np.testing.assert_allclose(err, np.sqrt(halfwidth ** 2 + 2. ** 2))  # systematic in quadrature
    log = (tmp_path / 'calculating.log').read_text()
    assert 'tau_line' in log and 'tau_meth' in log and '84%' in log


def test_hierarchical_quantiles_stable_to_grid_extent():
    # with two methods the mu posterior has power-law tails: the SD grows with the tau grid, the quantiles do not
    args = _two_methods(LINE_OFFSETS, method_offset=6.)
    narrow = utils.hierarchical_combine(*args, tau_max_factor=10.)
    wide = utils.hierarchical_combine(*args, tau_max_factor=1000.)
    for key in ('median', 'q16', 'q84'):
        np.testing.assert_allclose(wide[key], narrow[key], atol=0.05)
    assert wide['sd'] > narrow['sd']
