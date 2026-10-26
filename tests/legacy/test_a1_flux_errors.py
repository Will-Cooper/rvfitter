"""
A1: flux uncertainties from inverse variance, the file-not-found error and a non-mutating normaliser.
"""
import numpy as np
import pytest
from astropy.io import fits

import utils


def _write_pypeit_like(path, wave, flux, ivar):
    cols = [fits.Column(name='wave', format='D', array=wave),
            fits.Column(name='flux', format='D', array=flux),
            fits.Column(name='ivar', format='D', array=ivar)]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(path)


def test_ivar_to_sigma():
    ivar = np.array([4., 0.25, 0., -1., np.nan, np.inf])
    sigma = utils.ivar_to_sigma(ivar)
    np.testing.assert_allclose(sigma[:2], [0.5, 2.])
    assert np.all(np.isnan(sigma[2:]))


def test_freader_sigma_from_ivar_and_masks_bad_pixels(tmp_path):
    npix = 200
    wave = np.linspace(8000., 8200., npix)
    flux = np.full(npix, 2.)
    ivar = np.full(npix, 4.)  # sigma = 0.5, i.e. a relative error of 0.25
    bad = [0, 50, 51, 199]
    flux[bad] = 0.
    ivar[bad[:2]] = 0.
    ivar[bad[2]] = -3.
    ivar[bad[3]] = np.nan
    fname = str(tmp_path / 'synthetic.fits')
    _write_pypeit_like(fname, wave, flux, ivar)

    spec = utils.freader(fname)
    outflux = spec.flux.value
    outerr = spec.uncertainty.array
    assert len(outflux) == npix - len(bad)
    assert np.all(np.isfinite(outerr))
    # after normalisation the flux is 1 and the relative error must be sigma / flux = 0.5 / 2
    np.testing.assert_allclose(outflux, 1.)
    np.testing.assert_allclose(outerr / outflux, 0.25)


def test_freader_missing_txt_raises_file_not_found(tmp_path):
    missing = str(tmp_path / 'does_not_exist.txt')
    with pytest.raises(FileNotFoundError, match='does_not_exist.txt'):
        utils.freader(missing)


def test_freader_txt_drops_nonfinite_rows(tmp_path):
    wave = np.linspace(8000., 8200., 50)
    flux = np.linspace(1., 2., 50)
    flux[[3, 20]] = [np.nan, np.inf]
    fname = str(tmp_path / 'synthetic.txt')
    np.savetxt(fname, np.column_stack([wave, flux]))
    spec = utils.freader(fname)
    assert len(spec.flux) == 48
    assert np.all(np.isfinite(spec.flux.value))
    np.testing.assert_array_equal(spec.uncertainty.array, 0.)  # text files keep their zero errors


def test_normaliser_does_not_mutate_inputs():
    x = np.linspace(8000., 8300., 31)
    flux = np.linspace(1., 4., 31)
    err = np.full(31, 0.1)
    flux_copy, err_copy = flux.copy(), err.copy()
    out = utils.normaliser(x, flux, err, xmin=8100, xmax=8200)
    np.testing.assert_array_equal(flux, flux_copy)
    np.testing.assert_array_equal(err, err_copy)
    normval = np.median(flux_copy[(x > 8100) & (x < 8200)])
    np.testing.assert_allclose(out[1], flux_copy / normval)
    np.testing.assert_allclose(out[2], err_copy / normval)
