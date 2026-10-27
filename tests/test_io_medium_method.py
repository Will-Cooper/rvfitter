"""The air-to-vacuum formula is selectable through read_spectrum(medium_method=...)."""
import numpy as np
from astropy.io import fits

from rvfitter import io
from rvfitter.medium import air_to_vac

WAVE = np.linspace(7000., 9000., 200)


def test_medium_method(tmp_path):
    cols = [fits.Column(name=n, format='D', array=a) for n, a in
            (('OPT_WAVE', WAVE), ('OPT_FLAM', np.ones(200)), ('OPT_FLAM_IVAR', np.ones(200)))]
    prim = fits.PrimaryHDU()
    prim.header['AIRORVAC'] = 'air'
    fits.HDUList([prim, fits.BinTableHDU.from_columns(cols)]).writeto(tmp_path / 's.fits')
    ciddor = io.read_spectrum(tmp_path / 's.fits')
    morton = io.read_spectrum(tmp_path / 's.fits', medium_method='morton')
    np.testing.assert_allclose(morton.wave, air_to_vac(WAVE, 'morton'), rtol=1e-12)
    assert not np.allclose(ciddor.wave, morton.wave, rtol=0, atol=1e-6)
