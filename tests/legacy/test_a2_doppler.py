"""
A2: the Doppler formula must use the vacuum speed of light, not c / n_air.
"""
from types import SimpleNamespace

import astropy.units as u
import numpy as np

import utils
from splot import Splot

C_KMS = 299792.458


def test_splot_rvcalc_uses_vacuum_c():
    labline = 7664.8991 * u.AA  # K I, air
    rv_true = 30.
    shift = labline * rv_true / C_KMS
    fake = SimpleNamespace(rvunit=u.km / u.s, labline=labline)
    rv = Splot.__rvcalc__(fake, shift)
    assert rv.unit == u.km / u.s
    np.testing.assert_allclose(rv.value, rv_true, rtol=1e-12)


def test_inv_rv_calc_uses_vacuum_c():
    wave = np.array([7664.8991, 8183.2556])
    np.testing.assert_allclose(utils.inv_rv_calc(30., wave), 30. * wave / C_KMS, rtol=1e-12)


def test_rvcalc_and_inv_rv_calc_round_trip():
    labline = 8521.13165 * u.AA
    shift = utils.inv_rv_calc(-12.5, labline.value) * u.AA
    rv = Splot.__rvcalc__(SimpleNamespace(rvunit=u.km / u.s, labline=labline), shift)
    np.testing.assert_allclose(rv.value, -12.5, rtol=1e-12)
