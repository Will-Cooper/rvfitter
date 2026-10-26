import numpy as np
import pytest

from rvfitter.medium import air_to_vac, refractive_index, to_vacuum, vac_to_air


@pytest.mark.parametrize('method', ['ciddor', 'edlen', 'morton'])
def test_known_line(method):
    # K I D1: NIST 7698.9646 Å air, 7701.0835 Å vacuum
    assert vac_to_air(np.array([7701.0835]), method)[0] == pytest.approx(7698.9646, abs=0.003)
    assert air_to_vac(np.array([7698.9646]), method)[0] == pytest.approx(7701.0835, abs=0.003)


@pytest.mark.parametrize('method', ['ciddor', 'edlen', 'morton'])
def test_round_trip(method):
    wave = np.linspace(2100., 50000., 1001)
    assert np.allclose(air_to_vac(vac_to_air(wave, method), method), wave, rtol=1e-11, atol=0)


def test_methods_agree_to_a_few_m_s():
    wave = np.linspace(4000., 25000., 500)
    nc, ne, nm = (refractive_index(wave, m) for m in ('ciddor', 'edlen', 'morton'))
    # differences in n map to velocity offsets c·Δn; standard formulae agree to < 0.05 km/s
    assert np.max(np.abs(nc - ne)) * 299792.458 < 0.05
    assert np.max(np.abs(nc - nm)) * 299792.458 < 0.05


def test_uv_unchanged():
    wave = np.array([1215.67, 1999.])
    assert np.array_equal(vac_to_air(wave), wave)
    assert np.array_equal(air_to_vac(wave), wave)


def test_to_vacuum_and_bad_inputs():
    wave = np.array([8000.])
    assert to_vacuum(wave, 'vacuum') is not None and to_vacuum(wave, 'vacuum')[0] == 8000.
    assert to_vacuum(wave, 'air')[0] > 8000.
    with pytest.raises(ValueError):
        to_vacuum(wave, 'water')
    with pytest.raises(ValueError):
        refractive_index(wave, 'nonsense')
