import numpy as np
import pytest

from rvfitter.constants import C_KMS
from rvfitter.types import Spectrum
from synth import make_spectrum, make_template


def test_spectrum_validation():
    with pytest.raises(ValueError):
        Spectrum(np.array([2., 1.]), np.ones(2), np.ones(2))
    with pytest.raises(ValueError):
        Spectrum(np.array([1., 2.]), np.ones(3), np.ones(2))
    s = Spectrum(np.arange(10.), np.ones(10), np.ones(10))
    assert s.cut(2, 5).wave.tolist() == [2., 3., 4., 5.]


def test_synth_shift_matches_rv():
    # a noiseless spectrum at +50 km/s puts the K I 7701 Å minimum at λ0·sqrt((1+β)/(1−β))
    template = make_template((7690., 7712.), n_random=0)  # anchor lines only
    spec, _ = make_spectrum(rv=50., resolution=20000., snr=np.inf, wave_range=(7690., 7712.), template=template)
    imin = np.argmin(spec.flux)
    beta = 50. / C_KMS
    expected = 7701.084 * np.sqrt((1 + beta) / (1 - beta))
    assert spec.wave[imin] == pytest.approx(expected, abs=0.1)


def test_synth_is_deterministic():
    a, _ = make_spectrum(rv=10., rng=np.random.default_rng(5))
    b, _ = make_spectrum(rv=10., rng=np.random.default_rng(5))
    assert np.array_equal(a.flux, b.flux)
