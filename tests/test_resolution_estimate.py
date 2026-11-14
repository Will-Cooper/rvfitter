"""R from the pixel spacing (io.estimate_resolution) and its sanity check against data/instruments.csv"""
from pathlib import Path

import numpy as np
import pytest

from rvfitter import io
from rvfitter.calibration import slit

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
TOLERANCE = 1.5  # the pixel-spacing estimate must be within this factor of the known R (it assumes 2.5 px)


def _log_grid(r: float, px: float, wave_range=(7500., 10000.)) -> np.ndarray:
    """A wavelength grid with ``px`` pixels per resolution element of resolving power ``r``"""
    dlnw = 1. / (r * px)
    return np.exp(np.arange(np.log(wave_range[0]), np.log(wave_range[1]), dlnw))


@pytest.mark.parametrize('r', [300., 2500., 20000.])
def test_estimate_at_nyquist_sampling(r):
    assert io.estimate_resolution(_log_grid(r, io.SAMPLING_PX)) == pytest.approx(r, rel=1e-3)


@pytest.mark.parametrize('px', [2., 3., 4.])
def test_estimate_scales_with_sampling(px):
    """Other sampling: the estimate is R px / 2.5, i.e. off by exactly the sampling ratio"""
    assert io.estimate_resolution(_log_grid(2000., px)) == pytest.approx(2000. * px / io.SAMPLING_PX, rel=1e-3)


def test_estimate_needs_increasing_steps():
    with pytest.raises(ValueError, match='no increasing'):
        io.estimate_resolution(np.array([5000., 5000.]))


def test_instrument_range_check():
    meta = {'pypeit_spec': None, 'instrument_names': ('ARCoIRIS',)}
    assert io.implausible_resolution(3000., meta) is None
    assert 'ARCoIRIS' in io.implausible_resolution(3500. * 2, meta)  # a 2x finer (rebinned) grid
    assert 'without a documented R range' in io.implausible_resolution(io.R_ESTIMATE_CEILING + 1, {})
    assert io.implausible_resolution(io.R_ESTIMATE_CEILING, {}) is None


def test_every_resolution_range_has_a_source():
    rows = slit._table().values()
    assert all(r.resolution_source for r in rows if r.resolution_min is not None)


def _write_ascii(path: Path, wave: np.ndarray, header: str = '') -> Path:
    rng = np.random.default_rng(0)
    rows = '\n'.join(f'{w:.6f} {1 + 0.01 * rng.normal():.6f} 0.01' for w in wave)
    path.write_text(f'# AIRORVAC = vac\n{header}# wave flux err\n{rows}\n')
    return path


def test_read_spectrum_flags_a_rebinned_spectrum(tmp_path, caplog):
    native = _log_grid(2500., io.SAMPLING_PX)
    ok = io.read_spectrum(_write_ascii(tmp_path / 'native.txt', native))
    assert ok.meta['resolution_source'] == 'estimated' and not ok.meta['resolution_implausible']
    assert ok.resolution == pytest.approx(2500., rel=0.01)
    fine = np.linspace(native[0], native[-1], 3 * len(native))  # rebinned onto a 3x finer grid
    spec = io.read_spectrum(_write_ascii(tmp_path / 'rebinned.txt', fine))
    assert spec.meta['resolution_implausible'] and spec.resolution > io.R_ESTIMATE_CEILING
    assert 'IMPLAUSIBLE' in caplog.text


def test_header_and_config_resolution_are_not_estimated(tmp_path):
    path = _write_ascii(tmp_path / 's.txt', _log_grid(2500., 2.5), '# RESOLUTION = 3000\n')
    assert io.read_spectrum(path).meta['resolution_source'] == 'header'
    assert io.read_spectrum(path, resolution=4000.).meta['resolution_source'] == 'config'


@pytest.mark.parametrize('name, known', [
    ('gtc_osiris/DENISJ174534.6-164053_tellcorr.fits', 2602.),  # Cooper et al. 2024
    ('near_infrared/sxd_J0808+3157.fits', 2000.),  # SpeX SXD
    ('near_infrared/xtc_J1536+0646.fits', 3500.),  # ARCoIRIS
])
def test_examples_within_tolerance(name, known):
    spec = io.read_spectrum(EXAMPLES / name)
    r = io.estimate_resolution(spec.wave)
    assert known / TOLERANCE < r < known * TOLERANCE
    assert io.implausible_resolution(r, spec.meta) is None


def test_j1745_tellcorr_is_estimated_and_plausible():
    spec = io.read_spectrum(EXAMPLES / 'gtc_osiris/DENISJ174534.6-164053_tellcorr.fits')
    assert spec.meta['resolution_source'] == 'estimated' and not spec.meta['resolution_implausible']
