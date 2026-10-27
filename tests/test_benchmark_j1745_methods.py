"""
J1745−1640 benchmark for the methods package (needs RVFITTER_TESTDATA).

The GTC/OSIRIS R2500I telluric-corrected spectrum is PypeIt vacuum and already heliocentric, so no barycentric
step is applied. Until the io and regions packages are merged, the spectrum is read with astropy and the
regions are built by hand: two broad windows split at the middle of the spectrum, with the O2 A band and the
strongest H2O band masked (the plan's R ≲ 3000 optical default). Methods are compared with the benchmark
26.2 ± 2.3 km/s through z = |RV − 26.2| / √(2.3² + σ²); z ≤ 2 is required, and the values are recorded in
docs/dev_notes_methods.md. Nothing here is tuned towards the benchmark.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from rvfitter.types import Region, Spectrum

pytestmark = pytest.mark.data

BENCH_RV, BENCH_ERR = 26.2, 2.3
REL = Path('pypeit/R2500I/GTC54-15A0/OB0055/Science/DENISJ174534.6-164053_tellcorr.fits')
REGIONS = [Region('W1', 'ccf', 7400., 8600., reason='benchmark window'),
           Region('W2', 'ccf', 8600., 10000., reason='benchmark window'),
           Region('O2 A', 'telluric', 7590., 7710., reason='O2 A band'),
           Region('H2O', 'telluric', 9270., 9680., reason='strong H2O')]


def _root() -> Path:
    root = os.environ.get('RVFITTER_TESTDATA')
    if not root:
        pytest.skip('RVFITTER_TESTDATA not set')
    return Path(root)


@pytest.fixture(scope='module')
def j1745():
    from astropy.io import fits
    path = _root() / REL
    d = fits.getdata(path, 1)
    wave, flux, ivar = (np.asarray(d[c], float) for c in ('wave', 'flux', 'ivar'))
    good = ivar > 0
    sigma = np.where(good, 1. / np.sqrt(np.where(good, ivar, 1.)), np.nan)
    return Spectrum(wave, np.where(good, flux, np.nan), sigma, 2500.,
                    {'object': 'J1745-1640', 'medium_in': 'vacuum', 'frame_in': 'heliocentric'}, str(path))


@pytest.fixture(scope='module')
def cfg():
    return {'templates': {'local_dir': str(_root() / 'templates' / 'bt-settl-cifist' / 'useful'),
                          'grid': 'bt-settl-cifist', 'teff_prior': 2100.},
            'methods': {'options': {'ccf': {'n_mc': 100}, 'template_fit': {}}}}


def _z(rv, err):
    return abs(rv - BENCH_RV) / np.hypot(BENCH_ERR, err)


@pytest.mark.parametrize('name', ['ccf', 'template_fit'])
def test_method_on_j1745(j1745, cfg, name):
    from rvfitter.methods import REGISTRY
    res = REGISTRY[name](j1745, REGIONS, cfg, rng=np.random.default_rng(42))
    assert res.n_regions == 2 and np.isfinite(res.rv) and np.isfinite(res.rv_err)
    assert _z(res.rv, res.rv_err) <= 2., (res.rv, res.rv_err)


def test_resolution_of_j1745(j1745, cfg):
    from rvfitter.methods.template_fit import measure_resolution
    out = measure_resolution(j1745, REGIONS, cfg)
    assert 1500. < out['resolution'] < 3000.
