"""
J1745−1640 benchmark for the methods package (needs RVFITTER_TESTDATA).

The GTC/OSIRIS R2500I telluric-corrected spectrum is PypeIt vacuum and already heliocentric, so the barycentric
step is skipped. Regions come from `regions.select` with `telluric_corrected=True`, and R from
`estimate_resolution` (the file has none). Methods are compared with the benchmark 26.2 ± 2.3 km/s through
z = |RV − 26.2| / √(2.3² + σ²); z ≤ 2 is required, and the values are recorded in docs/dev_notes_methods.md.
Nothing here is tuned towards the benchmark.
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.data

BENCH_RV, BENCH_ERR = 26.2, 2.3
REL = Path('pypeit/R2500I/GTC54-15A0/OB0055/Science/DENISJ174534.6-164053_tellcorr.fits')


def _root() -> Path:
    root = os.environ.get('RVFITTER_TESTDATA')
    if not root:
        pytest.skip('RVFITTER_TESTDATA not set')
    return Path(root)


@pytest.fixture(scope='module')
def j1745():
    from rvfitter import io
    return io.read_spectrum(_root() / REL, medium='vacuum')


@pytest.fixture(scope='module')
def cfg():
    from rvfitter import config
    c = config.Config()
    c.templates.local_dir = str(_root() / 'templates' / 'bt-settl-cifist' / 'useful')
    c.templates.teff_prior = 2100.
    c.input.telluric_corrected = True
    c.velocity_frame.correction = 'skip'
    c.velocity_frame.frame = 'heliocentric'
    c.methods.enabled = ['ccf', 'template_fit']
    c.methods.options = {'ccf': {'n_mc': 100}}
    return c


@pytest.fixture(scope='module')
def resolution(j1745, cfg):
    from rvfitter.methods import estimate_resolution
    return estimate_resolution(j1745, cfg=cfg)


def _z(rv, err):
    return abs(rv - BENCH_RV) / np.hypot(BENCH_ERR, err)


def test_estimate_resolution(resolution):
    assert resolution['method'] == 'template_fit'
    assert 1800. < resolution['resolution'] < 2800. and resolution['resolution_err'] < 400.


@pytest.mark.parametrize('name', ['ccf', 'template_fit'])
def test_method_on_j1745(j1745, cfg, resolution, name):
    from rvfitter import regions
    from rvfitter.methods import REGISTRY
    cfg.input.resolution = round(resolution['resolution'])
    sel = regions.select(j1745, teff=cfg.templates.teff_prior, methods=[name],
                         options=regions.SelectOptions(telluric_corrected=True))
    res = REGISTRY[name].measure(j1745, sel[name], cfg, rng=np.random.default_rng(42))
    assert res.n_regions >= 3  # the full-span window is split into segments
    assert np.isfinite(res.rv) and np.isfinite(res.rv_err)
    assert _z(res.rv, res.rv_err) <= 2., (res.rv, res.rv_err)


def test_cli_measure_writes_a_row(j1745, cfg, resolution, tmp_path):
    import pandas as pd

    from rvfitter import cli, config
    cfg.input.resolution = round(resolution['resolution'])
    toml = tmp_path / 'j1745.toml'
    config.dump(cfg, toml)
    out = tmp_path / 'results.csv'
    code = cli.main(['measure', str(_root() / REL), '--config', str(toml), '--bary', 'skip', '--telluric-corrected',
                     '--medium', 'vacuum', '--out', str(out), '--n-procs', '1', '--no-plots', '-q'])
    assert code == 0
    table = pd.read_csv(out)
    assert len(table) == 1
    for name in ('ccf', 'template_fit'):
        assert np.isfinite(table[f'rv_{name}'].iloc[0])
    assert np.isfinite(table['rv_adopted'].iloc[0])
