"""rvfitter inject on a tiny run: a synthetic template through the pipeline, the summary table and the CLI."""
import numpy as np
import pytest
from astropy.table import Table

import injection
import synth
from rvfitter import cli, forecast, inject
from rvfitter.types import Spectrum


def _setup():
    w, f = synth.make_template((7600., 8800.), rng=np.random.default_rng(0))
    tpl = Spectrum(w, f, np.full(len(w), np.nan), None, {'object': 'synthetic'}, 'synthetic')
    cfg = injection.cfg_of({'methods': {'enabled': ['template_fit'],
                                        'options': {'template_fit': {'template': (w, f), 'segment_res': 0.}}}})
    return cfg, tpl, forecast.wave_grid(7700., 8700., 2500.), 2500.


def test_summarise():
    s = inject.summarise(np.zeros(4), np.array([1., -1., 1., np.nan]), np.ones(4))
    assert s['n'] == 3 and s['n_failed'] == 1 and s['pull_std'] == pytest.approx(np.std([1, -1, 1], ddof=1))


def test_tiny_run():
    cfg, tpl, wave, r = _setup()
    out = inject.inject(cfg, tpl, wave, r, 50., 0., [10., -30.], 2, seed=3)
    assert list(out['method']) == ['template_fit', 'adopted']
    row = out[0]
    assert row['n'] == 4 and row['n_failed'] == 0 and abs(row['bias']) < 3. and 0.2 < row['pull_std'] < 5.
    again = inject.inject(cfg, tpl, wave, r, 50., 0., [10., -30.], 2, seed=3)
    assert np.array_equal(out['bias'], again['bias'])  # seeded


def test_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(forecast, 'setup', lambda args: _setup())
    out = tmp_path / 'inj.csv'
    assert cli.main(['inject', '--snr', '40', '--wave-range', '7700', '8700', '--rv', '5', '--n', '2',
                     '--out', str(out)]) == 0
    t = Table.read(out, format='ascii.csv')
    assert list(t['method']) == ['template_fit', 'adopted'] and t['n'][0] == 2
