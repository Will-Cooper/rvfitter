import logging
import types

import numpy as np
import pytest

from rvfitter import archives, barycentric, combine, io, pipeline, regions, results, systematics
from rvfitter import methods as methods_pkg
from rvfitter.types import Measurement, MethodResult, Spectrum


def spec(name, i=0):
    w = np.linspace(7000, 8000, 50)
    return Spectrum(w, np.ones_like(w), np.ones_like(w), 5000.0, {'object': name}, f'{name}_{i}.fits')


class FakeMethod:
    def __init__(self, name, fail=False):
        self.name, self.fail = name, fail

    def measure(self, spectrum, regs, cfg, *, executor, rng):
        if self.fail:
            raise RuntimeError('boom')
        rv = 10.0 + rng.normal()
        return MethodResult(self.name, rv, 1.0, 1, [Measurement(self.name, 'r', rv, 1.0)])


def fresh_spectra():
    return [spec('B', 0), spec('A', 0), spec('A', 1), spec('C', 0)]


@pytest.fixture
def fakes(monkeypatch):
    spectra = fresh_spectra()
    monkeypatch.setattr(archives, 'collect_inputs', lambda p: [p], raising=False)
    monkeypatch.setattr(archives, 'group_by_object', lambda sp, override=None: {
        n: [s for s in sp if s.meta['object'] == n] for n in dict.fromkeys(s.meta['object'] for s in sp)},
        raising=False)
    monkeypatch.setattr(io, 'read_spectrum', lambda p, **kw: spectra.pop(0), raising=False)
    monkeypatch.setattr(barycentric, 'correction', lambda s, mode='auto', frame='barycentric', overrides=None:
                        (-5.0, 'computed'), raising=False)
    monkeypatch.setattr(barycentric, 'apply', lambda s, kms: s, raising=False)
    monkeypatch.setattr(regions, 'select', lambda s, **kw: {m: [] for m in kw['methods']}, raising=False)
    monkeypatch.setattr(combine, 'combine_regions',
                        lambda ms: (float(np.mean([m.rv for m in ms])), 1.0), raising=False)
    monkeypatch.setattr(combine, 'hierarchical', lambda v, e, ln, m: {
        'median': float(np.median(v)), 'err': 1.0, 'q16': 0.0, 'q84': 1.0, 'n_methods': len(set(m)),
        'n_lines': len(v)}, raising=False)
    monkeypatch.setattr(systematics, 'wavelength_zero_point', lambda s: 0.5, raising=False)
    monkeypatch.setattr(systematics, 'template_spread', lambda r: 0.4, raising=False)
    monkeypatch.setattr(methods_pkg, 'REGISTRY', {
        'ccf': FakeMethod('ccf'), 'linecentre': FakeMethod('linecentre'), 'lsd': FakeMethod('lsd', fail=True)},
        raising=False)
    return spectra


CFG = types.SimpleNamespace(seed=7, methods=['linecentre', 'ccf', 'lsd'], hash=lambda: 'h')
INPUTS = ['p1', 'p2', 'p3', 'p4']


def test_sequencing_and_failure(fakes, caplog):
    caplog.set_level(logging.INFO, logger='rvfitter')
    caplog.set_level(logging.INFO)
    res = pipeline.run(CFG, INPUTS, n_procs=1)
    assert [r.object for r in res] == ['A', 'B', 'C']
    msgs = [r.getMessage() for r in caplog.records]
    idx = [msgs.index(f'[{i}/3] {n}') for i, n in enumerate('ABC', 1)]
    assert idx == sorted(idx)
    assert any(m.startswith('ccf: ') and 'km/s' in m for m in msgs)
    assert any('adopted RV' in m for m in msgs)
    a = res[0]
    assert len(a.files) == 2 and a.barycentric['source'] == 'computed'
    assert np.isnan(results.to_table([a]).loc[0, 'rv_lsd'])
    assert a.methods['lsd'].measurements[0].flag & 1
    assert a.adopted['n_methods'] == 2 and a.systematics['template_spread'] == 0.4
    assert a.config_hash == 'h'
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_determinism_across_procs(fakes, monkeypatch):
    t1 = results.to_table(pipeline.run(CFG, INPUTS, n_procs=1))
    spectra = fresh_spectra()
    monkeypatch.setattr(io, 'read_spectrum', lambda p, **kw: spectra.pop(0), raising=False)
    t2 = results.to_table(pipeline.run(CFG, INPUTS, n_procs=2))
    assert t1.equals(t2)
    assert t1['rv_ccf'].nunique() == 3


def test_methods_override(fakes):
    res = pipeline.run(CFG, ['p1'], n_procs=1, methods=['ccf'])
    assert list(res[0].methods) == ['ccf']
