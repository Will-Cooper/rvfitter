import logging
import threading
import types

import numpy as np
import pytest

from rvfitter import archives, barycentric, combine, io, pipeline, regions, results, systematics
from rvfitter import methods as methods_pkg
from rvfitter.types import Measurement, MethodResult, Spectrum


def spec(name, i=0):
    w = np.linspace(7000, 8000, 50)
    return Spectrum(w, np.ones_like(w), np.ones_like(w), 5000.0, {'object': name}, f'{name}_{i}.fits')


def _work(x):
    return x * x


class FakeMethod:
    def __init__(self, name, fail=False):
        self.name, self.fail = name, fail

    def measure(self, spectrum, regs, cfg, *, executor, rng):
        if self.fail:
            raise RuntimeError('boom')
        extra = executor.submit(_work, 3).result() if executor is not None else _work(3)
        rv = 10.0 + rng.normal() + extra
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
    calls = {}

    def select(s, **kw):
        calls['select'] = kw
        return {m: [] for m in kw['methods']}

    monkeypatch.setattr(regions, 'select', select, raising=False)
    monkeypatch.setattr(regions, 'SelectOptions', types.SimpleNamespace, raising=False)
    monkeypatch.setattr(combine, 'combine_regions',
                        lambda ms: (float(np.mean([m.rv for m in ms])), 1.0), raising=False)

    def hierarchical(v, e, ln, m, *, tau_scale=None, ngrid=200):
        calls['tau'] = tau_scale
        return {'median': float(np.median(v)), 'err': 1.0, 'q16': 0.0, 'q84': 1.0, 'n_methods': len(set(m)),
                'n_lines': len(v)}

    monkeypatch.setattr(combine, 'hierarchical', hierarchical, raising=False)
    monkeypatch.setattr(systematics, 'wavelength_zero_point', lambda s, c: 0.5, raising=False)
    monkeypatch.setattr(systematics, 'template_spread', lambda s, r, c, *, executor=None, rng: 0.4, raising=False)
    monkeypatch.setattr(methods_pkg, 'REGISTRY', {
        'ccf': FakeMethod('ccf'), 'linecentre': FakeMethod('linecentre'), 'lsd': FakeMethod('lsd', fail=True)},
        raising=False)
    return types.SimpleNamespace(calls=calls)


NS = types.SimpleNamespace
CFG = NS(seed=7, methods=NS(enabled=['linecentre', 'ccf', 'lsd'], options={}), hash=lambda: 'h',
         input=NS(medium='auto', hdu=None, resolution=None, paths=['p1']),
         regions=NS(overrides={'ccf': [{'name': 'x', 'remove': True}]}, min_snr=7.0, telluric_strictness=None),
         combine=NS(tau_scale=0.3, systematic_floor=0.2), grouping=NS(override=None),
         velocity_frame=NS(correction='auto', frame='barycentric', overrides={}), templates=NS(teff_prior=None),
         parallel=NS(n_procs=1))
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


def test_config_overrides_passed(fakes):
    res = pipeline.run(CFG, ['p1'], n_procs=1)
    assert fakes.calls['select']['overrides'] == CFG.regions.overrides
    assert fakes.calls['tau'] == 0.3
    opts = fakes.calls['select']['options']
    assert opts.snr_min == 7.0 and not hasattr(opts, 'telluric')
    assert res[0].systematics['floor'] == 0.2
    df = results.to_table(res)
    expect = np.sqrt(df.loc[0, 'rv_adopted_err'] ** 2 + 0.5**2 + 0.4**2 + 0.2**2)
    assert df.loc[0, 'rv_total_err'] == pytest.approx(expect)


def test_on_object_callback(fakes):
    seen = []
    res = pipeline.run(CFG, INPUTS, n_procs=1, on_object=lambda r: seen.append(r.object))
    assert seen == [r.object for r in res] == ['A', 'B', 'C']


def test_cancel_between_objects(fakes):
    cancel = threading.Event()
    seen = []

    def cb(r):
        seen.append(r.object)
        cancel.set()

    res = pipeline.run(CFG, INPUTS, n_procs=1, on_object=cb, cancel=cancel)
    assert [r.object for r in res] == ['A'] == seen


def test_cancel_between_methods(fakes, monkeypatch):
    cancel = threading.Event()
    reg = methods_pkg.REGISTRY
    orig = reg['linecentre'].measure

    def measure(*a, **k):
        cancel.set()
        return orig(*a, **k)

    monkeypatch.setattr(reg['linecentre'], 'measure', measure)
    seen = []
    res = pipeline.run(CFG, INPUTS, n_procs=1, on_object=seen.append, cancel=cancel)
    assert res == [] and seen == []
