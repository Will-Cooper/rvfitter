import logging
import threading
import types

import numpy as np
import pytest

from rvfitter import archives, barycentric, combine, config, pipeline, regions, results, systematics
from rvfitter.calibration import wavelength_correction as wcorr
from rvfitter.types import Measurement, MethodResult, Spectrum


def spec(name, i=0):
    w = np.linspace(7000, 8000, 50)
    return Spectrum(w, np.ones_like(w), np.ones_like(w), 5000.0, {'object': name}, f'{name}_{i}.fits')


def _work(x):
    return x * x


class FakeMethod:
    def __init__(self, name, fail=False):
        self.name, self.fail = name, fail

    def __call__(self, spectrum, regs, cfg, *, executor, rng):
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
    monkeypatch.setattr(archives, 'read_spectrum', lambda p, **kw: spectra.pop(0))
    monkeypatch.setattr(barycentric, 'correction', lambda s, mode='auto', frame='barycentric', overrides=None:
                        (-5.0, 'computed'), raising=False)
    monkeypatch.setattr(barycentric, 'apply', lambda s, kms: s, raising=False)
    calls = {}

    def select(s, **kw):
        calls['select'] = kw
        return {m: [] for m in kw['methods']}

    monkeypatch.setattr(regions, 'select', select, raising=False)
    monkeypatch.setattr(regions, 'SelectOptions', types.SimpleNamespace, raising=False)
    def combine_regions(ms, details):
        details['tau_line'] = 0.
        return float(np.mean([m.rv for m in ms])), 1.0

    monkeypatch.setattr(combine, 'combine_regions', combine_regions)

    def hierarchical(v, e, ln, m, *, tau_scale=None, ngrid=200):
        calls['tau'] = tau_scale
        return {'median': float(np.median(v)), 'err': 1.0, 'q16': 0.0, 'q84': 1.0, 'n_methods': len(set(m)),
                'n_lines': len(v), 'tau_line': 0., 'tau_method': 0.}

    monkeypatch.setattr(combine, 'hierarchical', hierarchical, raising=False)
    monkeypatch.setattr(systematics, 'wavelength_zero_point', lambda s, **kw: 0.5)
    monkeypatch.setattr(systematics, 'template_spread', lambda s, r, c, *, executor=None, rng: 0.4)
    monkeypatch.setattr(pipeline, 'REGISTRY', {'ccf': FakeMethod('ccf'), 'template_fit': FakeMethod('template_fit'),
                                               'telluric_ref': FakeMethod('telluric_ref', fail=True)})
    return types.SimpleNamespace(calls=calls)


CFG = config.Config()
CFG.seed = 7
CFG.methods.enabled = ['template_fit', 'ccf', 'telluric_ref']
CFG.input.paths = ['p1']
CFG.regions.overrides = {'ccf': [{'name': 'x', 'remove': True}]}
CFG.regions.min_snr = 7.0
CFG.combine.tau_scale = 0.3
CFG.combine.systematic_floor = 0.2
INPUTS = ['p1', 'p2', 'p3', 'p4']


def test_sequencing_and_failure(fakes, caplog):
    caplog.set_level(logging.INFO, logger='rvfitter')
    caplog.set_level(logging.INFO)
    failures = []
    res = pipeline.run(CFG, INPUTS, n_procs=1, failures=failures)
    assert [r.object for r in res] == ['A', 'B', 'C']
    assert [f[0] for f in failures] == ['A telluric_ref'] * 2 + ['B telluric_ref', 'C telluric_ref']  # A has 2 epochs
    assert failures[0][1] == 'RuntimeError: boom'
    msgs = [r.getMessage() for r in caplog.records]
    idx = [msgs.index(f'[{i}/3] {n}') for i, n in enumerate('ABC', 1)]
    assert idx == sorted(idx)
    assert any(m.startswith('ccf: ') and 'km/s' in m for m in msgs)
    assert any('adopted RV' in m for m in msgs)
    a = res[0]
    assert len(a.files) == 2 and a.barycentric['source'] == 'computed'
    assert np.isnan(results.to_table([a]).loc[0, 'rv_telluric_ref'])
    failed = a.methods['telluric_ref'].measurements[0]
    assert failed.flag & 1 and failed.diagnostics['error'] == 'RuntimeError: boom'
    assert a.adopted['n_methods'] == 2 and a.systematics['template_spread'] == 0.4
    assert a.config_hash == CFG.hash()
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors[0].exc_info is not None  # the traceback is logged
    assert any('4 failure(s)' in r.getMessage() for r in errors)  # the end-of-run summary


def test_determinism_across_procs(fakes, monkeypatch):
    t1 = results.to_table(pipeline.run(CFG, INPUTS, n_procs=1))
    spectra = fresh_spectra()
    monkeypatch.setattr(archives, 'read_spectrum', lambda p, **kw: spectra.pop(0))
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
    assert opts.snr_min == 7.0 and opts.telluric == 1.0
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
    orig = pipeline.REGISTRY['template_fit']

    def measure(*a, **k):
        cancel.set()
        return orig(*a, **k)

    monkeypatch.setitem(pipeline.REGISTRY, 'template_fit', measure)
    seen = []
    res = pipeline.run(CFG, INPUTS, n_procs=1, on_object=seen.append, cancel=cancel)
    assert res == [] and seen == []


def test_valid_range_config_overrides_arc_range(caplog):
    s = spec('A')
    s.meta.update(arc_range=(7437., 9926.), arc_file='wvcalib.fits')
    cfg = config.Config()
    with caplog.at_level(logging.INFO, logger='rvfitter.calibration.wavelength_correction'):
        assert wcorr.valid_range(cfg, 'obj', None, s) == (7437., 9926.)
    assert 'arc-line range' in caplog.text
    cfg.calibration.wave_valid_range = (7600., 9000.)
    assert wcorr.valid_range(cfg, 'obj', s) == (7600., 9000.)
    assert wcorr.valid_range(config.Config(), 'obj', spec('B')) is None
