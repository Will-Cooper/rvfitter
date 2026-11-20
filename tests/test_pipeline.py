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
    monkeypatch.setattr(archives, 'read_spectra', lambda p, **kw: [spectra.pop(0)])
    monkeypatch.setattr(barycentric, 'correction', lambda s, mode='auto', frame='barycentric', overrides=None:
                        (-5.0, 'computed'), raising=False)
    monkeypatch.setattr(barycentric, 'apply', lambda s, kms: s, raising=False)
    calls = {}

    def select(s, **kw):
        calls['select'] = kw
        return {m: [] for m in kw['methods']}

    monkeypatch.setattr(regions, 'select', select, raising=False)
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
    monkeypatch.setattr(systematics, 'template_spread_details',
                        lambda s, r, c, *, executor=None, rng, method: dict(
                            total=0.4, delta_modes=0.3, window_modes=dict(full=1.3, features=1.)))
    monkeypatch.setattr(pipeline, 'REGISTRY', {'ccf': FakeMethod('ccf'), 'template_fit': FakeMethod('template_fit'),
                                               'telluric_ref': FakeMethod('telluric_ref', fail=True)})
    return types.SimpleNamespace(calls=calls)


CFG = config.Config()
CFG.seed = 7
CFG.methods.enabled = ['template_fit', 'ccf', 'telluric_ref']
CFG.input.paths = ['p1']
CFG.regions.overrides = {'ccf': [{'name': 'x', 'remove': True}]}
CFG.calibration.zero_point_known = True  # the fake spectra carry no slit information: no slit-centring term
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
    monkeypatch.setattr(archives, 'read_spectra', lambda p, **kw: [spectra.pop(0)])
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


def test_forward_mcmc_epochs_share_one_fit(fakes, monkeypatch):
    """forward_mcmc on a multi-epoch object goes through forward.fit_epochs once; single epochs use measure"""
    calls = []

    def fit_epochs(spectra, regs, cfg, *, executor, rng):
        calls.append(len(spectra))
        return [MethodResult('forward_mcmc', 10. + k, 1., 1, [Measurement('forward_mcmc', 'r', 10. + k, 1.)])
                for k in range(len(spectra))]

    monkeypatch.setattr(pipeline.forward, 'fit_epochs', fit_epochs)
    monkeypatch.setitem(pipeline.REGISTRY, 'forward_mcmc', FakeMethod('forward_mcmc'))
    cfg = config.Config()
    cfg.methods.enabled = ['ccf', 'forward_mcmc']
    res = {r.object: r for r in pipeline.run(cfg, INPUTS, n_procs=1)}
    assert calls == [2]  # A only
    assert [m.rv for m in res['A'].methods['forward_mcmc'].measurements] == [10., 11.]
    assert res['B'].methods['forward_mcmc'].rv > 12.  # FakeMethod: 10 + noise + 9


@pytest.mark.parametrize('enabled', [False, True])
def test_binary_test_only_when_enabled(fakes, monkeypatch, enabled):
    calls = []

    def fake_test(spectrum, regs, cfg):
        calls.append(spectrum.source)
        return {'dbic': 12. if spectrum.source == 'A_1.fits' else -3., 'flag': spectrum.source == 'A_1.fits'}

    monkeypatch.setattr(pipeline.binary, 'test', fake_test)
    cfg = config.Config()
    cfg.methods.enabled = ['ccf']
    cfg.methods.binary = enabled
    table = results.to_table(pipeline.run(cfg, INPUTS, n_procs=1)).set_index('object')
    if not enabled:
        assert not calls and table['binary_dbic'].isna().all()
        return
    assert len(calls) == 4  # every epoch; A has two
    assert table.loc['A', 'binary_dbic'] == 12. and bool(table.loc['A', 'binary_flag'])
    assert table.loc['B', 'binary_dbic'] == -3. and not bool(table.loc['B', 'binary_flag'])


def test_template_window_delta_is_a_column_and_warns_only_beyond_3_total_errors(fakes, monkeypatch, caplog):
    for delta, warns in ((0.3, False), (-317., True)):
        monkeypatch.setattr(systematics, 'template_spread_details', lambda *a, d=delta, **k: dict(
            total=0.4, delta_modes=d, window_modes=dict(full=1. + d, features=1.)))
        caplog.clear()
        res = pipeline.run(config.loads(config.dumps(CFG)), INPUTS[:1], n_procs=1, failures=[])
        row = results.to_table(res).iloc[0]
        assert row['template_window_delta_kms'] == delta and row['sys_template'] == 0.4
        assert any('full-span template fit (-316.00 km/s)' in r.getMessage() for r in caplog.records
                   if r.levelno == logging.WARNING) is warns


def test_failed_template_method_leaves_no_template_term(fakes, monkeypatch):
    """A template method enabled but failing everywhere (e.g. an optical grid on NIR data) contributes nothing, so
    model-free line centring alone has no template systematic (absent, not unknown)"""
    monkeypatch.setattr(pipeline, 'REGISTRY', {'linecentre': FakeMethod('linecentre'),
                                               'template_fit': FakeMethod('template_fit', fail=True)})
    monkeypatch.setattr(systematics, 'template_spread_details', lambda *a, **k: dict(total=np.nan, delta_modes=np.nan))
    cfg = config.loads(config.dumps(CFG))
    cfg.methods.enabled = ['linecentre', 'template_fit']
    res = pipeline.run(cfg, INPUTS, n_procs=1, failures=[])
    assert all(r.systematics['template_spread'] == 0. for r in res)


@pytest.mark.parametrize('enabled, expected', [(['ccf'], 'ccf'), (['lsd'], 'lsd'),
                                               (['ccf', 'template_fit'], 'template_fit'),
                                               (['linecentre', 'lsd', 'ccf'], 'lsd')])
def test_template_spread_uses_an_enabled_template_method(fakes, monkeypatch, enabled, expected):
    """The spread is measured with a used template method on its own regions (template_fit first), never with
    template_fit when it is not enabled (it had no regions: sys_template was NaN for ccf or lsd alone)"""
    seen = []

    def details(s, sel, c, *, executor=None, rng, method):
        assert method in sel  # its own selected regions
        seen.append(method)
        return dict(total=0.4, delta_modes=0.3, window_modes=dict(full=1.3, features=1.))

    monkeypatch.setattr(systematics, 'template_spread_details', details)
    monkeypatch.setattr(pipeline, 'REGISTRY', {m: FakeMethod(m) for m in enabled})
    cfg = config.loads(config.dumps(CFG))
    cfg.methods.enabled = enabled
    res = pipeline.run(cfg, INPUTS[:1], n_procs=1, failures=[])
    assert seen == [expected] and res[0].systematics['template_spread'] == 0.4
    assert np.isfinite(results.to_table(res).loc[0, 'rv_total_err'])


def test_shape_corrected_linecentre_alone_has_a_finite_template_term(fakes, monkeypatch):
    """Shape-corrected line centring alone has no template_fit regions (sys_template was NaN): its term is the
    weighted mean shift of the lines' shape offsets for each ±1 Teff-step template, RMS over the steps"""
    offs = [{'nom': 1., 'lo': 3., 'hi': 0.}, {'nom': 2., 'lo': 4., 'hi': 1.}]  # shifts +2 and −1 on both lines

    def lc(spectrum, regs, cfg, *, executor, rng):
        ms = [Measurement('linecentre', f'l{i}', 10., 1., 0, dict(shape_correction=True, shape_template='nom',
                                                                   shape_offsets=o)) for i, o in enumerate(offs)]
        return MethodResult('linecentre', 10., 1., 2, ms)

    monkeypatch.setattr(pipeline, 'REGISTRY', {'linecentre': lc})
    monkeypatch.setattr(systematics, 'template_spread_details', lambda *a, **k: pytest.fail('no template_fit run'))
    cfg = config.loads(config.dumps(CFG))
    cfg.methods.enabled = ['linecentre']
    res = pipeline.run(cfg, INPUTS[:1], n_procs=1, failures=[])
    assert res[0].systematics['template_spread'] == pytest.approx(np.sqrt((4. + 1.) / 2.))
    assert np.isfinite(results.to_table(res).loc[0, 'rv_total_err'])


@pytest.mark.parametrize('enabled, expected', [(['ccf', 'template_fit'], 1), (['ccf', 'forward_mcmc'], 8)])
def test_automatic_n_procs(fakes, monkeypatch, enabled, expected):
    """Automatic n_procs: serial for the default methods (a pool costs more than it saves), cpu_count - 1 for MCMC"""
    seen = []
    monkeypatch.setattr(pipeline, '_measure', lambda spectra, cfg, n, rng, **kw: seen.append(n) or
                        pipeline.measure_object(spectra, cfg, None, rng, **kw))
    monkeypatch.setattr(pipeline.os, 'cpu_count', lambda: 9)
    monkeypatch.setitem(pipeline.REGISTRY, 'forward_mcmc', FakeMethod('forward_mcmc'))
    cfg = config.loads(config.dumps(CFG))
    cfg.parallel.n_procs = None
    cfg.methods.enabled = enabled
    pipeline.run(cfg, INPUTS, failures=[])
    assert set(seen) == {expected}


def test_on_step_before_each_method_with_automatic_processes(fakes):
    """The GUI bar: on_step runs in the calling process before every (epoch, method), with the default n_procs"""
    assert CFG.parallel.n_procs is None  # the default: the automatic rule (one process without forward_mcmc)
    events = []
    pipeline.run(CFG, INPUTS, on_step=lambda obj, m: events.append((obj, m)),
                 on_object=lambda res: events.append((res.object, 'done')))
    methods = CFG.methods.enabled
    assert events == ([('A', m) for m in methods] * 2 + [('A', 'done')] + [('B', m) for m in methods]
                      + [('B', 'done')] + [('C', m) for m in methods] + [('C', 'done')])


def test_results_record_an_estimated_resolution(fakes, monkeypatch):
    spectra = fresh_spectra()
    for s in spectra:
        s.meta.update(resolution_source='estimated', resolution_implausible=s.meta['object'] == 'C')
    monkeypatch.setattr(archives, 'read_spectra', lambda p, **kw: [spectra.pop(0)])
    table = results.to_table(pipeline.run(CFG, INPUTS, n_procs=1))
    assert list(table['resolution_source']) == ['estimated;estimated', 'estimated', 'estimated']
    assert (table['rv_ccf_flag'] & results.FLAGS['resolution_estimated']).all()
    assert list(table['rv_ccf_flag'] & results.FLAGS['resolution_implausible'] > 0) == [False, False, True]
    assert (table['resolution'] == 5000.).all()


def test_resolution_too_low_is_warned_and_flagged(fakes, monkeypatch, caplog):
    from dataclasses import replace
    spectra = [replace(s, resolution=100. if s.meta['object'] == 'C' else s.resolution) for s in fresh_spectra()]
    monkeypatch.setattr(archives, 'read_spectra', lambda p, **kw: [spectra.pop(0)])
    caplog.set_level(logging.WARNING, logger='rvfitter')
    table = results.to_table(pipeline.run(CFG, INPUTS, n_procs=1))
    assert list(table['rv_ccf_flag'] & results.FLAGS['resolution_too_low'] > 0) == [False, False, True]
    assert 'C: R 100 is below 500' in caplog.text


REAL_CORRECTION = barycentric.correction  # before the fakes fixture replaces it


def test_unknown_frame_is_warned_and_flagged(fakes, monkeypatch, caplog):
    """What the command line does when auto cannot tell the frame: nothing applied, a loud WARNING, a flag"""
    monkeypatch.setattr(barycentric, 'correction', REAL_CORRECTION)
    caplog.set_level(logging.WARNING, logger='rvfitter')
    table = results.to_table(pipeline.run(CFG, INPUTS, n_procs=1))
    assert (table['bary_source'] == 'unknown').all() and (table['bary_kms'] == 0.).all()
    assert (table['zp_flags'] & wcorr.ZP_FLAGS['frame_unknown']).all()
    assert 'VELOCITY FRAME UNKNOWN for B_0.fits' in caplog.text and '±30.0 km/s' in caplog.text


def test_wavelength_shift_is_minus_the_applied_dv_at_the_regions():
    from rvfitter.calibration import sky_zeropoint
    from rvfitter.types import Region
    s = sky_zeropoint.apply(spec('A'), None, constant_kms=5.)
    sel = {'ccf': [Region('a', 'line', 7400., 7410., 7405.), Region('t', 'telluric', 7600., 7700., None)]}
    assert pipeline.wavelength_shift(s, sel) == pytest.approx(-5.)
    assert pipeline.wavelength_shift(spec('A'), sel) == 0.
