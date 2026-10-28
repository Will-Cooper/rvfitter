"""Tests of rvfitter.combine, ported from tests/legacy/test_a3_errors.py, plus equivalence with the legacy code."""
import ast
import logging
import typing
from pathlib import Path

import numpy as np
import pytest

from rvfitter import combine
from rvfitter.types import Measurement, MethodResult

LEGACY_UTILS = Path(__file__).resolve().parents[1] / 'legacy' / 'utils.py'


def _legacy_functions():
    """Extract the legacy combination functions from legacy/utils.py without importing its heavy dependencies"""
    tree = ast.parse(LEGACY_UTILS.read_text())
    wanted = {'combine_rvs', '_half_cauchy_logpdf', '_trapezoid_weights', 'mixture_pdf', 'hierarchical_combine'}
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {f.name for f in body} == wanted
    namespace = {'np': np, 'Tuple': tuple, 'Optional': typing.Optional, 'Sequence': typing.Sequence}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(LEGACY_UTILS), 'exec'), namespace)  # noqa: S102
    return namespace


LINES = ['k1-a', 'k1-b', 'rb1-a', 'rb1-b', 'na1-a', 'na1-b']
LINE_OFFSETS = np.array([8., -6., 3., -9., 5., -1.])
NOISE = np.array([0.3, -0.2, 0.1, -0.4, 0.2, 0.0])


def _two_methods(line_offsets, method_offset=0., err=1.):
    values = np.concatenate([20. + line_offsets + NOISE, 20. + line_offsets - NOISE + method_offset])
    return (values, np.full(len(values), err), LINES + LINES, ['lc'] * len(LINES) + ['xcorr'] * len(LINES))


def test_weighted_mean():
    assert combine.weighted_mean([12.], [3.]) == (12., 3.)
    np.testing.assert_allclose(combine.weighted_mean([10.] * 4, [2.] * 4), [10., 1.])
    np.testing.assert_allclose(combine.weighted_mean([0., 20.], [1., 1.]), [10., 10.])
    assert combine.weighted_mean([np.nan, 5.], [1., 2.]) == (5., 2.)
    assert np.isnan(combine.weighted_mean([], [])).all()


def test_combine_regions_random_effects():
    ms = [Measurement('ccf', 'a', 0., 1.), Measurement('ccf', 'b', 20., 1.)]
    details = {}
    rv, err = combine.combine_regions(ms, details=details)
    ref = combine.hierarchical([0., 20.], [1., 1.], ['a', 'b'], ['m', 'm'], tau_method=0.)
    np.testing.assert_allclose([rv, err], [ref['median'], ref['err']])
    np.testing.assert_allclose([details['birge_rv'], details['birge_err']], [10., 10.])
    assert details['tau_line'] > 5.
    # with tau >> sigma the regions are weighted almost equally, unlike inverse-variance weights
    ms = [Measurement('x', 'a', 50., 1.), Measurement('x', 'b', 30., 4.), Measurement('x', 'c', 25., 4.),
          Measurement('x', 'd', 35., 4.)]
    rv, _ = combine.combine_regions(ms, details=details)
    assert abs(rv - 35.) < 4. and details['birge_rv'] > 45.
    assert combine.combine_regions(ms[:1]) == (50., 1.)


def test_tau_zero_recovers_inverse_variance():
    values = np.array([10., 12., 15., 11.])
    errors = np.array([1., 2., 3., 1.5])
    res = combine.hierarchical(values, errors, ['a', 'b', 'a', 'b'], ['m', 'm', 'n', 'n'], tau_line=0.,
                               tau_method=0.)
    w = 1 / errors ** 2
    np.testing.assert_allclose(res['mean'], np.sum(w * values) / np.sum(w))
    np.testing.assert_allclose(res['sd'], 1 / np.sqrt(np.sum(w)))
    np.testing.assert_allclose(res['median'], res['mean'])
    np.testing.assert_allclose(res['err'], res['sd'], rtol=1e-8)


def test_single_measurement_returns_itself():
    res = combine.hierarchical([7.5], [2.], ['a'], ['m'])
    np.testing.assert_allclose([res['mean'], res['median'], res['sd'], res['err']], [7.5, 7.5, 2., 2.])


def test_single_method_warns(caplog):
    with caplog.at_level(logging.WARNING, logger='rvfitter.combine'):
        res = combine.hierarchical([1., 2., 3.], [1., 1., 1.], ['a', 'b', 'c'], ['m'] * 3)
    assert res['tau_method'] == 0. and 'tau_method' in caplog.text


def test_shared_line_offsets_inflate_error():
    quiet = combine.hierarchical(*_two_methods(np.zeros(len(LINES))))
    shared = combine.hierarchical(*_two_methods(LINE_OFFSETS))
    assert shared['sd'] > 2.5 > quiet['sd']
    assert shared['sd'] > 5 / np.sqrt(2 * len(LINES))
    assert shared['tau_line'] > 3. > quiet['tau_line']
    np.testing.assert_allclose(shared['mean'], 20. + LINE_OFFSETS.mean(), atol=0.5)


def test_method_offset_shows_in_tau_method():
    same = combine.hierarchical(*_two_methods(np.zeros(len(LINES))))
    offset = combine.hierarchical(*_two_methods(np.zeros(len(LINES)), method_offset=15.))
    assert offset['tau_method'] > 5. > same['tau_method']
    np.testing.assert_allclose(offset['mean'], 27.5, atol=0.5)
    assert offset['sd'] > 3. > same['sd']


def test_invariant_to_input_order():
    values, errors, lines, methods = _two_methods(LINE_OFFSETS, method_offset=4.)
    errors = errors * np.linspace(0.8, 1.6, len(errors))
    order = np.random.default_rng(1).permutation(len(values))
    res = combine.hierarchical(values, errors, lines, methods)
    perm = combine.hierarchical(values[order], errors[order], np.array(lines)[order], np.array(methods)[order])
    for key in ('mean', 'sd', 'median', 'q16', 'q84', 'tau_line', 'tau_method'):
        np.testing.assert_allclose(perm[key], res[key], rtol=1e-9)


def test_mixture_pdf_integrates_to_one():
    res = combine.hierarchical(*_two_methods(LINE_OFFSETS))
    x = np.linspace(res['mean'] - 15 * res['sd'], res['mean'] + 15 * res['sd'], 4001)
    pdf = combine.mixture_pdf(x, res['means'], res['sds'], res['weights'])
    np.testing.assert_allclose(np.trapezoid(pdf, x), 1., rtol=1e-3)


def test_quantiles_stable_to_grid_extent():
    args = _two_methods(LINE_OFFSETS, method_offset=6.)
    narrow = combine.hierarchical(*args, tau_max_factor=10.)
    wide = combine.hierarchical(*args, tau_max_factor=1000.)
    for key in ('median', 'q16', 'q84'):
        np.testing.assert_allclose(wide[key], narrow[key], atol=0.05)
    assert wide['sd'] > narrow['sd']


def test_large_problem_memory():
    import tracemalloc
    rng = np.random.default_rng(7)
    nlines, nmeths = 30, 6
    values = (25. + rng.normal(0., 4., nlines)[None, :] + rng.normal(0., 2., nmeths)[:, None]
              + rng.normal(0., 1., (nmeths, nlines)))
    errors = rng.uniform(0.5, 2., (nmeths, nlines))
    args = (values.ravel(), errors.ravel(), np.tile(np.arange(nlines), nmeths), np.repeat(np.arange(nmeths), nlines))
    tracemalloc.start()
    res = combine.hierarchical(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert res['n'] == 180 and res['n_lines'] == 30 and res['n_methods'] == 6
    assert peak < 100e6
    assert 1. < res['tau_line'] < 8. and 0.5 < res['tau_method'] < 6.
    assert abs(res['median'] - 25.) < 3.


def test_rejects_bad_tau_scale():
    for bad in (0., -1., np.nan):
        with pytest.raises(ValueError):
            combine.hierarchical([1., 2.], [1., 1.], ['a', 'b'], ['m', 'm'], tau_scale=bad)


def test_from_methods_skips_flagged():
    ms_a = [Measurement('ccf', 'K I', 25., 2.), Measurement('ccf', 'Rb I', 27., 2.),
            Measurement('ccf', 'Cs I', 90., 2., flag=1)]
    ms_b = [Measurement('linecentre', 'K I', 26., 3.), Measurement('linecentre', 'Rb I', 29., 3.)]
    methods = {'ccf': MethodResult('ccf', 0, 0, 3, ms_a), 'linecentre': MethodResult('linecentre', 0, 0, 2, ms_b)}
    res = combine.hierarchical_from_methods(methods)
    assert res['n'] == 4 and res['n_lines'] == 2 and res['n_methods'] == 2
    assert res['excluded'] == [('ccf', 'Cs I', 1, 'fit_failed')]
    assert combine.hierarchical_from_methods(methods, exclude_flags=0)['n'] == 5
    # poor_fit (64) is informative only; outlier (128) excludes
    ms_a[2].flag = 64
    assert combine.hierarchical_from_methods(methods)['n'] == 5
    ms_a[2].flag = 128
    assert combine.hierarchical_from_methods(methods)['n'] == 4


# ------------------------------------------------------------------------------------------------------------
# equivalence with the legacy Phase A implementation
# ------------------------------------------------------------------------------------------------------------
def _cases():
    rng = np.random.default_rng(3)
    nl, nm = 8, 3
    big = (26. + rng.normal(0, 6, nl)[None] + rng.normal(0, 2, nm)[:, None] + rng.normal(0, 1, (nm, nl))).ravel()
    return [
        (*_two_methods(LINE_OFFSETS), {}),
        (*_two_methods(LINE_OFFSETS, method_offset=6.), {'tau_max_factor': 100.}),
        (*_two_methods(np.zeros(6)), {'tau_scale': 0.5, 'ngrid': 120}),
        (np.array([30., 28., 35., 31., 25., 36., 22.]), np.array([5., 3., 4., 5., 5., 5., 5.]),
         ['a', 'b', 'c', 'a', 'b', 'c', 'd'], ['lc'] * 3 + ['x'] * 4, {}),
        (np.array([10., 12., 15., 11.]), np.array([1., 2., 3., 1.5]), ['a', 'b', 'a', 'b'], ['m', 'm', 'n', 'n'],
         {'tau_line': 0., 'tau_meth': 0.}),
        (np.array([1., 2., 3.]), np.ones(3), ['a', 'b', 'c'], ['m'] * 3, {}),
        (big, rng.uniform(0.5, 3., nl * nm), np.tile(np.arange(nl), nm), np.repeat(np.arange(nm), nl), {}),
    ]


@pytest.mark.parametrize('case', range(7))
def test_hierarchical_matches_legacy(case):
    legacy = _legacy_functions()
    *args, kwargs = _cases()[case]
    old = legacy['hierarchical_combine'](*args, **kwargs)
    new_kwargs = {('tau_method' if k == 'tau_meth' else k): v for k, v in kwargs.items()}
    new = combine.hierarchical(*args, **new_kwargs)
    for key_old, key_new in [('mean', 'mean'), ('sd', 'sd'), ('median', 'median'), ('q16', 'q16'), ('q84', 'q84'),
                             ('tau_line', 'tau_line'), ('tau_meth', 'tau_method'), ('tau_scale', 'tau_scale')]:
        np.testing.assert_allclose(new[key_new], old[key_old], rtol=0, atol=1e-8, err_msg=key_new)
    np.testing.assert_allclose(new['err'], (old['q84'] - old['q16']) / 2, atol=1e-8)
    np.testing.assert_allclose(new['weights'], old['weights'], atol=1e-12)
    assert (new['n'], new['n_lines'], new['n_methods']) == (old['n'], old['nlines'], old['nmethods'])


def test_weighted_mean_matches_legacy():
    legacy = _legacy_functions()
    rng = np.random.default_rng(5)
    for _ in range(20):
        n = rng.integers(1, 8)
        rvs, errs = rng.normal(25, 8, n), rng.uniform(0.5, 5, n)
        if n > 2:
            rvs[0] = np.nan
        np.testing.assert_allclose(combine.weighted_mean(rvs, errs), legacy['combine_rvs'](rvs, errs), atol=1e-12)


def _brute(values, errors, lines, methods, tau_scale, ngrid=200, tau_max_factor=10.):
    """Independent dense-covariance evaluation of the same model on the same τ grid"""
    values, errors = np.asarray(values, float), np.asarray(errors, float)
    lines, methods = np.asarray(lines).astype(str), np.asarray(methods).astype(str)
    same_l = (lines[:, None] == lines[None]).astype(float)
    same_m = (methods[:, None] == methods[None]).astype(float)
    tau_max = max(tau_max_factor * tau_scale, 3 * np.ptp(values))
    grid = np.concatenate([[0.], np.logspace(np.log10(1e-3 * tau_scale), np.log10(tau_max), ngrid - 1)])
    quad = np.zeros_like(grid)
    quad[:-1] += np.diff(grid) / 2
    quad[1:] += np.diff(grid) / 2
    prior = 2 / (np.pi * tau_scale) / (1 + (grid / tau_scale) ** 2) * quad
    one = np.ones(len(values))
    logw, means, sds = [], [], []
    for i, tl in enumerate(grid):
        for j, tm in enumerate(grid):
            cov = np.diag(errors ** 2) + tl ** 2 * same_l + tm ** 2 * same_m
            ci = np.linalg.inv(cov)
            a, b = one @ ci @ one, one @ ci @ values
            ll = -0.5 * (np.linalg.slogdet(cov)[1] + np.log(a) + values @ ci @ values - b ** 2 / a)
            logw.append(ll + np.log(prior[i]) + np.log(prior[j]))
            means.append(b / a)
            sds.append(1 / np.sqrt(a))
    w = np.exp(np.array(logw) - max(logw))
    w /= w.sum()
    means, sds = np.array(means), np.array(sds)
    mean = np.sum(w * means)
    return mean, np.sqrt(np.sum(w * (sds ** 2 + means ** 2)) - mean ** 2)


def test_hierarchical_matches_dense_brute_force():
    values = np.array([25., 31., 22., 27., 33., 20.])
    errors = np.array([2., 3., 2.5, 2., 3., 2.5])
    lines, methods = ['a', 'b', 'c'] * 2, ['x'] * 3 + ['y'] * 3
    res = combine.hierarchical(values, errors, lines, methods, ngrid=40)
    mean, sd = _brute(values, errors, lines, methods, res['tau_scale'], ngrid=40)
    np.testing.assert_allclose([res['mean'], res['sd']], [mean, sd], atol=1e-9)
