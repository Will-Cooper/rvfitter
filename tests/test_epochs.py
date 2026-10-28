"""Multi-epoch objects: per-epoch labels, honest combination and the variability test."""
from dataclasses import replace

import numpy as np

import synth
from rvfitter import config, pipeline, results


def _cfg(tpl):
    cfg = config.Config()
    cfg.velocity_frame.correction = 'skip'
    cfg.methods.enabled = ['template_fit']
    cfg.methods.options = {'template_fit': {'template': tpl, 'segment_res': 0.}}
    cfg.regions.template_windows = 'full'
    return cfg


def _epoch(rv, seed, name, tpl):
    spec, _ = synth.make_spectrum(rv=rv, resolution=2500., snr=150., rng=np.random.default_rng(seed), template=tpl)
    return replace(spec, source=name)


def test_disagreeing_epochs_are_flagged_and_inflate_the_error():
    tpl = synth.make_template((7500., 9000.), rng=np.random.default_rng(0))
    spectra = [_epoch(10., 1, 'night1.fits', tpl), _epoch(30., 2, 'night2.fits', tpl)]
    res = pipeline.measure_object(spectra, _cfg(tpl), None, np.random.default_rng(0), name='X')
    ep = res.adopted['epochs']
    assert ep['labels'] == ['night1', 'night2'] and ep['variable'] and ep['pvalue'] < 1e-3
    np.testing.assert_allclose(ep['rv'], [10., 30.], atol=3.)
    m = res.methods['template_fit']
    assert {x.region.split(':')[0] for x in m.measurements} == {'night1', 'night2'}
    assert m.rv_err > 5. and res.adopted['err'] > 5.  # not 20 ± 0.7: the epoch spread is in the error
    # two levels: regions -> epoch, then random effects over the epochs (many regions do not shrink the error)
    assert res.adopted['tau_epoch'] > 3. and 'pvalue_stat' in ep and len(ep['sys_wavezp']) == 2
    row = results.to_table([res]).iloc[0]
    assert bool(row['rv_epoch_variable']) and row['rv_epoch_pvalue'] < 1e-3 and row['n_epochs'] == 2
    det = results.details_table([res])
    assert {'diag_epoch', 'diag_epoch_method_rv', 'diag_epoch_adopted_rv'} <= set(det.columns)


def test_agreeing_epochs_are_not_flagged():
    tpl = synth.make_template((7500., 9000.), rng=np.random.default_rng(0))
    spectra = [_epoch(12., 3, 'a.fits', tpl), _epoch(12., 4, 'b.fits', tpl)]
    res = pipeline.measure_object(spectra, _cfg(tpl), None, np.random.default_rng(0), name='Y')
    ep = res.adopted['epochs']
    assert not ep['variable'] and ep['pvalue'] > 1e-3
    assert abs(res.adopted['median'] - 12.) < 3. * res.adopted['err']


def test_epoch_labels_unique():
    a, _ = synth.make_spectrum(rng=np.random.default_rng(1))
    s1, s2 = replace(a, source='x/obj.fits'), replace(a, source='y/obj.fits')
    assert pipeline.epoch_labels([s1, s2]) == ['e1', 'e2']
    s1 = replace(s1, meta={'mjd_mid': 57000.1})
    s2 = replace(s2, meta={'mjd_mid': 57001.2})
    assert pipeline.epoch_labels([s1, s2]) == ['mjd57000.10000', 'mjd57001.20000']


def test_epoch_combination_does_not_shrink_with_regions():
    """Two epochs disagreeing by 20 km/s give an error of order 10 km/s however many regions each has"""
    from rvfitter.types import Measurement, MethodResult

    def epoch(rv, n):
        meas = [Measurement('ccf', f'r{i}', rv + 0.1 * i, 1.) for i in range(n)]
        return MethodResult('ccf', rv, 1. / np.sqrt(n), n, meas)

    for n in (1, 8):
        res = pipeline._combine_epochs('ccf', [epoch(10., n), epoch(30., n)], ['a', 'b'], 'X')
        assert 15. < res.rv < 25. and res.rv_err > 5., (n, res.rv, res.rv_err)
