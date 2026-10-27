"""
Seeded injection–recovery helper for the method tests.

Each trial draws a synthetic spectrum (tests/synth.py) with a known RV and noise, measures it, and the
pull distribution (RV − truth)/σ_RV is summarised. Pass criteria (plan): |bias| < 0.2σ and pull width 0.8–1.2.
"""
import numpy as np

import synth
from rvfitter.types import Region


def full_region(spec, margin=20.):
    """One window covering the spectrum minus a margin [Å] (no typed wavelengths)"""
    return [Region('all', 'ccf', spec.wmin + margin, spec.wmax - margin, reason='test')]


def run(measure, opts_for, *, n, rv=23.4, resolution=2500., snr=30., vsini=0., seed=0,
        wave_range=(7500., 9000.), method='method', regions=full_region, **synth_kw):
    """
    Run n trials and return (rvs, errs, summary)

    Parameters
    ----------
    measure
        The method's measure function
    opts_for
        Function template → method options dict
    """
    tpl = synth.make_template(wave_range, rng=np.random.default_rng(0))
    ss = np.random.SeedSequence(seed)
    rvs, errs = [], []
    for child in ss.spawn(n):
        noise_seed, method_seed = child.spawn(2)
        spec, _ = synth.make_spectrum(rv=rv, resolution=resolution, snr=snr, vsini=vsini, wave_range=wave_range,
                                      rng=np.random.default_rng(noise_seed), template=tpl, **synth_kw)
        cfg = {'methods': {'options': {method: opts_for(tpl)}}}
        res = measure(spec, regions(spec), cfg, rng=np.random.default_rng(method_seed))
        rvs.append(res.rv)
        errs.append(res.rv_err)
    rvs, errs = np.array(rvs), np.array(errs)
    pull = (rvs - rv) / errs
    scatter = np.std(rvs, ddof=1)
    summary = dict(bias_sigma=(np.mean(rvs) - rv) / scatter, pull_std=np.std(pull, ddof=1),
                   pull_mean=np.mean(pull), scatter=scatter, mean_err=np.mean(errs))
    return rvs, errs, summary


def check(summary):
    assert abs(summary['bias_sigma']) < 0.2, summary
    assert 0.8 < summary['pull_std'] < 1.2, summary
