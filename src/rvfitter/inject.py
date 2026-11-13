"""
``rvfitter inject``: injection–recovery of the configured methods on a degraded grid template.

The template (as for ``forecast``: the configured grid at templates.teff_prior/logg_prior, or --teff/--logg) is
broadened to R and vsini, shifted to each --rv (relativistic, pixel-integrated), optionally multiplied by the
telluric transmission (at rest: observer frame), and given photon noise σ_i = √(A_i A_med)/SNR on the pixels of a
spectrum or a ln λ grid. Each of --n seeds per RV is measured by `pipeline.measure_object` with the user's config
(barycentric correction skipped; wavelength correction "none": the injected wavelengths are exact). Per method
(and the adopted RV), the table gives the bias (mean RV − truth), bias/scatter, and the pull (RV − truth)/σ mean and
width; a calibrated method has |bias| < 0.2 scatter and pull width 0.8–1.2.
"""
import argparse
import copy
import logging
from collections.abc import Sequence

import numpy as np
from astropy.table import Table

from . import forecast, pipeline
from .types import Spectrum

__all__ = ['main', 'inject', 'spectra', 'summarise']

log = logging.getLogger(__name__)


def spectra(template: Spectrum, wave: np.ndarray, resolution: float, snr: float, vsini: float,
            rvs: Sequence[float], n: int, rng: np.random.Generator, *, airmass: float | None = None,
            pwv: float | None = None) -> list[tuple[float, Spectrum]]:
    """(truth, Spectrum) for n noise realisations at each RV; tellurics when `airmass` is given"""
    tpl = forecast.model_spectrum(template, wave, resolution, vsini)
    trans = np.ones(len(wave))
    if airmass is not None:
        from .calibration import telluric
        kw = {} if pwv is None else {'pwv': pwv}
        trans = telluric.transmission(wave, airmass, resolution=resolution, **kw)
    out = []
    for rv in rvs:
        clean = tpl.at(wave, rv) * trans
        if not np.all(np.isfinite(clean)):
            raise ValueError(f'the template does not cover the pixels at {rv} km/s')
        sigma = np.sqrt(np.clip(clean, 0., None) * np.median(clean)) / snr
        for k in range(n):
            meta = {'object': f'inject rv={rv:g} #{k}', 'medium_in': 'vacuum'}
            if airmass is not None:
                meta['airmass'] = airmass
            out.append((float(rv), Spectrum(wave, clean + rng.normal(0., 1., len(wave)) * sigma, sigma, resolution,
                                            meta, 'inject')))
    return out


def summarise(truth: np.ndarray, rv: np.ndarray, err: np.ndarray) -> dict:
    """n, failures, bias [km/s], bias/scatter, pull mean and width, scatter and mean error"""
    ok = np.isfinite(rv) & np.isfinite(err) & (err > 0)
    d, e = rv[ok] - truth[ok], err[ok]
    n = int(ok.sum())
    if n < 2:
        return dict(n=n, n_failed=int((~ok).sum()), bias=np.nan, bias_sigma=np.nan, pull_mean=np.nan,
                    pull_std=np.nan, scatter=np.nan, mean_err=np.nan)
    scatter = float(np.std(d, ddof=1))
    return dict(n=n, n_failed=int((~ok).sum()), bias=float(np.mean(d)), bias_sigma=float(np.mean(d) / scatter),
                pull_mean=float(np.mean(d / e)), pull_std=float(np.std(d / e, ddof=1)), scatter=scatter,
                mean_err=float(np.mean(e)))


def inject(cfg, template: Spectrum, wave: np.ndarray, resolution: float, snr: float, vsini: float,
           rvs: Sequence[float], n: int, seed: int, *, methods: Sequence[str] | None = None,
           airmass: float | None = None, pwv: float | None = None) -> Table:
    """Run the injections and return the per-method summary table (one row per method, then 'adopted')"""
    cfg = copy.deepcopy(cfg)
    cfg.calibration.wavelength_correction = 'none'
    names = pipeline.enabled_methods(cfg, methods)
    noise_ss, method_ss = np.random.SeedSequence(seed).spawn(2)
    trials = spectra(template, wave, resolution, snr, vsini, rvs, n, np.random.default_rng(noise_ss),
                     airmass=airmass, pwv=pwv)
    seeds = method_ss.spawn(len(trials))
    got: dict[str, list[tuple[float, float, float]]] = {m: [] for m in [*names, 'adopted']}
    failures: list[tuple[str, str]] = []
    for k, ((truth, spec), s) in enumerate(zip(trials, seeds, strict=True)):
        res = pipeline.measure_object([spec], cfg, None, np.random.default_rng(s), name=spec.meta['object'],
                                      methods=names, bary='skip', failures=failures)
        for m in names:
            got[m].append((truth, res.methods[m].rv, res.methods[m].rv_err))
        got['adopted'].append((truth, res.adopted.get('median', np.nan), res.adopted.get('err', np.nan)))
        log.info('injection %d/%d (%s): %s', k + 1, len(trials), spec.meta['object'],
                 ', '.join(f'{m} {res.methods[m].rv:.2f} ± {res.methods[m].rv_err:.2f}' for m in names))
    rows = []
    for m, vals in got.items():
        a = np.array(vals, float).reshape(-1, 3)
        rows.append(dict(method=m, **summarise(a[:, 0], a[:, 1], a[:, 2])))
    out = Table(rows=rows)
    for c in ('bias', 'scatter', 'mean_err'):
        out[c].unit = 'km/s'
    out.meta.update(resolution=resolution, snr=snr, vsini=vsini, rvs=list(map(float, rvs)), n=n, seed=seed,
                    template=template.source, tellurics=airmass is not None)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog='rvfitter inject', description='Injection-recovery of the configured methods '
                                'on a degraded grid template: bias and pull width per method.')
    forecast.add_arguments(p)
    p.add_argument('--rv', type=float, nargs='+', default=[0.], help='injected RVs [km/s] (default 0)')
    p.add_argument('--n', type=int, default=20, help='noise realisations per RV (default 20)')
    p.add_argument('--seed', type=int, help='random seed (default: the config\'s)')
    p.add_argument('--methods', help='comma-separated methods (default: the config\'s)')
    p.add_argument('--tellurics', action='store_true', help='multiply by the telluric transmission')
    p.add_argument('--airmass', type=float, default=1.2, help='airmass for --tellurics (default 1.2)')
    p.add_argument('--pwv', type=float, help='precipitable water vapour [mm] for --tellurics (default: the model\'s)')
    p.add_argument('--out', default='inject.csv', help='output table (default: inject.csv)')
    args = p.parse_args(argv)
    if args.n < 2:
        p.error('--n must be at least 2')
    cfg, template, wave, resolution = forecast.setup(args)
    methods = [m for m in args.methods.split(',') if m] if args.methods else None
    out = inject(cfg, template, wave, resolution, args.snr, args.vsini, args.rv, args.n,
                 cfg.seed if args.seed is None else args.seed, methods=methods,
                 airmass=args.airmass if args.tellurics else None, pwv=args.pwv)
    log.info('Injection-recovery at R %.0f, SNR %.0f, vsini %.0f km/s, RVs %s, %d seeds each:', resolution, args.snr,
             args.vsini, ', '.join(f'{v:g}' for v in args.rv), args.n)
    log.info('%-14s %4s %6s %9s %8s %9s %9s %9s %9s', 'method', 'n', 'failed', 'bias', 'bias/sd', 'pull mean',
             'pull sd', 'scatter', 'mean err')
    for r in out:
        log.info('%-14s %4d %6d %9.3f %8.3f %9.3f %9.3f %9.3f %9.3f', r['method'], r['n'], r['n_failed'], r['bias'],
                 r['bias_sigma'], r['pull_mean'], r['pull_std'], r['scatter'], r['mean_err'])
    out.write(args.out, format='ascii.csv', overwrite=True)
    log.info('Wrote %s', args.out)
    return 0
