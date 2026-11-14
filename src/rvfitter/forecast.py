"""
``rvfitter forecast``: photon-noise RV precision (Bouchy et al. 2001, A&A 374, 733) per feature window and per band.

A template from the configured grid (at templates.teff_prior/logg_prior, or --teff/--logg) is broadened to the
given R and vsini and sampled on the pixels of a spectrum, or on a ln λ grid of `PIX_PER_RES` pixels per resolution
element over a wavelength range. Windows come from `regions.select` on that noiseless model, as a measurement would
choose them: the template-method feature windows and the line windows ('feature'), and the telluric-free spans
('band'). σ_RV = c/√ΣW_i with W_i = (λ_i dA_i/dλ)² / var_i (`regions.rv_precision`; SNR per pixel at the median
flux, photon noise). The 'total' row combines the bands. Recommended: the best feature windows that together hold
`RECOMMEND_FRACTION` of the features' summed information (1/σ²).
"""
import argparse
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table

from . import broadening, config, regions
from .constants import C_KMS
from .methods import _common as C
from .types import Spectrum

__all__ = ['main', 'template_for', 'wave_grid', 'model_spectrum', 'table', 'add_arguments', 'setup']

log = logging.getLogger(__name__)

PIX_PER_RES = 2.5
RECOMMEND_FRACTION = 0.9
PAD_KMS = 1500.


def template_for(cfg: Any, wave_range: tuple[float, float], teff: float | None = None,
                 logg: float | None = None) -> Spectrum:
    """The configured grid's rest-frame template at (teff, logg) [default: the config's priors], covering wave_range"""
    teff = cfg.templates.teff_prior if teff is None else teff
    if teff is None:
        raise ValueError('no template temperature: set templates.teff_prior in the config or give --teff')
    logg = (cfg.templates.logg_prior or 5.) if logg is None else logg
    pad = broadening.doppler_factor(PAD_KMS) * 1.01
    grid = C.open_grid(cfg.templates)
    return grid.get(teff, logg, cfg.templates.met_prior, wave_range=(wave_range[0] / pad, wave_range[1] * pad))


def wave_grid(wmin: float, wmax: float, resolution: float) -> np.ndarray:
    """ln λ grid of PIX_PER_RES pixels per resolution element"""
    return broadening.log_wave_grid(wmin, wmax, C_KMS / (PIX_PER_RES * resolution))


def model_spectrum(template: Spectrum, wave: np.ndarray, resolution: float, vsini: float = 0.) -> C.Template:
    """The template broadened to R and vsini (a `Template`: `.at(wave, v)` is the pixel-integrated flux)"""
    return C.template_from_spectrum(template, resolution, (float(wave[0]), float(wave[-1])), vsini=vsini,
                                    pad_kms=PAD_KMS)


def table(template: Spectrum, wave: np.ndarray, resolution: float, snr: float, vsini: float = 0., *,
          teff: float | None = None, telluric_corrected: bool = False) -> Table:
    """σ_RV [km/s] per feature window, per band and in total, with the recommended windows (see the module doc)"""
    flux = model_spectrum(template, wave, resolution, vsini).at(wave)
    spec = Spectrum(wave, flux, np.full(len(wave), np.nan), resolution, {'object': 'forecast'}, 'forecast')
    opts = regions.SelectOptions(telluric_corrected=telluric_corrected) if telluric_corrected else None
    sel = regions.select(spec, teff=teff, methods=['template_fit', 'linecentre', 'lsd'], options=opts)
    rows, seen = [], set()
    for kind, method in (('band', 'lsd'), ('feature', 'template_fit'), ('feature', 'linecentre')):
        for r in sel[method]:
            if (r.wmin, r.wmax) in seen:
                continue
            seen.add((r.wmin, r.wmax))
            m = (wave >= r.wmin) & (wave <= r.wmax)
            rows.append(dict(kind=kind, name=r.name, wmin=r.wmin, wmax=r.wmax,
                             sigma_rv=regions.rv_precision(wave[m], flux[m], snr)))
    if not rows:
        raise ValueError('no usable windows in the coverage')
    bands = np.array([r['sigma_rv'] for r in rows if r['kind'] == 'band'])
    rows.append(dict(kind='total', name='all bands', wmin=float(wave[0]), wmax=float(wave[-1]),
                     sigma_rv=float(1. / np.sqrt(np.sum(bands ** -2.))) if len(bands) else math.inf))
    out = Table(rows=rows, names=('kind', 'name', 'wmin', 'wmax', 'sigma_rv'))
    out['recommended'] = False
    feat = [i for i in np.argsort(out['sigma_rv']) if out['kind'][i] == 'feature' and np.isfinite(out['sigma_rv'][i])]
    info = np.array([out['sigma_rv'][i] ** -2. for i in feat])
    if len(info):
        n = int(np.searchsorted(np.cumsum(info), RECOMMEND_FRACTION * info.sum())) + 1
        out['recommended'][feat[:n]] = True
    out['wmin'].unit = out['wmax'].unit = 'Angstrom'
    out['sigma_rv'].unit = 'km/s'
    out.meta.update(resolution=resolution, snr=snr, vsini=vsini, template=template.source)
    return out


def add_arguments(p: argparse.ArgumentParser) -> None:
    """The template, resolution, SNR, vsini and coverage arguments shared by forecast and inject"""
    p.add_argument('--config', help='TOML configuration file (templates; default: the default config)')
    p.add_argument('--teff', type=float, help='template Teff [K] (default: templates.teff_prior)')
    p.add_argument('--logg', type=float, help='template log g [dex] (default: templates.logg_prior, else 5)')
    p.add_argument('--resolution', type=float, help='resolving power (default: the spectrum\'s, with --spectrum)')
    p.add_argument('--snr', type=float, required=True, help='signal-to-noise ratio per pixel at the median flux')
    p.add_argument('--vsini', type=float, default=0., help='projected rotation [km/s] (default 0)')
    cov = p.add_mutually_exclusive_group(required=True)
    cov.add_argument('--wave-range', type=float, nargs=2, metavar=('WMIN', 'WMAX'), help='vacuum wavelength range [Å]')
    cov.add_argument('--spectrum', help='use this spectrum\'s pixels (and resolution, unless --resolution)')


def setup(args: argparse.Namespace) -> tuple[Any, Spectrum, np.ndarray, float]:
    """(config, rest-frame template, pixel wavelengths, R) from the shared arguments"""
    cfg = config.load(args.config) if args.config else config.Config()
    if args.spectrum:
        from . import io
        spec = io.read_spectrum(args.spectrum, medium=cfg.input.medium, medium_method=cfg.input.medium_method)
        wave = spec.wave
        resolution = args.resolution or cfg.input.resolution or spec.resolution
    else:
        resolution = args.resolution or cfg.input.resolution
        if not resolution:
            raise ValueError('give --resolution (or input.resolution in the config)')
        wave = wave_grid(*args.wave_range, resolution)
    if not resolution:
        raise ValueError(f'{args.spectrum} has no resolution: give --resolution')
    template = template_for(cfg, (float(wave[0]), float(wave[-1])), args.teff, args.logg)
    return cfg, template, wave, float(resolution)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog='rvfitter forecast', description='Forecast the photon-noise RV precision '
                                '(Bouchy et al. 2001) per feature window and per band.')
    add_arguments(p)
    p.add_argument('--out', default='forecast.csv', help='output table (default: forecast.csv)')
    args = p.parse_args(argv)
    cfg, template, wave, resolution = setup(args)
    out = table(template, wave, resolution, args.snr, args.vsini, teff=args.teff or cfg.templates.teff_prior,
                telluric_corrected=cfg.input.telluric_corrected)
    log.info('Forecast σ_RV at R %.0f, SNR %.0f, vsini %.0f km/s (template %s):', resolution, args.snr, args.vsini,
             Path(template.source).name)
    log.info('%-8s %-32s %9s %9s %10s %s', 'kind', 'name', 'wmin [Å]', 'wmax [Å]', 'σ_RV [km/s]', 'recommended')
    for r in out:
        log.info('%-8s %-32s %9.1f %9.1f %10.2f %s', r['kind'], r['name'], r['wmin'], r['wmax'], r['sigma_rv'],
                 'yes' if r['recommended'] else '')
    out.write(args.out, format='ascii.csv', overwrite=True)
    log.info('Wrote %s', args.out)
    return 0
