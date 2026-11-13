"""
Systematic error terms, kept separate from the statistical errors (deep-research-report.md: statistical,
template-spread and wavelength zero-point terms are reported separately, plus their quadrature total).

- `template_spread`: the RV spread over plausible template choices: every grid candidate within the prior,
  each enabled grid, and the other template-window mode (features vs full spans), measured with a template
  method (template_fit by default; it is fast). The components are logged and returned by
  `template_spread_details`.
- `wavelength_zero_point`: the uncertainty of the wavelength zero point at the measurements' wavelengths.
"""
import copy
import logging
from collections.abc import Sequence
from concurrent.futures import Executor
from typing import Any

import numpy as np

from . import regions as _regions
from .calibration.sky_zeropoint import curve_at
from .calibration.telluric_zeropoint import frame_factor
from .calibration.wavelength_correction import arc_kms
from .methods import REGISTRY
from .methods import _common as C
from .numerics import rms
from .types import Region, Spectrum

__all__ = ['template_spread', 'template_spread_details', 'wavelength_zero_point']

log = logging.getLogger(__name__)


def _pinned(cfg: Any, **templates) -> Any:
    """A copy of cfg with some ``templates`` fields replaced"""
    new = copy.deepcopy(cfg)
    for k, v in templates.items():
        setattr(new.templates, k, v)
    return new


def _sd(vals) -> float:
    v = np.array([x for x in vals if np.isfinite(x)])
    return float(np.std(v, ddof=1)) if len(v) > 1 else np.nan


def template_spread_details(spectrum: Spectrum, regions: dict[str, list[Region]], cfg: Any, *,
                            executor: Executor | None = None, rng: np.random.Generator,
                            method: str = 'template_fit') -> dict:
    """
    RVs of one template method under every plausible template choice, and their spread

    Returns 'templates' {label: rv} for each grid candidate (pinned) in the configured window mode, 'grids'
    {grid: rv} for each enabled grid (when several), 'window_modes' {mode: rv} for 'features' and 'full', and
    'sd_templates', 'sd_grids', 'delta_modes' and 'total' [km/s], the standard deviation (ddof = 1) of every RV
    above. A variant that cannot be measured (ValueError) is NaN with a WARNING; NaN with fewer than two variants.
    """
    measure = REGISTRY[method]
    regs = regions.get(method, [])
    out: dict[str, Any] = dict(templates={}, grids={}, window_modes={}, sd_templates=np.nan, sd_grids=np.nan,
                               delta_modes=np.nan, total=np.nan)
    if not regs:
        return out
    seeds = rng.spawn(3)

    def run(c, rgs, seed) -> float:
        try:
            return float(measure(spectrum, rgs, c, executor=executor, rng=np.random.default_rng(seed)).rv)
        except ValueError as exc:
            log.warning('template spread: a variant could not be measured (%s); it is left out', exc)
            return np.nan

    try:
        cands = C.candidates(cfg, {'teff_window': 300.})
    except ValueError as exc:
        log.warning('template spread not measured (sys_template is NaN): %s', exc)
        return out
    # (1) every candidate of every configured grid, pinned (grid and all axes)
    cand_seeds = seeds[0].spawn(len(cands))
    for cand, seed in zip(cands, cand_seeds, strict=True):
        if cand.params is None:
            continue
        p = cand.param_dict
        std = {f'{a}_range': (p[a], p[a]) for a in C.TEMPLATE_COLUMNS}
        pinned = _pinned(cfg, grid=cand.grid.name, **std,
                         ranges={a: (v, v) for a, v in p.items() if a not in C.TEMPLATE_COLUMNS})
        out['templates'][cand.label] = run(pinned, regs, seed)
    # (2) each enabled grid (when several are configured)
    grids = C.grid_names(C.section(cfg, 'templates'))
    if len(grids) > 1:
        for grid, seed in zip(grids, seeds[1].spawn(len(grids)), strict=True):
            out['grids'][grid] = run(_pinned(cfg, grid=grid), regs, seed)
    # (3) the other window mode
    mode = cfg.regions.template_windows
    other = 'full' if mode == 'features' else 'features'
    out['window_modes'][mode] = run(cfg, regs, seeds[2])
    sel = _regions.select(spectrum, teff=cfg.templates.teff_prior, methods=[method],
                          options=_regions.SelectOptions(telluric_corrected=bool(cfg.input.telluric_corrected),
                                                         template_windows=other))
    out['window_modes'][other] = run(cfg, sel.get(method, []), seeds[2])
    out['sd_templates'] = _sd(out['templates'].values())
    out['sd_grids'] = _sd(out['grids'].values())
    modes = list(out['window_modes'].values())
    out['delta_modes'] = float(modes[1] - modes[0])
    out['total'] = _sd([*out['templates'].values(), *out['grids'].values(), *modes])
    log.info('template spread (%s): %.2f km/s; templates SD %.2f km/s over %d, grids SD %.2f km/s, '
             '%s − %s windows %.2f km/s', method, out['total'], out['sd_templates'], len(out['templates']),
             out['sd_grids'], other, mode, out['delta_modes'])
    return out


def template_spread(spectrum: Spectrum, regions: dict[str, list[Region]], cfg: Any, *,
                    executor: Executor | None = None, rng: np.random.Generator) -> float:
    """The template-spread systematic [km/s]: the 'total' of `template_spread_details`"""
    return float(template_spread_details(spectrum, regions, cfg, executor=executor, rng=rng)['total'])


def wavelength_zero_point(spectrum: Spectrum, *, wave: np.ndarray | Sequence[float] | None = None) -> float:
    """
    The wavelength zero-point systematic [km/s]: the larger of the arc-RMS term and the zero-point term

    - Arc RMS (`wavelength_correction.arc_kms`): a lower bound, since an internally consistent arc fit can hide a
      distortion of the whole solution (J1745: arc RMS ≈3 km/s against a +10 to +25 km/s sky-line distortion).
    - Zero point: the RMS, over the measurements' wavelengths `wave`, of the uncertainty σ(λ) of the applied curve
      (meta['wavelength_correction']), else of the measured but unapplied one (sky, else telluric); for a sky curve
      the model spread is added in quadrature. A measured offset's value is never an error (it is applied, or the
      slit-centring prior covers it). Without `wave`, the curve's summary error. NaN if neither term is known.
    """
    meta = spectrum.meta
    arc = arc_kms(spectrum)
    lam = None if wave is None else np.asarray(wave, dtype=float)
    if lam is not None:
        lam = lam[np.isfinite(lam)]
        lam = lam if len(lam) else None
    wc = meta.get('wavelength_correction') or {}
    if wc.get('sky') or wc.get('telluric'):  # applied (a star − sky constant's error is sys_star_sky)
        sky = bool(wc.get('sky'))
        curve = wc['sky' if sky else 'telluric']
    else:  # measured but not applied
        sky = bool(meta.get('sky_zeropoint'))
        curve = meta.get('sky_zeropoint') or meta.get('telluric_zeropoint')
    zp = np.nan
    if curve and lam is None:
        zp = float(curve.get('zp_sky_err' if sky else 'zp_tell_err', np.nan))
    elif curve:
        spread = curve.get('zp_sky_model_spread', np.nan) if sky else 0.
        spread = float(spread) if spread is not None and np.isfinite(spread) else 0.
        zp = float(np.hypot(rms(curve_at(curve, lam / frame_factor(meta))[1]), spread))  # observatory frame
    vals = [v for v in (arc, zp) if np.isfinite(v)]
    return float(max(vals)) if vals else np.nan
