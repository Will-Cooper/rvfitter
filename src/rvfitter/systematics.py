"""
Systematic error terms, kept separate from the statistical errors: the statistical,
template-spread and wavelength zero-point terms are reported separately, plus their quadrature total.

- `template_spread_details`: the RV spread over plausible template choices (one Teff grid step either side of
  the best-fitting candidate, and each enabled grid), measured with a template method (template_fit by
  default; it is fast), and the other template-window mode's RV difference as a diagnostic.
- `wavelength_zero_point`: the uncertainty of the wavelength zero point at the measurements' wavelengths.
"""
import copy
import dataclasses
import logging
from collections.abc import Sequence
from concurrent.futures import Executor
from typing import Any

import numpy as np

from . import regions as _regions
from .calibration.sky_zeropoint import curve_at
from .calibration.telluric_zeropoint import frame_factor
from .calibration.wavelength_correction import arc_kms
from .logging_setup import demoted
from .methods import REGISTRY
from .methods import _common as C
from .methods.guidance import TEMPLATE_COLUMNS, grid_names
from .numerics import rms
from .types import Region, Spectrum

__all__ = ['template_spread_details', 'wavelength_zero_point']

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


def _chi2(res) -> float:
    """A template method's χ² summed over its windows (NaN if a window failed)"""
    return float(sum(m.diagnostics['chi2'] for m in res.measurements))


def template_spread_details(spectrum: Spectrum, regions: dict[str, list[Region]], cfg: Any, *,
                            executor: Executor | None = None, rng: np.random.Generator,
                            method: str = 'template_fit') -> dict:
    """
    RVs of one template method under every plausible template choice, and their spread

    Returns 'templates' {label: rv} for each grid candidate (pinned) in the configured window mode, 'teff_steps'
    {label: rv} for the best-fitting candidate (least χ²) and the grid points one Teff step below and above it
    at its other parameters (one at a grid edge; measured if not a candidate), 'grids' {grid: rv} for each
    enabled grid (when several), 'window_modes' {mode: rv} for 'features' and 'full', and [km/s]
    'sd_templates' (the RMS of the Teff steps' RV − the best fit's RV), 'sd_grids' (the SD of the grids' RVs;
    NaN with one grid, which has no grid choice), 'delta_modes' (full − features RV) and 'total', the
    standard deviation (ddof = 1) of the 'teff_steps' and 'grids' RVs (sys_template; NaN with fewer than two).
    The window modes are a diagnostic only: below R 4500 'features' is used because the grids model the
    molecular bands poorly, so a 'full' fit that disagrees is a model failure, not an uncertainty of the
    features fit (from R 4500 up both are the full spans). A variant that cannot be measured (ValueError) is
    NaN with a WARNING.
    """
    measure = REGISTRY[method]
    regs = regions.get(method, [])
    out: dict[str, Any] = dict(templates={}, teff_steps={}, grids={}, window_modes={}, sd_templates=np.nan,
                               sd_grids=np.nan, delta_modes=np.nan, total=np.nan)
    if not regs:
        return out
    seeds = rng.spawn(3)

    def run(c, rgs, seed):  # the variants' logs repeat the main run's: DEBUG
        try:
            with demoted():
                return measure(spectrum, rgs, c, executor=executor, rng=np.random.default_rng(seed))
        except ValueError as exc:
            log.warning('template spread: a variant could not be measured (%s); it is left out', exc)
            return None

    def rv(res) -> float:
        return np.nan if res is None else float(res.rv)

    try:
        cands = C.candidates(cfg, {'teff_window': 300.})
    except ValueError as exc:
        log.warning('template spread not measured (sys_template is NaN): %s', exc)
        return out
    # (1) every candidate of every configured grid, pinned (grid and all axes); the Teff term is the RV change
    # for one Teff grid step either side of the best fit, so templates far from it never enter
    def pin(grid, p: dict) -> Any:
        std = {f'{a}_range': (p[a], p[a]) for a in TEMPLATE_COLUMNS}
        return _pinned(cfg, grid=grid.name, **std,
                       ranges={a: (v, v) for a, v in p.items() if a not in TEMPLATE_COLUMNS})

    def key(grid, p: dict) -> tuple:
        return grid.name, tuple(float(p[a]) for a in grid.axes)

    measured = {}  # key: (label, rv)
    best, best_chi2 = None, np.inf
    for cand, seed in zip(cands, seeds[0].spawn(len(cands)), strict=True):
        if cand.params is None:
            continue
        res = run(pin(cand.grid, cand.param_dict), regs, seed)
        out['templates'][cand.label] = measured[key(cand.grid, cand.param_dict)] = rv(res)
        if np.isfinite(rv(res)) and (c := _chi2(res)) < best_chi2:
            best, best_chi2 = cand, c
    if best is not None:
        grid, p = best.grid, best.param_dict
        it = grid.axes.index('teff')
        same = grid.points[np.all(np.delete(grid.points, it, 1) == [p[a] for a in grid.axes if a != 'teff'], 1)]
        teffs = same[:, it]
        lo, hi = teffs[teffs < p['teff']].max(initial=-np.inf), teffs[teffs > p['teff']].min(initial=np.inf)
        steps = [t for t in (lo, hi) if np.isfinite(t)]
        out['teff_steps'][best.label] = measured[key(grid, p)]
        for t, seed in zip(steps, seeds[0].spawn(len(steps)), strict=True):
            q = dict(p, teff=float(t))
            k = key(grid, q)
            if k not in measured:
                measured[k] = rv(run(pin(grid, q), regs, seed))
            out['teff_steps'][f'{grid.name} ' + '/'.join(f'{v:g}' for v in k[1])] = measured[k]
        dv = np.array(list(out['teff_steps'].values())[1:]) - out['teff_steps'][best.label]
        dv = dv[np.isfinite(dv)]
        out['sd_templates'] = float(np.sqrt(np.mean(dv ** 2))) if len(dv) else np.nan
    # (2) each enabled grid (when several are configured)
    grids = grid_names(cfg.templates)
    if len(grids) > 1:
        for grid, seed in zip(grids, seeds[1].spawn(len(grids)), strict=True):
            out['grids'][grid] = rv(run(_pinned(cfg, grid=grid), regs, seed))
    # (3) the other window mode
    mode = cfg.regions.template_windows
    other = 'full' if mode == 'features' else 'features'
    out['window_modes'][mode] = rv(run(cfg, regs, seeds[2]))
    with demoted():
        sel = _regions.select(spectrum, teff=cfg.templates.teff_prior, methods=[method],
                              options=dataclasses.replace(_regions.options_for(cfg), template_windows=other))
    out['window_modes'][other] = rv(run(cfg, sel.get(method, []), seeds[2]))
    out['sd_grids'] = _sd(out['grids'].values())
    out['delta_modes'] = float(out['window_modes']['full'] - out['window_modes']['features'])
    out['total'] = _sd([*out['teff_steps'].values(), *out['grids'].values()])
    log.info('template spread (%s): %.2f km/s; one Teff step %.2f km/s about %s (of %d candidates), grids SD '
             '%.2f km/s; full − features windows %.2f km/s (diagnostic, not in the spread)', method, out['total'],
             out['sd_templates'], best.label if best is not None else '-', len(out['templates']), out['sd_grids'],
             out['delta_modes'])
    return out


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
