"""
Systematic error terms, kept separate from the statistical errors (deep-research-report.md: statistical,
template-spread and wavelength zero-point terms are reported separately, plus their quadrature total).

- `template_spread`: the RV spread over plausible template choices: every grid candidate within the prior,
  each enabled grid, and the other template-window mode (features vs full spans), measured with a template
  method (template_fit by default; it is fast). The components are logged and returned by
  `template_spread_details`.
- `wavelength_zero_point`: the velocity equivalent of the wavelength-solution RMS, if known.
"""
import logging
from collections.abc import Sequence
from concurrent.futures import Executor
from typing import Any

import numpy as np

from .constants import C_KMS
from .types import Region, Spectrum

__all__ = ['template_spread', 'template_spread_details', 'wavelength_zero_point']

log = logging.getLogger(__name__)


def _cfg_get(cfg: Any, section: str, key: str, default=None):
    sec = cfg.get(section, {}) if isinstance(cfg, dict) else getattr(cfg, section, None)
    if sec is None:
        return default
    val = sec.get(key, default) if isinstance(sec, dict) else getattr(sec, key, default)
    return default if val is None else val


def _with(cfg: Any, section: str, **values) -> Any:
    """A copy of cfg with some fields of one section replaced (dataclass Config or mapping)"""
    import copy
    new = copy.deepcopy(cfg)
    sec = new.setdefault(section, {}) if isinstance(new, dict) else getattr(new, section)
    for k, v in values.items():
        if isinstance(sec, dict):
            sec[k] = v
        else:
            setattr(sec, k, v)
    return new


def template_spread_details(spectrum: Spectrum, regions: dict[str, list[Region]] | list[Region], cfg: Any, *,
                            executor: Executor | None = None, rng: np.random.Generator,
                            method: str = 'template_fit') -> dict:
    """
    RVs of one template method under every plausible template choice, and their spread

    Parameters
    ----------
    spectrum
        The (barycentric-corrected) spectrum
    regions
        The regions selected for the object ({method: [Region]} or the method's list)
    cfg
        Config (or mapping)
    executor, rng
        As for the methods
    method
        The template method used (default template_fit)

    Returns
    -------
    details
        'templates': {label: rv} for each grid candidate (pinned) in the configured window mode;
        'grids': {grid: rv} for each enabled grid; 'window_modes': {mode: rv} for 'features' and 'full';
        'sd_templates', 'sd_grids', 'delta_modes' and 'total' [km/s], where total is the standard deviation
        (ddof = 1) of every RV above; NaN when fewer than two variants exist
    """
    from . import regions as _regions
    from .methods import REGISTRY
    from .methods import _common as C
    measure = REGISTRY[method]
    regs = regions.get(method, []) if isinstance(regions, dict) else list(regions)
    out: dict[str, Any] = dict(templates={}, grids={}, window_modes={}, sd_templates=np.nan, sd_grids=np.nan,
                               delta_modes=np.nan, total=np.nan)
    if not regs:
        return out
    seeds = rng.spawn(3)
    grids = _cfg_get(cfg, 'templates', 'grid', 'bt-settl-cifist')
    grids = list(grids) if isinstance(grids, (list, tuple)) else [grids]

    def run(c, rgs, seed):
        try:
            return float(measure(spectrum, rgs, c, executor=executor, rng=np.random.default_rng(seed)).rv)
        except Exception as exc:  # noqa: BLE001 - a variant that cannot run is logged, not fatal
            log.debug('template spread variant failed: %s', exc)
            return np.nan

    # (1) every candidate of every configured grid, pinned (grid and all axes)
    tcfg = C.section(cfg, 'templates')
    cands = C.candidates(cfg, {'teff_window': 300.})
    for k, cand in enumerate(cands):
        if cand.params is None:
            continue
        p = cand.param_dict
        std = {f'{a}_range': (p[a], p[a]) for a in ('teff', 'logg', 'met', 'alpha')}
        pinned = _with(cfg, 'templates', grid=cand.grid.name, **std,
                       ranges={a: (v, v) for a, v in p.items() if a not in ('teff', 'logg', 'met', 'alpha')})
        out['templates'][cand.label] = run(pinned, regs, seeds[0].spawn(len(cands))[k])
    # (2) each enabled grid (when several are configured)
    if len(grids) > 1:
        for j, grid in enumerate(grids):
            out['grids'][str(grid)] = run(_with(cfg, 'templates', grid=grid), regs, seeds[1].spawn(len(grids))[j])
    # (3) the other window mode
    mode = _cfg_get(cfg, 'regions', 'template_windows', 'features')
    other = 'full' if mode == 'features' else 'features'
    out['window_modes'][mode] = run(cfg, regs, seeds[2])
    try:
        sel = _regions.select(spectrum, teff=tcfg.get('teff_prior'), methods=[method],
                              options=_regions.SelectOptions(
                                  telluric_corrected=bool(_cfg_get(cfg, 'input', 'telluric_corrected', False)),
                                  template_windows=other))
        out['window_modes'][other] = run(cfg, sel.get(method, []), seeds[2])
    except Exception as exc:  # noqa: BLE001
        log.debug('template spread: %s windows unavailable: %s', other, exc)

    def sd(vals):
        vals = np.array([v for v in vals if np.isfinite(v)])
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan

    out['sd_templates'] = sd(out['templates'].values())
    out['sd_grids'] = sd(out['grids'].values())
    modes = list(out['window_modes'].values())
    out['delta_modes'] = float(modes[1] - modes[0]) if len(modes) == 2 else np.nan
    out['total'] = sd([*out['templates'].values(), *out['grids'].values(), *modes])
    log.info('template spread (%s): %.2f km/s; templates SD %.2f km/s over %d, grids SD %.2f km/s, '
             '%s − %s windows %.2f km/s', method, out['total'], out['sd_templates'], len(out['templates']),
             out['sd_grids'], other, mode, out['delta_modes'])
    return out


def template_spread(spectrum: Spectrum, regions: dict[str, list[Region]] | list[Region], cfg: Any, *,
                    executor: Executor | None = None, rng: np.random.Generator) -> float:
    """
    The template-spread systematic [km/s]: the 'total' of `template_spread_details`

    Returns
    -------
    spread
        km/s, NaN if fewer than two template variants could be measured
    """
    return float(template_spread_details(spectrum, regions, cfg, executor=executor, rng=rng)['total'])


def wavelength_zero_point(spectrum: Spectrum, cfg: Any, *, wave: np.ndarray | Sequence[float] | None = None
                          ) -> float:
    """
    The wavelength zero-point systematic [km/s]: the larger of the arc-RMS term and the zero-point term

    - Arc RMS: c × rms / λ_mid, with the RMS [Å] from cfg.input.wave_rms or spectrum.meta['wave_rms']. This is a
      lower bound: an internally consistent arc fit can hide a distortion of the whole solution (J1745: arc RMS
      ≈3 km/s against a +10 to +25 km/s sky-line distortion).
    - Zero point, evaluated at the measurements' wavelengths `wave` (the RMS over them; the curves vary by tens
      of km/s across a spectrum, so a single mean offset would misstate it):
      - a correction applied (meta['wavelength_correction']): its own uncertainty, i.e. the sky curve's error
        σ_sky(λ) with the spread over reasonable curve models (`zp_sky_model_spread`) in quadrature; or the
        telluric curve's error σ_tell(λ). The error of an applied star − sky constant is `sys_star_sky`;
      - nothing applied: the measured curve's uncertainty σ(λ) only (sky, else telluric). A measured offset's value
        is never counted as an error: it is either applied or unknown (the pipeline warns when a significant
        offset is left unapplied, and the star's own constant gets the slit-centring prior, `sys_star_sky`).
      Without `wave`, the curves' summary errors (`zp_sky_err`, `zp_tell_err`) are used.

    Returns
    -------
    sys
        km/s; NaN if neither is known
    """
    from .calibration.sky_zeropoint import curve_at
    meta = spectrum.meta if hasattr(spectrum.meta, 'get') else {}
    rms = _cfg_get(cfg, 'input', 'wave_rms', None)
    if rms is None:
        rms = meta.get('wave_rms')
    arc = float(C_KMS * rms / np.median(spectrum.wave)) if rms is not None and np.isfinite(rms) else np.nan
    lam = None if wave is None else np.asarray(wave, dtype=float)
    if lam is not None:
        lam = lam[np.isfinite(lam)]
        lam = lam if len(lam) else None

    def rms_of(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return float(np.sqrt(np.mean(x * x))) if len(x) else np.nan

    def observatory(curve: dict) -> np.ndarray:  # the curves are in the observatory frame
        from .calibration.telluric_zeropoint import frame_factor
        return lam / frame_factor(meta)

    def spread_of(curve: dict) -> float:
        v = curve.get('zp_sky_model_spread', np.nan)
        return float(v) if v is not None and np.isfinite(v) else 0.

    zp = np.nan
    wc = meta.get('wavelength_correction') or {}
    if wc.get('sky'):  # applied: its uncertainty remains (a star − sky constant's error is sys_star_sky)
        c = wc['sky']
        zp = (float(np.hypot(rms_of(curve_at(c, observatory(c))[1]), spread_of(c))) if lam is not None
              else float(c.get('zp_sky_err', np.nan)))
    elif wc.get('telluric'):  # the star's telluric curve applied
        c = wc['telluric']
        zp = (rms_of(curve_at(c, observatory(c))[1]) if lam is not None
              else float(c.get('zp_tell_err', np.nan)))
    else:  # measured but not applied: only its uncertainty (a measured offset's value is never an error)
        c = meta.get('sky_zeropoint') or meta.get('telluric_zeropoint')
        if c:
            sky = bool(meta.get('sky_zeropoint'))
            if lam is not None:
                zp = float(np.hypot(rms_of(curve_at(c, observatory(c))[1]), spread_of(c) if sky else 0.))
            else:
                zp = float(c.get('zp_sky_err' if sky else 'zp_tell_err', np.nan))
    vals = [v for v in (arc, zp) if np.isfinite(v)]
    return float(max(vals)) if vals else np.nan
