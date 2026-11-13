"""The best template's parameters and, at an edge of the searched parameters, guidance for a refit."""
import logging
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from .. import templates as T

__all__ = ['TEMPLATE_COLUMNS', 'template_summary', 'grid_names']

log = logging.getLogger(__name__)

TEMPLATE_COLUMNS = ('teff', 'logg', 'met', 'alpha')


def grid_names(tcfg: Any) -> list[str]:
    """The configured grid names (``templates.grid`` may be one name or a list)"""
    return [tcfg.grid] if isinstance(tcfg.grid, str) else [str(x) for x in tcfg.grid]


def _encoded_axes(grid) -> set[str]:
    """Axes a grid really has: encoded in its file names, or taking more than one value"""
    named = set(re.compile(grid.meta.pattern).groupindex)
    return {a for a in grid.axes if a in named or len(grid.axis_values(a)) > 1}


def template_summary(method: str, cands: Sequence, scores: Sequence[float], best: Any, dof: int,
                     tcfg: Any = None) -> dict:
    """
    The best template, its χ², χ²-weighted parameter estimates and any parameter edge, with guidance

    Weights over the candidates of the best grid are exp(−Δχ²/(2s)), with s = max(1, χ²_min/ν), so that a fit
    dominated by model mismatch does not give unrealistically sharp parameters. If the best candidate is at the
    low or high end of a searched axis (one with at least two values), the edge is reported with guidance: widen
    `templates.<axis>_range` if the grid extends further, otherwise a known grid that covers the region (download
    command and configuration patch, also as data for a GUI). ``tcfg`` (the TemplatesConfig) gives the grids to
    keep in such a patch; None keeps the best grid only.

    Returns 'template_grid', 'template_label', 'template_<axis>' for teff, logg, met and alpha (NaN where the grid
    has no such axis), 'template_chi2', 'template_at_edge', 'estimate' {axis: (mean, sd)} and 'edges' [dict].
    """
    scores = np.asarray(scores, dtype=float)
    ibest = cands.index(best)
    out: dict[str, Any] = dict(template_grid=best.grid.name if best.grid is not None else 'user',
                               template_label=best.label,
                               template_chi2=float(scores[ibest]) if len(scores) > ibest else np.nan,
                               estimate={}, edges=[])
    for a in TEMPLATE_COLUMNS:
        out[f'template_{a}'] = np.nan
    if best.grid is None:
        out['template_at_edge'] = False
        return out
    grid = best.grid
    has = _encoded_axes(grid)
    for a in TEMPLATE_COLUMNS:
        if a in has:
            out[f'template_{a}'] = float(best.param_dict[a])
    same = [i for i, c in enumerate(cands) if c.grid is grid and np.isfinite(scores[i] if len(scores) > i else np.nan)]
    if len(same) > 1:
        chi = scores[same]
        s = max(1., float(np.nanmin(chi)) / max(dof, 1))
        w = np.exp(-(chi - np.nanmin(chi)) / (2. * s))
        w /= w.sum()
        for a in grid.axes:
            vals = np.array([cands[i].param_dict[a] for i in same])
            if a in has:
                mean = float(np.sum(w * vals))
                out['estimate'][a] = (mean, float(np.sqrt(np.sum(w * (vals - mean) ** 2))))
    grids = grid_names(tcfg) if tcfg is not None else [grid.name]
    for a in grid.axes:
        searched = sorted({c.param_dict[a] for c in cands if c.grid is grid})
        if len(searched) < 2:
            continue
        v = best.param_dict[a]
        direction = 'low' if np.isclose(v, searched[0]) else 'high' if np.isclose(v, searched[-1]) else None
        if direction is not None:
            out['edges'].append(_edge_guidance(grid, a, v, direction, (searched[0], searched[-1]), best, grids))
    out['template_at_edge'] = bool(out['edges'])
    est = ', '.join(f'{a} {m:.4g} ± {sd:.2g}' for a, (m, sd) in out['estimate'].items())
    log.info('%s: best template %s (χ² %.1f over %d candidates)%s', method, best.label, out['template_chi2'],
             len(cands), f'; χ²-weighted {est}' if est else '')
    for e in out['edges']:
        log.warning('%s: %s', method, e['message'])
    return out


def _range_key(axis: str) -> str:
    return f'{axis}_range' if axis in TEMPLATE_COLUMNS else f'ranges.{axis}'


def _ranges(axis: str, rng: tuple[float, float]) -> dict:
    """The templates-section keys setting one axis's range"""
    r = [float(rng[0]), float(rng[1])]
    return {f'{axis}_range': r} if axis in TEMPLATE_COLUMNS else {'ranges': {axis: r}}


def _edge_guidance(grid, axis: str, v: float, direction: str, searched: tuple[float, float], best,
                   grids: list[str]) -> dict:
    """Guidance for one axis at an edge (see template_summary)"""
    gvals = grid.axis_values(axis)
    glo, ghi = float(gvals[0]), float(gvals[-1])
    step = float(np.median(np.diff(gvals))) if len(gvals) > 1 else T.AXIS_SCALE.get(axis, 1.)
    e = dict(axis=axis, direction=direction, value=float(v), searched=tuple(map(float, searched)), grid=grid.name,
             grid_bounds=(glo, ghi), action='', cfg_patch={}, download=None, command='', message='')
    where = (f'best {axis} {v:g} ({best.label}) is at the {direction} edge of the searched {axis} range '
             f'{searched[0]:g} to {searched[1]:g}')
    if (v < ghi - 1e-9) if direction == 'high' else (v > glo + 1e-9):  # the grid extends further
        lo, hi = searched
        lo, hi = (max(glo, lo - 2 * step), hi) if direction == 'low' else (lo, min(ghi, hi + 2 * step))
        e.update(action='widen', cfg_patch={'templates': _ranges(axis, (lo, hi))})
        e['message'] = (f'{where}; grid {grid.name} extends to {glo if direction == "low" else ghi:g}: widen '
                        f'templates.{_range_key(axis)} to [{lo:g}, {hi:g}] and refit')
        return e
    # the grid itself ends here: suggest a known grid that extends further
    params = best.param_dict
    options = []
    for name, meta in T.KNOWN_GRIDS.items():
        cov = meta.coverage.get(axis)
        if cov is None or name == grid.name:
            continue
        beyond = cov[0] < v - 1e-9 if direction == 'low' else cov[1] > v + 1e-9
        fits = all(meta.coverage.get(a, (-np.inf, np.inf))[0] - 1e-9 <= params[a] <= meta.coverage.get(
            a, (-np.inf, np.inf))[1] + 1e-9 for a in ('teff', 'logg') if a != axis)
        if beyond and fits:
            options.append((name, meta, cov))
    if not options:
        e.update(action='none', message=f'{where}, which is also the edge of grid {grid.name}; no known grid '
                                        f'extends further: consider a custom local grid (templates.local_dir)')
        return e
    name, meta, cov = options[0]
    width = 3 * T.AXIS_SCALE.get(axis, step)
    new = (max(cov[0], v - width), v) if direction == 'low' else (v, min(cov[1], v + width))
    teff, logg = params['teff'], params['logg']
    dl = {'grid': name, 'teff_range': (teff - 300., teff + 300.), 'logg_range': (logg - 0.5, logg + 0.5)}
    dl |= {f'{axis}_range': new} if axis in TEMPLATE_COLUMNS else {'ranges': {axis: new}}
    cmd = (f'rvfitter templates download --grid {name} --teff {dl["teff_range"][0]:g} {dl["teff_range"][1]:g} '
           f'--logg {dl["logg_range"][0]:g} {dl["logg_range"][1]:g}')
    if axis in ('met', 'alpha'):
        cmd += f' --{axis} {new[0]:g} {new[1]:g}'
    elif axis not in ('teff', 'logg'):
        cmd += f' --range {axis} {new[0]:g} {new[1]:g}'
    pr = (min(new[0], searched[0]), max(new[1], searched[1]))
    e.update(action='download', download=dl, command=cmd,
             cfg_patch={'templates': {'grid': [*grids, name], **_ranges(axis, pr)}})
    e['message'] = (f'{where}, which is also the edge of grid {grid.name}. Grid {name} ({meta.reference}) covers '
                    f'{axis} {cov[0]:g} to {cov[1]:g}: run "{cmd}", then set templates.grid = {[*grids, name]} '
                    f'and templates.{_range_key(axis)} = [{pr[0]:g}, {pr[1]:g}], and refit'
                    + ('' if meta.verified else f' (the SVO id of {name} is not yet verified)'))
    return e
