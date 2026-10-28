"""
Helpers shared by the template-based methods: configuration access, region handling, log-λ data preparation,
template candidates and deterministic parallel maps.
"""
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import broadening
from ..constants import C_KMS
from ..types import Measurement, MethodResult, Region, Spectrum

log = logging.getLogger(__name__)

# measurement flag bits, from results.FLAGS
from ..results import FLAGS as _FLAGS  # noqa: E402

FLAG_FIT_FAILED = _FLAGS['fit_failed']  # the fit failed or gave a non-finite error
FLAG_FEW_PIXELS = _FLAGS['few_regions']  # too few usable pixels/regions
FLAG_OUTSIDE_GRID = _FLAGS['template_grid_edge']  # template requested outside the grid: nearest point used
FLAG_EDGE = _FLAGS['not_converged']  # optimum at the edge of the search range
FLAG_POOR_FIT = _FLAGS['poor_fit']  # reduced χ² > 3 (the error is scaled; check the template)
FLAG_OUTLIER = _FLAGS['outlier']  # gross outlier among the method's regions (excluded)
FLAG_GRID_HOLE = _FLAGS['template_grid_hole']  # template in a hole of the grid (missing corner): nearest used
FLAG_PARAM_EDGE = _FLAGS['template_param_edge']  # best template at the edge of the searched parameters
SERIOUS = FLAG_EDGE | FLAG_FIT_FAILED | FLAG_FEW_PIXELS

MASK_KINDS = ('telluric',)
SKIP_KINDS = ('telluric', 'continuum')


# ---------------------------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------------------------
def section(cfg: Any, name: str) -> dict:
    """A config section as a dict, from a Config dataclass tree, a mapping or None"""
    if cfg is None:
        return {}
    sec = cfg.get(name) if isinstance(cfg, Mapping) else getattr(cfg, name, None)
    if sec is None:
        return {}
    if isinstance(sec, Mapping):
        return dict(sec)
    return {k: getattr(sec, k) for k in dir(sec) if not k.startswith('_') and not callable(getattr(sec, k))}


def options(cfg: Any, method: str, defaults: Mapping[str, Any]) -> dict:
    """
    The method's options: `defaults` updated by cfg.methods.options[method]

    Unknown keys raise, so typos in a config are not silently ignored.
    """
    given = dict(section(cfg, 'methods').get('options', {}) or {}).get(method, {}) or {}
    unknown = set(given) - set(defaults)
    if unknown:
        raise ValueError(f'unknown {method} options {sorted(unknown)}; valid: {sorted(defaults)}')
    out = dict(defaults)
    out.update(given)
    return out


_ESTIMATED_R: dict[tuple, float] = {}


def resolution_of(spectrum: Spectrum, cfg: Any, opts: Mapping[str, Any], *, estimate: bool = True) -> float:
    """
    The data's resolving power: the method option, else cfg.input.resolution, else spectrum.resolution, else
    (if `estimate`) `template_fit.estimate_resolution`, cached per spectrum and logged as a warning
    """
    for r in (opts.get('resolution'), section(cfg, 'input').get('resolution'), spectrum.resolution):
        if r is not None and np.isfinite(r) and r > 0:
            return float(r)
    if not estimate:
        raise ValueError('the spectral resolution is unknown: set input.resolution or the method option '
                         '"resolution"')
    key = (spectrum.source, len(spectrum.wave), spectrum.wmin, spectrum.wmax, float(np.nansum(spectrum.flux)))
    if key not in _ESTIMATED_R:
        from .template_fit import estimate_resolution
        est_cfg = cfg
        given = {k: opts[k] for k in ('template', 'templates') if opts.get(k) is not None}
        if given:  # estimate with the calling method's own template
            est_cfg = {'templates': section(cfg, 'templates'), 'methods': {'options': {'template_fit': given}}}
        out = estimate_resolution(spectrum, cfg=est_cfg)
        if not np.isfinite(out['resolution']):
            raise ValueError('the spectral resolution is unknown and could not be estimated: set input.resolution')
        log.warning('Resolution of %s unknown: estimated R = %.0f ± %.0f from a template fit (set input.resolution '
                    'to override)', spectrum.source or 'the spectrum', out['resolution'], out['resolution_err'])
        _ESTIMATED_R[key] = out['resolution']
    return _ESTIMATED_R[key]


# ---------------------------------------------------------------------------------------------------------------
# regions
# ---------------------------------------------------------------------------------------------------------------
def split_regions(regions: Sequence[Region], kinds: Iterable[str] | None = None) -> tuple[list[Region], list[Region]]:
    """
    Fitting windows and masked intervals

    Parameters
    ----------
    regions
        The regions selected for this method
    kinds
        Region kinds used as windows (default: everything except telluric and continuum regions)

    Returns
    -------
    windows, masks
        The windows, and the telluric regions to exclude from them
    """
    kinds = None if kinds is None else tuple(kinds)
    windows = [r for r in regions if (r.kind in kinds if kinds else r.kind not in SKIP_KINDS)]
    masks = [r for r in regions if r.kind in MASK_KINDS]
    return windows, masks


def in_any(wave: np.ndarray, intervals: Iterable[tuple[float, float]]) -> np.ndarray:
    """Boolean mask of wavelengths inside any of the [wmin, wmax] intervals"""
    out = np.zeros(len(wave), dtype=bool)
    for lo, hi in intervals:
        out |= (wave >= lo) & (wave <= hi)
    return out


# ---------------------------------------------------------------------------------------------------------------
# data on a log-λ grid
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Prepared:
    """Data resampled onto a uniform ln λ grid, with a usable-pixel mask"""
    wave: np.ndarray
    flux: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray
    dv: float
    sigma_known: bool
    region: Region
    source_pixels: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.mask.sum())


def prepare(spectrum: Spectrum, window: Region, masks: Sequence[Region], dv: float | None = None,
            flux: np.ndarray | None = None, *, resample: bool = False) -> Prepared:
    """
    The part of a spectrum in `window`, on its own pixels (default) or resampled onto a uniform ln λ grid

    Working on the observed pixels keeps the noise independent and avoids the extra smoothing of a resampling,
    which the template rows would not share (on J1745-like BT-Settl injections the resampling biased the CCF by
    −0.1 to −0.55 km/s per feature window, noiseless). Within a window the velocity step is near-constant, and
    the CCF filters data and template rows identically, so a uniform ln λ grid is not needed.

    Parameters
    ----------
    spectrum
        The observed spectrum
    window
        The fitting window
    masks
        Intervals to exclude (e.g. telluric bands)
    dv
        Grid step [km/s] when resampling; default the median native step within the window
    flux
        Replacement flux on the spectrum's pixels (used by Monte Carlo trials)
    resample
        Resample onto a uniform ln λ grid (flux-conserving) instead of using the observed pixels

    Returns
    -------
    prepared
        The window's data; `dv` is the median velocity step. Pixels in a mask or non-finite are masked out
    """
    flux = spectrum.flux if flux is None else flux
    inwin = (spectrum.wave >= window.wmin) & (spectrum.wave <= window.wmax)
    if inwin.sum() < 4:
        raise ValueError(f'window {window.name} covers fewer than 4 pixels')
    w = spectrum.wave[inwin]
    sigma_known = bool(np.any(np.isfinite(spectrum.sigma) & (spectrum.sigma > 0)))
    if resample:
        if dv is None:
            dv = float(np.median(np.diff(np.log(w)))) * C_KMS
        grid = broadening.log_wave_grid(w[0], w[-1], dv)
        f = broadening.rebin(spectrum.wave, flux, grid)
        s = (broadening._rebin_sigma(spectrum.wave, spectrum.sigma, grid) if sigma_known
             else np.full(len(grid), np.nan))
    else:
        grid, f, s = w, np.asarray(flux, dtype=float)[inwin], spectrum.sigma[inwin]
        dv = float(np.median(np.diff(np.log(w)))) * C_KMS
    mask = np.isfinite(f) & ~in_any(grid, [(m.wmin, m.wmax) for m in masks])
    if sigma_known:
        mask &= np.isfinite(s) & (s > 0)
    return Prepared(grid, f, s, mask, dv, sigma_known, window, int(inwin.sum()))


# ---------------------------------------------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Template:
    """A rest-frame template on a uniform ln λ grid, already broadened to the data's resolution"""
    lnw0: float
    dlnw: float
    flux: np.ndarray
    label: str
    params: dict = field(default_factory=dict)

    @property
    def wave(self) -> np.ndarray:
        return np.exp(self.lnw0 + self.dlnw * np.arange(len(self.flux)))

    def at(self, wave: np.ndarray, v: float | np.ndarray = 0., *, integrate: bool = True) -> np.ndarray:
        """
        Template flux at observed pixels for radial velocities v (scalar → 1D, array → (len(v), len(wave)))

        By default the shifted template is averaged over each observed pixel (edges half-way between centres in
        ln λ, and extrapolated by half a pixel at the two ends, which makes the first and last pixels very slightly
        asymmetric; negligible), matching pixel-integrated data; `integrate=False` point-samples at the centres.
        """
        lnw = np.log(np.asarray(wave, dtype=float))
        v = np.asarray(v, dtype=float)
        shift = np.atleast_1d(np.log(broadening.doppler_factor(v)))
        idx = np.arange(len(self.flux), dtype=float)
        if not integrate or len(lnw) < 2:
            x = ((lnw[None, :] - shift[:, None]) - self.lnw0) / self.dlnw
            out = np.interp(x.ravel(), idx, self.flux, left=np.nan, right=np.nan).reshape(x.shape)
            return out[0] if v.ndim == 0 else out
        mid = 0.5 * (lnw[1:] + lnw[:-1])
        edges = np.concatenate([[2 * lnw[0] - mid[0]], mid, [2 * lnw[-1] - mid[-1]]])
        finite = np.isfinite(self.flux)
        f = np.where(finite, self.flux, 0.)
        cum = np.concatenate([[0.], np.cumsum(0.5 * (f[1:] + f[:-1]))])  # integral in index units
        bad = np.concatenate([[0.], np.cumsum(~finite[1:] | ~finite[:-1])]).astype(float)
        x = ((edges[None, :] - shift[:, None]) - self.lnw0) / self.dlnw
        inside = (x >= 0) & (x <= len(f) - 1)
        cx = np.interp(x.ravel(), idx, cum).reshape(x.shape)
        bx = np.interp(x.ravel(), idx, bad).reshape(x.shape)
        width = np.diff(x, axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            out = np.diff(cx, axis=1) / width
        ok = inside[:, :-1] & inside[:, 1:] & (np.diff(bx, axis=1) == 0)
        out[~ok] = np.nan
        return out[0] if v.ndim == 0 else out


def template_from_spectrum(spec: Spectrum, resolution: float, wave_range: tuple[float, float], *,
                           vsini: float = 0., label: str | None = None, dv: float | None = None,
                           pad_kms: float = 1500.) -> Template:
    """
    Broaden a rest-frame template spectrum to R (and vsini) on a uniform ln λ grid covering wave_range ± pad

    Parameters
    ----------
    spec
        Rest-frame template (vacuum); its `resolution` (None = infinite) is removed in quadrature
    resolution
        Target resolving power
    wave_range
        Observed range to cover [Å]
    vsini
        Rotational broadening [km/s]
    label
        Template label
    dv
        Grid step [km/s] (default c/(10 R))
    pad_kms
        Velocity padding on both sides, so shifted templates still cover the data
    """
    dv = C_KMS / (10. * resolution) if dv is None else dv
    lo = wave_range[0] * broadening.doppler_factor(-pad_kms)
    hi = wave_range[1] * broadening.doppler_factor(pad_kms)
    sel = (spec.wave >= lo * 0.999) & (spec.wave <= hi * 1.001)
    w, f = spec.wave[sel], spec.flux[sel]
    if len(w) < 4 or w[0] > lo or w[-1] < hi:
        raise ValueError(f'template does not cover {lo:.1f}–{hi:.1f} Å')
    dv_fine = min(float(np.median(np.diff(np.log(w)))) * C_KMS, dv)
    fine = broadening.log_wave_grid(w[0], w[-1], dv_fine)
    ff = broadening.rebin(w, f, fine)
    native = spec.resolution if spec.resolution and np.isfinite(spec.resolution) else None
    ff = broadening.broaden_loglam(ff, dv_fine, resolution=resolution, vsini=vsini, resolution_native=native)
    grid = broadening.log_wave_grid(lo, hi, dv)
    out = broadening.rebin(fine, ff, grid)
    return Template(float(np.log(grid[0])), dv / C_KMS, out, label or str(spec.meta.get('object', 'template')),
                    dict(spec.meta))


def template_from_grid_spec(spec: Spectrum, label: str) -> Template:
    """Wrap an already-broadened ln-uniform Spectrum (TemplateGrid.broadened) as a Template"""
    lnw = np.log(spec.wave)
    return Template(float(lnw[0]), float(lnw[1] - lnw[0]), np.asarray(spec.flux), label, dict(spec.meta))


@dataclass(frozen=True)
class Candidate:
    """A template candidate: either a given spectrum or a grid point (params in the grid's axes order)"""
    label: str
    spectrum: Spectrum | None = None
    grid: Any = None
    params: tuple[float, ...] | None = None

    @property
    def param_dict(self) -> dict[str, float]:
        """{axis: value} for a grid point, {} otherwise"""
        if self.grid is None or self.params is None:
            return {}
        given = dict(zip(self.grid.axes, self.params, strict=False))  # trailing axes may be omitted: defaults
        return self.grid.params(**given)


def grid_names(tcfg: Mapping[str, Any]) -> list[str]:
    """The configured grid names (templates.grid may be one name or a list)"""
    g = tcfg.get('grid') or 'bt-settl-cifist'
    return [g] if isinstance(g, str) else [str(x) for x in g]


def candidates(cfg: Any, opts: Mapping[str, Any]) -> list[Candidate]:
    """
    Template candidates from the method options or the config

    - opts['template']: a rest-frame Spectrum, or a (wave, flux) tuple, used alone;
    - opts['templates']: a list of such;
    - otherwise every configured grid (templates.grid: a name or a list) is opened and its points within the
      ranges become candidates: teff within teff_range (default teff_prior ± 300 K), log g within logg_range
      (default all), and met, alpha and extra axes within their ranges or held at the value nearest their
      prior/default (templates.met_prior, alpha_prior; extra-axis ranges in templates.ranges).
    """
    def as_spec(t, i):
        if isinstance(t, Spectrum):
            return t
        w, f = t
        return Spectrum(w, f, np.full(len(w), np.nan), None, {'object': f'template{i}'}, 'user')

    if opts.get('template') is not None:
        s = as_spec(opts['template'], 0)
        return [Candidate(str(s.meta.get('object', 'template')), spectrum=s)]
    if opts.get('templates'):
        specs = [as_spec(t, i) for i, t in enumerate(opts['templates'])]
        return [Candidate(str(s.meta.get('object', f'template{i}')), spectrum=s) for i, s in enumerate(specs)]
    tcfg = section(cfg, 'templates')
    teff = tcfg.get('teff_prior')
    teff_range = tcfg.get('teff_range')
    if teff_range is None and teff is not None:
        teff_range = (teff - opts.get('teff_window', 300.), teff + opts.get('teff_window', 300.))
    priors = {k: tcfg.get(f'{k}_prior') for k in ('met', 'alpha') if tcfg.get(f'{k}_prior') is not None}
    out = []
    for name in grid_names(tcfg):
        grid = open_grid(tcfg, name)
        extra = {k: tuple(v) for k, v in dict(tcfg.get('ranges') or {}).items() if k in grid.axes}
        points = grid.select(teff_range=teff_range, logg_range=tcfg.get('logg_range'),
                             met_range=tcfg.get('met_range'), alpha_range=tcfg.get('alpha_range'), ranges=extra,
                             priors=priors)
        if not points:
            points = [grid.nearest(teff or 2000., tcfg.get('logg_prior') or 5., **priors)]
            log.warning('%s: no grid points in the prior ranges: using %s', grid.name, points[0])
        out += [Candidate(f'{grid.name} ' + '/'.join(f'{v:g}' for v in p), grid=grid, params=p) for p in points]
    return out


def open_grid(tcfg: Mapping[str, Any], name: str | None = None):
    """
    The TemplateGrid for one configured grid: its local directory, else the downloaded grid in the cache

    `templates.local_dir` may be a directory (used for the first grid) or a mapping grid → directory. A `medium`
    given in the config overrides the first grid's declared medium (and is then verified like any other).
    """
    from dataclasses import replace as _replace
    from pathlib import Path

    from .. import templates as T
    names = grid_names(tcfg)
    name = name or names[0]
    local = tcfg.get('local_dir')
    directory = local.get(name) if isinstance(local, Mapping) else (local if name == names[0] else None)
    if not directory:
        cached = T.cache_dir() / name
        if not cached.is_dir():
            raise ValueError(f'no templates for grid {name}: set templates.local_dir, or run "rvfitter templates '
                             f'download --grid {name}", or give the method option "template"')
        directory = cached
    meta = None
    if tcfg.get('medium') and name == names[0]:
        base = T._read_metadata(Path(directory)) or T.KNOWN_GRIDS.get(name)
        if base is None:
            raise ValueError(f'unknown grid {name!r} and no metadata in {directory}')
        meta = _replace(base, medium=tcfg['medium'])
    return T.TemplateGrid(directory, grid=name, meta=meta)


def build_template(cand: Candidate, resolution: float, wave_range: tuple[float, float], *, vsini: float = 0.,
                   pad_kms: float = 1500.) -> Template:
    """Broadened Template for a candidate covering wave_range ± pad_kms"""
    if cand.spectrum is not None:
        return template_from_spectrum(cand.spectrum, resolution, wave_range, vsini=vsini, label=cand.label,
                                      pad_kms=pad_kms)
    lo = wave_range[0] * broadening.doppler_factor(-pad_kms)
    hi = wave_range[1] * broadening.doppler_factor(pad_kms)
    spec = cand.grid.broadened(**cand.param_dict, resolution=resolution, wave_range=(lo, hi), vsini=vsini)
    t = template_from_grid_spec(spec, cand.label)
    t.params.update(cand.param_dict)
    return t


# ---------------------------------------------------------------------------------------------------------------
# best template, its parameters and edge guidance
# ---------------------------------------------------------------------------------------------------------------
TEMPLATE_COLUMNS = ('teff', 'logg', 'met', 'alpha')


def _encoded_axes(grid) -> set[str]:
    """Axes a grid really has: encoded in its file names, or taking more than one value"""
    import re
    named = set(re.compile(grid.meta.pattern).groupindex)
    return {a for a in grid.axes if a in named or len(grid.axis_values(a)) > 1}


def template_summary(method: str, cands: Sequence[Candidate], scores: Sequence[float], best: Candidate,
                     dof: int, tcfg: Mapping[str, Any] | None = None) -> dict:
    """
    The best template, its χ², χ²-weighted parameter estimates and any parameter edge, with guidance

    Weights over the candidates of the best grid are exp(−Δχ²/(2s)), with s = max(1, χ²_min/ν), so that a fit
    dominated by model mismatch does not give unrealistically sharp parameters.
    If the best candidate is at the low or high end of a searched axis (one with at least two values), the edge is
    reported: widening `templates.<axis>_range` if the grid extends further, otherwise a known grid that covers
    the region, with the download command and the configuration change (both also as data for a GUI).

    Returns
    -------
    summary
        'template_grid', 'template_label', 'template_<axis>' for teff, logg, met and alpha (NaN where the grid has
        no such axis), 'template_chi2', 'template_at_edge', 'estimate' {axis: (mean, sd)}, 'edges' [dict]
    """
    from .. import templates as T
    tcfg = dict(tcfg or {})
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
    for a in grid.axes:
        searched = sorted({c.param_dict[a] for c in cands if c.grid is grid})
        if len(searched) < 2:
            continue
        v = best.param_dict[a]
        direction = 'low' if np.isclose(v, searched[0]) else 'high' if np.isclose(v, searched[-1]) else None
        if direction is None:
            continue
        out['edges'].append(_edge_guidance(grid, a, v, direction, (searched[0], searched[-1]), best, tcfg, T))
    out['template_at_edge'] = bool(out['edges'])
    est = ', '.join(f'{a} {m:.4g} ± {sd:.2g}' for a, (m, sd) in out['estimate'].items())
    log.info('%s: best template %s (χ² %.1f over %d candidates)%s', method, best.label, out['template_chi2'],
             len(cands), f'; χ²-weighted {est}' if est else '')
    for e in out['edges']:
        log.warning('%s: %s', method, e['message'])
    return out


def _range_key(axis: str) -> str:
    return f'{axis}_range' if axis in ('teff', 'logg', 'met', 'alpha') else f'ranges.{axis}'


def _patch(axis: str, rng: tuple[float, float], **extra) -> dict:
    t = dict(extra)
    if axis in ('teff', 'logg', 'met', 'alpha'):
        t[f'{axis}_range'] = [float(rng[0]), float(rng[1])]
    else:
        t['ranges'] = {axis: [float(rng[0]), float(rng[1])]}
    return {'templates': t}


def _edge_guidance(grid, axis, v, direction, searched, best, tcfg, T) -> dict:
    """Guidance for one axis at an edge (see template_summary)"""
    gvals = grid.axis_values(axis)
    glo, ghi = float(gvals[0]), float(gvals[-1])
    step = float(np.median(np.diff(gvals))) if len(gvals) > 1 else T.AXIS_SCALE.get(axis, 1.)
    e = dict(axis=axis, direction=direction, value=float(v), searched=tuple(map(float, searched)), grid=grid.name,
             grid_bounds=(glo, ghi), action='', cfg_patch={}, download=None, command='', message='')
    where = (f'best {axis} {v:g} ({best.label}) is at the {direction} edge of the searched {axis} range '
             f'{searched[0]:g} to {searched[1]:g}')
    extends = v < ghi - 1e-9 if direction == 'high' else v > glo + 1e-9
    if extends:
        lo, hi = searched
        lo, hi = (max(glo, lo - 2 * step), hi) if direction == 'low' else (lo, min(ghi, hi + 2 * step))
        e.update(action='widen', cfg_patch=_patch(axis, (lo, hi)))
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
    teff = params.get('teff', 2000.)
    logg = params.get('logg', 5.)
    dl = {'grid': name, 'teff_range': (teff - 300., teff + 300.), 'logg_range': (logg - 0.5, logg + 0.5)}
    if axis in ('teff', 'logg', 'met', 'alpha'):
        dl[f'{axis}_range'] = new
    else:
        dl['ranges'] = {axis: new}
    flag = {'teff': '--teff', 'logg': '--logg', 'met': '--met', 'alpha': '--alpha'}
    cmd = (f'rvfitter templates download --grid {name} --teff {dl["teff_range"][0]:g} {dl["teff_range"][1]:g} '
           f'--logg {dl["logg_range"][0]:g} {dl["logg_range"][1]:g}')
    if axis in ('met', 'alpha'):
        cmd += f' {flag[axis]} {new[0]:g} {new[1]:g}'
    elif axis not in ('teff', 'logg'):
        cmd += f' --range {axis} {new[0]:g} {new[1]:g}'
    grids = grid_names(tcfg) if tcfg else [grid.name]
    patch = _patch(axis, (min(new[0], searched[0]), max(new[1], searched[1])), grid=[*grids, name])
    e.update(action='download', download=dl, command=cmd, cfg_patch=patch)
    pr = patch['templates'].get(f'{axis}_range') or patch['templates'].get('ranges', {}).get(axis)
    e['message'] = (f'{where}, which is also the edge of grid {grid.name}. Grid {name} ({meta.reference}) covers '
                    f'{axis} {cov[0]:g} to {cov[1]:g}: run "{cmd}", then set templates.grid = {[*grids, name]} '
                    f'and templates.{_range_key(axis)} = [{pr[0]:g}, {pr[1]:g}], and refit'
                    + ('' if meta.verified else f' (the SVO id of {name} is not yet verified)'))
    return e


# ---------------------------------------------------------------------------------------------------------------
# parallel maps and results
# ---------------------------------------------------------------------------------------------------------------
def pmap(fn: Callable, items: Sequence, executor: Executor | None = None) -> list:
    """Map fn over items in order, serially or with an executor; results never depend on which"""
    if executor is None or len(items) < 2:
        return [fn(x) for x in items]
    return list(executor.map(fn, items))


def child_seeds(rng: np.random.Generator, n: int) -> list[np.random.SeedSequence]:
    """n independent seed sequences derived from rng (deterministic for a given rng state)"""
    entropy = [int(x) for x in rng.integers(0, 2 ** 63 - 1, size=4)]
    return np.random.SeedSequence(entropy).spawn(n)


def segment(windows: Sequence[Region], resolution: float, n_res: float | None) -> list[Region]:
    """
    Split windows longer than n_res resolution elements into near-equal segments (per-segment RVs)

    Each segment is a separate measurement, so the segment-to-segment scatter enters the Birge combination and
    the hierarchical line × method model, and a region that disagrees (e.g. telluric residuals or a badly
    matched molecular band) is visible instead of being averaged into one number.

    Parameters
    ----------
    windows
        The fitting windows
    resolution
        Resolving power
    n_res
        Maximum segment length in resolution elements (None or ≤0: no splitting)

    Returns
    -------
    segments
        The windows, with long ones replaced by segments named "<name> [a–b]"
    """
    if not n_res or n_res <= 0:
        return list(windows)
    out = []
    for w in windows:
        n_elem = np.log(w.wmax / w.wmin) * resolution
        k = int(np.ceil(n_elem / n_res - 1e-9))
        if k <= 1:
            out.append(w)
            continue
        edges = w.wmin * (w.wmax / w.wmin) ** (np.arange(k + 1) / k)
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            out.append(Region(f'{w.name} [{a:.0f}-{b:.0f}]', w.kind, float(a), float(b), w.centre, w.continuum,
                              f'{w.reason}; segment {a:.0f}-{b:.0f} of {k}'.lstrip('; ')))
    return out


def reject_outliers(measurements: list[Measurement], k: float = 5.) -> list[Measurement]:
    """
    Flag gross outliers among a method's usable measurements (at least three)

    A measurement is an outlier if |rv − median| > k √(s² + σ²), with s = 1.4826 × MAD of the RVs. With k = 5
    this only catches failures (e.g. a fit locked on a wrong optimum), not ordinary scatter, which the Birge
    and hierarchical combinations handle. Outliers get the `FLAG_OUTLIER` bit (excluded from the combination).
    """
    usable = [m for m in measurements if not (m.flag & SERIOUS) and np.isfinite(m.rv) and np.isfinite(m.rv_err)]
    if len(usable) < 3:
        return measurements
    rvs = np.array([m.rv for m in usable])
    med = float(np.median(rvs))
    s = 1.4826 * float(np.median(np.abs(rvs - med)))
    for m in usable:
        if abs(m.rv - med) > k * np.hypot(s, m.rv_err):
            m.flag |= FLAG_OUTLIER
            log.warning('%s %s: %.1f km/s is a gross outlier (median %.1f, robust scatter %.1f); excluded',
                        m.method, m.region, m.rv, med, s)
    return measurements


def finish(method: str, measurements: list[Measurement], outlier_k: float | None = 5.) -> MethodResult:
    """Combine a method's per-region measurements (random effects), ignoring seriously flagged ones and outliers"""
    from ..combine import combine_regions
    if outlier_k:
        reject_outliers(measurements, outlier_k)
    bad = SERIOUS | FLAG_OUTLIER
    good = [m for m in measurements if not (m.flag & bad) and np.isfinite(m.rv) and np.isfinite(m.rv_err)]
    details: dict = {}
    rv, err = combine_regions(good, details=details)
    for m in measurements:  # the combination is method-level; recorded on every row for the details table
        m.diagnostics.update(method_tau=details.get('tau_line', np.nan), method_birge_rv=details.get('birge_rv'),
                             method_birge_err=details.get('birge_err'))
    if len(good) > 1:
        log.info('%s: random-effects %.2f ± %.2f km/s (τ %.1f km/s over %d regions; Birge %.2f ± %.2f km/s)', method,
                 rv, err, details.get('tau_line', np.nan), len(good), details.get('birge_rv', np.nan),
                 details.get('birge_err', np.nan))
    return MethodResult(method, rv, err, len(good), measurements)


def highpass(wave: np.ndarray, flux: np.ndarray, mask: np.ndarray, fwhm_kms: float) -> np.ndarray:
    """
    Smooth continuum by normalised Gaussian convolution in ln λ, using only the masked-in pixels

    Parameters
    ----------
    wave
        Uniform ln λ grid
    flux
        Flux
    mask
        Pixels to use
    fwhm_kms
        Gaussian FWHM [km/s]

    Returns
    -------
    continuum
        The smoothed flux (NaN where no usable pixel lies within the kernel)
    """
    dv = float(np.log(wave[1] / wave[0])) * C_KMS
    kernel = broadening.gaussian_kernel(fwhm_kms * broadening.FWHM_TO_SIGMA / dv, nsigma=3.)
    from scipy.signal import fftconvolve
    m = mask & np.isfinite(flux)
    num = fftconvolve(np.where(m, flux, 0.), kernel, mode='same')
    den = fftconvolve(m.astype(float), kernel, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    out[den < 1e-3] = np.nan
    return out
