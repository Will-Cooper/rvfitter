"""
Helpers shared by the template-based methods: options, regions, window data, template candidates and their
selection, deterministic parallel maps, and the per-window and per-method results.
"""
import copy
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .. import broadening
from .. import templates as T
from ..combine import combine_regions
from ..constants import C_KMS
from ..results import FLAGS as _FLAGS
from ..types import Measurement, MethodResult, Region, Spectrum
from .guidance import grid_names, template_summary

log = logging.getLogger(__name__)

# measurement flag bits, from results.FLAGS
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

#: options shared by the template methods (ccf, template_fit)
TEMPLATE_DEFAULTS: dict[str, Any] = dict(
    resolution=None,  # override the data's R
    rv_min=-500., rv_max=500.,  # search range [km/s]
    rv_step=None,  # RV grid step [km/s]; the default is the method's
    n_chunks=8,  # parallel tasks (fixed, so results do not depend on the executor)
    template=None, templates=None,  # a rest-frame Spectrum or (wave, flux), or a list of them, instead of the grid
    teff_window=300.,  # grid candidates within teff_prior ± this [K] when templates.teff_range is unset
    segment_res=150.,  # split windows longer than this many resolution elements (0: never)
    outlier_k=5.,  # gross-outlier rejection among regions (0: off)
    pad_kms=1500.,  # template coverage beyond the data [km/s]
)


def options(cfg: Any, method: str, defaults: Mapping[str, Any]) -> dict:
    """The method's options: `defaults` updated by cfg.methods.options[method]; unknown keys raise"""
    given = cfg.methods.options.get(method) or {}
    unknown = set(given) - set(defaults)
    if unknown:
        raise ValueError(f'unknown {method} options {sorted(unknown)}; valid: {sorted(defaults)}')
    return {**defaults, **given}


_ESTIMATED_R: dict[tuple, float] = {}


def resolution_of(spectrum: Spectrum, cfg: Any, opts: Mapping[str, Any], *, estimate: bool = True) -> float:
    """
    The data's resolving power: the method option, else cfg.input.resolution, else spectrum.resolution, else
    (if `estimate`) `template_fit.estimate_resolution`, cached per spectrum and logged as a warning
    """
    for r in (opts.get('resolution'), cfg.input.resolution, spectrum.resolution):
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
            est_cfg = copy.deepcopy(cfg)
            est_cfg.methods.options = {'template_fit': given}
        out = estimate_resolution(spectrum, cfg=est_cfg)
        if not np.isfinite(out['resolution']):
            raise ValueError('the spectral resolution is unknown and could not be estimated: set input.resolution')
        log.warning('Resolution of %s unknown: estimated R = %.0f ± %.0f from a template fit (set input.resolution '
                    'to override)', spectrum.source or 'the spectrum', out['resolution'], out['resolution_err'])
        _ESTIMATED_R[key] = out['resolution']
    return _ESTIMATED_R[key]


def split_regions(regions: Sequence[Region]) -> tuple[list[Region], list[Region]]:
    """The fitting windows (all but telluric and continuum regions) and the telluric regions masked out of them"""
    return [r for r in regions if r.kind not in SKIP_KINDS], [r for r in regions if r.kind in MASK_KINDS]


def in_any(wave: np.ndarray, intervals: Iterable[tuple[float, float]]) -> np.ndarray:
    """Boolean mask of wavelengths inside any of the [wmin, wmax] intervals"""
    out = np.zeros(len(wave), dtype=bool)
    for lo, hi in intervals:
        out |= (wave >= lo) & (wave <= hi)
    return out


@dataclass
class Prepared:
    """A window's data on its own pixels, with a usable-pixel mask and the median velocity step `dv` [km/s]"""
    wave: np.ndarray
    flux: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray
    dv: float
    sigma_known: bool
    region: Region

    @property
    def n(self) -> int:
        return int(self.mask.sum())


def prepare(spectrum: Spectrum, window: Region, masks: Sequence[Region], flux: np.ndarray | None = None
            ) -> Prepared:
    """
    The part of a spectrum in `window`, on its own pixels; masked or non-finite pixels are masked out

    The observed pixels keep the noise independent and avoid the extra smoothing of a resampling, which the
    template rows would not share (on J1745-like BT-Settl injections the resampling biased the CCF by −0.1 to
    −0.55 km/s per feature window, noiseless). `flux` replaces the spectrum's flux (Monte Carlo trials).
    """
    flux = spectrum.flux if flux is None else flux
    inwin = (spectrum.wave >= window.wmin) & (spectrum.wave <= window.wmax)
    if inwin.sum() < 4:
        raise ValueError(f'window {window.name} covers fewer than 4 pixels')
    w = spectrum.wave[inwin]
    sigma_known = bool(np.any(np.isfinite(spectrum.sigma) & (spectrum.sigma > 0)))
    f, s = np.asarray(flux, dtype=float)[inwin], spectrum.sigma[inwin]
    mask = np.isfinite(f) & ~in_any(w, [(m.wmin, m.wmax) for m in masks])
    if sigma_known:
        mask &= np.isfinite(s) & (s > 0)
    return Prepared(w, f, s, mask, float(np.median(np.diff(np.log(w)))) * C_KMS, sigma_known, window)


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

    def at(self, wave: np.ndarray, v: float | np.ndarray = 0.) -> np.ndarray:
        """
        Template flux at observed pixels for radial velocities v (scalar → 1D, array → (len(v), len(wave)))

        The shifted template is averaged over each observed pixel (edges half-way between centres in ln λ, and
        extrapolated by half a pixel at the two ends, which makes the first and last pixels very slightly
        asymmetric; negligible), matching pixel-integrated data. A single pixel is point-sampled.
        """
        lnw = np.log(np.asarray(wave, dtype=float))
        v = np.asarray(v, dtype=float)
        shift = np.atleast_1d(np.log(broadening.doppler_factor(v)))
        idx = np.arange(len(self.flux), dtype=float)
        if len(lnw) < 2:
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
    Broaden a rest-frame (vacuum) template spectrum to R (and vsini) on a uniform ln λ grid of step `dv` [km/s;
    default c/(10 R)] covering wave_range ± pad_kms; its own `resolution` (None = infinite) is removed in quadrature
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


def candidates(cfg: Any, opts: Mapping[str, Any]) -> list[Candidate]:
    """
    Template candidates from the method options or the config

    - opts['template']: a rest-frame Spectrum, or a (wave, flux) tuple, used alone;
    - opts['templates']: a list of such;
    - otherwise every configured grid (templates.grid: a name or a list) is opened and its points within the
      ranges become candidates: teff within teff_range (default teff_prior ± teff_window), log g within
      logg_range (default all), and met, alpha and extra axes within their ranges or held at the value nearest
      their prior/default (templates.met_prior, alpha_prior; extra-axis ranges in templates.ranges). With no
      point in the ranges, the point nearest teff_prior (and logg_prior, else 5.0 dex) is used (WARNING).
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
    t = cfg.templates
    teff_range = t.teff_range
    if teff_range is None and t.teff_prior is not None:
        teff_range = (t.teff_prior - opts['teff_window'], t.teff_prior + opts['teff_window'])
    priors = {k: getattr(t, f'{k}_prior') for k in ('met', 'alpha') if getattr(t, f'{k}_prior') is not None}
    out = []
    for name in grid_names(t):
        grid = open_grid(t, name)
        extra = {k: tuple(v) for k, v in t.ranges.items() if k in grid.axes}
        points = grid.select(teff_range=teff_range, logg_range=t.logg_range, met_range=t.met_range,
                             alpha_range=t.alpha_range, ranges=extra, priors=priors)
        if not points:
            if t.teff_prior is None:
                raise ValueError(f'{grid.name}: no grid points in the template ranges; widen them or set '
                                 'templates.teff_prior')
            points = [grid.nearest(t.teff_prior, t.logg_prior or 5., **priors)]
            log.warning('%s: no grid points in the prior ranges: using %s', grid.name, points[0])
        out += [Candidate(f'{grid.name} ' + '/'.join(f'{v:g}' for v in p), grid=grid, params=p) for p in points]
    return out


def open_grid(tcfg: Any, name: str | None = None):
    """
    The TemplateGrid for one configured grid (``tcfg`` is the TemplatesConfig): its local directory, else the
    downloaded grid in the cache. `local_dir` may be a directory (used for the first grid) or a mapping
    grid → directory; a `medium` given in the config overrides the first grid's declared medium.
    """
    names = grid_names(tcfg)
    name = name or names[0]
    local = tcfg.local_dir
    directory = local.get(name) if isinstance(local, Mapping) else (local if name == names[0] else None)
    if not directory:
        cached = T.cache_dir() / name
        if not cached.is_dir():
            raise ValueError(f'no templates for grid {name}: set templates.local_dir, or run "rvfitter templates '
                             f'download --grid {name}", or give the method option "template"')
        directory = cached
    meta = None
    if tcfg.medium and name == names[0]:
        base = T._read_metadata(Path(directory)) or T.KNOWN_GRIDS.get(name)
        if base is None:
            raise ValueError(f'unknown grid {name!r} and no metadata in {directory}')
        meta = replace(base, medium=tcfg.medium)
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
    lnw = np.log(spec.wave)
    return Template(float(lnw[0]), float(lnw[1] - lnw[0]), np.asarray(spec.flux), cand.label,
                    {**spec.meta, **cand.param_dict})


def choose_template(method: str, cands: list[Candidate], score: Callable[[Candidate], float], cfg: Any,
                    opts: Mapping[str, Any], resolution: float, wave_range: tuple[float, float], n_pix: int,
                    executor: Executor | None, vsini: float = 0.) -> tuple[Template, int, dict]:
    """
    The candidate with the lowest `score` (χ²; scored only when there are several), broadened; its grid flags
    (outside the grid, in a hole, at a parameter edge) and the template diagnostics shared by every window
    """
    if len(cands) > 1:
        scores = pmap(score, cands, executor)
        best = cands[int(np.argmin(scores))]
        log.info('%s: template %s chosen by χ² from %d candidates', method, best.label, len(cands))
    else:
        best, scores = cands[0], [np.nan]
    tpl = build_template(best, resolution, wave_range, vsini=vsini, pad_kms=opts['pad_kms'])
    summary = template_summary(method, cands, scores, best, n_pix, cfg.templates)
    flag = ((FLAG_OUTSIDE_GRID if tpl.params.get('outside_grid') else 0)
            | (FLAG_GRID_HOLE if tpl.params.get('grid_hole') else 0)
            | (FLAG_PARAM_EDGE if summary['template_at_edge'] else 0))
    diag = dict(template_summary=summary, **{k: v for k, v in summary.items() if k.startswith('template_')},
                template=tpl.label, template_params=tpl.params,
                template_scores=dict(zip([c.label for c in cands], map(float, scores), strict=False)))
    return tpl, flag, diag


def pmap(fn: Callable, items: Sequence, executor: Executor | None = None) -> list:
    """Map fn over items in order, serially or with an executor; results never depend on which"""
    if executor is None or len(items) < 2:
        return [fn(x) for x in items]
    return list(executor.map(fn, items))


def child_seeds(rng: np.random.Generator, n: int) -> list[np.random.SeedSequence]:
    """n independent seed sequences derived from rng (deterministic for a given rng state)"""
    entropy = [int(x) for x in rng.integers(0, 2 ** 63 - 1, size=4)]
    return np.random.SeedSequence(entropy).spawn(n)


def run_chunks(fn: Callable, n_total: int, seeds: Sequence, args: tuple, executor: Executor | None) -> np.ndarray:
    """``n_total`` trials split over ``len(seeds)`` tasks ``fn((seed, n, *args)) -> list``, concatenated in order"""
    n_each = np.diff(np.linspace(0, n_total, len(seeds) + 1).astype(int))
    tasks = [(seeds[j], int(n), *args) for j, n in enumerate(n_each) if n > 0]
    return np.concatenate([np.asarray(x, float) for x in pmap(fn, tasks, executor)])


def window_measurement(method: str, window: Region, rv: float, err: float, flag: int, chi2_red: float,
                       sigma_known: bool, diag: dict) -> Measurement:
    """One window's measurement: fit_failed for a non-finite error, poor_fit for χ²/ν > 3 with known errors"""
    if not np.isfinite(err):
        flag |= FLAG_FIT_FAILED
    if np.isfinite(chi2_red) and chi2_red > 3. and sigma_known:
        flag |= FLAG_POOR_FIT
    m = Measurement(method, window.name, float(rv), float(err), int(flag), diag)
    log.debug('%s %s: %.2f ± %.2f km/s (χ²/ν %.2f, flag %d)', method, window.name, m.rv, m.rv_err, chi2_red, flag)
    return m


def segment(windows: Sequence[Region], resolution: float, n_res: float | None) -> list[Region]:
    """
    Split windows longer than n_res resolution elements (None or ≤ 0: never) into near-equal segments named
    "<name> [a-b]"

    Each segment is a separate measurement, so the segment-to-segment scatter enters the random-effects
    combinations, and a region that disagrees (e.g. telluric residuals or a badly matched molecular band) is
    visible instead of being averaged into one number.
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
