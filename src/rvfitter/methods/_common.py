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
            flux: np.ndarray | None = None) -> Prepared:
    """
    Resample the part of a spectrum in `window` onto a uniform ln λ grid at (by default) its native sampling

    Parameters
    ----------
    spectrum
        The observed spectrum
    window
        The fitting window
    masks
        Intervals to exclude (e.g. telluric bands)
    dv
        Grid step [km/s]; default the median native step within the window
    flux
        Replacement flux on the spectrum's pixels (used by Monte Carlo trials)

    Returns
    -------
    prepared
        The resampled data; pixels outside the window, in a mask, or non-finite are masked out
    """
    flux = spectrum.flux if flux is None else flux
    inwin = (spectrum.wave >= window.wmin) & (spectrum.wave <= window.wmax)
    if inwin.sum() < 4:
        raise ValueError(f'window {window.name} covers fewer than 4 pixels')
    w = spectrum.wave[inwin]
    if dv is None:
        dv = float(np.median(np.diff(np.log(w)))) * C_KMS
    grid = broadening.log_wave_grid(w[0], w[-1], dv)
    f = broadening.rebin(spectrum.wave, flux, grid)
    sigma_known = bool(np.any(np.isfinite(spectrum.sigma) & (spectrum.sigma > 0)))
    s = broadening._rebin_sigma(spectrum.wave, spectrum.sigma, grid) if sigma_known else np.full(len(grid), np.nan)
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

    def at(self, wave: np.ndarray, v: float | np.ndarray = 0.) -> np.ndarray:
        """
        Template flux at observed wavelengths for radial velocities v (scalar → 1D, array → (len(v), len(wave)))
        """
        lnw = np.log(wave)
        v = np.asarray(v, dtype=float)
        shift = np.log(broadening.doppler_factor(v))
        x = ((lnw[None, :] - np.atleast_1d(shift)[:, None]) - self.lnw0) / self.dlnw
        out = np.interp(x.ravel(), np.arange(len(self.flux), dtype=float), self.flux, left=np.nan, right=np.nan)
        out = out.reshape(x.shape)
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
    """A template candidate: either a given spectrum or a grid point"""
    label: str
    spectrum: Spectrum | None = None
    grid: Any = None
    params: tuple[float, float, float] | None = None


def candidates(cfg: Any, opts: Mapping[str, Any]) -> list[Candidate]:
    """
    Template candidates from the method options or the config

    - opts['template']: a rest-frame Spectrum, or a (wave, flux) tuple, used alone;
    - opts['templates']: a list of such;
    - otherwise cfg.templates (local_dir, grid, teff_prior, teff/logg/met ranges) opens a TemplateGrid and the
      candidates are its points within the ranges (default teff_prior ± 300 K).
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
    grid = open_grid(tcfg)
    teff = tcfg.get('teff_prior')
    teff_range = tcfg.get('teff_range')
    if teff_range is None and teff is not None:
        teff_range = (teff - opts.get('teff_window', 300.), teff + opts.get('teff_window', 300.))
    points = grid.select(teff_range=teff_range, logg_range=tcfg.get('logg_range'), met_range=tcfg.get('met_range'))
    if not points:
        points = [grid.nearest(teff or 2000., tcfg.get('logg_prior') or 5.)]
        log.warning('no grid points in the prior ranges: using %s', points[0])
    return [Candidate(f'{grid.name} {t:g}/{g:g}/{m:+g}', grid=grid, params=(t, g, m)) for t, g, m in points]


def open_grid(tcfg: Mapping[str, Any]):
    """
    The TemplateGrid named by a templates config section: `local_dir`, else the downloaded grid in the cache

    A `medium` given in the config overrides the grid's declared medium (and is then verified like any other).
    """
    from dataclasses import replace as _replace
    from pathlib import Path

    from .. import templates as T
    name = tcfg.get('grid') or 'bt-settl-cifist'
    directory = tcfg.get('local_dir')
    if not directory:
        cached = T.cache_dir() / name
        if not cached.is_dir():
            raise ValueError(f'no templates: set templates.local_dir, or run "rvfitter templates download --grid '
                             f'{name}", or give the method option "template"')
        directory = cached
    meta = None
    if tcfg.get('medium'):
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
    spec = cand.grid.broadened(*cand.params, resolution=resolution, wave_range=(lo, hi), vsini=vsini)
    t = template_from_grid_spec(spec, cand.label)
    t.params.update(teff=cand.params[0], logg=cand.params[1], met=cand.params[2])
    return t


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
    """Combine a method's per-region measurements (Birge), ignoring seriously flagged ones and gross outliers"""
    from ..combine import combine_regions
    if outlier_k:
        reject_outliers(measurements, outlier_k)
    bad = SERIOUS | FLAG_OUTLIER
    good = [m for m in measurements if not (m.flag & bad) and np.isfinite(m.rv) and np.isfinite(m.rv_err)]
    rv, err = combine_regions(good)
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
