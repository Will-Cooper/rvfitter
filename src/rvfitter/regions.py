"""
Automatic, band-agnostic selection of line, continuum and fitting regions for each method.

No wavelength is written here: every window comes from the feature catalogue (`rvfitter.features`) intersected with
the spectrum.
The selection follows five steps:

1. Intersect with the coverage, minus NaN gaps, chip gaps (a step > 5x the median) and the edges.
2. Filter by resolving power (R >= r_min), signal-to-noise ratio (a robust ``der_snr`` estimate) and Teff prior.
3. Mask telluric bands (strictness is configurable and scaled with airmass).
4. Rank by the Bouchy et al. (2001) RV information content, computed on the smoothed observed spectrum.
5. Build windows scaled to the resolution element.
"""
import dataclasses
import logging
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .constants import C_KMS
from .features import Feature, features
from .types import Region, Spectrum

__all__ = ['SelectOptions', 'select', 'der_snr', 'rv_precision', 'LINE_METHODS', 'TELLURIC_METHODS']

log = logging.getLogger(__name__)

LINE_METHODS = frozenset({'linecentre'})
TELLURIC_METHODS = frozenset({'telluric_ref'})
_FWHM_TO_SIGMA = 1. / (2. * math.sqrt(2. * math.log(2.)))
# telluric strictness -> (minimum effective depth masked, padding [FWHM_inst], line-overlap half-width mode)
_STRICTNESS = {'strict': (0., 2.), 'standard': (0.15, 0.), 'loose': (0.5, 0.), 'off': None}


@dataclass
class SelectOptions:
    """
    Tunable knobs of `select`; the defaults were chosen by experiment on synthetic spectra

    Attributes
    ----------
    snr_min
        Minimum robust SNR per pixel in the neighbourhood of a window
    k_line, k_core
        Line-window half-width in units of the instrumental FWHM: ``k_line`` (plus the intrinsic width) for normal
        lines, ``k_core`` (and no intrinsic width) for 'core-only window' lines
    intrinsic_kms
        Intrinsic half-width [km/s] added to normal line windows (rotation plus thermal and pressure broadening)
    cont_fwhm, cont_gap_fwhm
        Width of each continuum window and its gap from the line window, in FWHM
    edge_fwhm
        Trim applied at the ends of every usable segment, in FWHM
    telluric
        'strict' (mask every band, pad by 2 FWHM, reject lines whose full window overlaps a band), 'standard'
        (bands whose effective depth >= 0.15; reject lines whose core overlaps), 'loose' (depth >= 0.5, reject
        lines whose centre is inside) or 'off'
    min_span_resel, max_span_resel
        Minimum and maximum length of a span for ccf/template_fit/forward, in resolution elements
    snr_window_fwhm
        Half-width of the neighbourhood used to estimate the SNR, in FWHM
    smooth_fwhm
        Gaussian smoothing (sigma = smooth_fwhm x FWHM / 2.355) applied before the information content is computed
    gap_factor, nan_gap_pix
        A step larger than ``gap_factor`` x the median step is a chip gap; NaN runs of at least ``nan_gap_pix``
        pixels are gaps
    exclude_species
        Line species never chosen automatically (H-alpha is chromospheric)
    max_lines
        Keep only the best N lines per method (None = all)
    """
    snr_min: float = 3.
    k_line: float = 2.5
    k_core: float = 1.
    intrinsic_kms: float = 20.
    cont_fwhm: float = 3.
    cont_gap_fwhm: float = 0.5
    edge_fwhm: float = 2.
    telluric: str = 'standard'
    min_span_resel: float = 30.
    max_span_resel: float = 2000.
    snr_window_fwhm: float = 10.
    smooth_fwhm: float = 0.5
    gap_factor: float = 5.
    nan_gap_pix: int = 3
    exclude_species: tuple[str, ...] = ('H I',)
    max_lines: int | None = None

    @classmethod
    def from_config(cls, cfg: Any = None) -> 'SelectOptions':
        """
        Build options from a config section (attribute object or mapping)

        Config names are mapped as ``min_snr`` -> ``snr_min`` and ``telluric_strictness`` -> ``telluric``;
        any field of this class may also be given by its own name. Missing or None entries keep the defaults.
        """
        if cfg is None:
            return cls()
        alias = {'min_snr': 'snr_min', 'telluric_strictness': 'telluric'}
        names = {f.name for f in dataclasses.fields(cls)}
        if isinstance(cfg, Mapping):
            def get(k):
                return cfg.get(k)
        else:
            def get(k):
                return getattr(cfg, k, None)
        kwargs = {}
        for key in [*names, *alias]:
            val = get(key)
            if val is not None:
                kwargs[alias.get(key, key)] = val
        return cls(**kwargs)


# ---------------------------------------------------------------------------------------------------- statistics
def der_snr(flux: np.ndarray) -> float:
    """
    Robust SNR per pixel from the DER_SNR algorithm (Stoehr et al. 2008)

    The noise is the median of |2 f_i - f_(i-2) - f_(i+2)| scaled by 1.482602/sqrt(6), so smooth spectral structure
    cancels. Returns NaN with fewer than five finite pixels, and 0 for a non-positive median.
    """
    f = np.asarray(flux, dtype=float)
    f = f[np.isfinite(f)]
    if f.size < 5:
        return float('nan')
    signal = float(np.median(f))
    noise = 1.482602 / math.sqrt(6.) * float(np.median(np.abs(2. * f[2:-2] - f[:-4] - f[4:])))
    if signal <= 0.:
        return 0.
    return math.inf if noise == 0. else signal / noise


def rv_precision(wave: np.ndarray, flux_smooth: np.ndarray, snr: float) -> float:
    """
    Photon-noise RV precision [km/s] of a window (Bouchy et al. 2001)

    The per-pixel weight is W_i = (lambda_i dA_i/dlambda)^2 / var_i with photon noise var_i = A_i / snr^2 x A_c,
    where A_c is the median flux (so snr is the continuum SNR per pixel); sigma_RV = c / sqrt(sum W_i).
    Returns inf if the window has fewer than three usable pixels or no gradient.
    """
    ok = np.isfinite(flux_smooth) & (flux_smooth > 0.05 * np.nanmedian(flux_smooth))
    if ok.sum() < 3 or not np.isfinite(snr) or snr <= 0.:
        return math.inf
    grad = np.gradient(flux_smooth, wave)
    cont = float(np.nanmedian(flux_smooth[ok]))
    w = (wave[ok] * grad[ok]) ** 2 / (flux_smooth[ok] * cont) * snr ** 2
    total = float(np.sum(w))
    return C_KMS / math.sqrt(total) if total > 0. else math.inf


# ---------------------------------------------------------------------------------------------------- helpers
class _Context:
    """Everything derived once per spectrum"""

    def __init__(self, spec: Spectrum, opts: SelectOptions):
        self.spec, self.opts = spec, opts
        self.wave, self.flux = spec.wave, spec.flux
        step = float(np.median(np.diff(self.wave)))
        if spec.resolution:
            self.R = float(spec.resolution)
            self.r_note = 'given'
        else:
            # assume the usual ~2.5 pixels per resolution element
            self.R = float(np.median(self.wave) / (2.5 * step))
            self.r_note = 'ESTIMATED from the pixel scale (resolution unknown)'
            log.warning('Spectrum has no resolution; assuming R~%.0f from the pixel scale', self.R)
        sigma_pix = max(opts.smooth_fwhm * float(np.median(self.wave)) / self.R / step * _FWHM_TO_SIGMA, 0.5)
        good = np.isfinite(self.flux)
        filled = np.interp(self.wave, self.wave[good], self.flux[good]) if good.sum() >= 2 else np.full_like(
            self.flux, np.nan)
        self.smooth = np.where(good, gaussian_filter1d(filled, sigma_pix, mode='nearest'), np.nan)
        self.airmass = _airmass(spec)
        self.segments = self._segments()

    def fwhm(self, w: float) -> float:
        """Instrumental FWHM [Å] at ``w``"""
        return w / self.R

    def _segments(self) -> list[tuple[float, float]]:
        o, w = self.opts, self.wave
        good = np.isfinite(self.flux)
        steps = np.diff(w)
        big = steps > o.gap_factor * np.median(steps)
        # break points: after pixel i where there is a chip gap, or at NaN runs of >= nan_gap_pix pixels
        cut = np.zeros(len(w), dtype=bool)  # cut[i]: a break between i and i+1
        cut[:-1] |= big
        i = 0
        while i < len(w):
            if good[i]:
                i += 1
                continue
            j = i
            while j < len(w) and not good[j]:
                j += 1
            if j - i >= o.nan_gap_pix:  # a long NaN run is a gap; shorter runs are simply left out
                cut[max(i - 1, 0):j] = True
            i = j
        segs: list[tuple[float, float]] = []
        start = 0
        for k in range(len(w)):
            if cut[k] or k == len(w) - 1:
                idx = np.arange(start, k + 1)
                idx = idx[good[idx]]
                if idx.size >= 5:
                    lo, hi = w[idx[0]], w[idx[-1]]
                    # a run of NaN shorter than the gap threshold does not split; a long run's pixels are not good
                    lo += o.edge_fwhm * self.fwhm(lo)
                    hi -= o.edge_fwhm * self.fwhm(hi)
                    if hi > lo:
                        segs.append((float(lo), float(hi)))
                start = k + 1
        return segs

    def covered(self, a: float, b: float) -> bool:
        return any(lo <= a and b <= hi for lo, hi in self.segments)

    def snr(self, a: float, b: float) -> float:
        """Robust SNR in the neighbourhood [a, b] widened by the SNR window"""
        pad = self.opts.snr_window_fwhm * self.fwhm(0.5 * (a + b))
        m = (self.wave >= a - pad) & (self.wave <= b + pad)
        return der_snr(self.flux[m])

    def sigma_rv(self, a: float, b: float, snr: float) -> float:
        m = (self.wave >= a) & (self.wave <= b)
        return rv_precision(self.wave[m], self.smooth[m], snr)


def _airmass(spec: Spectrum) -> float:
    try:
        x = float(spec.meta.get('airmass', 1.))
    except (TypeError, ValueError):
        return 1.
    return x if math.isfinite(x) and x >= 1. else 1.


def _tellurics(ctx: _Context, feats: Sequence[Feature]) -> list[tuple[float, float, str]]:
    """Masked telluric intervals within the coverage, for the configured strictness"""
    rule = _STRICTNESS.get(ctx.opts.telluric)
    if ctx.opts.telluric not in _STRICTNESS:
        raise ValueError(f'telluric must be one of {sorted(_STRICTNESS)}, not {ctx.opts.telluric!r}')
    if rule is None:
        return []
    threshold, pad_fwhm = rule
    out = []
    for f in feats:
        if f.kind != 'telluric':
            continue
        eff = 1. - (1. - min(f.depth, 0.999)) ** ctx.airmass  # deeper at higher airmass
        if eff < threshold:
            continue
        lo = f.wmin - pad_fwhm * ctx.fwhm(f.wmin)
        hi = f.wmax + pad_fwhm * ctx.fwhm(f.wmax)
        if any(hi > s_lo and lo < s_hi for s_lo, s_hi in ctx.segments):
            out.append((lo, hi, f.name))
    return out


def _overlaps(a: float, b: float, intervals: Iterable[tuple[float, float, str]]) -> str | None:
    for lo, hi, name in intervals:
        if b > lo and a < hi:
            return name
    return None


def _line_window(ctx: _Context, f: Feature) -> tuple[float, float]:
    o = ctx.opts
    fwhm = ctx.fwhm(f.wave_vac)
    if f.core_only:
        half = o.k_core * fwhm
    else:
        half = o.k_line * fwhm + f.wave_vac * o.intrinsic_kms / C_KMS
    return f.wave_vac - half, f.wave_vac + half


def _continuum(ctx: _Context, wmin: float, wmax: float, avoid: Sequence[tuple[float, float, str]]
               ) -> tuple[tuple[float, float], ...]:
    """Feature-free continuum windows either side of a line window"""
    o = ctx.opts
    fw = ctx.fwhm(0.5 * (wmin + wmax))
    width, gap = o.cont_fwhm * fw, o.cont_gap_fwhm * fw
    out = []
    for side in (-1, 1):
        for shift in range(4):
            if side < 0:
                hi = wmin - gap - shift * width
                lo = hi - width
            else:
                lo = wmax + gap + shift * width
                hi = lo + width
            if ctx.covered(lo, hi) and _overlaps(lo, hi, avoid) is None:
                out.append((float(lo), float(hi)))
                break
    return tuple(sorted(out))


def _feature_list(feats: Sequence[Feature], lo: float, hi: float, teff: float | None, ctx: _Context) -> list[str]:
    """Names of non-telluric catalogue features passing the R and Teff filters that fall in [lo, hi]"""
    names = []
    for f in feats:
        if f.kind == 'telluric' or f.species in ctx.opts.exclude_species:
            continue
        if ctx.R < f.r_min or not f.teff_ok(teff):
            continue
        a, b = f.span
        if b >= lo and a <= hi:
            names.append(f.name)
    return names


def _default_kind(method: str) -> str:
    return 'line' if method in LINE_METHODS else ('telluric' if method in TELLURIC_METHODS else 'band')


def _avoid_windows(ctx: _Context, masks) -> list[tuple[float, float, str]]:
    """Windows (all catalogue lines plus masked tellurics) that continuum windows must avoid"""
    out = []
    for f in features():
        if f.kind == 'line':
            a, b = _line_window(ctx, f)
            out.append((a - ctx.fwhm(f.wave_vac), b + ctx.fwhm(f.wave_vac), f.name))
    return out + list(masks)


def _make_override(item: Any, method: str, ctx: _Context, masks) -> Region | None:
    """
    Turn one override entry (Region, mapping or (wmin, wmax)) into a Region clipped to the coverage

    A line override without a centre takes it from the catalogue entry of the same name; failing that a
    `ValueError` is raised. A line override without continuum windows gets feature-free ones.
    """
    kind = _default_kind(method)
    if isinstance(item, Region):
        d = dict(name=item.name, kind=item.kind, wmin=item.wmin, wmax=item.wmax, centre=item.centre,
                 continuum=item.continuum)
    elif isinstance(item, Mapping):
        d = dict(item)
        d.setdefault('kind', kind)
        if 'name' not in d and 'wmin' in d and 'wmax' in d:
            d['name'] = f"{method} {d['wmin']:.0f}-{d['wmax']:.0f}"
    else:
        wmin, wmax = (float(x) for x in item)
        d = dict(name=f'{method} {wmin:.0f}-{wmax:.0f}', kind=kind, wmin=wmin, wmax=wmax)
    for key in ('name', 'wmin', 'wmax'):
        if key not in d:
            raise ValueError(f"override for {method!r} lacks {key!r}: {d}")
    extra = set(d) - {'name', 'kind', 'wmin', 'wmax', 'centre', 'continuum', 'reason', 'remove'}
    if extra:
        raise ValueError(f'override for {method!r} region {d["name"]!r} has unknown keys {sorted(extra)}')
    centre = d.get('centre')
    if d['kind'] == 'line' and centre is None:
        match = [f for f in features() if f.kind == 'line' and f.name == d['name']]
        if not match:
            raise ValueError(f"line override {d['name']!r} for {method!r} needs a 'centre' (it is not a catalogue "
                             'line name)')
        centre = match[0].wave_vac
    lo, hi = max(float(d['wmin']), ctx.wave[0]), min(float(d['wmax']), ctx.wave[-1])
    if hi <= lo:
        log.warning('Override region %s (%.1f-%.1f A) lies outside the coverage (%.1f-%.1f A); dropped',
                    d['name'], d['wmin'], d['wmax'], ctx.wave[0], ctx.wave[-1])
        return None
    if (lo, hi) != (float(d['wmin']), float(d['wmax'])):
        log.warning('Override region %s clipped to the coverage: %.1f-%.1f A', d['name'], lo, hi)
    cont = tuple((float(c[0]), float(c[1])) for c in d.get('continuum', ()))
    if d['kind'] == 'line' and not cont:
        cont = _continuum(ctx, lo, hi, [w for w in _avoid_windows(ctx, masks) if w[2] != d['name']])
    return Region(str(d['name']), str(d['kind']), float(lo), float(hi), None if centre is None else float(centre),
                  cont, 'user override')


def _apply_overrides(auto: list[Region], entries: Sequence[Any], method: str, ctx: _Context, masks
                     ) -> list[Region]:
    """Apply override entries on top of the automatic regions: replace by name, add, or remove"""
    regs = list(auto)
    for item in entries:
        if isinstance(item, Mapping) and item.get('remove'):
            name = item.get('name')
            keep = [r for r in regs if r.name != name]
            if len(keep) == len(regs):
                log.warning('%s: cannot remove region %r (not selected)', method, name)
            regs = keep
            continue
        new = _make_override(item, method, ctx, masks)
        if new is None:
            continue
        idx = next((i for i, r in enumerate(regs) if r.name == new.name), None)
        if idx is None:
            regs.append(new)
        else:
            regs[idx] = new
    return regs


# ---------------------------------------------------------------------------------------------------- per method
def _select_lines(ctx: _Context, feats: Sequence[Feature], masks, teff, drops: Counter) -> list[Region]:
    o = ctx.opts
    lines = [f for f in feats if f.kind == 'line']
    # windows of all catalogue lines inside the coverage, to keep continuum windows feature-free
    all_windows = []
    for f in lines:
        a, b = _line_window(ctx, f)
        all_windows.append((a - ctx.fwhm(f.wave_vac), b + ctx.fwhm(f.wave_vac), f.name))
    avoid = all_windows + list(masks)
    mode = o.telluric
    cands = []
    for f in lines:
        a, b = _line_window(ctx, f)
        why = cat = None
        if f.species in o.exclude_species:
            why = cat = 'excluded species'
        elif not f.teff_ok(teff):
            why = cat = 'Teff prior'
        elif ctx.R < f.r_min:
            why = cat = 'R < r_min'
        elif not ctx.covered(a, b):
            why = cat = 'outside coverage'
        else:
            half = {'strict': b - f.wave_vac, 'standard': o.k_core * ctx.fwhm(f.wave_vac), 'loose': 0.,
                    'off': 0.}[mode]
            hit = _overlaps(f.wave_vac - half, f.wave_vac + half, masks)
            if hit:
                why, cat = f'telluric ({hit})', 'telluric'
        if why is None:
            snr = ctx.snr(a, b)
            if (math.isnan(snr) or snr < o.snr_min):
                why, cat = f'SNR {snr:.1f} < {o.snr_min:g}', 'SNR'
            else:
                cands.append((f, a, b, snr))
        if why:
            drops[cat] += 1
            log.debug('line %s dropped: %s', f.name, why)
    ranked = []
    for f, a, b, snr in cands:
        sig = ctx.sigma_rv(a, b, snr)
        ranked.append((sig, f, a, b, snr))
    ranked.sort(key=lambda t: t[0])
    if o.max_lines is not None:
        ranked = ranked[:o.max_lines]
    regions = []
    for sig, f, a, b, snr in ranked:
        cont = _continuum(ctx, a, b, [w for w in avoid if w[2] != f.name])
        width = 'core-only' if f.core_only else f'+-{o.k_line:g} FWHM + {o.intrinsic_kms:g} km/s'
        reason = (f'{f.species} line, Teff {f.teff_min:.0f}-{f.teff_max:.0f} K, R {ctx.R:.0f} >= {f.r_min:.0f}; '
                  f'window {width}; SNR {snr:.1f}; sigma_RV~{sig:.2f} km/s (Bouchy)'
                  f'; {len(cont)} continuum window(s)')
        regions.append(Region(f.name, 'line', float(a), float(b), float(f.wave_vac), cont, reason))
        log.debug('line %s selected: %s', f.name, reason)
    return regions


def _split_spans(ctx: _Context, masks) -> list[tuple[float, float]]:
    """Coverage minus masked tellurics, as spans of at least min_span_resel resolution elements, chunked"""
    o = ctx.opts
    pieces = []
    for lo, hi in ctx.segments:
        cuts = sorted((max(a, lo), min(b, hi)) for a, b, _ in masks if b > lo and a < hi)
        start = lo
        for a, b in cuts:
            if a > start:
                pieces.append((start, a))
            start = max(start, b)
        if hi > start:
            pieces.append((start, hi))
    spans = []
    for lo, hi in pieces:
        n = ctx.R * math.log(hi / lo)
        if n < o.min_span_resel:
            continue
        k = max(1, math.ceil(n / o.max_span_resel))
        edges = np.exp(np.linspace(math.log(lo), math.log(hi), k + 1))
        edges[0], edges[-1] = lo, hi  # no rounding error at the ends
        spans.extend((float(edges[i]), float(edges[i + 1])) for i in range(k))
    return spans


def _select_spans(ctx: _Context, feats: Sequence[Feature], masks, teff, method: str, drops: Counter) -> list[Region]:
    kind = 'ccf' if method == 'ccf' else 'band'
    out = []
    for lo, hi in _split_spans(ctx, masks):
        snr = ctx.snr(lo, hi)
        if (math.isnan(snr) or snr < ctx.opts.snr_min):
            drops['SNR'] += 1
            log.debug('span %.0f-%.0f dropped: SNR %.1f', lo, hi, snr)
            continue
        sig = ctx.sigma_rv(lo, hi, snr)
        names = _feature_list(feats, lo, hi, teff, ctx)
        n = ctx.R * math.log(hi / lo)
        shown = ', '.join(names[:6]) + (f' (+{len(names) - 6} more)' if len(names) > 6 else '')
        reason = (f'non-telluric span of {n:.0f} resolution elements, SNR {snr:.1f}, sigma_RV~{sig:.2f} km/s '
                  f'(Bouchy); features: {shown or "none catalogued"}')
        out.append((sig, Region(f'{kind.upper() if kind == "ccf" else "BAND"} {lo:.0f}-{hi:.0f}', kind, lo, hi,
                                None, (), reason)))
    out.sort(key=lambda t: t[0])
    return [r for _, r in out]


def _select_telluric(ctx: _Context, feats: Sequence[Feature], drops: Counter) -> list[Region]:
    out = []
    for f in feats:
        if f.kind != 'telluric':
            continue
        if ctx.R < f.r_min:
            drops['R < r_min'] += 1
            continue
        pieces = [(max(f.wmin, lo), min(f.wmax, hi)) for lo, hi in ctx.segments if f.wmax > lo and f.wmin < hi]
        if not pieces:
            continue
        lo, hi = max(pieces, key=lambda p: p[1] - p[0])
        if ctx.R * math.log(hi / lo) < ctx.opts.min_span_resel / 3.:
            drops['too short'] += 1
            continue
        snr = ctx.snr(lo, hi)
        if (math.isnan(snr) or snr < ctx.opts.snr_min):
            drops['SNR'] += 1
            continue
        sig = ctx.sigma_rv(lo, hi, snr)
        part = '' if (lo, hi) == (f.wmin, f.wmax) else ' (partly covered)'
        reason = (f'telluric {f.species} band as wavelength reference{part}; R {ctx.R:.0f} >= {f.r_min:.0f}; '
                  f'SNR {snr:.1f}')
        out.append((sig, Region(f.name, 'telluric', float(lo), float(hi), None, (), reason)))
    out.sort(key=lambda t: t[0])
    return [r for _, r in out]


# ---------------------------------------------------------------------------------------------------- public
def select(spectrum: Spectrum, *, teff: float | None = None, methods: Iterable[str],
           overrides: Mapping[str, Sequence[Any]] | None = None,
           options: Any = None) -> dict[str, list[Region]]:
    """
    Choose the regions each method should use on this spectrum

    Parameters
    ----------
    spectrum
        The spectrum (vacuum wavelengths). If ``resolution`` is None it is estimated from the pixel scale
    teff
        Teff prior [K]; features whose useful range excludes it are dropped. None disables the prior
    methods
        Method names: 'linecentre' gets line windows with continuum, 'telluric_ref' gets telluric bands, and every
        other ('ccf', 'template_fit', 'forward_mcmc', 'lsd'...) gets merged non-telluric spans
    overrides
        ``{method: [entry, ...]}`` applied on top of the automatic selection of that method. An entry is a
        mapping ``{'name', 'kind', 'wmin', 'wmax'[, 'centre', 'continuum']}`` (adds a region, or replaces the one
        of the same name), ``{'name', 'remove': True}`` (drops it), a `Region` or a ``(wmin, wmax)`` tuple.
        Entries are clipped to the coverage (wholly uncovered ones are dropped with a warning). A 'linecentre'
        entry without a centre takes it from the catalogue line of the same name, else a `ValueError` is raised
    options
        `SelectOptions`, a mapping or object with config-style names (``min_snr``, ``telluric_strictness``;
        see `SelectOptions.from_config`), or None for the defaults

    Returns
    -------
    regions
        ``{method: [Region, ...]}``, best (smallest photon-noise sigma_RV) first. A method with nothing usable
        gets an empty list (a warning is logged); this function never raises for missing bands.
    """
    opts = options if isinstance(options, SelectOptions) else SelectOptions.from_config(options)
    methods = list(methods)
    overrides = dict(overrides or {})
    ctx = _Context(spectrum, opts)
    feats = features()
    result: dict[str, list[Region]] = {}
    if not ctx.segments:
        log.warning('No usable coverage in %s (all NaN or too short); no regions selected',
                    spectrum.source or 'spectrum')
        return {m: [] for m in methods}
    masks = _tellurics(ctx, feats)
    cov = ', '.join(f'{a:.0f}-{b:.0f}' for a, b in ctx.segments)
    log.info('Region selection: usable coverage %s A, R=%.0f (%s), telluric=%s (%d band(s) masked), Teff prior %s',
             cov, ctx.R, ctx.r_note, opts.telluric, len(masks), 'none' if teff is None else f'{teff:.0f} K')
    for name, lo, hi in ((m[2], m[0], m[1]) for m in masks):
        log.debug('telluric mask %s: %.1f-%.1f A', name, lo, hi)
    for method in methods:
        drops: Counter = Counter()
        if method in LINE_METHODS:
            regs = _select_lines(ctx, feats, masks, teff, drops)
        elif method in TELLURIC_METHODS:
            regs = _select_telluric(ctx, feats, drops)
        else:
            regs = _select_spans(ctx, feats, masks, teff, method, drops)
        how = 'auto'
        if overrides.get(method):
            regs = _apply_overrides(regs, overrides[method], method, ctx, masks)
            how = 'auto + overrides'
        result[method] = regs
        if not regs:
            log.warning('%s: no usable regions (%s)', method, dict(drops) or 'none available in the coverage')
        log.info('%s (%s): %d region(s): %s%s', method, how, len(regs), ', '.join(r.name for r in regs[:8]) +
                 (' ...' if len(regs) > 8 else ''), f'; dropped {dict(drops)}' if drops else '')
        for r in regs:
            log.debug('  %s [%s] %.2f-%.2f: %s', r.name, r.kind, r.wmin, r.wmax, r.reason)
    return result
