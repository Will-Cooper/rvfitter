"""
Wavelength zero point from the OH night-sky emission lines.

The sky spectrum extracted alongside the object (e.g. PypeIt `OPT_COUNTS_SKY`) shares the object's wavelength
solution, so its OH lines measure the solution's zero point and its variation with wavelength. The sky is
uniformly illuminated, so it cannot see a star's own offset from slit centring; that constant comes from the
telluric absorption lines (`telluric_zeropoint`).

Method:
1. Work in the observatory frame: any velocity correction already applied to the wavelengths in the header
   (PypeIt `VEL_CORR`) is undone.
2. Subtract a running low percentile from the sky, on its own pixels (no resampling), leaving the emission lines.
3. Build a model sky from the packaged vacuum OH line list (`data/oh_sky_vacuum.csv`: PypeIt's Rousselot
   et al. 2000 model, BSD-3-Clause), with Gaussian lines of the observed sky FWHM, measured by maximising the
   correlation over trial widths.
4. Cut the coverage minus the strong telluric bands (which eat the OH emission) into segments of at most 250 Å,
   so the segments start right after a band (e.g. the O2 A band). In each, correlate the data with the model
   evaluated at the data's own pixels for every trial velocity (2 km/s steps; comparing like with like, since
   resampling coarse pixels biases a few-line segment by several km/s), fit the peak with a parabola, and take
   the Zucker (2003) error with the number of native pixels. Segments with too few lines or a poor correlation are
   rejected.
5. Δv(λ) is local: within each telluric-free span, the piecewise-linear interpolation between its segment
   values, held constant beyond them; it is never interpolated across an unmeasured gap (a wavelength in a gap
   takes the nearer span's edge value), and its error grows with the distance from the nearest node. Node errors
   are scaled by the Birge factor of linear fits within the spans. The spread over reasonable curve models
   (interpolation, linear, constant) of the main span's mean Δv is reported and added to the zero-point error.

Sign: positive Δv means the data's wavelengths are too large (too red). `apply` divides the wavelengths by the
Doppler factor of Δv(λ) (plus an optional constant), which lowers every measured RV by Δv at that wavelength.
"""
import functools
import logging
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import percentile_filter

from .. import broadening
from ..types import Spectrum

__all__ = ['SkyZeroPoint', 'oh_lines', 'model_sky', 'measure', 'apply', 'observatory_frame_factor']

log = logging.getLogger(__name__)

_DV = 2.  # grid step [km/s]


@functools.lru_cache(maxsize=1)
def oh_lines() -> tuple[np.ndarray, np.ndarray]:
    """The packaged vacuum OH line list: (wavelengths [Å], relative amplitudes)"""
    from importlib import resources
    with resources.files('rvfitter').joinpath('data', 'oh_sky_vacuum.csv').open() as fh:
        df = pd.read_csv(fh, comment='#')
    return df['wave_vac'].to_numpy(float), df['amplitude'].to_numpy(float)


def model_sky(wave: np.ndarray, fwhm: float) -> np.ndarray:
    """OH emission model: Gaussian lines of FWHM `fwhm` [Å] at the packaged vacuum positions"""
    lw, la = oh_lines()
    sel = (lw > wave[0] - 5 * fwhm) & (lw < wave[-1] + 5 * fwhm)
    sigma = fwhm * broadening.FWHM_TO_SIGMA
    out = np.zeros(len(wave))
    for c, a in zip(lw[sel], la[sel], strict=True):
        m = np.abs(wave - c) < 6 * sigma
        out[m] += a * np.exp(-0.5 * ((wave[m] - c) / sigma) ** 2)
    return out


def observatory_frame_factor(meta: Any) -> float:
    """
    The factor f that a header-declared velocity correction multiplied into the wavelengths (1 if none)

    PypeIt records it as VEL_CORR (`meta['vel_corr']`); `meta['vel_corr_kms']` is used if only that exists.
    """
    meta = meta or {}
    f = meta.get('vel_corr')
    if f is not None and np.isfinite(f) and f > 0:
        return float(f)
    v = meta.get('vel_corr_kms')
    if v is not None and np.isfinite(v):
        return float(broadening.doppler_factor(v))
    return 1.


def _curve(wave, node_wave, node_dv, node_err, node_span, inflate, span_bounds=None):
    """
    Δv and its error at `wave` from per-span piecewise-linear nodes

    Within a telluric-free span, Δv is interpolated linearly between its nodes and held at the edge values out to
    the span's boundaries (`span_bounds`), so one span's values never set another's. Only inside an unmeasured gap
    between two spans does Δv ramp linearly from one span's boundary value to the next, which keeps the curve
    continuous (and the corrected wavelengths monotonic). The error grows with the distance from the nearest
    node of the span (or the nearer span) at the rate `inflate` [km/s per Å].
    """
    wave = np.atleast_1d(np.asarray(wave, dtype=float))
    node_wave, node_dv = np.asarray(node_wave, float), np.asarray(node_dv, float)
    node_err, node_span = np.asarray(node_err, float), np.asarray(node_span, int)
    spans = np.unique(node_span)
    aug_w, aug_v = [], []
    for k in spans:
        sel = node_span == k
        w_k, v_k = node_wave[sel], node_dv[sel]
        lo, hi = (span_bounds[k] if span_bounds is not None else (w_k.min(), w_k.max()))
        aug_w += [min(lo, w_k.min()), *w_k, max(hi, w_k.max())]
        aug_v += [v_k[0], *v_k, v_k[-1]]
    order = np.argsort(aug_w, kind='stable')
    aug_w, aug_v = np.asarray(aug_w)[order], np.asarray(aug_v)[order]
    keep = np.concatenate([[True], np.diff(aug_w) > 0])
    dv = np.interp(wave, aug_w[keep], aug_v[keep])
    lo = np.array([node_wave[node_span == k].min() for k in spans])
    hi = np.array([node_wave[node_span == k].max() for k in spans])
    dist = np.maximum(lo[None, :] - wave[:, None], 0.) + np.maximum(wave[:, None] - hi[None, :], 0.)
    which = spans[np.argmin(dist, axis=1)]
    err = np.empty(len(wave))
    for k in spans:
        m = which == k
        sel = node_span == k
        base = np.interp(wave[m], node_wave[sel], node_err[sel])
        d = np.maximum(node_wave[sel].min() - wave[m], 0.) + np.maximum(wave[m] - node_wave[sel].max(), 0.)
        err[m] = np.hypot(base, inflate * d)
    return dv, err


@dataclass
class SkyZeroPoint:
    """
    The measured zero-point curve Δv(λ) [km/s]: local, per telluric-free span (see `_curve`)

    `zp_kms`, `slope` and `model_spread` describe the main span (the one with the most nodes): its mean Δv, the
    slope of a weighted linear fit, and the standard deviation over reasonable curve models (interpolation,
    linear, constant) of the mean Δv over that span. `zp_err` combines the mean node error and the model spread.
    """
    node_wave: np.ndarray
    node_dv: np.ndarray
    node_err: np.ndarray
    node_span: np.ndarray
    fwhm: float
    segments: list[dict] = field(default_factory=list)
    frame_factor: float = 1.
    zp_kms: float = np.nan
    zp_mean_err: float = np.nan
    slope: float = np.nan  # km/s per 1000 Å (main span)
    slope_err: float = np.nan
    model_spread: float = np.nan
    inflate: float = 0.
    birge: float = 1.
    main_span: tuple[float, float] = (np.nan, np.nan)
    span_bounds: dict = field(default_factory=dict)  # span id → (wmin, wmax) of its segments
    kind: str = 'sky'

    @property
    def n_used(self) -> int:
        return sum(1 for s in self.segments if s['used'])

    @property
    def zp_err(self) -> float:
        main = (self.node_wave >= self.main_span[0]) & (self.node_wave <= self.main_span[1])
        stat = float(np.mean(self.node_err[main])) if main.any() else float(np.mean(self.node_err))
        spread = self.model_spread if np.isfinite(self.model_spread) else 0.
        return float(np.hypot(stat, spread))

    def dv(self, wave: np.ndarray | float) -> np.ndarray:
        out = _curve(wave, self.node_wave, self.node_dv, self.node_err, self.node_span, self.inflate,
                     self.span_bounds)[0]
        return out if np.ndim(wave) else out[0]

    def dv_err(self, wave: np.ndarray | float) -> np.ndarray:
        return _curve(wave, self.node_wave, self.node_dv, self.node_err, self.node_span, self.inflate,
                      self.span_bounds)[1]

    def as_dict(self) -> dict:
        return dict(zp_sky_kms=self.zp_kms, zp_sky_err=self.zp_err, zp_sky_slope=self.slope,
                    zp_sky_slope_err=self.slope_err, zp_sky_model_spread=self.model_spread,
                    node_wave=[float(x) for x in self.node_wave], node_dv=[float(x) for x in self.node_dv],
                    node_err=[float(x) for x in self.node_err], node_span=[int(x) for x in self.node_span],
                    inflate=self.inflate, birge=self.birge, main_span=list(self.main_span),
                    span_bounds={int(k): [float(a), float(b)] for k, (a, b) in self.span_bounds.items()},
                    fwhm=self.fwhm,
                    n_used=self.n_used, segments=self.segments)


def _segments(lo: float, hi: float, tell: list[tuple[float, float]], width: float, min_width: float
              ) -> list[tuple[float, float, int]]:
    """The coverage minus telluric intervals, cut into near-equal segments of at most `width` Å: (lo, hi, span)"""
    spans, start = [], lo
    for a, b in sorted(tell):
        if b <= start or a >= hi:
            continue
        if a > start:
            spans.append((start, min(a, hi)))
        start = max(start, b)
    if start < hi:
        spans.append((start, hi))
    out = []
    for k, (a, b) in enumerate(spans):
        if b - a < min_width:
            continue
        n = max(1, int(np.ceil((b - a) / width - 1e-9)))
        edges = np.linspace(a, b, n + 1)
        out += [(lo, hi, k) for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    return out


def _emission(flux: np.ndarray, width_pix: int) -> np.ndarray:
    """Sky emission above a running 20th-percentile baseline"""
    good = np.isfinite(flux)
    f = np.where(good, flux, np.interp(np.arange(len(flux)), np.flatnonzero(good), flux[good]))
    return np.where(good, f - percentile_filter(f, 20, size=max(width_pix, 5)), np.nan)


def _corr_curve(w: np.ndarray, d: np.ndarray, fwhm: float, lags: np.ndarray) -> np.ndarray:
    """
    Correlation of the emission `d` on pixels `w` with the OH model shifted by each lag [km/s]

    The model is evaluated at the data's own pixels for every trial velocity (no resampling of the data), and
    both are Hann-tapered so lines cut by the segment edges get little weight.
    """
    good = np.isfinite(d)
    w, d = w[good], d[good]
    taper = np.hanning(len(w) + 2)[1:-1]
    a = (d - d.mean()) * taper
    na = np.sqrt(np.sum(a * a))
    out = np.empty(len(lags))
    for k, lag in enumerate(lags):
        m = model_sky(w / broadening.doppler_factor(lag), fwhm)
        bm = (m - m.mean()) * taper
        nb = np.sqrt(np.sum(bm * bm))
        out[k] = np.sum(a * bm) / (na * nb) if nb > 0 else np.nan
    return out


def _peak(lags: np.ndarray, c: np.ndarray) -> tuple[float, float, float]:
    """Lag [km/s], peak correlation and curvature [(km/s)⁻²] of a parabola through the top; NaN at the edges"""
    if not np.isfinite(c).any():
        return np.nan, np.nan, np.nan
    i = int(np.nanargmax(c))
    if i < 3 or i > len(c) - 4:
        return np.nan, float(c[i]), np.nan
    x = lags[i - 3:i + 4] - lags[i]
    p2, p1, p0 = np.polyfit(x, c[i - 3:i + 4], 2)
    if p2 >= 0:
        return np.nan, float(c[i]), np.nan
    x0 = -p1 / (2 * p2)
    return float(lags[i] + x0), float(np.polyval([p2, p1, p0], x0)), float(2 * p2)


def measure(spectrum: Spectrum, *, sky: np.ndarray | None = None, seg_width: float = 250., min_width: float = 60.,
            min_corr: float = 0.7, max_lag_kms: float = 150., telluric_depth: float = 0.3,
            min_lines: int = 3, fwhm: float | None = None) -> SkyZeroPoint | None:
    """
    Measure the wavelength zero point Δv(λ) from the OH sky lines

    Parameters
    ----------
    spectrum
        A spectrum with a sky (`spectrum.sky`, or `sky` on the same pixels); its meta says whether a velocity
        correction was applied to the wavelengths (undone here)
    sky
        Sky flux overriding `spectrum.sky`
    seg_width
        Segment width [Å]
    min_width
        Telluric-free spans narrower than this [Å] are not used
    min_corr
        Minimum peak correlation of a usable segment
    max_lag_kms
        Search range [km/s]
    telluric_depth
        Telluric bands at least this deep (catalogue depth) are cut out before segmenting: the absorption distorts
        the OH emission
    min_lines
        Minimum number of model lines in a segment with at least 10 per cent of the segment's strongest amplitude
    fwhm
        Sky-line FWHM [Å]; measured if None

    Returns
    -------
    zp
        The SkyZeroPoint, or None if there is no sky or fewer than two usable segments
    """
    from ..features import features
    sky = spectrum.sky if sky is None else np.asarray(sky, dtype=float)
    if sky is None:
        return None
    ff = observatory_frame_factor(spectrum.meta)
    wave = spectrum.wave / ff
    good = np.isfinite(sky) & np.isfinite(wave)
    w_all, d_all = wave[good], _emission(sky[good], int(40. / float(np.median(np.diff(wave[good])))))
    if fwhm is None:  # the width maximising the correlation peak over lags (a zero-point shift does not bias it)
        coarse = np.arange(-max_lag_kms, max_lag_kms + 1e-9, 10.)
        trials = np.geomspace(1., 15., 21)
        heights = [np.nanmax(_corr_curve(w_all, d_all, fw, coarse)) for fw in trials]
        i = int(np.nanargmax(heights))
        fwhm = float(trials[i])
        if 0 < i < len(trials) - 1:
            p = np.polyfit(np.log(trials[i - 1:i + 2]), heights[i - 1:i + 2], 2)
            if p[0] < 0:
                fwhm = float(np.exp(-p[1] / (2 * p[0])))
    lw, la = oh_lines()
    tell = [(f.wmin, f.wmax) for f in features() if f.kind == 'telluric' and np.isfinite(f.depth)
            and f.depth >= telluric_depth]
    lags = np.arange(-max_lag_kms, max_lag_kms + 1e-9, _DV)
    segs = []
    for lo, hi, span in _segments(float(w_all[0]), float(w_all[-1]), tell, seg_width, min_width):
        seg = dict(wmin=float(lo), wmax=float(hi), centre=float(0.5 * (lo + hi)), dv=np.nan, err=np.nan, corr=np.nan,
                   used=False, reason='', span=int(span))
        inseg = (lw >= lo) & (lw <= hi)
        n_lines = int(np.sum(inseg & (la >= 0.1 * la[inseg].max()))) if inseg.any() else 0
        idx = np.flatnonzero((w_all >= lo) & (w_all <= hi))
        if n_lines < min_lines or len(idx) < 20:
            seg['reason'] = f'{n_lines} strong OH lines'
        else:
            lag, corr, curv = _peak(lags, _corr_curve(w_all[idx], d_all[idx], fwhm, lags))
            seg['corr'] = corr
            if not np.isfinite(lag):
                seg['reason'] = 'no correlation peak'
            elif corr < min_corr:
                seg['reason'] = f'correlation {corr:.2f} < {min_corr}'
            else:
                n = int(np.sum(np.isfinite(d_all[idx])))  # native pixels: independent
                var = -1. / (n * curv / corr * corr ** 2 / (1. - corr ** 2)) if 0 < corr < 1 else np.nan
                seg.update(dv=lag, err=float(np.sqrt(var)) if np.isfinite(var) and var > 0 else np.nan,
                           used=True, reason='ok')
                if not np.isfinite(seg['err']):
                    seg.update(used=False, reason='undefined error')
        segs.append(seg)
    used = [s for s in segs if s['used']]
    if len(used) < 2:
        log.warning('Sky zero point: only %d usable segment(s) (%s); not measured', len(used),
                    '; '.join(f"{s['wmin']:.0f}-{s['wmax']:.0f}: {s['reason']}" for s in segs if not s['used']))
        return None
    x_c = np.array([s['centre'] for s in used])
    v = np.array([s['dv'] for s in used])
    e = np.array([s['err'] for s in used])
    k = np.array([s['span'] for s in used])
    w = 1. / e ** 2
    # realism of the node errors: Birge factor of a linear fit within each span (pooled)
    chi2, dof = 0., 0
    for span in np.unique(k):
        m = k == span
        if m.sum() > 2:
            c = np.polyfit(x_c[m], v[m], 1, w=np.sqrt(w[m]))
            chi2 += float(np.sum(w[m] * (v[m] - np.polyval(c, x_c[m])) ** 2))
            dof += int(m.sum()) - 2
    birge = float(np.sqrt(max(1., chi2 / dof))) if dof > 0 else 1.
    node_err = e * birge
    steps = [abs(v[i + 1] - v[i]) / (x_c[i + 1] - x_c[i]) for i in range(len(v) - 1) if k[i + 1] == k[i]]
    inflate = float(np.median(steps)) if steps else 0.02
    bounds = {}
    for s_ in segs:
        a_, b_ = bounds.get(s_['span'], (np.inf, -np.inf))
        bounds[s_['span']] = (min(a_, s_['wmin']), max(b_, s_['wmax']))
    zp = SkyZeroPoint(x_c, v, node_err, k, fwhm, segs, ff, inflate=inflate, birge=birge, span_bounds=bounds)
    main = int(max(np.unique(k), key=lambda span: (np.sum(k == span), -span)))
    m = k == main
    zp.main_span = (float(x_c[m].min()), float(x_c[m].max()))
    wn = 1. / node_err[m] ** 2
    lam = np.linspace(x_c[m].min(), x_c[m].max(), 200) if m.sum() > 1 else x_c[m]
    zp.zp_kms = float(np.mean(zp.dv(lam)))
    zp.zp_mean_err = float(1. / np.sqrt(np.sum(wn)))
    means = [zp.zp_kms, float(np.sum(wn * v[m]) / np.sum(wn))]
    if m.sum() > 1:
        a = np.vander(x_c[m] - x_c[m].mean(), 2)
        aw = a * np.sqrt(wn)[:, None]
        c, *_ = np.linalg.lstsq(aw, v[m] * np.sqrt(wn), rcond=None)
        cov = np.linalg.inv(aw.T @ aw)
        zp.slope, zp.slope_err = float(c[0] * 1000.), float(np.sqrt(cov[0, 0]) * 1000.)
        means.append(float(np.mean(np.polyval(c, lam - x_c[m].mean()))))
    zp.model_spread = float(np.std(means, ddof=1)) if len(means) > 1 else np.nan
    log.info('Sky zero point: %+.1f ± %.1f km/s over %.0f-%.0f Å (main span; slope %+.1f km/s per 1000 Å, model '
             'spread %.1f km/s); %d of %d segments in %d span(s); sky FWHM %.2f Å', zp.zp_kms, zp.zp_err,
             *zp.main_span, zp.slope, zp.model_spread, zp.n_used, len(segs), len(np.unique(k)), fwhm)
    for s_ in segs:
        log.debug('  sky segment %.0f-%.0f Å (span %d): %+.1f ± %.1f km/s, corr %.2f, %s', s_['wmin'], s_['wmax'],
                  s_['span'], s_['dv'], s_['err'], s_['corr'], s_['reason'])
    return zp


def apply(spectrum: Spectrum, zp: Any = None, *, constant_kms: float = 0., constant_err: float = np.nan,
          source: str = 'sky') -> Spectrum:
    """
    Correct the wavelengths for a zero point: λ' = λ / D(Δv(λ) + constant), D the relativistic Doppler factor

    Parameters
    ----------
    spectrum
        The spectrum (in any frame: the correction is a multiplicative factor)
    zp
        A zero-point curve with `dv(λ)` and `as_dict()` (`SkyZeroPoint`, or a `TelluricZeroPoint`, whose `kind`
        is 'telluric'); None: only the constant
    constant_kms
        An additional constant offset [km/s] (e.g. the star − sky offset from the tellurics)
    constant_err
        Its uncertainty [km/s] (recorded for the zero-point systematic)
    source
        Label recorded in meta['wavelength_correction']['source']

    Returns
    -------
    corrected
        The spectrum with corrected wavelengths; meta['wavelength_correction'] records what was applied (the
        curve under 'sky' or 'telluric')
    """
    dv = np.full(len(spectrum.wave), float(constant_kms))
    if zp is not None:
        dv = dv + zp.dv(spectrum.wave)
    new_wave = spectrum.wave / broadening.doppler_factor(dv)
    if np.any(np.diff(new_wave) <= 0):
        raise ValueError('the zero-point correction would make the wavelengths non-monotonic; not applied')
    meta = dict(spectrum.meta)
    kind = getattr(zp, 'kind', 'sky')
    meta['wavelength_correction'] = dict(source=source, constant_kms=float(constant_kms),
                                         constant_err=float(constant_err),
                                         sky=zp.as_dict() if zp is not None and kind == 'sky' else None,
                                         telluric=zp.as_dict() if zp is not None and kind == 'telluric' else None,
                                         dv_min=float(dv.min()), dv_max=float(dv.max()))
    return replace(spectrum, wave=new_wave, meta=meta)


def applied_dv(meta: Any, wave: float | np.ndarray) -> np.ndarray:
    """The velocity correction applied at `wave` according to meta['wavelength_correction'] (0 if none)"""
    wc = (meta or {}).get('wavelength_correction')
    if not wc:
        return np.zeros_like(np.asarray(wave, dtype=float))
    dv = np.full_like(np.asarray(wave, dtype=float), wc.get('constant_kms', 0.))
    for key in ('sky', 'telluric'):
        curve = wc.get(key)
        if curve:
            bounds = {int(k): tuple(v) for k, v in (curve.get('span_bounds') or {}).items()} or None
            dv = dv + _curve(wave, curve['node_wave'], curve['node_dv'], curve['node_err'], curve['node_span'],
                             curve.get('inflate', 0.), bounds)[0].reshape(np.shape(dv))
    return dv
