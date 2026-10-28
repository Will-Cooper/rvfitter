"""
Wavelength zero point from the telluric absorption bands.

Telluric absorption is imprinted on the starlight itself, so it shares the star's path through the slit and
the spectrograph. Its lines therefore measure the star's own wavelength zero point Δv(λ), including any offset
from slit centring, which the uniformly illuminated sky (`sky_zeropoint`) cannot see.

Method:
1. Work in the observatory frame: a velocity correction applied to the wavelengths in the header (PypeIt
   `VEL_CORR`) and a barycentric correction applied by the pipeline (`meta['barycentric_applied_kms']`) are
   undone.
2. Take the telluric bands of the feature catalogue (or the `telluric` regions given) that the HITRAN
   transmission model (`telluric.model`, built into the user cache) shows to be deeper than `min_depth` at the
   spectrum's airmass and resolution, cut them into segments of at most `seg_width` Å, and pad each by
   `pad_resel` resolution elements of continuum. Catalogue bands whose Teff range excludes the target's `teff`
   are skipped (O2 A below 2800 K, filled by the K I wings). Catalogue stellar lines (± `mask_resel` resolution
   elements) and bandhead edges are masked.
3. In each segment, fit the observed pixels with

       F(λ) = P(λ) · [T(λ / D(v); a_dry, a_H2O) ⊗ LSF],

   averaged over each pixel, where T = exp(−a_dry τ_dry − a_H2O τ_H2O) is the HITRAN model, the τ scalings
   (airmass and airmass × PWV) are free when the species is deeper than `min_depth` in that segment, and P is a
   Legendre polynomial of degree `degree` (the smooth stellar continuum; solved linearly at each trial).
   With a stellar `template` (rest frame, already at the data's resolution), the model is instead
   P · S(λ / D(v_star)) · [T ⊗ LSF], with the star's observatory-frame velocity v_star free, and the stellar
   lines are not masked: the star's features are then modelled rather than assumed smooth, and v_star − v (the
   star's velocity relative to the telluric lines in the same segment, independent of the wavelength solution)
   is reported. The velocities are found by scans (star, telluric, star again; telluric over ±`max_lag_kms`)
   and refined jointly with the scalings.
4. The errors come from the curvature of the χ² profile in the velocities (the scalings re-optimised at each
   point; a parabola for v alone, a quadratic surface for (v, v_star)), scaled by the Birge factor
   √max(1, χ²/ν) of the segment's fit.
5. The resolution is `resolution`, else `spectrum.resolution`, else (without a template) estimated by
   minimising the total χ² over R (a telluric-line estimate of the LSF).

Δv(λ) is the piecewise-linear interpolation between the segment values (held constant beyond them), as for the
sky; its error grows with the distance from the nearest node at the median node-to-node slope. With
`spread_templates`, each node's error also includes the spread of Δv over neighbouring stellar templates. `zp_kms`
is only the error-weighted mean of the nodes (Birge-inflated error), a summary that is not "the" zero point when
Δv varies across the spectrum; the systematics evaluate the curve at the measurements' wavelengths instead.

Sign: positive Δv means the data's wavelengths are too large (too red), as for `sky_zeropoint`.
"""
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.polynomial import legendre
from scipy.optimize import minimize

from .. import broadening
from ..constants import C_KMS
from ..types import Region, Spectrum
from . import telluric
from .sky_zeropoint import _curve, observatory_frame_factor

__all__ = ['TelluricZeroPoint', 'measure', 'star_minus_sky', 'frame_factor']

log = logging.getLogger(__name__)

_A_BOUNDS = {'dry': (0.2, 10.), 'h2o': (0.005, 200.)}  # airmass; airmass × PWV [mm]


def frame_factor(meta: Any) -> float:
    """The factor that converts observatory-frame wavelengths into the spectrum's (header + pipeline corrections)"""
    meta = meta or {}
    f = observatory_frame_factor(meta)
    v = meta.get('barycentric_applied_kms')
    if v is not None and np.isfinite(v) and v != 0:
        f *= float(broadening.doppler_factor(v))
    return f


@dataclass
class TelluricZeroPoint:
    """
    The star's zero-point curve Δv(λ) [km/s] from the telluric bands

    Unpacks as ``(zp_kms, zp_err)`` (the architecture's ``measure -> (dv, err)``); ``dv(λ)`` and ``dv_err(λ)``
    give the curve.
    """
    node_wave: np.ndarray
    node_dv: np.ndarray
    node_err: np.ndarray
    segments: list[dict] = field(default_factory=list)
    frame_factor: float = 1.
    resolution: float = np.nan
    resolution_source: str = ''
    airmass: float = np.nan
    template: str | None = None
    zp_kms: float = np.nan
    zp_err: float = np.nan
    birge: float = 1.
    inflate: float = 0.
    span_bounds: dict = field(default_factory=dict)
    kind: str = 'telluric'

    def __iter__(self):
        return iter((self.zp_kms, self.zp_err))

    @property
    def node_span(self) -> np.ndarray:
        return np.zeros(len(self.node_wave), dtype=int)

    @property
    def n_used(self) -> int:
        return sum(1 for s in self.segments if s['used'])

    def dv(self, wave: np.ndarray | float) -> np.ndarray:
        out = _curve(wave, self.node_wave, self.node_dv, self.node_err, self.node_span, self.inflate,
                     self.span_bounds)[0]
        return out if np.ndim(wave) else out[0]

    def dv_err(self, wave: np.ndarray | float) -> np.ndarray:
        out = _curve(wave, self.node_wave, self.node_dv, self.node_err, self.node_span, self.inflate,
                     self.span_bounds)[1]
        return out if np.ndim(wave) else out[0]

    def as_dict(self) -> dict:
        return dict(zp_tell_kms=self.zp_kms, zp_tell_err=self.zp_err, birge=self.birge, inflate=self.inflate,
                    node_wave=[float(x) for x in self.node_wave], node_dv=[float(x) for x in self.node_dv],
                    node_err=[float(x) for x in self.node_err], node_span=[0] * len(self.node_wave),
                    span_bounds={int(k): [float(a), float(b)] for k, (a, b) in self.span_bounds.items()},
                    resolution=self.resolution, resolution_source=self.resolution_source, airmass=self.airmass,
                    template=self.template,
                    n_used=self.n_used, segments=self.segments)


# ------------------------------------------------------------------------------------------------ segments
def _band_windows(regions: Sequence[Region] | None, lo: float, hi: float, teff: float | None = None
                  ) -> list[tuple[float, float, str]]:
    """
    Telluric bands (wmin, wmax, name) within [lo, hi]: the given telluric regions, else the catalogue's

    A catalogue band whose Teff range excludes `teff` is dropped (and logged): for a telluric band, the catalogue's
    Teff range is where it is a safe zero-point reference (e.g. O2 A below 2800 K, where the K I resonance wings
    fill the band and make the fitted Δv depend on the stellar template by tens of km/s).
    """
    from ..features import features
    cat = {f.name: f for f in features()}
    out = None
    if regions:
        out = [(r.wmin, r.wmax, r.name) for r in regions if r.kind == 'telluric'] or None
    if out is None:
        out = [(f.wmin, f.wmax, f.name) for f in cat.values() if f.kind == 'telluric']
    keep = []
    for a, b, n in out:
        if not (b > lo and a < hi):
            continue
        f = cat.get(n)
        if f is not None and not f.teff_ok(teff):
            log.info('Telluric zero point: %s not used at Teff %.0f K (a safe reference only for %.0f-%.0f K: %s)', n,
                     teff, f.teff_min, f.teff_max, f.caveat)
            continue
        keep.append((max(a, lo), min(b, hi), n))
    return keep


def _stellar_mask(wave: np.ndarray, resolution: float, mask_resel: float) -> np.ndarray:
    """True on pixels near catalogue stellar lines (± mask_resel resolution elements) and bandhead edges"""
    from ..features import features
    bad = np.zeros(len(wave), dtype=bool)
    for f in features():
        if f.kind == 'line' and np.isfinite(f.wave_vac):
            hw = mask_resel * f.wave_vac / resolution
            bad |= np.abs(wave - f.wave_vac) < hw
        elif f.kind == 'bandhead' and np.isfinite(f.wave_vac):
            hw = mask_resel * f.wave_vac / resolution
            bad |= (wave > f.wave_vac - hw) & (wave < f.wave_vac + hw)
    return bad


def _cut(a: float, b: float, width: float) -> list[tuple[float, float]]:
    n = max(1, int(np.ceil(np.log(b / a) / np.log1p(width / a) - 1e-9)))
    e = np.exp(np.linspace(np.log(a), np.log(b), n + 1))
    return list(zip(e[:-1], e[1:], strict=True))


# ------------------------------------------------------------------------------------------------ fitting
class _Segment:
    """One segment's data, the fine telluric model around it, and an optional stellar template"""

    def __init__(self, w_all, f_all, s_all, use, template, degree, max_lag):
        # models are evaluated on the contiguous pixels (so pixel edges are right) and compared on `use`
        self.w_all = w_all
        self.template = template
        self.degree = degree
        pad = 1. + 3. * max_lag / C_KMS + 0.004
        m = telluric.model()
        self.fw, self.td, self.th = m.slice(w_all[0] / pad, w_all[-1] * pad)
        self.dlnlam = m.dlnlam
        self._key, self._tc = None, None
        self._f_all, self._s_all = f_all, s_all
        self.set_use(use)

    def set_use(self, use: np.ndarray) -> None:
        """Select the pixels entering χ² (e.g. after clipping outliers)"""
        self.use = np.asarray(use, dtype=bool)
        self.w, self.f, s = self.w_all[self.use], self._f_all[self.use], self._s_all[self.use]
        self.wt = 1. / s ** 2 if np.all(np.isfinite(s)) and np.all(s > 0) else np.ones(len(self.w))
        x = 2. * (self.w - self.w_all[0]) / (self.w_all[-1] - self.w_all[0]) - 1.
        self.basis = legendre.legvander(x, self.degree)

    def residuals(self, vt: float, vs: float, a_dry: float, a_h2o: float, resolution: float) -> np.ndarray:
        """Normalised residuals (data − model)/σ on every pixel of the segment, the continuum fitted on `use`"""
        t_all = self.model_all(vt, vs, a_dry, a_h2o, resolution)
        a = self.basis * t_all[self.use][:, None]
        sw = np.sqrt(self.wt)
        c, *_ = np.linalg.lstsq(a * sw[:, None], self.f * sw, rcond=None)
        x = 2. * (self.w_all - self.w_all[0]) / (self.w_all[-1] - self.w_all[0]) - 1.
        model = legendre.legvander(x, self.degree) @ c * t_all
        s = self._s_all
        if not np.any(np.isfinite(s) & (s > 0)):
            s = np.ones(len(self.w_all))
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(np.isfinite(s) & (s > 0), (self._f_all - model) / s, np.nan)

    def depth(self, which: str, a: float, resolution: float) -> float:
        tau = self.td if which == 'dry' else self.th
        t = broadening.convolve(np.exp(-a * tau), broadening.gaussian_kernel(
            telluric.lsf_sigma_pix(resolution, self.dlnlam)))
        inside = (self.fw >= self.w[0]) & (self.fw <= self.w[-1])
        return float(1. - np.min(t[inside])) if inside.any() else 0.

    def model_all(self, vt: float, vs: float, a_dry: float, a_h2o: float, resolution: float) -> np.ndarray:
        """Telluric (shifted by vt) × stellar template (shifted by vs), averaged over the contiguous pixels"""
        key = (a_dry, a_h2o, resolution)
        if key != self._key:
            t = np.exp(-a_dry * self.td - a_h2o * self.th)
            self._tc = broadening.convolve(t, broadening.gaussian_kernel(
                telluric.lsf_sigma_pix(resolution, self.dlnlam)))
            self._key = key
        tm = broadening.rebin(self.fw * broadening.doppler_factor(vt), self._tc, self.w_all)
        return tm if self.template is None else tm * self.template.at(self.w_all, vs)

    def chi2(self, vt: float, vs: float, a_dry: float, a_h2o: float, resolution: float) -> float:
        t = self.model_all(vt, vs, a_dry, a_h2o, resolution)[self.use]
        if not np.all(np.isfinite(t)):
            return np.inf
        a = self.basis * t[:, None]
        sw = np.sqrt(self.wt)
        c, *_ = np.linalg.lstsq(a * sw[:, None], self.f * sw, rcond=None)
        r = self.f - a @ c
        return float(np.sum(self.wt * r * r))


class _Fitter:
    """Least squares of one segment over (v_tell[, v_star], free τ scalings), with the continuum solved linearly"""

    def __init__(self, seg: _Segment, resolution: float, a0: dict, free: Sequence[str], max_lag: float,
                 star_range: tuple[float, float]):
        self.seg, self.resolution, self.a0, self.names = seg, resolution, dict(a0), list(free)
        self.star = seg.template is not None
        self.max_lag, self.star_range = max_lag, star_range

    def scalings(self, p) -> dict:
        a = dict(self.a0)
        for k, n in enumerate(self.names):
            a[n] = float(np.clip(np.exp(p[k]), *_A_BOUNDS[n]))
        return a

    def chi2(self, vt: float, vs: float, p) -> float:
        """χ² at the velocities and scalings; infinite outside the search ranges (which bound the optimisers)"""
        if abs(vt) > self.max_lag or (self.star and not self.star_range[0] <= vs <= self.star_range[1]):
            return np.inf
        a = self.scalings(p)
        return self.seg.chi2(vt, vs, a['dry'], a['h2o'], self.resolution)

    def profile(self, vt: float, vs: float, p_init) -> tuple[float, np.ndarray]:
        """χ² minimised over the free scalings at fixed velocities"""
        if not self.names:
            return self.chi2(vt, vs, p_init), p_init
        r = minimize(lambda p: self.chi2(vt, vs, p), p_init, method='Nelder-Mead',
                     options=dict(xatol=1e-3, fatol=1e-3, maxiter=400))
        return float(r.fun), r.x

    def best(self, max_lag: float, step: float, star_range: tuple[float, float]) -> dict:
        """Scans (star, then telluric, then star again), then a joint Nelder–Mead refinement"""
        p0 = np.log([self.a0[n] for n in self.names]) if self.names else np.zeros(0)
        vs0 = 0.
        if self.star:
            stars = np.arange(star_range[0], star_range[1] + 1e-9, step)
            vs0 = float(stars[int(np.argmin([self.chi2(0., v, p0) for v in stars]))])
        lags = np.arange(-max_lag, max_lag + 1e-9, step)
        vt0 = float(lags[int(np.argmin([self.chi2(v, vs0, p0) for v in lags]))])
        if self.star:
            stars = np.arange(vs0 - 5 * step, vs0 + 5 * step + 1e-9, step / 2.)
            vs0 = float(stars[int(np.argmin([self.chi2(vt0, v, p0) for v in stars]))])
        start = np.concatenate([[vt0], [vs0] if self.star else [], p0])
        nv = 2 if self.star else 1
        scale = [step] * nv + [0.2] * len(p0)
        simplex = np.vstack([start] + [start + np.eye(len(start))[k] * scale[k] for k in range(len(start))])

        def fun(q):
            return self.chi2(q[0], q[1] if self.star else 0., q[nv:])

        r = minimize(fun, start, method='Nelder-Mead',
                     options=dict(xatol=1e-3, fatol=1e-4, maxiter=3000, initial_simplex=simplex))
        return dict(vt=float(r.x[0]), vs=float(r.x[1]) if self.star else np.nan, p=r.x[nv:], chi2=float(r.fun))

    def curvature(self, fit: dict, h: float) -> dict:
        """
        Covariance of the velocities from the χ² profile (scalings re-optimised) on a grid of ±h steps

        One velocity: a parabola through v ± k·h, k = −2..2. Two: a quadratic surface through the 3 × 3 grid.
        cov = 2 H⁻¹. NaN if the surface is not positive definite.
        """
        p = fit['p']
        if not self.star:
            vs = fit['vt'] + h * np.arange(-2, 3)
            c2 = []
            for v in vs:
                val, p = self.profile(v, np.nan, p)
                c2.append(val)
            cc = np.polyfit(vs - fit['vt'], c2, 2)
            if cc[0] <= 0:
                return dict(vt=fit['vt'], cov=np.full((1, 1), np.nan), chi2=float(np.min(c2)))
            d = -cc[1] / (2 * cc[0])
            vt = fit['vt'] + d if abs(d) < 2 * h else fit['vt']
            return dict(vt=float(vt), cov=np.array([[1. / cc[0]]]),
                        chi2=float(min(np.min(c2), np.polyval(cc, d))))
        rows, vals = [], []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                val, p = self.profile(fit['vt'] + i * h, fit['vs'] + j * h, p)
                x, y = i * h, j * h
                rows.append([1., x, y, 0.5 * x * x, 0.5 * y * y, x * y])
                vals.append(val)
        c, *_ = np.linalg.lstsq(np.array(rows), np.array(vals), rcond=None)
        hess = np.array([[c[3], c[5]], [c[5], c[4]]])
        if np.any(np.linalg.eigvalsh(hess) <= 0):
            return dict(vt=fit['vt'], vs=fit['vs'], cov=np.full((2, 2), np.nan), chi2=float(np.min(vals)))
        cov = 2. * np.linalg.inv(hess)
        d = -np.linalg.solve(hess, c[1:3])
        if np.all(np.abs(d) < 2 * h):
            vt, vs = fit['vt'] + d[0], fit['vs'] + d[1]
            chi2 = float(min(np.min(vals), c[0] + 0.5 * c[1:3] @ d))
        else:
            vt, vs, chi2 = fit['vt'], fit['vs'], float(np.min(vals))
        return dict(vt=float(vt), vs=float(vs), cov=cov, chi2=chi2)


def _segment_fit(seg: _Segment, resolution: float, airmass: float, pwv: float, min_depth: float, max_lag: float,
                 star_range: tuple[float, float], clip: float = 5.) -> dict:
    """Fit one segment at fixed R: Δv (and the star's velocity with a template), errors and diagnostics"""
    a0 = {'dry': airmass, 'h2o': airmass * pwv}
    depths = {k: seg.depth(k, a0[k], resolution) for k in ('dry', 'h2o')}
    free = [k for k in ('dry', 'h2o') if depths[k] >= min_depth]
    out = dict(depth_dry=depths['dry'], depth_h2o=depths['h2o'], free=','.join(free))
    if not free:
        return dict(out, reason=f'telluric depth {max(depths.values()):.3f} < {min_depth}')
    step = max(1., 0.1 * C_KMS / resolution)
    fitter = _Fitter(seg, resolution, a0, free, max_lag, star_range)
    fit = fitter.best(max_lag, step, star_range)
    # one round of outlier clipping (bad pixels, cosmic rays): |residual| > clip × robust scatter
    a = fitter.scalings(fit['p'])
    r = seg.residuals(fit['vt'], fit['vs'], a['dry'], a['h2o'], resolution)
    scatter = 1.4826 * np.median(np.abs(r[seg.use] - np.median(r[seg.use])))
    bad = seg.use & (np.abs(r - np.median(r[seg.use])) > clip * scatter)
    out['n_clipped'] = int(bad.sum())
    if bad.any():
        seg.set_use(seg.use & ~bad)
        fit = fitter.best(max_lag, step, star_range)
    if abs(fit['vt']) > max_lag - step:
        return dict(out, dv=fit['vt'], reason='optimum at the edge of the search range')
    if fitter.star and not star_range[0] + step < fit['vs'] < star_range[1] - step:
        return dict(out, dv=fit['vt'], v_star=fit['vs'], reason='stellar velocity at the edge of the search range')
    cv = fitter.curvature(fit, max(0.5, 0.05 * C_KMS / resolution))
    n = len(seg.w)
    dof = n - (seg.basis.shape[1] + len(free) + (2 if fitter.star else 1))
    red = cv['chi2'] / dof if dof > 0 else np.nan
    cov = cv['cov']
    if not np.all(np.isfinite(cov)) or not np.isfinite(red):
        return dict(out, dv=cv['vt'], reason='undefined χ² curvature')
    birge = float(np.sqrt(max(1., red)))
    a = fitter.scalings(fit['p'])
    stat = float(np.sqrt(cov[0, 0]))
    res = dict(out, dv=cv['vt'], err=stat * birge, err_stat=stat, chi2_red=red, birge=birge, used=True, reason='ok',
               a_dry=a['dry'], a_h2o=a['h2o'], n_pix=n)
    if fitter.star:
        s_star = float(np.sqrt(cov[1, 1]))
        s_diff = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0.)))
        res.update(v_star=cv['vs'], v_star_err=s_star * birge, star_minus_tell=cv['vs'] - cv['vt'],
                   star_minus_tell_err=s_diff * birge, star_minus_tell_err_stat=s_diff,
                   corr_star_tell=float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])))
    return res


def _total_chi2(segs: list[_Segment], resolution: float, airmass: float, pwv: float, min_depth: float,
                max_lag: float, star_range: tuple[float, float]) -> float:
    total = 0.
    for seg in segs:
        a0 = {'dry': airmass, 'h2o': airmass * pwv}
        free = [k for k in ('dry', 'h2o') if seg.depth(k, a0[k], resolution) >= min_depth]
        if free:
            fitter = _Fitter(seg, resolution, a0, free, max_lag, star_range)
            total += fitter.best(max_lag, max(1., 0.1 * C_KMS / resolution), star_range)['chi2']
    return total


def _estimate_resolution(segs, airmass, pwv, min_depth, max_lag, star_range) -> float:
    """R minimising the total χ² of the segments: a coarse geometric grid, refined, then a parabola in ln R"""
    args = (airmass, pwv, min_depth, max_lag, star_range)
    grid = np.geomspace(500., 100_000., 14)
    c2 = np.array([_total_chi2(segs, r, *args) for r in grid])
    i = int(np.argmin(c2))
    fine = grid[i] * np.geomspace(1 / 1.5, 1.5, 7)
    c2f = np.array([_total_chi2(segs, r, *args) for r in fine])
    j = int(np.argmin(c2f))
    if 0 < j < len(fine) - 1:
        p = np.polyfit(np.log(fine[j - 1:j + 2]), c2f[j - 1:j + 2], 2)
        if p[0] > 0:
            return float(np.exp(-p[1] / (2 * p[0])))
    return float(fine[j])


# ------------------------------------------------------------------------------------------------ public
def measure(spectrum: Spectrum, regions: Sequence[Region] | None = None, *, resolution: float | None = None,
            airmass: float | None = None, pwv: float = telluric.DEFAULT_PWV, seg_width: float = 250.,
            pad_resel: float = 15., mask_resel: float = 5., degree: int = 2, min_depth: float = 0.05,
            max_lag_kms: float = 150., template: Any = None,
            star_rv_range: tuple[float, float] = (-400., 400.), min_pixels: int = 20, teff: float | None = None,
            spread_templates: Sequence[Any] = ()) -> TelluricZeroPoint | None:
    """
    Measure the star's wavelength zero point Δv(λ) from the telluric absorption bands

    Parameters
    ----------
    spectrum
        A spectrum whose telluric absorption has NOT been corrected; its meta says which velocity corrections
        were applied to the wavelengths (undone here)
    regions
        Telluric regions to use (kind 'telluric'); default: the catalogue's telluric bands within the coverage
    resolution
        Resolving power; default `spectrum.resolution`, else estimated from the telluric lines
    airmass
        Starting airmass (default meta['airmass'], else 1.2); the τ scalings are fitted where the bands are deep
    pwv
        Starting precipitable water vapour [mm]
    seg_width
        Maximum segment width [Å]
    pad_resel
        Continuum padding on each side of a segment [resolution elements]
    mask_resel
        Half-width of the masks around catalogue stellar lines and bandheads [resolution elements]; not used
        with a template
    degree
        Legendre degree of the smooth stellar continuum in each segment
    min_depth
        A species is fitted (and a segment used) only if its model absorption depth exceeds this
    max_lag_kms
        Search range [km/s]
    template
        Optional rest-frame stellar template already at the data's resolution, with ``at(wave, v)`` (e.g.
        `methods._common.Template`); requires a known resolution
    star_rv_range
        Search range of the star's observatory-frame velocity with a template [km/s]
    min_pixels
        Minimum number of usable pixels in a segment
    teff
        The target's Teff [K] (e.g. the prior): catalogue bands that are not a safe reference at this Teff are
        skipped (see `_band_windows`); None keeps every band
    spread_templates
        Alternative stellar templates (e.g. the neighbouring grid points in Teff); each used segment is refitted
        with each, and the standard deviation of Δv over the templates (including `template`) is added in
        quadrature to its error (`err_template`)

    Returns
    -------
    zp
        The TelluricZeroPoint (unpacks as (zp_kms, zp_err)), or None if no segment is usable or the telluric model
        has not been built (logged with the instruction to build it)
    """
    if not telluric.available():
        log.warning('Telluric zero point not measured: %s', telluric.unavailable_message())
        return None
    ff = frame_factor(spectrum.meta)
    wave = spectrum.wave / ff
    flux, sigma = np.asarray(spectrum.flux, float), np.asarray(spectrum.sigma, float)
    if airmass is None:
        am = spectrum.meta.get('airmass') if hasattr(spectrum.meta, 'get') else None
        airmass = float(am) if am is not None and np.isfinite(am) and am >= 1 else 1.2
        if am is None:
            log.info('Telluric zero point: no airmass in the header; starting from %.1f', airmass)
    if template is not None and resolution is None and not spectrum.resolution:
        raise ValueError('a stellar template needs a known resolution')
    tm = telluric.model()
    lo_cov, hi_cov = max(float(wave[0]), tm.wmin), min(float(wave[-1]), tm.wmax)
    r_guess = resolution or spectrum.resolution or 3000.
    bands = _band_windows(regions, lo_cov, hi_cov, teff)
    pieces = []
    for a, b, name in bands:
        if b - a < 3 * a / r_guess:
            continue
        pieces += [(x, y, name) for x, y in _cut(a, b, seg_width)]
    segs, meta_segs = [], []
    good = np.isfinite(flux) & np.isfinite(wave)
    if np.isfinite(sigma).any():
        good &= np.isfinite(sigma) & (sigma > 0)
    for a, b, name in pieces:
        pad = pad_resel * 0.5 * (a + b) / r_guess
        lo, hi = max(a - pad, lo_cov), min(b + pad, hi_cov)
        span = (wave >= lo) & (wave <= hi)
        use = good[span]
        if template is None:
            use &= ~_stellar_mask(wave[span], r_guess, mask_resel)
        info = dict(band=name, wmin=float(a), wmax=float(b), centre=float(np.sqrt(a * b)), dv=np.nan, err=np.nan,
                    used=False, reason='', n_pix=int(use.sum()))
        meta_segs.append(info)
        if use.sum() < min_pixels:
            info['reason'] = f'{int(use.sum())} usable pixels'
            segs.append(None)
            continue
        f_span = np.where(good[span], flux[span], 0.)  # unused pixels never enter χ²
        segs.append(_Segment(wave[span], f_span, sigma[span], use, template, degree, max_lag_kms))
    usable = [s for s in segs if s is not None]
    if not usable:
        log.warning('Telluric zero point: no telluric band with enough pixels in %.0f-%.0f Å; not measured',
                    lo_cov, hi_cov)
        return None
    if resolution is None and spectrum.resolution:
        resolution, r_src = float(spectrum.resolution), 'spectrum'
    elif resolution is None:
        resolution = _estimate_resolution(usable, airmass, pwv, min_depth, max_lag_kms, star_rv_range)
        r_src = 'telluric lines'
        log.info('Telluric zero point: R = %.0f estimated from the telluric lines', resolution)
    else:
        r_src = 'given'
    for seg, info in zip(segs, meta_segs, strict=True):
        if seg is None:
            continue
        res = _segment_fit(seg, resolution, airmass, pwv, min_depth, max_lag_kms, star_rv_range)
        if res.get('used') and spread_templates and template is not None:
            # template systematic of this node: refit with each alternative template
            alts = []
            for alt in spread_templates:
                s2 = _Segment(seg.w_all, seg._f_all, seg._s_all, seg.use, alt, degree, max_lag_kms)
                r2 = _segment_fit(s2, resolution, airmass, pwv, min_depth, max_lag_kms, star_rv_range)
                if r2.get('used'):
                    alts.append(float(r2['dv']))
            spread = float(np.std([res['dv'], *alts], ddof=1)) if alts else 0.
            res.update(err_fit=res['err'], err_template=spread, dv_templates=alts,
                       err=float(np.hypot(res['err'], spread)))
        info.update(res)
        log.debug('  telluric segment %.0f-%.0f Å (%s): %+.1f ± %.1f km/s, χ²/ν %.2f, depths dry %.2f H2O %.2f%s, '
                  '%s', info['wmin'], info['wmax'], info['band'], info['dv'], info['err'],
                  info.get('chi2_red', np.nan), info.get('depth_dry', np.nan), info.get('depth_h2o', np.nan),
                  f"; star − telluric {info['star_minus_tell']:+.1f} ± {info['star_minus_tell_err']:.1f} km/s"
                  if 'star_minus_tell' in info else '', info['reason'])
    used = [s for s in meta_segs if s['used']]
    if not used:
        log.warning('Telluric zero point: no usable segment (%s); not measured',
                    '; '.join(f"{s['wmin']:.0f}-{s['wmax']:.0f}: {s['reason']}" for s in meta_segs))
        return None
    x = np.array([s['centre'] for s in used])
    v = np.array([s['dv'] for s in used])
    e = np.array([s['err'] for s in used])
    w = 1. / e ** 2
    mean = float(np.sum(w * v) / np.sum(w))
    chi2 = float(np.sum(w * (v - mean) ** 2))
    birge = float(np.sqrt(max(1., chi2 / (len(v) - 1)))) if len(v) > 1 else 1.
    steps = [abs(v[i + 1] - v[i]) / (x[i + 1] - x[i]) for i in range(len(v) - 1)]
    inflate = float(np.median(steps)) if steps else 0.02
    zp = TelluricZeroPoint(x, v, e, meta_segs, ff, resolution, r_src, airmass,
                           template=getattr(template, 'label', None) if template is not None else None, zp_kms=mean,
                           zp_err=float(birge / np.sqrt(np.sum(w))), birge=birge, inflate=inflate,
                           span_bounds={0: (float(min(s['wmin'] for s in used)),
                                            float(max(s['wmax'] for s in used)))})
    log.info('Telluric zero point: Δv(λ) %+.1f to %+.1f km/s over %.0f-%.0f Å from %d of %d segments (node-weighted '
             'mean %+.1f ± %.1f km/s, Birge %.2f; not a single zero point when Δv varies); R %.0f (%s)', v.min(),
             v.max(), x.min(), x.max(), zp.n_used, len(meta_segs), zp.zp_kms, zp.zp_err, birge, resolution, r_src)
    return zp




def star_minus_sky(tell: TelluricZeroPoint, sky: Any) -> dict:
    """
    The star − sky zero-point constant from the telluric nodes that lie inside the sky's measured range

    Only a telluric node whose wavelength falls inside a used (measured) sky segment is compared: there the sky
    value is a measurement, not an interpolation across an unmeasured gap, a value held flat beyond the last node
    or a ramp towards an edge node. Each such node gives d_i = Δv_tell,i − Δv_sky(λ_i), with error
    √(σ_tell,i² + σ_sky(λ_i)²). The constant is their weighted mean, its error Birge-inflated, with the sky curve's
    model spread (the spread of reasonable curve models, as in `zp_sky_model_spread`) added in quadrature; the χ²
    against a constant tests whether the star sees the same distortion as the sky (with two or more nodes).

    Returns
    -------
    info
        {'zp_star_sky', 'zp_star_sky_err', 'n_nodes', 'chi2', 'dof', 'birge', 'reason', 'nodes': [...]}; the
        constant and its error are NaN when no node lies in the measured sky range (`reason` says why)
    """
    lam = np.asarray(tell.node_wave, float)
    s_dv = np.asarray(sky.dv(lam), float)
    s_err = np.asarray(sky.dv_err(lam), float)
    measured = [(float(s['wmin']), float(s['wmax'])) for s in getattr(sky, 'segments', []) if s.get('used')]
    inside = np.array([any(a <= x <= b for a, b in measured) for x in lam], dtype=bool)
    d = tell.node_dv - s_dv
    e = np.hypot(tell.node_err, s_err)
    ok = inside & np.isfinite(d) & np.isfinite(e) & (e > 0)
    nodes = [dict(wave=float(lam[i]), tell=float(tell.node_dv[i]), tell_err=float(tell.node_err[i]),
                  sky=float(s_dv[i]), sky_err=float(s_err[i]), diff=float(d[i]), err=float(e[i]),
                  sky_measured=bool(inside[i]), used=bool(ok[i]))
             for i in range(len(lam))]
    nan = float('nan')
    if not ok.any():
        reason = (f'none of the {len(lam)} telluric node(s) ({", ".join(f"{x:.0f}" for x in lam)} Å) lies inside a '
                  f'measured sky segment ({", ".join(f"{a:.0f}-{b:.0f}" for a, b in measured) or "none"} Å)')
        log.info('Star - sky zero point not measured: %s', reason)
        return dict(zp_star_sky=nan, zp_star_sky_err=nan, n_nodes=0, chi2=nan, dof=0, birge=nan, reason=reason,
                    nodes=nodes)
    w = 1. / e[ok] ** 2
    mean = float(np.sum(w * d[ok]) / np.sum(w))
    chi2 = float(np.sum(w * (d[ok] - mean) ** 2))
    dof = int(ok.sum()) - 1
    birge = float(np.sqrt(max(1., chi2 / dof))) if dof > 0 else 1.
    spread = getattr(sky, 'model_spread', nan)
    spread = float(spread) if spread is not None and np.isfinite(spread) else 0.
    err = float(np.hypot(birge / np.sqrt(np.sum(w)), spread))
    log.info('Star - sky zero point: %+.1f ± %.1f km/s from %d telluric node(s) inside the measured sky range (%d '
             'outside it not used; χ² %.1f for %d dof; sky model spread %.1f km/s included)', mean, err,
             int(ok.sum()), int((~inside).sum()), chi2, dof, spread)
    return dict(zp_star_sky=mean, zp_star_sky_err=err, n_nodes=int(ok.sum()), chi2=chi2, dof=dof, birge=birge,
                reason='ok', nodes=nodes)
