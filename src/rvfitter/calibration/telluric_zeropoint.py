"""
Wavelength zero point from the telluric absorption bands.

Telluric absorption shares the starlight's path through the slit, so its lines measure the star's own zero point
Δv(λ), including the slit-centring offset that the uniformly illuminated sky (`sky_zeropoint`) cannot see.

In the observatory frame (header `VEL_CORR` and any applied barycentric correction undone), each telluric band
segment (catalogue bands safe at the target's Teff, padded with continuum, stellar lines masked) is fitted with

    F(λ) = P(λ) · [S(λ / D(v_star))] · [T(λ / D(v); a_dry, a_H2O) ⊗ LSF],

averaged over each pixel: T the HITRAN model with its τ scalings free where the species is deep enough, P a
Legendre continuum solved linearly, and the stellar template S (with v_star free) only when one is given. The
velocity errors come from the χ² profile's curvature (scalings re-optimised), scaled by the segment's Birge
factor. Without a known resolution, R is estimated by minimising the total χ².

Δv(λ) interpolates the segment values as for the sky; `zp_kms` is only their weighted mean, a summary.
Sign: positive Δv means the data's wavelengths are too red, as for `sky_zeropoint`.
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
from ..numerics import parabola_vertex, weighted_mean_chi2
from ..types import Region, Spectrum
from . import telluric
from .sky_zeropoint import INFLATE_DEFAULT, _Curve, frame_factor

__all__ = ['TelluricZeroPoint', 'measure', 'star_minus_sky_direct', 'node_windows', 'frame_factor']

log = logging.getLogger(__name__)

_A_BOUNDS = {'dry': (0.2, 10.), 'h2o': (0.005, 200.)}  # airmass; airmass × PWV [mm]
_AIRMASS_DEFAULT = 1.2  # starting airmass without a header value (the scalings are fitted)
_R_GUESS = 3000.  # resolving power used to size the segments before R is known or estimated
_PAD_RESEL = 15.  # continuum padding on each side of a segment [resolution elements]
_MASK_RESEL = 5.  # half-width of the stellar-line masks without a template [resolution elements]
_MIN_PIXELS = 20  # minimum usable pixels in a segment


@dataclass
class TelluricZeroPoint(_Curve):
    """The star's zero-point curve Δv(λ) [km/s] from the telluric bands; unpacks as ``(zp_kms, zp_err)``."""
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

    def as_dict(self) -> dict:
        return dict(zp_tell_kms=self.zp_kms, zp_tell_err=self.zp_err, **self._nodes_dict(),
                    resolution=self.resolution, resolution_source=self.resolution_source, airmass=self.airmass,
                    template=self.template)


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
        x = 2. * (w_all - w_all[0]) / (w_all[-1] - w_all[0]) - 1.
        self.basis_all = legendre.legvander(x, degree)
        self.set_use(use)

    def set_use(self, use: np.ndarray) -> None:
        """Select the pixels entering χ² (e.g. after clipping outliers)"""
        self.use = np.asarray(use, dtype=bool)
        self.w, self.f, s = self.w_all[self.use], self._f_all[self.use], self._s_all[self.use]
        # without σ the fit is unweighted (measure warns): χ² and the Birge factor are then in data units
        self.wt = 1. / s ** 2 if np.all(np.isfinite(s)) and np.all(s > 0) else np.ones(len(self.w))
        self.basis = self.basis_all[self.use]

    def _continuum(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The design matrix P·t on `use` and its weighted least-squares coefficients"""
        a = self.basis * t[:, None]
        sw = np.sqrt(self.wt)
        c, *_ = np.linalg.lstsq(a * sw[:, None], self.f * sw, rcond=None)
        return a, c

    def residuals(self, vt: float, vs: float, a_dry: float, a_h2o: float, resolution: float) -> np.ndarray:
        """Normalised residuals (data − model)/σ on every pixel of the segment, the continuum fitted on `use`"""
        t_all = self.model_all(vt, vs, a_dry, a_h2o, resolution)
        c = self._continuum(t_all[self.use])[1]
        model = self.basis_all @ c * t_all
        s = self._s_all
        if not np.any(np.isfinite(s) & (s > 0)):
            s = np.ones(len(self.w_all))
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(np.isfinite(s) & (s > 0), (self._f_all - model) / s, np.nan)

    def _lsf(self, t: np.ndarray, resolution: float) -> np.ndarray:
        return broadening.convolve(t, broadening.gaussian_kernel(telluric.lsf_sigma_pix(resolution, self.dlnlam)))

    def depth(self, which: str, a: float, resolution: float) -> float:
        t = self._lsf(np.exp(-a * (self.td if which == 'dry' else self.th)), resolution)
        inside = (self.fw >= self.w[0]) & (self.fw <= self.w[-1])
        return float(1. - np.min(t[inside])) if inside.any() else 0.

    def model_all(self, vt: float, vs: float, a_dry: float, a_h2o: float, resolution: float) -> np.ndarray:
        """Telluric (shifted by vt) × stellar template (shifted by vs), averaged over the contiguous pixels"""
        key = (a_dry, a_h2o, resolution)
        if key != self._key:
            self._tc = self._lsf(np.exp(-a_dry * self.td - a_h2o * self.th), resolution)
            self._key = key
        tm = broadening.rebin(self.fw * broadening.doppler_factor(vt), self._tc, self.w_all)
        return tm if self.template is None else tm * self.template.at(self.w_all, vs)

    def chi2(self, vt: float, vs: float, a_dry: float, a_h2o: float, resolution: float) -> float:
        t = self.model_all(vt, vs, a_dry, a_h2o, resolution)[self.use]
        if not np.all(np.isfinite(t)):
            return np.inf
        a, c = self._continuum(t)
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

    def best(self, step: float) -> dict:
        """Scans (star, then telluric, then star again), then a joint Nelder–Mead refinement"""
        max_lag, star_range = self.max_lag, self.star_range
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
            d, c2_min, curv = parabola_vertex(vs - fit['vt'], c2)
            if curv <= 0:
                return dict(vt=fit['vt'], cov=np.full((1, 1), np.nan), chi2=float(np.min(c2)))
            vt = fit['vt'] + d if abs(d) < 2 * h else fit['vt']
            return dict(vt=float(vt), cov=np.array([[2. / curv]]), chi2=float(min(np.min(c2), c2_min)))
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


def _scalings(seg: _Segment, resolution: float, airmass: float, pwv: float, min_depth: float
              ) -> tuple[dict, dict, list[str]]:
    """Starting τ scalings, model depths, and the species deep enough to be fitted"""
    a0 = {'dry': airmass, 'h2o': airmass * pwv}
    depths = {k: seg.depth(k, a0[k], resolution) for k in ('dry', 'h2o')}
    return a0, depths, [k for k in ('dry', 'h2o') if depths[k] >= min_depth]


def _segment_fit(seg: _Segment, resolution: float, airmass: float, pwv: float, min_depth: float, max_lag: float,
                 star_range: tuple[float, float], clip: float = 5.) -> dict:
    """Fit one segment at fixed R: Δv (and the star's velocity with a template), errors and diagnostics"""
    a0, depths, free = _scalings(seg, resolution, airmass, pwv, min_depth)
    out = dict(depth_dry=depths['dry'], depth_h2o=depths['h2o'], free=','.join(free))
    if not free:
        return dict(out, reason=f'telluric depth {max(depths.values()):.3f} < {min_depth}')
    step = max(1., 0.1 * C_KMS / resolution)
    fitter = _Fitter(seg, resolution, a0, free, max_lag, star_range)
    fit = fitter.best(step)
    # one round of outlier clipping (bad pixels, cosmic rays): |residual| > clip × robust scatter
    a = fitter.scalings(fit['p'])
    r = seg.residuals(fit['vt'], fit['vs'], a['dry'], a['h2o'], resolution)
    scatter = 1.4826 * np.median(np.abs(r[seg.use] - np.median(r[seg.use])))
    bad = seg.use & (np.abs(r - np.median(r[seg.use])) > clip * scatter)
    out['n_clipped'] = int(bad.sum())
    if bad.any():
        seg.set_use(seg.use & ~bad)
        fit = fitter.best(step)
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
        a0, _, free = _scalings(seg, resolution, airmass, pwv, min_depth)
        if free:
            total += _Fitter(seg, resolution, a0, free, max_lag, star_range).best(max(1., 0.1 * C_KMS / resolution))[
                'chi2']
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
        x0, _, curv = parabola_vertex(np.log(fine[j - 1:j + 2]), c2f[j - 1:j + 2])
        if curv > 0:
            return float(np.exp(x0))
    return float(fine[j])


# ------------------------------------------------------------------------------------------------ public
def measure(spectrum: Spectrum, regions: Sequence[Region] | None = None, *, resolution: float | None = None,
            seg_width: float = 250., degree: int = 2, min_depth: float = 0.05, max_lag_kms: float = 150.,
            template: Any = None, star_rv_range: tuple[float, float] = (-400., 400.), teff: float | None = None,
            spread_templates: Sequence[Any] = ()) -> TelluricZeroPoint | None:
    """
    Measure the star's zero point Δv(λ) from the telluric bands of a spectrum NOT telluric-corrected.

    `regions` of kind 'telluric' replace the catalogue's bands. `resolution` defaults to `spectrum.resolution`,
    else is estimated from the telluric lines (a `template` needs it known). `seg_width` [Å] caps the segments,
    `degree` is the Legendre continuum degree, a species is fitted only where its model depth exceeds `min_depth`,
    and `max_lag_kms` / `star_rv_range` [km/s] bound the telluric and stellar velocity searches. `template` is a
    rest-frame stellar template at the data's resolution with ``at(wave, v)``. Catalogue bands that are not a safe
    reference at `teff` [K] are skipped. Each used segment is refitted with every template in `spread_templates`,
    and the standard deviation of Δv over the templates is added in quadrature to its error (`err_template`).
    Returns None (logged) if no segment is usable or the telluric model is not built.
    """
    if not telluric.available():
        log.warning('Telluric zero point not measured: %s', telluric.unavailable_message())
        return None
    from ..regions import meta_airmass
    ff = frame_factor(spectrum.meta)
    wave = spectrum.wave / ff
    flux, sigma = np.asarray(spectrum.flux, float), np.asarray(spectrum.sigma, float)
    airmass = meta_airmass(spectrum.meta, _AIRMASS_DEFAULT, 'the starting telluric scalings (then fitted)')
    pwv = telluric.DEFAULT_PWV
    if template is not None and resolution is None and not spectrum.resolution:
        raise ValueError('a stellar template needs a known resolution')
    tm = telluric.model()
    lo_cov, hi_cov = max(float(wave[0]), tm.wmin), min(float(wave[-1]), tm.wmax)
    r_guess = resolution or spectrum.resolution or _R_GUESS
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
    else:
        log.warning('Telluric zero point: the spectrum has no σ; the segments are fitted unweighted, so their χ² '
                    'and Birge factors are in data units')
    for a, b, name in pieces:
        pad = _PAD_RESEL * 0.5 * (a + b) / r_guess
        lo, hi = max(a - pad, lo_cov), min(b + pad, hi_cov)
        span = (wave >= lo) & (wave <= hi)
        use = good[span]
        if template is None:
            use &= ~_stellar_mask(wave[span], r_guess, _MASK_RESEL)
        info = dict(band=name, wmin=float(a), wmax=float(b), centre=float(np.sqrt(a * b)), dv=np.nan, err=np.nan,
                    used=False, reason='', n_pix=int(use.sum()))
        meta_segs.append(info)
        if use.sum() < _MIN_PIXELS:
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
    mean, formal, chi2 = weighted_mean_chi2(v, e)
    birge = float(np.sqrt(max(1., chi2 / (len(v) - 1)))) if len(v) > 1 else 1.
    if len(v) > 1:
        inflate = float(np.median([abs(v[i + 1] - v[i]) / (x[i + 1] - x[i]) for i in range(len(v) - 1)]))
    else:
        inflate = INFLATE_DEFAULT
        log.warning('Telluric zero point: a single node; its error grows at the default %.2f km/s per Å', inflate)
    zp = TelluricZeroPoint(x, v, e, meta_segs, ff, resolution, r_src, airmass,
                           template=getattr(template, 'label', None) if template is not None else None, zp_kms=mean,
                           zp_err=birge * formal, birge=birge, inflate=inflate,
                           span_bounds={0: (float(min(s['wmin'] for s in used)),
                                            float(max(s['wmax'] for s in used)))})
    log.info('Telluric zero point: Δv(λ) %+.1f to %+.1f km/s over %.0f-%.0f Å from %d of %d segments (node-weighted '
             'mean %+.1f ± %.1f km/s, Birge %.2f; not a single zero point when Δv varies); R %.0f (%s)', v.min(),
             v.max(), x.min(), x.max(), zp.n_used, len(meta_segs), zp.zp_kms, zp.zp_err, birge, resolution, r_src)
    return zp


#: star − sky from the direct sky: at least this many nodes, and the nodes' χ² p-value must exceed STAR_SKY_P_MIN
#: (otherwise the error is Birge-inflated and the result flagged inconsistent)
STAR_SKY_MIN_NODES = 2
STAR_SKY_P_MIN = 0.01


def node_windows(tell: TelluricZeroPoint) -> list[tuple[float, float]]:
    """The (wmin, wmax) [Å, observatory frame] of each telluric node (the used segments, in node order)"""
    return [(float(s['wmin']), float(s['wmax'])) for s in tell.segments if s.get('used')]


def _fit_constant_dar(d: np.ndarray, e: np.ndarray, dar: np.ndarray | None, s_prior: tuple[float, float] | None
                      ) -> dict:
    """
    d_i = a + s·D_i by marginalising over a grid in s: s ≡ 0 without `dar`; uniform in [−1, 1] when `s_prior` is
    None; else Gaussian (s₀, σ_s) truncated to [−1, 1] (a single point when σ_s = 0). Returns a, s, their
    posterior variances and covariance, and the χ² at the posterior means.
    """
    w = 1. / e ** 2
    if dar is None:
        grid, prior = np.zeros(1), np.ones(1)
    elif s_prior is None:
        grid = np.linspace(-1., 1., 2001)
        prior = np.ones(len(grid))
    elif s_prior[1] > 0:
        grid = np.clip(s_prior[0] + s_prior[1] * np.linspace(-5., 5., 401), -1., 1.)
        prior = np.exp(-0.5 * ((grid - s_prior[0]) / s_prior[1]) ** 2)
    else:
        grid, prior = np.array([float(s_prior[0])]), np.ones(1)
    dd = np.zeros(len(d)) if dar is None else dar
    a_hat = np.sum(w * (d - grid[:, None] * dd), axis=1) / np.sum(w)
    chi2 = np.sum(w * (d - a_hat[:, None] - grid[:, None] * dd) ** 2, axis=1)
    post = prior * np.exp(-0.5 * (chi2 - chi2.min()))
    post /= post.sum()
    s_mean, a_mean = float(np.sum(post * grid)), float(np.sum(post * a_hat))
    var_s = float(np.sum(post * (grid - s_mean) ** 2))
    var_a = float(1. / np.sum(w) + np.sum(post * (a_hat - a_mean) ** 2))
    cov = float(np.sum(post * (a_hat - a_mean) * (grid - s_mean)))
    return dict(a=a_mean, s=s_mean, var_a=var_a, var_s=var_s, cov=cov,
                chi2=float(np.sum(w * (d - a_mean - s_mean * dd) ** 2)))


def star_minus_sky_direct(tell: TelluricZeroPoint, sky_windows: Sequence[dict], dar: np.ndarray | None = None,
                          s_prior: tuple[float, float] | None = None) -> dict:
    """
    The star − sky zero point a + s·D(λ) from the sky measured directly in each telluric node's own window

    `sky_windows` holds one `sky_zeropoint.measure_windows` result per node (`node_windows`), so the sky and the
    star are compared over the same pixels, not through the sky curve's interpolation. Each usable node gives
    d_i = Δv_tell,i − Δv_sky,i with error √(σ_tell,i² + σ_sky,i²). `dar` is the across-slit DAR D(λ) at the nodes
    [km/s per unit s] (`dar.dar_kms`; None: a constant, s ≡ 0) and `s_prior` the slit's (s, σ_s) when its PA is
    known (None: s is fitted with a uniform prior in [−1, 1]); see `_fit_constant_dar`. It is measured only with at
    least `STAR_SKY_MIN_NODES` nodes. The χ² of the nodes tests "the star sees the sky's distortion plus a
    constant (and DAR)": when its p-value is ≤ `STAR_SKY_P_MIN`, the errors are inflated by the Birge factor and
    `consistent` is False. With fewer nodes everything is NaN (`reason` says why).
    """
    from scipy.stats import chi2 as chi2_dist
    lam = np.asarray(tell.node_wave, float)
    wins = node_windows(tell)
    if len(sky_windows) != len(lam) or len(wins) != len(lam):
        raise ValueError(f'{len(sky_windows)} sky windows for {len(lam)} telluric nodes')
    if dar is not None and len(dar) != len(lam):
        raise ValueError(f'{len(dar)} DAR values for {len(lam)} telluric nodes')
    s_dv = np.array([float(s.get('dv', np.nan)) for s in sky_windows])
    s_err = np.array([float(s.get('err', np.nan)) for s in sky_windows])
    s_ok = np.array([bool(s.get('used')) for s in sky_windows])
    bands = [str(s.get('band', '')) for s in tell.segments if s.get('used')]
    d = tell.node_dv - s_dv
    e = np.hypot(tell.node_err, s_err)
    ok = s_ok & np.isfinite(d) & np.isfinite(e) & (e > 0)
    nodes = [dict(band=bands[i], wmin=wins[i][0], wmax=wins[i][1], wave=float(lam[i]), tell=float(tell.node_dv[i]),
                  tell_err=float(tell.node_err[i]), sky=float(s_dv[i]), sky_err=float(s_err[i]),
                  sky_reason=str(sky_windows[i].get('reason', '')), diff=float(d[i]), err=float(e[i]),
                  dar=float(dar[i]) if dar is not None else 0., used=bool(ok[i])) for i in range(len(lam))]
    nan = float('nan')
    mode = 'none' if dar is None else ('fitted' if s_prior is None else 'known')
    base = dict(method='direct sky', nodes=nodes, dar_mode=mode)
    if ok.sum() < STAR_SKY_MIN_NODES:
        reason = (f'{int(ok.sum())} telluric node(s) with a measured sky in the same window (< {STAR_SKY_MIN_NODES}'
                  f'; ' + ', '.join(f"{n['band']} {n['wmin']:.0f}-{n['wmax']:.0f} Å: {n['sky_reason'] or 'no sky'}"
                                     for n in nodes if not n['used']) + ')')
        log.info('Star - sky zero point not measured: %s', reason)
        return dict(zp_star_sky=nan, zp_star_sky_err=nan, s=nan, s_err=nan, cov_as=nan, n_nodes=int(ok.sum()),
                    chi2=nan, dof=0, p=nan, birge=nan, consistent=False, reason=reason, **base)
    fit = _fit_constant_dar(d[ok], e[ok], None if dar is None else dar[ok], s_prior)
    dof = int(ok.sum()) - 1 - (mode == 'fitted')
    p = float(chi2_dist.sf(fit['chi2'], dof)) if dof > 0 else nan
    consistent = not p <= STAR_SKY_P_MIN
    birge = 1. if consistent else float(np.sqrt(fit['chi2'] / dof))
    err, s_err_post = birge * np.sqrt(fit['var_a']), birge * np.sqrt(fit['var_s'])
    log.info('Star - sky zero point: %+.1f ± %.1f km/s%s from %d telluric node(s) with the sky measured in the same '
             'windows (%s); χ² %.1f for %d dof, p = %.2g%s', fit['a'], err,
             '' if mode == 'none' else f" + s·D(λ) with s = {fit['s']:+.3f} ± {s_err_post:.3f} ({mode})",
             int(ok.sum()), ', '.join(f"{n['band']} {n['diff']:+.1f} ± {n['err']:.1f}" for n in nodes if n['used']),
             fit['chi2'], dof, p, '' if consistent else f' ≤ {STAR_SKY_P_MIN:g}: the nodes disagree, errors '
                                                        f'inflated by the Birge factor {birge:.2f}')
    return dict(zp_star_sky=fit['a'], zp_star_sky_err=float(err), s=fit['s'] if dar is not None else nan,
                s_err=float(s_err_post) if dar is not None else nan, cov_as=birge ** 2 * fit['cov'],
                n_nodes=int(ok.sum()), chi2=fit['chi2'], dof=dof, p=p, birge=birge, consistent=bool(consistent),
                reason='ok' if consistent else 'inconsistent nodes (Birge-inflated)', **base)
