"""
Wavelength zero point from the OH night-sky emission lines.

The sky spectrum extracted alongside the object (e.g. PypeIt `OPT_COUNTS_SKY`) shares the object's wavelength
solution, so its OH lines measure the solution's zero point and its variation with wavelength. The sky is
uniformly illuminated, so it cannot see a star's own offset from slit centring; that constant comes from the
telluric absorption lines (`telluric_zeropoint`).

Method:
1. Work in the observatory frame: any velocity correction already applied to the wavelengths in the header
   (PypeIt `VEL_CORR`) is undone.
2. Resample the sky onto a uniform ln λ grid and subtract a running low percentile, leaving the emission lines.
3. Build a model sky from the packaged vacuum OH line list (`data/oh_sky_vacuum.csv`: PypeIt's Rousselot
   et al. 2000 model, BSD-3-Clause), with Gaussian lines of the observed sky FWHM, measured by maximising the
   correlation over trial widths.
4. Cross-correlate data and model segment by segment (default 250 Å), fit the peak with a parabola, and take
   the Zucker (2003) error. Segments that overlap strong telluric absorption (which eats the OH emission), have
   too few lines, or correlate poorly are rejected.
5. Fit a low-order polynomial Δv(λ) to the segments (degree 0–3 chosen by BIC, keeping at least one degree of
   freedom), with the covariance scaled by max(1, χ²/ν), so the segment scatter sets the uncertainty.

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
from ..constants import C_KMS
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


@dataclass
class SkyZeroPoint:
    """The measured zero-point curve Δv(λ) = Σ c_k x^k, x = (λ − ref_wave)/1000 Å [km/s]"""
    ref_wave: float
    coef: np.ndarray
    cov: np.ndarray
    fwhm: float
    segments: list[dict] = field(default_factory=list)
    frame_factor: float = 1.
    wmin_fit: float = -np.inf  # Δv is held constant outside the span of the used segment centres
    wmax_fit: float = np.inf

    @property
    def zp_kms(self) -> float:
        return float(self.coef[0])

    @property
    def zp_err(self) -> float:
        return float(np.sqrt(self.cov[0, 0]))

    @property
    def slope(self) -> float:
        """km/s per 1000 Å (NaN for a constant fit)"""
        return float(self.coef[1]) if len(self.coef) > 1 else np.nan

    @property
    def slope_err(self) -> float:
        return float(np.sqrt(self.cov[1, 1])) if len(self.coef) > 1 else np.nan

    @property
    def n_used(self) -> int:
        return sum(1 for s in self.segments if s['used'])

    def _x(self, wave) -> np.ndarray:
        return (np.clip(np.asarray(wave, dtype=float), self.wmin_fit, self.wmax_fit) - self.ref_wave) / 1000.

    def dv(self, wave: np.ndarray | float) -> np.ndarray:
        """Δv at `wave` [km/s]; constant beyond the used segments (no extrapolation)"""
        return np.polynomial.polynomial.polyval(self._x(wave), self.coef)

    def dv_err(self, wave: np.ndarray | float) -> np.ndarray:
        x = np.atleast_1d(self._x(wave))
        jac = np.vander(x, len(self.coef), increasing=True)
        return np.sqrt(np.einsum('ij,jk,ik->i', jac, self.cov, jac))

    def as_dict(self) -> dict:
        return dict(zp_sky_kms=self.zp_kms, zp_sky_err=self.zp_err, zp_sky_slope=self.slope,
                    zp_sky_slope_err=self.slope_err, ref_wave=self.ref_wave, coef=[float(c) for c in self.coef],
                    wmin_fit=self.wmin_fit, wmax_fit=self.wmax_fit,
                    fwhm=self.fwhm, n_used=self.n_used, segments=self.segments)


def _emission(flux: np.ndarray, width_pix: int) -> np.ndarray:
    """Sky emission above a running 20th-percentile baseline"""
    good = np.isfinite(flux)
    f = np.where(good, flux, np.interp(np.arange(len(flux)), np.flatnonzero(good), flux[good]))
    return np.where(good, f - percentile_filter(f, 20, size=max(width_pix, 5)), np.nan)


def _xc(d: np.ndarray, t: np.ndarray, idx: np.ndarray, maxlag: int) -> tuple[float, float, float]:
    """Lag (pixels, parabola-refined) and peak correlation of d against t over the pixels idx; NaN on failure"""
    lags = np.arange(-maxlag, maxlag + 1)
    ok = (idx - maxlag >= 0) & (idx + maxlag < len(t))
    idx = idx[ok]
    a = d[idx]
    good = np.isfinite(a)
    idx, a = idx[good], a[good]
    if len(idx) < 20:
        return np.nan, np.nan, np.nan
    taper = np.hanning(len(idx) + 2)[1:-1]  # lines cut by the segment edges get little weight
    a = (a - a.mean()) * taper
    c = np.empty(len(lags))
    for k, lag in enumerate(lags):
        b = (t[idx - lag] - t[idx - lag].mean()) * taper
        c[k] = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))
    i = int(np.nanargmax(c))
    if i < 3 or i > len(c) - 4:
        return np.nan, float(c[i]), np.nan
    p2, p1, _ = np.polyfit(lags[i - 3:i + 4] - lags[i], c[i - 3:i + 4], 2)
    if p2 >= 0:
        return np.nan, float(c[i]), np.nan
    x0 = -p1 / (2 * p2)
    peak = float(np.polyval([p2, p1, _], x0))
    return float(lags[i] + x0), peak, float(2 * p2)


def measure(spectrum: Spectrum, *, sky: np.ndarray | None = None, seg_width: float = 250., degree: int | None = None,
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
    degree
        Polynomial degree of Δv(λ); None (default) chooses it by BIC among 0–3 (at most n − 2 for n usable
        segments)
    min_corr
        Minimum peak correlation of a usable segment
    max_lag_kms
        Search range [km/s]
    telluric_depth
        Segments overlapping telluric bands at least this deep (catalogue depth) by more than 20 per cent are
        rejected: the absorption distorts the OH emission
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
    grid = broadening.log_wave_grid(wave[good][0], wave[good][-1], _DV)
    data = broadening.rebin(wave[good], sky[good], grid)
    pix_per_a = C_KMS / _DV / float(np.median(grid))
    d = _emission(data, int(40. * pix_per_a))
    if fwhm is None:  # the width maximising the correlation peak (over lags, so a zero-point shift does not bias it)
        trials = np.geomspace(1., 15., 25)
        heights = []
        all_idx = np.arange(len(grid))
        for fw in trials:
            m = _emission(model_sky(grid, fw), int(40. * pix_per_a))
            heights.append(_xc(d, m, all_idx, int(np.ceil(max_lag_kms / _DV)))[1])
        i = int(np.argmax(heights))
        fwhm = float(trials[i])
        if 0 < i < len(trials) - 1:
            x = np.log(trials[i - 1:i + 2])
            p = np.polyfit(x, heights[i - 1:i + 2], 2)
            if p[0] < 0:
                fwhm = float(np.exp(-p[1] / (2 * p[0])))
    model = _emission(model_sky(grid, fwhm), int(40. * pix_per_a))
    lw, la = oh_lines()
    tell = [(f.wmin, f.wmax) for f in features() if f.kind == 'telluric' and np.isfinite(f.depth)
            and f.depth >= telluric_depth]
    maxlag = int(np.ceil(max_lag_kms / _DV))
    n_seg = max(1, int(round((grid[-1] - grid[0]) / seg_width)))
    edges = np.linspace(grid[0], grid[-1], n_seg + 1)
    segs = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        seg = dict(wmin=float(lo), wmax=float(hi), centre=float(0.5 * (lo + hi)), dv=np.nan, err=np.nan, corr=np.nan,
                   used=False, reason='')
        overlap = sum(max(0., min(hi, b) - max(lo, a)) for a, b in tell) / (hi - lo)
        inseg = (lw >= lo) & (lw <= hi)
        n_lines = int(np.sum(inseg & (la >= 0.1 * la[inseg].max()))) if inseg.any() else 0
        idx = np.flatnonzero((grid >= lo) & (grid <= hi))
        if overlap > 0.2:
            seg['reason'] = f'telluric absorption ({overlap:.0%} of the segment)'
        elif n_lines < min_lines:
            seg['reason'] = f'{n_lines} strong OH lines'
        else:
            lag, corr, curv = _xc(d, model, idx, maxlag)
            seg['corr'] = corr
            if not np.isfinite(lag):
                seg['reason'] = 'no correlation peak'
            elif corr < min_corr:
                seg['reason'] = f'correlation {corr:.2f} < {min_corr}'
            else:
                n = int(np.sum(np.isfinite(d[idx])))
                var = -1. / (n * curv / corr * corr ** 2 / (1. - corr ** 2)) if 0 < corr < 1 else np.nan
                seg.update(dv=lag * _DV, err=float(np.sqrt(var)) * _DV if np.isfinite(var) and var > 0 else np.nan,
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
    w = 1. / e ** 2
    ref = float(np.sum(w * x_c) / np.sum(w))
    x = (x_c - ref) / 1000.

    def fit(deg):
        a = np.vander(x, deg + 1, increasing=True)
        aw = a * np.sqrt(w)[:, None]
        c, *_ = np.linalg.lstsq(aw, v * np.sqrt(w), rcond=None)
        return c, np.linalg.inv(aw.T @ aw), float(np.sum(w * (v - a @ c) ** 2))

    if degree is None:  # choose by BIC among 0 .. min(3, n - 2), so at least one degree of freedom remains
        fits = {d: fit(d) for d in range(0, max(0, min(3, len(used) - 2)) + 1)}
        deg = min(fits, key=lambda d: fits[d][2] + (d + 1) * np.log(len(used)))
    else:
        deg = min(degree, len(used) - 1)
    coef, cov, chi2 = fit(deg)
    dof = len(used) - (deg + 1)
    if dof > 0:
        cov = cov * max(1., chi2 / dof)
    zp = SkyZeroPoint(ref, coef, cov, fwhm, segs, ff, float(x_c.min()), float(x_c.max()))
    log.info('Sky zero point: %+.1f ± %.1f km/s at %.0f Å, slope %+.1f ± %.1f km/s per 1000 Å, degree %d (%d of %d '
             'segments, sky FWHM %.2f Å)', zp.zp_kms, zp.zp_err, ref, zp.slope, zp.slope_err, len(coef) - 1,
             zp.n_used, len(segs), fwhm)
    for s in segs:
        log.debug('  sky segment %.0f-%.0f Å: %+.1f ± %.1f km/s, corr %.2f, %s', s['wmin'], s['wmax'], s['dv'],
                  s['err'], s['corr'], s['reason'])
    return zp


def apply(spectrum: Spectrum, zp: SkyZeroPoint | None = None, *, constant_kms: float = 0.,
          source: str = 'sky') -> Spectrum:
    """
    Correct the wavelengths for a zero point: λ' = λ / D(Δv(λ) + constant), D the relativistic Doppler factor

    Parameters
    ----------
    spectrum
        The spectrum (in any frame: the correction is a multiplicative factor)
    zp
        The sky zero-point curve (None: only the constant)
    constant_kms
        An additional constant offset [km/s] (e.g. the star's own offset from tellurics)
    source
        Label recorded in meta['wavelength_correction']['source']

    Returns
    -------
    corrected
        The spectrum with corrected wavelengths; meta['wavelength_correction'] records what was applied
    """
    dv = np.full(len(spectrum.wave), float(constant_kms))
    if zp is not None:
        dv = dv + zp.dv(spectrum.wave)
    new_wave = spectrum.wave / broadening.doppler_factor(dv)
    meta = dict(spectrum.meta)
    meta['wavelength_correction'] = dict(source=source, constant_kms=float(constant_kms),
                                         sky=None if zp is None else zp.as_dict(),
                                         dv_min=float(dv.min()), dv_max=float(dv.max()))
    return replace(spectrum, wave=new_wave, meta=meta)


def applied_dv(meta: Any, wave: float | np.ndarray) -> np.ndarray:
    """The velocity correction applied at `wave` according to meta['wavelength_correction'] (0 if none)"""
    wc = (meta or {}).get('wavelength_correction')
    if not wc:
        return np.zeros_like(np.asarray(wave, dtype=float))
    dv = np.full_like(np.asarray(wave, dtype=float), wc.get('constant_kms', 0.))
    sky = wc.get('sky')
    if sky:
        x = (np.clip(np.asarray(wave, dtype=float), sky.get('wmin_fit', -np.inf), sky.get('wmax_fit', np.inf))
             - sky['ref_wave']) / 1000.
        dv = dv + np.polynomial.polynomial.polyval(x, sky['coef'])
    return dv
