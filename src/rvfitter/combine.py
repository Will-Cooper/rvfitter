"""
Combination of per-region and per-method radial velocities.

- `weighted_mean` / `combine_regions`: the inverse-variance mean of one method's regions, with the formal error
  inflated by the Birge ratio √(χ²/(N − 1)) when the regions scatter more than their errors allow.
- `hierarchical`: the adopted RV from a line × method model, v_{m,i} = μ + δ_i + β_m + ε_{m,i}, with δ and β
  marginalised analytically (Woodbury identity) and the two scatter terms τ_line and τ_method integrated on a
  deterministic grid under half-Cauchy priors. The adopted value is the posterior median of μ and its error
  half the central 68 per cent interval.

Both are ports of the Phase A legacy functions (`legacy/utils.py` `combine_rvs` and `hierarchical_combine`),
reproducing them to ≤1e-8 km/s.
"""
import logging
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr

from .types import Measurement, MethodResult

__all__ = ['weighted_mean', 'combine_regions', 'hierarchical', 'hierarchical_from_methods', 'mixture_pdf']

log = logging.getLogger(__name__)

_Q16, _Q84 = float(ndtr(-1.)), float(ndtr(1.))


def weighted_mean(rvs: Sequence[float], errs: Sequence[float]) -> tuple[float, float]:
    """
    Inverse-variance weighted mean with the Birge-ratio error inflation

    Parameters
    ----------
    rvs
        Radial velocities [km/s]; entries that are non-finite, or whose error is non-finite or non-positive,
        are ignored
    errs
        Their 1σ uncertainties [km/s]

    Returns
    -------
    rv, err
        The weighted mean, and its error: the single value's own error for N = 1, otherwise the formal error
        1/√Σw multiplied by max(1, √(χ²/(N − 1))) with χ² = Σw(v − mean)². NaN, NaN if nothing is usable.
    """
    rvs, errs = np.asarray(rvs, dtype=float), np.asarray(errs, dtype=float)
    good = np.isfinite(rvs) & np.isfinite(errs) & (errs > 0)
    rvs, errs = rvs[good], errs[good]
    if not len(rvs):
        return np.nan, np.nan
    if len(rvs) == 1:
        return float(rvs[0]), float(errs[0])
    weights = 1. / errs ** 2
    rv = np.average(rvs, weights=weights)
    formal = 1. / np.sqrt(np.sum(weights))
    chisq = np.sum(weights * (rvs - rv) ** 2)
    return float(rv), float(formal * max(1., np.sqrt(chisq / (len(rvs) - 1))))


def combine_regions(measurements: Iterable[Measurement]) -> tuple[float, float]:
    """
    Combine one method's per-region measurements (Birge-inflated inverse-variance mean; see `weighted_mean`)

    Parameters
    ----------
    measurements
        The per-region measurements; every finite one is used, whatever its flag (callers filter first)

    Returns
    -------
    rv, err
        The combined RV and its error [km/s]
    """
    measurements = list(measurements)
    return weighted_mean([m.rv for m in measurements], [m.rv_err for m in measurements])


def _half_cauchy_logpdf(tau: np.ndarray, scale: float) -> np.ndarray:
    """Log density of a half-Cauchy distribution of the given scale, for tau ≥ 0"""
    return np.log(2. / (np.pi * scale)) - np.log1p((tau / scale) ** 2)


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Trapezoidal quadrature weights for an increasing, possibly non-uniform, grid"""
    w = np.zeros_like(x)
    dx = np.diff(x)
    w[:-1] += dx / 2
    w[1:] += dx / 2
    return w


def mixture_pdf(x: np.ndarray, means: np.ndarray, sds: np.ndarray, weights: np.ndarray,
                extra_sd: float = 0.) -> np.ndarray:
    """
    Density of a mixture of Gaussians, e.g. the posterior of μ returned by `hierarchical`

    Parameters
    ----------
    x
        Where to evaluate the density
    means, sds, weights
        Component means, standard deviations and normalised weights
    extra_sd
        An extra standard deviation added in quadrature to every component (e.g. a systematic)

    Returns
    -------
    pdf
        The density at x
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    sds = np.sqrt(np.asarray(sds, dtype=float) ** 2 + extra_sd ** 2)
    weights = np.asarray(weights, dtype=float)
    keep = weights > 1e-12 * np.max(weights)
    m, s, w = np.asarray(means, dtype=float)[keep], sds[keep], weights[keep]
    norm = w / (np.sqrt(2 * np.pi) * s)
    pdf = np.empty(len(x))
    for start in range(0, len(x), 256):  # chunked over x to bound the memory
        z = (x[start:start + 256, None] - m[None, :]) / s[None, :]
        pdf[start:start + 256] = np.sum(norm * np.exp(-0.5 * z ** 2), axis=1)
    return pdf


def hierarchical(values: Sequence[float], errors: Sequence[float], line_ids: Sequence, method_ids: Sequence, *,
                 tau_scale: float | None = None, ngrid: int = 200, tau_max_factor: float = 10.,
                 tau_line: float | None = None, tau_method: float | None = None) -> dict:
    """
    Adopted RV from a hierarchical line × method model

    The model is v_{m,i} = μ + δ_i + β_m + ε_{m,i}, with δ_i ~ N(0, τ_line²) shared by all methods measuring
    line (region) i, β_m ~ N(0, τ_method²) shared by all lines measured by method m, and ε ~ N(0, σ²).
    Marginalising δ and β gives v ~ N(μ1, C), C = diag(σ²) + τ_line² [same line] + τ_method² [same method].
    With a flat prior on μ, its conditional posterior is Gaussian for fixed τ; the τ have half-Cauchy priors
    and are integrated on a fixed grid (τ = 0 plus ngrid − 1 log-spaced points from 1e-3 × scale to τ_max,
    trapezoidal weights), so the posterior of μ is a mixture of Gaussians. No random numbers are used.
    C⁻¹ and |C| come from the Woodbury identity and the matrix determinant lemma, so memory scales with
    ngrid × (n_lines + n_methods)² rather than with the number of measurements squared.
    A factor with a single level (one line, or one method) has an offset that cannot be separated from μ, so
    its τ is fixed at 0 unless given explicitly, and a warning is logged.

    Parameters
    ----------
    values
        RV measurements [km/s] (entries with non-finite values or non-finite/non-positive errors are dropped)
    errors
        Their 1σ uncertainties [km/s]
    line_ids
        The line or region of each measurement (hashable labels)
    method_ids
        The method of each measurement (hashable labels)
    tau_scale
        Scale of both half-Cauchy priors; by default the median measurement error (a mild empirical Bayes)
    ngrid
        Grid points per τ axis, including τ = 0
    tau_max_factor
        The τ grid reaches at least this many prior scales (and at least three times the range of the values)
    tau_line, tau_method
        Fix that τ at this value instead of integrating over it

    Returns
    -------
    result
        'median' and 'err' (half of q84 − q16) are the adopted RV and its error; also 'q16', 'q84', the
        posterior 'mean' and 'sd' of μ, the posterior medians 'tau_line' and 'tau_method', 'tau_scale', the
        counts 'n', 'n_lines' and 'n_methods', and the mixture components 'means', 'sds', 'weights'
    """
    values, errors = np.asarray(values, dtype=float), np.asarray(errors, dtype=float)
    line_ids, method_ids = np.asarray(line_ids, dtype=object), np.asarray(method_ids, dtype=object)
    good = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    values, errors, line_ids, method_ids = values[good], errors[good], line_ids[good], method_ids[good]
    n = len(values)
    out = dict(median=np.nan, err=np.nan, q16=np.nan, q84=np.nan, mean=np.nan, sd=np.nan, tau_line=np.nan,
               tau_method=np.nan, tau_scale=np.nan, n=n, n_lines=len(set(line_ids)),
               n_methods=len(set(method_ids)), means=np.array([]), sds=np.array([]), weights=np.array([]))
    if not n:
        return out
    if tau_scale is None:
        tau_scale = float(np.median(errors))
    if not np.isfinite(tau_scale) or tau_scale <= 0:
        raise ValueError(f'tau_scale must be positive and finite, not {tau_scale}')
    out['tau_scale'] = tau_scale
    tau_max = max(tau_max_factor * tau_scale, 3. * float(np.ptp(values)))

    def taugrid(fixed):
        if fixed is not None:
            return np.array([float(fixed)]), np.zeros(1), np.ones(1)
        grid = np.concatenate([[0.], np.logspace(np.log10(1e-3 * tau_scale), np.log10(tau_max), ngrid - 1)])
        return grid, _half_cauchy_logpdf(grid, tau_scale), _trapezoid_weights(grid)

    if out['n_lines'] == 1 and tau_line is None and n > 1:
        log.warning('Only one line/region: tau_line is fixed at 0 (the line systematic is not estimable)')
    if out['n_methods'] == 1 and tau_method is None and n > 1:
        log.warning('Only one method: tau_method is fixed at 0 (the between-method systematic is not estimable)')
    if out['n_lines'] == 1 and tau_line is None:
        tau_line = 0.
    if out['n_methods'] == 1 and tau_method is None:
        tau_method = 0.
    tl, logprior_l, quad_l = taugrid(tau_line)
    tm, logprior_m, quad_m = taugrid(tau_method)
    # C = D + U S S U^T, D = diag(σ²), U = [line indicators | method indicators] (n × r), S = diag(τ...).
    # With W = I + S U^T D^-1 U S (r × r): x^T C^-1 y = x^T D^-1 y − (S U^T D^-1 x)^T W^-1 (S U^T D^-1 y)
    # and |C| = |D| |W|; τ = 0 is allowed.
    _, linv = np.unique(line_ids.astype(str), return_inverse=True)
    _, minv = np.unique(method_ids.astype(str), return_inverse=True)
    nl, nm = linv.max() + 1, minv.max() + 1
    umat = np.zeros((n, nl + nm))
    umat[np.arange(n), linv] = 1.
    umat[np.arange(n), nl + minv] = 1.
    dinv = 1. / errors ** 2
    amat = umat.T @ (dinv[:, None] * umat)
    p1, pv = umat.T @ dinv, umat.T @ (dinv * values)
    a0, b0, c0 = np.sum(dinv), np.sum(dinv * values), np.sum(dinv * values ** 2)
    logdet_d = np.sum(np.log(errors ** 2))
    a, b, vcv, logdet = (np.empty((len(tl), len(tm))) for _ in range(4))
    eye = np.eye(nl + nm)
    for k, tline in enumerate(tl):  # one τ_line at a time keeps the memory at len(tm) × r × r
        smat = np.concatenate([np.full((len(tm), nl), tline), np.repeat(tm[:, None], nm, axis=1)], axis=1)
        wmat = eye[None] + smat[:, :, None] * amat[None] * smat[:, None, :]
        rhs = np.stack([smat * p1, smat * pv], axis=-1)
        sol = np.linalg.solve(wmat, rhs)
        a[k] = a0 - np.sum(rhs[..., 0] * sol[..., 0], axis=-1)  # 1^T C^-1 1
        b[k] = b0 - np.sum(rhs[..., 0] * sol[..., 1], axis=-1)  # 1^T C^-1 v
        vcv[k] = c0 - np.sum(rhs[..., 1] * sol[..., 1], axis=-1)  # v^T C^-1 v
        logdet[k] = logdet_d + np.linalg.slogdet(wmat)[1]
    loglike = -0.5 * (logdet + np.log(a) + vcv - b ** 2 / a)
    with np.errstate(divide='ignore'):  # zero quadrature weight at the end of a grid
        logpost = loglike + (logprior_l + np.log(quad_l))[:, None] + (logprior_m + np.log(quad_m))[None, :]
    weights = np.exp(logpost - np.max(logpost))
    weights /= weights.sum()
    means, sds = b / a, np.sqrt(1. / a)

    mean = float(np.sum(weights * means))
    sd = float(np.sqrt(np.sum(weights * (sds ** 2 + means ** 2)) - mean ** 2))
    keep = weights > 1e-12 * weights.max()
    km, ks, kw = means[keep], sds[keep], weights[keep] / weights[keep].sum()

    def quantile(q):
        lo, hi = np.min(km - 10 * ks), np.max(km + 10 * ks)
        return brentq(lambda x: np.sum(kw * ndtr((x - km) / ks)) - q, lo, hi, xtol=1e-10)

    def tau_median(grid, marginal):
        if len(grid) == 1:
            return float(grid[0])
        cdf = np.cumsum(marginal) - marginal / 2  # mid-point rule
        return float(np.interp(0.5, cdf, grid))

    q16, q84 = quantile(_Q16), quantile(_Q84)
    out.update(median=quantile(0.5), err=(q84 - q16) / 2., q16=q16, q84=q84, mean=mean, sd=sd,
               tau_line=tau_median(tl, weights.sum(axis=1)), tau_method=tau_median(tm, weights.sum(axis=0)),
               means=means.ravel(), sds=sds.ravel(), weights=weights.ravel())
    return out


def hierarchical_from_methods(methods: Mapping[str, MethodResult], *, include_flagged: bool = False,
                              **kwargs) -> dict:
    """
    Run `hierarchical` over every per-region measurement of every method

    The region name is the line label, so regions shared between methods share a line offset.

    Parameters
    ----------
    methods
        Method name → MethodResult
    include_flagged
        Also use measurements with a non-zero flag
    **kwargs
        Passed to `hierarchical`

    Returns
    -------
    result
        As `hierarchical`
    """
    values, errors, lines, meths = [], [], [], []
    for name, res in methods.items():
        for m in res.measurements:
            if m.flag and not include_flagged:
                continue
            values.append(m.rv)
            errors.append(m.rv_err)
            lines.append(m.region)
            meths.append(name)
    return hierarchical(values, errors, lines, meths, **kwargs)
