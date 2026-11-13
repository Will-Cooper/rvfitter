"""
Unresolved-binary test: one template against two shifted copies with a flux ratio, compared by ΔBIC.

Over the template-fit windows (same template choice and per-window polynomial continuum P as `template_fit`), the
single model is P · T(v) and the double model P · (T(v₁) + q T(v₂)), with v, v₁, v₂ and q ∈ (0, 1] shared by every
window. Each is found on a velocity grid (and a q grid), then refined (bounded scalar / Nelder–Mead).
ΔBIC = (χ²₁ − χ²₂)/s − 2 ln N, with N the number of pixels, two extra parameters and s = max(1, χ²₁/ν₁): the
single fit's reduced χ² rescales both, so template mismatch on real data is not read as a second star.
The object is flagged when ΔBIC > `DBIC_THRESHOLD` = 10 ("very strong" evidence; Kass & Raftery 1995, JASA 90, 773).
"""
import logging
from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from ..constants import C_KMS
from ..types import Region, Spectrum
from . import _common as C
from . import template_fit as TF

__all__ = ['test', 'DBIC_THRESHOLD']

log = logging.getLogger(__name__)

DBIC_THRESHOLD = 10.
SPAN_KMS = 300.  # v₁ and v₂ are searched within the single-star RV ± this
N_GRID = 49  # velocity grid points over that span (coarse search; refined afterwards)
Q_GRID = np.geomspace(0.1, 1., 8)


class _Window:
    """One window's weighted data, polynomial basis and template"""

    def __init__(self, d: TF._Data, tpl: C.Template, degree: int):
        self.d, self.tpl = d, tpl
        self.sw = np.sqrt(d.w)
        self.basis = np.vander(d.x, degree + 1, increasing=True) * self.sw[:, None]
        self.y = d.flux * self.sw

    def chi2(self, t: np.ndarray) -> float:
        """χ² of the best P · t (t: template flux on the pixels); inf if t is not finite"""
        if not np.all(np.isfinite(t)):
            return np.inf
        a = self.basis * t[:, None]
        coef, *_ = np.linalg.lstsq(a, self.y, rcond=None)
        return float(np.sum((self.y - a @ coef) ** 2))

    def stack(self, v: np.ndarray) -> np.ndarray:
        return self.tpl.at(self.d.wave, v)  # (len(v), n_pix)


def _single(wins: list[_Window], v: float) -> float:
    return sum(w.chi2(w.tpl.at(w.d.wave, v)) for w in wins)


def _double(wins: list[_Window], p: np.ndarray) -> float:
    v1, v2, q = p
    return sum(w.chi2(w.tpl.at(w.d.wave, v1) + q * w.tpl.at(w.d.wave, v2)) for w in wins)


def _grid_double(wins: list[_Window], vgrid: np.ndarray) -> tuple[float, float, float, float]:
    """Best (χ², v₁, v₂, q) on the grid, by normal equations batched over every (v₁, v₂) pair"""
    total = np.zeros((len(Q_GRID), len(vgrid), len(vgrid)))
    for w in wins:
        t = w.stack(vgrid)
        bad = ~np.all(np.isfinite(t), axis=1)
        t = np.nan_to_num(t)
        bt = w.basis[None, :, :] * t[:, :, None]  # (n_v, n_pix, n_c)
        g = np.einsum('aic,bid->abcd', bt, bt)
        h = np.einsum('aic,i->ac', bt, w.y)
        yy = float(w.y @ w.y)
        gd = g.diagonal(axis1=0, axis2=1).transpose(2, 0, 1)  # g[i, i]
        for k, q in enumerate(Q_GRID):  # basis B_i + q B_j: G = g_ii + q (g_ij + g_ji) + q² g_jj
            gq = gd[:, None] + q * (g + g.transpose(1, 0, 2, 3)) + q * q * gd[None, :]
            hq = h[:, None, :] + q * h[None, :, :]
            sol = np.linalg.solve(gq + 1e-12 * np.eye(gq.shape[-1]), hq[..., None])[..., 0]
            c2 = yy - np.einsum('abc,abc->ab', hq, sol)
            c2[bad, :] = np.inf
            c2[:, bad] = np.inf
            total[k] += c2
    k, i, j = np.unravel_index(int(np.argmin(total)), total.shape)
    return float(total[k, i, j]), float(vgrid[i]), float(vgrid[j]), float(Q_GRID[k])


def test(spectrum: Spectrum, regions: Sequence[Region], cfg: Any) -> dict:
    """
    Single- versus two-component template fit; returns 'dbic', 'flag', the single ('rv', 'chi2') and double
    ('rv1', 'rv2', 'q', 'chi2') solutions, 'n_pix', 'chi2_red' and 'template'
    """
    opts = C.options(cfg, 'template_fit', TF.DEFAULTS)
    resolution = C.resolution_of(spectrum, cfg, opts)
    cands = C.candidates(cfg, opts)
    datas, _ = TF._prepare(spectrum, C.covered_regions('binary', regions, cands), opts, resolution)
    if not datas:
        raise ValueError(f'binary test: no template-fit window has {opts["poly_degree"] + 10} usable pixels')
    rv_grid = TF._rv_grid(spectrum, opts)
    wave_range = (min(d.wave[0] for d in datas), max(d.wave[-1] for d in datas))
    score = partial(TF._score_candidate, datas=datas, resolution=resolution, wave_range=wave_range, opts=opts,
                    rv_grid=rv_grid)
    tpl, _, _ = C.choose_template('binary', cands, score, cfg, opts, resolution, wave_range,
                                  sum(len(d.wave) for d in datas), None)
    wins = [_Window(d, tpl, opts['poly_degree']) for d in datas]
    n_pix = sum(len(d.wave) for d in datas)

    scan = np.array([_single(wins, v) for v in rv_grid])
    v0 = float(rv_grid[int(np.nanargmin(scan))])
    step = rv_grid[1] - rv_grid[0]
    r = minimize_scalar(lambda v: _single(wins, v), bounds=(v0 - 2 * step, v0 + 2 * step), method='bounded')
    v_single, chi2_1 = float(r.x), float(r.fun)

    vgrid = np.linspace(v_single - SPAN_KMS, v_single + SPAN_KMS, N_GRID)
    _, v1, v2, q = _grid_double(wins, vgrid)
    best = None
    bounds = [(vgrid[0], vgrid[-1])] * 2 + [(1e-3, 1.)]
    for start in ((v1, v2, q), (v_single, v_single + 2 * C_KMS / resolution, 0.5),
                  (v_single, v_single - 2 * C_KMS / resolution, 0.5)):
        res = minimize(lambda p: _double(wins, p), np.array(start), method='Nelder-Mead', bounds=bounds,
                       options=dict(xatol=1e-3, fatol=1e-6, maxiter=4000))
        if best is None or res.fun < best.fun:
            best = res
    chi2_2 = min(float(best.fun), chi2_1)  # nested: the double model is never worse than the single one
    v1, v2, q = (float(x) for x in best.x)

    nu1 = n_pix - len(wins) * (opts['poly_degree'] + 1) - 1
    red = chi2_1 / max(nu1, 1)
    scale = max(1., red) if wins[0].d.sigma_known else red
    dbic = (chi2_1 - chi2_2) / scale - 2. * np.log(n_pix)
    flag = bool(dbic > DBIC_THRESHOLD)
    (log.warning if flag else log.info)(
        'binary test: ΔBIC %.1f (%s; threshold %.0f): single %.1f km/s, double %.1f and %.1f km/s, flux ratio %.2f '
        '(χ²/ν %.2f, %d pixels, template %s)', dbic, 'flagged' if flag else 'not flagged', DBIC_THRESHOLD,
        v_single, v1, v2, q, red, n_pix, tpl.label)
    return dict(dbic=float(dbic), flag=flag, rv=v_single, chi2_single=chi2_1, rv1=v1, rv2=v2, q=q,
                chi2_double=chi2_2, n_pix=n_pix, chi2_red=red, template=tpl.label)
