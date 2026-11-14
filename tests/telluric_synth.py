"""
A tiny synthetic telluric transmission model for the tests (the HITRAN-derived model is built on demand and never
shipped). It has the real model's format and band layout, not its line list: O2 bands of regularly spaced lines and
H2O bands of randomly placed lines (fixed seed), Gaussian in τ, so fits, masks and shifts behave as with the real
model.
"""
import json
from pathlib import Path

import numpy as np

R_S = 100_000  # stored sampling
WMIN, WMAX = 6000., 25000.
# (wmin, wmax, line spacing or density [per Å], peak τ at airmass 1 / per mm PWV, σ [Å], species)
O2_BANDS = ((6867., 6905., 0.9, 1.5, 0.25), (7594., 7700., 0.9, 3.0, 0.25))
H2O_BANDS = ((6900., 7000., 0.3, 0.02, 0.15), (7162., 7342., 0.6, 0.12, 0.15), (8122., 8352., 0.5, 0.06, 0.15),
             (8952., 9903., 0.8, 0.25, 0.15), (11100., 11600., 0.8, 0.15, 0.2), (13300., 14900., 1.0, 1.0, 0.3),
             (17900., 20500., 1.0, 1.0, 0.4))


def _add(tau: np.ndarray, lnw: np.ndarray, centres: np.ndarray, depths: np.ndarray, sigma: float) -> None:
    w = np.exp(lnw)
    for c, d in zip(centres, depths, strict=True):
        lo, hi = np.searchsorted(w, [c - 6 * sigma, c + 6 * sigma])
        tau[lo:hi] += d * np.exp(-0.5 * ((w[lo:hi] - c) / sigma) ** 2)


def write_model(path: str | Path, seed: int = 2026) -> Path:
    """Write the synthetic model to `path` (an .npz in the format `rvfitter.calibration.telluric` reads)"""
    rng = np.random.default_rng(seed)
    dl = 1. / R_S
    n = int(np.floor(np.log(WMAX / WMIN) * R_S))
    lnw = np.log(WMIN) + (np.arange(n) + 0.5) * dl
    tau_dry, tau_h2o = np.zeros(n), np.zeros(n)
    for a, b, step, peak, sigma in O2_BANDS:
        c = np.arange(a + 0.3, b, step) + rng.uniform(-0.3, 0.3, len(np.arange(a + 0.3, b, step)))
        _add(tau_dry, lnw, c, peak * np.exp(-(c - a) / (b - a)) * rng.uniform(0.5, 1., len(c)), sigma)
    for a, b, density, peak, sigma in H2O_BANDS:
        c = np.sort(rng.uniform(a, b, int(density * (b - a))))
        _add(tau_h2o, lnw, c, peak * rng.uniform(0.1, 1., len(c)), sigma)
    meta = dict(description='synthetic test model (not HITRAN)', source='tests/telluric_synth.py',
                citation='HITRAN: none (synthetic)', licence='test fixture', terms='test fixture', molecules={})
    path = Path(path)
    np.savez_compressed(path, lnlam0=np.log(WMIN), dlnlam=dl, n=n, tau_dry=tau_dry.astype(np.float16),
                        tau_h2o=tau_h2o.astype(np.float16), meta=json.dumps(meta))
    return path
