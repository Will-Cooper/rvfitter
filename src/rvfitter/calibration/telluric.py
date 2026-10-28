"""
Telluric transmission model and masks.

The packaged model (`data/telluric_hitran.npz`, built by `tools/build_telluric_model.py`) holds zenith optical
depths on a grid uniform in ln λ (vacuum, 6000–25 000 Å, R_s = 300 000), computed from HITRAN line-by-line data
(H2O, O2, CO2, CH4; HITRAN2024, Gordon et al. 2026) through a layered US Standard 1976 atmosphere above a 2.4 km
site. The dry gases and water vapour are stored separately, so

    T(λ; X, w) = exp(−X · (τ_dry(λ) + w · τ_H2O(λ)))

for airmass X and precipitable water vapour w [mm]. The model is valid for spectral resolutions up to about
100 000; at lower resolution it is convolved with a Gaussian line-spread function and averaged over the data's
pixels. Only the transmission, not the HITRAN line parameters, is shipped (see the builder for the licence and
citation). Continuum absorption (MT_CKD), Rayleigh scattering, aerosols, O3 and N2O are omitted; they are smooth at
these wavelengths and absorbed by any continuum fit.

The wavelengths are vacuum. At R = 20 000 the blue edges of the O2 A and B bands lie 1.0–1.3 Å redward of the
Fraunhofer A and B lines (759.370 and 686.719 nm in air) converted to vacuum, and 2.8–3.4 Å redward of the air
values (tested). An independent ESO SkyCalc/LBLRTM transmission (vacuum; airmass 3, R = 60 000) agrees within
0.2 km/s in the O2 B, O2 A and H2O 8200 and 9300 bands (cross-correlation; docs/dev_notes_methods.md).
"""
import functools
import json
import logging
from dataclasses import dataclass

import numpy as np

from .. import broadening
from ..constants import C_KMS

__all__ = ['TelluricModel', 'model', 'transmission', 'mask', 'DEFAULT_PWV']

log = logging.getLogger(__name__)

DEFAULT_PWV = 2.5  # [mm]; a typical night at a high site


@dataclass(frozen=True)
class TelluricModel:
    """Zenith optical depths on a grid uniform in ln λ (see the module docstring)"""
    lnlam0: float
    dlnlam: float
    tau_dry: np.ndarray
    tau_h2o: np.ndarray
    meta: dict

    @property
    def wave(self) -> np.ndarray:
        """Vacuum wavelengths of the grid [Å]"""
        return np.exp(self.lnlam0 + (np.arange(len(self.tau_dry)) + 0.5) * self.dlnlam)

    @property
    def wmin(self) -> float:
        return float(np.exp(self.lnlam0))

    @property
    def wmax(self) -> float:
        return float(np.exp(self.lnlam0 + len(self.tau_dry) * self.dlnlam))

    def slice(self, wmin: float, wmax: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(wave, tau_dry, tau_h2o) of the grid points covering [wmin, wmax]"""
        i0 = max(0, int(np.floor((np.log(wmin) - self.lnlam0) / self.dlnlam)) - 1)
        i1 = min(len(self.tau_dry), int(np.ceil((np.log(wmax) - self.lnlam0) / self.dlnlam)) + 2)
        if i1 <= i0:
            return np.empty(0), np.empty(0), np.empty(0)
        w = np.exp(self.lnlam0 + (np.arange(i0, i1) + 0.5) * self.dlnlam)
        return w, self.tau_dry[i0:i1], self.tau_h2o[i0:i1]


@functools.lru_cache(maxsize=1)
def model() -> TelluricModel:
    """The packaged telluric model (loaded once)"""
    from importlib import resources
    with resources.files('rvfitter').joinpath('data', 'telluric_hitran.npz').open('rb') as fh:
        z = np.load(fh)
        m = TelluricModel(float(z['lnlam0']), float(z['dlnlam']), z['tau_dry'].astype(float),
                          z['tau_h2o'].astype(float), json.loads(str(z['meta'])))
    log.debug('telluric model: %d points, %.0f-%.0f Å (%s)', len(m.tau_dry), m.wmin, m.wmax, m.meta.get('source'))
    return m


def lsf_sigma_pix(resolution: float, dlnlam: float) -> float:
    """The Gaussian LSF σ in grid pixels of step `dlnlam` for resolving power `resolution`"""
    return float(broadening.FWHM_TO_SIGMA / (resolution * dlnlam))


def transmission(wave: np.ndarray, airmass: float = 1., pwv: float = DEFAULT_PWV, *,
                 resolution: float | None = None, dv: float = 0.) -> np.ndarray:
    """
    The telluric transmission at `wave`

    Parameters
    ----------
    wave
        Vacuum wavelengths [Å], strictly increasing
    airmass
        Airmass X (scales every optical depth)
    pwv
        Precipitable water vapour [mm]
    resolution
        Resolving power of a Gaussian LSF; the transmission is then convolved and averaged over each pixel of
        `wave`. None: the model sampled at `wave` (pixel-averaged where the pixels are coarser than the grid)
    dv
        Velocity shift of the telluric lines [km/s]: the model is evaluated at λ / D(dv), so positive dv moves
        the features to the red (as a wavelength scale that is too red would)

    Returns
    -------
    t
        Transmission (1 outside the model's 6000–25 000 Å coverage)
    """
    wave = np.atleast_1d(np.asarray(wave, dtype=float))
    out = np.ones(len(wave))
    if not len(wave):
        return out
    m = model()
    f = float(broadening.doppler_factor(dv))
    pad = 1. + 10. / (resolution if resolution else 1e5) + 0.01 * abs(dv) / C_KMS + 1e-3
    lo, hi = wave[0] / f / pad, wave[-1] / f * pad
    if hi < m.wmin or lo > m.wmax:
        return out
    w, td, th = m.slice(lo, hi)
    t = np.exp(-airmass * (td + pwv * th))
    if resolution is not None and np.isfinite(resolution) and resolution > 0:
        t = broadening.convolve(t, broadening.gaussian_kernel(lsf_sigma_pix(resolution, m.dlnlam)))
    inside = (wave >= w[0] * f) & (wave <= w[-1] * f)
    if inside.sum() > 1:
        out[inside] = broadening.rebin(w * f, t, wave[inside])
    elif inside.any():
        out[inside] = np.interp(wave[inside], w * f, t)
    if not inside.all():
        log.debug('telluric transmission: %d pixel(s) outside the model coverage set to 1', int((~inside).sum()))
    return out


def mask(wave: np.ndarray, threshold: float = 0.05, *, airmass: float = 1., pwv: float = DEFAULT_PWV,
         resolution: float | None = None) -> np.ndarray:
    """
    Pixels whose telluric absorption 1 − T exceeds `threshold`

    Parameters
    ----------
    wave
        Vacuum wavelengths [Å]
    threshold
        Absorption depth above which a pixel is masked
    airmass, pwv, resolution
        As for `transmission`

    Returns
    -------
    masked
        Boolean array, True where telluric absorption is deeper than `threshold`
    """
    return 1. - transmission(wave, airmass, pwv, resolution=resolution) > threshold
