"""
Slit-centring prior on a star's absolute wavelength zero point.

A point source in a slit of width W [arcsec] with seeing FWHM F is placed somewhere across the slit, while the
uniformly illuminated sky (and the arc lamp) fills it. The light that passes the slit then has its centroid
displaced along the dispersion, so the star's wavelength zero point differs from the sky's by an unknown constant.
We take the profile centre x as uniform within ±δ, δ = max((W − F)/2, `ACQUISITION_ARCSEC`), i.e. anywhere
the whole core fits, but never better than the acquisition accuracy, and the profile as a Gaussian clipped by
the slit edges. The centroid of the transmitted light is

    c(x) = x + σ [φ(a) − φ(b)] / [Φ(b) − Φ(a)],   a = (−W/2 − x)/σ,  b = (W/2 − x)/σ,  σ = F/√(8 ln 2),

and σ_x is the RMS of c(x) over the prior. For F ≪ W, c(x) = x and σ_x = (W − F)/√12; when the star
overfills the slit (F ≥ W) the clipping damps the shift, so σ_x stays finite and small rather than undefined.

which becomes a velocity through the plate scale along the dispersion (arcsec per binned pixel) and the dispersion
D(λ) [Å per pixel] of the wavelength solution:

    σ_v(λ) = c · (σ_x / s_spec) · D(λ) / λ.

The pixels are assumed square on the sky and the anamorphic magnification of the grating is ignored.

Inputs:
- W: the header (``SLITW``, else 'N.NNarcsec' in ``MASKNAME``/``DECKER``…; `rvfitter.io`);
- F: the object's spatial FWHM along the slit (PypeIt ``FWHMFIT``) times the spatial plate scale;
- the plate scales: ``PIXSCALE``-like header keywords, else the unbinned scale of the instrument in
  ``data/instruments.csv`` times the header binning;
- D(λ): the spacing of the native wavelength solution of the spectrum that provided the header.

When an input is missing the prior cannot be computed and NaN is returned (the caller warns that the absolute
zero point is unconstrained).
"""
import functools
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import ndtr

from ..constants import C_KMS

__all__ = ['SlitInputs', 'ACQUISITION_ARCSEC', 'centroid_rms', 'instrument_pixscale', 'inputs', 'prior_kms']

log = logging.getLogger(__name__)

ACQUISITION_ARCSEC = 0.1  # the smallest half-range of the star's position across the slit (target acquisition)


@functools.lru_cache(maxsize=1)
def _table() -> dict[str, tuple[float, str]]:
    import csv
    from importlib import resources
    with resources.files('rvfitter').joinpath('data', 'instruments.csv').open() as fh:
        rows = csv.DictReader(line for line in fh if not line.startswith('#'))
        return {r['key'].strip().lower(): (float(r['pixscale_unbinned_arcsec']), r['source'].strip()) for r in rows}


def instrument_pixscale(pypeit_spec: str | None, instrument: str | None) -> tuple[float | None, str]:
    """The unbinned pixel scale [arcsec/pix] of an instrument from ``data/instruments.csv`` (PYP_SPEC, then INSTRUME)"""
    table = _table()
    for key in (pypeit_spec, instrument):
        if key and str(key).strip().lower() in table:
            scale, source = table[str(key).strip().lower()]
            return scale, f'data/instruments.csv[{str(key).strip()}]: {source}'
    return None, ''


@dataclass
class SlitInputs:
    """The inputs of the slit-centring prior, with where each came from (None when unknown)"""
    width_arcsec: float | None = None
    width_source: str = ''
    fwhm_arcsec: float | None = None
    fwhm_source: str = ''
    scale_spat: float | None = None  # arcsec per binned pixel along the slit
    scale_spec: float | None = None  # arcsec per binned pixel along the dispersion
    scale_source: str = ''
    disp_wave: np.ndarray | None = field(default=None, repr=False)  # [Å] where the dispersion is tabulated
    disp: np.ndarray | None = field(default=None, repr=False)  # [Å per binned pixel]
    disp_source: str = ''

    @property
    def missing(self) -> list[str]:
        out = []
        if self.width_arcsec is None:
            out.append('slit width (SLITW, or MASKNAME/DECKER with "arcsec")')
        if self.fwhm_arcsec is None:
            out.append('seeing FWHM (PypeIt FWHMFIT and a plate scale)')
        if self.scale_spec is None:
            out.append('plate scale (PIXSCALE, or the instrument table and BINNING)')
        if self.disp is None:
            out.append('dispersion')
        return out

    @property
    def sigma_arcsec(self) -> float:
        """RMS centroid offset of the transmitted light across the slit [arcsec]; NaN if W or F is unknown"""
        if self.width_arcsec is None or self.fwhm_arcsec is None or not self.width_arcsec > 0:
            return float('nan')
        return centroid_rms(self.width_arcsec, self.fwhm_arcsec)

    def as_dict(self) -> dict[str, Any]:
        return dict(slit_width_arcsec=self.width_arcsec, slit_width_source=self.width_source,
                    seeing_fwhm_arcsec=self.fwhm_arcsec, seeing_fwhm_source=self.fwhm_source,
                    scale_spat_arcsec=self.scale_spat, scale_spec_arcsec=self.scale_spec,
                    scale_source=self.scale_source, disp_source=self.disp_source,
                    disp_median=float(np.median(self.disp)) if self.disp is not None else None,
                    sigma_arcsec=self.sigma_arcsec, missing=self.missing)


def _scales(meta: dict) -> tuple[float | None, float | None, str]:
    """(arcsec per binned spatial pixel, arcsec per binned spectral pixel, source) from one spectrum's meta"""
    hdr = meta.get('pixscale_arcsec')
    if hdr is not None and np.isfinite(hdr) and hdr > 0:
        return float(hdr), float(hdr), 'header PIXSCALE/PLATESCL/SECPIX (binned)'
    unb, source = instrument_pixscale(meta.get('pypeit_spec'), meta.get('instrument'))
    if unb is None:
        return None, None, ''
    bspat, bspec = meta.get('binning_spat'), meta.get('binning_spec')
    if bspat is None or bspec is None:
        return None, None, f'{source}; binning unknown'
    return unb * bspat, unb * bspec, f'{source} × BINNING {bspec},{bspat} (spectral,spatial)'


def inputs(spectra: Sequence[Any]) -> SlitInputs:
    """
    Collect the prior's inputs from candidate spectra, in order of preference (e.g. the target's own spec1d table,
    then the input spectrum); each input is taken from the first spectrum that has it, and the dispersion from the
    first one that gave a slit width (its native wavelength solution), else the first spectrum
    """
    out = SlitInputs()
    first_with_width = None
    for s in spectra:
        if s is None:
            continue
        meta = s.meta if hasattr(s.meta, 'get') else {}
        src = f"{getattr(s, 'source', '')}" + (f" HDU {meta['hdu']}" if meta.get('hdu') is not None else '')
        if out.width_arcsec is None and meta.get('slit_width_arcsec'):
            out.width_arcsec = float(meta['slit_width_arcsec'])
            out.width_source = f"{meta.get('slit_width_key')} of {src}"
            first_with_width = s
        if out.scale_spec is None:
            sp, ss, why = _scales(meta)
            if ss is not None:
                out.scale_spat, out.scale_spec, out.scale_source = sp, ss, why
        if out.fwhm_arcsec is None and meta.get('trace_fwhm_pix'):
            sp, _, _ = _scales(meta)
            if sp is not None:
                out.fwhm_arcsec = float(meta['trace_fwhm_pix']) * sp
                out.fwhm_source = f"trace FWHM {float(meta['trace_fwhm_pix']):.2f} pix × {sp:.3f} arcsec/pix of {src}"
    disp_spec = first_with_width or next((s for s in spectra if s is not None), None)
    if disp_spec is not None and len(disp_spec.wave) > 2:
        w = np.asarray(disp_spec.wave, dtype=float)
        out.disp_wave, out.disp = w, np.gradient(w)
        out.disp_source = f'wavelength solution of {getattr(disp_spec, "source", "")}'
    return out


def centroid_rms(width: float, fwhm: float, acquisition: float = ACQUISITION_ARCSEC, n: int = 401) -> float:
    """
    RMS [arcsec] of the transmitted-light centroid c(x) of a Gaussian profile (FWHM `fwhm`) clipped by a slit of
    width `width`, for a profile centre x uniform within ±max((width − fwhm)/2, acquisition)
    """
    half = max((width - fwhm) / 2., acquisition)
    x = half * ((2. * np.arange(n) + 1.) / n - 1.)  # midpoints of n equal bins: an unbiased uniform average
    sig = fwhm / np.sqrt(8. * np.log(2.))
    if sig <= 0:
        return float(np.sqrt(np.mean(np.clip(x, -width / 2., width / 2.) ** 2)))
    a, b = (-width / 2. - x) / sig, (width / 2. - x) / sig
    pdf = np.exp(-0.5 * a * a) - np.exp(-0.5 * b * b)
    c = x + sig * pdf / np.sqrt(2. * np.pi) / np.maximum(ndtr(b) - ndtr(a), 1e-300)
    return float(np.sqrt(np.mean(c * c)))


def prior_kms(inp: SlitInputs, wave: np.ndarray | Sequence[float] | float) -> float:
    """
    The slit-centring prior σ_v [km/s]: the RMS over `wave` [Å] of c (σ_x / s_spec) D(λ) / λ; NaN if unavailable
    """
    sig = inp.sigma_arcsec
    if not np.isfinite(sig) or inp.scale_spec is None or inp.disp is None:
        return float('nan')
    lam = np.atleast_1d(np.asarray(wave, dtype=float))
    lam = lam[np.isfinite(lam)]
    if not len(lam):
        return float('nan')
    d = np.interp(lam, inp.disp_wave, inp.disp)
    v = C_KMS * sig / inp.scale_spec * d / lam
    return float(np.sqrt(np.mean(v * v)))
