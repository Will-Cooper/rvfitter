"""
Telluric-referenced RVs: the star's velocity relative to the telluric lines in the same stretch of spectrum.

Telluric absorption is imprinted on the starlight, so within one telluric band the stellar and telluric features
share the same wavelength-solution error and the same offset from slit centring. Each telluric segment is fitted
with P(λ) · S(λ / D(v_star)) · [T(λ / D(v_tell)) ⊗ LSF] (`calibration.telluric_zeropoint`, with a stellar
template S at the data's resolution and the HITRAN transmission T), and

    RV = (v_star − v_tell) ⊕ v_frame

is independent of the wavelength solution and of the star's position in the slit; v_frame is the velocity
correction already applied to the wavelengths (header, e.g. PypeIt VEL_CORR, and the pipeline's barycentric step),
so the RV is in the spectrum's frame, like the other methods'. Its error is that of v_star − v_tell from the χ²
curvature surface (their covariance included), scaled by the segment's Birge factor √max(1, χ²/ν). The segments
are combined by random effects (`_common.finish`).

The input must not be telluric-corrected; set `calibration.telluric_source` for a corrected input (e.g. a PypeIt
spec1d sharing its wavelength solution). Regions are the `telluric` regions from `regions.select` (telluric bands
at R ≥ their r_min); without any, the catalogue's telluric bands within the coverage are used and this is logged.
The template is the grid point nearest the configured priors (templates.teff_prior, logg_prior; 5.0 dex if unset),
or the method option `template`; it is fixed a priori, not chosen by χ², so the method stays cheap.
"""
import logging
from collections.abc import Sequence
from concurrent.futures import Executor
from typing import Any

import numpy as np

from ..calibration import telluric_zeropoint
from ..constants import C_KMS
from ..types import Measurement, MethodResult, Region, Spectrum
from . import _common as C

__all__ = ['measure', 'DEFAULTS', 'prior_template']

log = logging.getLogger(__name__)

METHOD = 'telluric_ref'
DEFAULTS: dict[str, Any] = dict(
    resolution=None,  # override the data's R
    template=None, templates=None, teff_window=300.,  # as for the other template methods
    seg_width=250.,  # maximum telluric segment width [Å]
    degree=2,  # Legendre continuum degree per segment
    min_depth=0.05,  # minimum model telluric depth of a fitted species / usable segment
    max_lag_kms=150.,  # telluric zero-point search range [km/s]
    star_rv_min=-400., star_rv_max=400.,  # observatory-frame stellar velocity search range [km/s]
    outlier_k=5.,  # gross-outlier rejection among segments (0: off)
)


def prior_template(cfg: Any, resolution: float, wave_range: tuple[float, float], opts: dict | None = None):
    """
    The template nearest the configured priors (teff_prior, logg_prior; 5.0 dex if unset), at `resolution`

    Returns
    -------
    template
        A `_common.Template` covering wave_range (± 1500 km/s), or None if no template is configured
    """
    opts = dict(opts or {})
    try:
        cands = C.candidates(cfg, opts)
    except ValueError as exc:
        log.info('telluric_ref: no stellar template available (%s)', exc)
        return None
    tcfg = C.section(cfg, 'templates')
    teff, logg = tcfg.get('teff_prior'), tcfg.get('logg_prior') or 5.

    def distance(c):
        p = c.param_dict
        return (abs(p.get('teff', 0.) - teff) if teff is not None and 'teff' in p else 0.,
                abs(p.get('logg', 5.) - logg) if 'logg' in p else 0.)

    best = min(cands, key=distance)
    return C.build_template(best, resolution, wave_range)


def _frame_kms(meta: Any) -> float:
    """The velocity [km/s] of the wavelength corrections already applied (header and pipeline)"""
    f = telluric_zeropoint.frame_factor(meta)
    return float(C_KMS * (f * f - 1.) / (f * f + 1.))


def measure(spectrum: Spectrum, regions: Sequence[Region], cfg: Any = None, *, executor: Executor | None = None,
            rng: np.random.Generator | None = None) -> MethodResult:
    """
    Telluric-referenced RVs, one per telluric segment, combined with a random-effects model

    Parameters
    ----------
    spectrum
        Observed spectrum (vacuum), not telluric-corrected
    regions
        Telluric regions (kind 'telluric'); others are ignored
    cfg
        Config or mapping; options under cfg.methods.options['telluric_ref'] (see `DEFAULTS`)
    executor, rng
        Unused (the fits are deterministic); accepted for the common method interface

    Returns
    -------
    result
        The combined RV; each measurement's diagnostics hold v_star, v_tell (the local telluric zero point), their
        errors and correlation, χ²/ν and the τ scalings
    """
    opts = C.options(cfg, METHOD, DEFAULTS)
    if C.section(cfg, 'input').get('telluric_corrected'):
        log.warning('telluric_ref: the input is telluric-corrected; the telluric lines it relies on may be gone')
    resolution = C.resolution_of(spectrum, cfg, opts)
    tell = [r for r in regions if r.kind == 'telluric']
    if not tell:
        log.info('telluric_ref: no telluric regions selected; using the catalogue telluric bands in the coverage')
    template = prior_template(cfg, resolution, (float(spectrum.wave[0]), float(spectrum.wave[-1])), opts)
    if template is None:
        log.warning('telluric_ref: needs a stellar template (templates config or the option "template")')
        return MethodResult(METHOD, np.nan, np.nan, 0, [])
    zp = telluric_zeropoint.measure(spectrum, tell or None, resolution=resolution, template=template,
                                    seg_width=opts['seg_width'], degree=opts['degree'], min_depth=opts['min_depth'],
                                    max_lag_kms=opts['max_lag_kms'],
                                    star_rv_range=(opts['star_rv_min'], opts['star_rv_max']))
    if zp is None:
        return MethodResult(METHOD, np.nan, np.nan, 0, [])
    v_frame = _frame_kms(spectrum.meta)
    measurements = []
    for s in zp.segments:
        name = f"{s['band']} {s['wmin']:.0f}-{s['wmax']:.0f}"
        diag = {k: s.get(k) for k in ('v_star', 'v_star_err', 'dv', 'err', 'err_stat', 'corr_star_tell', 'chi2_red',
                                      'a_dry', 'a_h2o', 'depth_dry', 'depth_h2o', 'n_pix', 'n_clipped', 'reason')}
        diag.update(wave_centre=s['centre'], v_frame=v_frame, template=template.label, resolution=resolution,
                    zp_tell_kms=s.get('dv'), zp_tell_err=s.get('err'))
        if not s.get('used') or 'star_minus_tell' not in s:
            flag = C.FLAG_EDGE if 'edge' in s.get('reason', '') else C.FLAG_FIT_FAILED
            measurements.append(Measurement(METHOD, name, np.nan, np.nan, flag, diag))
            continue
        rv = float(C_KMS * np.tanh(np.arctanh(s['star_minus_tell'] / C_KMS)
                                              + np.arctanh(v_frame / C_KMS)))
        flag = C.FLAG_POOR_FIT if s['chi2_red'] > 3. else 0
        flag |= C.FLAG_OUTSIDE_GRID if template.params.get('outside_grid') else 0
        diag.update(rv_stat_err=s['star_minus_tell_err_stat'])
        measurements.append(Measurement(METHOD, name, rv, float(s['star_minus_tell_err']), flag, diag))
        log.info('telluric_ref %s: %.1f ± %.1f km/s (star %+.1f, telluric %+.1f km/s, observatory frame; χ²/ν %.1f)',
                 name, rv, s['star_minus_tell_err'], s['v_star'], s['dv'], s['chi2_red'])
    return C.finish(METHOD, measurements, outlier_k=opts['outlier_k'] or None)
