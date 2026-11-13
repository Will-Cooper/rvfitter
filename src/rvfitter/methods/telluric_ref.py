"""
Telluric-referenced RVs: the star's velocity relative to the telluric lines in the same stretch of spectrum.

Within one telluric band the stellar and telluric features share the wavelength-solution error and the
slit-centring offset. Each segment is fitted with P(λ) · S(λ / D(v_star)) · [T(λ / D(v_tell)) ⊗ LSF]
(`calibration.telluric_zeropoint`; S the stellar template nearest the priors, T the HITRAN transmission), and
RV = (v_star − v_tell) ⊕ v_frame, with v_frame the velocity correction already applied to the wavelengths, so the
RV is in the spectrum's frame. Its error (covariance included) is scaled by √max(1, χ²/ν); segments are combined
by random effects. The input must not be telluric-corrected (set `calibration.telluric_source` otherwise).
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

__all__ = ['measure', 'DEFAULTS', 'prior_template', 'neighbour_templates']

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


def prior_template(cfg: Any, resolution: float, wave_range: tuple[float, float], opts: dict | None = None
                   ) -> C.Template:
    """
    The template nearest the configured priors (teff_prior, logg_prior; 5.0 dex if unset) at `resolution`,
    covering wave_range ± 1500 km/s; ValueError if none is available
    """
    cands = C.candidates(cfg, opts or DEFAULTS)
    teff, logg = cfg.templates.teff_prior, cfg.templates.logg_prior or 5.

    def distance(c):
        p = c.param_dict
        return (abs(p.get('teff', 0.) - teff) if teff is not None and 'teff' in p else 0.,
                abs(p.get('logg', 5.) - logg) if 'logg' in p else 0.)

    best = min(cands, key=distance)
    return C.build_template(best, resolution, wave_range)


def neighbour_templates(cfg: Any, resolution: float, wave_range: tuple[float, float], nominal: C.Template,
                        opts: dict | None = None) -> list:
    """
    The grid points one step below and above the nominal template's Teff (other parameters equal), at
    `resolution`: the alternatives whose spread is a node's template systematic. Empty for a given template.
    """
    cands = C.candidates(cfg, opts or DEFAULTS)
    nom = next((c for c in cands if c.label == nominal.label), None)
    if nom is None or not nom.param_dict or 'teff' not in nom.param_dict:
        return []
    p0 = nom.param_dict
    same = [c for c in cands if c.param_dict and all(c.param_dict.get(k) == v for k, v in p0.items() if k != 'teff')]
    lower = [c for c in same if c.param_dict['teff'] < p0['teff']]
    upper = [c for c in same if c.param_dict['teff'] > p0['teff']]
    out = []
    for group, pick in ((lower, max), (upper, min)):
        if group:
            c = pick(group, key=lambda c: c.param_dict['teff'])
            try:
                out.append(C.build_template(c, resolution, wave_range))
            except (ValueError, OSError) as exc:  # a missing neighbour narrows the spread estimate
                log.warning('neighbour template %s unavailable (%s): the template spread uses fewer templates',
                            c.label, exc)
    return out


def _frame_kms(meta: Any) -> float:
    """The velocity [km/s] of the wavelength corrections already applied (header and pipeline)"""
    f = telluric_zeropoint.frame_factor(meta)
    return float(C_KMS * (f * f - 1.) / (f * f + 1.))


def measure(spectrum: Spectrum, regions: Sequence[Region], cfg: Any, *, executor: Executor | None = None,
            rng: np.random.Generator | None = None) -> MethodResult:
    """
    Telluric-referenced RVs, one per telluric segment (the 'telluric' regions, else the catalogue bands in the
    coverage), combined by random effects; options in cfg.methods.options['telluric_ref'] (`DEFAULTS`). The fits
    are deterministic: `executor` and `rng` are accepted for the common interface only. Diagnostics hold v_star,
    v_tell (the local telluric zero point), their errors and correlation, χ²/ν and the τ scalings.
    """
    opts = C.options(cfg, METHOD, DEFAULTS)
    from ..calibration import telluric
    if not telluric.available():
        log.warning('telluric_ref skipped: %s', telluric.unavailable_message())
        diag = {'reason': 'telluric model unavailable'}
        meas = Measurement(METHOD, 'telluric model unavailable', np.nan, np.nan, C.FLAG_FIT_FAILED, diag)
        return MethodResult(METHOD, np.nan, np.nan, 0, [meas])
    if cfg.input.telluric_corrected:
        log.warning('telluric_ref: the input is telluric-corrected; the telluric lines it relies on may be gone')
    resolution = C.resolution_of(spectrum, cfg, opts)
    tell = [r for r in regions if r.kind == 'telluric']
    if not tell:
        log.info('telluric_ref: no telluric regions selected; using the catalogue telluric bands in the coverage')
    template = prior_template(cfg, resolution, (float(spectrum.wave[0]), float(spectrum.wave[-1])), opts)
    zp = telluric_zeropoint.measure(spectrum, tell or None, resolution=resolution, template=template,
                                    seg_width=opts['seg_width'], degree=opts['degree'], min_depth=opts['min_depth'],
                                    max_lag_kms=opts['max_lag_kms'],
                                    star_rv_range=(opts['star_rv_min'], opts['star_rv_max']),
                                    teff=cfg.templates.teff_prior)
    if zp is None:
        raise ValueError('no usable telluric segment')
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
        flag |= C.FLAG_GRID_HOLE if template.params.get('grid_hole') else 0
        diag.update(rv_stat_err=s['star_minus_tell_err_stat'])
        measurements.append(Measurement(METHOD, name, rv, float(s['star_minus_tell_err']), flag, diag))
        log.info('telluric_ref %s: %.1f ± %.1f km/s (star %+.1f, telluric %+.1f km/s, observatory frame; χ²/ν %.1f)',
                 name, rv, s['star_minus_tell_err'], s['v_star'], s['dv'], s['chi2_red'])
    return C.finish(METHOD, measurements, outlier_k=opts['outlier_k'] or None)
