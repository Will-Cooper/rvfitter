"""
Wavelength zero-point policy: measure the sky and telluric zero points of a spectrum, choose what to apply
(``calibration.wavelength_correction``) and give the star's absolute zero-point term (`star_sky_term`).

A measured offset's value never becomes an error: it is either applied (its uncertainty is then a systematic) or
unknown (the slit-centring prior covers it).
"""
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
from scipy.stats import chi2 as chi2_dist

from .. import io
from ..constants import C_KMS
from ..features import features
from ..numerics import rms
from ..types import Spectrum
from . import dar, sky_zeropoint, slit, telluric, telluric_zeropoint

__all__ = ['correct', 'star_sky_term', 'combine_epochs', 'arc_kms', 'valid_range', 'matching_object', 'ZP_KEYS',
           'ZP_FLAGS', 'zp_flag_names']

log = logging.getLogger(__name__)

ZP_KEYS = ('zp_sky_kms', 'zp_sky_err', 'zp_sky_slope', 'zp_sky_model_spread', 'zp_tell_kms', 'zp_tell_err',
           'zp_star_sky', 'zp_star_sky_err', 'zp_star_sky_n', 'zp_dar_s', 'zp_dar_s_err')

#: Bits of the ``zp_flags`` results column: what the wavelength zero-point step did, and why
ZP_FLAGS: dict[str, int] = {
    'sky_applied': 1,  # the sky curve Δv_sky(λ) was applied
    'telluric_applied': 2,  # the star's telluric curve was applied
    'star_sky_applied': 4,  # the star − sky constant was applied with the sky curve ('sky+star')
    'sky_not_needed': 8,  # sky-if-needed: the offset is not significant or below the practical floor
    'sky_unsafe': 16,  # sky-if-needed: too few sky segments, or the regions rely on unmeasured parts of the curve
    'telluric_unavailable': 32,  # the telluric model is not built: telluric terms skipped
    'telluric_refused': 64,  # the telluric curve does not cover the regions: not applied
    'star_sky_unmeasured': 128,  # fewer than 2 telluric nodes with the sky measured in the same window: NaN
    'no_sky': 256,  # no sky spectrum: the sky zero point is not measured
    'star_sky_inconsistent': 512,  # the star − sky nodes disagree (χ² p ≤ 0.01): error Birge-inflated
    'zp_unconstrained': 1024,  # neither the star − sky constant nor the slit prior is available: not in the errors
    'slit_prior': 2048,  # sys_star_sky is the slit-centring prior (or a user value), not a measurement
    'dar_pa_unknown': 4096,  # no slit PA (raw header): the DAR factor s is fitted (star - sky) or marginalised (prior)
    'dar_unknown': 8192,  # no zenith distance or plate scale: the DAR across the slit is not modelled
    'sky_window_fallback': 16384,  # a region's own sky window failed: the sky curve is used there instead
}

# sky-if-needed criteria (calibration data only; fixed a priori)
SKY_MIN_SEGMENTS = 3  # well-measured (used) sky segments
SKY_MAX_UNMEASURED = 0.5  # each region: at most this fraction outside the measured sky segments
SKY_SIGNIFICANCE = 3.  # |weighted mean Δv| / σ(curve ⊕ arc RMS)
SKY_VARIATION_P = 1e-3  # χ² p-value of the segments against a constant
SKY_FLOOR_RESEL = 0.05  # the correction must exceed this fraction of a resolution element (c/R)
TELLURIC_MAX_NODE_GAP = 250.  # [Å] a region farther than this from every telluric node is not covered


def zp_flag_names(flags: int) -> list[str]:
    """Names of the bits set in a ``zp_flags`` value"""
    return [k for k, b in ZP_FLAGS.items() if int(flags) & b]


def arc_kms(spec: Spectrum) -> float:
    """The arc-solution RMS (``meta['wave_rms']``, Å) as a velocity [km/s] at the median wavelength; NaN if unknown"""
    r = spec.meta.get('wave_rms')
    return float(C_KMS * r / np.median(spec.wave)) if r is not None and np.isfinite(r) else float('nan')


def _region_list(sel: dict | Sequence | None) -> list:
    """The distinct non-telluric regions of a `regions.select` result (dict method → regions, or a list)"""
    if not sel:
        return []
    regs = [r for lst in sel.values() for r in lst] if isinstance(sel, dict) else list(sel)
    out, seen = [], set()
    for r in regs:
        key = (round(r.wmin, 2), round(r.wmax, 2))
        if r.kind != 'telluric' and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _sky_needed(zp: Any, spec: Spectrum, cfg: Any, regions: Sequence, name: str) -> tuple[bool, dict]:
    """
    The sky-if-needed decision: apply the sky curve only if (a) it is well measured where the regions are, (b) the
    offset or the distortion is significant and (c) the correction exceeds a practical floor. Calibration data and
    region positions only (never the stellar line scatter).
    """
    t: dict[str, Any] = {'n_segments': int(zp.n_used)}
    # (a) coverage: the regions in the observatory frame against the measured sky segments
    ff = telluric_zeropoint.frame_factor(spec.meta)
    fracs = [zp.measured_fraction(r.wmin / ff, r.wmax / ff) for r in regions]
    t['region_measured'] = {r.name: round(f, 3) for r, f in zip(regions, fracs, strict=True)}
    t['min_measured_fraction'] = float(min(fracs)) if fracs else float('nan')
    poor = [r.name for r, f in zip(regions, fracs, strict=True) if f <= 1. - SKY_MAX_UNMEASURED]
    # regions on unmeasured parts of the curve are dropped if the correction is applied (see `uncovered`), so they
    # veto it only when they are the majority
    t['uncovered'] = sorted(set(poor))
    ok_a = zp.n_used >= SKY_MIN_SEGMENTS and bool(regions) and 2 * len(poor) < len(regions)
    # (b) significance: weighted mean of the segment values, σ from the curve (with its model spread) and the arc
    v, e = np.asarray(zp.node_dv, float), np.asarray(zp.node_err, float)
    w = 1. / e ** 2
    mean = float(np.sum(w * v) / np.sum(w))
    spread = zp.model_spread if np.isfinite(zp.model_spread) else 0.
    arc = arc_kms(spec)
    sigma = float(np.sqrt(1. / np.sum(w) + spread ** 2 + (arc ** 2 if np.isfinite(arc) else 0.)))
    chi2 = float(np.sum(w * (v - mean) ** 2))
    p = float(chi2_dist.sf(chi2, len(v) - 1)) if len(v) > 1 else 1.
    t.update(mean_kms=mean, sigma_kms=sigma, arc_kms=arc, n_sigma=abs(mean) / sigma, chi2=chi2, dof=len(v) - 1,
             p_variation=p)
    ok_b = t['n_sigma'] > SKY_SIGNIFICANCE or p < SKY_VARIATION_P
    # (c) floor: the largest correction at the regions against 0.05 resolution elements
    res = spec.resolution or cfg.input.resolution
    if not res and np.isfinite(zp.fwhm) and zp.fwhm > 0:
        res = float(np.median(spec.wave) / zp.fwhm)  # from the sky-line width
    t['resolution'] = float(res) if res else float('nan')
    t['floor_kms'] = SKY_FLOOR_RESEL * C_KMS / res if res else 0.
    centres = np.array([0.5 * (r.wmin + r.wmax) / ff for r in regions])
    t['max_correction_kms'] = float(np.max(np.abs(zp.dv(centres)))) if len(centres) else abs(mean)
    ok_c = t['max_correction_kms'] > t['floor_kms']
    t.update(coverage_ok=bool(ok_a), significant=bool(ok_b), above_floor=bool(ok_c))
    apply = ok_a and ok_b and ok_c
    if apply:
        verdict = 'applying'
        if poor:
            verdict += (f'; dropping region(s) {", ".join(t["uncovered"])}, which rely on unmeasured parts of the '
                        f'sky curve for more than {SKY_MAX_UNMEASURED:.0%} of their width')
    elif not ok_a:
        if zp.n_used < SKY_MIN_SEGMENTS:
            why = f'only {zp.n_used} measured sky segments (< {SKY_MIN_SEGMENTS})'
        elif not regions:
            why = 'no regions to check the coverage against'
        else:
            why = (f'most regions ({", ".join(t["uncovered"])}) rely on unmeasured (held-flat, ramped or '
                   f'extrapolated) parts of the sky curve for more than {SKY_MAX_UNMEASURED:.0%} of their width')
        verdict = f'not applied because unsafe: {why}'
    elif not ok_b:
        verdict = (f'not applied: neither the offset (≤ {SKY_SIGNIFICANCE:g}σ) nor the variation '
                   f'(p ≥ {SKY_VARIATION_P:g}) is significant')
    else:
        verdict = f'not applied: the correction is below {SKY_FLOOR_RESEL:g} resolution elements'
    t['verdict'] = verdict
    # unsafe is a fallback the user should see (WARNING); "not needed" is ordinary (INFO)
    (log.info if ok_a and not (apply and poor) else log.warning)(
        '%s: sky zero point %+.1f ± %.1f km/s (%.1fσ incl. arc RMS %.1f km/s; segment variation χ² %.1f for %d dof, '
        'p = %.2g; %d segments; regions measured ≥ %.0f per cent; max |Δv| at the regions %.1f km/s, floor %.1f '
        'km/s): %s', name, mean, sigma, t['n_sigma'], arc if np.isfinite(arc) else 0., chi2, len(v) - 1, p,
        zp.n_used, 100. * t['min_measured_fraction'] if fracs else 0., t['max_correction_kms'], t['floor_kms'],
        verdict)
    return apply, t


def _telluric_coverage(tzp: Any, spec: Spectrum, regions: Sequence) -> tuple[bool, str]:
    """Whether the telluric curve may be applied at the regions: a node within 250 Å of each, and not O2 A alone"""
    used = [s for s in tzp.segments if s.get('used')]
    if used and all(str(s['band']).startswith('O2') for s in used):
        return False, 'the curve is anchored by the O2 bands alone'
    if not regions:
        return True, 'ok'
    ff = telluric_zeropoint.frame_factor(spec.meta)
    nodes = np.asarray(tzp.node_wave, float)
    far = [(r.name, float(np.min(np.abs(nodes - 0.5 * (r.wmin + r.wmax) / ff)))) for r in regions]
    far = [(n, d) for n, d in far if d > TELLURIC_MAX_NODE_GAP]
    if far:
        return False, (f'{len(far)} region(s) lie more than {TELLURIC_MAX_NODE_GAP:.0f} Å from every telluric node ('
                       + ', '.join(f'{n}: {d:.0f} Å' for n, d in far[:5]) + (', ...' if len(far) > 5 else '') + ')')
    return True, 'ok'


def _read_source(path: str, cfg: Any, hdu: int | None = None) -> Spectrum:
    """Read a calibration file the user named (sky_source, telluric_source); a failure raises"""
    return io.read_spectrum(path, medium=cfg.input.medium, medium_method=cfg.input.medium_method, hdu=hdu)


def _object_tables(spec: Spectrum, cfg: Any) -> list[tuple[int, Spectrum]]:
    """Every object table of a multi-object PypeIt file (hdu, spectrum), the one read first included"""
    if int(spec.meta.get('n_objects', 1) or 1) < 2 or not Path(str(spec.source)).is_file():
        return [(spec.meta.get('hdu', -1), spec)]
    with fits.open(spec.source) as hdul:
        tables = [i for i in range(1, len(hdul)) if getattr(hdul[i], 'columns', None) is not None
                  and ('OPT_WAVE' in hdul[i].columns.names or 'BOX_WAVE' in hdul[i].columns.names)]
    out = []
    for k in tables:
        if k == spec.meta.get('hdu'):
            out.append((k, spec))
            continue
        try:
            out.append((k, _read_source(spec.source, cfg, hdu=k)))
        except (ValueError, KeyError) as exc:  # not an object table
            log.debug('object table %d of %s unreadable: %s', k, spec.source, exc)
    return out


def valid_range(cfg: Any, name: str, *spectra: Spectrum | None) -> tuple[float, float] | None:
    """``calibration.wave_valid_range`` if set, else the arc-line range found next to a PypeIt file (INFO-logged)"""
    if cfg.calibration.wave_valid_range:
        return cfg.calibration.wave_valid_range
    for s in spectra:
        arc = s.meta.get('arc_range') if s is not None else None
        if arc:
            log.info('%s: calibration.wave_valid_range not set; using the arc-line range %.0f-%.0f Å of %s', name,
                     arc[0], arc[1], s.meta.get('arc_file', 'wvcalib'))
            return (float(arc[0]), float(arc[1]))
    return None


def _measure_sky(src: Spectrum, cfg: Any, name: str, fallback: Spectrum | None = None):
    """
    The sky zero point of `src`; for a multi-object PypeIt file, every object's sky is measured (the same slit and
    wavelength solution) and the best measured (most usable segments, then the smallest node errors) is used
    """
    kw = dict(seg_width=cfg.calibration.sky_segment_width, valid_range=valid_range(cfg, name, src, fallback))
    tables = [(h, s) for h, s in _object_tables(src, cfg) if s.sky is not None]
    if len(tables) < 2:
        zp = sky_zeropoint.measure(src, **kw)
        if zp is not None:
            zp.source_hdu = src.meta.get('hdu')
            zp.source_spectrum = src
        return zp
    found = []
    for h, s in tables:
        z = sky_zeropoint.measure(s, **kw)
        if z is not None:
            z.source_hdu = h
            z.source_spectrum = s
            found.append(z)
    if not found:
        return None
    best = max(found, key=lambda z: (z.n_used, -float(np.mean(z.node_err))))
    log.info('%s: sky zero point from the sky of HDU %s of %s (%s)', name, best.source_hdu, Path(str(src.source)).name,
             '; '.join(f'HDU {z.source_hdu}: {z.n_used} segments, mean node error {np.mean(z.node_err):.1f} km/s'
                       for z in found))
    return best


def _highpass(f: np.ndarray) -> np.ndarray:
    """Stellar features only: flux calibration and slopes divided out"""
    g = np.where(np.isfinite(f), f, np.nanmedian(f))
    return g / np.maximum(median_filter(g, size=51, mode='nearest'), 1e-30) - 1.


def matching_object(path: str, raw: Spectrum, cfg: Any, name: str, what: str = 'telluric_source') -> Spectrum:
    """
    The object of a calibration file whose flux best matches the input (a PypeIt spec1d holds one table per
    object in the slit, and the highest-S/N one is not necessarily the input's target)
    """
    first = _read_source(path, cfg)
    found = _object_tables(first, cfg)
    if len(found) == 1:
        return first
    ref = _highpass(raw.flux)
    tell = np.zeros(len(raw.wave), dtype=bool)
    for f in features():
        if f.kind == 'telluric':
            tell |= (raw.wave > f.wmin) & (raw.wave < f.wmax)

    def score(s: Spectrum) -> float:
        """Correlation of the high-passed fluxes outside the telluric bands, less |ln| of the flux-level ratio"""
        f = np.interp(raw.wave, s.wave, s.flux, left=np.nan, right=np.nan)
        ok = np.isfinite(f) & np.isfinite(raw.flux) & ~tell
        if ok.sum() < 10:
            return -np.inf
        corr = float(np.corrcoef(_highpass(f)[ok], ref[ok])[0, 1])
        ratio = np.nanmedian(f[ok]) / np.nanmedian(raw.flux[ok])
        return corr - (abs(float(np.log(ratio))) if ratio > 0 else np.inf)

    scores = [score(s) for _, s in found]
    k = int(np.argmax(scores))
    log.info('%s: calibration.%s %s: object HDU %d matches the input best (score %.3f: high-pass '
             'correlation − |ln flux ratio|; others %s)', name, what, path, found[k][0], scores[k],
             ', '.join(f'{h}: {c:.3f}' for (h, _), c in zip(found, scores, strict=True) if h != found[k][0]))
    return found[k][1]


def _telluric_zeropoint(src: Spectrum, cfg: Any, name: str):
    """
    The star's telluric zero point; the star is modelled with the template nearest the priors when available, and
    each node's error includes the spread over the neighbouring templates (± one Teff grid step)
    """
    from ..methods.telluric_ref import neighbour_templates, prior_template
    width = cfg.calibration.telluric_segment_width
    teff = cfg.templates.teff_prior
    resolution = cfg.input.resolution or src.resolution
    template, alts = None, []
    if cfg.calibration.telluric_template:
        if resolution is None:  # a first pass without the template measures R from the telluric lines
            first = telluric_zeropoint.measure(src, seg_width=width, teff=teff)
            resolution = first.resolution if first is not None else None
        if resolution is not None:
            wr = (float(src.wave[0]), float(src.wave[-1]))
            try:
                template = prior_template(cfg, resolution, wr)
                alts = neighbour_templates(cfg, resolution, wr, template)
            except ValueError as exc:  # no template covers the spectrum: a smooth stellar continuum
                log.warning('%s: no template for the telluric zero point (%s); assuming a smooth star', name, exc)
    return telluric_zeropoint.measure(src, resolution=resolution, template=template, seg_width=width, teff=teff,
                                      spread_templates=alts)


def _check_sky_source(raw: Spectrum, other: Spectrum, name: str, path: str) -> None:
    """Warn if a calibration file does not look like it shares the input's wavelength solution"""
    lo, hi = max(raw.wmin, other.wmin), min(raw.wmax, other.wmax)
    overlap = max(0., hi - lo) / (raw.wmax - raw.wmin)
    ratio = len(other.wave) / max(len(raw.wave), 1)
    if overlap < 0.9 or not 0.67 < ratio < 1.5:
        log.warning('%s: calibration.sky_source %s covers %.0f per cent of the input range with %d pixels against %d; '
                    'it should share the input\'s wavelength solution (e.g. the spec1d of a coadd)', name, path,
                    100 * overlap, len(other.wave), len(raw.wave))


def correct(raw: Spectrum, spec: Spectrum, cfg: Any, name: str = '', regions: dict | Sequence | None = None
            ) -> tuple[Spectrum, dict]:
    """
    Measure the wavelength zero point of one spectrum and apply what ``calibration.wavelength_correction`` asks for

    ``raw`` is the spectrum as read, ``spec`` as it will be measured (e.g. barycentric-corrected; the correction is
    a multiplicative factor). ``regions`` (selected before any correction) feed the sky-if-needed and telluric
    coverage tests. Modes: 'sky-if-needed' the sky curve when `_sky_needed` passes; 'auto' the same plus the
    star − sky constant when measured ('sky+star'); 'sky' the sky curve; 'telluric' the telluric curve if it covers
    the regions; 'none' nothing. The tellurics are measured for 'telluric', when the sky is applied in
    sky-if-needed/auto, or whenever ``calibration.telluric_source`` is given. Returns the spectrum (with
    meta['sky_zeropoint'], ['telluric_zeropoint'] and ['star_minus_sky'] when measured) and {'mode', 'applied',
    'zp_flags', 'sky_tests', 'star_sky_applied', 'star_sky', 'slit', *ZP_KEYS}.
    """
    cal = cfg.calibration
    mode = cal.wavelength_correction
    nan = float('nan')
    info: dict[str, Any] = {'mode': mode, 'applied': 'none', 'zp_flags': 0, 'sky_tests': {},
                            'star_sky_applied': False, 'star_sky': {}, 'slit': None, **dict.fromkeys(ZP_KEYS, nan)}
    regions = _region_list(regions)
    # sky: the shape of the wavelength solution
    zp = None
    src = raw if raw.sky is not None else None
    if src is None and cal.sky_source:
        other = _read_source(cal.sky_source, cfg)
        if other.sky is None:
            log.warning('%s: calibration.sky_source %s has no sky spectrum', name, cal.sky_source)
        else:
            _check_sky_source(raw, other, name, cal.sky_source)
            src = other
    if src is not None:
        zp = _measure_sky(src, cfg, name, raw)
    else:
        info['zp_flags'] |= ZP_FLAGS['no_sky']
        log.info('%s: no sky spectrum: the sky zero point is not measured%s', name,
                 ' and nothing is applied' if mode in ('sky', 'sky-if-needed') else '')
    if zp is not None:
        info.update(zp_sky_kms=zp.zp_kms, zp_sky_err=zp.zp_err, zp_sky_slope=zp.slope,
                    zp_sky_model_spread=zp.model_spread)
        spec = replace(spec, meta={**spec.meta, 'sky_zeropoint': {**zp.as_dict(), 'source_hdu': zp.source_hdu}})
    # does the sky need correcting? (calibration data and region positions only)
    sky_ok = False
    if zp is not None and mode in ('sky-if-needed', 'auto'):
        sky_ok, info['sky_tests'] = _sky_needed(zp, spec, cfg, regions, name)
        if not sky_ok:
            info['zp_flags'] |= ZP_FLAGS['sky_not_needed' if info['sky_tests']['coverage_ok'] else 'sky_unsafe']
    elif zp is not None and mode == 'sky':
        sky_ok = True
    # telluric (the star's own zero point): for 'telluric', and for the star - sky constant when the sky correction
    # is applied; always measured when a telluric_source is given
    tzp, tsrc = None, None
    if mode == 'telluric' or cal.telluric_source or (mode in ('sky-if-needed', 'auto') and sky_ok):
        loud = log.warning if (mode == 'telluric' or cal.telluric_source) else log.info
        if not telluric.available():
            info['zp_flags'] |= ZP_FLAGS['telluric_unavailable']
            loud('%s: telluric zero point skipped: %s', name, telluric.unavailable_message())
        else:
            if cal.telluric_source:
                tsrc = matching_object(cal.telluric_source, raw, cfg, name)
                _check_sky_source(raw, tsrc, name, cal.telluric_source)
            elif not cfg.input.telluric_corrected:
                tsrc = raw
            else:
                loud('%s: the input is telluric-corrected and no calibration.telluric_source is set; the telluric '
                     'zero point (and so the star - sky constant) is not measured', name)
            if tsrc is not None:
                tzp = _telluric_zeropoint(tsrc, cfg, name)
    # inputs of the slit-centring prior and of the DAR (always reported when available): the target's own spec1d
    # table first (the telluric match, else the sky_source object matching the input), then the input
    cands = [tsrc, raw]
    if tsrc is None and cal.sky_source and not raw.meta.get('trace_fwhm_pix'):
        cands.insert(0, matching_object(cal.sky_source, raw, cfg, name, what='sky_source'))
    info['slit'] = slit.inputs(cands)
    info['dar'] = _dar_geometry(raw, tsrc, cfg, name, info)
    if tzp is not None:
        info.update(zp_tell_kms=tzp.zp_kms, zp_tell_err=tzp.zp_err)
        spec = replace(spec, meta={**spec.meta, 'telluric_zeropoint': tzp.as_dict()})
        if zp is not None:
            ss = _star_minus_sky(tzp, zp, info, tsrc, cfg, name)
            info.update(zp_star_sky=ss['zp_star_sky'], zp_star_sky_err=ss['zp_star_sky_err'],
                        zp_star_sky_n=ss['n_nodes'], zp_dar_s=ss['s'], zp_dar_s_err=ss['s_err'], star_sky=ss)
            if not np.isfinite(ss['zp_star_sky']):
                info['zp_flags'] |= ZP_FLAGS['star_sky_unmeasured']
            elif not ss['consistent']:
                info['zp_flags'] |= ZP_FLAGS['star_sky_inconsistent']
            spec = replace(spec, meta={**spec.meta, 'star_minus_sky': ss})
    elif mode == 'telluric':
        log.warning('%s: wavelength_correction = "telluric" but the telluric zero point could not be measured; '
                    'nothing applied', name)
    # apply: the sky measured in each region's own window where it can be (`_region_sky`)
    if sky_ok:
        zp = _region_sky(zp, regions, spec, cfg, name, info)
    if sky_ok and mode == 'auto' and np.isfinite(info['zp_star_sky']):  # opt-in until validated beyond J1745
        spec = sky_zeropoint.apply(spec, zp, constant_kms=info['zp_star_sky'], constant_err=info['zp_star_sky_err'],
                                   dar=_dar_table(info, spec), source='sky+star')
        info['applied'] = 'sky+star'
        info['star_sky_applied'] = True
        info['zp_flags'] |= ZP_FLAGS['sky_applied'] | ZP_FLAGS['star_sky_applied']
    elif sky_ok:
        spec = sky_zeropoint.apply(spec, zp, source='sky')
        info['applied'] = 'sky'
        info['zp_flags'] |= ZP_FLAGS['sky_applied']
    elif mode == 'telluric' and tzp is not None:
        ok, why = _telluric_coverage(tzp, spec, regions)
        if ok:
            spec = sky_zeropoint.apply(spec, tzp, source='telluric')
            info['applied'] = 'telluric'
            info['zp_flags'] |= ZP_FLAGS['telluric_applied']
        else:
            info['zp_flags'] |= ZP_FLAGS['telluric_refused']
            log.warning('%s: the telluric zero-point curve is not applied: %s', name, why)
    if info['applied'] != 'none':
        wc = spec.meta['wavelength_correction']
        log.info('%s: wavelength correction applied (%s): Δv %+.1f to %+.1f km/s over the spectrum%s', name,
                 info['applied'], wc['dv_min'], wc['dv_max'],
                 f"; the star - sky constant {info['zp_star_sky']:+.1f} ± {info['zp_star_sky_err']:.1f} km/s "
                 f"({info['zp_star_sky_n']} nodes; DAR s = {info['zp_dar_s']:+.3f}) is included and its error is "
                 f"sys_star_sky" if info['star_sky_applied'] else '')
    else:
        log.info('%s: no wavelength correction applied (mode %s)%s', name, mode,
                 '; only the measured curve\'s uncertainty (not its value) is counted in sys_wavezp'
                 if zp or tzp else '')
        if zp is not None and abs(zp.zp_kms) > 3. * zp.zp_err:
            log.warning('%s: the OH sky lines say the wavelength solution is offset by %+.1f ± %.1f km/s (%.0f-%.0f '
                        'Å), but no correction is applied (mode %s) and that offset is not in the errors', name,
                        zp.zp_kms, zp.zp_err, *zp.main_span, mode)
    if np.isfinite(info['zp_star_sky']) and not info['star_sky_applied']:
        log.info('%s: the measured star - sky offset %+.1f ± %.1f km/s (%d nodes) is not applied (mode %s, %s): its '
                 'value is not an error; the slit-centring prior is used instead', name, info['zp_star_sky'],
                 info['zp_star_sky_err'], info['zp_star_sky_n'], mode, info['applied'])
    return spec, info


def _region_sky(zp: Any, regions: Sequence, spec: Spectrum, cfg: Any, name: str, info: dict) -> Any:
    """
    The sky curve to apply, with one node per region: the sky Δv measured directly in a window centred on the
    region, so each region gets its local zero point rather than the curve interpolated between 250 Å segments

    The window starts at the region's width (at least `sky_zeropoint.MIN_WIDTH`) and grows by 25 per cent until
    `measure_windows` uses it (correlation ≥ 0.7 and the synthetic bias test within 0.1 resolution elements: the
    minimum width the bias test allows) and its error (statistical ⊕ bias) is no larger than the curve's there,
    so the local value is never noisier than the curve; up to ``calibration.sky_segment_width``, clipped to the
    telluric-free span holding the region, so no deep telluric band enters it. Where no window passes, the
    curve's own value and error are used (WARNING, flag sky_window_fallback). Regions that the coverage test
    drops (`_sky_needed`) are skipped. Without regions the curve is returned unchanged.
    """
    dropped = set(info.get('sky_tests', {}).get('uncovered', ()))
    regions = [r for r in regions if r.name not in dropped]
    if not regions:
        return zp
    ff = telluric_zeropoint.frame_factor(spec.meta)
    nodes, fallback = [], []
    for r in regions:
        lo, hi = r.wmin / ff, r.wmax / ff
        c = 0.5 * (lo + hi)
        curve_err = float(zp.dv_err(c))
        span = next(((a, b) for a, b in zp.span_bounds.values() if a <= c <= b), None)
        width, res = max(hi - lo, sky_zeropoint.MIN_WIDTH), None
        while span is not None:
            a, b = max(c - width / 2., span[0]), min(c + width / 2., span[1])
            res = sky_zeropoint.measure_windows(zp.source_spectrum, [(a, b)], zp.fwhm)[0]
            res['used'] &= bool(res['err'] <= curve_err)
            if res['used'] or width >= cfg.calibration.sky_segment_width or (a, b) == span:
                break
            width = min(1.25 * width, cfg.calibration.sky_segment_width)
        if res is not None and res['used']:
            nodes.append((c, res['dv'], res['err'], r.name, f"window {res['wmin']:.0f}-{res['wmax']:.0f} Å"))
        else:
            fallback.append(r.name)
            nodes.append((c, float(zp.dv(c)), curve_err, r.name, 'sky curve (fallback)'))
    nodes.sort()
    if fallback:
        info['zp_flags'] |= ZP_FLAGS['sky_window_fallback']
        log.warning('%s: no usable sky window for region(s) %s (too few lines, the bias test failed or noisier than '
                    'the curve); the sky curve is applied there instead', name, ', '.join(fallback))
    info['sky_windows'] = {n: dict(wave=w, dv=v, err=e, source=src) for w, v, e, n, src in nodes}
    log.info('%s: sky Δv per region: %s', name, '; '.join(f'{n} {v:+.1f} ± {e:.1f} km/s ({src})'
                                                           for w, v, e, n, src in nodes))
    w = np.array([n[0] for n in nodes])
    keep = np.concatenate([[True], np.diff(w) > 0])  # regions sharing a centre share a node
    return replace(zp, node_wave=w[keep], node_dv=np.array([n[1] for n in nodes])[keep],
                   node_err=np.array([n[2] for n in nodes])[keep], node_span=np.zeros(int(keep.sum()), dtype=int),
                   main_span=(float(w.min()), float(w.max())),
                   span_bounds={0: (float(min(a for a, _ in zp.span_bounds.values())),
                                    float(max(b for _, b in zp.span_bounds.values())))})


def _dar_geometry(raw: Spectrum, tsrc: Spectrum | None, cfg: Any, name: str, info: dict) -> dict:
    """
    The DAR geometry (`dar.geometry`) from the raw headers of the input's frames; header items the input lacks (a
    tellcorr file has almost none) come from the telluric source, the target's own spec1d table
    """
    meta = {**(tsrc.meta if tsrc is not None else {}),
            **{k: v for k, v in raw.meta.items() if v is not None and v != []}}
    geo = dar.geometry(io.raw_headers(meta, cfg.input.raw_header, cfg.input.raw_header_dir), {**meta, 'object': name})
    if not geo['pa_known']:
        info['zp_flags'] |= ZP_FLAGS['dar_pa_unknown']
    if not np.isfinite(geo['zenith_deg']) or info['slit'].scale_spec is None:
        info['zp_flags'] |= ZP_FLAGS['dar_unknown']
        log.warning('%s: the DAR across the slit cannot be modelled (no zenith distance or plate scale): it is left '
                    'out of the star - sky fit and the slit prior', name)
    elif geo['pa_known']:
        log.info('%s: DAR across the slit: s = %+.3f ± %.3f (slit PA %.1f deg east of north, parallactic angle '
                 '%.1f-%.1f deg, zenith distance %.1f deg; %s)', name, geo['s'], geo['s_err'],
                 geo['pa_disp_deg'][0] + 90., min(min(q) for q in geo['q_deg']), max(max(q) for q in geo['q_deg']),
                 geo['zenith_deg'], geo['source'])
    else:
        log.warning('%s: the slit PA is unknown (no raw header: set input.raw_header or input.raw_header_dir): the '
                    'DAR factor s is fitted with a uniform prior in [-1, 1] (star - sky) or marginalised over it '
                    '(slit prior)', name)
    return geo


def _dar_nodes(info: dict, lam: np.ndarray) -> np.ndarray | None:
    """
    D(λ) [km/s per unit s] at observatory-frame `lam` relative to info['dar_ref'] (the star − sky fit's reference;
    without it the mean of `lam`, for the prior, which only uses D − mean D); None if unknown
    """
    if info['zp_flags'] & ZP_FLAGS['dar_unknown']:
        return None
    return dar.dar_kms(info['dar'], info['slit'], lam, info.get('dar_ref', float(np.mean(lam))))


def _star_minus_sky(tzp: Any, zp: Any, info: dict, tsrc: Spectrum, cfg: Any, name: str) -> dict:
    """
    The star − sky fit a + s·D(λ) on the telluric nodes, the sky measured directly in each node's window. The
    telluric fit is repeated with the other template families (`_family_templates`), and the standard deviation of
    a over the families (`err_family`) is added in quadrature to its error.
    """
    info['dar_ref'] = float(np.mean(tzp.node_wave))
    geo = info['dar']
    s_prior = (geo['s'], geo['s_err']) if geo['pa_known'] else None

    def fit(t: Any) -> dict:
        windows = sky_zeropoint.measure_windows(zp.source_spectrum, telluric_zeropoint.node_windows(t), zp.fwhm)
        return telluric_zeropoint.star_minus_sky_direct(t, windows, _dar_nodes(info, t.node_wave), s_prior)

    ss = fit(tzp)
    ss['dar_ref'] = info['dar_ref']
    if not np.isfinite(ss['zp_star_sky']):
        return ss
    family = {tzp.template or 'smooth star': ss['zp_star_sky']}
    for label, template in _family_templates(tsrc, cfg, name, tzp):
        alt = telluric_zeropoint.measure(tsrc, resolution=tzp.resolution, template=template,
                                         teff=cfg.templates.teff_prior,
                                         seg_width=cfg.calibration.telluric_segment_width)
        family[label] = fit(alt)['zp_star_sky'] if alt is not None else float('nan')
    vals = [v for v in family.values() if np.isfinite(v)]
    spread = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.
    log.info('%s: star - sky constant over the telluric-fit template families: %s; spread %.1f km/s added in '
             'quadrature to its error', name, ', '.join(f'{k} {v:+.1f}' for k, v in family.items()), spread)
    ss.update(family=family, err_family=spread, err_fit=ss['zp_star_sky_err'],
              zp_star_sky_err=float(np.hypot(ss['zp_star_sky_err'], spread)))
    return ss


def _family_templates(tsrc: Spectrum, cfg: Any, name: str, tzp: Any) -> list[tuple[str, Any]]:
    """
    The other star models of the telluric fit: a smooth star, and the prior template of every other configured or
    cached grid (e.g. SAND next to BT-Settl); a grid with no template near the priors is skipped (INFO)
    """
    import copy

    from .. import templates as T
    from ..methods.guidance import grid_names
    from ..methods.telluric_ref import prior_template
    out: list[tuple[str, Any]] = [('smooth star', None)] if tzp.template is not None else []
    if not cfg.calibration.telluric_template or not np.isfinite(tzp.resolution):
        return out
    configured = grid_names(cfg.templates)
    cache = T.cache_dir()
    # a downloaded grid has its metadata file (an empty or partial directory is not a cached grid)
    cached = sorted(d.name for d in cache.iterdir() if d.name in T.KNOWN_GRIDS
                    and (d / 'rvfitter_grid.toml').is_file()) if cache.is_dir() else []
    wr = (float(tsrc.wave[0]), float(tsrc.wave[-1]))
    for grid in dict.fromkeys(configured[1:] + cached):
        if grid == configured[0]:
            continue
        alt = copy.deepcopy(cfg)
        alt.templates.grid = grid
        if not isinstance(alt.templates.local_dir, dict):
            alt.templates.local_dir, alt.templates.medium = None, None  # that directory was the first grid's
        try:
            t = prior_template(alt, tzp.resolution, wr)
        except ValueError as exc:
            log.info('%s: grid %s gives no telluric-fit template (%s)', name, grid, exc)
            continue
        out.append((t.label, t))
    return out


def _dar_table(info: dict, spec: Spectrum) -> dict | None:
    """The fitted s·D(λ) tabulated over the spectrum (observatory frame), for `sky_zeropoint.apply`"""
    ff = telluric_zeropoint.frame_factor(spec.meta)
    wave = np.linspace(spec.wave[0] / ff, spec.wave[-1] / ff, 200)
    d = _dar_nodes(info, wave)
    return None if d is None else dict(wave=[float(x) for x in wave], kms=[float(x) for x in info['zp_dar_s'] * d])


def star_sky_term(info: dict, spec: Spectrum, waves: np.ndarray | None, cfg: Any, name: str) -> dict[str, Any]:
    """
    The star's absolute zero-point term (``sys_star_sky``) of one epoch

    The applied star − sky correction's error, the RMS over the measurements' wavelengths of
    σ²(a + s·D) = σ_a² + D²σ_s² + 2D cov(a, s); none (NaN) when the telluric curve was applied or with
    ``calibration.zero_point_known``; otherwise the slit-centring prior (``calibration.slit_prior``: a number
    [km/s], 'auto' for `slit.prior_kms` at the measurements' wavelengths, or 'none'). The 'auto' prior gains the
    DAR term in quadrature: the RMS over those wavelengths of s·(D − mean D) (the constant part is inside the
    position prior), with s² + σ_s² for a known slit PA and ⟨s²⟩ = 1/3 (s uniform in [−1, 1]) otherwise. When it
    cannot be computed a WARNING says the absolute zero point is unconstrained (NaN, flag zp_unconstrained). The
    computed prior is always reported (``prior``) when its inputs exist.
    """
    choice = cfg.calibration.slit_prior
    nan = float('nan')
    inp = info.get('slit')
    lam = waves if waves is not None and len(waves) else np.array([float(np.median(spec.wave))])
    lam = lam / float(telluric_zeropoint.frame_factor(spec.meta))  # the slit offset is in the observatory frame
    prior = slit.prior_kms(inp, lam) if inp is not None else nan
    out = dict(value=nan, term='', prior=prior, dar=nan, flags=0, slit=inp.as_dict() if inp is not None else None)
    if np.isfinite(prior):
        log.info('%s: slit-centring prior %.1f km/s (slit %.2f arcsec from %s; seeing FWHM %.2f arcsec from %s; '
                 'σ_x = (W - F)/√12 = %.3f arcsec; %.3f arcsec per spectral pixel from %s; dispersion from the %s)',
                 name, prior, inp.width_arcsec, inp.width_source, inp.fwhm_arcsec, inp.fwhm_source,
                 inp.sigma_arcsec, inp.scale_spec, inp.scale_source, inp.disp_source)
    dar_d = _dar_nodes(info, lam) if info.get('dar') is not None else None
    if dar_d is not None:  # the DAR's share of the term (see dar.py and the docstring)
        dc = dar_d - np.mean(dar_d)
        if info.get('star_sky_applied'):
            out['dar'] = float(np.sqrt(np.mean(info['zp_dar_s_err'] ** 2 * dar_d ** 2)))
        elif info['dar']['pa_known']:
            out['dar'] = float(np.hypot(info['dar']['s'], info['dar']['s_err']) * np.sqrt(np.mean(dc * dc)))
        else:
            out['dar'] = float(np.sqrt(np.mean(dc * dc) / 3.))  # s uniform in [-1, 1]: <s²> = 1/3
    if info.get('star_sky_applied'):
        ss = info['star_sky']
        var = info['zp_star_sky_err'] ** 2 + (0. if dar_d is None else
                                              dar_d ** 2 * ss['s_err'] ** 2 + 2. * dar_d * ss['cov_as'])
        out.update(value=float(np.sqrt(np.mean(np.broadcast_to(var, lam.shape)))),
                   term='star - sky constant + DAR (measured, applied)')
    elif info.get('applied') == 'telluric':
        out.update(term='none: the star\'s telluric curve was applied')
    elif cfg.calibration.zero_point_known:
        out.update(term='none: calibration.zero_point_known')
        log.info('%s: calibration.zero_point_known: no slit-centring term', name)
    elif isinstance(choice, (int, float)) and not isinstance(choice, bool):
        out.update(value=float(choice), term='slit prior (calibration.slit_prior)', flags=ZP_FLAGS['slit_prior'])
    elif str(choice) == 'none':
        out.update(term='none: calibration.slit_prior = "none"', flags=ZP_FLAGS['zp_unconstrained'])
        log.warning('%s: calibration.slit_prior = "none": the absolute wavelength zero point (slit centring) is '
                    'unconstrained and not in the errors', name)
    elif np.isfinite(prior):
        out.update(value=float(np.hypot(prior, out.get('dar', 0.))), term='slit prior (W - F)/√12 ⊕ DAR',
                   flags=ZP_FLAGS['slit_prior'])
    else:
        if inp is None:
            why = 'no slit information'
        elif inp.missing:
            why = 'missing ' + '; '.join(inp.missing)
        else:
            why = (f'the seeing FWHM {inp.fwhm_arcsec:.2f} arcsec is not smaller than the slit width '
                   f'{inp.width_arcsec:.2f} arcsec')
        out.update(term=f'unconstrained ({why})', flags=ZP_FLAGS['zp_unconstrained'])
        log.warning('%s: the absolute wavelength zero point is unconstrained: the star - sky constant is not applied '
                    'and the slit-centring prior cannot be computed (%s); sys_star_sky is NaN. Set '
                    'calibration.slit_prior [km/s], or calibration.zero_point_known = true for fibre/IFU data', name,
                    why)
    log.info('%s: sys_star_sky = %.1f km/s (%s; DAR share %.1f km/s)', name, out['value'], out['term'],
             out['dar'])
    return out


def combine_epochs(zp_epochs: list[dict]) -> dict[str, Any]:
    """
    One set of zero-point terms for the object: an epoch's own with one epoch; with several, the mean of the
    offsets, the RMS of the errors and systematics, the applied corrections joined by '/' and the flags OR-ed
    """
    if len(zp_epochs) == 1:
        z = dict(zp_epochs[0])
    else:
        z = {}
        for key in ZP_KEYS:
            vals = [e[key] for e in zp_epochs]
            fin = [v for v in vals if np.isfinite(v)]
            if key.endswith(('_err', '_spread')):
                z[key] = rms(vals)
            elif key == 'zp_star_sky_n':
                z[key] = float(np.sum(fin)) if fin else float('nan')
            else:
                z[key] = float(np.mean(fin)) if fin else float('nan')
        for key in ('wavelength_zero_point', 'star_sky', 'slit_prior'):
            z[key] = rms([e[key] for e in zp_epochs])
        z['star_sky_term'] = '/'.join(dict.fromkeys(e['star_sky_term'] for e in zp_epochs))
        z['slit'] = zp_epochs[0]['slit']
        z['applied'] = '/'.join(dict.fromkeys(e['applied'] for e in zp_epochs))
        z['mode'] = zp_epochs[0]['mode']
        z['zp_flags'] = int(np.bitwise_or.reduce([e['zp_flags'] for e in zp_epochs]))
        z['sky_tests'] = zp_epochs[0]['sky_tests']
    z['star_sky_systematic'] = z.pop('star_sky')
    return z
