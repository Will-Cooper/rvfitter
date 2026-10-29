"""Orchestration of an analysis: objects, methods, combination and output.

Objects are processed sequentially; parallelism is used inside an object (a process pool passed
down to the methods), so results do not depend on ``n_procs``.

The configuration attributes follow the interface addendum in ``docs/architecture.md``.
"""
import importlib
import logging
import multiprocessing
import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .calibration import sky_zeropoint
from .types import Measurement, MethodResult, ObjectResult, Spectrum

__all__ = ['run', 'measure_object', 'enabled_methods', 'NoMethodsError', 'method_problem', 'wavelength_correction',
           'ZP_FLAGS', 'zp_flag_names',
           'refit_object', 'suggested_refits', 'apply_patch', 'config_for_object', 'object_seed']

log = logging.getLogger(__name__)

FLAG_FIT_FAILED = 1


def enabled_methods(cfg: Any, override: Sequence[str] | None = None) -> list[str]:
    """Names of the methods to run, from ``override`` or ``cfg.methods.enabled``"""
    if override:
        return list(override)
    return list(cfg.methods.enabled)


class NoMethodsError(RuntimeError):
    """None of the enabled methods is available; the message is user-ready"""


def _registry() -> dict[str, Any]:
    """The method registry, or an empty dict if ``rvfitter.methods`` has none yet"""
    return dict(getattr(importlib.import_module('rvfitter.methods'), 'REGISTRY', None) or {})


def method_problem(cfg: Any, methods: Sequence[str] | None = None) -> str | None:
    """Why no method can run (a user-ready message), or None if at least one enabled method is registered"""
    names = enabled_methods(cfg, methods)
    registry = _registry()
    if not names:
        return 'no RV methods available: none is enabled in the configuration'
    if not registry:
        return ('no RV methods available: rvfitter.methods has no registered methods yet '
                f'(enabled: {", ".join(names)})')
    if not any(m in registry for m in names):
        return (f'no RV methods available: none of the enabled methods ({", ".join(names)}) is registered '
                f'(registered: {", ".join(sorted(registry))})')
    return None


def _config_hash(cfg: Any) -> str:
    try:
        return str(cfg.hash())
    except Exception:  # noqa: BLE001
        return ''


class _Cancelled(Exception):
    """Raised inside measure_object when the cancel event is set"""


def _failed(method: str, region: str = '') -> MethodResult:
    meas = Measurement(method, region, float('nan'), float('nan'), FLAG_FIT_FAILED)
    return MethodResult(method, float('nan'), float('nan'), 0, [meas])


def measure_object(spectra: Sequence[Spectrum], cfg: Any, executor: Executor | None, rng: np.random.Generator, *,
                   name: str | None = None, methods: Sequence[str] | None = None,
                   bary: str | None = None, cancel: threading.Event | None = None) -> ObjectResult:
    """
    Measure one object (one or more spectra)

    Runs barycentric.correction, the wavelength zero point, regions.select, each enabled method,
    combine.hierarchical and systematics. A method that raises is logged and recorded as a failed measurement
    (flag 1); the object carries on. If ``cancel`` is set between methods, ``_Cancelled`` is raised (handled by
    ``run``). Several spectra are epochs: each is measured separately and the epochs are combined in a second
    level (random effects over the epochs' RVs), so a common per-epoch offset is not averaged down by the number of
    regions.
    """
    from . import barycentric, combine, regions, systematics

    names = enabled_methods(cfg, methods)
    registry = _registry()
    name = name or str(spectra[0].meta.get('object', Path(spectra[0].source).stem))
    mode = bary or cfg.velocity_frame.correction
    frame = cfg.velocity_frame.frame
    overrides = cfg.velocity_frame.overrides

    corrected, kmss, sources = [], [], []
    for spec in spectra:
        kms, source = barycentric.correction(spec, mode=mode, frame=frame, overrides=overrides)
        log.info('%s: barycentric correction %.3f km/s (%s)', name, kms, source)
        if source != 'skipped' and kms:  # recorded, so the telluric zero point can return to the observatory frame
            shifted = barycentric.apply(spec, kms)
            corrected.append(replace(shifted, meta={**shifted.meta, 'barycentric_applied_kms': float(kms)}))
        else:
            corrected.append(spec)
        kmss.append(float(kms))
        sources.append(source)
    # several epochs: report the mean correction applied, and the source of the first
    bary_info = {'applied_kms': float(np.mean(kmss)), 'source': sources[0], 'frame': frame}

    # None means "use the SelectOptions default"
    knobs = {'snr_min': cfg.regions.min_snr, 'telluric': cfg.regions.telluric_strictness,
             'telluric_corrected': True if cfg.input.telluric_corrected else None,
             'template_windows': getattr(cfg.regions, 'template_windows', None)}
    opts = regions.SelectOptions(**{k: v for k, v in knobs.items() if v is not None})

    def select(spec: Spectrum) -> dict:
        return regions.select(spec, teff=cfg.templates.teff_prior, methods=names, overrides=cfg.regions.overrides,
                              options=opts)

    # wavelength zero point (sky shape and/or the star's telluric zero point), applied once to the wavelengths; the
    # regions the methods will use (selected before any correction, which moves them by much less than their width)
    # feed the sky-if-needed and telluric coverage tests
    cal_mode = getattr(getattr(cfg, 'calibration', None), 'wavelength_correction', 'sky-if-needed')
    cal_infos = []
    for k, (raw, spec) in enumerate(zip(spectra, corrected, strict=True)):
        pre = select(spec) if cal_mode in ('sky-if-needed', 'telluric', 'auto') else None
        corrected[k], info = wavelength_correction(raw, spec, cfg, name, regions=pre)
        cal_infos.append(info)

    method_rngs = dict(zip(names, rng.spawn(len(names)), strict=True))
    per_method: dict[str, list[MethodResult]] = {m: [] for m in names}
    sel0: dict[str, list] = {}
    for spec in corrected:
        sel = select(spec)
        if not sel0:
            sel0 = sel
        for m in names:
            if cancel is not None and cancel.is_set():
                raise _Cancelled
            if m not in registry:
                log.error('%s: unknown method %r', name, m)
                per_method[m].append(_failed(m))
                continue
            try:
                res = registry[m].measure(spec, sel.get(m, []), cfg, executor=executor, rng=method_rngs[m])
            except Exception as exc:  # noqa: BLE001
                log.error('%s: method %s failed: %s', name, m, exc)
                log.debug('traceback', exc_info=True)
                res = _failed(m)
            per_method[m].append(res)

    # first-order uncorrected RVs (the correction lowered each RV by the Δv applied at its wavelength), and the
    # wavelengths of each epoch's usable measurements (where the zero-point systematic is evaluated)
    from .results import EXCLUDE_FLAGS
    epoch_waves: list[list[float]] = [[] for _ in corrected]
    for lst in per_method.values():
        for k, res in enumerate(lst):
            spec = corrected[min(k, len(corrected) - 1)]
            for meas in res.measurements:
                lam = _measurement_wave(meas)
                dv = float(sky_zeropoint.applied_dv(spec.meta, lam)) if np.isfinite(lam) else np.nan
                meas.diagnostics['zp_applied_kms'] = dv
                meas.diagnostics['rv_uncorrected'] = float(meas.rv + dv) if np.isfinite(dv) else np.nan
                if np.isfinite(lam) and np.isfinite(meas.rv) and not meas.flag & EXCLUDE_FLAGS:
                    epoch_waves[min(k, len(corrected) - 1)].append(lam)

    # zero-point systematics of each epoch, at the wavelengths of its measurements
    zp_epochs = []
    for k, (spec, info) in enumerate(zip(corrected, cal_infos, strict=True)):
        waves = np.asarray(epoch_waves[k]) if epoch_waves[k] else None
        try:
            wavezp = float(systematics.wavelength_zero_point(spec, cfg, wave=waves))
        except Exception as exc:  # noqa: BLE001
            log.debug('wavelength zero-point unavailable: %s', exc)
            wavezp = float('nan')
        term = _star_sky_term(info, spec, waves, cfg, name)
        zp_epochs.append({**{key: info[key] for key in ZP_KEYS}, 'mode': info['mode'], 'applied': info['applied'],
                          'zp_flags': int(info['zp_flags']) | term['flags'], 'sky_tests': info['sky_tests'],
                          'wavelength_zero_point': wavezp, 'star_sky': term['value'], 'star_sky_term': term['term'],
                          'slit_prior': term['prior'], 'slit': term['slit']})

    # several epochs: each was measured separately; its regions become '<epoch>:<region>', and the epochs are
    # combined in a second level (random effects over the per-epoch RVs), so a common per-epoch offset counts once
    # per epoch rather than once per region
    labels = epoch_labels(corrected)
    epoch_info: dict[str, Any] = {}
    if len(corrected) > 1:
        epoch_info = _epoch_summaries(per_method, labels, cfg, name,
                                      sys=[z['wavelength_zero_point'] for z in zp_epochs])
        for lst in per_method.values():
            for k, res in enumerate(lst):
                for meas in res.measurements:
                    meas.diagnostics['epoch'] = labels[k]
                    meas.region = f'{labels[k]}:{meas.region}'

    results: dict[str, MethodResult] = {}
    for m, lst in per_method.items():
        res = lst[0] if len(lst) == 1 else _combine_epochs(m, lst, labels, name)
        results[m] = res
        if np.isfinite(res.rv):
            log.info('%s: %.1f ± %.1f km/s', m, res.rv, res.rv_err)
        else:
            log.info('%s: no result', m)

    adopted: dict[str, Any] = {}
    good = [r for r in results.values() if np.isfinite(r.rv)]
    if good and epoch_info:
        adopted = _adopted_over_epochs(epoch_info, cfg, name)
    elif good:
        try:
            # every usable per-region measurement of every method; flagged ones are excluded (and logged)
            adopted = combine.hierarchical_from_methods({r.method: r for r in good}, tau_scale=cfg.combine.tau_scale)
            if not np.isfinite(adopted.get('median', float('nan'))):  # nothing usable after the exclusions
                adopted = {}
        except Exception as exc:  # noqa: BLE001
            log.error('%s: hierarchical combination failed: %s', name, exc)
            log.debug('traceback', exc_info=True)
    if adopted and epoch_info:
        adopted['epochs'] = epoch_info
    if adopted:
        log.info('%s: adopted RV %.2f ± %.2f km/s', name, adopted.get('median', float('nan')),
                 adopted.get('err', float('nan')))
    else:
        log.warning('%s: no adopted RV', name)

    sysd: dict[str, Any] = _combine_zp_epochs(zp_epochs)
    bary_info['wavelength_correction'] = sysd.pop('applied')
    bary_info['wavelength_correction_mode'] = sysd.pop('mode')
    bary_info['zp_flags'] = int(sysd.pop('zp_flags'))
    bary_info['sky_tests'] = sysd.pop('sky_tests')
    if len(zp_epochs) > 1:
        bary_info['wavelength_correction_epochs'] = [
            {k: v for k, v in z.items() if k != 'sky_tests'} | {'epoch': lab}
            for z, lab in zip(zp_epochs, labels, strict=True)]
    try:
        sysd['template_spread'] = float(systematics.template_spread(corrected[0], sel0, cfg, executor=executor,
                                                                    rng=rng.spawn(1)[0]))
    except Exception as exc:  # noqa: BLE001
        log.debug('template spread unavailable: %s', exc)

    sysd['floor'] = float(cfg.combine.systematic_floor)
    return ObjectResult(object=name, files=[s.source for s in spectra], methods=results, adopted=adopted,
                        systematics=sysd, barycentric=bary_info, binary={}, config_hash=_config_hash(cfg),
                        version=__version__)


def _star_sky_term(info: dict, spec: Spectrum, waves: np.ndarray | None, cfg: Any, name: str) -> dict[str, Any]:
    """
    The star's absolute zero-point term (``sys_star_sky``) of one epoch

    - the star − sky constant applied ('sky+star'): its error;
    - the telluric curve applied, or ``calibration.zero_point_known`` (fibre, IFU or an absolute calibration):
      none (NaN);
    - otherwise the slit-centring prior: ``calibration.slit_prior`` as a number [km/s], or with 'auto' the
      prior `calibration.slit.prior_kms` at the measurements' wavelengths; 'none' disables it. When it cannot be
      computed, a WARNING says the absolute zero point is unconstrained and the term is NaN (flag
      zp_unconstrained).

    A measured but unapplied offset never enters as its value. The computed slit prior is always reported
    (``prior``) when its inputs exist, whether or not it is used.
    """
    from .calibration import slit
    from .calibration.telluric_zeropoint import frame_factor
    cal = getattr(cfg, 'calibration', None)
    choice = getattr(cal, 'slit_prior', 'auto')
    known = bool(getattr(cal, 'zero_point_known', False))
    nan = float('nan')
    inp = info.get('slit')
    lam = waves if waves is not None and len(waves) else np.array([float(np.median(spec.wave))])
    lam = lam / float(frame_factor(spec.meta))  # the slit offset is in the observatory frame's pixels
    prior = slit.prior_kms(inp, lam) if inp is not None else nan
    out = dict(value=nan, term='', prior=prior, flags=0, slit=inp.as_dict() if inp is not None else None)
    if np.isfinite(prior):
        log.info('%s: slit-centring prior %.1f km/s (slit %.2f arcsec from %s; seeing FWHM %.2f arcsec from %s; '
                 'σ_x = (W - F)/√12 = %.3f arcsec; %.3f arcsec per spectral pixel from %s; dispersion from the %s)',
                 name, prior, inp.width_arcsec, inp.width_source, inp.fwhm_arcsec, inp.fwhm_source,
                 inp.sigma_arcsec, inp.scale_spec, inp.scale_source, inp.disp_source)
    if info.get('star_sky_applied'):
        out.update(value=float(info['zp_star_sky_err']), term='star - sky constant (measured, applied)')
    elif info.get('applied') == 'telluric':
        out.update(term='none: the star\'s telluric curve was applied')
    elif known:
        out.update(term='none: calibration.zero_point_known')
        log.info('%s: calibration.zero_point_known: no slit-centring term', name)
    elif isinstance(choice, (int, float)) and not isinstance(choice, bool):
        out.update(value=float(choice), term='slit prior (calibration.slit_prior)', flags=ZP_FLAGS['slit_prior'])
    elif str(choice) == 'none':
        out.update(term='none: calibration.slit_prior = "none"', flags=ZP_FLAGS['zp_unconstrained'])
        log.warning('%s: calibration.slit_prior = "none": the absolute wavelength zero point (slit centring) is '
                    'unconstrained and not in the errors', name)
    elif np.isfinite(prior):
        out.update(value=prior, term='slit prior (W - F)/√12', flags=ZP_FLAGS['slit_prior'])
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
    log.info('%s: sys_star_sky = %.1f km/s (%s)', name, out['value'], out['term'])
    return out


def _rms(values: Sequence[float]) -> float:
    v = np.array([x for x in values if np.isfinite(x)], dtype=float)
    return float(np.sqrt(np.mean(v * v))) if len(v) else float('nan')


def _combine_zp_epochs(zp_epochs: list[dict]) -> dict[str, Any]:
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
                z[key] = _rms(vals)
            elif key == 'zp_star_sky_n':
                z[key] = float(np.sum(fin)) if fin else float('nan')
            else:
                z[key] = float(np.mean(fin)) if fin else float('nan')
        z['wavelength_zero_point'] = _rms([e['wavelength_zero_point'] for e in zp_epochs])
        z['star_sky'] = _rms([e['star_sky'] for e in zp_epochs])
        z['slit_prior'] = _rms([e['slit_prior'] for e in zp_epochs])
        z['star_sky_term'] = '/'.join(dict.fromkeys(e['star_sky_term'] for e in zp_epochs))
        z['slit'] = zp_epochs[0]['slit']
        z['applied'] = '/'.join(dict.fromkeys(e['applied'] for e in zp_epochs))
        z['mode'] = zp_epochs[0]['mode']
        z['zp_flags'] = int(np.bitwise_or.reduce([e['zp_flags'] for e in zp_epochs]))
        z['sky_tests'] = zp_epochs[0]['sky_tests']
    z['star_sky_systematic'] = z.pop('star_sky')
    return z


def _combine_epochs(method: str, lst: list[MethodResult], labels: list[str], name: str) -> MethodResult:
    """A method over several epochs: random effects over the epochs' own (region-combined) RVs"""
    from .combine import combine_regions
    per_epoch = [Measurement(method, lab, float(r.rv), float(r.rv_err), 0)
                 for r, lab in zip(lst, labels, strict=False)]
    meas = [x for r in lst for x in r.measurements]
    try:
        details: dict = {}
        rv, err = combine_regions([x for x in per_epoch if np.isfinite(x.rv) and np.isfinite(x.rv_err)],
                                  details=details)
    except Exception as exc:  # noqa: BLE001
        log.error('%s: combining %s over epochs failed: %s', name, method, exc)
        return MethodResult(method, float('nan'), float('nan'), 0, meas)
    for x in meas:
        x.diagnostics.update(method_tau_epoch=details.get('tau_line', np.nan))
    n = sum(r.n_regions for r in lst)
    n_ep = sum(1 for x in per_epoch if np.isfinite(x.rv) and np.isfinite(x.rv_err))
    log.info('%s: %s over %d epochs: %.2f ± %.2f km/s (τ_epoch %.1f km/s)', name, method, n_ep, rv, err,
             details.get('tau_line', np.nan))
    return MethodResult(method, rv, err, n, meas)


def _adopted_over_epochs(epoch_info: dict, cfg: Any, name: str) -> dict:
    """The adopted RV from the epochs' adopted RVs: random effects over epochs (the second level)"""
    from . import combine
    rv, err = np.asarray(epoch_info['rv'], float), np.asarray(epoch_info['err'], float)
    ok = np.isfinite(rv) & np.isfinite(err) & (err > 0)
    if not ok.any():
        return {}
    labels = [lab for lab, g in zip(epoch_info['labels'], ok, strict=True) if g]
    res = combine.hierarchical(rv[ok], err[ok], labels, ['epoch'] * int(ok.sum()), tau_scale=cfg.combine.tau_scale)
    res['tau_epoch'] = res.get('tau_line', float('nan'))
    for key in ('tau_line', 'tau_method'):
        vals = [v for v in epoch_info.get(key, []) if np.isfinite(v)]
        res[key] = float(np.median(vals)) if vals else float('nan')
    res['n_lines'] = int(np.nansum(epoch_info.get('n_lines', [0])))
    res['n_methods'] = int(np.nanmax(epoch_info.get('n_methods', [0]) or [0]))
    log.info('%s: adopted over %d epochs: %.2f ± %.2f km/s (τ_epoch %.1f km/s)', name, int(ok.sum()), res['median'],
             res['err'], res['tau_epoch'])
    return res


def _measurement_wave(meas: Measurement) -> float:
    """A representative wavelength of a measurement's region (line centre, else the median fitted wavelength)"""
    d = meas.diagnostics or {}
    for key in ('centre', 'wave_centre'):
        if d.get(key) is not None and np.isfinite(d[key]):
            return float(d[key])
    w = d.get('wave')
    if w is not None and len(w):
        return float(np.median(np.asarray(w, dtype=float)))
    return float('nan')


ZP_KEYS = ('zp_sky_kms', 'zp_sky_err', 'zp_sky_slope', 'zp_sky_model_spread', 'zp_tell_kms', 'zp_tell_err',
           'zp_star_sky', 'zp_star_sky_err', 'zp_star_sky_n')

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


def _region_list(sel: Any) -> list:
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


def _arc_kms(spec: Spectrum, cfg: Any) -> float:
    """The arc-solution RMS as a velocity [km/s] (cfg.input.wave_rms, else meta['wave_rms']); NaN if unknown"""
    from .constants import C_KMS
    rms = getattr(getattr(cfg, 'input', None), 'wave_rms', None)
    if rms is None:
        rms = spec.meta.get('wave_rms') if hasattr(spec.meta, 'get') else None
    return float(C_KMS * rms / np.median(spec.wave)) if rms is not None and np.isfinite(rms) else float('nan')


def _sky_needed(zp: Any, spec: Spectrum, cfg: Any, regions: Sequence | None, name: str) -> tuple[bool, dict]:
    """
    The sky-if-needed decision: apply the sky curve only if (a) it is well measured where the regions are, (b) the
    offset or the distortion is significant and (c) the correction exceeds a practical floor. Calibration data and
    region positions only (never the stellar line scatter).
    """
    from scipy.stats import chi2 as chi2_dist

    from .calibration import telluric_zeropoint
    from .constants import C_KMS
    t: dict[str, Any] = {'n_segments': int(zp.n_used)}
    # (a) coverage: the regions in the observatory frame against the measured sky segments
    ff = telluric_zeropoint.frame_factor(spec.meta)
    regs = list(regions or [])
    fracs = [zp.measured_fraction(r.wmin / ff, r.wmax / ff) for r in regs]
    t['region_measured'] = {r.name: round(f, 3) for r, f in zip(regs, fracs, strict=True)}
    t['min_measured_fraction'] = float(min(fracs)) if fracs else float('nan')
    poor = [r.name for r, f in zip(regs, fracs, strict=True) if f <= 1. - SKY_MAX_UNMEASURED]
    ok_a = zp.n_used >= SKY_MIN_SEGMENTS and bool(regs) and any(f > 0 for f in fracs) and not poor
    # (b) significance: weighted mean of the segment values, σ from the curve (with its model spread) and the arc
    v, e = np.asarray(zp.node_dv, float), np.asarray(zp.node_err, float)
    w = 1. / e ** 2
    mean = float(np.sum(w * v) / np.sum(w))
    spread = zp.model_spread if np.isfinite(zp.model_spread) else 0.
    arc = _arc_kms(spec, cfg)
    sigma = float(np.sqrt(1. / np.sum(w) + spread ** 2 + (arc ** 2 if np.isfinite(arc) else 0.)))
    chi2 = float(np.sum(w * (v - mean) ** 2))
    p = float(chi2_dist.sf(chi2, len(v) - 1)) if len(v) > 1 else 1.
    t.update(mean_kms=mean, sigma_kms=sigma, arc_kms=arc, n_sigma=abs(mean) / sigma, chi2=chi2, dof=len(v) - 1,
             p_variation=p)
    ok_b = t['n_sigma'] > SKY_SIGNIFICANCE or p < SKY_VARIATION_P
    # (c) floor: the largest correction at the regions against 0.05 resolution elements
    res = spec.resolution or getattr(getattr(cfg, 'input', None), 'resolution', None)
    if not res and np.isfinite(zp.fwhm) and zp.fwhm > 0:
        res = float(np.median(spec.wave) / zp.fwhm)  # from the sky-line width
    t['resolution'] = float(res) if res else float('nan')
    t['floor_kms'] = SKY_FLOOR_RESEL * C_KMS / res if res else 0.
    centres = np.array([0.5 * (r.wmin + r.wmax) / ff for r in regs]) if regs else np.array([])
    t['max_correction_kms'] = float(np.max(np.abs(zp.dv(centres)))) if len(centres) else abs(mean)
    ok_c = t['max_correction_kms'] > t['floor_kms']
    t.update(coverage_ok=bool(ok_a), significant=bool(ok_b), above_floor=bool(ok_c))
    apply = ok_a and ok_b and ok_c
    if apply:
        verdict = 'applying'
    elif not ok_a:
        if zp.n_used < SKY_MIN_SEGMENTS:
            why = f'only {zp.n_used} measured sky segments (< {SKY_MIN_SEGMENTS})'
        elif not regs:
            why = 'no regions to check the coverage against'
        else:
            why = (f'region(s) {", ".join(poor) or "all"} rely on unmeasured (held-flat, ramped or extrapolated) '
                   f'parts of the sky curve for more than {SKY_MAX_UNMEASURED:.0%} of their width')
        verdict = f'not applied: {why}'
    elif not ok_b:
        verdict = (f'not applied: neither the offset (≤ {SKY_SIGNIFICANCE:g}σ) nor the variation '
                   f'(p ≥ {SKY_VARIATION_P:g}) is significant')
    else:
        verdict = f'not applied: the correction is below {SKY_FLOOR_RESEL:g} resolution elements'
    t['verdict'] = verdict
    log.info('%s: sky zero point %+.1f ± %.1f km/s (%.1fσ incl. arc RMS %.1f km/s; segment variation χ² %.1f for %d '
             'dof, p = %.2g; %d segments; regions measured ≥ %.0f per cent; max |Δv| at the regions %.1f km/s, floor '
             '%.1f km/s): %s', name, mean, sigma, t['n_sigma'], arc if np.isfinite(arc) else 0., chi2, len(v) - 1, p,
             zp.n_used, 100. * t['min_measured_fraction'] if fracs else 0., t['max_correction_kms'], t['floor_kms'],
             verdict)
    return apply, t


def _telluric_coverage(tzp: Any, spec: Spectrum, regions: Sequence | None) -> tuple[bool, str]:
    """Whether the telluric curve may be applied at the regions: a node within 250 Å of each, and not O2 A alone"""
    from .calibration import telluric_zeropoint
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


def _read_source(path: str, cfg: Any, what: str, name: str, hdu: int | None = None) -> Spectrum | None:
    """Read a calibration file (sky_source, telluric_source); a failure is logged, not fatal"""
    from . import io
    try:
        return io.read_spectrum(path, medium=cfg.input.medium, medium_method=cfg.input.medium_method, hdu=hdu)
    except Exception as exc:  # noqa: BLE001
        log.warning('%s: could not read calibration.%s %s: %s', name, what, path, exc)
        return None


def _object_tables(spec: Spectrum, cfg: Any) -> list[tuple[int, Spectrum]]:
    """Every object table of a multi-object PypeIt file (hdu, spectrum), the one read first included"""
    from . import io
    n = int(spec.meta.get('n_objects', 1) or 1) if hasattr(spec.meta, 'get') else 1
    if n < 2 or not Path(str(spec.source)).is_file():
        return [(spec.meta.get('hdu', -1) if hasattr(spec.meta, 'get') else -1, spec)]
    from astropy.io import fits
    out = []
    with fits.open(spec.source) as hdul:
        tables = [i for i in range(1, len(hdul)) if getattr(hdul[i], 'columns', None) is not None
                  and ('OPT_WAVE' in hdul[i].columns.names or 'BOX_WAVE' in hdul[i].columns.names)]
    for k in tables:
        if k == spec.meta.get('hdu'):
            out.append((k, spec))
            continue
        try:
            out.append((k, io.read_spectrum(spec.source, medium=cfg.input.medium,
                                            medium_method=cfg.input.medium_method, hdu=k)))
        except Exception as exc:  # noqa: BLE001 - not an object table
            log.debug('object table %d of %s unreadable: %s', k, spec.source, exc)
    return out


def _valid_range(cfg: Any, name: str, *spectra: Spectrum | None) -> tuple[float, float] | None:
    """``calibration.wave_valid_range`` if set, else the arc-line range found next to a PypeIt file (INFO-logged)"""
    cal = getattr(cfg, 'calibration', None)
    given = getattr(cal, 'wave_valid_range', None)
    if given:
        return given
    for s in spectra:
        arc = (getattr(s, 'meta', None) or {}).get('arc_range')
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
    cal = getattr(cfg, 'calibration', None)
    kw = dict(seg_width=getattr(cal, 'sky_segment_width', 250.),
              valid_range=_valid_range(cfg, name, src, fallback))
    tables = [(h, s) for h, s in _object_tables(src, cfg) if s.sky is not None]
    if len(tables) < 2:
        zp = sky_zeropoint.measure(src, **kw)
        if zp is not None:
            zp.source_hdu = src.meta.get('hdu') if hasattr(src.meta, 'get') else None
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


def _matching_object(path: str, raw: Spectrum, cfg: Any, name: str, what: str = 'telluric_source'
                     ) -> Spectrum | None:
    """
    The object of a calibration file whose flux best matches the input (a PypeIt spec1d holds one table per
    object in the slit, and the highest-S/N one is not necessarily the input's target)
    """
    from scipy.ndimage import median_filter
    first = _read_source(path, cfg, what, name)
    if first is None:
        return None
    found = _object_tables(first, cfg)
    if len(found) == 1:
        return first

    def highpass(f: np.ndarray) -> np.ndarray:  # stellar features only: flux calibration and slopes divided out
        g = np.where(np.isfinite(f), f, np.nanmedian(f))
        return g / np.maximum(median_filter(g, size=51, mode='nearest'), 1e-30) - 1.

    ref = highpass(raw.flux)

    from .features import features
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
        corr = float(np.corrcoef(highpass(f)[ok], ref[ok])[0, 1])
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
    from .calibration import telluric_zeropoint
    cal = getattr(cfg, 'calibration', None)
    width = getattr(cal, 'telluric_segment_width', 250.)
    teff = getattr(getattr(cfg, 'templates', None), 'teff_prior', None)
    resolution = getattr(cfg.input, 'resolution', None) or src.resolution
    template, alts = None, []
    if getattr(cal, 'telluric_template', True):
        if resolution is None:  # a first pass without the template measures R from the telluric lines
            first = telluric_zeropoint.measure(src, seg_width=width, teff=teff)
            resolution = first.resolution if first is not None else None
        if resolution is not None:
            from .methods.telluric_ref import neighbour_templates, prior_template
            wr = (float(src.wave[0]), float(src.wave[-1]))
            try:
                template = prior_template(cfg, resolution, wr)
                alts = neighbour_templates(cfg, resolution, wr, template) if template is not None else []
            except Exception as exc:  # noqa: BLE001 - fall back to a smooth stellar continuum
                log.warning('%s: no template for the telluric zero point (%s); assuming a smooth star', name, exc)
    return telluric_zeropoint.measure(src, resolution=resolution, template=template, seg_width=width, teff=teff,
                                      spread_templates=alts)


def wavelength_correction(raw: Spectrum, spec: Spectrum, cfg: Any, name: str = '',
                          regions: Sequence | None = None) -> tuple[Spectrum, dict]:
    """
    Measure the wavelength zero point of one spectrum and apply it if ``cfg.calibration`` asks for it

    - The sky zero point Δv_sky(λ) is measured whenever a sky is available (``raw.sky``, or the sky of
      ``calibration.sky_source``; of a multi-object PypeIt file, the best-measured object's sky), in the
      observatory frame, and always reported.
    - The star's own (telluric) zero point Δv_tell(λ) is measured from the telluric absorption of
      ``calibration.telluric_source`` (a non-telluric-corrected spectrum sharing the input's wavelength solution),
      else of the input itself unless ``input.telluric_corrected``. It is measured when the mode is 'telluric', when
      the sky correction is applied in 'sky-if-needed'/'auto', or whenever a telluric_source is given, and needs the
      telluric model (`rvfitter templates telluric --build`); without it the telluric terms are skipped and flagged.
      With both, the star − sky constant (slit centring) compares each telluric node with the sky measured
      directly in the same window (`telluric_zeropoint.star_minus_sky_direct`: ≥ 2 nodes; Birge-inflated and
      flagged when the nodes disagree, p ≤ 0.01).
    - What is applied to ``spec`` (which may already be barycentric-corrected; the correction is a multiplicative
      factor) follows ``wavelength_correction``: 'sky-if-needed' (default) and its synonym 'auto' the sky curve
      when `_sky_needed` passes, plus the star − sky constant when it is measured ('sky+star'); 'sky' the sky
      curve alone; 'telluric' the telluric curve if it covers the regions; 'none' nothing.
    - A measured offset's value never becomes an error: what is not applied counts only through its uncertainty
      (sys_wavezp) or, for the star's slit-centring constant, through the slit prior (``info['slit']``, the inputs
      of `calibration.slit`; evaluated in `measure_object`).

    Parameters
    ----------
    raw, spec
        The spectrum as read, and as it will be measured (e.g. barycentric-corrected)
    cfg
        The configuration
    name
        Object name for the log
    regions
        The regions the methods will measure (for the sky-if-needed and telluric coverage tests)

    Returns
    -------
    spectrum, info
        The (possibly) corrected spectrum, with meta['sky_zeropoint'], meta['telluric_zeropoint'] and
        meta['star_minus_sky'] whenever measured, and {'mode', 'applied', 'zp_flags', 'sky_tests',
        'star_sky_applied', 'star_sky', 'slit', *ZP_KEYS}
    """
    from .calibration import slit, telluric, telluric_zeropoint
    cal = getattr(cfg, 'calibration', None)
    mode = getattr(cal, 'wavelength_correction', 'sky-if-needed') if cal is not None else 'sky-if-needed'
    nan = float('nan')
    info: dict[str, Any] = {'mode': mode, 'applied': 'none', 'zp_flags': 0, 'sky_tests': {},
                            'star_sky_applied': False, 'star_sky': {}, 'slit': None, **dict.fromkeys(ZP_KEYS, nan)}
    regions = _region_list(regions)
    # --- sky (the shape of the wavelength solution)
    zp = None
    src = raw if raw.sky is not None else None
    sky_source = getattr(cal, 'sky_source', None) if cal is not None else None
    if src is None and sky_source:
        other = _read_source(sky_source, cfg, 'sky_source', name)
        if other is not None and other.sky is None:
            log.warning('%s: calibration.sky_source %s has no sky spectrum', name, sky_source)
        elif other is not None:
            _check_sky_source(raw, other, name, sky_source)
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
        spec = replace(spec, meta={**spec.meta, 'sky_zeropoint': {**zp.as_dict(),
                                                                  'source_hdu': getattr(zp, 'source_hdu', None)}})
    # --- does the sky need correcting? (calibration data and region positions only)
    sky_ok = False
    if zp is not None and mode in ('sky-if-needed', 'auto'):
        sky_ok, info['sky_tests'] = _sky_needed(zp, spec, cfg, regions, name)
        if not sky_ok:
            info['zp_flags'] |= ZP_FLAGS['sky_not_needed' if info['sky_tests']['coverage_ok'] else 'sky_unsafe']
    elif zp is not None and mode == 'sky':
        sky_ok = True
    # --- telluric (the star's own zero point): needed for 'telluric', and for the star - sky constant when the sky
    # correction is applied; always measured when a telluric_source is given
    tzp, tsrc = None, None
    tell_source = getattr(cal, 'telluric_source', None) if cal is not None else None
    wanted = mode == 'telluric' or bool(tell_source) or (mode in ('sky-if-needed', 'auto') and sky_ok)
    if wanted:
        loud = log.warning if (mode == 'telluric' or tell_source) else log.info
        if not telluric.available():
            info['zp_flags'] |= ZP_FLAGS['telluric_unavailable']
            loud('%s: telluric zero point skipped: %s', name, telluric.unavailable_message())
        else:
            if tell_source:
                tsrc = _matching_object(tell_source, raw, cfg, name)
                if tsrc is not None:
                    _check_sky_source(raw, tsrc, name, tell_source)
            elif not getattr(cfg.input, 'telluric_corrected', False):
                tsrc = raw
            else:
                loud('%s: the input is telluric-corrected and no calibration.telluric_source is set; the telluric '
                     'zero point (and so the star - sky constant) is not measured', name)
            if tsrc is not None:
                tzp = _telluric_zeropoint(tsrc, cfg, name)
    if tzp is not None:
        info.update(zp_tell_kms=tzp.zp_kms, zp_tell_err=tzp.zp_err)
        spec = replace(spec, meta={**spec.meta, 'telluric_zeropoint': tzp.as_dict()})
        if zp is not None:
            # the sky measured directly in each telluric node's own window (inside the bands too), never the
            # interpolated sky curve
            sky_src = getattr(zp, 'source_spectrum', None) or src
            windows = sky_zeropoint.measure_windows(sky_src, telluric_zeropoint.node_windows(tzp), zp.fwhm)
            ss = telluric_zeropoint.star_minus_sky_direct(tzp, windows)
            info.update(zp_star_sky=ss['zp_star_sky'], zp_star_sky_err=ss['zp_star_sky_err'],
                        zp_star_sky_n=ss['n_nodes'], star_sky=ss)
            if not np.isfinite(ss['zp_star_sky']):
                info['zp_flags'] |= ZP_FLAGS['star_sky_unmeasured']
            elif not ss['consistent']:
                info['zp_flags'] |= ZP_FLAGS['star_sky_inconsistent']
            spec = replace(spec, meta={**spec.meta, 'star_minus_sky': ss})
    elif mode == 'telluric':
        log.warning('%s: wavelength_correction = "telluric" but the telluric zero point could not be measured; '
                    'nothing applied', name)

    # --- apply
    def use_telluric(spectrum: Spectrum) -> Spectrum:
        ok, why = _telluric_coverage(tzp, spectrum, regions)
        if not ok:
            info['zp_flags'] |= ZP_FLAGS['telluric_refused']
            log.warning('%s: the telluric zero-point curve is not applied: %s', name, why)
            return spectrum
        info['applied'] = 'telluric'
        info['zp_flags'] |= ZP_FLAGS['telluric_applied']
        return sky_zeropoint.apply(spectrum, tzp, source='telluric')

    if sky_ok and mode in ('sky-if-needed', 'auto') and np.isfinite(info['zp_star_sky']):
        spec = sky_zeropoint.apply(spec, zp, constant_kms=info['zp_star_sky'], constant_err=info['zp_star_sky_err'],
                                   source='sky+star')
        info['applied'] = 'sky+star'
        info['star_sky_applied'] = True
        info['zp_flags'] |= ZP_FLAGS['sky_applied'] | ZP_FLAGS['star_sky_applied']
    elif sky_ok:
        spec = sky_zeropoint.apply(spec, zp, source='sky')
        info['applied'] = 'sky'
        info['zp_flags'] |= ZP_FLAGS['sky_applied']
    elif mode == 'telluric' and tzp is not None:
        spec = use_telluric(spec)
    if info['applied'] != 'none':
        wc = spec.meta['wavelength_correction']
        log.info('%s: wavelength correction applied (%s): Δv %+.1f to %+.1f km/s over the spectrum%s', name,
                 info['applied'], wc['dv_min'], wc['dv_max'],
                 f"; the star - sky constant {info['zp_star_sky']:+.1f} ± {info['zp_star_sky_err']:.1f} km/s "
                 f"({info['zp_star_sky_n']} nodes) is included and its error is sys_star_sky"
                 if info['star_sky_applied'] else '')
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
    # --- inputs of the slit-centring prior (always reported when available): the target's own spec1d table first
    # (the telluric match, else the sky_source object matching the input), then the input
    cands = [tsrc, raw]
    if tsrc is None and sky_source and not (raw.meta or {}).get('trace_fwhm_pix'):
        cands.insert(0, _matching_object(sky_source, raw, cfg, name, what='sky_source'))
    info['slit'] = slit.inputs(cands)
    return spec, info


def epoch_labels(spectra: Sequence[Spectrum]) -> list[str]:
    """Epoch labels: the file stems, else the mid-exposure MJD, else e1, e2, ... (always unique)"""
    stems = [Path(s.source).stem or f'e{k + 1}' for k, s in enumerate(spectra)]
    if len(set(stems)) == len(stems):
        return stems
    mjds = [s.meta.get('mjd_mid') for s in spectra]
    if all(m is not None for m in mjds) and len({f'{m:.5f}' for m in mjds}) == len(mjds):
        return [f'mjd{m:.5f}' for m in mjds]
    return [f'e{k + 1}' for k in range(len(spectra))]


def _epoch_summaries(per_method: dict[str, list[MethodResult]], labels: list[str], cfg: Any, name: str,
                     sys: Sequence[float] | None = None) -> dict:
    """
    Per-epoch method and adopted RVs (recorded in each measurement's diagnostics) and a variability test

    The epochs' adopted RVs are compared with their weighted mean: χ² = Σ (v_e − v̄)²/σ_e² with n − 1 degrees of
    freedom, where σ_e is the epoch's statistical error combined with its wavelength zero-point systematic `sys`
    (each epoch has its own wavelength solution and slit centring); p < 0.001 flags the object as variable (a
    possible binary or a calibration problem). The statistical-only test is reported too (`pvalue_stat`).
    """
    from scipy.stats import chi2 as chi2_dist

    from . import combine
    rv_e, err_e, taus_l, taus_m, n_l, n_m = [], [], [], [], [], []
    for k, label in enumerate(labels):
        per = {m: lst[k] for m, lst in per_method.items() if k < len(lst) and np.isfinite(lst[k].rv)}
        try:
            ad = combine.hierarchical_from_methods(per, tau_scale=cfg.combine.tau_scale) if per else {}
        except Exception as exc:  # noqa: BLE001
            log.debug('epoch %s: hierarchical combination failed: %s', label, exc)
            ad = {}
        rv, err = float(ad.get('median', np.nan)), float(ad.get('err', np.nan))
        rv_e.append(rv)
        err_e.append(err)
        taus_l.append(float(ad.get('tau_line', np.nan)))
        taus_m.append(float(ad.get('tau_method', np.nan)))
        n_l.append(int(ad.get('n_lines', 0)))
        n_m.append(int(ad.get('n_methods', 0)))
        for lst in per_method.values():
            if k < len(lst):
                for meas in lst[k].measurements:
                    meas.diagnostics.update(epoch_method_rv=float(lst[k].rv), epoch_method_err=float(lst[k].rv_err),
                                            epoch_adopted_rv=rv, epoch_adopted_err=err)
        log.info('%s: epoch %s: adopted %.2f ± %.2f km/s (%s)', name, label, rv, err,
                 ', '.join(f'{m} {r.rv:.2f} ± {r.rv_err:.2f}' for m, r in per.items()))
    v, e = np.array(rv_e), np.array(err_e)
    s = np.array([x if x is not None and np.isfinite(x) else 0. for x in (sys or [0.] * len(v))], dtype=float)
    ok = np.isfinite(v) & np.isfinite(e) & (e > 0)

    def test(errors: np.ndarray) -> tuple[float, float]:
        if ok.sum() < 2:
            return np.nan, np.nan
        w = 1. / errors[ok] ** 2
        mean = np.sum(w * v[ok]) / np.sum(w)
        c2 = float(np.sum(w * (v[ok] - mean) ** 2))
        return c2, float(chi2_dist.sf(c2, int(ok.sum()) - 1))

    chi2_stat, p_stat = test(e)
    chi2, pvalue = test(np.hypot(e, s))
    variable = bool(np.isfinite(pvalue) and pvalue < 1e-3)
    log.info('%s: epoch variability: χ² %.1f for %d epochs, p = %.3g with the zero-point systematics (statistical '
             'errors only: χ² %.1f, p = %.3g)%s', name, chi2, int(ok.sum()), pvalue, chi2_stat, p_stat,
             ' (variable: p < 0.001)' if variable else '')
    return dict(n=len(labels), labels=list(labels), rv=[float(x) for x in v], err=[float(x) for x in e],
                sys_wavezp=[float(x) for x in s], chi2=chi2, pvalue=pvalue, chi2_stat=chi2_stat, pvalue_stat=p_stat,
                variable=variable, tau_line=taus_l, tau_method=taus_m, n_lines=n_l, n_methods=n_m)


def _check_sky_source(raw: Spectrum, other: Spectrum, name: str, path: str) -> None:
    """Warn if a sky_source file does not look like it shares the input's wavelength solution"""
    lo, hi = max(raw.wmin, other.wmin), min(raw.wmax, other.wmax)
    overlap = max(0., hi - lo) / (raw.wmax - raw.wmin)
    ratio = len(other.wave) / max(len(raw.wave), 1)
    if overlap < 0.9 or not 0.67 < ratio < 1.5:
        log.warning('%s: calibration.sky_source %s covers %.0f per cent of the input range with %d pixels against %d; '
                    'it should share the input\'s wavelength solution (e.g. the spec1d of a coadd)', name, path,
                    100 * overlap, len(other.wave), len(raw.wave))


def _read_all(cfg: Any, inputs: Iterable[str | Path], medium: str) -> list[Spectrum]:
    from . import archives, io

    spectra = []
    for inp in ([inputs] if isinstance(inputs, (str, Path)) else inputs):
        for path in archives.collect_inputs(Path(inp)):
            try:
                spectra.append(io.read_spectrum(path, medium=medium, hdu=cfg.input.hdu,
                                                resolution=cfg.input.resolution))
            except Exception as exc:  # noqa: BLE001
                log.warning('Skipping %s: %s', path, exc)
    return spectra


def run(cfg: Any, inputs: str | Path | Iterable[str | Path] | None = None, *, n_procs: int | None = None,
        medium: str | None = None, methods: Sequence[str] | None = None, bary: str | None = None,
        on_object: Callable[[ObjectResult], None] | None = None, cancel: threading.Event | None = None
        ) -> list[ObjectResult]:
    """
    Run the analysis on all spectra found in ``inputs`` (default ``cfg.input.paths``)

    Objects are processed one after another (``[i/N] name``); within each object a
    ``ProcessPoolExecutor`` (spawn context, safe with Qt threads) is created (none if ``n_procs`` is 1) and
    passed to the methods.
    The keywords ``n_procs``, ``medium``, ``methods`` and ``bary`` override the config.
    ``on_object`` is called with each finished object; if ``cancel`` is set (checked between objects and between
    methods) the results so far are returned and the interrupted object is dropped.
    """
    from . import archives

    problem = method_problem(cfg, methods)
    if problem:
        raise NoMethodsError(problem)
    if inputs is None:
        inputs = cfg.input.paths
    if n_procs is None:
        n_procs = cfg.parallel.n_procs
    spectra = _read_all(cfg, inputs, medium or cfg.input.medium)
    groups = archives.group_by_object(spectra, cfg.grouping.override)
    names = sorted(groups)
    log.info('Read %d spectra of %d objects', len(spectra), len(names))
    out: list[ObjectResult] = []
    for i, name in enumerate(names, start=1):
        if cancel is not None and cancel.is_set():
            log.warning('Cancelled after %d of %d objects', len(out), len(names))
            break
        log.info('[%d/%d] %s', i, len(names), name)
        obj_cfg = config_for_object(cfg, name)
        rng = np.random.default_rng(object_seed(cfg.seed, name))
        try:
            if n_procs and n_procs > 1:
                with ProcessPoolExecutor(n_procs, mp_context=multiprocessing.get_context('spawn')) as ex:
                    res = measure_object(groups[name], obj_cfg, ex, rng, name=name, methods=methods, bary=bary,
                                         cancel=cancel)
            else:
                res = measure_object(groups[name], obj_cfg, None, rng, name=name, methods=methods, bary=bary,
                                     cancel=cancel)
        except _Cancelled:
            log.warning('Cancelled during %s', name)
            break
        res.refitted = obj_cfg is not cfg  # `config_for_object` returns cfg itself when no patch applies
        out.append(res)
        if on_object is not None:
            on_object(res)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# refitting one object with an updated configuration (e.g. a wider parameter range or an extra grid)
# ----------------------------------------------------------------------------------------------------------------------
def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in patch.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def apply_patch(cfg: Any, cfg_patch: dict) -> Any:
    """
    A new, validated Config: ``cfg`` with ``cfg_patch`` deep-merged into it (``cfg`` is not modified)

    Parameters
    ----------
    cfg
        A `config.Config`
    cfg_patch
        Nested mapping of the sections and keys to change, e.g. ``{'templates': {'met_range': [-1.0, 0.0]}}``
    """
    from . import config
    return config.from_dict(_deep_merge(cfg.to_dict(), _plain(cfg_patch)))


def object_seed(seed: int, name: str) -> np.random.SeedSequence:
    """
    The seed sequence of one object: from the run seed and a stable hash of the object name

    It does not depend on the other objects in the input or their order, so a run on any set of inputs and
    `refit_object` give the same random numbers for the same object.
    """
    import hashlib
    h = int.from_bytes(hashlib.blake2b(str(name).encode('utf-8'), digest_size=8).digest(), 'little')
    return np.random.SeedSequence([int(seed), h])


def config_for_object(cfg: Any, name: str) -> Any:
    """
    The configuration for one object: ``cfg`` with the patches of its ``[[refits]]`` entries applied, in order

    Entries recorded for other objects are ignored, so a refit changes only its own object when the written TOML
    is run again.
    """
    patches = [r.get('patch', {}) for r in (getattr(cfg, 'refits', None) or []) if r.get('object') == name]
    out = cfg
    for patch in patches:
        out = apply_patch(out, patch)
    return out


def suggested_refits(result: ObjectResult) -> list[dict]:
    """
    Actions suggested by the template-edge guidance of an object (for the CLI log or a GUI button)

    Returns
    -------
    actions
        One dict per distinct (axis, direction, action): 'method', 'axis', 'direction', 'action' ('widen' |
        'download' | 'none'), 'message', 'cfg_patch' (for `refit_object`), 'download' (keyword arguments for
        `templates.download`, or None) and 'command' (the equivalent CLI command, or '')
    """
    seen, out = set(), []
    for name, mres in result.methods.items():
        for meas in mres.measurements:
            summ = (meas.diagnostics or {}).get('template_summary') or {}
            for e in summ.get('edges', []):
                key = (e['axis'], e['direction'], e['action'])
                if key in seen:
                    continue
                seen.add(key)
                out.append(dict(method=name, axis=e['axis'], direction=e['direction'], action=e['action'],
                                message=e['message'], cfg_patch=e['cfg_patch'], download=e['download'],
                                command=e['command']))
    return out


def refit_object(result: ObjectResult, cfg: Any, cfg_patch: dict, *, inputs: Sequence[str | Path] | None = None,
                 n_procs: int | None = 1, medium: str | None = None, methods: Sequence[str] | None = None,
                 bary: str | None = None) -> tuple[ObjectResult, Any]:
    """
    Rerun one object with an updated configuration, recording the change

    Nothing changes mid-run. The patch (e.g. a suggestion from `suggested_refits`, applied after downloading
    any grid it needs) is recorded as ``{'object': name, 'patch': ...}`` in ``new_cfg.refits`` (and so in any TOML
    written from it); the global values are unchanged, and `run` applies the patch to that object only
    (`config_for_object`). The object is measured again from its files with the patched configuration.

    Parameters
    ----------
    result
        The object's previous result (its ``files`` are re-read unless ``inputs`` is given)
    cfg
        The Config used for ``result`` (not modified)
    cfg_patch
        Nested mapping merged into the configuration, e.g. ``{'templates': {'met_range': [-1.0, 0.0]}}``
    inputs, n_procs, medium, methods, bary
        As for `run`

    Returns
    -------
    result, new_cfg
        The new ObjectResult, and the Config with the refit recorded. The object's seed is `object_seed`, so
        running the written TOML on the same inputs (any set containing this object) reproduces it exactly
    """
    new_cfg = apply_patch(cfg, {})
    new_cfg.refits.append({'object': result.object, 'patch': _plain(cfg_patch)})
    obj_cfg = config_for_object(new_cfg, result.object)
    obj_cfg.validate()
    log.info('Refitting %s with %s', result.object, _plain(cfg_patch))
    spectra = _read_all(obj_cfg, list(inputs or result.files), medium or obj_cfg.input.medium)
    if not spectra:
        raise FileNotFoundError(f'none of the files of {result.object} could be read: {result.files}')
    rng = np.random.default_rng(object_seed(new_cfg.seed, result.object))
    if n_procs and n_procs > 1:
        with ProcessPoolExecutor(n_procs, mp_context=multiprocessing.get_context('spawn')) as ex:
            res = measure_object(spectra, obj_cfg, ex, rng, name=result.object, methods=methods, bary=bary)
    else:
        res = measure_object(spectra, obj_cfg, None, rng, name=result.object, methods=methods, bary=bary)
    res.refitted = True
    return res, new_cfg
