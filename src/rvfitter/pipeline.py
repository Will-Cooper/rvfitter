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

__all__ = ['run', 'measure_object', 'enabled_methods', 'NoMethodsError', 'method_problem',
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

    Runs barycentric.correction, regions.select, each enabled method, combine.hierarchical and systematics.
    A method that raises is logged and recorded as a failed measurement (flag 1); the object carries on.
    If ``cancel`` is set between methods, ``_Cancelled`` is raised (handled by ``run``).
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

    # wavelength zero point (sky shape and/or the star's telluric zero point), applied once to the wavelengths
    cal_infos = []
    for k, (raw, spec) in enumerate(zip(spectra, corrected, strict=True)):
        corrected[k], info = wavelength_correction(raw, spec, cfg, name)
        cal_infos.append(info)
    bary_info['wavelength_correction'] = cal_infos[0]['applied']

    # None means "use the SelectOptions default"
    knobs = {'snr_min': cfg.regions.min_snr, 'telluric': cfg.regions.telluric_strictness,
             'telluric_corrected': True if cfg.input.telluric_corrected else None,
             'template_windows': getattr(cfg.regions, 'template_windows', None)}
    opts = regions.SelectOptions(**{k: v for k, v in knobs.items() if v is not None})
    method_rngs = dict(zip(names, rng.spawn(len(names)), strict=True))
    per_method: dict[str, list[MethodResult]] = {m: [] for m in names}
    sel0: dict[str, list] = {}
    for spec in corrected:
        sel = regions.select(spec, teff=cfg.templates.teff_prior, methods=names, overrides=cfg.regions.overrides,
                             options=opts)
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

    # first-order uncorrected RVs (the correction lowered each RV by the Δv applied at its wavelength)
    for lst in per_method.values():
        for k, res in enumerate(lst):
            spec = corrected[min(k, len(corrected) - 1)]
            for meas in res.measurements:
                lam = _measurement_wave(meas)
                dv = float(sky_zeropoint.applied_dv(spec.meta, lam)) if np.isfinite(lam) else np.nan
                meas.diagnostics['zp_applied_kms'] = dv
                meas.diagnostics['rv_uncorrected'] = float(meas.rv + dv) if np.isfinite(dv) else np.nan

    # several epochs: each was measured separately; its regions become '<epoch>:<region>', so epoch-to-epoch
    # disagreement enters the random-effects scatter (method level) and the hierarchical τ_line (adopted), instead
    # of being averaged away as if the epochs were the same region
    labels = epoch_labels(corrected)
    epoch_info: dict[str, Any] = {}
    if len(corrected) > 1:
        epoch_info = _epoch_summaries(per_method, labels, cfg, name)
        for lst in per_method.values():
            for k, res in enumerate(lst):
                for meas in res.measurements:
                    meas.diagnostics['epoch'] = labels[k]
                    meas.region = f'{labels[k]}:{meas.region}'

    results: dict[str, MethodResult] = {}
    for m, lst in per_method.items():
        if len(lst) == 1:
            res = lst[0]
        else:
            meas = [x for r in lst for x in r.measurements]
            try:
                from .methods._common import finish
                res = finish(m, meas, outlier_k=None)  # random effects over every epoch:region, flags respected
            except Exception as exc:  # noqa: BLE001
                log.error('%s: combining %s over epochs failed: %s', name, m, exc)
                res = MethodResult(m, float('nan'), float('nan'), 0, meas)
        results[m] = res
        if np.isfinite(res.rv):
            log.info('%s: %.1f ± %.1f km/s', m, res.rv, res.rv_err)
        else:
            log.info('%s: no result', m)

    adopted: dict[str, Any] = {}
    good = [r for r in results.values() if np.isfinite(r.rv)]
    if good:
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

    sysd: dict[str, float] = {k: cal_infos[0][k] for k in ZP_KEYS}
    try:
        sysd['wavelength_zero_point'] = float(systematics.wavelength_zero_point(corrected[0], cfg))
    except Exception as exc:  # noqa: BLE001
        log.debug('wavelength zero-point unavailable: %s', exc)
    try:
        sysd['template_spread'] = float(systematics.template_spread(corrected[0], sel0, cfg, executor=executor,
                                                                    rng=rng.spawn(1)[0]))
    except Exception as exc:  # noqa: BLE001
        log.debug('template spread unavailable: %s', exc)

    sysd['floor'] = float(cfg.combine.systematic_floor)
    return ObjectResult(object=name, files=[s.source for s in spectra], methods=results, adopted=adopted,
                        systematics=sysd, barycentric=bary_info, binary={}, config_hash=_config_hash(cfg),
                        version=__version__)


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
           'zp_star_sky', 'zp_star_sky_err')


def _read_source(path: str, cfg: Any, what: str, name: str) -> Spectrum | None:
    """Read a calibration file (sky_source, telluric_source); a failure is logged, not fatal"""
    from . import io
    try:
        return io.read_spectrum(path, medium=cfg.input.medium, medium_method=cfg.input.medium_method)
    except Exception as exc:  # noqa: BLE001
        log.warning('%s: could not read calibration.%s %s: %s', name, what, path, exc)
        return None


def _matching_object(path: str, raw: Spectrum, cfg: Any, name: str) -> Spectrum | None:
    """
    The object of a calibration file whose flux best matches the input (a PypeIt spec1d holds one table per
    object in the slit, and the first is not necessarily the target)
    """
    from astropy.io import fits
    from scipy.ndimage import median_filter

    from . import io
    first = _read_source(path, cfg, 'telluric_source', name)
    if first is None:
        return None
    try:
        with fits.open(path) as hdul:
            n = len(hdul)
    except (OSError, ValueError):
        return first
    found = [(1, first)]
    for k in range(2, n):
        try:
            found.append((k, io.read_spectrum(path, medium=cfg.input.medium, medium_method=cfg.input.medium_method,
                                              hdu=k)))
        except Exception:  # noqa: BLE001 - not an object table
            continue
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
    log.info('%s: calibration.telluric_source %s: object HDU %d matches the input best (score %.3f: high-pass '
             'correlation − |ln flux ratio|; others %s)', name, path, found[k][0], scores[k],
             ', '.join(f'{h}: {c:.3f}' for (h, _), c in zip(found, scores, strict=True) if h != found[k][0]))
    return found[k][1]


def _telluric_zeropoint(src: Spectrum, cfg: Any, name: str):
    """The star's telluric zero point; the star is modelled with the template nearest the priors when available"""
    from .calibration import telluric_zeropoint
    cal = getattr(cfg, 'calibration', None)
    width = getattr(cal, 'telluric_segment_width', 250.)
    resolution = getattr(cfg.input, 'resolution', None) or src.resolution
    template = None
    if getattr(cal, 'telluric_template', True):
        if resolution is None:  # a first pass without the template measures R from the telluric lines
            first = telluric_zeropoint.measure(src, seg_width=width)
            resolution = first.resolution if first is not None else None
        if resolution is not None:
            from .methods.telluric_ref import prior_template
            try:
                template = prior_template(cfg, resolution, (float(src.wave[0]), float(src.wave[-1])))
            except Exception as exc:  # noqa: BLE001 - fall back to a smooth stellar continuum
                log.warning('%s: no template for the telluric zero point (%s); assuming a smooth star', name, exc)
    return telluric_zeropoint.measure(src, resolution=resolution, template=template, seg_width=width)


def wavelength_correction(raw: Spectrum, spec: Spectrum, cfg: Any, name: str = '') -> tuple[Spectrum, dict]:
    """
    Measure the wavelength zero point of one spectrum and apply it if ``cfg.calibration`` asks for it

    - The sky zero point Δv_sky(λ) is measured whenever a sky is available (``raw.sky``, or the sky of
      ``calibration.sky_source``), in the observatory frame, and always reported.
    - The star's own (telluric) zero point Δv_tell(λ) is measured from the telluric absorption of
      ``calibration.telluric_source`` (a non-telluric-corrected spectrum sharing the input's wavelength solution),
      else of the input itself unless ``input.telluric_corrected``. It is measured when the mode is 'telluric' or
      'auto', or whenever a telluric_source is given. With both curves, the star − sky constant (slit centring) is
      the weighted mean of Δv_tell − Δv_sky at the telluric segments (``zp_star_sky``).
    - What is applied to ``spec`` (which may already be barycentric-corrected; the correction is a multiplicative
      factor) follows ``wavelength_correction``: 'none' (default) nothing; 'sky' the sky curve; 'telluric' the
      telluric curve; 'auto' the sky shape plus the star − sky constant when both exist, else whichever exists.
      When nothing is applied, the measured zero point enters the zero-point systematic.

    Returns
    -------
    spectrum, info
        The (possibly) corrected spectrum, with meta['sky_zeropoint'], meta['telluric_zeropoint'] and
        meta['star_minus_sky'] whenever measured, and {'mode', 'applied', *ZP_KEYS}
    """
    from .calibration import telluric_zeropoint
    cal = getattr(cfg, 'calibration', None)
    mode = getattr(cal, 'wavelength_correction', 'none') if cal is not None else 'none'
    nan = float('nan')
    info: dict[str, Any] = {'mode': mode, 'applied': 'none', **dict.fromkeys(ZP_KEYS, nan)}
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
        zp = sky_zeropoint.measure(src, seg_width=getattr(cal, 'sky_segment_width', 250.))
    elif mode == 'sky':
        log.warning('%s: wavelength_correction = "sky" but no sky spectrum is available; nothing applied', name)
    if zp is not None:
        info.update(zp_sky_kms=zp.zp_kms, zp_sky_err=zp.zp_err, zp_sky_slope=zp.slope,
                    zp_sky_model_spread=zp.model_spread)
        spec = replace(spec, meta={**spec.meta, 'sky_zeropoint': zp.as_dict()})
        if abs(zp.zp_kms) > 2. * zp.zp_err:
            log.info('%s: the OH sky lines suggest the wavelength solution is offset by %+.1f ± %.1f km/s (%.0f-%.0f '
                     'Å); check the wavelength calibration', name, zp.zp_kms, zp.zp_err, *zp.main_span)
    # --- telluric (the star's own zero point)
    tzp = None
    tell_source = getattr(cal, 'telluric_source', None) if cal is not None else None
    if mode in ('auto', 'telluric') or tell_source:
        tsrc = None
        if tell_source:
            tsrc = _matching_object(tell_source, raw, cfg, name)
            if tsrc is not None:
                _check_sky_source(raw, tsrc, name, tell_source)
        elif not getattr(cfg.input, 'telluric_corrected', False):
            tsrc = raw
        else:
            log.warning('%s: the input is telluric-corrected and no calibration.telluric_source is set; the telluric '
                        'zero point is not measured', name)
        if tsrc is not None:
            tzp = _telluric_zeropoint(tsrc, cfg, name)
    if tzp is not None:
        info.update(zp_tell_kms=tzp.zp_kms, zp_tell_err=tzp.zp_err)
        spec = replace(spec, meta={**spec.meta, 'telluric_zeropoint': tzp.as_dict()})
        if zp is not None:
            ss = telluric_zeropoint.star_minus_sky(tzp, zp)
            info.update(zp_star_sky=ss['zp_star_sky'], zp_star_sky_err=ss['zp_star_sky_err'])
            spec = replace(spec, meta={**spec.meta, 'star_minus_sky': ss})
    elif mode == 'telluric':
        log.warning('%s: wavelength_correction = "telluric" but the telluric zero point could not be measured; '
                    'nothing applied', name)
    # --- apply
    if mode == 'sky' and zp is not None:
        spec, info['applied'] = sky_zeropoint.apply(spec, zp, source='sky'), 'sky'
    elif mode == 'telluric' and tzp is not None:
        spec, info['applied'] = sky_zeropoint.apply(spec, tzp, source='telluric'), 'telluric'
    elif mode == 'auto':
        if zp is not None and tzp is not None and np.isfinite(info['zp_star_sky']):
            spec = sky_zeropoint.apply(spec, zp, constant_kms=info['zp_star_sky'],
                                       constant_err=info['zp_star_sky_err'], source='sky+telluric')
            info['applied'] = 'sky+telluric'
        elif zp is not None:
            spec, info['applied'] = sky_zeropoint.apply(spec, zp, source='sky'), 'sky'
        elif tzp is not None:
            spec, info['applied'] = sky_zeropoint.apply(spec, tzp, source='telluric'), 'telluric'
    if info['applied'] != 'none':
        wc = spec.meta['wavelength_correction']
        log.info('%s: wavelength correction applied (%s): Δv %+.1f to %+.1f km/s over the spectrum', name,
                 info['applied'], wc['dv_min'], wc['dv_max'])
        if info['applied'] == 'sky':
            log.info('%s: the star-sky offset (slit centring) is not measured (no telluric zero point) and is not '
                     'included in the errors', name)
    else:
        log.info('%s: no wavelength correction applied (mode %s)%s', name, mode,
                 '; the measured zero point is counted in the zero-point systematic' if zp or tzp else '')
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


def _epoch_summaries(per_method: dict[str, list[MethodResult]], labels: list[str], cfg: Any, name: str) -> dict:
    """
    Per-epoch method and adopted RVs (recorded in each measurement's diagnostics) and a variability test

    The epochs' adopted RVs are compared with their weighted mean: χ² = Σ (v_e − v̄)²/σ_e² with n − 1 degrees of
    freedom. p < 0.001 flags the object as variable (a possible binary or a calibration problem).
    """
    from scipy.stats import chi2 as chi2_dist

    from . import combine
    rv_e, err_e = [], []
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
        for lst in per_method.values():
            if k < len(lst):
                for meas in lst[k].measurements:
                    meas.diagnostics.update(epoch_method_rv=float(lst[k].rv), epoch_method_err=float(lst[k].rv_err),
                                            epoch_adopted_rv=rv, epoch_adopted_err=err)
        log.info('%s: epoch %s: adopted %.2f ± %.2f km/s (%s)', name, label, rv, err,
                 ', '.join(f'{m} {r.rv:.2f} ± {r.rv_err:.2f}' for m, r in per.items()))
    v, e = np.array(rv_e), np.array(err_e)
    ok = np.isfinite(v) & np.isfinite(e) & (e > 0)
    pvalue, chi2 = np.nan, np.nan
    if ok.sum() > 1:
        w = 1. / e[ok] ** 2
        mean = np.sum(w * v[ok]) / np.sum(w)
        chi2 = float(np.sum(w * (v[ok] - mean) ** 2))
        pvalue = float(chi2_dist.sf(chi2, int(ok.sum()) - 1))
    variable = bool(np.isfinite(pvalue) and pvalue < 1e-3)
    log.info('%s: epoch variability: χ² %.1f for %d epochs, p = %.3g%s', name, chi2, int(ok.sum()), pvalue,
             ' (variable: p < 0.001)' if variable else '')
    return dict(n=len(labels), labels=list(labels), rv=[float(x) for x in v], err=[float(x) for x in e], chi2=chi2,
                pvalue=pvalue, variable=variable)


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
    return res, new_cfg
