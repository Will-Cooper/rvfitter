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
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .calibration import sky_zeropoint
from .types import Measurement, MethodResult, ObjectResult, Spectrum

__all__ = ['run', 'measure_object', 'enabled_methods', 'NoMethodsError', 'method_problem',
           'refit_object', 'suggested_refits', 'apply_patch']

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
        corrected.append(barycentric.apply(spec, kms) if source != 'skipped' and kms else spec)
        kmss.append(float(kms))
        sources.append(source)
    # several epochs: report the mean correction applied, and the source of the first
    bary_info = {'applied_kms': float(np.mean(kmss)), 'source': sources[0], 'frame': frame}

    # wavelength zero point (sky shape, later + telluric constant), applied once to the wavelengths
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

    results: dict[str, MethodResult] = {}
    for m, lst in per_method.items():
        if len(lst) == 1:
            res = lst[0]
        else:
            meas = [x for r in lst for x in r.measurements]
            try:
                rv, err = combine.combine_regions(meas)
            except Exception as exc:  # noqa: BLE001
                log.error('%s: combining %s over epochs failed: %s', name, m, exc)
                rv, err = float('nan'), float('nan')
            res = MethodResult(m, float(rv), float(err), sum(r.n_regions for r in lst), meas)
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
    if adopted:
        log.info('%s: adopted RV %.2f ± %.2f km/s', name, adopted.get('median', float('nan')),
                 adopted.get('err', float('nan')))
    else:
        log.warning('%s: no adopted RV', name)

    sysd: dict[str, float] = {k: cal_infos[0][k] for k in ('zp_sky_kms', 'zp_sky_err', 'zp_sky_slope')}
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


def wavelength_correction(raw: Spectrum, spec: Spectrum, cfg: Any, name: str = '') -> tuple[Spectrum, dict]:
    """
    Measure and apply the wavelength zero point to one spectrum, according to ``cfg.calibration``

    The sky zero point Δv(λ) is measured on the input spectrum (``raw``: its own sky, or the sky of
    ``calibration.sky_source``) in the observatory frame and applied to ``spec`` (which may already be
    barycentric-corrected; the correction is a multiplicative factor). 'auto' applies whatever is available
    (sky shape; the telluric constant arrives with calibration.telluric_zeropoint); 'none' applies nothing.

    Returns
    -------
    spectrum, info
        The (possibly) corrected spectrum and {'mode', 'applied', 'zp_sky_kms', 'zp_sky_err', 'zp_sky_slope'}
    """
    from . import io
    from .calibration import sky_zeropoint
    cal = getattr(cfg, 'calibration', None)
    mode = getattr(cal, 'wavelength_correction', 'none') if cal is not None else 'none'
    nan = float('nan')
    info = {'mode': mode, 'applied': 'none', 'zp_sky_kms': nan, 'zp_sky_err': nan, 'zp_sky_slope': nan}
    if mode == 'none':
        return spec, info
    zp = None
    if mode in ('auto', 'sky'):
        src = raw if raw.sky is not None else None
        sky_source = getattr(cal, 'sky_source', None)
        if src is None and sky_source:
            try:
                other = io.read_spectrum(sky_source, medium=cfg.input.medium, medium_method=cfg.input.medium_method)
                src = other if other.sky is not None else None
                if src is None:
                    log.warning('%s: calibration.sky_source %s has no sky spectrum', name, sky_source)
            except Exception as exc:  # noqa: BLE001 - a missing sky file is reported, not fatal
                log.warning('%s: could not read calibration.sky_source %s: %s', name, sky_source, exc)
        if src is not None:
            zp = sky_zeropoint.measure(src, seg_width=getattr(cal, 'sky_segment_width', 250.),
                                       degree=getattr(cal, 'sky_degree', None))
        elif mode == 'sky':
            log.warning('%s: wavelength_correction = "sky" but no sky spectrum is available; nothing applied', name)
    if mode == 'telluric':
        log.warning('%s: the telluric zero point is not implemented yet; nothing applied', name)
    if zp is not None:
        info.update(zp_sky_kms=zp.zp_kms, zp_sky_err=zp.zp_err, zp_sky_slope=zp.slope)
        spec = sky_zeropoint.apply(spec, zp, source='sky')
        info['applied'] = 'sky'
        wc = spec.meta['wavelength_correction']
        log.info('%s: wavelength correction applied (sky): Δv %+.1f to %+.1f km/s over the spectrum', name,
                 wc['dv_min'], wc['dv_max'])
    else:
        log.info('%s: no wavelength correction applied (mode %s)', name, mode)
    return spec, info


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
    seeds = np.random.SeedSequence(cfg.seed).spawn(len(names))
    out: list[ObjectResult] = []
    for i, (name, seed) in enumerate(zip(names, seeds, strict=True), start=1):
        if cancel is not None and cancel.is_set():
            log.warning('Cancelled after %d of %d objects', len(out), len(names))
            break
        log.info('[%d/%d] %s', i, len(names), name)
        rng = np.random.default_rng(seed)
        try:
            if n_procs and n_procs > 1:
                with ProcessPoolExecutor(n_procs, mp_context=multiprocessing.get_context('spawn')) as ex:
                    res = measure_object(groups[name], cfg, ex, rng, name=name, methods=methods, bary=bary,
                                         cancel=cancel)
            else:
                res = measure_object(groups[name], cfg, None, rng, name=name, methods=methods, bary=bary,
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

    Nothing changes mid-run: the patch (e.g. a suggestion from `suggested_refits`, applied after downloading
    any grid it needs) gives a new Config, which is validated, recorded in ``new_cfg.refits`` and so in any TOML
    written from it, and the object is measured again from its files.

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
        The new ObjectResult and the Config that produced it. The random seed is
        ``SeedSequence(new_cfg.seed).spawn(1)[0]``, i.e. that of a run on this object alone, so ``rvfitter run``
        with the written TOML on the same files reproduces it exactly
    """
    new_cfg = apply_patch(cfg, cfg_patch)
    new_cfg.refits.append({'object': result.object, 'patch': _plain(cfg_patch)})
    log.info('Refitting %s with %s', result.object, _plain(cfg_patch))
    spectra = _read_all(new_cfg, list(inputs or result.files), medium or new_cfg.input.medium)
    if not spectra:
        raise FileNotFoundError(f'none of the files of {result.object} could be read: {result.files}')
    rng = np.random.default_rng(np.random.SeedSequence(new_cfg.seed).spawn(1)[0])
    if n_procs and n_procs > 1:
        with ProcessPoolExecutor(n_procs, mp_context=multiprocessing.get_context('spawn')) as ex:
            res = measure_object(spectra, new_cfg, ex, rng, name=result.object, methods=methods, bary=bary)
    else:
        res = measure_object(spectra, new_cfg, None, rng, name=result.object, methods=methods, bary=bary)
    return res, new_cfg
