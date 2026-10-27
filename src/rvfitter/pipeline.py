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
from .types import Measurement, MethodResult, ObjectResult, Spectrum

__all__ = ['run', 'measure_object', 'enabled_methods', 'NoMethodsError', 'method_problem']

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

    # None means "use the SelectOptions default"
    knobs = {'snr_min': cfg.regions.min_snr, 'telluric': cfg.regions.telluric_strictness,
             'telluric_corrected': True if cfg.input.telluric_corrected else None}
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
            vals, errs, lines, meths = [], [], [], []
            for r in good:
                for x in r.measurements:
                    if np.isfinite(x.rv):
                        vals.append(x.rv)
                        errs.append(x.rv_err)
                        lines.append(x.region)
                        meths.append(r.method)
            adopted = combine.hierarchical(np.array(vals), np.array(errs), lines, meths,
                                           tau_scale=cfg.combine.tau_scale)
        except Exception as exc:  # noqa: BLE001
            log.error('%s: hierarchical combination failed: %s', name, exc)
            log.debug('traceback', exc_info=True)
    if adopted:
        log.info('%s: adopted RV %.2f ± %.2f km/s', name, adopted.get('median', float('nan')),
                 adopted.get('err', float('nan')))
    else:
        log.warning('%s: no adopted RV', name)

    sysd: dict[str, float] = {}
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
