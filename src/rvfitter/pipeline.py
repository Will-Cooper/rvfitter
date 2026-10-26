"""Orchestration of an analysis: objects, methods, combination and output.

Objects are processed sequentially; parallelism is used inside an object (a process pool passed
down to the methods), so results do not depend on ``n_procs``.

The configuration is accessed defensively (``getattr`` with defaults) so that the skeleton works with whatever
``config.Config`` provides; see ``docs/dev_notes_cli_results.md`` for the attributes used.
"""
import importlib
import logging
from collections.abc import Iterable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .types import Measurement, MethodResult, ObjectResult, Spectrum

__all__ = ['run', 'measure_object', 'enabled_methods']

log = logging.getLogger(__name__)

DEFAULT_METHODS = ('linecentre', 'ccf', 'template_fit')
FLAG_FIT_FAILED = 1


def enabled_methods(cfg: Any, override: Sequence[str] | None = None) -> list[str]:
    """Names of the methods to run, from ``override`` or ``cfg.methods`` (list of names or name -> options)"""
    if override:
        return list(override)
    methods = getattr(cfg, 'methods', None)
    if methods is None:
        return list(DEFAULT_METHODS)
    if isinstance(methods, dict):
        return [k for k, v in methods.items()
                if (v.get('enabled', True) if isinstance(v, dict) else getattr(v, 'enabled', True))]
    if isinstance(methods, str):
        return [m for m in methods.split(',') if m]
    return [getattr(m, 'name', m) for m in methods if getattr(m, 'enabled', True)]


def _registry() -> dict[str, Any]:
    return importlib.import_module('rvfitter.methods').REGISTRY


def _config_hash(cfg: Any) -> str:
    try:
        return str(cfg.hash())
    except Exception:  # noqa: BLE001
        return ''


def _failed(method: str, region: str = '') -> MethodResult:
    meas = Measurement(method, region, float('nan'), float('nan'), FLAG_FIT_FAILED)
    return MethodResult(method, float('nan'), float('nan'), 0, [meas])


def measure_object(spectra: Sequence[Spectrum], cfg: Any, executor: Executor | None, rng: np.random.Generator, *,
                   name: str | None = None, methods: Sequence[str] | None = None,
                   bary: str | None = None) -> ObjectResult:
    """
    Measure one object (one or more spectra)

    Runs barycentric.correction, regions.select, each enabled method, combine.hierarchical and systematics.
    A method that raises is logged and recorded as a failed measurement (flag 1); the object carries on.
    """
    from . import barycentric, combine, regions, systematics

    names = enabled_methods(cfg, methods)
    registry = _registry()
    name = name or str(spectra[0].meta.get('object', Path(spectra[0].source).stem))
    mode = bary or getattr(cfg, 'bary', None) or getattr(getattr(cfg, 'barycentric', None), 'mode', None) or 'auto'
    frame = getattr(getattr(cfg, 'barycentric', None), 'frame', 'barycentric')

    corrected, bary_info = [], {'applied_kms': 0.0, 'source': 'skipped', 'frame': frame}
    for spec in spectra:
        kms, source = barycentric.correction(spec, mode=mode, frame=frame)
        log.info('%s: barycentric correction %.3f km/s (%s)', name, kms, source)
        corrected.append(barycentric.apply(spec, kms) if source != 'skipped' and kms else spec)
        bary_info = {'applied_kms': float(kms), 'source': source, 'frame': frame}

    method_rngs = dict(zip(names, rng.spawn(len(names)), strict=True))
    per_method: dict[str, list[MethodResult]] = {m: [] for m in names}
    for spec in corrected:
        sel = regions.select(spec, teff=getattr(cfg, 'teff', None), methods=names)
        for m in names:
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
            adopted = combine.hierarchical(np.array(vals), np.array(errs), lines, meths)
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
        sysd['wavelength_zero_point'] = float(systematics.wavelength_zero_point(corrected[0]))
    except Exception as exc:  # noqa: BLE001
        log.debug('wavelength zero-point unavailable: %s', exc)
    try:
        sysd['template_spread'] = float(systematics.template_spread(results))
    except Exception as exc:  # noqa: BLE001
        log.debug('template spread unavailable: %s', exc)

    return ObjectResult(object=name, files=[s.source for s in spectra], methods=results, adopted=adopted,
                        systematics=sysd, barycentric=bary_info, binary={}, config_hash=_config_hash(cfg),
                        version=__version__)


def _read_all(cfg: Any, inputs: Iterable[str | Path], medium: str) -> list[Spectrum]:
    from . import archives, io

    spectra = []
    for inp in ([inputs] if isinstance(inputs, (str, Path)) else inputs):
        for path in archives.collect_inputs(Path(inp)):
            try:
                spectra.append(io.read_spectrum(path, medium=medium, resolution=getattr(cfg, 'resolution', None)))
            except Exception as exc:  # noqa: BLE001
                log.warning('Skipping %s: %s', path, exc)
    return spectra


def run(cfg: Any, inputs: str | Path | Iterable[str | Path], *, n_procs: int = 1, medium: str = 'auto',
        methods: Sequence[str] | None = None, bary: str | None = None) -> list[ObjectResult]:
    """
    Run the analysis on all spectra found in ``inputs``

    Objects are processed one after another (``[i/N] name``); within each object a
    ``ProcessPoolExecutor(n_procs)`` is created (none if ``n_procs`` is 1) and passed to the methods.
    """
    from . import archives

    spectra = _read_all(cfg, inputs, medium)
    groups = archives.group_by_object(spectra, getattr(cfg, 'objects', None))
    names = sorted(groups)
    log.info('Read %d spectra of %d objects', len(spectra), len(names))
    seeds = np.random.SeedSequence(getattr(cfg, 'seed', 0)).spawn(len(names))
    out = []
    for i, (name, seed) in enumerate(zip(names, seeds, strict=True), start=1):
        log.info('[%d/%d] %s', i, len(names), name)
        rng = np.random.default_rng(seed)
        if n_procs and n_procs > 1:
            with ProcessPoolExecutor(n_procs) as ex:
                out.append(measure_object(groups[name], cfg, ex, rng, name=name, methods=methods, bary=bary))
        else:
            out.append(measure_object(groups[name], cfg, None, rng, name=name, methods=methods, bary=bary))
    return out
