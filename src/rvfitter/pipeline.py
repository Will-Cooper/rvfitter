"""
Orchestration of an analysis: objects, methods, combination and output.

Objects are processed sequentially; parallelism is used inside an object (a process pool passed down to the
methods), so results do not depend on ``n_procs``.
"""
import logging
import multiprocessing
import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__, archives, barycentric, combine, regions, systematics
from .calibration import sky_zeropoint
from .calibration import wavelength_correction as wcorr
from .methods import REGISTRY, binary, forward
from .refit import apply_patch, config_for_object, object_seed, plain
from .results import EXCLUDE_FLAGS, FLAGS
from .types import Measurement, MethodResult, ObjectResult, Spectrum

__all__ = ['run', 'measure_object', 'refit_object', 'enabled_methods', 'region_selector', 'NoMethodsError']

log = logging.getLogger(__name__)

Failures = list[tuple[str, str]]


class NoMethodsError(ValueError):
    """No method, or an unknown one, is requested; the message is user-ready"""


class _Cancelled(Exception):
    """Raised inside measure_object when the cancel event is set"""


def enabled_methods(cfg: Any, override: Sequence[str] | None = None) -> list[str]:
    """Names of the methods to run, from ``override`` or ``cfg.methods.enabled``; none or an unknown one raises"""
    names = list(override or cfg.methods.enabled)
    unknown = [m for m in names if m not in REGISTRY]
    if not names or unknown:
        raise NoMethodsError('no RV methods available: ' + (f'unknown {", ".join(unknown)}' if unknown else
                                                            'none is enabled') +
                             f' (implemented: {", ".join(sorted(REGISTRY))})')
    return names


def _failed(method: str, exc: Exception) -> MethodResult:
    meas = Measurement(method, '', float('nan'), float('nan'), FLAGS['fit_failed'],
                       {'error': f'{type(exc).__name__}: {exc}'})
    return MethodResult(method, float('nan'), float('nan'), 0, [meas])


def _barycentric(spectra: Sequence[Spectrum], cfg: Any, mode: str, name: str) -> tuple[list[Spectrum], dict]:
    """The spectra in the requested frame, and the record of what was applied (the mean over epochs)"""
    frame = cfg.velocity_frame.frame
    corrected, kmss, sources = [], [], []
    for spec in spectra:
        kms, source = barycentric.correction(spec, mode=mode, frame=frame, overrides=cfg.velocity_frame.overrides)
        log.info('%s: barycentric correction %.3f km/s (%s)', name, kms, source)
        if source != 'skipped' and kms:  # recorded, so the telluric zero point can return to the observatory frame
            shifted = barycentric.apply(spec, kms)
            corrected.append(replace(shifted, meta={**shifted.meta, 'barycentric_applied_kms': float(kms)}))
        else:
            corrected.append(spec)
        kmss.append(float(kms))
        sources.append(source)
    return corrected, {'applied_kms': float(np.mean(kmss)), 'source': sources[0], 'frame': frame}


def region_selector(cfg: Any, names: list[str], overrides: bool = True) -> Callable[[Spectrum], dict]:
    """`regions.select` for these methods with the configured options (and ``regions.overrides`` if `overrides`)"""
    knobs = {'snr_min': cfg.regions.min_snr, 'telluric': cfg.regions.telluric_strictness,
             'telluric_corrected': True if cfg.input.telluric_corrected else None,
             'template_windows': cfg.regions.template_windows}
    opts = regions.SelectOptions(**{k: v for k, v in knobs.items() if v is not None})  # None: the default
    ov = cfg.regions.overrides if overrides else None
    return lambda spec: regions.select(spec, teff=cfg.templates.teff_prior, methods=names, overrides=ov, options=opts)


def _drop_uncovered(sel: dict, info: dict) -> dict:
    """Without the regions the applied sky correction does not cover (logged by `wavelength_correction`)"""
    gone = set(info.get('sky_tests', {}).get('uncovered', ())) if info.get('applied', 'none') != 'none' else set()
    return {m: [r for r in regs if r.name not in gone] for m, regs in sel.items()} if gone else sel


def _run_methods(spec: Spectrum, sel: dict, names: list[str], cfg: Any, executor: Executor | None,
                 rngs: dict[str, np.random.Generator], name: str, cancel: threading.Event | None,
                 failures: Failures) -> dict[str, MethodResult]:
    out = {}
    for m in names:
        if cancel is not None and cancel.is_set():
            raise _Cancelled
        out[m] = _guarded(lambda m=m: REGISTRY[m](spec, sel.get(m, []), cfg, executor=executor, rng=rngs[m]),
                          m, name, failures)
    return out


def _binary(spectra: list[Spectrum], infos: list[dict], cfg: Any, name: str, failures: Failures) -> dict:
    """The single- versus two-template test of each epoch (`methods.binary`): the largest ΔBIC, flag if any"""
    select = region_selector(cfg, ['template_fit'])
    try:
        tests = [binary.test(s, _drop_uncovered(select(s), i)['template_fit'], cfg)
                 for s, i in zip(spectra, infos, strict=True)]
    except ValueError as exc:  # the test cannot run on these data
        log.error('%s: binary test failed: %s', name, exc)
        failures.append((f'{name} binary', f'ValueError: {exc}'))
        return {}
    return dict(dbic=max(t['dbic'] for t in tests), flag=any(t['flag'] for t in tests), epochs=tests)


def _guarded(call: Callable[[], Any], method: str, name: str, failures: Failures, n: int | None = None) -> Any:
    """call(); if it raises, the failure is logged and recorded and `_failed` is returned (a list of n with n)"""
    try:
        return call()
    except Exception as exc:  # noqa: BLE001 - the one (object, method) guard: a failure never stops a batch
        if isinstance(exc, ValueError):  # a method's documented refusal ("cannot measure"): its message says why
            log.error('%s: method %s failed: %s', name, method, exc)
            log.debug('%s: method %s traceback', name, method, exc_info=exc)  # ValueError may also be a bug
        else:
            log.exception('%s: method %s failed', name, method)
        failures.append((f'{name} {method}', f'{type(exc).__name__}: {exc}'))
        return _failed(method, exc) if n is None else [_failed(method, exc) for _ in range(n)]


def _measurement_wave(meas: Measurement) -> float:
    """A representative wavelength of a measurement's region (line centre, else the median fitted wavelength)"""
    d = meas.diagnostics
    for key in ('centre', 'wave_centre'):
        if d.get(key) is not None and np.isfinite(d[key]):
            return float(d[key])
    w = d.get('wave')
    return float(np.median(np.asarray(w, dtype=float))) if w is not None and len(w) else float('nan')


def _uncorrect(per_epoch: list[dict[str, MethodResult]], corrected: list[Spectrum]) -> list[list[float]]:
    """
    Record each measurement's first-order uncorrected RV (the correction lowered it by the Δv applied at its
    wavelength), and return the wavelengths of each epoch's usable measurements
    """
    waves: list[list[float]] = []
    for res_by_method, spec in zip(per_epoch, corrected, strict=True):
        lams = []
        for res in res_by_method.values():
            for meas in res.measurements:
                lam = _measurement_wave(meas)
                dv = float(sky_zeropoint.applied_dv(spec.meta, lam)) if np.isfinite(lam) else np.nan
                meas.diagnostics['zp_applied_kms'] = dv
                meas.diagnostics['rv_uncorrected'] = float(meas.rv + dv) if np.isfinite(dv) else np.nan
                if np.isfinite(lam) and np.isfinite(meas.rv) and not meas.flag & EXCLUDE_FLAGS:
                    lams.append(lam)
        waves.append(lams)
    return waves


def _zero_point_terms(corrected: list[Spectrum], infos: list[dict], waves: list[list[float]], cfg: Any,
                      name: str) -> list[dict]:
    """The zero-point systematics of each epoch, at the wavelengths of its measurements"""
    out = []
    for spec, info, lams in zip(corrected, infos, waves, strict=True):
        w = np.asarray(lams) if lams else None
        term = wcorr.star_sky_term(info, spec, w, cfg, name)
        out.append({**{key: info[key] for key in wcorr.ZP_KEYS}, 'mode': info['mode'], 'applied': info['applied'],
                    'zp_flags': int(info['zp_flags']) | term['flags'], 'sky_tests': info['sky_tests'],
                    'wavelength_zero_point': float(systematics.wavelength_zero_point(spec, wave=w)),
                    'star_sky': term['value'], 'star_sky_term': term['term'], 'slit_prior': term['prior'],
                    'slit': term['slit']})
    return out


def measure_object(spectra: Sequence[Spectrum], cfg: Any, executor: Executor | None, rng: np.random.Generator, *,
                   name: str | None = None, methods: Sequence[str] | None = None, bary: str | None = None,
                   cancel: threading.Event | None = None, failures: Failures | None = None) -> ObjectResult:
    """
    Measure one object (one or more spectra)

    Runs the barycentric correction, the wavelength zero point, region selection, each enabled method, the
    combination and the systematics. A method that raises is logged with its traceback, recorded as a failed
    measurement (flag fit_failed) and appended to ``failures``; the object carries on. If ``cancel`` is set
    between methods, ``_Cancelled`` is raised (handled by ``run``). Several spectra are epochs: each is measured
    on its own and the epochs are combined in a second level (random effects over the epochs' RVs), so a common
    per-epoch offset is not averaged down by the number of regions.
    """
    names = enabled_methods(cfg, methods)
    failures = [] if failures is None else failures
    name = name or str(spectra[0].meta.get('object', Path(spectra[0].source).stem))
    corrected, bary_info = _barycentric(spectra, cfg, bary or cfg.velocity_frame.correction, name)
    select = region_selector(cfg, names)

    # wavelength zero point, applied once to the wavelengths; the regions the methods will use (selected before
    # any correction, which moves them by much less than their width) feed the sky-if-needed and telluric tests
    needs_regions = cfg.calibration.wavelength_correction in ('sky-if-needed', 'telluric', 'auto')
    cal_infos = []
    for k, (raw, spec) in enumerate(zip(spectra, corrected, strict=True)):
        corrected[k], info = wcorr.correct(raw, spec, cfg, name, regions=select(spec) if needs_regions else None)
        cal_infos.append(info)

    method_rngs = dict(zip(names, rng.spawn(len(names)), strict=True))
    sels = [_drop_uncovered(select(spec), info) for spec, info in zip(corrected, cal_infos, strict=True)]
    joint = forward.METHOD in names and len(corrected) > 1  # one forward model shared by the epochs
    per_epoch = [_run_methods(spec, sel, [m for m in names if not (joint and m == forward.METHOD)], cfg, executor,
                              method_rngs, name, cancel, failures)
                 for spec, sel in zip(corrected, sels, strict=True)]
    if joint:
        if cancel is not None and cancel.is_set():
            raise _Cancelled
        fits = _guarded(lambda: forward.fit_epochs(corrected, [s.get(forward.METHOD, []) for s in sels], cfg,
                                                   executor=executor, rng=method_rngs[forward.METHOD]),
                        forward.METHOD, name, failures, n=len(corrected))
        for res, fit in zip(per_epoch, fits, strict=True):
            res[forward.METHOD] = fit
    zp_epochs = _zero_point_terms(corrected, cal_infos, _uncorrect(per_epoch, corrected), cfg, name)

    # several epochs: regions become '<epoch>:<region>', and the epochs are combined in a second level
    labels = combine.epoch_labels(corrected)
    per_method = {m: [r[m] for r in per_epoch] for m in names}
    epoch_info: dict[str, Any] = {}
    if len(corrected) > 1:
        epoch_info = combine.epoch_summaries(per_method, labels, cfg.combine.tau_scale, name,
                                             sys=[z['wavelength_zero_point'] for z in zp_epochs])
        for lst in per_method.values():
            for label, res in zip(labels, lst, strict=True):
                for meas in res.measurements:
                    meas.diagnostics['epoch'] = label
                    meas.region = f'{label}:{meas.region}'
    results = {m: lst[0] if len(lst) == 1 else combine.method_over_epochs(m, lst, labels, name)
               for m, lst in per_method.items()}
    for m, res in results.items():
        if np.isfinite(res.rv):
            log.info('%s: %.1f ± %.1f km/s', m, res.rv, res.rv_err)
        else:
            log.info('%s: no result', m)

    good = {m: r for m, r in results.items() if np.isfinite(r.rv)}
    adopted: dict[str, Any] = {}
    if good and epoch_info:
        adopted = combine.adopted_over_epochs(epoch_info, cfg.combine.tau_scale, name)
    elif good:  # every usable per-region measurement of every method; flagged ones are excluded (and logged)
        adopted = combine.hierarchical_from_methods(good, tau_scale=cfg.combine.tau_scale)
        if not np.isfinite(adopted['median']):  # nothing usable after the exclusions
            adopted = {}
    if adopted and epoch_info:
        adopted['epochs'] = epoch_info
    if adopted:
        log.info('%s: adopted RV %.2f ± %.2f km/s', name, adopted['median'], adopted['err'])
    else:
        log.warning('%s: no adopted RV', name)

    binary_info = _binary(corrected, cal_infos, cfg, name, failures) if cfg.methods.binary else {}
    sysd = wcorr.combine_epochs(zp_epochs)
    bary_info.update(wavelength_correction=sysd.pop('applied'), wavelength_correction_mode=sysd.pop('mode'),
                     zp_flags=int(sysd.pop('zp_flags')), sky_tests=sysd.pop('sky_tests'))
    meta = corrected[0].meta  # the curves `plotting.plot_wavelength_correction` draws (first epoch)
    bary_info['wavelength_curves'] = {k: meta.get(k) for k in ('sky_zeropoint', 'telluric_zeropoint',
                                                              'wavelength_correction')}
    if len(zp_epochs) > 1:
        bary_info['wavelength_correction_epochs'] = [
            {k: v for k, v in z.items() if k != 'sky_tests'} | {'epoch': lab}
            for z, lab in zip(zp_epochs, labels, strict=True)]
    sysd['template_spread'] = float(systematics.template_spread(corrected[0], sels[0], cfg, executor=executor,
                                                                rng=rng.spawn(1)[0]))
    sysd['floor'] = float(cfg.combine.systematic_floor)
    unknown = [k for k, v in (('the adopted error', adopted.get('err')), ('sys_template', sysd['template_spread']),
                              ('sys_wavezp', sysd.get('wavelength_zero_point')),
                              ('sys_star_sky', sysd.get('star_sky_systematic')))
               if v is None or not np.isfinite(v)]
    if unknown:  # results._quad never counts an unknown term as 0
        bary_info['zp_flags'] |= wcorr.ZP_FLAGS['total_err_undefined']
        log.warning('%s: rv_total_err is NaN: %s unknown (zp flag total_err_undefined)', name, ', '.join(unknown))
    if 'linecentre' in results:  # True only with the opt-in, template-calibrated shape correction
        sysd['linecentre_shape_correction'] = any(m.diagnostics.get('shape_correction')
                                                  for m in results['linecentre'].measurements)
    return ObjectResult(object=name, files=[s.source for s in spectra], methods=results, adopted=adopted,
                        systematics=sysd, barycentric=bary_info, binary=binary_info, config_hash=_config_hash(cfg),
                        version=__version__)


def _config_hash(cfg: Any) -> str:
    try:
        return str(cfg.hash())
    except TypeError as exc:  # option values that are not TOML data, e.g. an in-memory template Spectrum
        log.warning('the configuration cannot be hashed (%s): config_hash is empty', exc)
        return ''


def _read_all(cfg: Any, inputs: Iterable[str | Path], medium: str, failures: Failures | None = None
              ) -> list[Spectrum]:
    spectra = []
    for inp in ([inputs] if isinstance(inputs, (str, Path)) else inputs):
        spectra += archives.load_spectra(archives.collect_inputs(Path(inp)), failures, medium=medium,
                                         hdu=cfg.input.hdu, resolution=cfg.input.resolution)
    return spectra


def _measure(spectra: Sequence[Spectrum], cfg: Any, n_procs: int | None, rng: np.random.Generator, **kw
             ) -> ObjectResult:
    """`measure_object` with a spawn-context process pool (safe with Qt threads) when ``n_procs`` > 1"""
    pool = (ProcessPoolExecutor(n_procs, mp_context=multiprocessing.get_context('spawn'))
            if n_procs and n_procs > 1 else nullcontext())
    with pool as ex:
        return measure_object(spectra, cfg, ex, rng, **kw)


def run(cfg: Any, inputs: str | Path | Iterable[str | Path] | None = None, *, n_procs: int | None = None,
        medium: str | None = None, methods: Sequence[str] | None = None, bary: str | None = None,
        on_object: Callable[[ObjectResult], None] | None = None, cancel: threading.Event | None = None,
        failures: Failures | None = None) -> list[ObjectResult]:
    """
    Run the analysis on all spectra found in ``inputs`` (default ``cfg.input.paths``)

    Objects are processed one after another (``[i/N] name``). ``n_procs``, ``medium``, ``methods`` and ``bary``
    override the config. ``on_object`` is called with each finished object; if ``cancel`` is set (checked between
    objects and between methods) the results so far are returned and the interrupted object is dropped. Every
    skipped file and failed (object, method) is appended to ``failures`` as ``(what, why)`` and summarised at
    ERROR at the end.
    """
    enabled_methods(cfg, methods)  # fail before reading anything
    failures = [] if failures is None else failures
    n_procs = cfg.parallel.n_procs if n_procs is None else n_procs
    spectra = _read_all(cfg, cfg.input.paths if inputs is None else inputs, medium or cfg.input.medium, failures)
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
        try:
            res = _measure(groups[name], obj_cfg, n_procs, np.random.default_rng(object_seed(cfg.seed, name)),
                           name=name, methods=methods, bary=bary, cancel=cancel, failures=failures)
        except _Cancelled:
            log.warning('Cancelled during %s', name)
            break
        res.refitted = obj_cfg is not cfg  # `config_for_object` returns cfg itself when no patch applies
        out.append(res)
        if on_object is not None:
            on_object(res)
    if failures:
        log.error('%d failure(s) during the run:', len(failures))
        for what, why in failures:
            log.error('  %s: %s', what, why)
    return out


def refit_object(result: ObjectResult, cfg: Any, cfg_patch: dict, *, inputs: Sequence[str | Path] | None = None,
                 n_procs: int | None = 1, medium: str | None = None, methods: Sequence[str] | None = None,
                 bary: str | None = None) -> tuple[ObjectResult, Any]:
    """
    Rerun one object with ``cfg_patch`` merged into ``cfg`` (not modified); returns the result and the new Config

    The patch (e.g. from `refit.suggested_refits`, after downloading any grid it needs) is recorded as
    ``{'object': name, 'patch': ...}`` in ``new_cfg.refits`` (written as ``[[refits]]``), so `run` applies it to
    that object only. The seed is `refit.object_seed`, so running the written TOML on any inputs containing this
    object reproduces the refit.
    """
    new_cfg = apply_patch(cfg, {})
    new_cfg.refits.append({'object': result.object, 'patch': plain(cfg_patch)})
    obj_cfg = config_for_object(new_cfg, result.object)
    obj_cfg.validate()
    log.info('Refitting %s with %s', result.object, plain(cfg_patch))
    spectra = _read_all(obj_cfg, list(inputs or result.files), medium or obj_cfg.input.medium)
    if not spectra:
        raise FileNotFoundError(f'none of the files of {result.object} could be read: {result.files}')
    res = _measure(spectra, obj_cfg, n_procs, np.random.default_rng(object_seed(new_cfg.seed, result.object)),
                   name=result.object, methods=methods, bary=bary)
    res.refitted = True
    return res, new_cfg
