"""Thin adapter between the GUI and the rvfitter core.

The GUI never imports ``config``, ``archives``, ``pipeline``, ``results`` or ``plotting`` directly.
Everything goes through a :class:`Backend`, so that tests can substitute a fake and so that a missing or
unfinished core module only disables the feature that needs it.
"""
from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

METHODS = ('linecentre', 'ccf', 'template_fit', 'forward_mcmc', 'lsd', 'telluric_ref')
FORMATS = ('csv', 'json', 'tsv', 'html')
PLOT_KINDS = {
    'line_fits': 'plot_line_fits',
    'ccf': 'plot_ccf',
    'corner': 'plot_corner',
    'adopted': 'plot_adopted',
}


class Backend(Protocol):
    """What the GUI needs from the core."""

    def version(self) -> str: ...

    def default_config(self) -> Any: ...

    def load_config(self, path: Path) -> Any: ...

    def save_config(self, cfg: Any, path: Path) -> None: ...

    def collect_inputs(self, path: Path) -> list[Path]: ...

    def group_objects(self, files: Sequence[Path], cfg: Any) -> dict[str, list[str]]: ...

    def read_spectrum(self, path: str | Path, cfg: Any) -> Any: ...

    def select_regions(self, spectrum: Any, cfg: Any) -> list[Any]: ...

    def run(self, cfg: Any, groups: Mapping[str, Sequence[str]], n_procs: int,
            on_result: Callable[[Any], None], should_cancel: Callable[[], bool]) -> list[Any]: ...

    def to_table(self, results: Sequence[Any]) -> Any: ...

    def write_results(self, results: Sequence[Any], path: Path, formats: Sequence[str]) -> None: ...

    def plot(self, kind: str, result: Any) -> Any | None: ...


class LazyBackend:
    """The real backend: imports each core module on first use."""

    def version(self) -> str:
        from rvfitter import __version__
        return __version__

    def default_config(self) -> Any:
        from rvfitter import config
        return config.Config()

    def load_config(self, path: Path) -> Any:
        from rvfitter import config
        return config.load(path)

    def save_config(self, cfg: Any, path: Path) -> None:
        from rvfitter import config
        config.dump(cfg, path)

    def collect_inputs(self, path: Path) -> list[Path]:
        from rvfitter import archives
        return list(archives.collect_inputs(path))

    def read_spectrum(self, path: str | Path, cfg: Any) -> Any:
        from rvfitter import io
        return io.read_spectrum(path)

    def group_objects(self, files: Sequence[Path], cfg: Any) -> dict[str, list[str]]:
        from rvfitter import archives
        spectra = []
        for f in files:
            try:
                spectra.append(self.read_spectrum(f, cfg))
            except Exception as exc:  # unreadable files are skipped, never fatal
                log.warning('Skipping %s: %s', f, exc)
        override = getattr(getattr(cfg, 'io', None), 'object_override', None)
        groups = archives.group_by_object(spectra, override) if override else archives.group_by_object(spectra)
        return {name: [s.source for s in specs] for name, specs in groups.items()}

    def select_regions(self, spectrum: Any, cfg: Any) -> list[Any]:
        from rvfitter import regions

        from .cfgutil import enabled_methods
        by_method = regions.select(spectrum, methods=enabled_methods(cfg) or list(METHODS), overrides=None)
        seen: dict[str, Any] = {}
        for lst in by_method.values():
            for reg in lst:
                seen.setdefault(reg.name, reg)
        return list(seen.values())

    def run(self, cfg: Any, groups: Mapping[str, Sequence[str]], n_procs: int,
            on_result: Callable[[Any], None], should_cancel: Callable[[], bool]) -> list[Any]:
        """Run ``pipeline.run`` one object at a time so each result can be reported as it finishes."""
        from rvfitter import pipeline
        out: list[Any] = []
        for name, files in groups.items():
            if should_cancel():
                log.info('Cancelled before %s', name)
                break
            log.info('Measuring %s (%d file%s)', name, len(files), '' if len(files) == 1 else 's')
            for res in pipeline.run(cfg, [Path(f) for f in files], n_procs=n_procs):
                out.append(res)
                on_result(res)
        return out

    def to_table(self, results: Sequence[Any]) -> Any:
        from rvfitter import results as res_mod
        return res_mod.to_table(list(results))

    def write_results(self, results: Sequence[Any], path: Path, formats: Sequence[str]) -> None:
        from rvfitter import results as res_mod
        res_mod.write(list(results), path, list(formats))

    def plot(self, kind: str, result: Any) -> Any | None:
        """Return a matplotlib Figure from ``plotting``, or ``None`` if the function is missing or fails."""
        try:
            from rvfitter import plotting
        except Exception as exc:
            log.debug('plotting unavailable: %s', exc)
            return None
        func = getattr(plotting, PLOT_KINDS[kind], None)
        if func is None:
            return None
        try:
            params = inspect.signature(func).parameters
            kwargs = {'show': False} if 'show' in params else {}
            return func(result, **kwargs)
        except Exception as exc:
            log.warning('Could not draw %s for %s: %s', kind, getattr(result, 'object', '?'), exc)
            return None
