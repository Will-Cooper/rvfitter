"""Thin adapter between the GUI and the rvfitter core; tests substitute a fake :class:`Backend`."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

FORMATS = ('csv', 'json', 'tsv', 'html')


class Backend(Protocol):
    """What the GUI needs from the core."""

    def version(self) -> str: ...

    def default_config(self) -> Any: ...

    def load_config(self, path: Path) -> Any: ...

    def save_config(self, cfg: Any, path: Path) -> None: ...

    def collect_inputs(self, path: Path) -> list[Path]: ...

    def group_objects(self, files: Sequence[Path], cfg: Any) -> dict[str, list[str]]: ...

    def read_spectrum(self, path: str | Path, cfg: Any) -> Any: ...

    def select_regions(self, spectrum: Any, cfg: Any, with_overrides: bool = False) -> dict[str, list[Any]]: ...

    def run(self, cfg: Any, groups: Mapping[str, Sequence[str]], n_procs: int,
            on_result: Callable[[Any], None], cancel: threading.Event) -> list[Any]: ...

    def to_table(self, results: Sequence[Any]) -> Any: ...

    def write_results(self, results: Sequence[Any], path: Path, formats: Sequence[str]) -> None: ...

    def suggested_refits(self, result: Any) -> list[dict]: ...

    def refit(self, result: Any, cfg: Any, suggestion: Mapping[str, Any]) -> tuple[Any, Any]: ...


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
        """Read one spectrum honouring ``cfg.input`` (medium, medium_method, hdu, resolution)."""
        from rvfitter import io
        i = cfg.input
        return io.read_spectrum(path, medium=i.medium, medium_method=i.medium_method, hdu=i.hdu,
                                resolution=i.resolution)

    def group_objects(self, files: Sequence[Path], cfg: Any) -> dict[str, list[str]]:
        from rvfitter import archives
        spectra = []
        for f in files:
            try:
                spectra.append(self.read_spectrum(f, cfg))
            except (OSError, ValueError) as exc:  # an unreadable file is skipped, never fatal
                log.warning('Skipping %s: %s', f, exc)
        groups = archives.group_by_object(spectra, cfg.grouping.override)
        return {name: [s.source for s in specs] for name, specs in groups.items()}

    def select_regions(self, spectrum: Any, cfg: Any, with_overrides: bool = False) -> dict[str, list[Any]]:
        """Regions per method, as the pipeline selects them, with or without ``cfg.regions.overrides``."""
        from rvfitter import pipeline
        return pipeline.region_selector(cfg, list(cfg.methods.enabled), with_overrides)(spectrum)

    def run(self, cfg: Any, groups: Mapping[str, Sequence[str]], n_procs: int,
            on_result: Callable[[Any], None], cancel: threading.Event) -> list[Any]:
        from rvfitter import pipeline
        files = [Path(f) for fs in groups.values() for f in fs]
        return pipeline.run(cfg, files, n_procs=n_procs, on_object=on_result, cancel=cancel)

    def to_table(self, results: Sequence[Any]) -> Any:
        from rvfitter import results as res_mod
        return res_mod.to_table(list(results))

    def write_results(self, results: Sequence[Any], path: Path, formats: Sequence[str]) -> None:
        from rvfitter import results as res_mod
        res_mod.write(list(results), path, list(formats))

    def suggested_refits(self, result: Any) -> list[dict]:
        """The template-edge guidance of ``result`` as actions (``refit.suggested_refits``)."""
        from rvfitter import refit
        return refit.suggested_refits(result)

    def refit(self, result: Any, cfg: Any, suggestion: Mapping[str, Any]) -> tuple[Any, Any]:
        """
        Download the grid of ``suggestion`` (if it needs one), then refit the object with its configuration patch

        Returns ``(new ObjectResult, new Config)``; the new Config records the refit in its ``[[refits]]``.
        """
        from rvfitter import pipeline, templates
        if suggestion.get('download'):
            log.info('Downloading template grid: %s', suggestion['download'])
            templates.download(**suggestion['download'])
            log.info('Download finished.')
        return pipeline.refit_object(result, cfg, suggestion['cfg_patch'], n_procs=cfg.parallel.n_procs)
