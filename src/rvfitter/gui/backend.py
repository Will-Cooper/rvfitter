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

    def list_spectra(self, path: str | Path, cfg: Any) -> list[Any]: ...

    def cube_image(self, path: str | Path) -> tuple[Any, tuple[int, int]]: ...

    def default_for(self, spectrum: Any) -> Any: ...

    def regime(self, spectrum: Any) -> str: ...

    def frame_status(self, spectrum: Any, cfg: Any) -> Any: ...

    def wavelength_status(self, spectrum: Any, cfg: Any) -> str: ...

    def setting_advice(self, spectrum: Any, cfg: Any, result: Any = None) -> dict[str, tuple[Any, str]]: ...

    def select_regions(self, spectrum: Any, cfg: Any, with_overrides: bool = False) -> dict[str, list[Any]]: ...

    def run(self, cfg: Any, groups: Mapping[str, Sequence[str]], n_procs: int | None,
            on_result: Callable[[Any], None], on_step: Callable[[str, str], None],
            cancel: threading.Event) -> list[Any]: ...

    def to_table(self, results: Sequence[Any]) -> Any: ...

    def write_results(self, results: Sequence[Any], path: Path, formats: Sequence[str]) -> None: ...

    def suggested_refits(self, result: Any) -> list[dict]: ...

    def refit(self, result: Any, cfg: Any, suggestion: Mapping[str, Any]) -> tuple[Any, Any]: ...

    def template_status(self, cfg: Any) -> dict[str, tuple[float, float] | None]: ...

    def template_methods(self, cfg: Any) -> list[str]: ...

    def download_templates(self, cfg: Any, grids: Sequence[str], source_dir: str | None = None) -> None: ...


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
        """Read one spectrum ('file.fits[hdu]' for one entry) honouring ``cfg.input`` (medium, hdu, R, cube)."""
        from rvfitter import io
        i = cfg.input
        return io.read_spectrum(path, medium=i.medium, medium_method=i.medium_method, hdu=i.hdu,
                                resolution=i.resolution, cube=i.cube)

    def list_spectra(self, path: str | Path, cfg: Any) -> list[Any]:
        """The candidate spectra of a file (``io.list_spectra``: hdu, name, kind, snr, usable, reason, named)."""
        from rvfitter import io
        return io.list_spectra(path, cube=cfg.input.cube)

    def cube_image(self, path: str | Path) -> tuple[Any, tuple[int, int]]:
        """The median-collapsed image of an IFU cube and its brightest spaxel (x, y)"""
        from rvfitter import io
        img = io.cube_image(path)
        return img, io.brightest_spaxel(img)

    def default_for(self, spectrum: Any) -> Any:
        """The regime defaults of ``spectrum`` (``config.default_for``)."""
        from rvfitter import config
        return config.default_for(spectrum)

    def setting_advice(self, spectrum: Any, cfg: Any, result: Any = None) -> dict[str, tuple[Any, str]]:
        """``config.setting_advice`` for ``spectrum``, with the best-fit Teff of ``result`` (its object's last run)"""
        import math

        from rvfitter import config, results
        teff = results._template_row(result)['template_teff'] if result is not None else math.nan
        return config.setting_advice(spectrum, cfg, None if math.isnan(teff) else teff)

    def frame_status(self, spectrum: Any, cfg: Any) -> Any:
        """What the velocity correction does for ``spectrum`` under ``cfg`` (``barycentric.status``)"""
        from rvfitter import barycentric
        vf = cfg.velocity_frame
        return barycentric.status(spectrum, vf.correction, vf.frame, vf.overrides)

    def wavelength_status(self, spectrum: Any, cfg: Any) -> str:
        """What the wavelength correction does for ``spectrum`` (``config.wavelength_status``)"""
        from rvfitter import config
        return config.wavelength_status(spectrum, cfg)

    def regime(self, spectrum: Any) -> str:
        """E.g. 'low-resolution optical' (``config.regime_for``)."""
        from rvfitter import config
        regime, band = config.regime_for(spectrum.resolution, (spectrum.wmin, spectrum.wmax))
        return f'{regime}-resolution {band}'

    def group_objects(self, files: Sequence[Path], cfg: Any) -> dict[str, list[str]]:
        from rvfitter import archives, io
        spectra = []
        i = cfg.input
        for f in files:
            try:  # every spectrum input.objects selects ('file.fits[hdu]' sources for multi-object files)
                spectra += io.read_spectra(f, objects=i.objects, medium=i.medium, medium_method=i.medium_method,
                                           hdu=i.hdu, resolution=i.resolution, cube=i.cube)
            except (OSError, ValueError) as exc:  # an unreadable file is skipped, never fatal
                log.warning('Skipping %s: %s', f, exc)
        groups = archives.group_by_object(spectra, cfg.grouping.override)
        return {name: [s.source for s in specs] for name, specs in groups.items()}

    def select_regions(self, spectrum: Any, cfg: Any, with_overrides: bool = False) -> dict[str, list[Any]]:
        """Regions per method, as the pipeline selects them, with or without ``cfg.regions.overrides``."""
        from rvfitter import pipeline
        return pipeline.region_selector(cfg, list(cfg.methods.enabled), with_overrides)(spectrum)

    def run(self, cfg: Any, groups: Mapping[str, Sequence[str]], n_procs: int | None,
            on_result: Callable[[Any], None], on_step: Callable[[str, str], None],
            cancel: threading.Event) -> list[Any]:
        from rvfitter import pipeline
        files = [Path(f) for fs in groups.values() for f in fs]
        return pipeline.run(cfg, files, n_procs=n_procs, on_object=on_result, on_step=on_step, cancel=cancel)

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

    def template_status(self, cfg: Any) -> dict[str, tuple[float, float] | None]:
        """Each configured grid → the Teff range [K] present locally (``templates.local_dir`` or the cache), or None"""
        from rvfitter import templates
        from rvfitter.methods import _common as C
        out: dict[str, tuple[float, float] | None] = {}
        for name in C.grid_names(cfg.templates):
            directory = C.grid_directory(cfg.templates, name)
            out[name] = None if directory is None else \
                templates.TemplateGrid(directory, grid=name, verify=False).bounds()['teff']
        return out

    def template_methods(self, cfg: Any) -> list[str]:
        """
        The enabled methods that read the template grid: all but linecentre (unless its shape_correction is on), and
        none given its own template by the method option ``template``/``templates``
        """
        out = []
        for m in cfg.methods.enabled:
            opts = cfg.methods.options.get(m, {})
            if opts.get('template') is not None or opts.get('templates'):
                continue
            if m != 'linecentre' or opts.get('shape_correction'):
                out.append(m)
        return out

    def download_templates(self, cfg: Any, grids: Sequence[str], source_dir: str | None = None) -> None:
        """
        Download ``grids`` into the cache over the configured parameter ranges (unset: ``templates.download``'s), or
        convert them from the original files in the folder ``source_dir`` (no download)
        """
        from rvfitter import templates
        t = cfg.templates
        ranges: dict[str, Any] = {k: tuple(v) for k, v in (('teff_range', t.teff_range), ('logg_range', t.logg_range),
                                                           ('met_range', t.met_range), ('alpha_range', t.alpha_range))
                                  if v is not None}
        if t.ranges:
            ranges['ranges'] = {k: tuple(v) for k, v in t.ranges.items()}
        for grid in grids:
            log.info('%s template grid %s (%s).', f'Converting from {source_dir}:' if source_dir else 'Downloading',
                     grid, ranges or 'default ranges')
            templates.download(grid, **ranges, source_dir=source_dir)
            log.info('Template grid %s ready.', grid)
