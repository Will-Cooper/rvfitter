"""A fake GUI backend and a toy Config tree, so the GUI can be tested without the core modules."""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pandas as pd

from rvfitter.io import SpectrumEntry
from rvfitter.types import Region
from synth import make_spectrum


@dataclass
class CubeCfg:
    position: tuple[float, float] | None = None
    radius: float | None = None


@dataclass
class InputCfg:  # the paths of rvfitter.config that the Setup tab binds to
    medium: Literal['auto', 'air', 'vacuum'] = 'auto'
    medium_method: Literal['ciddor', 'edlen', 'morton'] = 'ciddor'
    resolution: float | None = None
    objects: str | list[int | str] = 'auto'
    cube: CubeCfg = field(default_factory=CubeCfg)


@dataclass
class VelocityFrameCfg:
    correction: Literal['auto', 'apply', 'skip'] = 'auto'
    frame: Literal['barycentric', 'heliocentric'] = 'barycentric'


@dataclass
class CalibrationCfg:
    wavelength_correction: Literal['sky-if-needed', 'sky', 'auto', 'telluric', 'none'] = 'sky-if-needed'


@dataclass
class MethodsCfg:
    enabled: list[str] = field(default_factory=lambda: ['ccf'])
    options: dict[str, dict] = field(default_factory=dict)


@dataclass
class TemplatesCfg:
    grid: str = 'bt-settl-cifist'
    local_dir: str | None = None
    teff_prior: tuple[float, ...] | None = None


@dataclass
class RegionsCfg:
    overrides: dict[str, list[dict]] = field(default_factory=dict)


@dataclass
class ParallelCfg:
    n_procs: int | None = None  # as rvfitter.config: None = automatic


@dataclass
class OutputCfg:
    path: str = ''
    formats: list[str] = field(default_factory=lambda: ['csv'])


@dataclass
class FakeConfig:
    seed: int = 42
    input: InputCfg = field(default_factory=InputCfg)
    velocity_frame: VelocityFrameCfg = field(default_factory=VelocityFrameCfg)
    calibration: CalibrationCfg = field(default_factory=CalibrationCfg)
    methods: MethodsCfg = field(default_factory=MethodsCfg)
    templates: TemplatesCfg = field(default_factory=TemplatesCfg)
    regions: RegionsCfg = field(default_factory=RegionsCfg)
    parallel: ParallelCfg = field(default_factory=ParallelCfg)
    output: OutputCfg = field(default_factory=OutputCfg)


def _from_dict(d: dict) -> FakeConfig:
    cfg = FakeConfig()
    cfg.seed = d.pop('seed')
    for sect, vals in d.items():
        obj = getattr(cfg, sect)
        for k, v in vals.items():
            if k == 'teff_prior' and v is not None:
                v = tuple(v)
            if k == 'cube':
                v = CubeCfg(**v)
            setattr(obj, k, v)
    return cfg


def make_result(name: str, rv: float = 10.0):
    return SimpleNamespace(object=name, files=[f'{name}.txt'], rv=rv, methods={}, adopted={},
                           barycentric={'applied_kms': -12.345, 'source': 'computed', 'frame': 'barycentric'})


class FakeBackend:
    """
    Satisfies ``gui.backend.Backend``; the run blocks on ``gate`` (if set) before the second object. Spectra have
    R ``resolution`` from ``resolution_source``; the run reports a step per (file, method). A file whose stem is in
    ``multi`` holds those entries (one object per entry, picked by ``input.objects``); one whose stem is in ``cubes``
    is an IFU cube with that collapsed image.
    """

    def __init__(self, tmp: Path | None = None, objects=('alpha', 'beta')):
        self.objects = objects
        self.resolution, self.resolution_source = 3000., 'header'
        self.gate: threading.Event | None = None
        self.started = threading.Event()
        self.runs: list[dict] = []
        self.written: list[tuple] = []
        self.tmp = tmp
        self.edge_objects: set[str] = set()
        self.downloads: list[dict] = []
        self.refits: list[tuple] = []
        self.missing_grids: set[str] = set()  # template_status reports these not downloaded
        self.download_error: str | None = None  # download_templates raises this
        self.multi: dict[str, list[SpectrumEntry]] = {}
        self.cubes: dict[str, np.ndarray] = {}
        self.reads: list[tuple[str, tuple | None, float | None]] = []  # (path, cube position, radius) per read

    def version(self):
        return '0.0-test'

    def default_config(self):
        return FakeConfig()

    def load_config(self, path):
        return _from_dict(json.loads(Path(path).read_text()))

    def save_config(self, cfg, path):
        Path(path).write_text(json.dumps(asdict(cfg), indent=1))

    def collect_inputs(self, path):
        path = Path(path)
        if path.is_dir():
            return sorted(p for p in path.iterdir() if p.suffix == '.txt')
        return [path]

    def group_objects(self, files, cfg):
        out: dict[str, list[str]] = {}
        objects = cfg.input.objects
        for f in files:
            entries = self.multi.get(Path(f).stem)
            if entries is None:
                out.setdefault(Path(f).stem.split('_')[0], []).append(str(f))
                continue
            usable = [e for e in entries if e.usable]
            picks = (usable if objects in ('auto', 'all') else usable[:1] if objects == 'target' else
                     [e for e in entries if e.name in objects or str(e.hdu) in map(str, objects)])
            for e in picks:
                out.setdefault(e.name, []).append(f'{f}[{e.hdu}]')
        return out

    def list_spectra(self, path, cfg):
        stem = Path(path).stem
        if stem in self.cubes:
            return [SpectrumEntry(1, 'SCI', 'cube', 50., True),
                    SpectrumEntry(2, 'ERR', 'error', float('nan'), False, 'EXTNAME ERR: error plane, not a spectrum')]
        return self.multi.get(stem, [SpectrumEntry(None, Path(path).name, 'object', float('nan'), True)])

    def cube_image(self, path):
        img = self.cubes[Path(path).stem]
        y, x = np.unravel_index(np.nanargmax(img), img.shape)
        return img, (int(x), int(y))

    def read_spectrum(self, path, cfg):
        self.reads.append((str(path), cfg.input.cube.position, cfg.input.cube.radius))
        spec, _ = make_spectrum(rv=0., resolution=self.resolution, snr=50., wave_range=(7500., 8000.),
                                rng=np.random.default_rng(1))
        spec.meta.update(resolution_source=self.resolution_source, medium_in='vacuum', medium_source='header:TEST',
                         resolution_implausible=False)
        return spec

    def default_for(self, spectrum):
        cfg = FakeConfig()
        cfg.methods.enabled = ['ccf', 'template_fit']
        return cfg

    def setting_advice(self, spectrum, cfg, result=None):
        return {'input.medium': ('air', 'the test advice'), 'input.resolution': (None, 'a reason only')}

    def regime(self, spectrum):
        return 'low-resolution optical'

    def select_regions(self, spectrum, cfg, with_overrides=False):
        regs = {'ccf': [Region('K I 7701', 'line', 7695., 7707., 7701.084, (), 'test'),
                               Region('K I 7667', 'line', 7660., 7672., 7667.009, (), 'test')]}
        if with_overrides:
            for method, edits in cfg.regions.overrides.items():
                cur = {r.name: r for r in regs.get(method, [])}
                for e in edits:
                    if e.get('remove'):
                        cur.pop(e['name'], None)
                    else:
                        cur[e['name']] = Region(e['name'], e['kind'], e['wmin'], e['wmax'], e.get('centre'))
                regs[method] = list(cur.values())
        return regs

    def run(self, cfg, groups, n_procs, on_result, on_step, cancel):
        self.runs.append({'cfg': cfg, 'groups': dict(groups), 'n_procs': n_procs})
        self.started.set()
        out = []
        for i, name in enumerate(groups):
            if i == 1 and self.gate is not None:
                self.gate.wait(10)
            if cancel.is_set():
                break
            for _f in groups[name]:
                for m in cfg.methods.enabled:
                    on_step(name, m)
            res = make_result(name, 10.0 + i)
            out.append(res)
            on_result(res)
        return out

    def to_table(self, results):
        return pd.DataFrame([{'object': r.object, 'n_spectra': len(r.files), 'rv_adopted': r.rv,
                              'rv_adopted_err': 0.5} for r in results])

    def write_results(self, results, path, formats):
        self.written.append((Path(path), tuple(formats), len(results)))

    def suggested_refits(self, result):
        if result.object not in self.edge_objects:
            return []
        return [dict(method='template_fit', axis='met', direction='low', action='download',
                     message=f'{result.object}: best met is at the low edge',
                     cfg_patch={'templates': {'grid': 'elf-owl'}},
                     download={'grid': 'elf-owl', 'teff_range': (1500., 2100.)}, command='rvfitter templates download'),
                dict(method='template_fit', axis='logg', direction='high', action='none',
                     message='logg edge: no grid extends further', cfg_patch={}, download=None, command='')]

    def refit(self, result, cfg, suggestion):
        import copy
        self.downloads.append(dict(suggestion['download'] or {}))
        self.refits.append((result.object, suggestion['cfg_patch']))
        new_cfg = copy.deepcopy(cfg)
        new_cfg.templates.grid = 'elf-owl'
        new_cfg.refits = [{'object': result.object, 'patch': suggestion['cfg_patch']}]
        new = make_result(result.object, 99.0)
        self.edge_objects.discard(result.object)
        return new, new_cfg

    def template_status(self, cfg):
        grids = [cfg.templates.grid] if isinstance(cfg.templates.grid, str) else cfg.templates.grid
        return {g: None if g in self.missing_grids else (1200., 3000.) for g in grids}

    def template_methods(self, cfg):
        return [m for m in cfg.methods.enabled if m != 'linecentre']

    def download_templates(self, cfg, grids):
        if self.download_error:
            raise ValueError(self.download_error)
        self.downloads.append({'grids': list(grids)})
        self.missing_grids -= set(grids)
