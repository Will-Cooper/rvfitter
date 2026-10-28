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

from rvfitter.types import Region
from synth import make_spectrum


@dataclass
class MediumCfg:
    medium: Literal['auto', 'air', 'vacuum'] = 'auto'
    formula: Literal['ciddor', 'edlen', 'morton'] = 'ciddor'


@dataclass
class BaryCfg:
    correction: Literal['auto', 'apply', 'skip'] = 'auto'
    frame: Literal['barycentric', 'heliocentric'] = 'barycentric'


@dataclass
class MethodsCfg:
    linecentre: bool = True
    ccf: bool = True
    template_fit: bool = False
    forward_mcmc: bool = False
    lsd: bool = False
    telluric_ref: bool = False


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
    n_procs: int = 1
    seed: int = 42


@dataclass
class OutputCfg:
    path: str = ''
    formats: list[str] = field(default_factory=lambda: ['csv'])


@dataclass
class FakeConfig:
    medium: MediumCfg = field(default_factory=MediumCfg)
    barycentric: BaryCfg = field(default_factory=BaryCfg)
    methods: MethodsCfg = field(default_factory=MethodsCfg)
    templates: TemplatesCfg = field(default_factory=TemplatesCfg)
    regions: RegionsCfg = field(default_factory=RegionsCfg)
    parallel: ParallelCfg = field(default_factory=ParallelCfg)
    output: OutputCfg = field(default_factory=OutputCfg)


def _from_dict(d: dict) -> FakeConfig:
    cfg = FakeConfig()
    for sect, vals in d.items():
        obj = getattr(cfg, sect)
        for k, v in vals.items():
            if k == 'teff_prior' and v is not None:
                v = tuple(v)
            setattr(obj, k, v)
    return cfg


def make_result(name: str, rv: float = 10.0):
    return SimpleNamespace(object=name, files=[f'{name}.txt'], rv=rv,
                           barycentric={'applied_kms': -12.345, 'source': 'computed', 'frame': 'barycentric'})


class FakeBackend:
    """Satisfies ``gui.backend.Backend``; the run blocks on ``gate`` (if set) before the second object."""

    def __init__(self, tmp: Path | None = None, objects=('alpha', 'beta')):
        self.objects = objects
        self.gate: threading.Event | None = None
        self.started = threading.Event()
        self.runs: list[dict] = []
        self.written: list[tuple] = []
        self.tmp = tmp
        self.edge_objects: set[str] = set()
        self.downloads: list[dict] = []
        self.refits: list[tuple] = []

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
        for f in files:
            out.setdefault(Path(f).stem.split('_')[0], []).append(str(f))
        return out

    def read_spectrum(self, path, cfg):
        spec, _ = make_spectrum(rv=0., resolution=3000., snr=50., wave_range=(7500., 8000.),
                                rng=np.random.default_rng(1))
        return spec

    def select_regions(self, spectrum, cfg, with_overrides=False):
        regs = {'linecentre': [Region('K I 7701', 'line', 7695., 7707., 7701.084, (), 'test'),
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

    def run(self, cfg, groups, n_procs, on_result, should_cancel):
        self.runs.append({'cfg': cfg, 'groups': dict(groups), 'n_procs': n_procs})
        self.started.set()
        out = []
        for i, name in enumerate(groups):
            if i == 1 and self.gate is not None:
                self.gate.wait(10)
            if should_cancel():
                break
            res = make_result(name, 10.0 + i)
            out.append(res)
            on_result(res)
        return out

    def to_table(self, results):
        return pd.DataFrame([{'object': r.object, 'n_spectra': len(r.files), 'rv_adopted': r.rv,
                              'rv_adopted_err': 0.5} for r in results])

    def write_results(self, results, path, formats):
        self.written.append((Path(path), tuple(formats), len(results)))

    def plot(self, kind, result):
        if kind == 'ccf':
            return None  # exercises the placeholder
        from matplotlib.figure import Figure
        fig = Figure()
        fig.add_subplot(111).plot([0, 1], [0, 1])
        return fig

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
