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
    overrides: dict = field(default_factory=dict)


@dataclass
class RunCfg:
    n_procs: int = 1
    seed: int = 42
    output_dir: str = ''
    formats: list[str] = field(default_factory=lambda: ['csv'])


@dataclass
class FakeConfig:
    medium: MediumCfg = field(default_factory=MediumCfg)
    barycentric: BaryCfg = field(default_factory=BaryCfg)
    methods: MethodsCfg = field(default_factory=MethodsCfg)
    templates: TemplatesCfg = field(default_factory=TemplatesCfg)
    regions: RegionsCfg = field(default_factory=RegionsCfg)
    run: RunCfg = field(default_factory=RunCfg)


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

    def select_regions(self, spectrum, cfg):
        return [Region('K I 7701', 'line', 7695., 7707., 7701.084, (), 'test'),
                Region('K I 7667', 'line', 7660., 7672., 7667.009, (), 'test')]

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
