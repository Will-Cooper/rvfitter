"""Per-object configuration patches: refit guidance, ``[[refits]]`` entries and per-object seeds."""
import hashlib
from typing import Any

import numpy as np

from .types import ObjectResult

__all__ = ['apply_patch', 'config_for_object', 'object_seed', 'suggested_refits', 'plain']


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for k, v in patch.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def plain(obj: Any) -> Any:
    """``obj`` with numpy scalars and tuples made TOML-friendly (recursively)"""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def apply_patch(cfg: Any, cfg_patch: dict) -> Any:
    """A new, validated Config: ``cfg`` with ``cfg_patch`` (e.g. ``{'templates': {'met_range': [-1, 0]}}``) merged in"""
    from . import config
    return config.from_dict(_deep_merge(cfg.to_dict(), plain(cfg_patch)))


def object_seed(seed: int, name: str) -> np.random.SeedSequence:
    """
    The seed sequence of one object: from the run seed and a stable hash of the object name

    It does not depend on the other objects in the input or their order, so a run on any set of inputs and
    `pipeline.refit_object` give the same random numbers for the same object.
    """
    h = int.from_bytes(hashlib.blake2b(str(name).encode('utf-8'), digest_size=8).digest(), 'little')
    return np.random.SeedSequence([int(seed), h])


def config_for_object(cfg: Any, name: str) -> Any:
    """``cfg`` with the patches of the object's ``[[refits]]`` entries applied in order (``cfg`` itself if none)"""
    out = cfg
    for r in cfg.refits:
        if r.get('object') == name:
            out = apply_patch(out, r.get('patch', {}))
    return out


def suggested_refits(result: ObjectResult) -> list[dict]:
    """
    One action per distinct (axis, direction, action) of an object's template-edge guidance, for the CLI log or a
    GUI button: 'method', 'axis', 'direction', 'action' ('widen' | 'download' | 'none'), 'message', 'cfg_patch'
    (for `pipeline.refit_object`), 'download' (`templates.download` keywords, or None) and 'command' (CLI)
    """
    seen, out = set(), []
    for name, mres in result.methods.items():
        for meas in mres.measurements:
            for e in (meas.diagnostics.get('template_summary') or {}).get('edges', []):
                key = (e['axis'], e['direction'], e['action'])
                if key not in seen:
                    seen.add(key)
                    out.append(dict(method=name, **{k: e[k] for k in ('axis', 'direction', 'action', 'message',
                                                                      'cfg_patch', 'download', 'command')}))
    return out
