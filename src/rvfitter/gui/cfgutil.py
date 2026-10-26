"""Qt-free helpers for reading values out of a ``Config`` dataclass tree without knowing its exact layout."""
from __future__ import annotations

import dataclasses
from typing import Any

METHOD_NAMES = ('linecentre', 'ccf', 'template_fit', 'forward_mcmc', 'lsd', 'telluric_ref')
_MISSING = object()


def iter_fields(cfg: Any, prefix: str = ''):
    """Yield ``(dotted_path, value)`` for every leaf field of a dataclass tree."""
    for f in dataclasses.fields(cfg):
        value = getattr(cfg, f.name)
        path = f'{prefix}{f.name}'
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            yield from iter_fields(value, path + '.')
        else:
            yield path, value


def get_path(cfg: Any, path: str, default: Any = None) -> Any:
    obj = cfg
    for part in path.split('.'):
        obj = getattr(obj, part, _MISSING)
        if obj is _MISSING:
            return default
    return obj


def set_path(cfg: Any, path: str, value: Any) -> None:
    *head, last = path.split('.')
    obj = cfg
    for part in head:
        obj = getattr(obj, part)
    setattr(obj, last, value)


def find_field(cfg: Any, *names: str, default: Any = None) -> Any:
    """Value of the first leaf field whose last path component is one of ``names``."""
    for want in names:
        for path, value in iter_fields(cfg):
            if path.rsplit('.', 1)[-1] == want:
                return value
    return default


def enabled_methods(cfg: Any) -> list[str]:
    """Names of the enabled methods, whether ``methods`` is a list or a group of booleans."""
    methods = get_path(cfg, 'methods')
    if methods is None:
        return []
    if isinstance(methods, list | tuple | set | frozenset):
        return [str(m) for m in methods]
    if dataclasses.is_dataclass(methods):
        out = []
        for f in dataclasses.fields(methods):
            v = getattr(methods, f.name)
            if v is True or (dataclasses.is_dataclass(v) and getattr(v, 'enabled', False)):
                out.append(f.name)
        return out
    return []
