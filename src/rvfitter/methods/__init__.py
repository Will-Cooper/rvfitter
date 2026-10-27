"""
RV measurement methods.

Every method has the signature `measure(spectrum, regions, cfg, *, executor=None, rng) -> MethodResult` and is
listed in `REGISTRY` under the name used in the config and the results table (`rv_<name>`):
`REGISTRY['ccf'](spectrum, regions, cfg, executor=None, rng=rng)`.
Methods are imported lazily, so optional dependencies (e.g. emcee) are only needed when used.
"""
import importlib
from collections.abc import Callable

from ..types import MethodResult

__all__ = ['REGISTRY', 'get']


class _Lazy:
    """A picklable callable that imports a method module on first use and calls its `measure`"""

    def __init__(self, module: str):
        self.module = module

    def __call__(self, *args, **kwargs) -> MethodResult:
        return importlib.import_module(self.module).measure(*args, **kwargs)

    def __repr__(self) -> str:
        return f'<rvfitter method {self.module}.measure>'


REGISTRY: dict[str, Callable[..., MethodResult]] = {
    'ccf': _Lazy('rvfitter.methods.ccf'),
    'template_fit': _Lazy('rvfitter.methods.template_fit'),
}


def get(name: str) -> Callable[..., MethodResult]:
    """
    The `measure` function of a registered method

    Parameters
    ----------
    name
        A key of `REGISTRY`

    Returns
    -------
    measure
        The method's measure function
    """
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f'unknown method {name!r}; choose from {sorted(REGISTRY)}') from None
