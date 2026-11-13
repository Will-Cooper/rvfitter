"""
RV measurement methods: ``REGISTRY[name](spectrum, regions, cfg, *, executor=None, rng) -> MethodResult``, keyed by
the name used in the config and the results table (``rv_<name>``).
"""
from . import ccf, linecentre, telluric_ref, template_fit
from .template_fit import estimate_resolution

__all__ = ['REGISTRY', 'estimate_resolution']

REGISTRY = {'linecentre': linecentre.measure, 'ccf': ccf.measure, 'template_fit': template_fit.measure,
            'telluric_ref': telluric_ref.measure}
