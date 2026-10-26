"""
Shared set-up for the legacy (flat-module) tests: put the repository root on the import path
and force a non-interactive matplotlib backend.
"""
import os
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# splat is only used by fullrv for spectral indices (not RVs) and is not installed: stub it if missing
try:
    import splat  # noqa: F401
except ImportError:
    import types
    _splat = types.ModuleType('splat')
    _splat.Spectrum = lambda *a, **k: None
    _splat.measureIndexSet = lambda *a, **k: {}
    sys.modules['splat'] = _splat
