"""rvfitter: radial velocities of brown dwarfs from reduced 1D spectra."""
from importlib import metadata as _metadata

try:
    __version__ = _metadata.version('rvfitter')
except _metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = '2.0.0.dev0'

# Public API placeholder: populated as the pipeline modules are implemented.
__all__ = ['__version__']
