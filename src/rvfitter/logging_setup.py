"""Logging configuration: console handler on stdout and a full DEBUG log file."""
import logging
import sys
from pathlib import Path

__all__ = ['configure']

_NOISY = ('astropy', 'matplotlib', 'PIL', 'urllib3', 'numba', 'asyncio')


def configure(verbosity: int = 0, logfile: str | Path | None = None) -> None:
    """Configure the ``rvfitter`` logger, replacing earlier handlers.

    Console (stdout): INFO for ``verbosity`` 0, DEBUG for >0, WARNING for <0. ``logfile`` gets DEBUG with timestamps.
    """
    root = logging.getLogger('rvfitter')
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    level = logging.DEBUG if verbosity > 0 else logging.WARNING if verbosity < 0 else logging.INFO
    root.setLevel(logging.DEBUG)
    root.propagate = False
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(logging.Formatter('%(levelname)s %(message)s' if level <= logging.DEBUG else '%(message)s'))
    root.addHandler(console)
    if logfile is not None:
        logfile = Path(logfile)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)-7s %(name)s: %(message)s'))
        root.addHandler(fh)
    # Only touch loggers that already exist: creating e.g. 'astropy' here would stop astropy installing its own
    # logger class when it is imported later.
    for name in _NOISY:
        if name in logging.root.manager.loggerDict:
            logging.getLogger(name).setLevel(logging.WARNING)
