"""
Logging configuration: console handler on stdout and a full DEBUG log file.

The console (and the GUI's log pane) shows an identical INFO-or-higher message once per session (`OnceFilter`): a
file read at load and again for the run, or by several steps, repeats its warnings word for word. A message that
differs in any character, e.g. another file's name, is always shown, and the log file keeps every record.
"""
import contextlib
import logging
import sys
from pathlib import Path

__all__ = ['configure', 'OnceFilter', 'demoted']


class OnceFilter(logging.Filter):
    """Handler filter: each distinct (level, message) of INFO or WARNING once; DEBUG and ERROR or worse always (a
    repeated failure must stay visible). `reset` forgets what was seen (e.g. when the log pane is cleared)."""

    def __init__(self):
        super().__init__()
        self._seen: set[tuple[int, str]] = set()

    def reset(self) -> None:
        self._seen.clear()

    def filter(self, record: logging.LogRecord) -> bool:
        if not logging.INFO <= record.levelno < logging.ERROR:
            return True
        key = (record.levelno, record.getMessage())
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


class _Demote(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno not in (logging.INFO, logging.WARNING):
            return True
        if not logging.getLogger(record.name).isEnabledFor(logging.DEBUG):
            return False
        record.levelno, record.levelname = logging.DEBUG, 'DEBUG'
        return True


@contextlib.contextmanager
def demoted():
    """
    Within the block, INFO and WARNING records of the rvfitter loggers are logged at DEBUG (ERROR is unchanged): for
    reruns (e.g. the systematics' variants) that would repeat the main run's summaries and warnings, or warn about a
    variant the main result does not use. The caller reports a variant that fails.
    """
    # chisle: process-wide while active; a per-thread switch (contextvar) if runs ever overlap in threads
    loggers = [lg for name, lg in logging.root.manager.loggerDict.items()
               if name.startswith('rvfitter') and isinstance(lg, logging.Logger)]
    demote = _Demote()
    for lg in loggers:
        lg.addFilter(demote)
    try:
        yield
    finally:
        for lg in loggers:
            lg.removeFilter(demote)

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
    console.addFilter(OnceFilter())
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
