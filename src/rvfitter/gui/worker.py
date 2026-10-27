"""Background threads: the pipeline run and generic one-shot jobs."""
from __future__ import annotations

import logging
import threading
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QThread, Signal

log = logging.getLogger(__name__)


class RunWorker(QThread):
    """Runs the backend pipeline. Any process pool lives inside ``backend.run``, so the UI never blocks."""

    result = Signal(object)        # one ObjectResult as soon as it is finished
    progress = Signal(int, int)    # done, total
    done = Signal(object, bool)    # list of results, cancelled
    failed = Signal(str)

    def __init__(self, backend, cfg, groups: dict[str, list[str]], n_procs: int, parent=None):
        super().__init__(parent)
        self.backend, self.cfg, self.groups, self.n_procs = backend, cfg, groups, n_procs
        self._cancel = threading.Event()

    def cancel(self) -> None:
        log.info('Cancel requested: stopping after the current object.')
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self) -> None:
        n_done = 0
        total = len(self.groups)

        def on_result(res: Any) -> None:
            nonlocal n_done
            n_done += 1
            self.result.emit(res)
            self.progress.emit(n_done, total)

        try:
            results = self.backend.run(self.cfg, self.groups, self.n_procs, on_result, self._cancel.is_set)
        except Exception as exc:
            log.error('Run failed: %s', exc)
            log.debug(traceback.format_exc())
            self.failed.emit(str(exc))
            return
        self.done.emit(results, self._cancel.is_set())


class JobWorker(QThread):
    """Run ``func()`` in a thread and report its return value or error."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, func: Callable[[], Any], parent=None):
        super().__init__(parent)
        self.func = func

    def run(self) -> None:
        try:
            self.succeeded.emit(self.func())
        except Exception as exc:
            log.error('%s', exc)
            log.debug(traceback.format_exc())
            self.failed.emit(str(exc))
