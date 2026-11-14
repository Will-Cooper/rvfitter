"""GUI application entry point (``rvfitter gui``)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

INSTALL_HINT = 'The rvfitter GUI needs PySide6. Install it with:  pip install "rvfitter[gui]"'


def _qt_available() -> bool:
    try:
        import PySide6  # noqa: F401
    except ImportError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    """Launch the GUI. Optional ``argv`` entries are spectra, folders or archives to open at start-up."""
    if not _qt_available():
        log.error(INSTALL_HINT)
        sys.stderr.write(INSTALL_HINT + '\n')
        return 1
    from PySide6.QtWidgets import QApplication

    args = sys.argv[1:] if argv is None else list(argv)
    app = QApplication.instance() or QApplication(['rvfitter'])
    app.setOrganizationName('rvfitter')
    app.setApplicationName('rvfitter')
    from .window import MainWindow
    try:
        win = MainWindow()
    except Exception as exc:
        log.error('Could not start the GUI: %s', exc)
        sys.stderr.write(f'Could not start the GUI: {exc}\n')
        return 1
    win.show()
    paths = [Path(a) for a in args if Path(a).exists()]
    if paths:
        win.load_paths(paths)
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
