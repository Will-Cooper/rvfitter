"""Choice of the Qt platform plugin (Linux Wayland/X11), shared by ``rvfitter gui`` and the frozen launcher."""
from __future__ import annotations

import ctypes.util
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path

log = logging.getLogger(__name__)

XCB_CURSOR_HINT = ('The Qt xcb platform plugin needs the XCB cursor library: install libxcb-cursor0 '
                   '(Debian/Ubuntu: sudo apt install libxcb-cursor0; Fedora: xcb-util-cursor; '
                   'Arch: xcb-util-cursor; conda: xcb-util-cursor)')


def plugin_names(platforms_dir: Path) -> list[str]:
    """Platform plugins in a Qt ``plugins/platforms`` directory, e.g. ``['offscreen', 'wayland', 'xcb']``."""
    pat = 'q*' if sys.platform == 'win32' else 'libq*'
    return sorted(p.stem.removeprefix('libq').removeprefix('q') for p in platforms_dir.glob(pat)) \
        if platforms_dir.is_dir() else []


def platforms_dir() -> Path:
    from PySide6.QtCore import QLibraryInfo
    return Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)) / 'platforms'


def choose_platform(env: Mapping[str, str], plugins: list[str], system: str = sys.platform) -> str | None:
    """``QT_QPA_PLATFORM`` to set, or ``None`` to leave Qt alone (non-Linux, or the user already chose).

    On Wayland sessions: ``wayland;xcb`` if the wayland plugin exists, else ``xcb`` (XWayland).
    """
    if not system.startswith('linux') or env.get('QT_QPA_PLATFORM'):
        return None
    if env.get('XDG_SESSION_TYPE') != 'wayland':
        return None
    return 'wayland;xcb' if any(p.startswith('wayland') for p in plugins) else 'xcb'


def configure_platform() -> None:
    """Apply :func:`choose_platform` to ``os.environ``; call before the ``QApplication`` exists."""
    choice = choose_platform(os.environ, plugin_names(platforms_dir()))
    if choice is None:
        return
    os.environ['QT_QPA_PLATFORM'] = choice
    if choice == 'xcb':
        log.info('No Qt Wayland plugin in this PySide6: running through XWayland (QT_QPA_PLATFORM=xcb)')
    else:  # a harmless Qt/GNOME text-input warning floods the terminal on Wayland; user rules win
        os.environ.setdefault('QT_LOGGING_RULES', 'qt.qpa.wayland.textinput=false')


def xcb_cursor_missing(env: Mapping[str, str], plugins: list[str], system: str = sys.platform) -> bool:
    """True if Qt would use the xcb plugin but ``libxcb-cursor`` cannot be found (Qt 6 then aborts at start-up)."""
    wanted = env.get('QT_QPA_PLATFORM', '')
    return (system.startswith('linux') and 'xcb' in plugins and wanted in ('', 'xcb')
            and ctypes.util.find_library('xcb-cursor') is None)
