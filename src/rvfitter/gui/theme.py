"""Meaning-bearing colours over the native palette: blue is the primary action and adopted RV, gold needs
attention, rose is failed. The palette itself is never replaced."""
from __future__ import annotations

import functools

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPixmap

COLOURS = {  # role -> (light window, dark window)
    'primary': ('#004488', '#6699CC'),
    'flagged': ('#DDAA33', '#DDAA33'),
    'failed': ('#BB5566', '#EE99AA'),
}


def is_dark(palette: QPalette) -> bool:
    return palette.color(QPalette.ColorRole.Window).lightness() < 128


def colour(role: str, palette: QPalette) -> QColor:
    """The colour of ``role`` ('primary', 'flagged' or 'failed'), chosen from the palette's window lightness."""
    return QColor(COLOURS[role][is_dark(palette)])


def dot(role: str, palette: QPalette) -> QIcon:
    """A filled circle in the colour of ``role``."""
    return _dot(role, is_dark(palette))


@functools.cache
def _dot(role: str, dark: bool, size: int = 10) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(COLOURS[role][dark]))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.end()
    return QIcon(pix)


def tabular(font: QFont) -> QFont:
    """``font`` with tabular figures, where Qt supports it (6.7 and later)."""
    if hasattr(QFont, 'Tag'):
        font.setFeature(QFont.Tag('tnum'), 1)
    return font
