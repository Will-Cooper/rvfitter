"""Diagnostics of the selected object: its adopted RV, barycentric correction and the `rvfitter.plotting` figures."""
from __future__ import annotations

import math
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QGroupBox, QLabel, QTabWidget, QVBoxLayout, QWidget

from .. import plotting
from . import theme

EMPTY_TEXT = 'Select an object to see its fits.'


def rv_text(rv: float | None, err: float | None = None) -> str:
    """'36.6 ± 21.7' (or '36.6' without an error); empty when there is no RV."""
    if rv is None or math.isnan(rv):
        return ''
    if err is None or math.isnan(err):
        return f'{rv:.1f}'
    return f'{rv:.1f} ± {err:.1f}'


def _coloured(label: QLabel, role: str) -> None:
    pal = label.palette()
    pal.setColor(QPalette.ColorRole.WindowText, theme.colour(role, pal))
    label.setPalette(pal)


class DiagnosticsPanel(QWidget):
    """The adopted RV, barycentric correction and diagnostic figures of the selected object, one tab each."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.empty = QLabel(EMPTY_TEXT, alignment=Qt.AlignmentFlag.AlignCenter)
        self.empty.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        self.name = QLabel()
        self.readout = QLabel()
        font = theme.tabular(self.readout.font())
        font.setPointSizeF(2 * font.pointSizeF())
        self.readout.setFont(font)
        self.readout.setToolTip('The adopted radial velocity and its error.')
        self.note = QLabel()
        self.note.setWordWrap(True)
        box = QGroupBox('Barycentric correction')
        box.setToolTip('The velocity correction applied to this object, where it came from and its frame.')
        self.bary = QLabel()
        self.bary.setWordWrap(True)
        QVBoxLayout(box).addWidget(self.bary)
        self.figures = QTabWidget()
        self.content = QWidget()
        cl = QVBoxLayout(self.content)
        cl.setContentsMargins(0, 0, 0, 0)
        for w in (self.name, self.readout, self.note, box):
            cl.addWidget(w)
        cl.addWidget(self.figures, 1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.empty, 1)
        lay.addWidget(self.content, 1)
        self.clear()

    def clear(self) -> None:
        self.empty.setVisible(True)
        self.content.setVisible(False)
        self._clear_figures()

    def _clear_figures(self) -> None:
        while self.figures.count():
            page = self.figures.widget(0)
            self.figures.removeTab(0)
            page.deleteLater()

    @staticmethod
    def bary_text(result: Any) -> str:
        b = result.barycentric
        if not b:
            return 'No barycentric information recorded.'
        kms = b.get('applied_kms')
        value = 'n/a' if kms is None else f'{kms:+.3f} km/s'
        return f'Value applied: {value}    Source: {b.get("source", "?")}    Frame: {b.get("frame", "?")}'

    def show_result(self, result: Any, rv: float | None = None, err: float | None = None,
                    note: str = '') -> None:
        """Show ``result``; ``rv`` and ``err`` are its adopted RV in km/s (none: no RV could be measured)."""
        self.empty.setVisible(False)
        self.content.setVisible(True)
        self.name.setText(result.object)
        text = rv_text(rv, err)
        self.readout.setText(f'{text} km s⁻¹' if text else 'No RV measured')
        _coloured(self.readout, 'primary' if text else 'failed')
        self.note.setText(note)
        self.note.setVisible(bool(note))
        _coloured(self.note, 'flagged')
        self.bary.setText(self.bary_text(result))
        self._clear_figures()
        for name, fig in plotting.make_all(result).items():
            canvas = FigureCanvasQTAgg(fig)
            page = QWidget()
            lay = QVBoxLayout(page)
            lay.addWidget(NavigationToolbar2QT(canvas, page))
            lay.addWidget(canvas, 1)
            self.figures.addTab(page, name.replace('_', ' '))
