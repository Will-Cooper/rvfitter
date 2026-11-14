"""Diagnostics of the selected object: its barycentric correction and the `rvfitter.plotting` figures."""
from __future__ import annotations

from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from PySide6.QtWidgets import QGroupBox, QLabel, QTabWidget, QVBoxLayout, QWidget

from .. import plotting


class DiagnosticsPanel(QWidget):
    """The barycentric correction and the diagnostic figures of the selected object, one tab each."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        box = QGroupBox('Barycentric correction')
        box.setToolTip('The velocity correction applied to this object, where it came from and its frame.')
        self.bary = QLabel()
        self.bary.setWordWrap(True)
        QVBoxLayout(box).addWidget(self.bary)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(box)
        self.figures = QTabWidget()
        lay.addWidget(self.figures, 1)
        self.clear()

    def clear(self) -> None:
        self.bary.setText('No object selected.')
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

    def show_result(self, result: Any) -> None:
        self.bary.setText(f'{result.object}:  {self.bary_text(result)}')
        self._clear_figures()
        for name, fig in plotting.make_all(result).items():
            canvas = FigureCanvasQTAgg(fig)
            page = QWidget()
            lay = QVBoxLayout(page)
            lay.addWidget(NavigationToolbar2QT(canvas, page))
            lay.addWidget(canvas, 1)
            self.figures.addTab(page, name.replace('_', ' '))
