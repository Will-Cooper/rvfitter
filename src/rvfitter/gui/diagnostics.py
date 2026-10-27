"""Diagnostic tabs: embedded matplotlib canvases fed by ``plotting.*`` and the barycentric info box."""
from __future__ import annotations

import logging
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QGroupBox, QLabel, QStackedLayout, QTabWidget, QVBoxLayout, QWidget

from .backend import PLOT_KINDS

log = logging.getLogger(__name__)

TAB_TITLES = {'line_fits': 'Line fits', 'ccf': 'CCF', 'corner': 'Corner', 'adopted': 'Adopted RV'}


class FigureView(QWidget):
    """Shows a matplotlib Figure, or a text placeholder when there is none."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.canvas: FigureCanvasQTAgg | None = None
        self.toolbar: NavigationToolbar2QT | None = None
        self.placeholder = QLabel('Nothing to show yet.')
        self.placeholder.setWordWrap(True)
        self.placeholder.setStyleSheet('color: #666666; padding: 12px;')
        self._plot = QWidget()
        self._plot_lay = QVBoxLayout(self._plot)
        self._plot_lay.setContentsMargins(0, 0, 0, 0)
        self._stack = QStackedLayout(self)
        self._stack.addWidget(self.placeholder)
        self._stack.addWidget(self._plot)

    def show_message(self, text: str) -> None:
        self.placeholder.setText(text)
        self._stack.setCurrentWidget(self.placeholder)

    def show_figure(self, fig: Figure) -> None:
        self._clear_canvas()
        self.canvas = FigureCanvasQTAgg(fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self._plot_lay.addWidget(self.toolbar)
        self._plot_lay.addWidget(self.canvas)
        self.canvas.draw_idle()
        self._stack.setCurrentWidget(self._plot)

    def _clear_canvas(self) -> None:
        for w in (self.toolbar, self.canvas):
            if w is not None:
                self._plot_lay.removeWidget(w)
                w.deleteLater()
        self.canvas = self.toolbar = None


class DiagnosticsTabs(QWidget):
    """Line-fit, CCF, corner and adopted-RV tabs for one object, plus a barycentric info box."""

    def __init__(self, backend, parent: QWidget | None = None):
        super().__init__(parent)
        self.backend = backend
        self.views: dict[str, FigureView] = {}
        self.tabs = QTabWidget()
        for kind, title in TAB_TITLES.items():
            view = FigureView()
            self.views[kind] = view
            self.tabs.addTab(view, title)
        box = QGroupBox('Barycentric correction')
        box.setToolTip('The velocity correction applied to this object, where it came from and its frame.')
        self.bary = QLabel('No object selected.')
        self.bary.setWordWrap(True)
        QVBoxLayout(box).addWidget(self.bary)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(box)
        lay.addWidget(self.tabs, 1)
        self.clear()

    def clear(self) -> None:
        self.bary.setText('No object selected.')
        for view in self.views.values():
            view.show_message('Run an analysis and select an object to see its diagnostics.')

    @staticmethod
    def bary_text(result: Any) -> str:
        b = getattr(result, 'barycentric', None) or {}
        if not b:
            return 'No barycentric information recorded.'
        kms = b.get('applied_kms')
        value = 'n/a' if kms is None else f'{kms:+.3f} km/s'
        return f'Value applied: {value}    Source: {b.get("source", "?")}    Frame: {b.get("frame", "?")}'

    def show_result(self, result: Any) -> None:
        self.bary.setText(f'{getattr(result, "object", "")}:  {self.bary_text(result)}')
        for kind, view in self.views.items():
            fig = None
            try:
                fig = self.backend.plot(kind, result)
            except Exception as exc:  # a plotting bug must never take the GUI down
                log.warning('Plot %s failed: %s', kind, exc)
            if fig is None:
                func = PLOT_KINDS[kind]
                view.show_message(f'This diagnostic is not available (plotting.{func} is missing or failed).')
            else:
                view.show_figure(fig)
