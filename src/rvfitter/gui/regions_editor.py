"""Interactive region editor: the spectrum with the auto-selected regions as draggable shaded spans.

Edits are expressed as ``config.regions.overrides``, a mapping from region name to either
``{"wmin": .., "wmax": ..}`` (move an automatic region, or add a user region; user regions also carry
``"kind"``) or ``{"remove": true}`` (drop an automatic region).
All wavelengths are vacuum Angstrom.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

log = logging.getLogger(__name__)

AUTO_COLOUR = '#DDAA33'
USER_COLOUR = '#004488'
GREY = '#777777'
USER_KIND = 'line'


@dataclass
class Span:
    name: str
    kind: str
    wmin: float
    wmax: float
    auto: bool                  # True if it came from regions.select
    orig: tuple[float, float] | None = None   # the automatic limits
    patch: Any = None
    label: Any = None


class RegionEditor(QWidget):
    """Matplotlib canvas with draggable spans plus Add/Remove/Reset buttons."""

    overridesChanged = Signal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.spans: dict[str, Span] = {}
        self.removed: dict[str, Span] = {}
        self.selected: str | None = None
        self._drag: tuple[str, str, float] | None = None
        self._counter = 0
        self.wave = np.array([])
        self.flux = np.array([])

        self.figure = Figure(figsize=(8, 3.5))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.info = QLabel('Select an object to edit its regions.')
        self.add_btn = QPushButton('Add region')
        self.add_btn.setCheckable(True)
        self.add_btn.setToolTip('Click, then drag across the spectrum to add a new region.')
        self.rm_btn = QPushButton('Remove selected')
        self.rm_btn.setToolTip('Remove the selected region (click a shaded span to select it).')
        self.reset_btn = QPushButton('Reset to automatic')
        self.reset_btn.setToolTip('Discard all edits and go back to the automatically selected regions.')
        bar = QHBoxLayout()
        for b in (self.add_btn, self.rm_btn, self.reset_btn):
            bar.addWidget(b)
        bar.addStretch(1)
        lay = QVBoxLayout(self)
        lay.addLayout(bar)
        lay.addWidget(self.info)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)

        self._selector: SpanSelector | None = None
        self.add_btn.toggled.connect(self._toggle_add)
        self.rm_btn.clicked.connect(lambda: self.remove_span(self.selected))
        self.reset_btn.clicked.connect(self.reset)
        self.canvas.mpl_connect('button_press_event', self._press)
        self.canvas.mpl_connect('motion_notify_event', self._motion)
        self.canvas.mpl_connect('button_release_event', self._release)
        self.canvas.mpl_connect('key_press_event', self._key)
        self._style_axes()

    def _toggle_add(self, on: bool) -> None:
        if self._selector is not None:
            self._selector.set_active(False)
            self._selector.disconnect_events()
            self._selector = None
        if on:
            self._selector = SpanSelector(self.ax, self._on_new_span, 'horizontal', useblit=False, interactive=False,
                                          props={'alpha': 0.25, 'facecolor': USER_COLOUR})

    # ---- data in -------------------------------------------------------------------------------------------
    def set_data(self, wave, flux, regions, overrides: dict | None = None) -> None:
        """Show a spectrum with automatic ``regions`` (objects with name, kind, wmin, wmax) and apply ``overrides``."""
        self.wave, self.flux = np.asarray(wave, float), np.asarray(flux, float)
        self.spans.clear()
        self.removed.clear()
        self.selected = None
        self.ax.clear()
        overrides = dict(overrides or {})
        for reg in regions:
            span = Span(reg.name, reg.kind, float(reg.wmin), float(reg.wmax), True, (float(reg.wmin), float(reg.wmax)))
            ov = overrides.get(reg.name)
            if ov and ov.get('remove'):
                self.removed[reg.name] = span
                continue
            if ov and 'wmin' in ov:
                span.wmin, span.wmax = float(ov['wmin']), float(ov['wmax'])
            self.spans[reg.name] = span
        for name, ov in overrides.items():
            if name not in self.spans and name not in self.removed and 'wmin' in ov:
                self.spans[name] = Span(name, ov.get('kind', USER_KIND), float(ov['wmin']), float(ov['wmax']), False)
        self._redraw()

    def reset(self) -> None:
        for span in list(self.removed.values()):
            self.spans[span.name] = span
        self.removed.clear()
        for name in [n for n, s in self.spans.items() if not s.auto]:
            del self.spans[name]
        for span in self.spans.values():
            if span.orig:
                span.wmin, span.wmax = span.orig
        self.selected = None
        self._redraw()
        self._emit()

    # ---- programmatic edits (also used by the mouse handlers and the tests) -------------------------------------
    def add_span(self, wmin: float, wmax: float, kind: str = USER_KIND) -> str:
        wmin, wmax = sorted((float(wmin), float(wmax)))
        self._counter += 1
        name = f'user_{self._counter}'
        while name in self.spans:
            self._counter += 1
            name = f'user_{self._counter}'
        self.spans[name] = Span(name, kind, wmin, wmax, False)
        self.selected = name
        self._redraw()
        self._emit()
        return name

    def remove_span(self, name: str | None) -> None:
        if name is None or name not in self.spans:
            return
        span = self.spans.pop(name)
        if span.auto:
            self.removed[name] = span
        if self.selected == name:
            self.selected = None
        self._redraw()
        self._emit()

    def move_span(self, name: str, wmin: float, wmax: float) -> None:
        span = self.spans[name]
        span.wmin, span.wmax = sorted((float(wmin), float(wmax)))
        self._redraw()
        self._emit()

    def overrides(self) -> dict[str, dict]:
        """The edits as a ``regions.overrides`` mapping (empty if nothing was changed)."""
        out: dict[str, dict] = {}
        for name in self.removed:
            out[name] = {'remove': True}
        for name, s in self.spans.items():
            if not s.auto:
                out[name] = {'kind': s.kind, 'wmin': s.wmin, 'wmax': s.wmax}
            elif s.orig and (not np.isclose(s.wmin, s.orig[0]) or not np.isclose(s.wmax, s.orig[1])):
                out[name] = {'wmin': s.wmin, 'wmax': s.wmax}
        return out

    def _emit(self) -> None:
        self.overridesChanged.emit(self.overrides())

    # ---- drawing -------------------------------------------------------------------------------------------
    def _style_axes(self) -> None:
        ax = self.ax
        ax.tick_params(which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        ax.set_xlabel('Wavelength [Å]')
        ax.set_ylabel('Flux')

    def _redraw(self) -> None:
        xlim = self.ax.get_xlim() if len(self.wave) and self.ax.lines else None
        self.ax.clear()
        self._style_axes()
        if len(self.wave):
            self.ax.step(self.wave, self.flux, where='mid', color='k', lw=0.6)
            if xlim is not None and xlim != (0.0, 1.0):
                self.ax.set_xlim(xlim)
        for span in self.spans.values():
            colour = AUTO_COLOUR if span.auto else USER_COLOUR
            span.patch = self.ax.axvspan(span.wmin, span.wmax, color=colour, alpha=0.35 if span.name != self.selected
                                         else 0.6, lw=2 if span.name == self.selected else 0)
            span.label = self.ax.annotate(span.name, (0.5 * (span.wmin + span.wmax), 1.0),
                                          xycoords=('data', 'axes fraction'), rotation=90, ha='center',
                                          va='bottom', fontsize=7, color=GREY, clip_on=True)
        n_auto = sum(s.auto for s in self.spans.values())
        self.info.setText(f'{len(self.spans)} regions ({n_auto} automatic, {len(self.spans) - n_auto} added); '
                          f'{len(self.removed)} removed. Drag a span or its edges to adjust it.')
        self.canvas.draw_idle()

    def _update_patch(self, span: Span) -> None:
        span.patch.set_x(span.wmin)
        span.patch.set_width(span.wmax - span.wmin)
        self.canvas.draw_idle()

    # ---- mouse ---------------------------------------------------------------------------------------------------
    def _navigating(self) -> bool:
        return bool(getattr(self.toolbar, 'mode', ''))

    def _hit(self, x: float) -> tuple[str, str] | None:
        lo, hi = self.ax.get_xlim()
        tol = 0.006 * (hi - lo)
        best = None
        for name, s in self.spans.items():
            if abs(x - s.wmin) <= tol:
                return name, 'left'
            if abs(x - s.wmax) <= tol:
                return name, 'right'
            if s.wmin <= x <= s.wmax:
                best = (name, 'move')
        return best

    def _press(self, event) -> None:
        if event.inaxes is not self.ax or self._navigating() or self.add_btn.isChecked() or event.button != 1:
            return
        hit = self._hit(event.xdata)
        if hit is None:
            self.selected = None
            self._redraw()
            return
        name, mode = hit
        self.selected = name
        self._drag = (name, mode, event.xdata)
        self._redraw()

    def _motion(self, event) -> None:
        if self._drag is None or event.inaxes is not self.ax or event.xdata is None:
            return
        name, mode, x0 = self._drag
        s = self.spans[name]
        if mode == 'left':
            s.wmin = min(event.xdata, s.wmax)
        elif mode == 'right':
            s.wmax = max(event.xdata, s.wmin)
        else:
            dx = event.xdata - x0
            s.wmin, s.wmax = s.wmin + dx, s.wmax + dx
            self._drag = (name, mode, event.xdata)
        self._update_patch(s)

    def _release(self, event) -> None:
        if self._drag is not None:
            self._drag = None
            self._redraw()
            self._emit()

    def _key(self, event) -> None:
        if event.key in ('delete', 'backspace'):
            self.remove_span(self.selected)

    def _on_new_span(self, xmin: float, xmax: float) -> None:
        if xmax - xmin > 0:
            self.add_span(xmin, xmax)
        self.add_btn.setChecked(False)
