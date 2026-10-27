"""Interactive region editor: the spectrum with the auto-selected regions as draggable shaded spans.

Edits are expressed as ``config.regions.overrides``, a mapping from method name to a list of
``{"name", "kind", "wmin", "wmax"[, "centre", "continuum"]}`` (adds or replaces a region) or
``{"name", "remove": true}`` (drops one), as applied by ``regions.select``.
All wavelengths are vacuum Angstrom.
"""
from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

log = logging.getLogger(__name__)

AUTO_COLOUR = '#DDAA33'
SELECT_COLOUR = '#004488'
GREY = '#777777'
USER_KIND = 'line'

Key = tuple[str, str]   # (method, region name)


@dataclass
class Span:
    method: str
    name: str
    kind: str
    wmin: float
    wmax: float
    auto: bool                                 # True if it came from the automatic selection
    orig: tuple[float, float] | None = None    # the automatic limits
    extra: dict | None = None                  # centre / continuum, carried through to the overrides
    patch: Any = None
    label: Any = None

    @property
    def key(self) -> Key:
        return (self.method, self.name)


def _style_context():
    """``plotting.style_context()`` when available, else a no-op (never touches global rcParams)."""
    try:
        from rvfitter import plotting
        return plotting.style_context()
    except Exception:
        return contextlib.nullcontext()


class RegionEditor(QWidget):
    """Matplotlib canvas with draggable spans plus Add/Remove/Reset buttons."""

    overridesChanged = Signal(dict)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.spans: dict[Key, Span] = {}
        self.removed: dict[Key, Span] = {}
        self.selected: Key | None = None
        self._drag: tuple[Key, str, float] | None = None
        self._counter = 0
        self.wave = np.array([])
        self.flux = np.array([])

        self.figure = Figure(figsize=(8, 3.5))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.info = QLabel('Select an object to edit its regions.')
        self.method_box = QComboBox()
        self.method_box.setToolTip('The method that a newly added region applies to.')
        self.add_btn = QPushButton('Add region')
        self.add_btn.setCheckable(True)
        self.add_btn.setToolTip('Click, then drag across the spectrum to add a new region.')
        self.rm_btn = QPushButton('Remove selected')
        self.rm_btn.setToolTip('Remove the selected region (click a shaded span to select it).')
        self.reset_btn = QPushButton('Reset to automatic')
        self.reset_btn.setToolTip('Discard all edits and go back to the automatically selected regions.')
        bar = QHBoxLayout()
        for b in (self.add_btn, QLabel('for method'), self.method_box, self.rm_btn, self.reset_btn):
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
                                          props={'alpha': 0.25, 'facecolor': SELECT_COLOUR})

    # ---- data in -------------------------------------------------------------------------------------------
    def set_data(self, wave, flux, auto: dict[str, list], effective: dict[str, list] | None = None) -> None:
        """Show a spectrum with the automatic regions per method and the regions in force (with overrides)."""
        self.wave = np.asarray(wave, float)
        flux = np.asarray(flux, float)
        med = np.nanmedian(flux) if flux.size else 1.0
        self.flux = flux / med if np.isfinite(med) and med != 0 else flux
        self.spans.clear()
        self.removed.clear()
        self.selected = None
        self.ax.clear()
        effective = auto if effective is None else effective
        auto_keys = {}
        for method, regs in auto.items():
            for r in regs:
                auto_keys[(method, r.name)] = r
        eff_keys = set()
        for method, regs in effective.items():
            for r in regs:
                key = (method, r.name)
                eff_keys.add(key)
                base = auto_keys.get(key)
                extra = {}
                if getattr(r, 'centre', None) is not None:
                    extra['centre'] = r.centre
                if getattr(r, 'continuum', None):
                    extra['continuum'] = [list(c) for c in r.continuum]
                self.spans[key] = Span(method, r.name, r.kind, float(r.wmin), float(r.wmax), base is not None,
                                       (float(base.wmin), float(base.wmax)) if base is not None else None, extra)
        for key, r in auto_keys.items():
            if key not in eff_keys:
                self.removed[key] = Span(key[0], r.name, r.kind, float(r.wmin), float(r.wmax), True,
                                         (float(r.wmin), float(r.wmax)))
        methods = sorted({m for m, _ in self.spans} | set(auto) | set(effective))
        self.method_box.clear()
        self.method_box.addItems(methods or ['linecentre'])
        self._redraw()

    def reset(self) -> None:
        for span in list(self.removed.values()):
            self.spans[span.key] = span
        self.removed.clear()
        for key in [k for k, s in self.spans.items() if not s.auto]:
            del self.spans[key]
        for span in self.spans.values():
            if span.orig:
                span.wmin, span.wmax = span.orig
        self.selected = None
        self._redraw()
        self._emit()

    # ---- programmatic edits (also used by the mouse handlers and the tests) -------------------------------------
    def add_span(self, wmin: float, wmax: float, kind: str = USER_KIND, method: str | None = None) -> Key:
        wmin, wmax = sorted((float(wmin), float(wmax)))
        method = method or self.method_box.currentText() or 'linecentre'
        self._counter += 1
        while (method, f'user_{self._counter}') in self.spans:
            self._counter += 1
        key = (method, f'user_{self._counter}')
        self.spans[key] = Span(method, key[1], kind, wmin, wmax, False)
        self.selected = key
        self._redraw()
        self._emit()
        return key

    def remove_span(self, key: Key | None) -> None:
        if key is None or key not in self.spans:
            return
        span = self.spans.pop(key)
        if span.auto:
            self.removed[key] = span
        if self.selected == key:
            self.selected = None
        self._redraw()
        self._emit()

    def move_span(self, key: Key, wmin: float, wmax: float) -> None:
        span = self.spans[key]
        span.wmin, span.wmax = sorted((float(wmin), float(wmax)))
        self._redraw()
        self._emit()

    def overrides(self) -> dict[str, list[dict]]:
        """The edits as a ``regions.overrides`` mapping (empty if nothing was changed)."""
        out: dict[str, list[dict]] = {}
        for (method, name) in self.removed:
            out.setdefault(method, []).append({'name': name, 'remove': True})
        for (method, name), s in self.spans.items():
            changed = s.orig is None or not (np.isclose(s.wmin, s.orig[0]) and np.isclose(s.wmax, s.orig[1]))
            if changed:
                entry = {'name': name, 'kind': s.kind, 'wmin': s.wmin, 'wmax': s.wmax}
                entry.update(s.extra or {})
                out.setdefault(method, []).append(entry)
        return out

    def _emit(self) -> None:
        self.overridesChanged.emit(self.overrides())

    # ---- drawing -------------------------------------------------------------------------------------------
    def _style_axes(self) -> None:
        ax = self.ax
        ax.minorticks_on()
        ax.tick_params(which='both', direction='in', top=True, right=True, bottom=True, left=True)
        ax.set_xlabel('Wavelength [Å]')
        ax.set_ylabel(r'Normalised Flux [$F_\lambda$]')

    def _redraw(self) -> None:
        xlim = self.ax.get_xlim() if len(self.wave) and self.ax.lines else None
        with _style_context():
            self.ax.clear()
            self._style_axes()
            if len(self.wave):
                self.ax.step(self.wave, self.flux, where='mid', color='k', lw=0.6)
                if xlim is not None:
                    self.ax.set_xlim(xlim)
            for span in self.spans.values():
                sel = span.key == self.selected
                colour = SELECT_COLOUR if sel else AUTO_COLOUR
                span.patch = self.ax.axvspan(span.wmin, span.wmax, facecolor=colour, edgecolor=colour,
                                             alpha=0.5 if sel else 0.35, lw=1.5, ls='-' if span.auto else '--')
                span.label = self.ax.annotate(span.name, (0.5 * (span.wmin + span.wmax), 1.0),
                                              xycoords=('data', 'axes fraction'), xytext=(0, 3),
                                              textcoords='offset points', rotation=90, ha='center', va='bottom',
                                              fontsize=7, color=GREY, annotation_clip=False)
            self.canvas.draw_idle()
        n_auto = sum(s.auto for s in self.spans.values())
        self.info.setText(f'{len(self.spans)} regions ({n_auto} automatic, {len(self.spans) - n_auto} added); '
                          f'{len(self.removed)} removed. Drag a span or its edges to adjust it.')

    def _update_patch(self, span: Span) -> None:
        span.patch.set_x(span.wmin)
        span.patch.set_width(span.wmax - span.wmin)
        self.canvas.draw_idle()

    # ---- mouse ---------------------------------------------------------------------------------------------------
    def _navigating(self) -> bool:
        return bool(getattr(self.toolbar, 'mode', ''))

    def _hit(self, x: float) -> tuple[Key, str] | None:
        lo, hi = self.ax.get_xlim()
        tol = 0.006 * (hi - lo)
        best = None
        for key, s in self.spans.items():
            if abs(x - s.wmin) <= tol:
                return key, 'left'
            if abs(x - s.wmax) <= tol:
                return key, 'right'
            if s.wmin <= x <= s.wmax:
                best = (key, 'move')
        return best

    def _press(self, event) -> None:
        if event.inaxes is not self.ax or self._navigating() or self.add_btn.isChecked() or event.button != 1:
            return
        hit = self._hit(event.xdata)
        if hit is None:
            self.selected = None
            self._redraw()
            return
        key, mode = hit
        self.selected = key
        self._drag = (key, mode, event.xdata)
        self._redraw()

    def _motion(self, event) -> None:
        if self._drag is None or event.inaxes is not self.ax or event.xdata is None:
            return
        key, mode, x0 = self._drag
        s = self.spans[key]
        if mode == 'left':
            s.wmin = min(event.xdata, s.wmax)
        elif mode == 'right':
            s.wmax = max(event.xdata, s.wmin)
        else:
            dx = event.xdata - x0
            s.wmin, s.wmax = s.wmin + dx, s.wmax + dx
            self._drag = (key, mode, event.xdata)
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
