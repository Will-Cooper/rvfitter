"""The Setup tab: the few settings most runs need, filled in from the spectra, each saying where its value came from."""
from __future__ import annotations

import copy
import dataclasses
import typing
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..config import METHOD_LABELS, method_advice, regime_for
from . import theme
from .forms import METHOD_TIPS, TIPS, Editor, make_editor

# (config path, label); the editors are the Advanced form's (`forms.make_editor`), bound to the same fields
FIELDS = (
    ('input.resolution', 'Resolving power'),
    ('input.medium', 'Wavelengths'),
    ('methods.enabled', 'Methods'),
    ('velocity_frame.correction', 'Velocity correction'),
    ('calibration.wavelength_correction', 'Wavelength fix'),
    ('templates.grid', 'Templates'),
)
SOURCE_TEXT = {'header': 'from header', 'estimated': 'estimated: check', 'implausible': 'estimate out of range: type R',
               'default': 'default', 'user': 'set by you'}
SOURCE_ROLE = {'estimated': 'flagged', 'implausible': 'failed'}  # theme colour roles of the tags that need a look
R_PATH = 'input.resolution'
GRID_PATH = 'templates.grid'
METHODS_PATH = 'methods.enabled'
BAND_TEXT = {'optical': 'optical', 'nir': 'near-infrared', 'mixed': 'optical and near-infrared'}
NO_DATA = 'Open spectra to fill these in from the data.'


def _field(cls: type, path: str) -> tuple[Any, dataclasses.Field]:
    """Type hint and dataclass field of a dotted config path"""
    *parents, leaf = path.split('.')
    for p in parents:
        cls = typing.get_type_hints(cls)[p]
    return typing.get_type_hints(cls)[leaf], next(f for f in dataclasses.fields(cls) if f.name == leaf)


def _get(cfg: Any, path: str) -> Any:
    section, leaf = path.split('.')
    return getattr(getattr(cfg, section), leaf)


def _set(cfg: Any, path: str, value: Any) -> None:
    section, leaf = path.split('.')
    setattr(getattr(cfg, section), leaf, value)


class SetupPanel(QWidget):
    """
    The common settings, bound to ``cfg`` (the one Config of the session; `get_config` returns a copy with the boxes
    applied). A box the user edits is 'set by you' and `autofill` no longer changes it.
    """

    advancedRequested = Signal()

    def __init__(self, cfg: Any, template_status: Callable[[Any], dict], parent: QWidget | None = None):
        super().__init__(parent)
        self._template_status = template_status  # Backend.template_status
        self.grid_status = QLabel()
        self.cfg = copy.deepcopy(cfg)
        self.editors: dict[str, Editor] = {}
        self.tags: dict[str, QLabel] = {}
        self.sources: dict[str, str] = {}
        self.details: dict[str, str] = {}  # extra tag text, e.g. the medium the header gives
        self._auto_options = copy.deepcopy(cfg.methods.options)  # the method options auto-fill last set
        self.detected = QLabel(NO_DATA)
        self.detected.setWordWrap(True)
        self.advice: dict[str, Any] = {}  # method -> config.MethodAdvice for the detected spectrum
        self._detected: tuple[Any, str] | None = None  # (spectrum, where its R came from) of the advice
        self.recommended = QLabel()  # which methods suit the detected R and band
        self.method_note = QLabel()  # why a ticked method is not recommended
        self.method_note.setWordWrap(True)
        form = QFormLayout()
        for path, label in FIELDS:
            tp, f = _field(type(cfg), path)
            ed = make_editor(tp, f, path, TIPS)
            ed.widget.setToolTip(TIPS[path])
            tag = QLabel()
            row = QHBoxLayout()
            row.addWidget(ed.widget)
            row.addWidget(tag)
            if path == GRID_PATH:
                row.addWidget(self.grid_status)
            if path == METHODS_PATH:
                row.addWidget(self.method_note, 1)
                for box in ed.boxes.values():
                    box.toggled.connect(self._show_method_note)
            row.addStretch(1)
            form.addRow(label, row)
            self.editors[path], self.tags[path] = ed, tag
            self._watch(ed, path)
        self.advanced_btn = QPushButton('Advanced settings…')
        self.advanced_btn.setToolTip('Every setting, including those not shown here.')
        self.advanced_btn.clicked.connect(self.advancedRequested)
        lay = QVBoxLayout(self)
        lay.addWidget(self.detected)
        lay.addWidget(self.recommended)
        lay.addLayout(form)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(self.advanced_btn)
        lay.addLayout(btn_row)
        lay.addStretch(1)
        self._load(self.cfg, {path: 'default' for path, _ in FIELDS})

    def _watch(self, ed: Editor, path: str) -> None:
        """Mark ``path`` 'set by you' on any user edit (signals that programmatic changes do not emit)"""
        w = ed.widget
        for edit in [w] * isinstance(w, QLineEdit) + w.findChildren(QLineEdit):
            edit.textEdited.connect(lambda _t, p=path: self._mark_user(p))
        for combo in [w] * isinstance(w, QComboBox) + w.findChildren(QComboBox):
            combo.activated.connect(lambda _i, p=path: self._mark_user(p))
        for btn in w.findChildren(QAbstractButton):
            btn.clicked.connect(lambda _c=False, p=path: self._mark_user(p))

    def _mark_user(self, path: str) -> None:
        self.sources[path] = 'user'
        self._show_tag(path)
        if path == GRID_PATH:
            self.show_grid_status()
        if path == R_PATH and self._detected:
            try:
                self._show_advice(*self._detected)
            except ValueError:  # not (yet) a number: keep the advice for the last one
                pass

    def show_grid_status(self) -> None:
        """Say whether each selected grid is present locally and over which Teff range"""
        try:
            status = self._template_status(self.get_config())
        except FileNotFoundError as exc:  # a templates.local_dir that is not a usable grid
            self.grid_status.setText(f'not usable: {exc}')
            return
        parts = [(f'{g}: ' if len(status) > 1 else '') + (f'downloaded, {t[0]:.0f}–{t[1]:.0f} K' if t else
                                                          'not downloaded: downloaded when you press Measure RVs')
                 for g, t in status.items()]
        self.grid_status.setText('; '.join(parts))

    def _show_tag(self, path: str) -> None:
        src, tag = self.sources[path], self.tags[path]
        tag.setText(SOURCE_TEXT[src] + (f' ({self.details[path]})' if self.details.get(path) and src != 'user'
                                        else ''))
        role = SOURCE_ROLE.get(src)
        tag.setStyleSheet(f'color: {theme.colour(role, tag.palette()).name()}' if role else '')

    def _load(self, cfg: Any, sources: dict[str, str]) -> None:
        self.cfg = copy.deepcopy(cfg)
        for path, ed in self.editors.items():
            ed.set(_get(self.cfg, path))
            self.sources[path] = sources[path]
            self._show_tag(path)
        self.show_grid_status()
        if self._detected:  # programmatic sets emit no edit signal: the advice follows the new R and methods here
            self._show_advice(*self._detected)

    def set_config(self, cfg: Any) -> None:
        """Show ``cfg`` (opened settings, the Advanced dialog): boxes whose value changes become 'set by you'"""
        current = self.get_config()
        self._load(cfg, {p: 'user' if _get(cfg, p) != _get(current, p) else self.sources[p] for p in self.editors})

    def get_config(self) -> Any:
        """A copy of the session config with the boxes applied"""
        for path, ed in self.editors.items():
            _set(self.cfg, path, ed.get())
        return copy.deepcopy(self.cfg)

    def autofill(self, name: str, spectrum: Any, defaults: Any, description: str) -> None:
        """
        Fill the boxes not set by the user from the first spectrum of object ``name`` and its regime ``defaults``
        (`config.default_for`). The R box only shows the spectrum's R (``input.resolution`` stays empty, so every
        spectrum keeps its own); the medium box stays 'auto' (each file's header is read).
        """
        meta = spectrum.meta
        r_src = meta.get('resolution_source', 'header')
        self.detected.setText(f'Detected for {name}: {description}')
        if self.sources[R_PATH] != 'user':
            src = 'implausible' if meta.get('resolution_implausible') else \
                'estimated' if r_src == 'estimated' else 'header'
            self.editors[R_PATH].set(None)
            self.editors[R_PATH].edit.setPlaceholderText(f'{spectrum.resolution:.0f}')
            self.sources[R_PATH] = src
            self._show_tag(R_PATH)
        if self.sources['input.medium'] != 'user':
            self.sources['input.medium'] = 'header' if str(meta.get('medium_source', '')).startswith('header') \
                else 'default'
            self.details['input.medium'] = meta.get('medium_in', '')
            self._show_tag('input.medium')
        if self.sources['methods.enabled'] != 'user':
            self.editors['methods.enabled'].set(defaults.methods.enabled)
            if self.cfg.methods.options == self._auto_options:  # options from settings or Advanced are kept
                self.cfg.methods.options = copy.deepcopy(defaults.methods.options)
                self._auto_options = copy.deepcopy(defaults.methods.options)
            self.sources['methods.enabled'] = 'default'
            self._show_tag('methods.enabled')
        self._show_advice(spectrum, r_src)

    def _show_advice(self, spectrum: Any, r_src: str) -> None:
        """Say which methods suit the R (the box's, else the spectrum's) and band, and why on each method's box"""
        self._detected = spectrum, r_src
        R = self.editors[R_PATH].get() if self.sources[R_PATH] == 'user' else None
        estimated = R is None and r_src == 'estimated'
        R = spectrum.resolution if R is None else R
        regime, band = regime_for(R, (spectrum.wmin, spectrum.wmax))
        self.advice = method_advice(regime)
        names = [METHOD_LABELS[m] for m, a in self.advice.items() if a.recommended]
        r_text = 'unknown R' if R is None else f'R {R:.0f}' + (' (estimated)' if estimated else '')
        self.recommended.setText(f'Recommended for {r_text}, {BAND_TEXT[band]}: {", ".join(names)}. '
                                 'Hover over a method for why.')
        for m, box in self.editors[METHODS_PATH].boxes.items():
            box.setToolTip(f'{METHOD_TIPS[m]}\n{self.advice[m].reason}')
        self._show_method_note()

    def _show_method_note(self) -> None:
        """A gold note on each ticked method that is not recommended for the detected spectrum (never blocks it)"""
        boxes = self.editors[METHODS_PATH].boxes
        self.method_note.setText('\n'.join(f'{METHOD_LABELS[m]}: {a.reason}' for m, a in self.advice.items()
                                           if not a.recommended and boxes[m].isChecked()))
        self.method_note.setStyleSheet(f'color: {theme.colour("flagged", self.method_note.palette()).name()}')
