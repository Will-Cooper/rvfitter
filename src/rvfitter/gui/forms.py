"""Generic settings forms generated from the fields of the ``Config`` dataclasses.

Widgets are chosen from the type hints (and, for a few well-known names, from the name), so a new config option
appears in the GUI without any GUI change.
A field may provide ``metadata={"help": "...", "choices": [...]}`` to control its tooltip and choices.
"""
from __future__ import annotations

import copy
import dataclasses
import enum
import json
import types
import typing
from typing import Any, Literal, Union, get_args, get_origin

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .backend import FORMATS, METHODS

TIPS = {
    'medium': 'Wavelength medium of the input files. "auto" reads the header; otherwise air or vacuum is forced.',
    'medium_formula': 'Air/vacuum conversion formula (ciddor, edlen or morton). Used only when the input is in air.',
    'conversion': 'Air/vacuum conversion formula (ciddor, edlen or morton). Used only when the input is in air.',
    'formula': 'Air/vacuum conversion formula (ciddor, edlen or morton). Used only when the input is in air.',
    'correction': 'Barycentric correction: auto (use the header, else compute, else skip), apply (force) or skip.',
    'mode': 'Barycentric correction: auto (use the header, else compute, else skip), apply (force) or skip.',
    'frame': 'Velocity frame of the reported RVs: barycentric or heliocentric.',
    'methods': 'Which RV techniques to run. Each is a separate, independent measurement.',
    'linecentre': 'Fit line profiles and measure their centres.',
    'ccf': 'Cross-correlate with a template or mask.',
    'template_fit': 'Chi-squared fit of a (broadened, shifted) model-atmosphere template.',
    'forward_mcmc': 'Forward-model the spectrum with emcee (needs the [mcmc] extra). Slow but gives full posteriors.',
    'lsd': 'Least-squares deconvolution line profile.',
    'telluric_ref': 'Use telluric lines as a wavelength reference (zero-point check).',
    'grid': 'Template grid name (e.g. bt-settl-cifist). Templates are downloaded and cached on first use.',
    'template_grid': 'Template grid name (e.g. bt-settl-cifist). Templates are downloaded and cached on first use.',
    'local_dir': 'Folder with your own template spectra. Leave empty to use the downloaded cache.',
    'template_dir': 'Folder with your own template spectra. Leave empty to use the downloaded cache.',
    'teff_prior': 'Effective-temperature prior [K] used to select features and templates, e.g. 1800, 2400.',
    'teff': 'Effective-temperature prior [K] used to select features and templates.',
    'n_procs': 'Number of worker processes used within each object (1 = serial). Results do not depend on it.',
    'seed': 'Random seed. The same seed always gives the same results, for any number of processes.',
    'path': 'Where the results are written (a file name or folder; the format extensions are added).',
    'output_dir': 'Folder where the results tables are written.',
    'out_dir': 'Folder where the results tables are written.',
    'formats': 'Results file formats to write.',
    'overrides': 'Region overrides (JSON). Edit them graphically on the Regions tab.',
}

_PATH_NAMES = {'output_dir', 'out_dir', 'local_dir', 'template_dir', 'cache_dir', 'logfile'}
_CHOICE_LISTS = {'methods': METHODS, 'formats': FORMATS}


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) < len(get_args(tp)) and len(args) == 1:
            return args[0], True
    return tp, False


class Editor:
    """A widget plus ``get``/``set`` of the python value."""

    widget: QWidget

    def get(self) -> Any:
        raise NotImplementedError

    def set(self, value: Any) -> None:
        raise NotImplementedError


class BoolEditor(Editor):
    def __init__(self):
        self.widget = QCheckBox()

    def get(self):
        return self.widget.isChecked()

    def set(self, value):
        self.widget.setChecked(bool(value))


class IntEditor(Editor):
    def __init__(self):
        self.widget = QSpinBox()
        self.widget.setRange(-2_000_000_000, 2_000_000_000)

    def get(self):
        return int(self.widget.value())

    def set(self, value):
        self.widget.setValue(int(value))


class ChoiceEditor(Editor):
    def __init__(self, choices: list[Any], optional: bool = False):
        self.widget = QComboBox()
        self.optional = optional
        if optional:
            self.widget.addItem('(none)', None)
        for c in choices:
            self.widget.addItem(str(c.value if isinstance(c, enum.Enum) else c), c)

    def get(self):
        return self.widget.currentData()

    def set(self, value):
        idx = self.widget.findData(value)
        if idx < 0:
            idx = self.widget.findText(str(value))
        if idx < 0:
            self.widget.addItem(str(value), value)
            idx = self.widget.count() - 1
        self.widget.setCurrentIndex(idx)


class TextEditor(Editor):
    """Line edit parsing to ``kind`` (str, float, int or json); an empty box is ``None`` if optional."""

    def __init__(self, kind: type | str = str, optional: bool = False, browse: bool = False):
        self.kind, self.optional = kind, optional
        self.edit = QLineEdit()
        if browse:
            self.widget = QWidget()
            lay = QHBoxLayout(self.widget)
            lay.setContentsMargins(0, 0, 0, 0)
            btn = QPushButton('Browse…')
            btn.clicked.connect(self._browse)
            lay.addWidget(self.edit, 1)
            lay.addWidget(btn)
        else:
            self.widget = self.edit

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self.widget, 'Choose folder', self.edit.text())
        if d:
            self.edit.setText(d)

    def get(self):
        text = self.edit.text().strip()
        if text == '':
            if self.optional:
                return None
            if self.kind == 'json':
                return {}
            return None if self.kind in (float, int) else ''
        if self.kind is float:
            return float(text)
        if self.kind is int:
            return int(text)
        if self.kind == 'json':
            return json.loads(text)
        return text

    def set(self, value):
        if value is None:
            self.edit.clear()
        elif self.kind == 'json':
            self.edit.setText(json.dumps(value, sort_keys=True))
        else:
            self.edit.setText(str(value))


class SeqEditor(Editor):
    """Comma-separated list/tuple of scalars."""

    def __init__(self, container: type, elem: type, optional: bool = False):
        self.container, self.elem, self.optional = container, elem, optional
        self.widget = QLineEdit()

    def get(self):
        text = self.widget.text().strip()
        if not text:
            return None if self.optional else self.container()
        items = [t.strip() for t in text.split(',') if t.strip()]
        conv = self.elem if self.elem in (int, float, str) else str
        return self.container(conv(t) for t in items)

    def set(self, value):
        self.widget.setText('' if value is None else ', '.join(str(v) for v in value))


class CheckListEditor(Editor):
    """A row of check boxes for a list of names (methods, formats)."""

    def __init__(self, choices: tuple[str, ...], container: type = list):
        self.container = container
        self.widget = QWidget()
        self.boxes: dict[str, QCheckBox] = {}
        self._lay = QHBoxLayout(self.widget)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.addStretch(1)
        for c in choices:
            self._add(c)

    def _add(self, name: str):
        box = QCheckBox(name)
        self.boxes[name] = box
        self._lay.insertWidget(self._lay.count() - 1, box)

    def get(self):
        return self.container(n for n, b in self.boxes.items() if b.isChecked())

    def set(self, value):
        value = list(value or [])
        for v in value:
            if v not in self.boxes:
                self._add(v)
        for n, b in self.boxes.items():
            b.setChecked(n in value)


class GroupEditor(Editor):
    """A nested dataclass: a group box holding one editor per field."""

    def __init__(self, cls: type, path: str, tooltips: dict[str, str]):
        self.cls, self.path = cls, path
        self.widget = QGroupBox(path.rsplit('.', 1)[-1].replace('_', ' '))
        self.layout = QFormLayout(self.widget)
        self.children: dict[str, Editor] = {}
        hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            if not f.init:
                continue
            sub = f'{path}.{f.name}'
            ed = make_editor(hints.get(f.name, f.type), f, sub, tooltips)
            self.children[f.name] = ed
            tip = f.metadata.get('help') or tooltips.get(f.name) or f'Config option "{sub}".'
            ed.widget.setToolTip(tip)
            label = f.name.replace('_', ' ')
            if isinstance(ed, GroupEditor):
                self.layout.addRow(ed.widget)
            else:
                self.layout.addRow(label, ed.widget)
                lab = self.layout.labelForField(ed.widget)
                if lab is not None:
                    lab.setToolTip(tip)

    def get(self, base: Any = None) -> Any:
        base = base if base is not None else self.cls()
        vals = {}
        for name, ed in self.children.items():
            vals[name] = ed.get(getattr(base, name)) if isinstance(ed, GroupEditor) else ed.get()
        return dataclasses.replace(base, **vals)

    def set(self, value: Any) -> None:
        for name, ed in self.children.items():
            ed.set(getattr(value, name))


def make_editor(tp: Any, f: dataclasses.Field, path: str, tooltips: dict[str, str]) -> Editor:
    """Choose an editor for a field from its type hint and name."""
    tp, optional = _unwrap_optional(tp)
    name = f.name
    origin = get_origin(tp)
    if dataclasses.is_dataclass(tp) and isinstance(tp, type):
        return GroupEditor(tp, path, tooltips)
    choices = f.metadata.get('choices')
    if choices:
        return ChoiceEditor(list(choices), optional)
    if origin is Literal:
        return ChoiceEditor(list(get_args(tp)), optional)
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return ChoiceEditor(list(tp), optional)
    if tp is bool:
        return BoolEditor()
    if tp is int and not optional:
        return IntEditor()
    if tp in (int, float):
        return TextEditor(tp, optional=True)
    if tp is str:
        return TextEditor(str, optional=optional, browse=name in _PATH_NAMES)
    if origin in (list, tuple, set, frozenset) or tp in (list, tuple, set, frozenset):
        container = origin or tp
        args = [a for a in get_args(tp) if a is not Ellipsis]
        elem = args[0] if args else str
        if name in _CHOICE_LISTS and elem is str:
            return CheckListEditor(_CHOICE_LISTS[name], container)
        return SeqEditor(container, elem, optional)
    return TextEditor('json', optional=optional)


class ConfigForm(QWidget):
    """Editable view of a Config dataclass tree."""

    def __init__(self, cfg: Any, parent: QWidget | None = None):
        super().__init__(parent)
        self._cfg = copy.deepcopy(cfg)
        self.root = GroupEditor(type(cfg), 'config', TIPS)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        # top-level group's box is only decoration: show its children directly
        self.root.widget.setTitle('')
        self.root.widget.setFlat(True)
        lay.addWidget(self.root.widget)
        self.root.set(cfg)

    def set_config(self, cfg: Any) -> None:
        self._cfg = copy.deepcopy(cfg)
        self.root.set(cfg)

    def get_config(self) -> Any:
        """The config described by the widgets (fields the form cannot show keep their loaded values)."""
        self._cfg = self.root.get(self._cfg)
        return copy.deepcopy(self._cfg)

    def editor(self, path: str) -> Editor | None:
        """Editor for a dotted path such as ``"regions.overrides"``."""
        ed: Editor = self.root
        for part in path.split('.'):
            if not isinstance(ed, GroupEditor) or part not in ed.children:
                return None
            ed = ed.children[part]
        return ed

    def get_value(self, path: str) -> Any:
        ed = self.editor(path)
        return None if ed is None else ed.get()

    def set_value(self, path: str, value: Any) -> bool:
        ed = self.editor(path)
        if ed is None:
            return False
        ed.set(value)
        return True
