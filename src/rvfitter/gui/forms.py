"""Generic settings forms generated from the fields of the ``Config`` dataclasses.

Widgets are chosen from the type hints (and, for a few well-known names, from the name), so a new config option
appears in the GUI without any GUI change.
A field's tooltip is its `TIPS` entry; ``metadata={"choices": [...]}`` sets its choices.
"""
from __future__ import annotations

import copy
import dataclasses
import enum
import json
import types
import typing
from typing import Any, Literal, Union, get_args, get_origin

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..config import METHOD_LABELS, METHOD_NAMES, OUTPUT_FORMATS
from ..templates import KNOWN_GRIDS

# tooltip of every setting the forms show, by dotted config path (a missing one is a KeyError when the form is built)
TIPS = {
    'seed': 'Random seed for the Monte Carlo errors and samplers. The same seed gives the same results, for any number '
            'of processes. Default 0.',
    'input.paths': 'Spectra (files, folders or globs) that "rvfitter run" reads from a settings file. Not used in the '
                   'GUI: open or drop spectra instead.',
    'input.medium': 'Medium of the wavelengths in the files. "auto" reads each header; air or vacuum forces it (a '
                    'conflict with the header is logged).',
    'input.medium_method': 'Air-to-vacuum formula (ciddor, edlen or morton); used only for spectra in air. Default '
                           'ciddor.',
    'input.hdu': 'FITS extension to read (number or name). Leave empty to choose it automatically.',
    'input.resolution': 'Resolving power R applied to every spectrum, overriding the header. Leave empty to use each '
                        'spectrum\'s own (its header, else estimated from the pixel spacing).',
    'input.telluric_corrected': 'The spectra are already telluric-corrected: relax the telluric masking. The '
                                'telluric zero point then needs an uncorrected file (Calibration > Telluric source).',
    'input.raw_header': 'Raw frame headers (a path or glob), needed for the slit position angle that the reduction '
                        'drops (differential refraction across the slit).',
    'input.raw_header_dir': 'Folder searched for the raw frames named in the reduced files (PypeIt FILENAME).',
    'grouping.override': 'Which object each spectrum belongs to, when the header or file names group them wrongly '
                         '(JSON): file path, name or stem, or header object name, to object, e.g. '
                         '{"spec_0012.fits": "J1745-1640"}. Empty = automatic.',
    'velocity_frame.correction': 'Barycentric correction: auto (skip a spectrum already corrected, else compute it '
                                 'from the header, else use an ESO BERV keyword, else warn and skip), apply (always '
                                 'compute) or skip.',
    'velocity_frame.frame': 'Velocity frame of the reported RVs: barycentric (default) or heliocentric.',
    'velocity_frame.overrides': 'Values used instead of the header (JSON): ra, dec [deg], mjd (mid-exposure), site '
                                '(an astropy site name) or site_lat and site_lon [deg], e.g. {"site": "lapalma"}.',
    'templates.grid': 'Model-atmosphere grid(s). Several are searched together and their spread is a systematic. '
                      'Templates are downloaded and cached on first use.',
    'templates.local_dir': 'Folder with your own template grid (or JSON {grid: folder}). Leave empty to use the '
                           'download cache.',
    'templates.medium': 'Medium of the template wavelengths; leave at (none) to take it from the grid\'s metadata.',
    'templates.teff_range': 'Teff range searched [K], e.g. 1500, 2500. Empty = the Teff prior ± 300 K, or the whole '
                            'grid without a prior.',
    'templates.logg_range': 'log g range searched [dex, cgs], e.g. 4.5, 5.5. Empty = the whole grid.',
    'templates.met_range': 'Metallicity range searched [dex]. Empty = held at the grid point nearest the '
                           'metallicity prior (or the grid default).',
    'templates.alpha_range': 'Alpha-enhancement range searched [dex]. Empty = held at the grid point nearest the '
                             'alpha prior (or the grid default).',
    'templates.ranges': 'Ranges of any other grid axes (JSON), e.g. {"co": [0.5, 1]}. An axis without a range is held '
                        'at its default.',
    'templates.teff_prior': 'Expected Teff [K] (e.g. from the spectral type), used to choose the features and the '
                            'templates, e.g. 1800 or 2400.',
    'templates.logg_prior': 'Expected log g [dex], used when no grid point lies in the ranges and for the telluric '
                            'fit\'s template. Empty = 5.0.',
    'templates.met_prior': 'Expected metallicity [dex]; held there when no metallicity range is set. Empty = the '
                           'grid default.',
    'templates.alpha_prior': 'Expected alpha enhancement [dex]; held there when no alpha range is set. Empty = the '
                             'grid default.',
    'methods.enabled': 'Which RV techniques to run. Each is a separate measurement; they are combined into the '
                       'adopted RV. The defaults follow the resolving power.',
    'methods.options': 'Keyword options per method (JSON), e.g. {"template_fit": {"fit_vsini": true}, "linecentre": '
                       '{"profile": "gaussian"}}. Empty = each method\'s defaults.',
    'methods.binary': 'Also test for an unresolved binary: one template against two shifted copies (binary_dbic, '
                      'binary_flag).',
    'regions.overrides': 'Region overrides (JSON). Edit them graphically on the Regions tab.',
    'regions.min_snr': 'Features whose local S/N is below this are dropped. Default 5; lower it for faint spectra, '
                       'at the cost of noisier regions.',
    'regions.telluric_strictness': 'Scales the airmass-dependent threshold for masking telluric bands; larger masks '
                                   'more. Default 1.',
    'regions.max_regions': 'Keep at most this many regions per method, ranked by their forecast RV information. '
                           'Default 12.',
    'regions.continuum_fwhm': 'Continuum windows extend this many resolution elements around a feature. Default 5.',
    'regions.template_windows': 'Windows for CCF and template fit at R ≤ 5000: features (around well-modelled atomic '
                                'and CO features; default, since model grids reproduce molecular bands poorly) or '
                                'full (every non-telluric span). Above R 5000 the full spans are always used.',
    'combine.tau_scale': 'Scale [km/s] of the half-Cauchy prior on the region-to-region and method-to-method scatter '
                         'when combining. Empty = automatic (the median error).',
    'combine.systematic_floor': 'Added in quadrature to the adopted error [km/s], e.g. a known instrument floor. '
                                'Default 0.',
    'calibration.wavelength_correction': 'Wavelength zero-point correction: sky-if-needed (default: the OH sky curve, '
                                         'only when safe and significant), auto (also the star - sky constant), sky, '
                                         'telluric or none (report only).',
    'calibration.slit_prior': 'Slit-centring error when the star - sky constant is not applied: auto ((W - F)/√12 '
                              'from the slit width and trace FWHM), a velocity [km/s], or none (unconstrained; the '
                              'total error is then undefined).',
    'calibration.seeing_fwhm': 'Seeing FWHM [arcsec] assumed for the auto slit prior when the file has no trace FWHM '
                               '(flagged seeing_assumed). Default 0.8.',
    'calibration.wave_rms_kms': 'Wavelength-calibration accuracy [km/s] assumed when the file records no arc RMS '
                                '(flagged wave_rms_assumed). Empty = the instrument\'s typical value, if known.',
    'calibration.zero_point_known': 'Fibre or IFU data, or an absolute calibration: no slit-centring term.',
    'calibration.sky_source': 'File whose sky spectrum shares the input\'s wavelength solution (e.g. a PypeIt spec1d), '
                              'for spectra without their own sky.',
    'calibration.sky_segment_width': 'Maximum width [Å] of the sky segments; the zero point is interpolated between '
                                     'them. Default 250.',
    'calibration.telluric_source': 'Uncorrected file with the input\'s wavelength solution, for the star\'s own '
                                   'zero point from its telluric lines; needed when the input is telluric-corrected.',
    'calibration.telluric_segment_width': 'Maximum width [Å] of the telluric-band segments. Default 250.',
    'calibration.telluric_template': 'Model the star with the template nearest the priors in the telluric fit '
                                     '(default). Untick to fit the telluric lines without a stellar model.',
    'calibration.wave_valid_range': 'Vacuum wavelength range [Å, observatory frame] where the wavelength solution is '
                                    'constrained, e.g. by the arc lines; sky segments outside it are not used. Empty = '
                                    'the arc-line range found next to a PypeIt file.',
    'parallel.n_procs': 'Worker processes within each object; auto = serial, or all but one core when the forward '
                        'model runs. Results do not depend on it.',
    'output.path': 'File the results are written to after each run (each format gets its own extension). Empty = '
                   'save only with Export.',
    'output.formats': 'Results file formats to write.',
    'output.plots': 'Write the diagnostic plots next to the results ("rvfitter run" only; the GUI shows them in the '
                    'Diagnostics tab).',
    'output.plot_style': 'Plots written by "rvfitter run". quick: bundled fonts, no LaTeX needed. paper: text '
                         'typeset by LaTeX (needs latex and dvipng).',
    'output.medium': 'Medium of the wavelengths in the reports and plots: vacuum (default) or air.',
}
METHOD_TIPS = {  # what each method does; the Setup tab adds why it is or is not recommended
    'template_fit': 'Chi-squared fit of a (broadened, shifted) model-atmosphere template.',
    'ccf': 'Cross-correlation with the model template, one RV per window.',
    'telluric_ref': 'The star\'s velocity against the telluric lines of the same band.',
    'linecentre': 'Fit a line profile (Gaussian, Lorentzian or Voigt) to each line core and measure its shift.',
    'forward_mcmc': 'Pixel-level forward model (grid template, vsini, LSF, continuum) sampled by MCMC.',
    'lsd': 'Least-squares deconvolution: the mean line profile against a template line mask; RV and vsini.',
}

_PATH_NAMES = {'output_dir', 'out_dir', 'local_dir', 'template_dir', 'cache_dir', 'logfile', 'raw_header_dir'}
_CHOICE_LISTS = {'methods': METHOD_NAMES, 'enabled': METHOD_NAMES, 'formats': OUTPUT_FORMATS,
                 'grid': tuple(KNOWN_GRIDS)}
LABELS = {'seed': 'Random seed', 'hdu': 'FITS HDU', 'raw_header_dir': 'Raw header folder', 'n_procs': 'Processes',
          'raw_header': 'Raw header file', 'medium_method': 'Air/vacuum formula', 'enabled': 'Methods',
          'telluric_corrected': 'Telluric corrected'}
# kept from the loaded config, never shown: the file-format version and the refit button's per-object patches
HIDDEN = {'config.schema_version', 'config.refits'}
OPEN_SECTIONS = {'input', 'methods', 'velocity_frame', 'calibration', 'output'}  # the common settings
NUMBER_WIDTH, CHOICE_WIDTH, TEXT_WIDTH = 120, 180, 320  # maximum widths [px]; paths and lists stretch


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
    """Spin box; if optional (only counts such as ``n_procs``, valid from 1), 0 is the sentinel for ``None``, shown
    as "auto", so one arrow click moves between automatic and an explicit value."""

    LIMIT = 2_000_000_000

    def __init__(self, optional: bool = False):
        self.optional = optional
        self.widget = QSpinBox()
        self.widget.setRange(0 if optional else -self.LIMIT, self.LIMIT)
        self.widget.setMaximumWidth(NUMBER_WIDTH)
        if optional:
            self.widget.setSpecialValueText('auto')

    def get(self):
        if self.optional and self.widget.value() == self.widget.minimum():
            return None
        return int(self.widget.value())

    def set(self, value):
        self.widget.setValue(self.widget.minimum() if value is None else int(value))


class ChoiceEditor(Editor):
    def __init__(self, choices: list[Any], optional: bool = False):
        self.widget = QComboBox()
        self.widget.setMaximumWidth(CHOICE_WIDTH)
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
        if kind in (int, float):
            self.edit.setMaximumWidth(NUMBER_WIDTH)
        elif not browse and kind is str:
            self.edit.setMaximumWidth(TEXT_WIDTH)
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
        self.widget.setPlaceholderText('comma-separated')

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
    """Check boxes (three per row) for a list of names (methods, formats, grids).

    With ``scalar_ok`` a single ticked name is returned as a plain string (the config accepts either).
    """

    COLUMNS = 3

    def __init__(self, choices: tuple[str, ...], container: type = list, scalar_ok: bool = False):
        self.container, self.scalar_ok = container, scalar_ok
        self._order: list[str] = []
        self.widget = QWidget()
        self.boxes: dict[str, QCheckBox] = {}
        self._lay = QGridLayout(self.widget)
        self._lay.setContentsMargins(0, 0, 0, 0)
        for c in choices:
            self._add(c)

    def _add(self, name: str):
        box = QCheckBox(METHOD_LABELS.get(name, name))
        if name in METHOD_TIPS:
            box.setToolTip(f'{METHOD_TIPS[name]} (config name "{name}")')
        n = len(self.boxes)
        self.boxes[name] = box
        self._lay.addWidget(box, n // self.COLUMNS, n % self.COLUMNS)

    def _rank(self, name: str) -> int:
        return self._order.index(name) if name in self._order else len(self._order)

    def get(self):
        names = sorted((n for n, b in self.boxes.items() if b.isChecked()), key=self._rank)  # keep loaded order
        return names[0] if self.scalar_ok and len(names) == 1 else self.container(names)

    def set(self, value):
        value = [value] if isinstance(value, str) else list(value or [])
        self._order = value
        for v in value:
            if v not in self.boxes:
                self._add(v)
        for n, b in self.boxes.items():
            b.setChecked(n in value)


class GroupEditor(Editor):
    """A nested dataclass: a group box holding one editor per field.

    A section (one level below the root) is collapsible: an arrow button with its title shows or hides the fields
    (an arrow, not a check box, so it does not read as switching the section on or off).
    """

    def __init__(self, cls: type, path: str, tooltips: dict[str, str]):
        self.cls, self.path = cls, path
        title = LABELS.get(path.rsplit('.', 1)[-1], path.rsplit('.', 1)[-1].replace('_', ' ').capitalize())
        section = path.count('.') == 1
        self.body = QWidget()
        self.layout = QFormLayout(self.body)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)  # same on every OS
        self.box = QGroupBox('' if section else title)
        QVBoxLayout(self.box).addWidget(self.body)
        if not section:
            self.widget = self.box
            self._add_fields(cls, path, tooltips)
            return
        self.widget = QWidget()
        outer = QVBoxLayout(self.widget)
        outer.setContentsMargins(0, 0, 0, 0)
        self.toggle = QToolButton(text=title, checkable=True, autoRaise=True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        font = self.toggle.font()
        font.setBold(True)
        self.toggle.setFont(font)
        self.toggle.toggled.connect(self._expand)
        outer.addWidget(self.toggle)
        outer.addWidget(self.box)
        self.toggle.setChecked(path.rsplit('.', 1)[-1] in OPEN_SECTIONS)
        self._expand(self.toggle.isChecked())
        self._add_fields(cls, path, tooltips)

    def _add_fields(self, cls: type, path: str, tooltips: dict[str, str]) -> None:
        self.children: dict[str, Editor] = {}
        hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            if not f.init or f'{path}.{f.name}' in HIDDEN:
                continue
            sub = f'{path}.{f.name}'
            ed = make_editor(hints.get(f.name, f.type), f, sub, tooltips)
            self.children[f.name] = ed
            tip = ed.cls.__doc__.strip().splitlines()[0] if isinstance(ed, GroupEditor) else \
                tooltips[sub.removeprefix('config.')]
            ed.widget.setToolTip(tip)
            label = LABELS.get(f.name, f.name.replace('_', ' ').capitalize())
            if isinstance(ed, GroupEditor):
                self.layout.addRow(ed.widget)
            else:
                self.layout.addRow(label, ed.widget)
                lab = self.layout.labelForField(ed.widget)
                if lab is not None:
                    lab.setToolTip(tip)

    def _expand(self, on: bool) -> None:
        self.box.setVisible(on)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)

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
    if name == 'grid':
        return CheckListEditor(_CHOICE_LISTS['grid'], list, scalar_ok=True)
    if tp is int:
        return IntEditor(optional)
    if tp is float:
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
    return TextEditor('json', optional=optional or type(None) in get_args(tp))  # e.g. hdu: int | str | None


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
