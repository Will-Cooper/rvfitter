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
import re
import types
import typing
from typing import Any, Literal, Union, get_args, get_origin

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPalette
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..config import LOCATION_FIELDS, METHOD_LABELS, METHOD_NAMES, OUTPUT_FORMATS
from ..templates import KNOWN_GRIDS
from . import theme

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
    'input.objects': 'Which spectra of a multi-object file (PypeIt, JWST x1d, multi-extension FITS) to measure, each '
                     'as its own object: auto (the target if named, else every spectrum with its own source name), '
                     'target, all, or a JSON list of names or HDU numbers.',
    'input.cube.position': 'IFU cube: aperture centre (x, y) in pixels of the collapsed image. Empty = the brightest '
                           'spaxel.',
    'input.cube.radius': 'IFU cube: aperture radius in spaxels. Empty = 3.',
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
    'templates.teff_range': 'Teff range searched [K], e.g. 1500, 2500, or one grid value, e.g. 2000 (fewer templates '
                            'to download and fit). Empty = the Teff prior ± 300 K, or the whole grid without a prior.',
    'templates.logg_range': 'log g range searched [dex, cgs], e.g. 4.5, 5.5, or one grid value, e.g. 5.0. Empty = '
                            'the whole grid.',
    'templates.met_range': 'Metallicity range searched [dex], or one grid value. Empty = held at the grid point '
                           'nearest the metallicity prior (or the grid default).',
    'templates.alpha_range': 'Alpha-enhancement range searched [dex], or one grid value. Empty = held at the grid '
                             'point nearest the alpha prior (or the grid default).',
    'templates.ranges': 'Ranges of any other grid axes (JSON), e.g. {"co": [0.5, 1]}, or one value, {"co": 1}. An '
                        'axis without a range is held at its default.',
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
                           'Default auto: no limit.',
    'regions.continuum_fwhm': 'Continuum windows extend this many resolution elements around a feature. Default 3.',
    'regions.template_windows': 'Windows for CCF and template fit below R 4500: features (around well-modelled atomic '
                                'and CO features; default, since model grids reproduce molecular bands poorly) or '
                                'full (every non-telluric span). From R 4500 up the full spans are always used.',
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
# why the default of each setting that `config.setting_advice` does not cover suits any spectrum (the Advanced
# dialog shows it as that setting's recommendation)
DEFAULT_WHY = {
    'seed': 'only the Monte Carlo draws depend on it.',
    'input.paths': 'the GUI uses the spectra you open.',
    'input.medium_method': 'Ciddor (1996) is the standard air-to-vacuum formula; the others differ by under 1 m/s.',
    'input.hdu': 'the extension is chosen from the file\'s structure.',
    'input.raw_header': 'needed only for the slit position angle, if the reduced file lacks it.',
    'input.raw_header_dir': 'needed only for PypeIt files whose raw frames are elsewhere.',
    'input.objects': 'auto measures the target, else every spectrum with its own source name.',
    'input.cube.position': 'the brightest spaxel is the target in most cubes.',
    'input.cube.radius': '3 spaxels holds most of the light of a point source.',
    'grouping.override': 'spectra are grouped by the object name in their headers.',
    'velocity_frame.overrides': 'the header values are used; set them only where the header is wrong or empty.',
    'templates.local_dir': 'the grids are downloaded to the cache when first needed.',
    'templates.medium': 'the grid\'s own metadata gives its medium.',
    'templates.met_range': '[M/H] is held at the prior (or the grid default); free it for subdwarfs.',
    'templates.alpha_range': '[α/M] is held at the prior (or the grid default).',
    'templates.ranges': 'the extra axes are held at their defaults.',
    'templates.teff_prior': 'set it from the spectral type to choose the features and the grid.',
    'templates.logg_prior': 'the grid\'s default is used.',
    'templates.met_prior': 'solar; set it for metal-poor objects (the grid advice then ranks the metal-poor grids '
                           'first).',
    'templates.alpha_prior': 'the grid\'s default is used.',
    'methods.binary': 'the single- versus two-template test is needed only for suspected binaries.',
    'regions.overrides': 'the regions are chosen from the spectrum.',
    'regions.min_snr': 'features below S/N 5 add noise rather than information.',
    'regions.telluric_strictness': 'the airmass-dependent threshold suits most sites.',
    'regions.max_regions': 'every region that passes the S/N and telluric thresholds adds information.',
    'regions.continuum_fwhm': '3 resolution elements give a local continuum clear of neighbouring features.',
    'combine.tau_scale': 'the median error sets the scale of the scatter prior.',
    'combine.systematic_floor': 'add only a known instrument floor.',
    'calibration.slit_prior': 'auto uses the slit width and the trace FWHM.',
    'calibration.seeing_fwhm': 'used only when the trace FWHM is unknown.',
    'calibration.wave_rms_kms': 'the file\'s arc RMS, else the instrument\'s typical value, is used.',
    'calibration.sky_source': 'needed only for spectra without their own sky.',
    'calibration.sky_segment_width': '250 Å holds enough sky lines per segment.',
    'calibration.telluric_source': 'needed only when the input is telluric-corrected.',
    'calibration.telluric_segment_width': '250 Å holds enough telluric lines per segment.',
    'calibration.telluric_template': 'modelling the star separates its lines from the telluric ones.',
    'calibration.wave_valid_range': 'the arc-line range next to a PypeIt file is used.',
    'output.path': 'results are saved with Export.',
    'output.formats': 'CSV opens everywhere.',
    'output.plots': 'the GUI shows the plots in the Diagnostics tab.',
    'output.plot_style': 'quick needs no LaTeX.',
    'output.medium': 'vacuum wavelengths are the standard.',
}


def guide_url(path: str = '') -> str:
    """The settings guide, at the entry of a dotted config path or a section name (anchor: dots replaced by '-')"""
    return f'{GUIDE_URL}#{path.replace(".", "-")}' if path else GUIDE_URL


def open_guide(path: str = '') -> None:
    QDesktopServices.openUrl(QUrl(guide_url(path)))

_CHOICE_LISTS = {'methods': METHOD_NAMES, 'enabled': METHOD_NAMES, 'formats': OUTPUT_FORMATS,
                 'grid': tuple(KNOWN_GRIDS)}
LABELS = {'seed': 'Random seed', 'hdu': 'FITS HDU', 'raw_header_dir': 'Raw header folder', 'n_procs': 'Processes',
          'raw_header': 'Raw header file', 'medium_method': 'Air/vacuum formula', 'enabled': 'Methods',
          'telluric_corrected': 'Telluric corrected'}
# kept from the loaded config, never shown: the file-format version and the refit button's per-object patches
HIDDEN = {'config.schema_version', 'config.refits'}
GUIDE_URL = 'https://github.com/Will-Cooper/rvfitter/blob/main/docs/settings.md'
MORE_TIP = '\nMore… in the settings guide (the link under the box).'  # ends every Advanced tooltip
BROWSE_TEXT = {'files': 'Add files…', 'file': 'Browse…', 'folder': 'Browse…', 'save': 'Browse…'}  # by kind
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


def _choose(parent: QWidget, kind: str, start: str) -> list[str]:
    """The paths picked in the file dialog of a `config.LOCATION_FIELDS` kind (empty if cancelled)"""
    if kind == 'files':
        return QFileDialog.getOpenFileNames(parent, 'Add files', start)[0]
    if kind == 'file':
        name = QFileDialog.getOpenFileName(parent, 'Choose file', start)[0]
    elif kind == 'folder':
        name = QFileDialog.getExistingDirectory(parent, 'Choose folder', start)
    elif kind == 'save':
        name = QFileDialog.getSaveFileName(parent, 'Save as', start)[0]
    else:
        raise ValueError(f'Unknown location kind {kind!r}: not one of {sorted(BROWSE_TEXT)}')
    return [name] if name else []


def _with_browse(edit: QWidget, kind: str | None, slot) -> tuple[QWidget, QPushButton | None]:
    """``edit`` alone, or in a row with a Browse button (its ``kind`` attribute is the dialog's kind)"""
    if kind is None:
        return edit, None
    if kind not in BROWSE_TEXT:
        raise ValueError(f'Unknown location kind {kind!r}: not one of {sorted(BROWSE_TEXT)}')
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    btn = QPushButton(BROWSE_TEXT[kind])
    btn.kind = kind
    btn.clicked.connect(slot)
    lay.addWidget(edit, 1)
    lay.addWidget(btn)
    return row, btn


class TextEditor(Editor):
    """Line edit parsing to ``kind`` (str, float, int, json, or a tuple of them for a union such as int | str);
    an empty box is ``None`` if optional."""

    def __init__(self, kind: type | str = str, optional: bool = False, browse: str | None = None):
        self.kind, self.optional = kind, optional
        self.edit = QLineEdit()
        if kind in (int, float):
            self.edit.setMaximumWidth(NUMBER_WIDTH)
        elif not browse and kind is str:
            self.edit.setMaximumWidth(TEXT_WIDTH)
        self.widget, self.browse_btn = _with_browse(self.edit, browse, self._browse)

    def _browse(self):
        """Fill the box from the dialog of its kind; a JSON mapping {grid: folder} gets the folder of its first grid"""
        chosen = _choose(self.widget, self.browse_btn.kind, self.edit.text())
        if not chosen:
            return
        if self.kind != 'json':
            self.edit.setText(chosen[0])
            return
        current = self.get() if self.edit.text().strip() else None
        if isinstance(current, dict) and current:
            current[next(iter(current))] = chosen[0]
            self.set(current)
        else:
            self.set(chosen[0])

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
        if isinstance(self.kind, tuple):  # a union such as int | str: the first type that parses (str last)
            for t in self.kind:
                try:
                    return t(text)
                except ValueError:
                    continue
            raise ValueError(f'{text!r} is not one of {", ".join(t.__name__ for t in self.kind)}')
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

    def __init__(self, container: type, elem: type, optional: bool = False, browse: str | None = None):
        self.container, self.elem, self.optional = container, elem, optional
        self.edit = QLineEdit()
        self.edit.setPlaceholderText('comma-separated')
        self.widget, self.browse_btn = _with_browse(self.edit, browse, self._browse)

    def _browse(self):
        chosen = _choose(self.widget, self.browse_btn.kind, '')
        if chosen:
            self.set([*(self.get() or []), *chosen])

    def get(self):
        text = self.edit.text().strip()
        if not text:
            return None if self.optional else self.container()
        items = [t.strip() for t in text.split(',') if t.strip()]
        conv = self.elem if self.elem in (int, float, str) else str
        return self.container(conv(t) for t in items)

    def set(self, value):
        self.edit.setText('' if value is None else ', '.join(str(v) for v in value))


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
        section_name = path.rsplit('.', 1)[-1]
        self.help = QToolButton(text='?', autoRaise=True, toolTip=f'{title} in the settings guide')
        self.help.clicked.connect(lambda: open_guide(section_name))
        header = QHBoxLayout()
        header.addWidget(self.toggle)
        header.addWidget(self.help)
        header.addStretch(1)
        outer.addLayout(header)
        outer.addWidget(self.box)
        self._expand(False)  # every section starts collapsed: the Setup tab is the everyday view
        self._add_fields(cls, path, tooltips)

    def _add_fields(self, cls: type, path: str, tooltips: dict[str, str]) -> None:
        self.children: dict[str, Editor] = {}
        self.rows: dict[str, _FieldRow] = {}  # the leaf fields' rows (editor, comment, recommendation)
        hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            if not f.init or f'{path}.{f.name}' in HIDDEN:
                continue
            sub = f'{path}.{f.name}'
            ed = make_editor(hints.get(f.name, f.type), f, sub, tooltips)
            self.children[f.name] = ed
            label = LABELS.get(f.name, f.name.replace('_', ' ').capitalize())
            if isinstance(ed, GroupEditor):
                ed.widget.setToolTip(ed.cls.__doc__.strip().splitlines()[0])
                self.layout.addRow(ed.widget)
                continue
            key = sub.removeprefix('config.')
            tip = tooltips[key] + MORE_TIP
            ed.widget.setToolTip(tip)
            self.rows[f.name] = row = _FieldRow(ed, key, tooltips[key])
            self.layout.addRow(label, row)
            self.layout.labelForField(row).setToolTip(tip)

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


def _same(a: Any, b: Any) -> bool:
    """Equal settings values (lists of names in any order; a lone name equals a one-name list)"""
    if isinstance(a, (str, list)) and isinstance(b, (str, list)):
        return sorted([a] if isinstance(a, str) else a) == sorted([b] if isinstance(b, str) else b)
    return a == b


def _shown(value: Any) -> str:
    if isinstance(value, tuple):
        return '–'.join(f'{v:g}' for v in value)
    if isinstance(value, list):
        return ', '.join(METHOD_LABELS.get(v, v) for v in value) or '(none)'
    if isinstance(value, dict):
        return json.dumps(value) if value else '(none)'
    return str(value)


class _FieldRow(QWidget):
    """An editor, the first sentence of its tip with a link to its settings-guide entry (grey), and a recommendation
    line with Apply (blue; gold when the box differs) once `show_advice` is called"""

    def __init__(self, ed: Editor, path: str, tip: str):
        super().__init__()
        self.ed = ed
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)
        lay.addWidget(ed.widget)
        first = re.split(r'(?<!e\.g)(?<!i\.e)\. (?=[A-Z])', tip)[0].rstrip('.')  # its first sentence
        self.comment = QLabel(f'{first}. <a href="{guide_url(path)}">More…</a>', wordWrap=True,
                              openExternalLinks=True)
        grey = self.comment.palette().color(QPalette.ColorRole.PlaceholderText).name()
        self.comment.setStyleSheet(f'color: {grey}')
        lay.addWidget(self.comment)
        self.advice = QLabel(wordWrap=True)
        self.apply = QPushButton('Apply', toolTip='Set the recommended value')
        self.apply.clicked.connect(self._apply)
        row = QHBoxLayout()
        row.addWidget(self.advice, 1)
        row.addWidget(self.apply, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(row)
        self.advice.hide()
        self.apply.hide()
        self.value: Any = None

    def show_advice(self, text: str, value: Any = None) -> None:
        """Show ``text``; a ``value`` (None: nothing to apply) gets Apply"""
        self.value = value
        self.advice.setText(text)
        self.advice.show()
        self.apply.setVisible(value is not None)
        self._colour()

    def _colour(self) -> None:
        role = 'flagged' if self.value is not None and not _same(self.ed.get(), self.value) else 'primary'
        self.advice.setStyleSheet(f'color: {theme.colour(role, self.advice.palette()).name()}')

    def _apply(self) -> None:
        self.ed.set(self.value)
        self._colour()


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
    browse = LOCATION_FIELDS.get(tuple(path.removeprefix('config.').split('.')))
    if tp is str:
        return TextEditor(str, optional=optional, browse=browse)
    if origin in (list, tuple, set, frozenset) or tp in (list, tuple, set, frozenset):
        container = origin or tp
        args = [a for a in get_args(tp) if a is not Ellipsis]
        elem = args[0] if args else str
        if name in _CHOICE_LISTS and elem is str:
            return CheckListEditor(_CHOICE_LISTS[name], container)
        return SeqEditor(container, elem, optional, browse=browse)
    scalars = set(get_args(tp)) - {type(None)}
    if str in scalars and scalars <= {str, int, float}:  # e.g. slit_prior ('auto' or km/s), hdu (index or name)
        return TextEditor(tuple(t for t in (int, float, str) if t in scalars),
                          optional=optional or type(None) in get_args(tp))
    return TextEditor('json', optional=optional or type(None) in get_args(tp), browse=browse)  # e.g. local_dir


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

    def rows(self) -> dict[str, _FieldRow]:
        """Every leaf field's row, by dotted path"""
        out, todo = {}, [('', self.root)]
        while todo:
            prefix, g = todo.pop()
            out.update({prefix + n: r for n, r in g.rows.items()})
            todo += [(f'{prefix}{n}.', e) for n, e in g.children.items() if isinstance(e, GroupEditor)]
        return out

    def show_advice(self, name: str, advice: dict[str, Any]) -> None:
        """Under every box, the recommendation for object ``name``: its `config.setting_advice` entry, else why the
        default suits it (`DEFAULT_WHY`)"""
        for path, row in self.rows().items():
            if path in advice:
                value, reason = advice[path]
                row.show_advice(f'Recommended for {name}: {_shown(value)}. {reason}' if value is not None else
                                f'For {name}: {reason}', value)
            elif path in DEFAULT_WHY:
                row.show_advice(f'The default suits this spectrum: {DEFAULT_WHY[path]}')

    def get_value(self, path: str) -> Any:
        ed = self.editor(path)
        return None if ed is None else ed.get()

    def set_value(self, path: str, value: Any) -> bool:
        ed = self.editor(path)
        if ed is None:
            return False
        ed.set(value)
        return True
