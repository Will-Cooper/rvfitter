"""The main window."""
from __future__ import annotations

import copy
import logging
import re
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPainter, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableView,
    QTabWidget,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)

from ..config import METHOD_LABELS
from ..io import _SPECTRUM_KINDS
from . import theme
from .backend import Backend, LazyBackend
from .cube import CubeDialog
from .diagnostics import DiagnosticsPanel, rv_text
from .forms import ConfigForm
from .logpane import LogPane, QtLogHandler
from .regions_editor import RegionEditor
from .setup import SetupPanel
from .tablemodel import PandasModel
from .worker import JobWorker, RunWorker

log = logging.getLogger('rvfitter.gui')

QUICK_START = """
<h3>Quick start</h3>
<ol>
<li><b>Open spectra</b>: files, a folder or a zip/tar archive (or drop them onto the window).
They are grouped into objects in the list on the left.</li>
<li><b>Check the Setup tab</b>. Its boxes are filled in from the spectra, and each says where its value came from.
A gold <i>estimated: check</i> or rose <i>estimate out of range: type R</i> means rvfitter guessed: type the value if
you know it.</li>
<li><b>Measure RVs</b>. The bar shows each method as it runs; each object appears in the <b>Results</b> table when
it is done. Select an object to see its <b>Diagnostics</b>; a gold dot marks a result that needs attention.</li>
</ol>
"""

TREE_EMPTY = 'Drop spectra, folders or .zip/.tar archives here, or use Open spectra.'
FLAGGED_TIP = 'Gold dot: needs attention. The best template is at the edge of the searched grid.'
FAILED_TIP = 'Rose dot: failed. No RV could be measured; see the log.'
MAX_SIZE = (1280, 800)  # default window size cap [px]
LOG_HEIGHT = 90  # default log pane height [px]
PROGRESS_WIDTH = 320  # [px], room for 'J1745-1640 · Template fit (2/3) · object 1/3'
R_PREFIX = {'header': 'R ', 'config': 'R ', 'estimated': 'R≈'}
ROW_ROLE = Qt.ItemDataRole.UserRole + 1  # True on the rows that show an object's R, then its RV
ENTRY_ROLE = Qt.ItemDataRole.UserRole + 2  # the entry name of a spectrum of a multi-object file


def _file_of(source: str) -> str:
    """The file of a spectrum source ('file.fits[3]' for one entry of a multi-object file)"""
    return source[:source.rindex('[')] if source.endswith(']') else source


def _scan(backend: Backend, files: list[Path], cfg: Any) -> tuple[dict, dict, dict, dict]:
    """
    Group ``files`` into objects and read each object's first spectrum; list the spectra of the files that hold
    several, and the collapsed image and brightest spaxel of the IFU cubes. Returns (groups, first spectra, entries
    per multi-object file, (image, brightest) per cube file).
    """
    entries, cubes = {}, {}
    for f in files:
        try:
            listed = backend.list_spectra(f, cfg)
        except (OSError, ValueError):  # group_objects says why it skips the file
            continue
        if sum(e.kind in _SPECTRUM_KINDS for e in listed) > 1:
            entries[str(f)] = listed
        if any(e.kind == 'cube' for e in listed):
            cubes[str(f)] = backend.cube_image(f)
    groups = backend.group_objects(files, cfg) if files else {}
    return groups, {name: backend.read_spectrum(fs[0], cfg) for name, fs in groups.items()}, entries, cubes


def _scrolled(widget: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(widget)
    return scroll


class ObjectTree(QTreeWidget):
    """The object list; says what to do when it is empty."""

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        if self.topLevelItemCount() == 0:
            p = QPainter(self.viewport())
            p.setPen(self.palette().color(QPalette.ColorRole.PlaceholderText))
            p.drawText(self.viewport().rect().adjusted(12, 0, -12, 0),
                       int(Qt.AlignmentFlag.AlignCenter) | int(Qt.TextFlag.TextWordWrap), TREE_EMPTY)


class MainWindow(QMainWindow):
    """rvfitter main window: inputs, settings, run control, results, diagnostics, regions and log."""

    def __init__(self, backend: Backend | None = None, settings: QSettings | None = None):
        super().__init__()
        self.backend: Backend = backend or LazyBackend()
        self.settings = settings or QSettings()
        self.setWindowTitle(f'rvfitter {self.backend.version()}')
        self.setAcceptDrops(True)

        self.groups: dict[str, list[str]] = {}
        self.entries: dict[str, list[Any]] = {}  # multi-object file -> its spectra (io.SpectrumEntry)
        self.cubes: dict[str, tuple[Any, tuple[int, int]]] = {}  # IFU cube file -> (collapsed image, brightest)
        self._regroups = 0  # selection changes so far: only the latest regrouping is applied
        self.results: list[Any] = []
        self.worker: RunWorker | None = None
        self._jobs: list[JobWorker] = []
        self._logger_level: int | None = None
        self.current_object: str | None = None
        self._run_cfg: Any = None
        self.guidance: dict[str, list[dict]] = {}  # object -> suggested refits
        self._refitting = False
        self.first_spectra: dict[str, Any] = {}  # object -> its first spectrum (R and where it came from)
        self._steps: dict[str, Any] = {}  # progress of the run: planned steps per object, done, current object

        self._build_log()
        self._build_ui()
        self._build_menus()
        self._restore_layout()
        self.statusBar().showMessage('Open spectra to begin.')
        self._update_summary()

    # ---- construction --------------------------------------------------------------------------------------
    def _build_log(self) -> None:
        self.log_handler = QtLogHandler()
        self.log_pane = LogPane(self.log_handler)
        root = logging.getLogger('rvfitter')
        root.addHandler(self.log_handler)
        self._logger_level = root.level
        if root.level == logging.NOTSET or root.level > logging.DEBUG:
            root.setLevel(logging.DEBUG)

    def _build_ui(self) -> None:
        cfg = self.backend.default_config()

        # left: the objects
        self.input_tree = ObjectTree()
        self.input_tree.setColumnCount(2)
        self.input_tree.setHeaderLabels(['Objects', 'R'])
        self.input_tree.header().setStretchLastSection(False)
        self.input_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.input_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.input_tree.setFont(theme.tabular(self.input_tree.font()))
        self.input_tree.setToolTip('Spectra grouped into objects. Drop files, folders or archives here.')
        self.input_tree.itemSelectionChanged.connect(self._tree_selected)
        self.input_tree.itemChanged.connect(self._entry_toggled)
        self.input_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        for text, mode in (('Tick all usable', 'all'), ('Target only', 'target')):
            act = QAction(text, self.input_tree)
            act.triggered.connect(lambda _c=False, m=mode: self.set_objects(m))
            self.input_tree.addAction(act)

        # toolbar
        self.open_btn = QToolButton(text='Open spectra')
        self.open_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.open_btn.setToolTip('Open spectra from files, a folder or an archive (or drop them on the window).')
        menu = QMenu(self.open_btn)
        for text, slot, tip in (
            ('Files…', self.open_file, 'Open one or more spectrum files.'),
            ('Folder…', self.open_folder, 'Open every spectrum in a folder (searched recursively).'),
            ('Archive…', self.open_archive, 'Open a .zip or .tar/.tar.gz/.tgz archive of spectra.'),
        ):
            act = menu.addAction(text, slot)
            act.setToolTip(tip)
        self.open_btn.setMenu(menu)
        self.measure_btn = QPushButton('Measure RVs')
        self.measure_btn.setDefault(True)
        font = self.measure_btn.font()
        font.setBold(True)
        self.measure_btn.setFont(font)
        pal = self.measure_btn.palette()
        pal.setColor(QPalette.ColorRole.ButtonText, theme.colour('primary', pal))
        self.measure_btn.setPalette(pal)
        self.measure_btn.setToolTip('Measure radial velocities for all loaded objects (Ctrl+Return).')
        self.measure_btn.clicked.connect(self.run)
        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.setToolTip('Stop measuring after the object currently being measured.')
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(PROGRESS_WIDTH)
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        self.summary = QLabel()
        self.summary.setToolTip('Objects loaded, and how many are flagged or failed after measuring.')
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb = QToolBar('Toolbar')
        tb.setMovable(False)
        for w in (self.open_btn, self.measure_btn, self.cancel_btn, self.progress, spacer, self.summary):
            tb.addWidget(w)
        self.addToolBar(tb)

        # tabs
        self.tabs = QTabWidget()
        self.setup = SetupPanel(cfg, self.backend.template_status)
        self.setup.advancedRequested.connect(self.advanced_settings)
        self.setup.cubeRequested.connect(self.choose_cube)
        self.tabs.addTab(_scrolled(self.setup), 'Setup')

        self.model = PandasModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSortingEnabled(False)
        self.table.setFont(theme.tabular(self.table.font()))
        self.table.clicked.connect(self._table_clicked)
        self.export_btn = QPushButton('Export…')
        self.export_btn.setToolTip('Save the results table (csv, json, tsv or html).')
        self.export_btn.clicked.connect(self.export_results)
        res = QWidget()
        rv = QVBoxLayout(res)
        row = QHBoxLayout()
        self.count_label = QLabel('No results yet. Press Measure RVs.')
        row.addWidget(self.count_label)
        row.addStretch(1)
        row.addWidget(self.export_btn)
        rv.addLayout(row)
        rv.addWidget(self.table, 1)
        self._build_guidance(rv)
        self.tabs.addTab(res, 'Results')

        self.diagnostics = DiagnosticsPanel()
        self.tabs.addTab(self.diagnostics, 'Diagnostics')
        self.region_editor = RegionEditor()
        self.region_editor.overridesChanged.connect(self._overrides_changed)
        self.tabs.addTab(self.region_editor, 'Regions')
        self.tabs.currentChanged.connect(self._tab_changed)

        self.top_split = QSplitter(Qt.Orientation.Horizontal)
        self.top_split.addWidget(self.input_tree)
        self.top_split.addWidget(self.tabs)
        self.top_split.setStretchFactor(1, 1)
        self.top_split.setCollapsible(1, False)
        self.main_split = QSplitter(Qt.Orientation.Vertical)
        self.main_split.addWidget(self.top_split)
        self.main_split.addWidget(self.log_pane)
        self.log_pane.toggle.toggled.connect(self._log_toggled)
        self.main_split.setStretchFactor(0, 1)
        self.setCentralWidget(self.main_split)

    def _log_toggled(self, on: bool) -> None:
        total = sum(self.main_split.sizes())
        self.main_split.setSizes([total - LOG_HEIGHT, LOG_HEIGHT] if on else [total, 0])

    def _restore_layout(self) -> None:
        """80 per cent of the screen (capped), centred, narrow file list, small log; then the saved state."""
        area = (self.screen() or QApplication.primaryScreen()).availableGeometry()
        w, h = min(int(0.8 * area.width()), MAX_SIZE[0]), min(int(0.8 * area.height()), MAX_SIZE[1])
        self.resize(w, h)
        self.move(area.center() - self.rect().center())
        left = max(220, min(260, w // 4))
        self.top_split.setSizes([left, w - left])
        self.main_split.setSizes([h, 0])
        for name, widget in (('geometry', self), ('top_split', self.top_split), ('main_split', self.main_split)):
            state = self.settings.value(name)
            if state is not None:
                (widget.restoreGeometry if widget is self else widget.restoreState)(state)

    def _build_guidance(self, layout: QVBoxLayout) -> None:
        """Banner, details pane and the 'Download grid & refit' controls for the selected object."""
        self.guidance_banner = QLabel()
        self.guidance_banner.setWordWrap(True)
        self.guidance_banner.setBackgroundRole(QPalette.ColorRole.ToolTipBase)  # palette-aware note colours
        self.guidance_banner.setForegroundRole(QPalette.ColorRole.ToolTipText)
        self.guidance_banner.setAutoFillBackground(True)
        self.guidance_banner.setFrameShape(QFrame.Shape.StyledPanel)
        self.guidance_banner.setMargin(4)
        self.guidance_banner.setVisible(False)
        self.guidance_details = QPlainTextEdit()
        self.guidance_details.setReadOnly(True)
        self.guidance_details.setMaximumHeight(110)
        self.guidance_details.setToolTip('What the fit found at the edge of its search and what to do about it.')
        self.guidance_details.setVisible(False)
        self.guidance_combo = QComboBox()
        self.guidance_combo.setToolTip('The suggested change to apply to the selected object.')
        self.refit_btn = QPushButton('Download grid && refit')
        self.refit_btn.setToolTip('Download the suggested template grid (if needed), then measure this object again '
                                  'with the patched configuration. The change is recorded in the saved analysis.')
        self.refit_btn.clicked.connect(self.refit_selected)
        self.guidance_row = QWidget()
        gl = QHBoxLayout(self.guidance_row)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.addWidget(self.guidance_combo, 1)
        gl.addWidget(self.refit_btn)
        self.guidance_row.setVisible(False)
        layout.addWidget(self.guidance_banner)
        layout.addWidget(self.guidance_details)
        layout.addWidget(self.guidance_row)

    def _build_menus(self) -> None:
        m = self.menuBar().addMenu('&File')
        for text, slot, shortcut in (
            ('Open files…', self.open_file, QKeySequence.StandardKey.Open),
            ('Open folder…', self.open_folder, None),
            ('Open archive (zip/tar)…', self.open_archive, None),
            (None, None, None),
            ('Measure RVs', self.run, 'Ctrl+Return'),
            (None, None, None),
            ('Open settings…', self.load_config_dialog, None),
            ('Save settings…', self.save_config_dialog, QKeySequence.StandardKey.Save),
            ('Export results…', self.export_results, None),
            (None, None, None),
            ('Quit', self.close, QKeySequence.StandardKey.Quit),
        ):
            if text is None:
                m.addSeparator()
                continue
            act = QAction(text, self)
            if shortcut:
                act.setShortcut(shortcut)
            act.triggered.connect(slot)
            m.addAction(act)
        h = self.menuBar().addMenu('&Help')
        qs = QAction('Quick start', self)
        qs.triggered.connect(self.show_quick_start)
        ab = QAction('About rvfitter', self)
        ab.triggered.connect(self.show_about)
        h.addAction(qs)
        h.addAction(ab)

    # ---- helpers -------------------------------------------------------------------------------------------
    def _last_dir(self, key: str) -> str:
        return str(self.settings.value(f'lastdir/{key}', str(Path.home())))

    def _remember(self, key: str, path: str | Path) -> None:
        p = Path(path)
        self.settings.setValue(f'lastdir/{key}', str(p if p.is_dir() else p.parent))

    def current_config(self) -> Any:
        """The session config, with the Setup boxes applied."""
        return self.setup.get_config()

    def set_config(self, cfg: Any) -> None:
        before = self.current_config().input
        self.setup.set_config(cfg)
        if (cfg.input.objects, cfg.input.cube) != (before.objects, before.cube):
            self._regroup([*self.entries, *self.cubes])

    def advanced_settings(self) -> QDialog:
        """Every setting (the generated form) in a dialog; OK applies it to the session config."""
        dlg = QDialog(self)
        dlg.setWindowTitle('Advanced settings')
        form = ConfigForm(self.current_config())
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay = QVBoxLayout(dlg)
        lay.addWidget(_scrolled(form), 1)
        lay.addWidget(buttons)
        dlg.accepted.connect(lambda: self.set_config(form.get_config()))
        dlg.finished.connect(dlg.deleteLater)
        dlg.form = form
        dlg.resize(720, 640)
        dlg.open()
        return dlg

    def _error(self, title: str, msg: str) -> None:
        log.error('%s: %s', title, msg)
        QMessageBox.warning(self, title, msg)

    def _job(self, func, on_ok, on_fail=None) -> JobWorker:
        job = JobWorker(func, self)
        job.succeeded.connect(on_ok)
        job.failed.connect(on_fail or (lambda msg: self._error('Error', msg)))
        job.finished.connect(lambda j=job: self._jobs.remove(j) if j in self._jobs else None)
        self._jobs.append(job)
        job.start()
        return job

    # ---- input ---------------------------------------------------------------------------------------------
    def open_file(self) -> None:
        names, _ = QFileDialog.getOpenFileNames(self, 'Open spectra', self._last_dir('input'))
        if names:
            self._remember('input', names[0])
            self.load_paths([Path(n) for n in names])

    def open_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, 'Open folder of spectra', self._last_dir('input'))
        if d:
            self._remember('input', d)
            self.load_paths([Path(d)])

    def open_archive(self) -> None:
        name, _ = QFileDialog.getOpenFileName(self, 'Open archive', self._last_dir('input'),
                                              'Archives (*.zip *.tar *.tar.gz *.tgz *.tar.bz2 *.tar.xz);;All files (*)')
        if name:
            self._remember('input', name)
            self.load_paths([Path(name)])

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.load_paths(paths)

    def load_paths(self, paths: list[Path]) -> JobWorker:
        """Collect spectra from files, folders and archives, group them by object, read each object's first
        spectrum (for the Setup boxes and the R column) and list them."""
        cfg = self.current_config()
        backend = self.backend
        log.info('Loading %d input%s…', len(paths), '' if len(paths) == 1 else 's')

        def job() -> tuple[dict, dict, dict, dict]:
            files: list[Path] = []
            for p in paths:
                try:
                    files.extend(backend.collect_inputs(p))
                except (OSError, ValueError) as exc:  # one bad input never stops the others
                    log.warning('Could not read %s: %s', p, exc)
            return _scan(backend, files, cfg)

        return self._job(job, self._loaded)

    def _loaded(self, loaded: tuple[dict, dict, dict, dict]) -> None:
        groups, first = loaded[:2]
        if not groups and not loaded[2]:
            log.warning('No usable spectra found.')
            self.statusBar().showMessage('No usable spectra found. Check that the files are 1D spectra.')
            return
        self._merge(loaded)
        if first:
            self._autofill(first)
        # sources of multi-object files and cubes are 'file.fits[hdu]': count the files, not the entries
        n = len({re.sub(r'\[[^\]]*\]$', '', str(s)) for v in self.groups.values() for s in v})
        log.info('Loaded %d file%s in %d object%s.', n, '' if n == 1 else 's', len(self.groups),
                 '' if len(self.groups) == 1 else 's')
        self.statusBar().showMessage(f'{n} files, {len(self.groups)} objects loaded.')

    def _merge(self, loaded: tuple[dict, dict, dict, dict]) -> None:
        """Add the objects, spectra and cubes of a `_scan` to the session and show them"""
        groups, first, entries, cubes = loaded
        for name, files in groups.items():
            known = self.groups.setdefault(name, [])
            known.extend(f for f in files if f not in known)
        self.first_spectra.update(first)
        self.entries.update(entries)
        self.cubes.update(cubes)
        self.setup.show_cube(next(iter(self.cubes.values()))[1] if self.cubes else None)
        self._fill_tree()
        self._update_summary()

    def _regroup(self, files: list[str]) -> JobWorker | None:
        """Read ``files`` again with the current input.objects and input.cube, replacing their objects"""
        if not files:
            return None
        cfg, backend = self.current_config(), self.backend
        self._regroups += 1
        n = self._regroups
        self.measure_btn.setEnabled(False)  # until the objects follow the new selection

        def ok(loaded: tuple[dict, dict, dict, dict]) -> None:
            if n != self._regroups:  # a later change is on its way
                return
            for name in list(self.groups):
                self.groups[name] = [s for s in self.groups[name] if _file_of(s) not in files]
                if not self.groups[name]:
                    del self.groups[name]
                    self.first_spectra.pop(name, None)
            self._merge(loaded)
            self.measure_btn.setEnabled(self.worker is None)

        def fail(msg: str) -> None:
            self.measure_btn.setEnabled(self.worker is None)
            self._error('Error', msg)

        return self._job(lambda: _scan(backend, [Path(f) for f in files], cfg), ok, fail)

    def set_objects(self, objects: str | list[str]) -> JobWorker | None:
        """Set input.objects (from the tree's ticks or its context menu) and read the multi-object files again"""
        self.setup.cfg.input.objects = objects
        log.info('input.objects = %r', objects)
        return self._regroup(list(self.entries))

    def _entry_toggled(self, item: QTreeWidgetItem, column: int) -> None:
        """A ticked or unticked spectrum: input.objects becomes the names of every ticked one (a name ticked in
        one file selects it in every file, as the same source observed again)"""
        if column != 0 or item.data(0, ENTRY_ROLE) is None or \
                (item.checkState(0) == Qt.CheckState.Checked) == bool(item.data(0, ROW_ROLE)):
            return  # not a spectrum, or not a change of its tick (measured rows are the ticked ones)
        ticked = []
        it = QTreeWidgetItemIterator(self.input_tree)
        while it.value():
            e = it.value().data(0, ENTRY_ROLE)
            if e is not None and it.value().checkState(0) == Qt.CheckState.Checked and e not in ticked:
                ticked.append(e)
            it += 1
        self.set_objects(ticked)

    def choose_cube(self) -> CubeDialog:
        """The IFU extraction dialog for the first loaded cube; OK sets input.cube and reads the cubes again"""
        path, (image, brightest) = next(iter(self.cubes.items()))
        cfg = self.current_config()

        def preview(position: tuple[float, float] | None, radius: float) -> Any:
            c = copy.deepcopy(cfg)
            c.input.cube.position, c.input.cube.radius = position, radius
            return self.backend.read_spectrum(path, c)

        dlg = CubeDialog(image, brightest, cfg.input.cube, preview, Path(path).name, self)

        def accepted() -> None:
            cube = self.setup.cfg.input.cube
            cube.position, cube.radius = dlg.position, dlg.radius
            log.info('input.cube: position %s, radius %g spaxels', dlg.position or 'brightest spaxel', dlg.radius)
            self.setup.show_cube(brightest)
            self._regroup(list(self.cubes))

        dlg.accepted.connect(accepted)
        dlg.finished.connect(dlg.deleteLater)
        dlg.open()
        return dlg

    def _autofill(self, first: dict[str, Any]) -> None:
        """Fill the Setup boxes from the first object of this load; warn if the objects need different defaults."""
        regimes = {name: self.backend.regime(spec) for name, spec in first.items()}
        name, spec = next(iter(first.items()))
        if len(set(regimes.values())) > 1:
            log.warning('The objects fall into different regimes (%s); the Setup boxes follow %s. Each spectrum still '
                        'uses its own R.', ', '.join(f'{n}: {r}' for n, r in regimes.items()), name)
        meta = spec.meta
        instrument = meta.get('pypeit_spec') or next(iter(meta.get('instrument_names') or ()), None)
        description = (f'{instrument or "unknown instrument"}, {spec.wave[0]:,.0f}–{spec.wave[-1]:,.0f} Å (vacuum), '
                       f'{meta.get("medium_in", "?")} wavelengths in the file, {regimes[name]}')
        self.setup.autofill(name, spec, self.backend.default_for(spec), description)

    def _object_row(self, item: QTreeWidgetItem, name: str) -> None:
        """Make ``item`` the row of object ``name``: its R (then its RV), and selecting it selects the object"""
        spec = self.first_spectra[name]
        src = 'implausible' if spec.meta.get('resolution_implausible') else spec.meta.get('resolution_source', '')
        item.setText(1, f'{R_PREFIX.get(src, "R≈")}{spec.resolution:.0f}')
        item.setToolTip(1, f'Resolving power of {Path(self.groups[name][0]).name}: {src or "header"}'
                           f'{"" if src in ("header", "config") else ", check it on the Setup tab"}.')
        if src in ('estimated', 'implausible'):
            item.setForeground(1, theme.colour('flagged' if src == 'estimated' else 'failed',
                                               self.input_tree.palette()))
        item.setData(0, Qt.ItemDataRole.UserRole, name)
        item.setData(0, ROW_ROLE, True)
        item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def _fill_tree(self) -> None:
        """Objects first, each with its files; then each multi-object file with its spectra, ticked where they are
        measured (an object all of whose spectra come from such files is shown only there)"""
        self.input_tree.blockSignals(True)  # setting the ticks is not a selection change
        self.input_tree.clear()
        source_object = {s: name for name, files in self.groups.items() for s in files}
        for name, files in self.groups.items():
            if all(_file_of(f) in self.entries for f in files):
                continue
            top = QTreeWidgetItem([name])
            self._object_row(top, name)
            for f in files:
                child = QTreeWidgetItem([Path(f).name])
                child.setToolTip(0, str(f))
                child.setData(0, Qt.ItemDataRole.UserRole, name)
                top.addChild(child)
            self.input_tree.addTopLevelItem(top)
            top.setExpanded(True)
        for path, entries in self.entries.items():
            top = QTreeWidgetItem([Path(path).name])
            top.setToolTip(0, f'{path}: {len(entries)} extensions. Tick the spectra to measure; right-click for '
                              'all usable or the target only.')
            for e in entries:
                snr = 'S/N ?' if e.snr != e.snr else f'S/N {e.snr:.0f}'  # the panel is narrow: kind if not 'object'
                child = QTreeWidgetItem([f'{e.name}{f" ({e.kind})" * (e.kind != "object")} · {snr}'])
                name = source_object.get(f'{path}[{e.hdu}]')
                tip = f'HDU {e.hdu}, {e.kind}, {snr}: '
                if e.usable:
                    child.setData(0, ENTRY_ROLE, e.name)
                    child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    child.setToolTip(0, tip + (f'object {name}' if name else 'not measured'))
                else:  # visible, greyed and unticked, saying why
                    child.setFlags(child.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                    child.setToolTip(0, f'{tip}not usable: {e.reason}')
                child.setCheckState(0, Qt.CheckState.Checked if name else Qt.CheckState.Unchecked)
                if name:
                    self._object_row(child, name)
                top.addChild(child)
            self.input_tree.addTopLevelItem(top)
            top.setExpanded(True)
        self.input_tree.blockSignals(False)
        self._refresh_rows()

    def _adopted(self, name: str) -> tuple[float, float] | None:
        """The adopted RV and error [km/s] of ``name`` (NaN where not measured); None before it is measured."""
        df = self.model.df
        if len(df) == 0 or 'rv_adopted' not in df:
            return None
        rows = df[df['object'] == name]
        if rows.empty:
            return None
        row = rows.iloc[-1]
        return float(row['rv_adopted']), float(row.get('rv_adopted_err', float('nan')))

    def _state(self, name: str) -> str | None:
        """'failed' (no RV), 'flagged' (needs attention), 'ok', or None before measuring."""
        adopted = self._adopted(name)
        if adopted is None:
            return None
        if adopted[0] != adopted[0]:
            return 'failed'
        return 'flagged' if name in self.guidance else 'ok'

    def _refresh_rows(self) -> None:
        """Show each object's adopted RV, with a gold or rose dot where it needs attention or failed."""
        pal = self.input_tree.palette()
        self.input_tree.headerItem().setText(1, 'RV [km/s]' if len(self.model.df) else 'R')
        it = QTreeWidgetItemIterator(self.input_tree)
        while it.value():
            top = it.value()
            it += 1
            if not top.data(0, ROW_ROLE):
                continue
            name = top.data(0, Qt.ItemDataRole.UserRole)
            state = self._state(name)
            if state is None:
                continue
            top.setText(1, rv_text(*self._adopted(name)) or 'failed')
            top.setForeground(1, theme.colour('failed' if state == 'failed' else 'primary', pal))
            top.setIcon(0, theme.dot(state, pal) if state in ('flagged', 'failed') else QIcon())
            top.setToolTip(0, {'flagged': FLAGGED_TIP, 'failed': FAILED_TIP}.get(state, ''))
            top.setToolTip(1, 'Adopted radial velocity and its error.')

    def _update_summary(self) -> None:
        n = len(self.groups)
        states = [self._state(name) for name in self.groups]
        text = f'{n} object{"" if n == 1 else "s"}' if n else 'No objects loaded'
        for state in ('flagged', 'failed'):
            if state in states:
                text += f', {states.count(state)} {state}'
        self.summary.setText(text)

    # ---- configuration -------------------------------------------------------------------------------------
    def load_config_dialog(self) -> None:
        name, _ = QFileDialog.getOpenFileName(self, 'Load config', self._last_dir('config'),
                                              'TOML (*.toml);;All files (*)')
        if name:
            self._remember('config', name)
            self.load_config(Path(name))

    def load_config(self, path: Path) -> bool:
        try:
            self.set_config(self.backend.load_config(Path(path)))
        except Exception as exc:
            self._error('Could not load config', str(exc))
            return False
        log.info('Loaded config %s', path)
        return True

    def save_config_dialog(self) -> None:
        start = str(Path(self._last_dir('config')) / 'analysis.toml')
        name, _ = QFileDialog.getSaveFileName(self, 'Save analysis', start, 'TOML (*.toml)')
        if name:
            self._remember('config', name)
            self.save_config(Path(name))

    def save_config(self, path: Path) -> bool:
        """Write exactly the config that a CLI run would reproduce."""
        try:
            self.backend.save_config(self.current_config(), Path(path))
        except Exception as exc:
            self._error('Could not save config', str(exc))
            return False
        log.info('Saved analysis to %s', path)
        return True

    # ---- regions -------------------------------------------------------------------------------------------
    def _overrides_changed(self, overrides: dict) -> None:
        self.setup.cfg.regions.overrides = overrides
        log.info('Region overrides updated (%d edit%s).', len(overrides), '' if len(overrides) == 1 else 's')

    def load_regions_for(self, name: str) -> None:
        """Show the first spectrum of object ``name`` and its automatic regions in the editor."""
        files = self.groups.get(name)
        if not files:
            return
        cfg = self.current_config()
        try:
            spec = self.backend.read_spectrum(files[0], cfg)
            auto = self.backend.select_regions(spec, cfg, False)
            effective = self.backend.select_regions(spec, cfg, True) if cfg.regions.overrides else auto
        except Exception as exc:
            log.warning('Could not select regions for %s: %s', name, exc)
            self.region_editor.show_message(f'Regions are unavailable for {name}: {exc}')
            return
        self.region_editor.set_data(spec.wave, spec.flux, auto, effective)

    # ---- running -------------------------------------------------------------------------------------------
    def run(self) -> RunWorker | None:
        if self.worker is not None:
            return None
        if not self.groups:
            self._error('No spectra loaded', 'Use Open spectra, or drop files on the window, then press Measure RVs.')
            return None
        try:
            cfg = self.current_config()
            needs = self.backend.template_methods(cfg)
            missing = [g for g, t in self.backend.template_status(cfg).items() if t is None] if needs else []
        except Exception as exc:
            self._error('Invalid settings', str(exc))
            return None
        if not missing:
            return self._start(cfg)
        choice = self._ask_missing_templates(missing, needs)
        if choice == 'download':
            self._download_then_start(cfg, missing)
            return None
        if choice == 'without':
            cfg.methods.enabled = [m for m in cfg.methods.enabled if m not in needs]
            if not cfg.methods.enabled:
                self._error('Nothing to measure', f'Every enabled method ({", ".join(needs)}) needs the templates '
                            f'({", ".join(missing)}). Download them, or enable linecentre in the Setup tab.')
                return None
            log.warning('Template grid %s not downloaded: skipping %s for this run; measuring with %s.',
                        ', '.join(missing), ', '.join(needs), ', '.join(cfg.methods.enabled))
            return self._start(cfg)
        return None

    def _ask_missing_templates(self, missing: list[str], needs: list[str]) -> str:
        """'download', 'without' or 'cancel', from a dialog about the grids `missing` that methods `needs` read"""
        box = QMessageBox(QMessageBox.Icon.Question, 'Templates not downloaded',
                          f'The template grid {", ".join(missing)} is not downloaded, and {", ".join(needs)} '
                          'need it. Download it now (once; it is kept in the cache), or measure with the '
                          'model-free methods only?', parent=self)
        buttons = {box.addButton('Download and measure', QMessageBox.ButtonRole.AcceptRole): 'download',
                   box.addButton('Measure without template methods', QMessageBox.ButtonRole.ActionRole): 'without',
                   box.addButton(QMessageBox.StandardButton.Cancel): 'cancel'}
        box.exec()
        return buttons.get(box.clickedButton(), 'cancel')

    def _download_then_start(self, cfg: Any, grids: list[str]) -> JobWorker:
        """Download `grids` in the background (busy bar), then measure with `cfg`"""
        self.measure_btn.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.setFormat('Downloading templates…')
        self.progress.setVisible(True)
        self.statusBar().showMessage(f'Downloading template grid {", ".join(grids)}…')

        def ok(_none: None) -> None:
            self.setup.show_grid_status()
            self._start(cfg)

        def fail(msg: str) -> None:
            self.progress.setVisible(False)
            self.measure_btn.setEnabled(True)
            self.statusBar().showMessage('Template download failed.')
            self._error('Template download failed',
                        f'{msg}\n\nCheck the internet connection and press Measure RVs again; or choose "Measure '
                        'without template methods"; or point templates.local_dir (Advanced settings) at a grid '
                        'you already have.')

        return self._job(lambda: self.backend.download_templates(cfg, grids), ok, fail)

    def _start(self, cfg: Any) -> RunWorker:
        n_procs = cfg.parallel.n_procs
        self.results.clear()
        self.model.clear()
        self.guidance.clear()
        self._show_guidance(None)
        self._fill_tree()  # back to R: the RVs of the last run are gone
        self._update_summary()
        self.count_label.setText('Measuring…')
        self.worker = RunWorker(self.backend, cfg, dict(self.groups), n_procs, self)
        self.worker.result.connect(self._on_result)
        self.worker.step.connect(self._on_step)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._worker_finished)
        self.measure_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        # one step per (object, epoch, method); a finished object moves the bar to its end in any case
        planned = {name: len(files) * len(cfg.methods.enabled) for name, files in self.groups.items()}
        self._steps = {'planned': planned, 'base': 0, 'k': 0, 'object': None, 'n_objects': 0}
        self.progress.setRange(0, sum(planned.values()))
        self.progress.setValue(0)
        self.progress.setFormat('Starting…')
        self.progress.setVisible(True)
        self._run_cfg = cfg
        log.info('Measuring RVs: %d object%s, %s.', len(self.groups), '' if len(self.groups) == 1 else 's',
                 'processes chosen automatically' if n_procs is None else f'{n_procs} process(es)')
        self.worker.start()
        return self.worker

    def cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.cancel_btn.setEnabled(False)

    def _on_step(self, name: str, method: str, i: int, n: int) -> None:
        """A method is starting: the bar counts the steps done, the label names this one."""
        s = self._steps
        if name != s['object']:
            s['object'], s['k'] = name, 0
            s['n_objects'] += 1
        self.progress.setValue(min(s['base'] + s['k'], s['base'] + s['planned'].get(name, 0)))
        s['k'] += 1
        text = (f'{name} · {METHOD_LABELS.get(method, method)} ({i}/{n}) · '
                f'object {s["n_objects"]}/{len(s["planned"])}')
        self.progress.setFormat(text)
        self.statusBar().showMessage(f'Measuring {text}')

    def _on_result(self, res: Any) -> None:
        s = self._steps
        s['base'] += s['planned'].get(getattr(res, 'object', None), 0)
        self.progress.setValue(s['base'])
        self.results.append(res)
        self.model.append(self.backend.to_table([res]))
        self.count_label.setText(f'{len(self.results)} object{"" if len(self.results) == 1 else "s"} measured.')
        self._note_guidance(len(self.results) - 1, res)
        self._refresh_rows()
        self._update_summary()
        if len(self.results) == 1:
            self.table.selectRow(0)
            self.select_object(getattr(res, 'object', None))

    def _on_done(self, results: list, cancelled: bool) -> None:
        log.info('%s: %d result%s.', 'Cancelled' if cancelled else 'Finished', len(self.results),
                 '' if len(self.results) == 1 else 's')
        n = len(self.results)
        _, sep, tail = self.summary.text().partition(',')
        text = f'{n} object{"" if n == 1 else "s"}{sep}{tail}'
        self.statusBar().showMessage(f'Cancelled after {text}' if cancelled else f'Measured {text}')
        self._autowrite()

    def _on_failed(self, msg: str) -> None:
        self.statusBar().showMessage('Measuring failed.')
        QMessageBox.warning(self, 'Measuring failed', f'{msg}\n\nCheck the log, then press Measure RVs again.')

    def _worker_finished(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
        self.worker = None
        self.measure_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setVisible(False)

    def wait_for_run(self, timeout_ms: int = 30000) -> bool:
        w = self.worker
        return True if w is None else w.wait(timeout_ms)

    def _autowrite(self) -> None:
        cfg = self._run_cfg
        out = cfg.output.path if cfg is not None else None
        if not out or not self.results:
            return
        formats = cfg.output.formats or ['csv']
        try:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            self.backend.write_results(self.results, Path(out), list(formats))
            log.info('Wrote results to %s (%s).', out, ', '.join(formats))
        except OSError as exc:  # shown: the results are not saved
            log.error('Could not write results to %s: %s', out, exc)
            self.statusBar().showMessage(f'Could not save the results to {out}: {exc}')

    # ---- results and selection -----------------------------------------------------------------------------
    def export_results(self) -> None:
        if not self.results:
            self._error('No results to export', 'Press Measure RVs first, then export the table.')
            return
        start = str(Path(self._last_dir('export')) / 'results.csv')
        name, _ = QFileDialog.getSaveFileName(self, 'Export results', start,
                                              'CSV (*.csv);;JSON (*.json);;TSV (*.tsv);;HTML (*.html)')
        if name:
            self._remember('export', name)
            self.export_to(Path(name))

    def export_to(self, path: Path) -> bool:
        fmt = path.suffix.lstrip('.').lower() or 'csv'
        try:
            self.backend.write_results(self.results, path, [fmt])
        except Exception as exc:
            self._error('Export failed', str(exc))
            return False
        log.info('Exported %d results to %s', len(self.results), path)
        return True

    def _table_clicked(self, index) -> None:
        if 0 <= index.row() < len(self.results):
            self.select_object(getattr(self.results[index.row()], 'object', None))

    def _tree_selected(self) -> None:
        items = self.input_tree.selectedItems()
        if items:
            name = items[0].data(0, Qt.ItemDataRole.UserRole)
            self.select_object(name, from_tree=True)

    def select_object(self, name: str | None, from_tree: bool = False) -> None:
        if not name:
            return
        self.current_object = name
        res = next((r for r in self.results if getattr(r, 'object', None) == name), None)
        if res is not None:
            self._show_diagnostics(res)
        self._show_guidance(name)
        if self.tabs.currentWidget() is self.region_editor:
            self.load_regions_for(name)

    def _show_diagnostics(self, res: Any) -> None:
        rv, err = self._adopted(res.object) or (None, None)
        note = 'Needs attention: the best template is at the edge of the searched grid.' \
            if res.object in self.guidance else ''
        self.diagnostics.show_result(res, rv, err, note)

    # ---- template guidance and refits ----------------------------------------------------------------------
    def _note_guidance(self, row: int, res: Any) -> None:
        """Record the guidance of the result in ``row`` and mark the row."""
        name = getattr(res, 'object', '')
        sugg = self.backend.suggested_refits(res)
        if sugg:
            self.guidance[name] = sugg
            self.model.set_note(row, f'{name}: ' + ' | '.join(s['message'] for s in sugg))
        else:
            self.guidance.pop(name, None)
            self.model.set_note(row, None)

    def _show_guidance(self, name: str | None) -> None:
        sugg = self.guidance.get(name or '', [])
        self.guidance_banner.setVisible(bool(sugg))
        self.guidance_details.setVisible(bool(sugg))
        self.guidance_combo.clear()
        actionable = [s for s in sugg if s.get('action') in ('widen', 'download') and s.get('cfg_patch')]
        for s in actionable:
            self.guidance_combo.addItem(f'{s["action"]} {s["axis"]} ({s["direction"]})', s)
        self.guidance_row.setVisible(bool(actionable))
        self.refit_btn.setEnabled(bool(actionable) and not self._refitting)
        if not sugg:
            self.guidance_details.clear()
            return
        self.guidance_banner.setText(f'{name}: the best template is at the edge of the searched parameters '
                                     f'({len(sugg)} suggestion{"" if len(sugg) == 1 else "s"}).')
        self.guidance_details.setPlainText('\n\n'.join(
            s['message'] + (f'\nCLI: {s["command"]}' if s.get('command') else '') for s in sugg))

    def refit_selected(self) -> JobWorker | None:
        """Download the chosen suggestion's grid and refit the selected object in a worker thread."""
        name = self.current_object
        idx = next((i for i, r in enumerate(self.results) if getattr(r, 'object', None) == name), None)
        s = self.guidance_combo.currentData()
        if self._refitting or idx is None or s is None:
            return None
        cfg = self._run_cfg if self._run_cfg is not None else self.current_config()
        old, backend = self.results[idx], self.backend
        self._refitting = True
        self.refit_btn.setEnabled(False)
        self.statusBar().showMessage(f'Refitting {name}…')
        log.info('Refitting %s: %s', name, s['message'])

        def job() -> tuple[Any, Any]:
            return backend.refit(old, cfg, s)

        def ok(out: tuple[Any, Any]) -> None:
            new_res, new_cfg = out
            self._refitting = False
            self.results[idx] = new_res
            self.model.replace(idx, backend.to_table([new_res]))
            self._run_cfg = new_cfg
            self.set_config(new_cfg)  # the session config now carries the [[refits]] entry
            self._note_guidance(idx, new_res)
            self._refresh_rows()
            self._update_summary()
            self._show_diagnostics(new_res)
            self._show_guidance(name)
            self.statusBar().showMessage(f'Refitted {name}.')
            log.info('Refit of %s finished.', name)
            self._autowrite()

        def fail(msg: str) -> None:
            self._refitting = False
            self._show_guidance(name)
            self.statusBar().showMessage('Refit failed.')
            self._error('Refit failed', f'{msg}\n\nCheck the log, then try again.')

        return self._job(job, ok, fail)

    def _tab_changed(self, _i: int) -> None:
        name = self.current_object
        if self.tabs.currentWidget() is self.region_editor and name:
            self.load_regions_for(name)

    # ---- help ----------------------------------------------------------------------------------------------
    def show_quick_start(self) -> None:
        QMessageBox.information(self, 'Quick start', QUICK_START)

    def show_about(self) -> None:
        QMessageBox.about(self, 'About rvfitter',
                          f'<b>rvfitter {self.backend.version()}</b><br>'
                          'Radial velocities of brown dwarfs from reduced 1D spectra.<br>'
                          '<a href="https://github.com/Will-Cooper/rvfitter">github.com/Will-Cooper/rvfitter</a>')

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait(5000)
        for job in list(self._jobs):
            job.wait(2000 if not self._refitting else 60000)
        root = logging.getLogger('rvfitter')
        root.removeHandler(self.log_handler)
        if self._logger_level is not None:
            root.setLevel(self._logger_level)
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('top_split', self.top_split.saveState())
        self.settings.setValue('main_split', self.main_split.saveState())
        super().closeEvent(event)
