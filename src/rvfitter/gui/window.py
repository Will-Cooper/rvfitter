"""The main window."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableView,
    QTabWidget,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .backend import Backend, LazyBackend
from .cfgutil import find_field, get_path, set_path
from .diagnostics import DiagnosticsTabs
from .forms import ConfigForm
from .logpane import LogPane, QtLogHandler
from .regions_editor import RegionEditor
from .tablemodel import PandasModel
from .worker import JobWorker, RunWorker

log = logging.getLogger('rvfitter.gui')

QUICK_START = """
<h3>Quick start</h3>
<ol>
<li><b>Open</b> a spectrum, a folder of spectra, or a zip/tar archive (File menu), or drop them onto the window.
The spectra are grouped into objects in the list on the left.</li>
<li>Check the <b>Settings</b> tab. The defaults are sensible; hover over any setting for an explanation.
Tick the methods you want.</li>
<li>Optionally open the <b>Regions</b> tab, select an object, and drag, add or remove the shaded spans.
Your edits are stored in the analysis file.</li>
<li>Press <b>Run</b>. Each object appears in the <b>Results</b> table as soon as it is measured.
Press <b>Cancel</b> to stop after the current object.</li>
<li>Select an object to see its <b>Diagnostics</b>. Use <b>Export…</b> to save the table.</li>
<li><b>File &rarr; Save analysis (TOML)…</b> writes the exact configuration used.
Re-run it from the command line with <tt>rvfitter run analysis.toml</tt>.</li>
</ol>
"""

OVERRIDES_PATH = 'regions.overrides'


class MainWindow(QMainWindow):
    """rvfitter main window: inputs, settings, run control, results, diagnostics, regions and log."""

    def __init__(self, backend: Backend | None = None, settings: QSettings | None = None):
        super().__init__()
        self.backend: Backend = backend or LazyBackend()
        self.settings = settings or QSettings()
        self.setWindowTitle(f'rvfitter {self.backend.version()}')
        self.setAcceptDrops(True)
        self.resize(1300, 850)

        self.groups: dict[str, list[str]] = {}
        self.results: list[Any] = []
        self.worker: RunWorker | None = None
        self._jobs: list[JobWorker] = []
        self._logger_level: int | None = None
        self.current_object: str | None = None
        self._run_cfg: Any = None

        self._build_log()
        self._build_ui()
        self._build_menus()
        self.statusBar().showMessage('Open spectra to begin.')

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

        # left: inputs
        self.input_tree = QTreeWidget()
        self.input_tree.setHeaderLabels(['Input files (grouped by object)'])
        self.input_tree.setToolTip('Spectra grouped into objects. Drop files, folders or archives here.')
        self.input_tree.itemSelectionChanged.connect(self._tree_selected)
        btns = QHBoxLayout()
        for text, slot, tip in (
            ('Open file…', self.open_file, 'Open one spectrum file.'),
            ('Open folder…', self.open_folder, 'Open every spectrum in a folder (searched recursively).'),
            ('Open archive…', self.open_archive, 'Open a .zip or .tar/.tar.gz/.tgz archive of spectra.'),
        ):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            btns.addWidget(b)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.addLayout(btns)
        lv.addWidget(self.input_tree, 1)

        # run controls
        self.run_btn = QPushButton('Run')
        self.run_btn.setToolTip('Measure radial velocities for all loaded objects.')
        self.run_btn.clicked.connect(self.run)
        self.cancel_btn = QPushButton('Cancel')
        self.cancel_btn.setToolTip('Stop after the object currently being measured.')
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        tb = QToolBar('Run')
        tb.setMovable(False)
        tb.addWidget(self.run_btn)
        tb.addWidget(self.cancel_btn)
        tb.addWidget(self.progress)
        self.addToolBar(tb)

        # tabs
        self.tabs = QTabWidget()
        self.form = ConfigForm(cfg)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.form)
        self.tabs.addTab(scroll, 'Settings')

        self.region_editor = RegionEditor()
        self.region_editor.overridesChanged.connect(self._overrides_changed)
        self.tabs.addTab(self.region_editor, 'Regions')

        self.model = PandasModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setSortingEnabled(False)
        self.table.clicked.connect(self._table_clicked)
        self.export_btn = QPushButton('Export…')
        self.export_btn.setToolTip('Save the results table (csv, json, tsv or html).')
        self.export_btn.clicked.connect(self.export_results)
        res = QWidget()
        rv = QVBoxLayout(res)
        row = QHBoxLayout()
        self.count_label = QLabel('No results yet.')
        row.addWidget(self.count_label)
        row.addStretch(1)
        row.addWidget(self.export_btn)
        rv.addLayout(row)
        rv.addWidget(self.table, 1)
        self.tabs.addTab(res, 'Results')

        self.diagnostics = DiagnosticsTabs(self.backend)
        self.tabs.addTab(self.diagnostics, 'Diagnostics')
        self.tabs.currentChanged.connect(self._tab_changed)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(left)
        top.addWidget(self.tabs)
        top.setStretchFactor(1, 4)
        main = QSplitter(Qt.Orientation.Vertical)
        main.addWidget(top)
        main.addWidget(self.log_pane)
        main.setStretchFactor(0, 4)
        main.setSizes([620, 200])
        self.setCentralWidget(main)

    def _build_menus(self) -> None:
        m = self.menuBar().addMenu('&File')
        for text, slot, shortcut in (
            ('Open file…', self.open_file, QKeySequence.StandardKey.Open),
            ('Open folder…', self.open_folder, None),
            ('Open archive (zip/tar)…', self.open_archive, None),
            (None, None, None),
            ('Load config…', self.load_config_dialog, None),
            ('Save analysis (TOML)…', self.save_config_dialog, QKeySequence.StandardKey.Save),
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
        """The config described by the Settings tab."""
        return self.form.get_config()

    def set_config(self, cfg: Any) -> None:
        self.form.set_config(cfg)

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
        """Collect spectra from files, folders and archives, group them by object and list them."""
        cfg = self.current_config()
        backend = self.backend
        log.info('Loading %d input%s…', len(paths), '' if len(paths) == 1 else 's')

        def job() -> dict[str, list[str]]:
            files: list[Path] = []
            for p in paths:
                try:
                    files.extend(backend.collect_inputs(p))
                except Exception as exc:
                    log.warning('Could not read %s: %s', p, exc)
            if not files:
                return {}
            return backend.group_objects(files, cfg)

        return self._job(job, self._loaded)

    def _loaded(self, groups: dict[str, list[str]]) -> None:
        if not groups:
            log.warning('No usable spectra found.')
            self.statusBar().showMessage('No usable spectra found.')
            return
        for name, files in groups.items():
            known = self.groups.setdefault(name, [])
            known.extend(f for f in files if f not in known)
        self._fill_tree()
        n = sum(len(v) for v in self.groups.values())
        log.info('Loaded %d file%s in %d object%s.', n, '' if n == 1 else 's', len(self.groups),
                 '' if len(self.groups) == 1 else 's')
        self.statusBar().showMessage(f'{n} files, {len(self.groups)} objects loaded.')

    def _fill_tree(self) -> None:
        self.input_tree.clear()
        for name, files in self.groups.items():
            top = QTreeWidgetItem([f'{name}  ({len(files)})'])
            top.setData(0, Qt.ItemDataRole.UserRole, name)
            for f in files:
                child = QTreeWidgetItem([Path(f).name])
                child.setToolTip(0, str(f))
                child.setData(0, Qt.ItemDataRole.UserRole, name)
                top.addChild(child)
            self.input_tree.addTopLevelItem(top)
            top.setExpanded(True)

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
        if not self.form.set_value(OVERRIDES_PATH, overrides):
            try:
                set_path(self.form._cfg, OVERRIDES_PATH, overrides)
            except Exception as exc:
                log.warning('Config has no %s: %s', OVERRIDES_PATH, exc)
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
            overrides = get_path(cfg, OVERRIDES_PATH) or {}
            effective = self.backend.select_regions(spec, cfg, True) if overrides else auto
        except Exception as exc:
            log.warning('Could not select regions for %s: %s', name, exc)
            self.region_editor.info.setText(f'Regions unavailable for {name}: {exc}')
            return
        self.region_editor.set_data(spec.wave, spec.flux, auto, effective)

    # ---- running -------------------------------------------------------------------------------------------
    def run(self) -> RunWorker | None:
        if self.worker is not None:
            return None
        if not self.groups:
            self._error('Nothing to run', 'Open some spectra first.')
            return None
        try:
            cfg = self.current_config()
        except Exception as exc:
            self._error('Invalid settings', str(exc))
            return None
        n_procs = get_path(cfg, 'parallel.n_procs') or find_field(cfg, 'n_procs') or max(1, (os.cpu_count() or 2) - 1)
        self.results.clear()
        self.model.clear()
        self.count_label.setText('Running…')
        self.worker = RunWorker(self.backend, cfg, dict(self.groups), int(n_procs), self)
        self.worker.result.connect(self._on_result)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._worker_finished)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress.setRange(0, len(self.groups))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self._run_cfg = cfg
        log.info('Starting run: %d object%s, %d process%s.', len(self.groups), '' if len(self.groups) == 1 else 's',
                 n_procs, '' if n_procs == 1 else 'es')
        self.worker.start()
        return self.worker

    def cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.cancel_btn.setEnabled(False)

    def _on_result(self, res: Any) -> None:
        self.results.append(res)
        try:
            self.model.append(self.backend.to_table([res]))
        except Exception as exc:
            log.error('Could not tabulate %s: %s', getattr(res, 'object', '?'), exc)
        self.count_label.setText(f'{len(self.results)} object{"" if len(self.results) == 1 else "s"} measured.')
        if len(self.results) == 1:
            self.table.selectRow(0)
            self.select_object(getattr(res, 'object', None))

    def _on_progress(self, done: int, total: int) -> None:
        self.progress.setValue(done)
        self.statusBar().showMessage(f'Measured {done} of {total} objects.')

    def _on_done(self, results: list, cancelled: bool) -> None:
        log.info('%s: %d result%s.', 'Cancelled' if cancelled else 'Finished', len(self.results),
                 '' if len(self.results) == 1 else 's')
        self.statusBar().showMessage('Cancelled.' if cancelled else 'Finished.')
        self._autowrite()

    def _on_failed(self, msg: str) -> None:
        self.statusBar().showMessage('Run failed.')
        QMessageBox.warning(self, 'Run failed', msg)

    def _worker_finished(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
        self.worker = None
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress.setVisible(False)

    def wait_for_run(self, timeout_ms: int = 30000) -> bool:
        w = self.worker
        return True if w is None else w.wait(timeout_ms)

    def _autowrite(self) -> None:
        cfg = self._run_cfg
        out = get_path(cfg, 'output.path') if cfg is not None else None
        if not out or not self.results:
            return
        formats = get_path(cfg, 'output.formats') or ['csv']
        try:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            self.backend.write_results(self.results, Path(out), list(formats))
            log.info('Wrote results to %s (%s).', out, ', '.join(formats))
        except Exception as exc:
            log.warning('Could not write results to %s: %s', out, exc)

    # ---- results and selection -----------------------------------------------------------------------------
    def export_results(self) -> None:
        if not self.results:
            self._error('No results', 'There are no results to export yet.')
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
            self.diagnostics.show_result(res)
        if self.tabs.currentWidget() is self.region_editor:
            self.load_regions_for(name)

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
            job.wait(2000)
        root = logging.getLogger('rvfitter')
        root.removeHandler(self.log_handler)
        if self._logger_level is not None:
            root.setLevel(self._logger_level)
        self.settings.setValue('geometry', self.saveGeometry())
        super().closeEvent(event)
