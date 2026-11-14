"""GUI tests: offscreen Qt with a fake backend."""
import logging
import os
import threading

import pytest

pytest.importorskip('PySide6')
pytest.importorskip('pytestqt')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QSettings  # noqa: E402

from gui_fakes import FakeBackend, FakeConfig  # noqa: E402
from rvfitter.gui.window import MainWindow  # noqa: E402

pytestmark = pytest.mark.gui


@pytest.fixture
def backend(tmp_path):
    return FakeBackend(tmp_path)


@pytest.fixture
def win(qtbot, backend, tmp_path):
    w = MainWindow(backend, QSettings(str(tmp_path / 'q.ini'), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    yield w
    w.close()


@pytest.fixture
def spectra_dir(tmp_path):
    d = tmp_path / 'spectra'
    d.mkdir()
    for n in ('alpha_1', 'alpha_2', 'beta_1'):
        (d / f'{n}.txt').write_text('x')
    return d


def _load(qtbot, win, d):
    win.load_paths([d])
    qtbot.waitUntil(lambda: bool(win.groups), timeout=5000)


def test_window_builds(win):
    assert 'rvfitter' in win.windowTitle()
    assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == ['Setup', 'Results', 'Diagnostics', 'Regions']


def test_loading_populates_list(qtbot, win, spectra_dir):
    _load(qtbot, win, spectra_dir)
    assert set(win.groups) == {'alpha', 'beta'}
    assert win.input_tree.topLevelItemCount() == 2
    assert win.input_tree.topLevelItem(0).childCount() == 2


def test_settings_round_trip(win):
    cfg = FakeConfig()
    cfg.input.medium = 'air'
    cfg.velocity_frame.correction = 'skip'
    cfg.methods.enabled = ['template_fit']
    cfg.templates.local_dir = '/data/templates'
    cfg.templates.teff_prior = (1800.0, 2400.0)
    cfg.parallel.n_procs = 3
    cfg.parallel.seed = 7
    cfg.output.formats = ['csv', 'html']
    cfg.regions.overrides = {'ccf': [{'name': 'K I 7701', 'remove': True}]}
    win.set_config(cfg)
    assert win.current_config() == cfg


def test_every_setting_has_a_tooltip(win):
    from rvfitter.gui.forms import GroupEditor

    def walk(g):
        for name, ed in g.children.items():
            assert ed.widget.toolTip(), name
            if isinstance(ed, GroupEditor):
                walk(ed)
    walk(win.advanced_settings().form.root)
    assert all(ed.widget.toolTip() for ed in win.setup.editors.values())


def test_setup_autofill_from_the_spectra(qtbot, win, backend, spectra_dir):
    r_box = win.setup.editors['input.resolution']
    assert win.setup.tags['input.resolution'].text() == 'default'
    _load(qtbot, win, spectra_dir)
    assert r_box.edit.placeholderText() == '3000' and r_box.get() is None  # each spectrum keeps its own R
    assert win.setup.tags['input.resolution'].text() == 'from header'
    assert win.setup.tags['input.medium'].text() == 'from header (vacuum)'
    assert win.setup.editors['methods.enabled'].get() == ['ccf', 'template_fit']  # backend.default_for
    assert 'Detected for alpha' in win.setup.detected.text()
    assert win.current_config().input.resolution is None


def test_setup_estimated_r_is_flagged(qtbot, win, backend, spectra_dir):
    backend.resolution_source = 'estimated'
    _load(qtbot, win, spectra_dir)
    assert win.setup.tags['input.resolution'].text() == 'estimated: check'
    assert win.input_tree.topLevelItem(0).text(1) == 'R≈3000'


def test_setup_keeps_boxes_set_by_you_across_a_reload(qtbot, win, backend, spectra_dir):
    from PySide6.QtTest import QTest
    _load(qtbot, win, spectra_dir)
    QTest.keyClicks(win.setup.editors['input.resolution'].edit, '2602')
    win.setup.editors['methods.enabled'].boxes['linecentre'].click()
    assert win.setup.tags['input.resolution'].text() == 'set by you'
    want = win.current_config()
    backend.resolution, backend.resolution_source = 750., 'estimated'
    win.groups.clear()
    _load(qtbot, win, spectra_dir)
    got = win.current_config()
    assert got.input.resolution == 2602. and got.methods.enabled == want.methods.enabled
    assert 'linecentre' in got.methods.enabled
    assert win.setup.tags['input.resolution'].text() == win.setup.tags['methods.enabled'].text() == 'set by you'


def test_advanced_dialog_round_trip_keeps_the_config_hash(qtbot, tmp_path):
    from rvfitter import config
    be = FakeBackend(tmp_path)
    be.default_config = config.Config
    w = MainWindow(be, QSettings(str(tmp_path / 'a.ini'), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    before = w.current_config().hash()
    w.advanced_settings().accept()
    assert w.current_config().hash() == before == config.Config().hash()
    dlg = w.advanced_settings()
    assert dlg.form.set_value('seed', 7) and dlg.form.set_value('input.medium', 'air')
    dlg.accept()
    cfg = w.current_config()
    assert cfg.seed == 7 and w.setup.tags['input.resolution'].text() == 'default'
    assert cfg.input.medium == 'air' and w.setup.tags['input.medium'].text() == 'set by you'
    w.close()


def test_progress_bar_steps_per_method(qtbot, win, backend, spectra_dir):
    _load(qtbot, win, spectra_dir)  # the fake regime defaults: ccf and template_fit
    messages, values = [], []
    win.statusBar().messageChanged.connect(messages.append)
    win.progress.valueChanged.connect(values.append)
    win.run()
    qtbot.waitUntil(lambda: win.worker is None, timeout=10000)
    assert win.progress.maximum() == 6  # alpha: 2 files, beta: 1, times 2 methods
    labels = [m.removeprefix('Measuring ') for m in messages if m.startswith('Measuring ')]
    assert labels[0] == 'alpha · CCF (1/2) · object 1/2'
    assert labels[-1] == 'beta · Template fit (2/2) · object 2/2'
    assert len(labels) == 6 and values[-1] == 6 and values == sorted(values) and len(set(values)) >= 4


def test_save_and_load_config(win, tmp_path):
    cfg = FakeConfig()
    cfg.methods.enabled = ['template_fit', 'telluric_ref']
    cfg.parallel.seed = 99
    win.set_config(cfg)
    path = tmp_path / 'analysis.toml'
    assert win.save_config(path)
    win.set_config(FakeConfig())
    assert win.load_config(path)
    assert win.current_config() == cfg


def test_optional_count_auto_sentinel_is_next_to_valid_values(win):
    """n_procs: 'auto' (None) sits just below 1, so one arrow click switches between automatic and explicit"""
    from rvfitter.gui.forms import IntEditor
    ed = IntEditor(optional=True)
    ed.set(None)
    assert ed.get() is None and ed.widget.text() == 'auto'
    ed.widget.stepUp()
    assert ed.get() == 1
    ed.widget.stepDown()
    assert ed.get() is None


def test_run_fills_table(qtbot, win, backend, spectra_dir):
    _load(qtbot, win, spectra_dir)
    win.setup.cfg.parallel.n_procs = 2
    win.run()
    qtbot.waitUntil(lambda: win.worker is None, timeout=10000)
    assert win.model.rowCount() == 2
    assert list(win.model.df['object']) == ['alpha', 'beta']
    assert backend.runs[0]['n_procs'] == 2
    assert 'computed' in win.diagnostics.bary.text()


def test_run_with_automatic_processes(qtbot, win, backend, spectra_dir):
    _load(qtbot, win, spectra_dir)
    win.setup.cfg.parallel.n_procs = None
    win.run()
    qtbot.waitUntil(lambda: win.worker is None, timeout=10000)
    assert backend.runs[0]['n_procs'] is None and win.model.rowCount() == 2


def test_run_without_inputs_does_not_start(win, monkeypatch):
    monkeypatch.setattr(win, '_error', lambda *a: None)
    assert win.run() is None


def test_export(qtbot, win, backend, spectra_dir, tmp_path):
    _load(qtbot, win, spectra_dir)
    win.run()
    qtbot.waitUntil(lambda: win.worker is None, timeout=10000)
    assert win.export_to(tmp_path / 'out.json')
    assert backend.written[-1][1] == ('json',)


def test_log_messages_appear(qtbot, win):
    logging.getLogger('rvfitter.test').info('hello from the log')
    logging.getLogger('rvfitter.test').warning('careful now')
    qtbot.waitUntil(lambda: 'careful now' in win.log_pane.plain_text(), timeout=2000)
    assert 'hello from the log' in win.log_pane.plain_text()


def test_log_from_thread(qtbot, win):
    t = threading.Thread(target=lambda: logging.getLogger('rvfitter.t').error('from a thread'))
    t.start()
    t.join()
    qtbot.waitUntil(lambda: 'from a thread' in win.log_pane.plain_text(), timeout=2000)


def test_cancel_stops_worker(qtbot, win, backend, spectra_dir):
    backend.gate = threading.Event()
    _load(qtbot, win, spectra_dir)
    win.run()
    qtbot.waitUntil(lambda: win.model.rowCount() == 1, timeout=5000)
    assert win.cancel_btn.isEnabled()
    win.cancel_btn.click()
    backend.gate.set()
    qtbot.waitUntil(lambda: win.worker is None, timeout=10000)
    assert win.model.rowCount() == 1
    assert win.measure_btn.isEnabled()
    assert not win.cancel_btn.isEnabled()


def test_region_edits_land_in_overrides(qtbot, win, spectra_dir):
    _load(qtbot, win, spectra_dir)
    win.tabs.setCurrentWidget(win.region_editor)
    win.select_object('alpha')
    ed = win.region_editor
    k1, k2 = ('ccf', 'K I 7701'), ('ccf', 'K I 7667')
    assert set(ed.spans) == {k1, k2}
    assert win.current_config().regions.overrides == {}
    ed.move_span(k1, 7696.0, 7706.0)
    ed.remove_span(k2)
    key = ed.add_span(7800.0, 7810.0)
    ov = win.current_config().regions.overrides
    assert {'name': 'K I 7667', 'remove': True} in ov['ccf']
    moved = next(e for e in ov['ccf'] if e['name'] == 'K I 7701')
    assert (moved['wmin'], moved['wmax']) == (7696.0, 7706.0)
    assert next(e for e in ov['ccf'] if e['name'] == key[1])['wmin'] == 7800.0
    # reloading the editor from the config (via regions.select with overrides) reproduces the same edits
    win.load_regions_for('alpha')
    assert win.region_editor.overrides() == ov
    ed.reset()
    assert win.current_config().regions.overrides == {}


def test_region_mouse_drag(qtbot, win, spectra_dir):
    from matplotlib.backend_bases import MouseEvent
    _load(qtbot, win, spectra_dir)
    win.tabs.setCurrentWidget(win.region_editor)
    win.select_object('alpha')
    ed = win.region_editor
    ed.canvas.draw()
    ymid = 0.5 * sum(ed.ax.get_ylim())

    def ev(kind, x):
        px, py = ed.ax.transData.transform((x, ymid))
        MouseEvent(kind, ed.canvas, px, py, button=1)._process()
    ev('button_press_event', 7701.0)
    ev('motion_notify_event', 7721.0)
    ev('button_release_event', 7721.0)
    span = ed.spans[('ccf', 'K I 7701')]
    assert span.wmin == pytest.approx(7695 + 20, abs=1.0)
    assert win.current_config().regions.overrides['ccf'][0]['wmin'] == pytest.approx(span.wmin)


def test_missing_pyside_message(monkeypatch, capsys):
    from rvfitter.gui import app
    monkeypatch.setattr(app, '_qt_available', lambda: False)
    assert app.main([]) == 1
    assert 'pip install "rvfitter[gui]"' in capsys.readouterr().err


def test_help_and_about(win, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, 'information', lambda *a: shown.append(a[2]))
    monkeypatch.setattr(QMessageBox, 'about', lambda *a: shown.append(a[2]))
    win.show_quick_start()
    win.show_about()
    assert 'Quick start' in shown[0]
    assert '0.0-test' in shown[1]


def test_lazy_backend_uses_on_object_and_cancel(monkeypatch):
    from rvfitter import pipeline
    from rvfitter.gui.backend import LazyBackend
    seen = {}

    def fake_run(cfg, inputs, *, n_procs, on_object=None, on_step=None, cancel=None):
        seen['on_step'] = on_step
        seen['cancel'] = cancel
        seen['n'] = len(inputs)
        for name in ('a', 'b'):
            on_object(name)
        return ['a', 'b']

    monkeypatch.setattr(pipeline, 'run', fake_run, raising=False)
    got = []
    out = LazyBackend().run(FakeConfig(), {'x': ['f1', 'f2'], 'y': ['f3']}, 1, got.append, print, threading.Event())
    assert out == got == ['a', 'b']
    assert seen['n'] == 3
    assert not seen['cancel'].is_set()
    assert seen['on_step'] is print


def test_lazy_backend_honours_config(monkeypatch):
    from types import SimpleNamespace

    from rvfitter import archives, config, io, regions
    from rvfitter.gui.backend import LazyBackend
    seen = {}

    def fake_read(path, *, medium='auto', medium_method='ciddor', hdu=None, resolution=None):
        seen['read'] = (medium, medium_method, hdu, resolution)
        return SimpleNamespace(source=str(path))

    def fake_group(spectra, override=None):
        seen['override'] = override
        return {'obj': spectra}

    def fake_select(spec, *, teff=None, methods, overrides=None, options=None):
        seen['select'] = (teff, list(methods), overrides, options)
        return {}

    monkeypatch.setattr(io, 'read_spectrum', fake_read, raising=False)
    monkeypatch.setattr(archives, 'group_by_object', fake_group, raising=False)
    monkeypatch.setattr(regions, 'select', fake_select, raising=False)
    cfg = config.Config()
    cfg.input.medium, cfg.input.medium_method, cfg.input.hdu, cfg.input.resolution = 'air', 'edlen', 2, 4000.
    cfg.grouping.override = {'a': 'obj'}
    cfg.templates.teff_prior = 2000.
    cfg.methods.enabled = ['ccf']
    cfg.regions.overrides = {'ccf': [{'name': 'x', 'remove': True}]}
    be = LazyBackend()
    spec = be.read_spectrum('f.fits', cfg)
    assert seen['read'] == ('air', 'edlen', 2, 4000.)
    assert be.group_objects(['f.fits'], cfg) == {'obj': ['f.fits']}
    assert seen['override'] == {'a': 'obj'}
    be.select_regions(spec, cfg, True)
    assert seen['select'][:3] == (2000., ['ccf'], cfg.regions.overrides)


def test_diagnostic_figures_render(qtbot, win):
    from plotting_synth import synthetic_result
    win.diagnostics.show_result(synthetic_result())
    tabs = win.diagnostics.figures
    assert {tabs.tabText(i) for i in range(tabs.count())} >= {'line fits', 'ccf', 'adopted'}
    tabs.currentWidget().layout().itemAt(1).widget().draw()
    win.diagnostics.clear()
    assert tabs.count() == 0


def test_choice_fields_are_combo_boxes_from_config(qtbot):
    from PySide6.QtWidgets import QCheckBox, QComboBox

    from rvfitter import config
    from rvfitter.gui.forms import ConfigForm
    form = ConfigForm(config.Config())
    for path, choices in {
        'input.medium': config.INPUT_MEDIA, 'input.medium_method': tuple(config.AIR_METHODS),
        'velocity_frame.correction': config.VELOCITY_CORRECTIONS, 'velocity_frame.frame': config.VELOCITY_FRAMES,
        'calibration.wavelength_correction': config.WAVELENGTH_CORRECTIONS,
        'regions.template_windows': config.TEMPLATE_WINDOWS, 'output.plot_style': config.PLOT_STYLES,
        'output.medium': config.MEDIA,
    }.items():
        widget = form.editor(path).widget
        assert isinstance(widget, QComboBox), path
        assert [widget.itemText(i) for i in range(widget.count())] == list(choices)
    assert set(form.editor('output.formats').boxes) == set(config.OUTPUT_FORMATS)
    assert set(form.editor('methods.enabled').boxes) == set(config.METHOD_NAMES)
    assert isinstance(form.editor('input.telluric_corrected').widget, QCheckBox)
    assert form.editor('schema_version') is None
    assert form.get_config() == config.Config()


def test_window_state_saved(win):
    win.main_split.setSizes([500, 150])
    win.close()
    assert win.settings.value('main_split') is not None


def _measure(qtbot, win, d):
    _load(qtbot, win, d)
    win.run()
    qtbot.waitUntil(lambda: win.worker is None, timeout=10000)


def test_object_list_shows_rv_after_run(qtbot, win, spectra_dir):
    assert win.input_tree.topLevelItem(0) is None
    _load(qtbot, win, spectra_dir)
    assert win.input_tree.topLevelItem(0).text(1) == 'R 3000'
    win.run()
    qtbot.waitUntil(lambda: win.worker is None, timeout=10000)
    assert [win.input_tree.topLevelItem(i).text(1) for i in range(2)] == ['10.0 ± 0.5', '11.0 ± 0.5']
    assert win.summary.text() == '2 objects'
    assert win.statusBar().currentMessage() == 'Measured 2 objects'


def test_flagged_object_gets_gold_dot_and_summary(qtbot, win, backend, spectra_dir):
    backend.edge_objects = {'beta'}
    _measure(qtbot, win, spectra_dir)
    alpha, beta = (win.input_tree.topLevelItem(i) for i in range(2))
    assert alpha.icon(0).isNull() and not beta.icon(0).isNull()
    assert 'Gold dot' in beta.toolTip(0)
    assert win.summary.text() == '2 objects, 1 flagged'
    assert win.statusBar().currentMessage() == 'Measured 2 objects, 1 flagged'


def test_colour_roles_follow_palette_lightness():
    from PySide6.QtGui import QColor, QPalette

    from rvfitter.gui import theme
    light, dark = QPalette(), QPalette()
    light.setColor(QPalette.ColorRole.Window, QColor('#ffffff'))
    dark.setColor(QPalette.ColorRole.Window, QColor('#202020'))
    names = {r: (theme.colour(r, light).name().lower(), theme.colour(r, dark).name().lower()) for r in theme.COLOURS}
    assert names == {'primary': ('#004488', '#6699cc'), 'flagged': ('#ddaa33', '#ddaa33'),
                     'failed': ('#bb5566', '#ee99aa')}


def test_empty_states_give_directions(win):
    from rvfitter.gui.window import TREE_EMPTY
    assert TREE_EMPTY == 'Drop spectra, folders or .zip/.tar archives here, or use Open spectra.'
    assert not win.diagnostics.empty.isHidden() and win.diagnostics.content.isHidden()
    assert win.diagnostics.empty.text() == 'Select an object to see its fits.'
    assert not win.region_editor.empty.isHidden() and win.region_editor.content.isHidden()


def test_selecting_an_object_replaces_the_empty_state(qtbot, win, spectra_dir):
    _measure(qtbot, win, spectra_dir)
    assert win.diagnostics.empty.isHidden()
    assert win.diagnostics.readout.text() == '10.0 ± 0.5 km s⁻¹'
    win.tabs.setCurrentWidget(win.region_editor)
    assert win.region_editor.empty.isHidden()


def test_log_is_collapsed_and_counts_warnings(qtbot, win):
    assert not win.log_pane.toggle.isChecked() and win.log_pane.text.isHidden()
    assert win.log_pane.warn_label.text() == ''
    logging.getLogger('rvfitter.test').warning('one')
    logging.getLogger('rvfitter.test').error('two')
    qtbot.waitUntil(lambda: win.log_pane.warnings == 2, timeout=2000)
    assert win.log_pane.warn_label.text() == '2 warnings'
    win.log_pane.clear()
    assert win.log_pane.warn_label.text() == ''
