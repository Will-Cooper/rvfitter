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
    cfg.seed = 7
    cfg.output.formats = ['csv', 'html']
    cfg.regions.overrides = {'ccf': [{'name': 'K I 7701', 'remove': True}]}
    win.set_config(cfg)
    assert win.current_config() == cfg


def test_every_setting_has_a_tooltip(qtbot, win):
    """Every field of the real Config that the forms show has its own TIPS entry (building the form fails without
    one), and TIPS has no entry for a field that does not exist"""
    from rvfitter import config
    from rvfitter.gui.forms import MORE_TIP, TIPS, ConfigForm, GroupEditor
    form = ConfigForm(config.Config())
    qtbot.addWidget(form)
    shown = {}

    def walk(g, prefix):
        for name, ed in g.children.items():
            if isinstance(ed, GroupEditor):
                assert ed.widget.toolTip() == ed.cls.__doc__.strip().splitlines()[0]
                walk(ed, f'{prefix}{name}.')
            else:
                shown[prefix + name] = ed.widget.toolTip()
    walk(form.root, '')
    assert shown == {k: TIPS[k] + MORE_TIP for k in shown} and set(shown) == set(TIPS)
    assert all(ed.widget.toolTip() == TIPS[p] for p, ed in win.setup.editors.items())


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
    assert win.setup.recommended.text().startswith('Recommended for R 3000 (estimated), optical:')


def test_setup_recommends_methods_for_the_detected_r(qtbot, win, backend, spectra_dir):
    from PySide6.QtTest import QTest
    setup = win.setup
    boxes = setup.editors['methods.enabled'].boxes
    assert setup.recommended.text() == '' and setup.method_note.text() == ''  # nothing loaded yet
    _load(qtbot, win, spectra_dir)
    assert setup.recommended.text() == ('Recommended for R 3000, optical: Template fit, CCF, Line centring. '
                                        'Hover over a method for why.')
    assert 'needs no templates' in boxes['linecentre'].toolTip()
    assert setup.method_note.text() == ''  # the fake defaults (ccf, template_fit) are both recommended
    boxes['lsd'].click()  # allowed, with a gold note saying why it is not recommended
    assert boxes['lsd'].isChecked() and 'lsd' in win.current_config().methods.enabled
    assert setup.method_note.text() == 'LSD: Not recommended below R 20000: the mean line profile needs many ' \
                                       'resolved lines.'
    assert setup.method_note.styleSheet() == 'color: #ddaa33'
    QTest.keyClicks(setup.editors['input.resolution'].edit, '60000')  # a typed R moves the advice
    assert setup.recommended.text().startswith('Recommended for R 60000, optical: Template fit, Telluric reference.')
    assert setup.method_note.text().splitlines()[0].startswith('CCF: Optional: above R 20000')
    assert setup.method_note.text().splitlines()[1].startswith('LSD: Optional')
    boxes['lsd'].click()
    boxes['ccf'].click()
    assert setup.method_note.text() == ''


def test_setup_advice_follows_applied_settings(qtbot, win, backend, spectra_dir):
    """Advanced settings and opened settings files set R without an edit signal; the advice must still follow"""
    _load(qtbot, win, spectra_dir)
    cfg = win.current_config()
    cfg.input.resolution = 60000.
    win.set_config(cfg)
    assert win.setup.recommended.text().startswith('Recommended for R 60000, optical: Template fit, Telluric '
                                                   'reference.')
    assert win.setup.method_note.text().startswith('CCF: Optional: above R 20000')


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


def test_setup_autofill_keeps_method_options_from_settings(qtbot, win, backend, spectra_dir):
    cfg = win.current_config()
    cfg.methods.options = {'template_fit': {'fit_vsini': True}}
    win.set_config(cfg)  # an opened settings file, or the Advanced dialog
    _load(qtbot, win, spectra_dir)
    assert win.current_config().methods.options == {'template_fit': {'fit_vsini': True}}


def test_download_passes_every_configured_range(monkeypatch):
    from rvfitter import config
    from rvfitter import templates as T
    from rvfitter.gui.backend import LazyBackend
    calls = []
    monkeypatch.setattr(T, 'download', lambda grid, **kw: calls.append((grid, kw)))
    cfg = config.Config()
    cfg.templates.teff_range, cfg.templates.logg_range, cfg.templates.ranges = (1500, 2500), (4., 5.), {'co': (0.5, 1)}
    LazyBackend().download_templates(cfg, ['sand'])
    assert calls == [('sand', {'teff_range': (1500, 2500), 'logg_range': (4., 5.), 'ranges': {'co': (0.5, 1)}})]


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
    cfg.seed = 99
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


def test_setup_says_whether_the_grid_is_downloaded(win, backend):
    assert win.setup.grid_status.text() == 'downloaded, 1200–3000 K'
    backend.missing_grids = {'bt-settl-cifist'}
    win.setup.show_grid_status()
    assert win.setup.grid_status.text() == 'not downloaded: downloaded when you press Measure RVs'


def _missing_and_answer(win, backend, monkeypatch, answer):
    backend.missing_grids = {'bt-settl-cifist'}
    asked = []
    monkeypatch.setattr(win, '_ask_missing_templates', lambda missing, needs: asked.append((missing, needs)) or answer)
    return asked


def test_measure_with_missing_grid_downloads_then_measures(qtbot, win, backend, spectra_dir, monkeypatch):
    _load(qtbot, win, spectra_dir)
    asked = _missing_and_answer(win, backend, monkeypatch, 'download')
    win.run()
    qtbot.waitUntil(lambda: bool(backend.runs) and win.worker is None, timeout=10000)
    assert asked == [(['bt-settl-cifist'], ['ccf', 'template_fit'])]
    assert backend.downloads == [{'grids': ['bt-settl-cifist']}]
    assert backend.runs[0]['cfg'].methods.enabled == ['ccf', 'template_fit']
    assert win.setup.grid_status.text().startswith('downloaded')


def test_measure_with_missing_grid_download_failure_says_what_to_do(qtbot, win, backend, spectra_dir, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    _load(qtbot, win, spectra_dir)
    _missing_and_answer(win, backend, monkeypatch, 'download')
    backend.download_error = 'SVO unreachable'
    shown = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a: shown.append(a[2]))
    win.run()
    qtbot.waitUntil(lambda: bool(shown), timeout=10000)
    assert 'SVO unreachable' in shown[0] and 'local_dir' in shown[0]
    assert not backend.runs and win.measure_btn.isEnabled()


def test_measure_without_template_methods(qtbot, win, backend, spectra_dir, monkeypatch, caplog):
    _load(qtbot, win, spectra_dir)
    win.setup.editors['methods.enabled'].boxes['linecentre'].click()
    _missing_and_answer(win, backend, monkeypatch, 'without')
    with caplog.at_level(logging.WARNING):
        win.run()
        qtbot.waitUntil(lambda: win.worker is None, timeout=10000)
    assert backend.runs[0]['cfg'].methods.enabled == ['linecentre']
    assert any(r.levelno == logging.WARNING and 'skipping ccf, template_fit' in r.getMessage() for r in caplog.records)
    assert 'linecentre' in win.current_config().methods.enabled and 'ccf' in win.current_config().methods.enabled


def test_measure_without_template_methods_and_none_left(qtbot, win, backend, spectra_dir, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    _load(qtbot, win, spectra_dir)
    _missing_and_answer(win, backend, monkeypatch, 'without')
    shown = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a: shown.append(a[1]))
    assert win.run() is None
    assert shown == ['Nothing to measure'] and not backend.runs


def test_lazy_backend_template_status_and_methods(tmp_path, monkeypatch):
    import numpy as np

    from rvfitter import config
    from rvfitter import templates as T
    from rvfitter.gui.backend import LazyBackend
    monkeypatch.setenv('RVFITTER_CACHE', str(tmp_path))
    cfg, be = config.Config(), LazyBackend()
    cfg.templates.grid = ['bt-settl-cifist', 'sphinx']
    grid = tmp_path / 'bt-settl-cifist'
    grid.mkdir()
    for teff in (1200, 3000):
        np.savetxt(grid / f'lte{teff / 100:05.1f}-5.0.txt', np.c_[[7000., 8000.], [1., 1.]])
    T.write_metadata(grid, T.KNOWN_GRIDS['bt-settl-cifist'])
    (grid / 'manifest.json').write_text('{}')  # written last by a finished download
    (tmp_path / 'sphinx').mkdir()  # a failed download's leftovers: missing, so Measure RVs offers it again
    assert be.template_status(cfg) == {'bt-settl-cifist': (1200., 3000.), 'sphinx': None}
    cfg.methods.enabled = ['linecentre', 'ccf', 'lsd']
    cfg.methods.options = {'lsd': {'template': 'star.fits'}}
    assert be.template_methods(cfg) == ['ccf']
    cfg.methods.options = {'linecentre': {'shape_correction': True}}
    assert be.template_methods(cfg) == ['linecentre', 'ccf', 'lsd']


def test_measure_with_missing_grid_cancelled(qtbot, win, backend, spectra_dir, monkeypatch):
    _load(qtbot, win, spectra_dir)
    _missing_and_answer(win, backend, monkeypatch, 'cancel')
    assert win.run() is None and not backend.runs and not backend.downloads


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

    def fake_read(path, *, medium='auto', medium_method='ciddor', hdu=None, resolution=None, cube=None):
        seen['read'] = (medium, medium_method, hdu, resolution, cube)
        return SimpleNamespace(source=str(path))

    def fake_read_many(path, *, objects='auto', **kw):
        seen['objects'] = objects
        return [fake_read(path, **kw)]

    def fake_group(spectra, override=None):
        seen['override'] = override
        return {'obj': spectra}

    def fake_select(spec, *, teff=None, methods, overrides=None, options=None):
        seen['select'] = (teff, list(methods), overrides, options)
        return {}

    monkeypatch.setattr(io, 'read_spectrum', fake_read, raising=False)
    monkeypatch.setattr(io, 'read_spectra', fake_read_many, raising=False)
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
    assert seen['read'] == ('air', 'edlen', 2, 4000., cfg.input.cube)
    assert be.group_objects(['f.fits'], cfg) == {'obj': ['f.fits']}
    assert seen['objects'] == 'auto'
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


def test_text_or_number_fields_are_plain_boxes(qtbot):
    """slit_prior ('auto' or km/s) and hdu (index or name) are shown without JSON quotes and parse to their type"""
    from rvfitter import config
    from rvfitter.gui.forms import ConfigForm
    form = ConfigForm(config.Config())
    slit = form.editor('calibration.slit_prior')
    assert slit.edit.text() == 'auto'
    slit.edit.setText('2.5')
    assert form.get_config().calibration.slit_prior == 2.5
    hdu = form.editor('input.hdu')
    hdu.edit.setText('3')
    assert form.get_config().input.hdu == 3
    hdu.edit.setText('SCI')
    assert form.get_config().input.hdu == 'SCI'


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


@pytest.fixture
def mos_dir(tmp_path, backend):
    """A multi-object file 'mos' (two usable sources, one without valid pixels, a background slit) and 'single'"""
    from rvfitter.io import SpectrumEntry
    d = tmp_path / 'mos'
    d.mkdir()
    for n in ('mos', 'single_1'):
        (d / f'{n}.txt').write_text('x')
    backend.multi['mos'] = [SpectrumEntry(1, 'src_a', 'object', 20., True, named=True),
                            SpectrumEntry(2, 'src_b', 'object', 8., True, named=True),
                            SpectrumEntry(3, 'src_c', 'object', float('nan'), False, 'fewer than two valid pixels',
                                          named=True),
                            SpectrumEntry(4, 'background', 'background', 3., False,
                                          'JWST background slit (SRCNAME background)')]
    return d


def _mos_item(win):
    return next(win.input_tree.topLevelItem(i) for i in range(win.input_tree.topLevelItemCount())
                if win.input_tree.topLevelItem(i).text(0) == 'mos.txt')


def test_multi_object_file_shows_its_spectra_with_reasons(qtbot, win, mos_dir):
    from PySide6.QtCore import Qt
    _load(qtbot, win, mos_dir)
    assert set(win.groups) == {'single', 'src_a', 'src_b'}
    top = _mos_item(win)
    rows = [top.child(i) for i in range(top.childCount())]
    assert [r.text(0) for r in rows] == ['src_a · S/N 20', 'src_b · S/N 8', 'src_c · S/N ?',
                                         'background (background) · S/N 3']
    assert [r.checkState(0) for r in rows] == [Qt.CheckState.Checked] * 2 + [Qt.CheckState.Unchecked] * 2
    assert not rows[2].flags() & Qt.ItemFlag.ItemIsEnabled  # greyed, with the reason
    assert rows[2].toolTip(0) == 'HDU 3, object, S/N ?: not usable: fewer than two valid pixels'
    assert rows[0].text(1) == 'R 3000'
    rows[0].setSelected(True)
    assert win.current_object == 'src_a'
    assert win.input_tree.topLevelItemCount() == 2  # 'single', then the file: its sources are shown only there


def test_ticks_set_input_objects_and_the_run_measures_only_them(qtbot, win, backend, mos_dir):
    from PySide6.QtCore import Qt
    _load(qtbot, win, mos_dir)
    _mos_item(win).child(1).setCheckState(0, Qt.CheckState.Unchecked)
    assert win.current_config().input.objects == ['src_a']
    qtbot.waitUntil(lambda: 'src_b' not in win.groups, timeout=5000)
    assert _mos_item(win).child(1).checkState(0) == Qt.CheckState.Unchecked
    assert win.measure_btn.isEnabled()
    win.run()
    qtbot.waitUntil(lambda: win.worker is None, timeout=10000)
    assert backend.runs[-1]['groups'] == {'single': [str(mos_dir / 'single_1.txt')],
                                          'src_a': [f'{mos_dir / "mos.txt"}[1]']}
    actions = {a.text(): a for a in win.input_tree.actions()}
    actions['Target only'].trigger()
    qtbot.waitUntil(lambda: win.current_config().input.objects == 'target' and 'src_a' in win.groups, timeout=5000)
    actions['Tick all usable'].trigger()
    qtbot.waitUntil(lambda: 'src_b' in win.groups, timeout=5000)
    assert win.current_config().input.objects == 'all'


def test_cube_row_and_dialog_write_input_cube(qtbot, win, backend, tmp_path):
    import numpy as np
    img = np.zeros((9, 11))
    img[6, 4] = 10.  # brightest spaxel (x, y) = (4, 6)
    backend.cubes['cube'] = img
    (tmp_path / 'cube.txt').write_text('x')
    setup = win.setup
    assert not setup.form.isRowVisible(setup._cube_row)
    _load(qtbot, win, tmp_path / 'cube.txt')
    assert setup.form.isRowVisible(setup._cube_row)
    assert setup.cube_text.text() == 'brightest spaxel (4, 6), radius 3 spaxels'
    dlg = win.choose_cube()
    assert backend.reads[-1][1:] == (None, 3.)  # the preview
    dlg.radius_box.setValue(2.)
    dlg.set_position((2., 3.))
    assert backend.reads[-1][1:] == ((2., 3.), 2.) and dlg.info.text() == 'Centre: spaxel (2, 3)'
    n = len(backend.reads)
    dlg.accept()
    cube = win.current_config().input.cube
    assert (cube.position, cube.radius) == ((2., 3.), 2.)
    assert setup.cube_text.text() == 'spaxel (2, 3), radius 2 spaxels'
    qtbot.waitUntil(lambda: len(backend.reads) > n, timeout=5000)  # the cube is read again with the aperture
    assert backend.reads[-1] == (str(tmp_path / 'cube.txt'), (2., 3.), 2.)
    dlg = win.choose_cube()
    dlg.reset()
    dlg.accept()
    assert win.current_config().input.cube.position is None
    assert setup.cube_text.text() == 'brightest spaxel (4, 6), radius 2 spaxels'


def test_setup_says_when_r_is_too_low_for_an_rv(qtbot, win, backend, spectra_dir):
    from PySide6.QtTest import QTest
    _load(qtbot, win, spectra_dir)
    assert win.setup.recommended.styleSheet() == ''
    QTest.keyClicks(win.setup.editors['input.resolution'].edit, '100')
    text = win.setup.recommended.text()
    assert text.startswith('R 100 is below') and 'cannot constrain an RV' in text
    assert 'Recommended' not in text and 'left at the low-R defaults for R 100' in text  # no contradiction
    assert win.setup.recommended.styleSheet() == 'color: #ddaa33'
    assert win.statusBar().currentMessage() == '3 files, 2 objects loaded.'


def test_real_jwst_examples_list_spectra_and_cube():
    """The committed JWST examples through the real backend: MOS sources with reasons, the cube's brightest spaxel"""
    from pathlib import Path

    import numpy as np

    from rvfitter.gui.backend import LazyBackend
    ex = Path(__file__).parent.parent / 'examples'
    b = LazyBackend()
    cfg = b.default_config()
    entries = b.list_spectra(ex / 'jwst_mos' / 'jw01229004001_05101_00001_nrs1_x1d.fits', cfg)
    assert [(e.name, e.usable) for e in entries] == [('1229_13', False), ('1229_6', True), ('1229_7', True)]
    assert entries[0].reason == 'fewer than two valid pixels'
    cube = ex / 'jwst_ifu' / 'jw01386013001_03104_00001_nrs1_s3d_cutout.fits.gz'
    assert [e.kind for e in b.list_spectra(cube, cfg)] == ['cube', 'error', 'dq']
    img, (x, y) = b.cube_image(cube)
    assert img.shape == (17, 17) and img[y, x] == np.nanmax(img)


def test_tick_of_a_detector_position_selects_only_its_file(qtbot, tmp_path):
    """Real backend: two nights share a PypeIt trace name; ticking it in one file never selects the other file's"""
    from PySide6.QtCore import Qt

    from rvfitter.gui.backend import LazyBackend
    from test_io import _multi_spec1d
    names = ['SPAT0123-SLIT0456-DET01', 'SPAT0300-SLIT0456-DET01']
    a, b = (_multi_spec1d(tmp_path / f'night{n}.fits', [17., 86.], names, ['science'] * 2) for n in (1, 2))
    w = MainWindow(LazyBackend(), QSettings(str(tmp_path / 'q.ini'), QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    _load(qtbot, w, a)
    w.load_paths([b])
    qtbot.waitUntil(lambda: len(w.entries) == 2, timeout=10000)
    night1 = next(w.input_tree.topLevelItem(i) for i in range(w.input_tree.topLevelItemCount())
                  if w.input_tree.topLevelItem(i).text(0) == 'night1.fits')
    night1.child(0).setCheckState(0, Qt.CheckState.Checked)
    assert f'{a}[1]' in w.current_config().input.objects
    qtbot.waitUntil(lambda: 'My Star SPAT0123-SLIT0456-DET01 (night1)' in w.groups, timeout=10000)
    assert not any('night2' in name for name in w.groups)
    w.close()


def test_every_location_has_a_browse_dialog_of_its_kind(qtbot, monkeypatch, tmp_path):
    """Each config.LOCATION_FIELDS entry (input.paths and output.path included) gets the dialog of its kind"""
    from PySide6.QtWidgets import QFileDialog, QLineEdit

    from rvfitter import config
    from rvfitter.gui import forms
    picked = str(tmp_path / 'x')
    calls = []
    for fn, ret in (('getOpenFileNames', ([picked], '')), ('getOpenFileName', (picked, '')),
                    ('getExistingDirectory', picked), ('getSaveFileName', (picked, ''))):
        monkeypatch.setattr(QFileDialog, fn, staticmethod(lambda *a, fn=fn, ret=ret: calls.append(fn) or ret))
    dialog = {'files': 'getOpenFileNames', 'file': 'getOpenFileName', 'folder': 'getExistingDirectory',
              'save': 'getSaveFileName'}
    form = forms.ConfigForm(config.Config())
    qtbot.addWidget(form)
    for (section, name), kind in config.LOCATION_FIELDS.items():
        ed = form.editor(f'{section}.{name}')
        assert ed.browse_btn.kind == kind and ed.browse_btn.text() == forms.BROWSE_TEXT[kind]
        calls.clear()
        ed.browse_btn.click()
        assert calls == [dialog[kind]]
        assert ed.get() in (picked, [picked]), (section, name)
    form.set_value('templates.local_dir', {'lowz': 'a', 'sand': 'b'})  # a mapping: the first grid's folder
    form.editor('templates.local_dir').browse_btn.click()
    assert form.get_value('templates.local_dir') == {'lowz': picked, 'sand': 'b'}
    with pytest.raises(ValueError, match='Unknown location kind'):
        forms._with_browse(QLineEdit(), 'archive', lambda: None)


def test_advanced_groups_start_collapsed(qtbot, win):
    from rvfitter.gui.forms import GroupEditor
    dlg = win.advanced_settings()
    sections = [e for e in dlg.form.root.children.values() if isinstance(e, GroupEditor)]
    assert sections and all(not e.toggle.isChecked() and e.box.isHidden() for e in sections)
    dlg.reject()


def test_advanced_rows_show_comments_advice_and_apply(qtbot, win, spectra_dir):
    from rvfitter.gui import theme
    from rvfitter.gui.forms import DEFAULT_WHY, guide_url
    _load(qtbot, win, spectra_dir)
    dlg = win.advanced_settings()
    rows = dlg.form.rows()
    medium = rows['input.medium']
    assert guide_url('input.medium') in medium.comment.text() and 'More…' in medium.comment.text()
    assert medium.advice.text() == 'Recommended for alpha: air. the test advice' and not medium.apply.isHidden()

    def colour(row):
        return row.advice.styleSheet().split(': ')[1]
    assert colour(medium) == theme.colour('flagged', medium.palette()).name()  # differs from the box
    medium.apply.click()
    assert dlg.form.get_value('input.medium') == 'air'
    assert colour(medium) == theme.colour('primary', medium.palette()).name()
    assert rows['input.resolution'].advice.text() == 'For alpha: a reason only'
    assert rows['input.resolution'].apply.isHidden()
    assert rows['seed'].advice.text() == f'The default suits this spectrum: {DEFAULT_WHY["seed"]}'
    dlg.reject()


def test_guide_links(win):
    from rvfitter.gui.forms import guide_url
    assert guide_url('input.resolution').endswith('docs/settings.md#input-resolution')
    assert guide_url('calibration').endswith('#calibration')
    assert 'Settings guide' in [a.text() for m in win.menuBar().actions() for a in m.menu().actions()]


def test_settings_guide_has_an_anchor_for_every_setting_shown():
    from pathlib import Path

    from rvfitter.gui.forms import TIPS, guide_url
    guide = (Path(__file__).resolve().parents[1] / 'docs' / 'settings.md').read_text()
    paths = set(TIPS) | {p.split('.')[0] for p in TIPS if '.' in p}
    missing = [p for p in sorted(paths) if f'<a id="{guide_url(p).split("#")[1]}"></a>' not in guide]
    assert not missing


def test_setup_shows_the_frame_and_wavelength_status(qtbot, win, backend, spectra_dir):
    from rvfitter.gui.setup import FRAME_PATH, WAVE_PATH
    backend.frame_state = 'unknown'
    _load(qtbot, win, spectra_dir)
    frame = win.setup.status[FRAME_PATH]
    assert frame.text().startswith('Unknown (test inputs): a wrong choice shifts RVs by up to ±25.0 km/s')
    assert 'two-settings' in frame.text() and 'color' in frame.styleSheet()  # gold, with the docs link
    assert win.setup.status[WAVE_PATH].text().startswith('test wavelength status (none)')


@pytest.mark.parametrize('answer, correction, frame', [('heliocentric', 'skip', 'heliocentric'),
                                                       ('apply', 'apply', 'barycentric')])
def test_measure_asks_for_an_unknown_frame_and_records_it(qtbot, win, backend, spectra_dir, monkeypatch, answer,
                                                          correction, frame):
    backend.frame_state = 'unknown'
    _load(qtbot, win, spectra_dir)
    asked = []
    monkeypatch.setattr(win, '_ask_frame', lambda unknown: asked.append(sorted(unknown)) or answer)
    win.run()
    qtbot.waitUntil(lambda: bool(backend.runs) and win.worker is None, timeout=10000)
    assert asked == [['alpha', 'beta']]
    run_cfg = backend.runs[0]['cfg']
    want = {'correction': 'apply'} if correction == 'apply' else {'correction': correction, 'frame': frame}
    assert run_cfg.refits == [{'object': n, 'patch': {'velocity_frame': want}} for n in ('alpha', 'beta')]
    assert run_cfg.velocity_frame.correction == 'auto'  # the answer is per object, never global
    assert win.current_config().refits == run_cfg.refits  # saved with the settings


def test_frame_answer_leaves_computable_objects_alone(qtbot, win, backend, spectra_dir, monkeypatch):
    """Only the unknown objects get the answer: a computable object keeps 'auto' (its correction is still applied)"""
    backend.frame_state = 'unknown'
    _load(qtbot, win, spectra_dir)
    backend.frame_state = 'computed'
    unknown_only = {'alpha': win.backend.frame_status(win.first_spectra['alpha'], win.current_config())}
    unknown_only['alpha'] = unknown_only['alpha'].__class__('unknown', 0., 'test inputs', 25., 'barycentric')
    monkeypatch.setattr(win, '_ask_frame', lambda unknown: 'barycentric')
    cfg = win.current_config()
    assert win._settle_frame(cfg, unknown_only)
    assert cfg.refits == [{'object': 'alpha', 'patch': {'velocity_frame': {'correction': 'skip',
                                                                          'frame': 'barycentric'}}}]
    assert cfg.velocity_frame.correction == 'auto'  # beta (computable) is untouched


def test_measure_does_not_run_on_an_unknown_frame_without_an_answer(qtbot, win, backend, spectra_dir, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    backend.frame_state = backend.frame_apply_state = 'unknown'
    _load(qtbot, win, spectra_dir)
    monkeypatch.setattr(win, '_ask_frame', lambda unknown: 'cancel')
    assert win.run() is None and not backend.runs
    shown = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a: shown.append(a[2]))
    monkeypatch.setattr(win, '_ask_frame', lambda unknown: 'apply')  # not corrected, but it cannot be computed
    assert win.run() is None and not backend.runs
    assert 'velocity_frame.overrides' in shown[0]
    assert win.current_config().velocity_frame.correction == 'auto'


def test_diagnostics_say_how_much_the_wavelength_correction_moved_the_rv():
    from types import SimpleNamespace

    from rvfitter.gui.diagnostics import DiagnosticsPanel
    res = SimpleNamespace(barycentric={'applied_kms': 0., 'source': 'unknown', 'frame': 'barycentric',
                                       'wavelength_correction': 'sky', 'wavelength_shift_kms': -21.4})
    text = DiagnosticsPanel.bary_text(res)
    assert 'The sky correction moved this RV by −21.4 km/s' in text and 'frame_unknown' in text
    res.barycentric.update(wavelength_correction='none', wavelength_correction_mode='none', source='computed')
    assert 'No wavelength correction applied (none)' in DiagnosticsPanel.bary_text(res)
