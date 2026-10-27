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
    assert [win.tabs.tabText(i) for i in range(win.tabs.count())] == ['Settings', 'Regions', 'Results', 'Diagnostics']


def test_loading_populates_list(qtbot, win, spectra_dir):
    _load(qtbot, win, spectra_dir)
    assert set(win.groups) == {'alpha', 'beta'}
    assert win.input_tree.topLevelItemCount() == 2
    assert win.input_tree.topLevelItem(0).childCount() == 2


def test_settings_round_trip(win):
    cfg = FakeConfig()
    cfg.medium.medium = 'air'
    cfg.barycentric.correction = 'skip'
    cfg.methods.template_fit = True
    cfg.methods.linecentre = False
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
    walk(win.form.root)


def test_save_and_load_config(win, tmp_path):
    cfg = FakeConfig()
    cfg.methods.ccf = False
    cfg.parallel.seed = 99
    win.set_config(cfg)
    path = tmp_path / 'analysis.toml'
    assert win.save_config(path)
    win.set_config(FakeConfig())
    assert win.load_config(path)
    assert win.current_config() == cfg


def test_run_fills_table(qtbot, win, backend, spectra_dir):
    _load(qtbot, win, spectra_dir)
    win.form.set_value('parallel.n_procs', 2)
    win.run()
    qtbot.waitUntil(lambda: win.worker is None, timeout=10000)
    assert win.model.rowCount() == 2
    assert list(win.model.df['object']) == ['alpha', 'beta']
    assert backend.runs[0]['n_procs'] == 2
    assert 'computed' in win.diagnostics.bary.text()
    assert 'not available' in win.diagnostics.views['ccf'].placeholder.text()
    assert win.diagnostics.views['line_fits'].canvas is not None


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
    assert win.run_btn.isEnabled()
    assert not win.cancel_btn.isEnabled()


def test_region_edits_land_in_overrides(qtbot, win, spectra_dir):
    _load(qtbot, win, spectra_dir)
    win.tabs.setCurrentWidget(win.region_editor)
    win.select_object('alpha')
    ed = win.region_editor
    k1, k2 = ('linecentre', 'K I 7701'), ('linecentre', 'K I 7667')
    assert set(ed.spans) == {k1, k2}
    assert win.current_config().regions.overrides == {}
    ed.move_span(k1, 7696.0, 7706.0)
    ed.remove_span(k2)
    key = ed.add_span(7800.0, 7810.0)
    ov = win.current_config().regions.overrides
    assert {'name': 'K I 7667', 'remove': True} in ov['linecentre']
    moved = next(e for e in ov['linecentre'] if e['name'] == 'K I 7701')
    assert (moved['wmin'], moved['wmax']) == (7696.0, 7706.0)
    assert next(e for e in ov['linecentre'] if e['name'] == key[1])['wmin'] == 7800.0
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
    span = ed.spans[('linecentre', 'K I 7701')]
    assert span.wmin == pytest.approx(7695 + 20, abs=1.0)
    assert win.current_config().regions.overrides['linecentre'][0]['wmin'] == pytest.approx(span.wmin)


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

    def fake_run(cfg, inputs, *, n_procs, on_object=None, cancel=None):
        seen['cancel'] = cancel
        seen['n'] = len(inputs)
        for name in ('a', 'b'):
            on_object(name)
        return ['a', 'b']

    monkeypatch.setattr(pipeline, 'run', fake_run, raising=False)
    got = []
    out = LazyBackend().run(FakeConfig(), {'x': ['f1', 'f2'], 'y': ['f3']}, 1, got.append, lambda: False)
    assert out == got == ['a', 'b']
    assert seen['n'] == 3
    assert not seen['cancel'].is_set()


def test_lazy_backend_honours_config(monkeypatch):
    from types import SimpleNamespace

    from rvfitter import archives, io, regions
    from rvfitter.gui.backend import LazyBackend
    seen = {}

    def fake_read(path, *, medium='auto', hdu=None, resolution=None):
        seen['read'] = (medium, hdu, resolution)
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
    cfg = SimpleNamespace(input=SimpleNamespace(medium='air', medium_method='edlen', hdu=2, resolution=4000.),
                          grouping=SimpleNamespace(override={'a': 'obj'}),
                          templates=SimpleNamespace(teff_prior=(2000., 200.)),
                          methods=SimpleNamespace(enabled=['ccf']),
                          regions=SimpleNamespace(overrides={'ccf': [{'name': 'x', 'remove': True}]}))
    be = LazyBackend()
    spec = be.read_spectrum('f.fits', cfg)
    assert seen['read'] == ('air', 2, 4000.)
    assert be.group_objects(['f.fits'], cfg) == {'obj': ['f.fits']}
    assert seen['override'] == {'a': 'obj'}
    be.select_regions(spec, cfg, True)
    assert seen['select'][:3] == (2000., ['ccf'], cfg.regions.overrides)
