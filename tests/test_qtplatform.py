"""Qt platform choice: a pure function of the environment and the plugin list."""
import pytest

from rvfitter.gui.qtplatform import choose_platform, plugin_names, xcb_cursor_missing

WAYLAND = {'XDG_SESSION_TYPE': 'wayland'}


def test_wayland_plugin_present():
    assert choose_platform(WAYLAND, ['offscreen', 'wayland-egl', 'xcb'], 'linux') == 'wayland;xcb'


def test_wayland_plugin_absent_uses_xwayland():
    assert choose_platform(WAYLAND, ['offscreen', 'xcb'], 'linux') == 'xcb'


@pytest.mark.parametrize('env', [{'XDG_SESSION_TYPE': 'x11'}, {}])
def test_x11_left_to_qt(env):
    assert choose_platform(env, ['wayland', 'xcb'], 'linux') is None


def test_user_choice_never_overridden():
    assert choose_platform({**WAYLAND, 'QT_QPA_PLATFORM': 'offscreen'}, ['wayland', 'xcb'], 'linux') is None


@pytest.mark.parametrize('system', ['darwin', 'win32'])
def test_other_systems_left_to_qt(system):
    assert choose_platform(WAYLAND, ['wayland', 'xcb'], system) is None


def test_plugin_names(tmp_path):
    for n in ('libqxcb.so', 'libqwayland-egl.so', 'libqoffscreen.so'):
        (tmp_path / n).touch()
    assert plugin_names(tmp_path) == ['offscreen', 'wayland-egl', 'xcb']
    assert plugin_names(tmp_path / 'missing') == []


def test_xcb_cursor_only_checked_for_xcb(monkeypatch):
    monkeypatch.setattr('ctypes.util.find_library', lambda name: None)
    assert xcb_cursor_missing({}, ['xcb'], 'linux')
    assert not xcb_cursor_missing({'QT_QPA_PLATFORM': 'offscreen'}, ['xcb'], 'linux')
    assert not xcb_cursor_missing({}, ['wayland'], 'linux')
    assert not xcb_cursor_missing({}, ['xcb'], 'darwin')
    # Wayland session: the effective platform decides (Codex review)
    assert not xcb_cursor_missing(WAYLAND, ['wayland', 'xcb'], 'linux')
    assert xcb_cursor_missing(WAYLAND, ['xcb'], 'linux')


def test_wayland_quiets_textinput_warning_unless_user_rules(monkeypatch):
    from rvfitter.gui import qtplatform
    monkeypatch.setattr(qtplatform, 'platforms_dir', lambda: None)  # no PySide6 needed
    monkeypatch.setattr(qtplatform, 'plugin_names', lambda d: ['wayland', 'xcb'])
    monkeypatch.setattr(qtplatform.sys, 'platform', 'linux')
    for var in ('QT_QPA_PLATFORM', 'QT_LOGGING_RULES'):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv('XDG_SESSION_TYPE', 'wayland')
    qtplatform.configure_platform()
    assert qtplatform.os.environ['QT_LOGGING_RULES'] == 'qt.qpa.wayland.textinput=false'
    monkeypatch.delenv('QT_QPA_PLATFORM')
    monkeypatch.setenv('QT_LOGGING_RULES', 'mine')
    qtplatform.configure_platform()
    assert qtplatform.os.environ['QT_LOGGING_RULES'] == 'mine'
