"""Smoke tests for the package skeleton."""
import importlib
import os
import pkgutil
import subprocess
import sys

import matplotlib
import pytest

import rvfitter


def test_version():
    assert isinstance(rvfitter.__version__, str)
    assert rvfitter.__version__.startswith('2.')


def test_cli_version():
    out = subprocess.run([sys.executable, '-m', 'rvfitter.cli', '--version'], capture_output=True, text=True)
    assert out.returncode == 0
    assert rvfitter.__version__ in out.stdout


def test_cli_placeholder_exits_2(capsys):
    from rvfitter.cli import main
    assert main(['measure']) == 2
    assert 'not yet implemented' in capsys.readouterr().out


def _stub_modules():
    names = []
    for info in pkgutil.walk_packages(rvfitter.__path__, 'rvfitter.'):
        names.append(info.name)
    return names


@pytest.mark.parametrize('name', _stub_modules())
def test_module_imports(name):
    if name.startswith('rvfitter.gui'):
        pytest.importorskip('PySide6')
    importlib.import_module(name)


def test_expected_modules_present():
    names = set(_stub_modules())
    for mod in ('io', 'archives', 'config', 'features', 'regions', 'barycentric', 'broadening', 'templates',
                'combine', 'systematics', 'results', 'pipeline', 'logging_setup', 'plotting', 'cli',
                'calibration.telluric', 'calibration.telluric_zeropoint', 'methods.linecentre', 'methods.ccf',
                'methods.template_fit', 'methods.forward', 'methods.lsd', 'methods.binary', 'gui.app'):
        assert f'rvfitter.{mod}' in names


def test_import_leaves_rcparams_alone():
    code = (
        'import matplotlib; matplotlib.use("Agg")\n'
        'from matplotlib import rcParams\n'
        'before = {k: repr(v) for k, v in rcParams.items()}\n'
        'import pkgutil, importlib, rvfitter\n'
        'for i in pkgutil.walk_packages(rvfitter.__path__, "rvfitter."):\n'
        '    if "gui" not in i.name:\n'
        '        importlib.import_module(i.name)\n'
        'print("CHANGED", sorted(k for k, v in rcParams.items() if repr(v) != before[k]))\n'
    )
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         env={**os.environ, 'MPLBACKEND': 'Agg'})
    assert out.returncode == 0, out.stderr
    assert 'CHANGED []' in out.stdout
    assert matplotlib.get_backend()
