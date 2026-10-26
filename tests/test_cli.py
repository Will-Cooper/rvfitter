import sys
import types

import pytest

from rvfitter import __version__, cli


def test_version(capsys):
    assert cli.main(['--version']) == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_and_usage_errors():
    assert cli.main([]) == 2
    assert cli.main(['bogus']) == 2
    assert cli.main(['measure']) == 2
    assert cli.main(['measure', 'x', '--medium', 'water']) == 2
    assert cli.main(['measure', 'x', '--bary', 'maybe']) == 2


def test_measure_parsing():
    args = cli.build_parser().parse_args(
        ['measure', 'a.fits', '--methods', 'ccf,lsd', '--out', 'o.json', '--format', 'csv,json', '--n-procs', '3',
         '--medium', 'air', '--bary', 'skip', '-v'])
    assert (args.methods, args.n_procs, args.medium, args.bary, args.verbose) == ('ccf,lsd', 3, 'air', 'skip', True)


@pytest.mark.parametrize('cmd', ['inject', 'forecast', 'templates', 'gui'])
def test_not_yet_available(cmd, capsys, monkeypatch):
    monkeypatch.setitem(cli.LAZY, cmd, (f'rvfitter.nonexistent_{cmd}', 'main', True))
    assert cli.main([cmd]) == 2
    assert 'not yet available' in capsys.readouterr().out


def test_lazy_dispatch(monkeypatch):
    mod = types.ModuleType('rvfitter.fake_mod')
    mod.main = lambda argv: 0 if argv == ['download', 'x'] else 3
    monkeypatch.setitem(sys.modules, 'rvfitter.fake_mod', mod)
    monkeypatch.setitem(cli.LAZY, 'templates', ('rvfitter.fake_mod', 'main', True))
    assert cli.main(['templates', 'download', 'x']) == 0


def test_runtime_error_exit_1(tmp_path, monkeypatch):
    mod = types.ModuleType('rvfitter.fake_mod2')

    def boom(argv):
        raise RuntimeError('boom')

    mod.main = boom
    monkeypatch.setitem(sys.modules, 'rvfitter.fake_mod2', mod)
    monkeypatch.setitem(cli.LAZY, 'inject', ('rvfitter.fake_mod2', 'main', True))
    assert cli.main(['inject']) == 1


def test_doctor_offline(capsys):
    code = cli.main(['doctor', '--offline'])
    out = capsys.readouterr().out
    assert code == 0, out
    assert '✓' in out and 'import numpy' in out and 'skipped (--offline)' in out


def test_doctor_failure(monkeypatch, capsys):
    monkeypatch.setattr(cli, '_doctor_checks', lambda offline: [('thing', False, 'broken', True)])
    assert cli.main(['doctor', '--offline']) == 1
    assert '✗' in capsys.readouterr().out
