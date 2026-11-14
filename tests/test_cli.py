import sys
import types

import pytest

from rvfitter import __version__, cli


def test_version(capsys):
    assert cli.main(['--version']) == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_and_usage_errors():
    assert cli.main([]) == 1
    assert cli.main(['bogus']) == 1
    assert cli.main(['measure']) == 1
    assert cli.main(['measure', 'x', '--medium', 'water']) == 1
    assert cli.main(['measure', 'x', '--bary', 'maybe']) == 1


def test_measure_parsing():
    args = cli.build_parser().parse_args(
        ['measure', 'a.fits', '--methods', 'ccf,lsd', '--out', 'o.json', '--format', 'csv,json', '--n-procs', '3',
         '--medium', 'air', '--bary', 'skip', '-v'])
    assert (args.methods, args.n_procs, args.medium, args.bary, args.verbose) == ('ccf,lsd', 3, 'air', 'skip', True)


@pytest.mark.parametrize('cmd', ['inject', 'forecast', 'templates', 'gui'])
def test_not_yet_available(cmd, capsys, monkeypatch):
    monkeypatch.setitem(cli.LAZY, cmd, (f'rvfitter.nonexistent_{cmd}', 'main', True))
    assert cli.main([cmd]) == 1
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


def test_doctor_offline(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, '_reachable', lambda url: pytest.fail('network used with --offline'))
    monkeypatch.setenv('RVFITTER_CACHE', str(tmp_path))
    code = cli.main(['doctor', '--offline'])
    out = capsys.readouterr().out
    assert 'OK    import numpy' in out and 'skipped (--offline)' in out and 'Template cache writable' in out
    assert ('FAIL' in out) == (code == 1)  # extras (e.g. PySide6) may be missing in CI


def test_doctor_network_checks(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv('RVFITTER_CACHE', str(tmp_path))
    monkeypatch.setattr(cli, '_reachable', lambda url: (False, f'{url}: URLError') if 'zenodo' in url else (True, url))
    checks = {c[0]: c for c in cli._doctor_checks(False)}
    assert checks['SVO reachable'][1] and not checks['Zenodo reachable'][1]
    assert cli.main(['doctor']) == 1
    out = capsys.readouterr().out
    assert 'FAIL  Zenodo reachable' in out and 'fix:' in out


def test_doctor_failure(monkeypatch, capsys):
    monkeypatch.setattr(cli, '_doctor_checks', lambda offline: [('thing', False, 'broken', 'do x')])
    assert cli.main(['doctor', '--offline']) == 1
    assert 'FAIL  thing' in capsys.readouterr().out


def test_missing_optional_dependency_exit_1(monkeypatch, capsys):
    mod = types.ModuleType('rvfitter.fake_gui')

    def main():
        raise ModuleNotFoundError("No module named 'PySide6'", name='PySide6')

    mod.main = main
    monkeypatch.setitem(sys.modules, 'rvfitter.fake_gui', mod)
    monkeypatch.setitem(cli.LAZY, 'gui', ('rvfitter.fake_gui', 'main', False))
    assert cli.main(['gui']) == 1
    assert 'rvfitter[gui]' in capsys.readouterr().out


def test_lazy_systemexit_propagates(monkeypatch):
    mod = types.ModuleType('rvfitter.fake_exit')

    def main(argv):
        raise SystemExit(4)

    mod.main = main
    monkeypatch.setitem(sys.modules, 'rvfitter.fake_exit', mod)
    monkeypatch.setitem(cli.LAZY, 'forecast', ('rvfitter.fake_exit', 'main', True))
    assert cli.main(['forecast']) == 4


def test_measure_defaults_do_not_override_config():
    args = cli.build_parser().parse_args(['measure', 'a.fits'])
    assert args.medium is None and args.bary is None and args.n_procs is None


def test_run_no_objects_exit_1(monkeypatch, tmp_path):
    from rvfitter import config, pipeline
    ns = types.SimpleNamespace
    cfg = ns(input=ns(paths=['x']), output=ns(path=str(tmp_path / 'o.csv'), formats=[], plots=False))
    monkeypatch.setattr(config, 'load', lambda p: cfg, raising=False)
    monkeypatch.setattr(pipeline, 'run', lambda *a, **k: [])
    assert cli.main(['run', 'c.toml']) == 1


def test_run_with_failures_exit_2(monkeypatch, tmp_path):
    from rvfitter import config, pipeline, results
    ns = types.SimpleNamespace
    cfg = ns(input=ns(paths=['x']), output=ns(path=str(tmp_path / 'o.csv'), formats=[], plots=False))
    monkeypatch.setattr(config, 'load', lambda p: cfg, raising=False)
    monkeypatch.setattr(pipeline, 'run',
                        lambda *a, failures, **k: failures.append(('bad.fits', 'not a spectrum')) or [1])
    monkeypatch.setattr(results, 'summary_lines', lambda r: [])
    monkeypatch.setattr(results, 'write', lambda *a, **k: [])
    assert cli.main(['run', 'c.toml']) == 2  # the summary itself is logged by pipeline.run (test_pipeline)


def test_run_writes_pdfs_unless_no_plots(monkeypatch, tmp_path):
    from plotting_synth import synthetic_result
    from rvfitter import config, pipeline, results
    ns = types.SimpleNamespace
    cfg = ns(input=ns(paths=['x']), output=ns(path=str(tmp_path / 'o.csv'), formats=['csv'], plots=True,
                                                                    plot_style='quick'))
    monkeypatch.setattr(config, 'load', lambda p: cfg, raising=False)
    monkeypatch.setattr(pipeline, 'run', lambda *a, **k: [synthetic_result()])
    monkeypatch.setattr(results, 'write', lambda *a, **k: [])
    assert cli.main(['run', 'c.toml', '--no-plots']) == 0
    assert not list(tmp_path.glob('*.pdf'))
    cfg.output.plots = True
    assert cli.main(['run', 'c.toml']) == 0
    assert {p.name for p in tmp_path.glob('*.pdf')} >= {'J0000_ccf.pdf', 'J0000_adopted.pdf'}


def test_measure_records_every_choice_in_the_saved_config(monkeypatch, tmp_path):
    """The results_config.toml of `measure` reproduces the run: inputs, output and overrides are in it, and its file
    locations are absolute, so `rvfitter run` finds them from any directory"""
    from plotting_synth import synthetic_result
    from rvfitter import config, pipeline
    seen = {}

    def fake_run(cfg, inputs=None, **k):
        seen['cfg'], seen['inputs'] = cfg, inputs
        return [synthetic_result()]
    monkeypatch.setattr(pipeline, 'run', fake_run)
    monkeypatch.setattr(config, 'default_for', lambda spec: config.Config())
    monkeypatch.setattr('rvfitter.archives.collect_inputs', lambda p: [p])
    monkeypatch.setattr('rvfitter.io.read_spectrum', lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    assert cli.main(['measure', 'data/s.fits', '--methods', 'linecentre', '--medium', 'air', '--bary', 'skip',
                     '--out', 'out/results.csv', '--no-plots']) == 0
    cfg = seen['cfg']
    assert seen['inputs'] is None and cfg.input.paths == ['data/s.fits'] and cfg.methods.enabled == ['linecentre']
    assert cfg.input.medium == 'air' and cfg.velocity_frame.correction == 'skip'
    saved = config.loads((tmp_path / 'out' / 'results_config.toml').read_text())
    assert saved.input.paths == [str(tmp_path / 'data' / 's.fits')]
    assert saved.output.path == str(tmp_path / 'out' / 'results.csv')
    assert saved.methods.enabled == ['linecentre'] and saved.hash() == cfg.hash()
