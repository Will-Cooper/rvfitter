import os
from pathlib import Path

import pytest

from rvfitter import cli, config, io, pipeline, regions

TESTDATA = os.environ.get('RVFITTER_TESTDATA')
J1745 = Path(TESTDATA or '.') / 'pypeit/R2500I/GTC54-15A0/OB0055/Science/DENISJ174534.6-164053_tellcorr.fits'
data = pytest.mark.skipif(not TESTDATA, reason='RVFITTER_TESTDATA not set')


@data
def test_telluric_corrected_makes_k_lines_eligible():
    spec = io.read_spectrum(J1745)
    off = regions.select(spec, methods=['linecentre', 'ccf'], options=regions.SelectOptions())
    on = regions.select(spec, methods=['linecentre', 'ccf'], options=regions.SelectOptions(telluric_corrected=True))
    names_off = {r.name for r in off['linecentre']}
    names_on = {r.name for r in on['linecentre']}
    assert not {'K I 7667', 'K I 7701'} & names_off
    assert {'K I 7667', 'K I 7701'} <= names_on
    # ccf/fit spans keep every telluric mask (H2O residuals are not noise), so they are unchanged
    assert [(r.wmin, r.wmax) for r in on['ccf']] == [(r.wmin, r.wmax) for r in off['ccf']]


@data
def test_cli_measure_without_methods_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, 'REGISTRY', {})
    out = tmp_path / 'r.csv'
    code = cli.main(['measure', str(J1745), '--out', str(out), '--bary', 'skip', '--no-plots', '--telluric-corrected'])
    text = capsys.readouterr().out
    assert code == 1
    assert text.count('no RV methods available') == 1 and 'Traceback' not in text and 'AttributeError' not in text
    assert not out.exists()


def test_unknown_or_no_methods_raise():
    cfg = config.Config()
    assert pipeline.enabled_methods(cfg) == ['template_fit', 'ccf', 'linecentre']
    with pytest.raises(ValueError, match='unknown bisector'):
        pipeline.enabled_methods(cfg, ['ccf', 'bisector'])
    cfg.methods.enabled = []
    with pytest.raises(pipeline.NoMethodsError, match='none is enabled'):
        pipeline.run(cfg)


def test_flags_parse():
    args = cli.build_parser().parse_args(['measure', 'a.fits', '--telluric-corrected', '--no-plots'])
    assert args.telluric_corrected and args.no_plots
    assert cli.build_parser().parse_args(['run', 'c.toml', '--no-plots']).no_plots


def test_config_field_roundtrip(tmp_path):
    cfg = config.Config()
    assert cfg.input.telluric_corrected is False
    cfg.input.telluric_corrected = True
    config.dump(cfg, tmp_path / 'c.toml')
    assert config.load(tmp_path / 'c.toml').input.telluric_corrected is True
