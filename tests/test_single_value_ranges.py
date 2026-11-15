"""A template range given as one value (e.g. logg_range = 5.0) holds just that grid value."""
import json
import os

import pytest

from rvfitter import config
from rvfitter import templates as T

TOML = '[templates]\nlogg_range = 5.0\nteff_range = [2000]\nranges = {co = 1}\nmet_range = [-0.5, 0.0]\n'


def test_toml_single_values_normalise_and_round_trip():
    cfg = config.loads(TOML)
    t = cfg.templates
    assert (t.logg_range, t.teff_range, t.ranges, t.met_range) == ((5.0, 5.0), (2000., 2000.), {'co': (1., 1.)},
                                                                    (-0.5, 0.))
    assert config.loads(config.dumps(cfg)) == cfg  # saved as the normalised [v, v]
    assert cfg.to_dict()['templates']['logg_range'] == [5.0, 5.0]


@pytest.mark.gui
def test_code_and_gui_single_values(qtbot):
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('PySide6')
    from rvfitter.gui.forms import ConfigForm
    assert config.TemplatesConfig(logg_range=5.0, ranges={'co': [1.]}).logg_range == (5.0, 5.0)
    form = ConfigForm(config.Config())
    qtbot.addWidget(form)
    form.editor('templates.logg_range').edit.setText('5.0')
    form.editor('templates.ranges').edit.setText(json.dumps({'co': 1}))
    t = form.get_config().templates
    assert (t.logg_range, t.ranges) == ((5.0, 5.0), {'co': (1, 1)})
    form.get_config().validate()
    with pytest.raises(config.ConfigError, match='min <= max'):
        config.loads('[templates]\nlogg_range = [5.5, 5.0]\n')
    with pytest.raises(config.ConfigError, match='min < max'):  # not a template range: one value is not allowed
        config.loads('[calibration]\nwave_valid_range = [5000.0, 5000.0]\n')


def test_single_value_selects_only_it_and_names_the_nearest_otherwise(tmp_path):
    from test_templates import _make_grid
    grid = T.TemplateGrid(_make_grid(tmp_path / 'g'))  # teff 1800-2000 (100 K), logg 4.5, 5.0
    assert grid.select(teff_range=(1900., 1900.), logg_range=(5.0, 5.0)) == [(1900., 5.0, 0., 0.)]
    with pytest.raises(ValueError, match=r'no logg = 4.8; the nearest values are 4.5, 5'):
        grid.select(logg_range=(4.8, 4.8))


def test_single_values_with_no_joint_model_raise(tmp_path):
    """Each value exists on its axis but no model has both: raise, so no nearest-to-prior fallback can break them"""
    from test_templates import _make_grid
    directory = _make_grid(tmp_path / 'g')
    (directory / 'lte019.0-4.5.txt').unlink()  # sparse: (1900, 4.5) missing, (1900, 5.0) and (1800, 4.5) present
    grid = T.TemplateGrid(directory)
    with pytest.raises(ValueError, match=r'no model with teff = 1900, logg = 4.5 within the other ranges'):
        grid.select(teff_range=(1900., 1900.), logg_range=(4.5, 4.5))


def test_single_value_reaches_the_download(monkeypatch):
    from rvfitter.gui.backend import LazyBackend
    calls = []
    monkeypatch.setattr(T, 'download', lambda grid, **kw: calls.append(kw))
    LazyBackend().download_templates(config.loads(TOML), ['sonora-bobcat'])
    assert calls[0]['logg_range'] == (5.0, 5.0) and calls[0]['teff_range'] == (2000., 2000.)
    assert calls[0]['ranges'] == {'co': (1., 1.)}


def test_svo_names_the_nearest_values_of_a_missing_single_value(tmp_path):
    from test_templates import _fake_ssap
    rows = [dict(url=f'http://svo/x?fid={i}', teff=t, logg=5.0, met=0.) for i, t in enumerate((1800., 1900.))]
    urls = []

    def fetch(url):
        urls.append(url)
        return _fake_ssap([] if 'teff_min=1850' in url else rows)

    with pytest.raises(FileNotFoundError, match=r'no grid point at teff = 1850: nearest 1800, 1900\)'):
        T.download('bt-settl-cifist', (1850., 1850.), (5.0, 5.0), directory=tmp_path / 'dl', fetch=fetch)
    assert 'teff_min=1850' in urls[0] and 'teff_max=1850' in urls[0] and 'teff_min' not in urls[1]
