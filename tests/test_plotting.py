"""Diagnostic figures on a synthetic ObjectResult (no method run needed)."""
import importlib.util

import matplotlib as mpl
import pytest
from matplotlib.figure import Figure

from plotting_synth import synthetic_result
from rvfitter import plotting
from rvfitter.types import ObjectResult


@pytest.mark.parametrize('fn', ['plot_line_fits', 'plot_ccf', 'plot_adopted', 'plot_wavelength_correction'])
def test_returns_figure(fn):
    fig = getattr(plotting, fn)(synthetic_result())
    assert isinstance(fig, Figure)
    fig.canvas.draw()


def test_corner():
    if importlib.util.find_spec('corner') is None:
        with pytest.raises(ImportError, match='corner'):
            plotting.plot_corner(synthetic_result())
    else:
        assert isinstance(plotting.plot_corner(synthetic_result()), Figure)


def test_missing_data_raises():
    with pytest.raises(ValueError, match='linecentre'):
        plotting.plot_line_fits(ObjectResult('X', []))


def test_make_all_and_save(tmp_path):
    names = set(plotting.make_all(synthetic_result()))
    assert {'line_fits', 'ccf', 'adopted', 'wavelength_correction'} <= names
    paths = plotting.save_all([synthetic_result()], tmp_path)
    assert paths
    assert all(p.stat().st_size > 0 and p.suffix == '.pdf' for p in paths)


def test_style_does_not_leak():
    before = dict(mpl.rcParams)
    with plotting.style_context():
        assert mpl.rcParams['xtick.direction'] == 'in'
        assert mpl.rcParams['font.family'] == ['serif']
    plotting.plot_adopted(synthetic_result())
    assert dict(mpl.rcParams) == before
