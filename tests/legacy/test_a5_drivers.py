"""
A5: driver fixes (teff range, --systematic type, suffix stripping, nearest template, the shared adoptedrv) and
the single-level identifiability rule of the hierarchical model.
"""
import numpy as np
import pandas as pd
import pytest
from matplotlib import rcParams
import matplotlib.pyplot as plt

import utils


@pytest.mark.parametrize('teff, expected', [(np.nan, 2000.), (1100., 2000.), (4100., 2000.), (1149., 2000.),
                                            (2049., 2000.), (2051., 2100.), (1200., 1200.), (4000., 4000.)])
def test_expected_teff(teff, expected):
    assert utils.expected_teff(teff) == expected


def test_parsers_take_float_systematic():
    import fullrv
    import fullrv_spex
    for module in (fullrv, fullrv_spex):
        args = module.build_parser().parse_args(['-f', 'x.fits', '-s', '2.5'])
        assert args.systematic == 2.5 and isinstance(args.systematic, float)
        assert module.build_parser().parse_args(['-f', 'x.fits']).systematic == 0.


def test_spex_target_name_removes_suffix_only():
    import fullrv_spex
    # str.strip('.fits') would also eat the trailing 's' of the target name
    assert fullrv_spex.spex_target_name('data/sxd_Teegardens.fits') == 'Teegardens'
    assert fullrv_spex.spex_target_name('prism_J1745-1640_tellcorr.fits') == 'J1745-1640'


@pytest.fixture
def template_grid(tmp_path):
    rows = {'lte019.0-5.0.txt': (1900, 5.0, 0.), 'lte020.0-4.5.txt': (2000, 4.5, 0.),
            'lte020.0-5.0.txt': (2000, 5.0, 0.), 'lte022.0-5.0.txt': (2200, 5.0, 0.),
            'lte025.0-5.0.txt': (2500, 5.0, 0.)}
    for name in list(rows)[:-1]:  # lte025.0-5.0.txt is in the lookup but its file is missing
        (tmp_path / name).write_text('8000 1\n8001 1\n')
    df = pd.DataFrame.from_dict(rows, 'index', columns=('teff', 'logg', 'met'))
    return df, str(tmp_path) + '/'


def test_nearest_template(template_grid):
    df, tdir = template_grid
    assert utils.nearest_template(df, 2000, 5.0, 0., tdir) == ('lte020.0-5.0.txt', 2000., 5.0, 0.)  # exact
    assert utils.nearest_template(df, 2040, 4.6, 0., tdir) == ('lte020.0-4.5.txt', 2000., 4.5, 0.)  # nearest
    assert utils.nearest_template(df, 2150, 5.5, 0., tdir) == ('lte022.0-5.0.txt', 2200., 5.0, 0.)
    # 2500 K exists in the lookup but not on disk, so the nearest existing template is used
    assert utils.nearest_template(df, 2500, 5.0, 0., tdir) == ('lte022.0-5.0.txt', 2200., 5.0, 0.)
    assert utils.nearest_template(df, 2000, 5.0, 0., tdir + 'empty/') is None


def test_spex_uses_shared_adoptedrv(tmp_path, monkeypatch):
    import astropy.units as u
    import fullrv
    import fullrv_spex
    assert fullrv_spex.adoptedrv is fullrv.adoptedrv
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(rcParams, 'text.usetex', False)
    spec_indices = {'k1-a': 1.1690219, 'k1-b': 1.1772838}
    df = pd.DataFrame({'Short Name': ['X']})
    df = fullrv.adoptedrv(df, 'Short Name', 'X', np.array([10., 12.]), np.array([2., 2.]),
                          np.array([11., 14.]), np.array([3., 3.]), spec_indices, outname='X_adoptedrv.pdf',
                          wunit=u.um)
    labels = [t.get_text() for t in plt.figure(5).axes[0].get_yticklabels()]
    plt.close('all')
    assert '1.17' in labels[0] and '1.18' in labels[1]
    assert (tmp_path / 'adoptedrvplots' / 'X_adoptedrv.pdf').exists()
    assert np.isfinite(df.thisrv.iloc[0])


def test_single_level_factors_have_tau_fixed_at_zero():
    # one method: tau_meth is not identifiable, so it is fixed at 0
    values, errors = np.array([10., 14., 12.]), np.array([1., 1., 1.])
    res = utils.hierarchical_combine(values, errors, ['a', 'b', 'c'], ['m'] * 3)
    assert res['tau_meth'] == 0. and res['nmethods'] == 1
    ref = utils.hierarchical_combine(values, errors, ['a', 'b', 'c'], ['m'] * 3, tau_meth=0.)
    np.testing.assert_allclose([res['median'], res['q16'], res['q84']], [ref['median'], ref['q16'], ref['q84']])
    # one line measured by two methods: tau_line is fixed at 0
    res = utils.hierarchical_combine([10., 12.], [1., 1.], ['a', 'a'], ['m', 'n'])
    assert res['tau_line'] == 0. and res['tau_meth'] > 0.
    # a single measurement returns itself with its own error
    res = utils.hierarchical_combine([7.5], [2.], ['a'], ['m'])
    np.testing.assert_allclose([res['mean'], res['sd'], res['median']], [7.5, 2., 7.5])


def test_adoptedrv_single_method(tmp_path, monkeypatch):
    import fullrv
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(rcParams, 'text.usetex', False)
    spec_indices = {'k1-a': 7664.9, 'k1-b': 7699.0, 'na1-a': 8183.3}
    df = pd.DataFrame({'shortname': ['X']})
    nan3 = np.full(3, np.nan)
    df = fullrv.adoptedrv(df, 'shortname', 'X', nan3, nan3, np.array([20., 25., 22.]), np.array([3., 3., 3.]),
                          spec_indices)
    plt.close('all')
    assert np.isfinite(df.thisrv.iloc[0])
    assert df.rv_adopted_nmethods.iloc[0] == 1 and df.rv_adopted_nlines.iloc[0] == 3
    log = (tmp_path / 'calculating.log').read_text()
    assert 'WARNING' in log and 'between-method systematic is not estimable' in log
