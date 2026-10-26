"""
A4: per-line results stay at the fixed slot of their line (NaN if rejected), fits are matched by line name,
and the adopted-RV model and figure pair the methods' values by line.
"""
from types import SimpleNamespace

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib import rcParams

import utils

SPEC_INDICES = {'k1-a': 7664.8991, 'k1-b': 7698.9646, 'rb1-a': 7800.268, 'rb1-b': 7947.603,
                'na1-a': 8183.2556, 'na1-b': 8194.8240}
NAMES = list(SPEC_INDICES)
LC = dict(zip(NAMES, [30., 27., 35., 20., 19., 45.]))
XC = dict(zip(NAMES, [31., 22., 36., 24., 26., 38.]))
LC_REJECT = {'k1-b', 'na1-a'}
XC_REJECT = {'rb1-a', 'na1-a', 'na1-b'}
KMS = u.km / u.s


def _fake_obj(name, rv, extra):
    obj = SimpleNamespace(spec_index=name, rv=rv * KMS, rverr=(2. + 0.1 * NAMES.index(name)) * KMS,
                          best_rmsdiqr=0.1, snr=10., ax=None, c1=None, c2=None, c3=None, c4=None, **extra)
    obj.plotter = lambda: None
    return obj


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(rcParams, 'text.usetex', False)
    (tmp_path / 'adoptedrvplots').mkdir()
    return tmp_path


def _df():
    return pd.DataFrame({'shortname': ['X']})


def test_auto_lc_fit_keeps_line_slots(workdir):
    import linecentering
    extra = dict(line_profile='gaussian', std=2. * u.AA)
    objlist = [_fake_obj(name, LC[name], extra) for name in reversed(NAMES)]  # deliberately out of order
    useset = [name for name in NAMES if name not in LC_REJECT]
    _, rvs, errs = linecentering.auto_lc_fit(useset, SPEC_INDICES, objlist, _df(), 'X', 'shortname',
                                             nrows=3, ncols=2)
    plt.close('all')
    for i, name in enumerate(NAMES):
        if name in LC_REJECT:
            assert np.isnan(rvs[i]) and np.isnan(errs[i])
        else:
            assert rvs[i] == LC[name]
            np.testing.assert_allclose(errs[i], 2. + 0.1 * i)


def test_auto_xcorr_fit_keeps_line_slots(workdir):
    import crosscorrelate
    extra = dict(teff=2000. * u.K, grav=5. * u.dimensionless_unscaled, met=0. * u.dimensionless_unscaled)
    objlist = [_fake_obj(name, XC[name], extra) for name in reversed(NAMES)]
    useset = [name for name in NAMES if name not in XC_REJECT]
    _, rvs, errs = crosscorrelate.auto_xcorr_fit(useset, SPEC_INDICES, objlist, _df(), 'X', 'shortname',
                                                 nrows=3, ncols=2)
    plt.close('all')
    for i, name in enumerate(NAMES):
        if name in XC_REJECT:
            assert np.isnan(rvs[i]) and np.isnan(errs[i])
        else:
            assert rvs[i] == XC[name]


def test_adoptedrv_pairs_and_plots_by_line(workdir, monkeypatch, last_figure):
    import fullrv
    lc = np.array([np.nan if n in LC_REJECT else LC[n] for n in NAMES])
    xc = np.array([np.nan if n in XC_REJECT else XC[n] for n in NAMES])
    lce, xce = np.where(np.isnan(lc), np.nan, 2.), np.where(np.isnan(xc), np.nan, 3.)
    captured = {}
    original = fullrv.hierarchical_combine

    def spy(values, errors, line_ids, method_ids, **kwargs):
        captured.update(values=np.asarray(values), line_ids=np.asarray(line_ids), method_ids=np.asarray(method_ids))
        return original(values, errors, line_ids, method_ids, **kwargs)

    monkeypatch.setattr(fullrv, 'hierarchical_combine', spy)
    fullrv.adoptedrv(_df(), 'shortname', 'X', lc, lce, xc, xce, SPEC_INDICES)

    # every finite value reaches the model labelled with its own line and method
    for value, line, method in zip(captured['values'], captured['line_ids'], captured['method_ids']):
        if np.isfinite(value):
            assert value == (LC if method == 'lc' else XC)[line]

    # plotted positions: the marker at row y belongs to the line whose tick label is at row y
    fig = last_figure()
    axlines = fig.axes[0]
    ticklabels = [t.get_text() for t in axlines.get_yticklabels()]
    assert len(ticklabels) == len(NAMES)
    for name, label in zip(NAMES, ticklabels):
        assert name.split('-')[0][:-1].capitalize() in label and f'{int(SPEC_INDICES[name])}' in label
    containers = axlines.containers
    assert len(containers) == 2  # cross-correlation first, then line centring
    for container, values, offset in ((containers[0], XC, -0.1), (containers[1], LC, 0.1)):
        xdata, ydata = container.lines[0].get_data()
        for x, y in zip(xdata, ydata):
            if np.isfinite(x):
                row = int(round(y - offset)) - 1
                assert x == values[NAMES[row]]
        rejected = LC_REJECT if values is LC else XC_REJECT
        for row, name in enumerate(NAMES):
            assert np.isnan(xdata[row]) == (name in rejected)
    plt.close('all')
