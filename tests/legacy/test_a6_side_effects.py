"""
A6: no import side-effects, opt-in styling with usetex only when LaTeX exists, guarded keymaps, no module
state, and narrowed exception handling.
"""
import os
import subprocess
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest
from matplotlib import rcParams

import utils

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'legacy'))

IMPORT_CHECK = '''
import sys, types
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
before = {k: repr(v) for k, v in rcParams.items()}
sys.modules.setdefault('splat', types.SimpleNamespace(Spectrum=None, measureIndexSet=None))
import utils, splot, xcorr, linecentering, crosscorrelate, fullrv, fullrv_spex
changed = sorted(k for k, v in rcParams.items() if repr(v) != before[k])
print('CHANGED', changed)
'''


def test_importing_legacy_modules_leaves_rcparams_alone():
    out = subprocess.run([sys.executable, '-c', IMPORT_CHECK], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, 'MPLBACKEND': 'Agg'})
    assert out.returncode == 0, out.stderr
    assert 'CHANGED []' in out.stdout, out.stdout


@pytest.mark.parametrize('which, expected', [(None, False), ('/usr/bin/latex', True)])
def test_set_plot_style_usetex_follows_latex(monkeypatch, which, expected):
    monkeypatch.setattr(utils.shutil, 'which', lambda name: which)
    with plt.rc_context():
        utils.set_plot_style()
        assert rcParams['text.usetex'] is expected
        assert rcParams['xtick.top'] and rcParams['font.family'] == ['serif']
        utils.set_plot_style(usetex=False)
        assert rcParams['text.usetex'] is False


def test_keymaps_without_arrows_is_pure_and_idempotent():
    import xcorr
    with plt.rc_context({'keymap.back': ['left', 'c', 'backspace'], 'keymap.forward': ['right', 'v']}):
        first = xcorr.keymaps_without_arrows()
        assert first == {'keymap.back': ['c', 'backspace'], 'keymap.forward': ['v']}
        assert rcParams['keymap.back'] == ['left', 'c', 'backspace']  # rcParams untouched
        with plt.rc_context(first):
            assert xcorr.keymaps_without_arrows() == first  # already removed: no error, same result


def test_no_module_level_fit_state():
    import splot
    import xcorr
    assert not hasattr(splot, 'curr_pos') and not hasattr(xcorr, 'curr_pos')


def test_fit_line_wrap_reraises_unexpected_errors(monkeypatch):
    import splot
    fake = SimpleNamespace(ax=None, fitter=SimpleNamespace(fit_info={'param_cov': None}), profilefound=True,
                           rescale=True)

    def raise_value_error(*args, **kwargs):
        raise ValueError('cannot fit')

    monkeypatch.setattr(splot, 'fit_lines', raise_value_error)
    assert splot.Splot.fit_line_wrap(fake) is None  # an expected fit failure is reported, not raised
    assert fake.profilefound is False

    def raise_unexpected_error(*args, **kwargs):
        raise KeyError('a bug')

    monkeypatch.setattr(splot, 'fit_lines', raise_unexpected_error)
    with pytest.raises(KeyError):
        splot.Splot.fit_line_wrap(fake)


def test_templates_query_without_trailing_slash(tmp_path):
    import astropy.units as u
    import numpy as np
    import pandas as pd
    from specutils import Spectrum1D
    import xcorr
    wave = np.linspace(8000., 8010., 50)
    np.savetxt(tmp_path / 'lte020.0-5.0.txt', np.column_stack([np.linspace(7990., 8020., 300), np.ones(300)]))
    fake = SimpleNamespace(tempchanged=True, templatedir=str(tmp_path),  # no trailing '/'
                           templatedf=pd.DataFrame({'teff': [2000.], 'logg': [5.], 'met': [0.]},
                                                   index=['lte020.0-5.0.txt']),
                           teff=2000. * u.K, grav=5. * u.dimensionless_unscaled, met=0. * u.dimensionless_unscaled,
                           spec=Spectrum1D(np.ones(50) * u.Jy, wave * u.AA), wunit=u.AA, spec_index='k1-a',
                           cutspec=lambda spec: spec)
    xcorr.Xcorr.templates_query(fake)
    assert fake.gottemplate
    assert fake.templatefname == os.path.join(str(tmp_path), 'lte020.0-5.0.txt')
