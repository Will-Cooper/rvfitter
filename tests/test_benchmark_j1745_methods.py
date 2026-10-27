"""
J1745−1640 benchmark for the methods package (needs RVFITTER_TESTDATA).

The GTC/OSIRIS R2500I telluric-corrected spectrum is PypeIt vacuum and already heliocentric, so the barycentric
step is skipped. Regions come from `regions.select` with `telluric_corrected=True` and the default feature
windows, and R from `estimate_resolution` (the file has none).
Methods are compared with the benchmark 26.2 ± 2.3 km/s through z = |RV − 26.2| / √(2.3² + σ²). With the
feature windows both methods give 37–44 km/s (z ≈ 2–3.6), driven by Cs I 8523 (≈50 km/s in every method,
template and R, as in the Cooper et al. 2024 line centring); this is investigated in docs/dev_notes_methods.md
and is not tuned away. The tests therefore record the values and guard against regressions (z < 4), rather
than require z ≤ 2.

`test_j1745_like_injection` (slow) injects BT-Settl into J1745's own pixel grid, σ, masks and windows; it is
the test that catches continuum-handling bugs which the toy spectra cannot (e.g. the CCF high-pass footprint
bug fixed in 34133d6).
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.data

BENCH_RV, BENCH_ERR = 26.2, 2.3
REL = Path('pypeit/R2500I/GTC54-15A0/OB0055/Science/DENISJ174534.6-164053_tellcorr.fits')


def _root() -> Path:
    root = os.environ.get('RVFITTER_TESTDATA')
    if not root:
        pytest.skip('RVFITTER_TESTDATA not set')
    return Path(root)


@pytest.fixture(scope='module')
def j1745():
    from rvfitter import io
    return io.read_spectrum(_root() / REL, medium='vacuum')


def _cfg(mode='features'):
    from rvfitter import config
    c = config.Config()
    c.templates.local_dir = str(_root() / 'templates' / 'bt-settl-cifist' / 'useful')
    c.templates.teff_prior = 2100.
    c.input.telluric_corrected = True
    c.regions.template_windows = mode
    c.velocity_frame.correction = 'skip'
    c.velocity_frame.frame = 'heliocentric'
    c.methods.enabled = ['ccf', 'template_fit']
    c.methods.options = {'ccf': {'n_mc': 100}}
    return c


@pytest.fixture(scope='module')
def cfg():
    return _cfg()


@pytest.fixture(scope='module')
def resolution(j1745, cfg):
    from rvfitter.methods import estimate_resolution
    return estimate_resolution(j1745, cfg=cfg)


def _z(rv, err):
    return abs(rv - BENCH_RV) / np.hypot(BENCH_ERR, err)


def _select(spec, mode, names):
    from rvfitter import regions
    return regions.select(spec, teff=2100., methods=names,
                          options=regions.SelectOptions(telluric_corrected=True, template_windows=mode))


def test_estimate_resolution(resolution):
    assert resolution['method'] == 'template_fit'
    assert 1800. < resolution['resolution'] < 3300. and resolution['resolution_err'] < 400.


@pytest.mark.parametrize('mode', ['features', 'full'])
@pytest.mark.parametrize('name', ['ccf', 'template_fit'])
def test_method_on_j1745(j1745, resolution, name, mode):
    from rvfitter.methods import REGISTRY
    cfg = _cfg(mode)
    cfg.input.resolution = round(resolution['resolution'])
    res = REGISTRY[name].measure(j1745, _select(j1745, mode, [name])[name], cfg, rng=np.random.default_rng(42))
    assert res.n_regions >= 3
    assert np.isfinite(res.rv) and np.isfinite(res.rv_err)
    assert _z(res.rv, res.rv_err) < 4., (res.rv, res.rv_err)  # regression guard; see the module docstring
    if mode == 'features':
        assert res.rv_err < 8.  # the feature windows agree far better than the full spans


def test_cli_measure_writes_a_row(j1745, cfg, resolution, tmp_path):
    import pandas as pd

    from rvfitter import cli, config
    cfg.input.resolution = round(resolution['resolution'])
    toml = tmp_path / 'j1745.toml'
    config.dump(cfg, toml)
    out = tmp_path / 'results.csv'
    code = cli.main(['measure', str(_root() / REL), '--config', str(toml), '--bary', 'skip', '--telluric-corrected',
                     '--medium', 'vacuum', '--out', str(out), '--n-procs', '1', '--no-plots', '-q'])
    assert code == 0
    table = pd.read_csv(out)
    assert len(table) == 1
    for name in ('ccf', 'template_fit'):
        assert np.isfinite(table[f'rv_{name}'].iloc[0])
    assert np.isfinite(table['rv_adopted'].iloc[0])


# ------------------------------------------------------------------------------------------------------------
# BT-Settl injected into J1745's own grid, σ, masks and windows
# ------------------------------------------------------------------------------------------------------------
TRUTH = (2000., 5.0)
R_INJ = 2300.


def make_j1745_like(spec, rng, v_true=BENCH_RV, cont_kms=20000.):
    """BT-Settl 2000/5.0 at R 2300, shifted by v_true, on J1745's pixels with its σ and broad flux shape"""
    from rvfitter.methods import _common as C
    from rvfitter.types import Spectrum
    grid = C.open_grid({'local_dir': str(_root() / 'templates' / 'bt-settl-cifist' / 'useful'),
                        'grid': 'bt-settl-cifist'})
    tpl = C.build_template(C.Candidate('truth', grid=grid, params=(*TRUTH, 0.)), R_INJ, (spec.wmin, spec.wmax))
    model = tpl.at(spec.wave, v_true)
    good = np.isfinite(spec.flux) & np.isfinite(spec.sigma) & (spec.sigma > 0) & np.isfinite(model)
    lnw = np.log(spec.wave)
    u = np.linspace(lnw[0], lnw[-1], len(lnw))
    ratio = np.interp(u, lnw[good], (spec.flux / model)[good])
    cont = np.interp(lnw, u, C.highpass(np.exp(u), ratio, np.ones(len(u), bool), cont_kms))
    if cont_kms is None or os.environ.get('RVFITTER_FLAT_CONTINUUM'):  # diagnostic: flat continuum
        cont = np.full_like(model, np.nanmedian(spec.flux[good] / model[good]))
    noisy = model * cont + rng.normal(0., 1., len(model)) * np.where(good, spec.sigma, 0.)
    return Spectrum(spec.wave, np.where(good, noisy, np.nan), np.where(good, spec.sigma, np.nan), R_INJ,
                    {'object': 'J1745-like'}, 'synthetic')


def j1745_like_pulls(spec, measure, name, n=20, seed=0):
    """
    Pull summaries of `measure` on n J1745-like injections with the exact template (pinned)

    Returns
    -------
    method, windows
        The method-level summary (random-effects combination) and the pooled per-window summary
    """
    cfg = _cfg()
    cfg.input.resolution = R_INJ
    cfg.templates.teff_range = (TRUTH[0], TRUTH[0])
    cfg.templates.logg_range = (TRUTH[1], TRUTH[1])
    cfg.methods.options = {'ccf': {'n_mc': 50}}
    rvs, errs, wrv, werr = [], [], [], []
    for child in np.random.SeedSequence(seed).spawn(n):
        noise, meth = child.spawn(2)
        fake = make_j1745_like(spec, np.random.default_rng(noise))
        regs = _select(fake, 'features', [name])[name]
        res = measure(fake, regs, cfg, rng=np.random.default_rng(meth))
        rvs.append(res.rv)
        errs.append(res.rv_err)
        for m in res.measurements:
            if np.isfinite(m.rv) and np.isfinite(m.rv_err):
                wrv.append(m.rv)
                werr.append(m.rv_err)

    def summ(r, e):
        r, e = np.array(r), np.array(e)
        pull = (r - BENCH_RV) / e
        return dict(bias_sigma=float(np.mean(pull)), pull_std=float(np.std(pull, ddof=1)), bias=r.mean() - BENCH_RV,
                    scatter=np.std(r, ddof=1), mean_err=e.mean(), n=len(r))

    return summ(rvs, errs), summ(wrv, werr)


@pytest.mark.slow
@pytest.mark.parametrize('name', ['ccf', 'template_fit'])
def test_j1745_like_injection(j1745, name):
    from rvfitter.methods import REGISTRY
    method, windows = j1745_like_pulls(j1745, REGISTRY[name].measure, name, n=80)
    # per-window statistical errors: calibrated (N ~ 320 pooled windows)
    assert abs(windows['bias_sigma']) < 0.2, windows
    assert 0.8 < windows['pull_std'] < 1.2, windows
    # method level: the random-effects combination over ~4 windows marginalises tau, so it is conservative
    assert abs(method['bias_sigma']) < 0.25, method
    assert 0.6 < method['pull_std'] < 1.2, method
    # the pre-34133d6 CCF (template high-passed on its own grid) gave a bias of -1.6 sigma and pull SD 0.38 here
