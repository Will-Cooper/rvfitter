"""Tests for rvfitter.features and rvfitter.regions"""
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from rvfitter.features import features, load_catalogue  # noqa: E402
from rvfitter.regions import SelectOptions, der_snr, select  # noqa: E402
from rvfitter.types import Spectrum  # noqa: E402
from synth import make_spectrum  # noqa: E402

METHODS = ['linecentre', 'ccf', 'template_fit', 'forward_mcmc', 'telluric_ref']


def _synth(wave_range, R=2500., snr=60., **kw):
    spec, _ = make_spectrum(0., R, snr, wave_range=wave_range, rng=np.random.default_rng(3), **kw)
    return spec


def _inside(spec, regions):
    return all(r.wmin >= spec.wmin and r.wmax <= spec.wmax for r in regions)


def test_catalogue_loads_and_is_cached():
    t = load_catalogue()
    assert {'name', 'species', 'kind', 'wave_vac', 'wmin', 'wmax', 'teff_min', 'teff_max', 'r_min', 'rest_ref',
            'caveat'} <= set(t.colnames)
    assert set(t['kind']) == {'line', 'band', 'bandhead', 'telluric'}
    assert features() is features()
    kI = next(f for f in features() if f.name == 'K I 7701')
    assert kI.core_only and kI.teff_ok(2000.) and not kI.teff_ok(500.)
    assert all(f.rest_ref for f in features())


def test_der_snr():
    rng = np.random.default_rng(0)
    f = 100. + rng.normal(0., 2., 5000)
    assert der_snr(f) == pytest.approx(50., rel=0.1)
    assert np.isnan(der_snr(np.ones(3)))


def test_optical_only():
    spec = _synth((7400., 9000.))
    res = select(spec, teff=1800., methods=METHODS)
    names = [r.name for r in res['linecentre']]
    assert {'Rb I 7802', 'Rb I 7949', 'Na I 8185', 'Na I 8197', 'Cs I 8523'} <= set(names)
    assert not any(n.startswith(('K I 11', 'K I 12', 'Al', 'Fe')) for n in names)
    for m, regs in res.items():
        assert _inside(spec, regs), m
        assert all(r.reason for r in regs)
    for r in res['linecentre']:
        assert r.centre is not None and r.wmin < r.centre < r.wmax
    assert res['ccf'] and res['template_fit']
    assert res['telluric_ref'] == []  # R=2500 is below every telluric r_min


def test_nir_only():
    spec = _synth((10000., 13500.))
    res = select(spec, teff=1800., methods=METHODS)
    names = {r.name for r in res['linecentre']}
    assert {'K I 11693', 'K I 11776', 'K I 12436', 'K I 12526'} <= names
    assert 'Na I 11385' not in names  # inside the 1.11-1.16 um water band
    assert not any(n.startswith(('Rb', 'Cs')) for n in names)
    for regs in res.values():
        assert _inside(spec, regs)
    # the 1.11-1.16 um water band is masked out of the spans
    for r in res['ccf']:
        assert not (r.wmin < 11600. and r.wmax > 11100.)


def test_combined():
    spec = _synth((7400., 13500.), snr=80.)
    res = select(spec, teff=1800., methods=['linecentre', 'ccf'])
    names = {r.name for r in res['linecentre']}
    assert {'Rb I 7802', 'K I 11693', 'Al I 13127'} - {'Al I 13127'} <= names
    assert any(r.wmax < 9000. for r in res['ccf']) and any(r.wmin > 9900. for r in res['ccf'])


def test_telluric_masking_and_strictness():
    spec = _synth((7400., 9000.))
    std = {r.name for r in select(spec, teff=1800., methods=['linecentre'])['linecentre']}
    assert 'K I 7701' not in std and 'K I 7667' not in std  # in the O2 A band
    off = {r.name for r in select(spec, teff=1800., methods=['linecentre'],
                                  options=SelectOptions(telluric='off'))['linecentre']}
    assert {'K I 7701', 'K I 7667'} <= off
    strict = {r.name for r in select(spec, teff=1800., methods=['linecentre'],
                                     options=SelectOptions(telluric='strict'))['linecentre']}
    assert strict <= std
    assert 'Cs I 8946' not in strict  # next to the strong 8950-9900 A water band
    # spans never overlap the O2 A band
    for r in select(spec, methods=['ccf'])['ccf']:
        assert r.wmax <= 7592. or r.wmin >= 7702.
    with pytest.raises(ValueError):
        select(spec, methods=['ccf'], options=SelectOptions(telluric='bogus'))


def test_continuum_windows_are_feature_free():
    spec = _synth((7400., 9000.))
    for r in select(spec, teff=1800., methods=['linecentre'])['linecentre']:
        assert r.continuum
        for lo, hi in r.continuum:
            assert hi <= r.wmin or lo >= r.wmax
            assert spec.wmin <= lo < hi <= spec.wmax


def test_low_resolution_drops_high_r_lines():
    wr = (10000., 13500.)
    hi = {r.name for r in select(_synth(wr, R=6000.), teff=2500., methods=['linecentre'])['linecentre']}
    lo = {r.name for r in select(_synth(wr, R=1200.), teff=2500., methods=['linecentre'])['linecentre']}
    assert any(n.startswith('Fe I') for n in hi)
    assert not any(n.startswith('Fe I') for n in lo)
    assert not any(n.startswith('Na I') for n in lo)  # r_min 1500
    assert lo & {'K I 11693', 'K I 12436'}


def test_teff_prior():
    spec = _synth((7400., 9000.))
    cool = {r.name for r in select(spec, teff=900., methods=['linecentre'])['linecentre']}
    assert not cool  # all optical alkali lines are for Teff >= 1200 K
    none = {r.name for r in select(spec, teff=None, methods=['linecentre'])['linecentre']}
    assert 'Rb I 7802' in none


def test_gaps_and_nan(caplog):
    spec = _synth((7400., 9000.))
    flux = spec.flux.copy()
    i = np.searchsorted(spec.wave, 7802.4)
    flux[i - 20:i + 20] = np.nan  # NaN gap on Rb I 7802
    wave = spec.wave.copy()
    j = np.searchsorted(wave, 8523.)
    wave[j:] += 40.  # chip gap just before Cs I 8523 (jump of 40 A >> 5x the step)
    s2 = Spectrum(wave, flux, spec.sigma, spec.resolution, spec.meta, 'x')
    names = {r.name for r in select(s2, teff=1800., methods=['linecentre'])['linecentre']}
    assert 'Rb I 7802' not in names and 'Rb I 7949' in names
    res = select(s2, teff=1800., methods=['ccf'])['ccf']
    for r in res:  # no span straddles the gaps
        assert not (r.wmin < 7802. < r.wmax) and not (r.wmin < 8520. < r.wmax)


def test_overrides_on_top_of_auto():
    spec = _synth((7400., 9000.))
    auto = select(spec, teff=1800., methods=['linecentre', 'ccf'])
    names = [r.name for r in auto['linecentre']]
    assert 'Rb I 7802' in names and 'Cs I 8523' in names
    ov = {'linecentre': [{'name': 'Rb I 7802', 'kind': 'line', 'wmin': 7798., 'wmax': 7806., 'centre': 7802.4148},
                         {'name': 'mine', 'kind': 'line', 'wmin': 8300., 'wmax': 8310., 'centre': 8305.},
                         {'name': 'Cs I 8523', 'remove': True},
                         {'name': 'K I 7701', 'kind': 'line', 'wmin': 7695., 'wmax': 7705.}],  # centre from catalogue
          'ccf': [(8000., 8500.)]}
    res = select(spec, teff=1800., methods=['linecentre', 'ccf'], overrides=ov)
    new = {r.name: r for r in res['linecentre']}
    assert 'Cs I 8523' not in new and 'mine' in new
    assert new['Rb I 7802'].reason == 'user override'
    assert (new['Rb I 7802'].wmin, new['Rb I 7802'].wmax) == (7798., 7806.)
    assert new['K I 7701'].centre == pytest.approx(7701.0835)
    assert new['Rb I 7949'].reason != 'user override'  # untouched auto regions stay
    assert new['mine'].continuum  # feature-free continuum is built for line overrides
    assert len(res['ccf']) == len(auto['ccf']) + 1 and res['ccf'][-1].reason == 'user override'
    # removing something not selected only warns
    select(spec, methods=['linecentre'], overrides={'linecentre': [{'name': 'nothing', 'remove': True}]})


def test_line_override_needs_centre():
    spec = _synth((7400., 9000.))
    with pytest.raises(ValueError, match='centre'):
        select(spec, methods=['linecentre'], overrides={'linecentre': [{'name': 'x', 'kind': 'line', 'wmin': 8000.,
                                                                       'wmax': 8010.}]})
    with pytest.raises(ValueError, match='centre'):
        select(spec, methods=['linecentre'], overrides={'linecentre': [(8000., 8010.)]})


def test_override_outside_coverage_warns(caplog):
    spec = _synth((7400., 9000.))
    with caplog.at_level(logging.WARNING):
        res = select(spec, methods=['ccf'], overrides={'ccf': [(20000., 21000.), (8900., 9500.)]})
    assert res['ccf'][-1].wmax == pytest.approx(spec.wmax)
    assert any('outside the coverage' in r.message for r in caplog.records)


def test_options_from_config():
    class Cfg:
        min_snr = 7.
        telluric_strictness = 'strict'

    o = SelectOptions.from_config(Cfg())
    assert (o.snr_min, o.telluric) == (7., 'strict')
    o = SelectOptions.from_config({'min_snr': 4., 'k_line': 3.})
    assert (o.snr_min, o.k_line, o.telluric) == (4., 3., 1.)
    spec = _synth((7400., 9000.))
    assert select(spec, teff=1800., methods=['linecentre'], options=Cfg())['linecentre'] == select(
        spec, teff=1800., methods=['linecentre'], options=SelectOptions(snr_min=7., telluric='strict'))['linecentre']


def test_missing_band_never_raises(caplog):
    # no catalogued feature in 5000-5500 A; all-NaN spectrum; unknown resolution; unknown method
    wave = np.linspace(5000., 5500., 400)
    spec = Spectrum(wave, np.ones(400) + np.random.default_rng(0).normal(0, 0.01, 400), np.full(400, 0.01), 2000.)
    with caplog.at_level(logging.WARNING):
        res = select(spec, teff=1500., methods=METHODS + ['mystery'])
    assert res['linecentre'] == [] and res['telluric_ref'] == []
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    nan = Spectrum(wave, np.full(400, np.nan), np.full(400, np.nan), 2000.)
    assert all(v == [] for v in select(nan, methods=METHODS).values())
    unknown_r = (lambda s: Spectrum(s.wave, s.flux, s.sigma))(_synth((7400., 9000.)))
    assert select(unknown_r, teff=1800., methods=['linecentre'])['linecentre']


def test_low_snr_drops():
    spec = _synth((7400., 9000.), snr=1.5)
    assert select(spec, teff=1800., methods=['linecentre'])['linecentre'] == []


def test_ranking_and_logging(caplog):
    spec = _synth((7400., 9000.), snr=100.)
    with caplog.at_level(logging.DEBUG, logger='rvfitter.regions'):
        res = select(spec, teff=1800., methods=['linecentre'])
    sig = [float(r.reason.split('sigma_RV~')[1].split()[0]) for r in res['linecentre']]
    assert sig == sorted(sig)
    assert any(r.levelno == logging.INFO and 'linecentre' in r.message for r in caplog.records)
    assert any(r.levelno == logging.DEBUG and 'selected' in r.message for r in caplog.records)
    # airmass deepens the water bands: a high-airmass spectrum cannot gain lines
    hi = Spectrum(spec.wave, spec.flux, spec.sigma, spec.resolution, {'airmass': 2.5}, 'x')
    assert {r.name for r in select(hi, teff=1800., methods=['linecentre'])['linecentre']} <= {
        r.name for r in res['linecentre']}


def test_strictness_float_and_aliases():
    spec = _synth((7400., 9000.))

    def names(**kw):
        return {r.name for r in select(spec, teff=1800., methods=['linecentre'],
                                       options=SelectOptions(**kw))['linecentre']}

    assert names(telluric=1.) == names(telluric='standard')
    assert names(telluric=0.) == names(telluric='off') and 'K I 7701' in names(telluric=0.)
    assert names(telluric=2.) == names(telluric='strict')
    assert names(telluric=0.5) == names(telluric='loose')
    assert names(telluric=3.) <= names(telluric=1.) <= names(telluric=0.5) <= names(telluric=0.)
    with pytest.raises(ValueError):
        select(spec, methods=['ccf'], options=SelectOptions(telluric=-1.))


def test_real_config_regions():
    """The repository's own Config().regions (P1) must be accepted"""

    Config = pytest.importorskip('rvfitter.config').Config
    cfg = Config().regions
    spec = _synth((7400., 13500.))
    res = select(spec, teff=1800., methods=['linecentre', 'ccf', 'template_fit'], overrides=cfg.overrides, options=cfg)
    assert res['linecentre'] and res['ccf']
    assert all(len(v) <= cfg.max_regions for v in res.values())
    o = SelectOptions.from_config(cfg)
    assert (o.snr_min, o.telluric, o.cont_fwhm) == (cfg.min_snr, cfg.telluric_strictness, cfg.continuum_fwhm)
