"""
Tests of the telluric transmission model, the telluric (star's own) wavelength zero point, the star − sky constant,
the 'telluric'/'auto' wavelength corrections and the telluric_ref method.

The tests use the tiny synthetic transmission model of `telluric_synth` (conftest); the HITRAN-derived model is
built on demand and tested only where marked (the real model in the cache, or a network build).

The synthetic spectra are built on a 100 000-resolution grid in the observed frame: tellurics at λ_t·D(Δv(λ)),
stellar lines at λ_rest·D(rv)·D(Δv(λ)) (the same distortion, as for light sharing the slit position), multiplied,
convolved with the LSF and averaged over the pixels.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from rvfitter import broadening, config, medium, pipeline
from rvfitter.calibration import sky_zeropoint, telluric
from rvfitter.calibration import telluric_zeropoint as TZ
from rvfitter.types import Region, Spectrum
from synth import make_template

R = 2500.
AIRMASS, PWV = 1.6, 4.


def _dv_true(w):
    return 10. + 20. * (np.asarray(w) - 8500.) / 1000.  # km/s: a zero point plus a distortion


def _observed(dv_func=_dv_true, rv_star=None, snr=80., rng=None, frame_kms=0., with_sky=False,
              star=None) -> Spectrum:
    """A synthetic observed spectrum (7300–10 150 Å, 1.39 Å pixels) with tellurics and optionally a star"""
    rng = rng or np.random.default_rng(1)
    wave = 7300. * np.exp(np.arange(2050) * 1.39 / 8600.)
    step = 299792.458 / 100_000.
    hr = broadening.log_wave_grid(wave[0] - 20., wave[-1] + 20., step)
    f_dv = broadening.doppler_factor(dv_func(hr))
    flux_hr = telluric.transmission(hr / f_dv, AIRMASS, PWV) * (1. + 0.3 * (hr - 8500.) / 1500.)
    if rv_star is not None:
        sw, sf = star
        flux_hr = flux_hr * np.interp(hr / f_dv / broadening.doppler_factor(rv_star), sw, sf)
    flux_hr = broadening.broaden_loglam(flux_hr, step, resolution=R)
    flux = broadening.rebin(hr, flux_hr, wave)
    sigma = flux / snr
    flux = flux + rng.normal(0., 1., len(wave)) * sigma
    meta = {'airmass': AIRMASS}
    sky = None
    if with_sky:
        sky = sky_zeropoint.model_sky(wave / broadening.doppler_factor(dv_func(wave)), 5.5) * 100. + 5.
    if frame_kms:
        f = broadening.doppler_factor(frame_kms)
        wave = wave * f
        meta.update(vel_corr=f, vel_corr_kms=frame_kms, frame_in='heliocentric')
    return Spectrum(wave, flux, sigma, None, meta, 'synthetic', sky)


# ----------------------------------------------------------------------------------------------------- model
def _check_vacuum_and_cited(m) -> None:
    assert m.wmin == pytest.approx(6000., abs=1.) and m.wmax == pytest.approx(25000., abs=1.)
    assert 'HITRAN' in m.meta['citation'] and 'citepolicy' in m.meta['terms']
    assert {'H2O', 'O2', 'CO2', 'CH4'} <= set(m.meta['molecules'])
    # the blue edges of the O2 A and B bands sit just redward of the Fraunhofer A and B lines in vacuum, and far
    # from their air values (an air-wavelength model would put them ~2 Å bluer)
    w = broadening.log_wave_grid(6800., 7700., 299792.458 / 400_000.)
    t = telluric.transmission(w, 1., 2.5, resolution=20_000.)
    for lo, air in ((7580., 7593.70), (6855., 6867.19)):
        s = w > lo
        edge = w[s][np.argmax(t[s] < 0.5)]
        vac = float(medium.air_to_vac(np.array([air]))[0])
        assert 0. < edge - vac < 1.6, (edge, vac)
        assert edge - air > 2.5


def test_cached_model_is_vacuum_and_cited(real_telluric_model):
    """The model built in the user cache (skipped if it has not been built)"""
    if 'terms' not in real_telluric_model.meta:
        pytest.skip('the cached model predates the provenance metadata; rebuild it')
    _check_vacuum_and_cited(real_telluric_model)


@pytest.mark.network
@pytest.mark.slow
def test_build_from_hitran(tmp_path, monkeypatch):
    """The real build from HITRANonline (a few minutes; about 90 MB of line lists)"""
    from rvfitter.calibration import telluric_build
    out = tmp_path / 'telluric_hitran.npz'
    monkeypatch.setenv('RVFITTER_TELLURIC_MODEL', str(out))
    try:
        telluric_build.build(out, line_cache=tmp_path / 'lines')
    except RuntimeError as exc:
        pytest.skip(f'HITRAN unreachable: {exc}')
    _check_vacuum_and_cited(telluric.model())
    side = json.loads(out.with_suffix('.json').read_text())
    assert side['sha256'] and all(v['sha256'] and v['fetched'] for v in side['molecules'].values())


def _record(nu: float, s: float, mol: int = 7, iso: int = 1) -> str:
    """One HITRAN 160-character record (the fields the builder reads)"""
    rec = (f'{mol:2d}{iso:1d}{nu:12.6f}{s:10.3E}{1.0:10.3E}{0.05:5.3f}{0.08:5.3f}{100.:10.4f}{0.7:4.2f}'
           f'{-0.005:8.6f}')
    return rec.ljust(160)


def test_build_offline_with_fake_lines(tmp_path, monkeypatch):
    """The build pipeline, provenance and loading, with the HTTP layer replaced by a one-line fake"""
    from rvfitter.calibration import telluric_build
    nu_o2 = 1e8 / 7621.  # an O2 line at 7621 Å (vacuum)
    calls = []

    def fake_get(url):
        calls.append(url)
        lo = float(url.split('numin=')[1].split('&')[0])
        hi = float(url.split('numax=')[1].split('&')[0])
        if 'iso_ids_list=36' in url and lo <= nu_o2 < hi:
            return (_record(nu_o2, 1e-22) + '\n').encode()
        return None  # the API's 404: no lines in the interval

    out = tmp_path / 'model.npz'
    monkeypatch.setenv('RVFITTER_TELLURIC_MODEL', str(out))
    path = telluric_build.build(out, line_cache=tmp_path / 'lines', get=fake_get)
    assert path == out and out.is_file() and calls
    m = telluric.model()
    assert m.meta['citation'] == telluric_build.CITATION and 'not be redistributed' in m.meta['terms']
    assert m.meta['molecules']['O2']['n_lines_file'] == 1 and len(m.meta['molecules']['O2']['sha256']) == 64
    side = json.loads(out.with_suffix('.json').read_text())
    assert len(side['sha256']) == 64 and side['file'] == 'model.npz'
    w = np.linspace(7615., 7627., 400)
    t = telluric.transmission(w, 1., 2.5, resolution=20_000.)
    assert abs(w[np.argmin(t)] - 7621.) < 0.3 and t.min() < 0.99
    # a second build reuses the cached line files (no network) and is identical
    n_calls = len(calls)
    telluric_build.build(tmp_path / 'again.npz', line_cache=tmp_path / 'lines', get=fake_get)
    assert len(calls) == n_calls
    a, b = np.load(out), np.load(tmp_path / 'again.npz')
    assert np.array_equal(a['tau_dry'], b['tau_dry']) and np.array_equal(a['tau_h2o'], b['tau_h2o'])


def test_unavailable_model_skips_cleanly(tmp_path, monkeypatch, caplog):
    """Without the model the telluric terms log how to build it and skip; nothing crashes"""
    monkeypatch.setenv('RVFITTER_TELLURIC_MODEL', str(tmp_path / 'missing.npz'))
    assert not telluric.available()
    with pytest.raises(telluric.TelluricModelUnavailable, match='templates telluric --build'):
        telluric.model()
    w = np.linspace(7300., 10000., 2000)
    spec = Spectrum(w, np.ones_like(w), np.full_like(w, 0.01), 2500., {'airmass': 1.2}, 'flat')
    caplog.set_level('WARNING')
    assert TZ.measure(spec) is None
    assert 'templates telluric --build' in caplog.text
    from rvfitter.methods import REGISTRY
    res = REGISTRY['telluric_ref'].measure(spec, [], {'input': {'resolution': 2500.}}, rng=np.random.default_rng(0))
    assert not np.isfinite(res.rv) and res.measurements[0].flag & 1
    cfg = config.Config()
    cfg.calibration.wavelength_correction = 'telluric'
    _, info = pipeline.wavelength_correction(spec, spec, cfg, 'X')
    assert info['applied'] == 'none' and info['zp_flags'] & pipeline.ZP_FLAGS['telluric_unavailable']


def test_transmission_scaling_shift_and_mask():
    w = broadening.log_wave_grid(7550., 7750., 299792.458 / 200_000.)
    t1 = telluric.transmission(w, 1., 2.5, resolution=R)
    t2 = telluric.transmission(w, 2., 2.5, resolution=R)
    assert t2.min() < t1.min() < 0.5 and np.all(t1 <= 1. + 1e-9)
    # a positive dv moves the band to the red
    ts = telluric.transmission(w, 1., 2.5, resolution=R, dv=50.)
    edge = lambda tt: w[np.argmax(tt < 0.6)]  # noqa: E731
    assert edge(ts) - edge(t1) == pytest.approx(7600. * 50. / 299792.458, abs=0.3)
    assert telluric.mask(w, 0.3, resolution=R).any() and not telluric.mask(np.array([8000.5, 8001.]), 0.3).any()
    assert np.all(telluric.transmission(np.array([5000., 30000.]), 1., 2.5) == 1.)  # outside the coverage


# ---------------------------------------------------------------------------------------- telluric zero point
@pytest.mark.parametrize('frame_kms', [0., -21.])
def test_recovers_zero_point_and_distortion(frame_kms):
    spec = _observed(frame_kms=frame_kms)
    zp = TZ.measure(spec, resolution=R)
    assert zp is not None and zp.n_used >= 5
    for s in zp.segments:
        if s['used']:
            assert abs(s['dv'] - _dv_true(s['centre'])) < 3.5 * s['err'] + 0.5, s
    assert abs(zp.dv(9300.) - _dv_true(9300.)) < 4.
    dv, err = zp  # unpacks as the architecture's (dv, err)
    assert np.isfinite(dv) and err > 0


def test_estimates_resolution_from_tellurics():
    zp = TZ.measure(_observed())
    assert zp.resolution_source == 'telluric lines' and abs(zp.resolution / R - 1.) < 0.1


def test_star_minus_sky_constant():
    spec = _observed(with_sky=True)
    sky = sky_zeropoint.measure(spec)
    # the star sits 12 km/s further to the red than the sky (e.g. off-centre in the slit)
    star = _observed(dv_func=lambda w: _dv_true(w) + 12.)
    tell = TZ.measure(star, resolution=R)
    ss = TZ.star_minus_sky(tell, sky)
    assert abs(ss['zp_star_sky'] - 12.) < 3. * ss['zp_star_sky_err'] + 1., ss
    assert len(ss['nodes']) == tell.n_used
    # only nodes inside a measured sky segment are compared: never the ramps across the O2 A / H2O 9300 gaps
    measured = [(s['wmin'], s['wmax']) for s in sky.segments if s['used']]
    assert ss['n_nodes'] >= 1
    for node in ss['nodes']:
        assert node['used'] == any(a <= node['wave'] <= b for a, b in measured)
    assert ss['n_nodes'] < tell.n_used  # the 9300 Å nodes lie in the sky's unmeasured gap
    assert ss['zp_star_sky_err'] >= sky.model_spread - 1e-9


def test_star_minus_sky_without_overlap_is_nan(caplog):
    sky = sky_zeropoint.measure(_observed(with_sky=True))
    tell = TZ.measure(_observed(), [Region('H2O 9300', 'telluric', 9000., 9800.)], resolution=R)
    caplog.set_level('INFO')
    ss = TZ.star_minus_sky(tell, sky)
    assert ss['n_nodes'] == 0 and np.isnan(ss['zp_star_sky']) and np.isnan(ss['zp_star_sky_err'])
    assert 'measured sky segment' in ss['reason'] and 'not measured' in caplog.text


def test_o2a_dropped_for_cool_targets():
    """The catalogue marks O2 A as a safe zero-point reference only above 2800 K (the K I wings fill it below)"""
    spec = _observed()
    warm = TZ.measure(spec, resolution=R, teff=3200.)
    cool = TZ.measure(spec, resolution=R, teff=2100.)
    assert any(s['band'] == 'O2 A' and s['used'] for s in warm.segments)
    assert not any(s['band'] == 'O2 A' for s in cool.segments)
    assert cool.n_used == warm.n_used - sum(1 for s in warm.segments if s['band'] == 'O2 A' and s['used'])


def test_pipeline_telluric_and_auto_modes():
    raw = _observed(with_sky=True)
    cfg = config.Config()
    cfg.input.resolution = R
    cfg.calibration.telluric_template = False
    cfg.calibration.wavelength_correction = 'none'
    spec0, info0 = pipeline.wavelength_correction(raw, raw, cfg, 'X')  # 'none': the telluric term is not measured
    assert info0['applied'] == 'none' and np.isnan(info0['zp_tell_kms'])
    cfg.calibration.wavelength_correction = 'telluric'
    spec, info = pipeline.wavelength_correction(raw, raw, cfg, 'X')
    assert info['applied'] == 'telluric' and np.isfinite(info['zp_tell_kms'])
    assert np.isfinite(info['zp_star_sky']) and abs(info['zp_star_sky']) < 3. * info['zp_star_sky_err'] + 2.
    assert info['zp_star_sky_n'] >= 1 and info['zp_flags'] & pipeline.ZP_FLAGS['telluric_applied']
    lam = 9300.
    corrected = float(np.interp(lam, raw.wave, spec.wave))
    assert (lam / corrected - 1.) * 299792.458 == pytest.approx(_dv_true(lam), abs=5.)
    # 'auto' is the default sky-if-needed: the sky curve when needed, plus the star - sky constant when measured
    regs = [Region('Na I', 'line', 8175., 8205.), Region('Rb I', 'line', 7940., 7960.),
            Region('Cs I', 'line', 8515., 8530.)]
    from rvfitter import systematics
    for mode in ('auto', 'sky-if-needed'):
        cfg.calibration.wavelength_correction = mode
        spec_a, info_a = pipeline.wavelength_correction(raw, raw, cfg, 'X', regions=regs)
        assert info_a['applied'] == 'sky+star' and info_a['star_sky_applied']
        assert info_a['zp_flags'] & pipeline.ZP_FLAGS['star_sky_applied']
        assert info_a['zp_star_sky_n'] >= 2 and info_a['star_sky']['method'] == 'direct sky'
        assert abs(info_a['zp_star_sky']) < 3. * info_a['zp_star_sky_err'] + 2.  # the star sees the sky's curve
        wc = spec_a.meta['wavelength_correction']
        assert wc['constant_kms'] == pytest.approx(info_a['zp_star_sky']) and wc['sky'] is not None
        # the constant's error is sys_star_sky, not part of sys_wavezp
        assert systematics.wavelength_zero_point(spec_a, {}) == pytest.approx(info_a['zp_sky_err'])
    nosky = Spectrum(raw.wave, raw.flux, raw.sigma, None, raw.meta, 'x')
    _, info_t = pipeline.wavelength_correction(nosky, nosky, cfg, 'X', regions=regs)
    assert info_t['applied'] == 'none' and info_t['zp_flags'] & pipeline.ZP_FLAGS['no_sky']


def test_telluric_curve_refused_far_from_nodes():
    """A region far from every telluric node is not corrected by the telluric curve"""
    raw = _observed()
    cfg = config.Config()
    cfg.input.resolution = R
    cfg.calibration.telluric_template = False
    cfg.calibration.wavelength_correction = 'telluric'
    _, ok = pipeline.wavelength_correction(raw, raw, cfg, 'X', regions=[Region('Na I', 'line', 8180., 8200.)])
    assert ok['applied'] == 'telluric'
    out, bad = pipeline.wavelength_correction(raw, raw, cfg, 'X', regions=[Region('X', 'line', 10700., 10720.)])
    assert bad['applied'] == 'none' and bad['zp_flags'] & pipeline.ZP_FLAGS['telluric_refused']
    assert np.array_equal(out.wave, raw.wave)


# ------------------------------------------------------------------------------------------------ telluric_ref
@pytest.fixture(scope='module')
def star():
    return make_template((7000., 10500.), n_random=400, rng=np.random.default_rng(3))


def test_telluric_ref_recovers_rv_despite_distortion(star):
    from rvfitter.methods import REGISTRY
    rv_true, frame = 30., -21.
    spec = _observed(rv_star=rv_true, star=star, frame_kms=frame, snr=100.)
    tmpl = Spectrum(star[0], star[1], np.full(len(star[0]), np.nan), None, {'object': 'toy'}, 'toy')
    cfg = {'input': {'resolution': R}, 'methods': {'options': {'telluric_ref': {'template': tmpl}}}}
    res = REGISTRY['telluric_ref'].measure(spec, [], cfg, rng=np.random.default_rng(0))
    assert res.n_regions >= 4
    expected = float(299792.458 * np.tanh(np.arctanh(rv_true / 299792.458) + np.arctanh(frame / 299792.458)))
    assert abs(res.rv - expected) < 3. * res.rv_err + 0.5, (res.rv, res.rv_err, expected)
    m = [x for x in res.measurements if np.isfinite(x.rv)]
    # each segment's telluric zero point is the injected distortion (observatory frame)
    assert all(abs(x.diagnostics['dv'] - _dv_true(x.diagnostics['wave_centre'])) < 4. * x.diagnostics['err'] + 1.
               for x in m)


# --------------------------------------------------------------------------------------------------- pull test
def _effective_truth(seg: dict) -> float:
    """
    The Δv a segment measures when Δv varies across it: the mean over its pixels weighted by the shift
    information (dT/dλ)² of the telluric lines, not the value at the segment centre
    """
    pad = 15. * 0.5 * (seg['wmin'] + seg['wmax']) / R
    w = 7300. * np.exp(np.arange(2050) * 1.39 / 8600.)
    w = w[(w >= seg['wmin'] - pad) & (w <= seg['wmax'] + pad)]
    info = np.gradient(telluric.transmission(w, AIRMASS, PWV, resolution=R), w) ** 2
    return float(np.sum(info * _dv_true(w)) / np.sum(info))


@pytest.mark.slow
@pytest.mark.parametrize('distorted', [False, True])
def test_injection_pulls(distorted):
    """A constant zero point, or the zero point plus distortion, in 30 noise realisations: bias and pull width"""
    pulls = []
    for k in range(30):
        func = _dv_true if distorted else (lambda w: np.full(np.shape(w), 15.))
        zp = TZ.measure(_observed(dv_func=func, rng=np.random.default_rng(100 + k), snr=60.), resolution=R)
        for s in zp.segments:
            if s['used'] and s['err'] < 20.:
                truth = _effective_truth(s) if distorted else 15.
                pulls.append((s['dv'] - truth) / s['err'])
    pulls = np.asarray(pulls)
    assert len(pulls) > 150
    assert abs(np.mean(pulls)) < 0.25 and 0.7 < np.std(pulls) < 1.3, (np.mean(pulls), np.std(pulls))


# --------------------------------------------------------------------------------------------------- J1745 data
REL = Path('pypeit/R2500I/GTC54-15A0/OB0055/sciencebk/nopipecorr/'
           'spec1d_0000809229-20150803-OSIRIS-OsirisLongSlitSpectroscopy-DENISJ174534.6-164053_OSIRIS_20150804T003233'
           '.951.fits')
TELLCORR = Path('pypeit/R2500I/GTC54-15A0/OB0055/Science/DENISJ174534.6-164053_tellcorr.fits')


@pytest.mark.data
def test_j1745_telluric_zero_point(real_telluric_model):
    root = os.environ.get('RVFITTER_TESTDATA')
    if not root:
        pytest.skip('RVFITTER_TESTDATA not set')
    from rvfitter import io
    target = io.read_spectrum(Path(root) / TELLCORR, medium='vacuum')
    cfg = config.Config()
    cfg.input.medium = 'vacuum'
    # the spec1d holds two objects; the one matching the (target) coadd is HDU 2, not the first
    src = pipeline._matching_object(str(Path(root) / REL), target, cfg, 'J1745')
    assert src.source.endswith('.fits') and np.nanmedian(src.flux / src.sigma) > 50.
    zp = TZ.measure(src, resolution=2602.)
    assert zp is not None and zp.n_used >= 5
    assert all(abs(s['dv']) < 100. for s in zp.segments if s['used'])
