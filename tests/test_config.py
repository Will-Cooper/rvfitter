"""Tests for rvfitter.config: round trips, validation messages, hashing and regime defaults."""
import numpy as np
import pytest

from rvfitter import config
from rvfitter.config import Config, ConfigError, RegionOverride, default_for, regime_for
from rvfitter.types import Spectrum


def _full() -> Config:
    cfg = Config(seed=42)
    cfg.input.paths = ['a.fits', 'dir/']
    cfg.input.medium = 'air'
    cfg.input.medium_method = 'edlen'
    cfg.input.hdu = 'SPAT0168-SLIT0457-DET02'
    cfg.input.resolution = 2500.
    cfg.velocity_frame.correction = 'skip'
    cfg.velocity_frame.frame = 'heliocentric'
    cfg.velocity_frame.ra, cfg.velocity_frame.dec, cfg.velocity_frame.mjd = 266.4, -16.6, 57238.1
    cfg.velocity_frame.site = 'GTC'
    cfg.templates.local_dir = '/tmp/t'
    cfg.templates.medium = 'air'
    cfg.templates.teff_range = (1500., 2500.)
    cfg.templates.logg_prior = (5., 0.3)
    cfg.methods.enabled = ['template_fit', 'ccf']
    cfg.methods.options = {'ccf': {'oversample': 5, 'mask': [1.0, 2.0], 'sub': {'x': True}}}
    cfg.regions.overrides = [RegionOverride(name='K I', method='ccf', wmin=7660., wmax=7710., centre=7701.,
                                            continuum=[(7600., 7650.), (7720., 7760.)]),
                             RegionOverride(name='bad', wmin=7600., wmax=7700., action='exclude')]
    cfg.combine.prior_scale = 2.
    cfg.combine.systematic_floor = 1.5
    cfg.parallel.n_procs = 3
    cfg.output.formats = ['csv', 'html']
    cfg.output.medium = 'air'
    return cfg


def test_round_trip(tmp_path):
    cfg = _full()
    config.dump(cfg, tmp_path / 'c.toml')
    back = config.load(tmp_path / 'c.toml')
    assert back == cfg
    assert back.hash() == cfg.hash()
    assert isinstance(back.templates.teff_range, tuple)
    assert back.regions.overrides[0].continuum == [(7600., 7650.), (7720., 7760.)]


def test_default_round_trip_and_loads(tmp_path):
    cfg = Config()
    config.dump(cfg, tmp_path / 'd.toml')
    assert config.load(tmp_path / 'd.toml') == cfg
    assert config.loads(config.dumps(cfg)) == cfg
    assert config.loads('') == cfg  # everything has a default


def test_hash_properties():
    a, b = Config(), Config()
    assert len(a.hash()) == 64 and a.hash() == b.hash()
    b.seed = 1
    assert a.hash() != b.hash()
    c = Config()
    c.methods.options = {'ccf': {'b': 1, 'a': 2}}
    d = Config()
    d.methods.options = {'ccf': {'a': 2, 'b': 1}}
    assert c.hash() == d.hash()  # canonical: key order does not matter


def test_partial_file_uses_defaults(tmp_path):
    (tmp_path / 'p.toml').write_text('seed = 7\n[methods]\nenabled = ["ccf"]\n')
    cfg = config.load(tmp_path / 'p.toml')
    assert cfg.seed == 7 and cfg.methods.enabled == ['ccf'] and cfg.input.medium == 'auto'


def test_velocity_overrides():
    cfg = Config()
    assert cfg.velocity_frame.overrides() == {}
    cfg.velocity_frame.ra = 10.
    cfg.velocity_frame.site = 'LCO'
    assert cfg.velocity_frame.overrides() == {'ra': 10., 'site': 'LCO'}


@pytest.mark.parametrize('text, needle', [
    ('[input]\nmedum = "air"\n', "did you mean 'medium'"),
    ('[nonsense]\nx = 1\n', "Unknown key in the top level"),
    ('[input]\nmedium = "mud"\n', "input.medium = 'mud' is not one of auto, air, vacuum"),
    ('[input]\nmedium = 3\n', 'input.medium: expected a string'),
    ('seed = "x"\n', 'seed: expected an integer'),
    ('[parallel]\nn_procs = 0\n', 'parallel.n_procs = 0 must be >= 1'),
    ('[methods]\nenabled = ["ccf", "ccf"]\n', 'duplicates'),
    ('[methods]\nenabled = ["cff"]\n', "did you mean 'ccf'"),
    ('[methods]\nenabled = []\n', 'methods.enabled is empty'),
    ('[methods.options.bogus]\nx = 1\n', 'methods.options key'),
    ('[templates]\nteff_range = [2500, 1500]\n', 'templates.teff_range'),
    ('[templates]\nteff_range = [1500]\n', 'templates.teff_range: expected a list of 2'),
    ('[templates]\nteff_prior = [2000, 0]\n', 'sd > 0'),
    ('[output]\nformats = ["pdf"]\n', "output.formats[0] = 'pdf'"),
    ('[velocity_frame]\ndec = 120.0\n', 'velocity_frame.dec'),
    ('[velocity_frame]\nsite_lat = 10.0\n', 'site_lat and site_lon must be given together'),
    ('[[regions.overrides]]\nname = "x"\nwmin = 8000.0\nwmax = 7000.0\n', 'regions.overrides[0]'),
    ('[combine]\nprior_scale = -1.0\n', 'combine.prior_scale'),
    ('schema_version = 99\n', 'schema_version'),
    ('seed = [\n', 'Invalid TOML'),
])
def test_validation_errors(text, needle, tmp_path):
    with pytest.raises(ConfigError) as exc:
        config.loads(text)
    assert needle in str(exc.value)
    (tmp_path / 'bad.toml').write_text(text)
    with pytest.raises(ConfigError, match='bad.toml'):
        config.load(tmp_path / 'bad.toml')


def test_all_problems_reported_together():
    with pytest.raises(ConfigError) as exc:
        config.loads('[parallel]\nn_procs = 0\n[combine]\nprior_scale = 0.0\n')
    assert 'n_procs' in str(exc.value) and 'prior_scale' in str(exc.value)


def test_dump_refuses_invalid(tmp_path):
    cfg = Config()
    cfg.parallel.n_procs = -2
    with pytest.raises(ConfigError):
        config.dump(cfg, tmp_path / 'x.toml')
    assert not (tmp_path / 'x.toml').exists()


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        config.load(tmp_path / 'absent.toml')


def _spec(wmin, wmax, R, medium='vacuum'):
    w = np.linspace(wmin, wmax, 50)
    return Spectrum(w, np.ones(50), np.ones(50), R, {'medium_in': medium}, 'x.fits')


@pytest.mark.parametrize('R, wr, expect', [
    (2500., (7300., 10150.), ('low', 'optical')),
    (None, (7300., 10150.), ('low', 'optical')),
    (150., (7000., 25000.), ('low', 'mixed')),
    (2000., (10000., 24000.), ('low', 'nir')),
    (4100., (3500., 9000.), ('low', 'optical')),
    (6000., (7000., 9000.), ('mid', 'optical')),
    (20000., (11500., 13000.), ('mid', 'nir')),
    (50000., (8000., 25000.), ('high', 'mixed')),
])
def test_regime_for(R, wr, expect):
    assert regime_for(R, wr) == expect


def test_default_for_regimes():
    opt = default_for(_spec(7300., 10150., 2500.))
    assert opt.methods.enabled == ['template_fit', 'ccf', 'linecentre'] and opt.input.resolution == 2500.
    assert opt.input.medium == 'vacuum'
    nir = default_for(_spec(10000., 24000., 2000., medium='air'))
    assert nir.methods.enabled == ['template_fit', 'ccf']
    assert nir.methods.options['template_fit']['telluric_model'] is True and nir.input.medium == 'air'
    mid = default_for(_spec(7000., 9000., 6000.))
    assert mid.methods.enabled[0] == 'forward_mcmc' and mid.methods.options['forward_mcmc']['fit_vsini'] is True
    high = default_for(_spec(9000., 25000., 60000.))
    assert 'telluric_ref' in high.methods.enabled
    assert high.methods.options['forward_mcmc']['fit_wave_offset'] is True
    for cfg in (opt, nir, mid, high):
        cfg.validate()
        assert config.loads(config.dumps(cfg)) == cfg


def test_default_for_unknown_resolution_warns(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger='rvfitter.config'):
        cfg = default_for(_spec(7300., 10150., None))
    assert 'unknown' in caplog.text and cfg.input.resolution is None
    assert 'resolution' not in cfg.to_dict()['input']  # None is omitted from the TOML
