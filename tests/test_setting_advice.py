"""config.setting_advice and grid_advice: the per-setting recommendations of the Advanced dialog."""
import dataclasses
from pathlib import Path

import pytest

from rvfitter import config
from rvfitter.gui.backend import LazyBackend

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
SKIPPED = {'config.schema_version', 'config.refits'}


def _example(name):
    """The example's config and the first spectrum of its first object, read through the real GUI backend"""
    be = LazyBackend()
    cfg = config.load(EXAMPLES / name / 'rvfitter.toml')
    files = [f for p in cfg.input.paths for f in be.collect_inputs(Path(p))]
    sources = next(iter(be.group_objects(files, cfg).values()))
    return be, cfg, be.read_spectrum(sources[0], cfg)


def _values(advice):
    return {k: a.value for k, a in advice.items()}


@pytest.mark.parametrize('name, expected', [
    ('gtc_osiris', {'input.telluric_corrected': True, 'calibration.wavelength_correction': 'sky-if-needed',
                    'velocity_frame.correction': None, 'templates.grid': 'bt-settl-cifist',
                    'templates.teff_range': (1800., 2400.), 'templates.logg_range': (4.5, 5.5),
                    'regions.template_windows': 'features', 'calibration.zero_point_known': None,
                    'methods.enabled': ['template_fit', 'ccf', 'linecentre']}),
    ('near_infrared', {'input.medium': None, 'input.telluric_corrected': None, 'velocity_frame.correction': 'auto',
                       'calibration.wavelength_correction': 'none', 'templates.teff_range': None}),
    ('jwst_ifu', {'input.telluric_corrected': True, 'velocity_frame.correction': 'skip',
                  'velocity_frame.frame': 'barycentric', 'calibration.wavelength_correction': 'none',
                  'calibration.zero_point_known': True}),
    ('jwst_mos', {'input.telluric_corrected': True, 'velocity_frame.correction': 'skip',
                  'calibration.zero_point_known': None, 'input.medium': 'vacuum'}),
])
def test_setting_advice_for_the_examples(name, expected):
    be, cfg, spec = _example(name)
    advice = be.setting_advice(spec, cfg)
    assert {k: _values(advice)[k] for k in expected} == expected
    assert all(a.reason for a in advice.values())
    if name == 'gtc_osiris':  # the velocity frame is unknown: the reader's note says what to do
        assert 'heliocentric' in advice['velocity_frame.correction'].reason
        assert advice['templates.grid'].reason.startswith('From the Teff prior (2100 K)')
    if name == 'jwst_mos':
        assert 'below 500' in advice['input.resolution'].reason
    if name == 'near_infrared':
        assert advice['templates.grid'].reason.startswith('No Teff known')


def test_every_setting_has_advice_or_a_default_reason():
    """The data-dependent settings (always the same paths) and DEFAULT_WHY together cover every field once"""
    from rvfitter.gui.forms import DEFAULT_WHY, TIPS
    _, cfg, spec = _example('jwst_ifu')
    advised = set(config.setting_advice(spec, cfg))
    assert advised == set(config.setting_advice(spec, config.Config(), fit_teff=1500.))
    assert not advised & set(DEFAULT_WHY) and advised | set(DEFAULT_WHY) == set(TIPS)


@pytest.mark.parametrize('teff, met, grids', [
    (5800., None, ['bt-settl-agss2009', 'bt-settl-cifist']),
    (3200., None, ['bt-settl-cifist', 'bt-settl-agss2009', 'sand']),
    (1500., None, ['bt-settl-cifist', 'sonora-diamondback', 'sonora-elfowl']),
    (1000., None, ['sonora-elfowl', 'sonora-bobcat', 'sonora-diamondback']),
    (300., None, ['sonora-elfowl', 'sonora-bobcat']),
    (3000., -1., ['sand', 'bt-settl-agss2009', 'bt-settl-cifist']),
    (1000., -1., ['sand', 'lowz', 'bt-settl-agss2009', 'sonora-elfowl', 'sonora-bobcat', 'sonora-diamondback']),
    (100., None, []),
])
def test_grid_advice_by_teff(teff, met, grids):
    from rvfitter.templates import KNOWN_GRIDS
    ranked = config.grid_advice(teff, met)
    assert [g for g, _ in ranked] == grids
    assert all(KNOWN_GRIDS[g].coverage['teff'][0] <= teff <= KNOWN_GRIDS[g].coverage['teff'][1] for g in grids)


def test_template_advice_follows_the_teff_source():
    _, cfg, spec = _example('jwst_ifu')
    fit = config.setting_advice(spec, cfg, fit_teff=1000.)
    assert fit['templates.grid'].value == 'sonora-elfowl' and 'last run' in fit['templates.grid'].reason
    cfg.templates.teff_prior = 5800.  # the prior wins over the fit; FGK: no catalogue features at this Teff
    hot = config.setting_advice(spec, cfg, fit_teff=1000.)
    assert hot['templates.grid'].value == 'bt-settl-agss2009' and hot['regions.template_windows'].value == 'full'
    assert hot['templates.logg_range'].value is None
    cfg.templates.teff_prior = 100.  # no grid covers it: said plainly, nothing to apply
    cold = config.setting_advice(spec, cfg)
    assert cold['templates.grid'] == cold['templates.teff_range'] and cold['templates.grid'].value is None
    assert 'No known grid covers 100 K' in cold['templates.grid'].reason


@pytest.mark.parametrize('name', ['gtc_osiris', 'near_infrared', 'jwst_ifu', 'jwst_mos'])
def test_default_for_agrees_with_setting_advice(name):
    """default_for takes its methods from setting_advice; every other value both set is the same"""
    _, _, spec = _example(name)
    defaults = config.default_for(spec)
    for path, (value, _) in config.setting_advice(spec, config.Config()).items():
        section, *rest = path.split('.')
        default = getattr(getattr(defaults, section), rest[0])
        if value is not None and default != getattr(getattr(config.Config(), section), rest[0]):
            assert default == value, path
    assert defaults.methods.enabled == config.setting_advice(spec, config.Config())['methods.enabled'].value


def test_location_fields_name_real_fields():
    for (section, name), kind in config.LOCATION_FIELDS.items():
        assert name in {f.name for f in dataclasses.fields(getattr(config.Config(), section))}
        assert kind in ('files', 'file', 'folder', 'save')
