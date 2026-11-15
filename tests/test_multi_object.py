"""Multi-object files (PypeIt, JWST x1d, multi-extension FITS), IFU cubes and the selection of their spectra"""
import logging
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits

from rvfitter import archives, barycentric, config, io
from rvfitter.calibration import wavelength_correction as wcorr
from test_io import WAVE, _multi_spec1d

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


# ----------------------------------------------------------------------------------------------------------------------
# PypeIt multi-object spec1d
# ----------------------------------------------------------------------------------------------------------------------

def test_pypeit_entries_and_selection(tmp_path, caplog):
    """Long-slit traces (no MASKDEF_OBJNAME): 'auto' is the target; 'all', lists and low S/N as stated"""
    path = _multi_spec1d(tmp_path / 'm.fits', [17., 86., 2.], ['SPAT0168', 'SPAT0240', 'SPAT0300'], ['science'] * 3)
    entries = io.list_spectra(path)
    assert [(e.hdu, e.name, e.kind, e.named) for e in entries] == [(1, 'SPAT0168', 'object', False),
                                                                    (2, 'SPAT0240', 'object', False),
                                                                    (3, 'SPAT0300', 'object', False)]
    assert np.allclose([e.snr for e in entries], [17., 86., 2.])
    assert [e.usable for e in entries] == [True, True, False] and 'below 3' in entries[2].reason
    caplog.set_level(logging.INFO, logger='rvfitter')
    (spec,) = io.read_spectra(path)  # auto -> target: the highest S/N, as before
    assert spec.source == f'{path}[2]' and spec.meta['object'] == 'My Star' and spec.meta['entry'] == 'SPAT0240'
    assert "'auto': the spectra are not named after their sources" in caplog.text
    both = io.read_spectra(path, objects='all')
    assert [s.meta['object'] for s in both] == ['My Star SPAT0168', 'My Star SPAT0240']  # never merged by grouping
    assert 'skipping HDU 3 (SPAT0300): median S/N 2.0' in caplog.text
    assert sorted(archives.group_by_object(both)) == ['My Star SPAT0168', 'My Star SPAT0240']
    picked = io.read_spectra(path, objects=['SPAT0300', 1])
    assert [s.meta['hdu'] for s in picked] == [1, 3] and 'is read because input.objects names it' in caplog.text
    with pytest.raises(ValueError, match='selects none'):
        io.read_spectra(path, objects=['nothing'])
    assert io.read_spectra(f'{path}[1]')[0].source == f'{path}[1]'  # a refit reads its own source back
    assert io.read_spectrum(f'{path}[3]').meta['hdu'] == 3


def test_pypeit_mask_objects_are_named_and_all_read(tmp_path):
    path = _multi_spec1d(tmp_path / 'm.fits', [50., 40.], ['SPAT0100', 'SPAT0200'], ['science'] * 2,
                         target='mask1', maskdef=['star A', 'star B'])
    assert all(e.named for e in io.list_spectra(path))
    assert [s.meta['object'] for s in io.read_spectra(path)] == ['star A', 'star B']
    named = _multi_spec1d(tmp_path / 'n.fits', [50., 40.], ['SPAT0100', 'SPAT0200'], ['science'] * 2,
                          target='J1745-1640', maskdef=['other star', 'J1745-1640'])
    assert [s.meta['hdu'] for s in io.read_spectra(named)] == [2]  # auto: the entry named after the target


# ----------------------------------------------------------------------------------------------------------------------
# JWST x1d and generic multi-extension FITS
# ----------------------------------------------------------------------------------------------------------------------

def _x1d(path: Path, names: list[str], snr: float = 50.) -> Path:
    """A JWST x1d with one EXTRACT1D per source: WAVELENGTH [µm], FLUX [Jy], DQ (pixel 5 DO_NOT_USE)"""
    prim = fits.PrimaryHDU()
    prim.header.update(TELESCOP='JWST', INSTRUME='NIRSPEC', GRATING='G395H', TARGNAME='', EXP_TYPE='NRS_MSASPEC')
    hdus = [prim]
    um = WAVE / 1e4 * 4.  # 2.8-3.6 µm
    for k, name in enumerate(names):
        dq = np.zeros(len(um), dtype=np.int32)
        dq[5] = 1
        cols = [fits.Column('WAVELENGTH', 'D', 'um', array=um), fits.Column('FLUX', 'D', 'Jy', array=np.ones_like(um)),
                fits.Column('FLUX_ERROR', 'D', 'Jy', array=np.full_like(um, 1. / snr)),
                fits.Column('DQ', 'J', array=dq)]
        tab = fits.BinTableHDU.from_columns(cols, name='EXTRACT1D', ver=k + 1)
        tab.header.update(SRCNAME=name, SLTNAME=str(k), SRCTYPE='POINT')
        hdus.append(tab)
    fits.HDUList(hdus).writeto(path)
    return path


def test_jwst_x1d_sources_units_and_frame(tmp_path, caplog):
    path = _x1d(tmp_path / 'x1d.fits', ['1229_9', 'background_7', '1229_1'])
    entries = io.list_spectra(path)
    assert [(e.name, e.kind, e.usable) for e in entries] == [('1229_9', 'object', True),
                                                              ('background_7', 'background', False),
                                                              ('1229_1', 'object', True)]
    caplog.set_level(logging.INFO, logger='rvfitter')
    spectra = io.read_spectra(path)  # auto: the sources carry their own names, so all usable ones
    assert [s.meta['object'] for s in spectra] == ['1229_9', '1229_1']
    assert [s.source for s in spectra] == [f'{path}[1]', f'{path}[3]']
    spec = spectra[0]
    assert len(spec.wave) == len(WAVE) - 1  # the DO_NOT_USE pixel is dropped
    lam = spec.wave * u.AA
    expected = (1. * u.Jy).to(u.erg / u.s / u.cm ** 2 / u.AA, equivalencies=u.spectral_density(lam)).value
    assert np.allclose(spec.flux, expected) and np.allclose(spec.flux / spec.sigma, 50.)
    assert 'Jy (per unit frequency) to fλ' in caplog.text and spec.meta['flux_conversion']
    assert spec.meta['medium_in'] == 'vacuum' and spec.meta['medium_source'] == 'default:jwst'
    assert spec.meta['frame_in'] == 'barycentric' and spec.meta['frame_source'] == io.JWST_FRAME_SOURCE
    assert barycentric.correction(spec, mode='auto') == (0., 'header')  # already corrected by the JWST pipeline


def test_specsys_marks_the_frame():
    meta = io._meta_from_header({'SPECSYS': 'BARYCENT', 'VELOSYS': 29037.15})
    assert meta['frame_in'] == 'barycentric' and meta['frame_source'] == 'header:SPECSYS'
    assert io._meta_from_header({'SPECSYS': 'TOPOCENT'})['frame_in'] is None


def test_surface_brightness_needs_pixar():
    wave, one = np.array([3e4, 4e4]), np.ones(2)
    with pytest.raises(ValueError, match='PIXAR_SR'):
        io._to_flambda(wave, one, one, 'MJy/sr', {}, 'x')
    flux, _, note = io._to_flambda(wave, one, one, 'MJy/sr', {'PIXAR_SR': 2e-13}, 'x')
    assert np.allclose(flux, io._to_flambda(wave, one * 2e-13 * 1e6, one, 'Jy', {}, 'x')[0]) and note
    assert io._to_flambda(wave, one, one, 'erg/s/cm2/Angstrom', {}, 'x')[2] == {}  # fλ: unchanged


def test_auxiliary_extensions_are_never_spectra(tmp_path):
    flux = np.ones(200)
    sci = fits.ImageHDU(flux, name='SCI')
    sci.header.update(CRVAL1=7000., CDELT1=1., CRPIX1=1., CTYPE1='WAVE')
    hdus = [fits.PrimaryHDU(), sci, fits.ImageHDU(flux * 0.1, name='ERR'), fits.ImageHDU(np.zeros(200), name='DQ'),
            fits.ImageHDU(flux, name='SKY')]
    for h in hdus[2:]:
        h.header.update(CRVAL1=7000., CDELT1=1., CRPIX1=1., CTYPE1='WAVE')
    path = tmp_path / 'mef.fits'
    fits.HDUList(hdus).writeto(path)
    entries = io.list_spectra(path)
    assert [(e.name, e.kind, e.usable) for e in entries] == [('SCI', 'object', True), ('ERR', 'error', False),
                                                              ('DQ', 'dq', False), ('SKY', 'sky', False)]
    assert 'error plane, not a spectrum' in entries[1].reason
    (spec,) = io.read_spectra(path)  # one candidate: read as a single spectrum, with its errors
    assert spec.source == str(path) and np.allclose(spec.sigma, 0.1)
    with pytest.raises(ValueError, match='sky extension, not a spectrum'):
        io.read_spectrum(path, hdu='SKY')


def test_source_notation_through_archives(tmp_path):
    path = _x1d(tmp_path / 'x1d.fits', ['a', 'b'])
    assert io.split_source(f'{path}[2]') == (path, 2) and io.split_source(path) == (path, None)
    assert archives.collect_inputs(f'{path}[2]') == [Path(f'{path}[2]')]
    (spec,) = archives.load_spectra(archives.collect_inputs(f'{path}[2]'))
    assert spec.meta['object'] == 'b'
    assert list(archives.group_by_object([spec], {'x1d.fits': 'renamed'})) == ['renamed']  # file-name override


# ----------------------------------------------------------------------------------------------------------------------
# IFU cubes
# ----------------------------------------------------------------------------------------------------------------------

def _cube(path: Path, x0: float = 12., y0: float = 7., pixar: float | None = 1e-13) -> tuple[Path, np.ndarray]:
    """A 20 × 25 spaxel cube in MJy/sr: a Gaussian point source (σ 1.2 spaxels) whose spectrum has absorption lines,
    an ERR plane, and a DQ plane flagging one aperture spaxel in plane 10 and the whole edge column x = 0"""
    ny, nx = 20, 25
    rng = np.random.default_rng(3)
    spec = 1. - 0.5 * np.exp(-0.5 * ((WAVE[:, None] - rng.uniform(7100., 8900., 25)) / 2.) ** 2).sum(axis=1)
    yy, xx = np.mgrid[:ny, :nx]
    psf = np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / 1.2 ** 2)
    sci = spec[:, None, None] * psf[None] + 1e-3
    dq = np.zeros(sci.shape, dtype=np.int32)
    dq[10, int(y0), int(x0) + 1] = 1
    dq[:, :, 0] = 513
    prim = fits.PrimaryHDU()
    prim.header.update(TELESCOP='JWST', INSTRUME='NIRSPEC', GRATING='G395H', TARGNAME='VHS 1256-1257 b')
    h = fits.ImageHDU(sci.astype(np.float32), name='SCI')
    h.header.update(CTYPE1='RA---TAN', CTYPE2='DEC--TAN', CTYPE3='WAVE', CUNIT3='Angstrom', CRVAL3=WAVE[0],
                    CDELT3=WAVE[1] - WAVE[0], CRPIX3=1., BUNIT='MJy/sr', SPECSYS='BARYCENT')
    if pixar is not None:
        h.header['PIXAR_SR'] = pixar
    fits.HDUList([prim, h, fits.ImageHDU(np.full(sci.shape, 0.01, np.float32), name='ERR'),
                  fits.ImageHDU(dq, name='DQ')]).writeto(path)
    return path, spec


def test_cube_extraction(tmp_path, caplog):
    path, spec = _cube(tmp_path / 's3d.fits')
    image = io.cube_image(path)
    assert io.brightest_spaxel(image) == (12, 7) and np.isnan(image[:, 0]).all()
    entries = io.list_spectra(path)
    assert [(e.name, e.kind) for e in entries] == [('SCI', 'cube'), ('ERR', 'error'), ('DQ', 'dq')]
    caplog.set_level(logging.INFO, logger='rvfitter')
    (out,) = io.read_spectra(path)
    ext = out.meta['extraction']
    assert (ext['x'], ext['y'], ext['radius'], ext['radius_source']) == (12., 7., io.CUBE_RADIUS_DEFAULT, 'default')
    assert ext['n_spaxels'] == 29 and ext['planes_dropped'] == 1 and len(out.wave) == len(WAVE) - 1
    assert out.meta['no_slit'] and out.meta['frame_in'] == 'barycentric' and 'PIXAR_SR' in out.meta['flux_conversion']
    keep = np.delete(np.arange(len(WAVE)), 10)
    yy, xx = np.mgrid[:20, :25]
    inside = (xx - 12) ** 2 + (yy - 7) ** 2 <= 9
    summed = spec[keep] * np.exp(-0.5 * ((xx - 12) ** 2 + (yy - 7) ** 2) / 1.2 ** 2)[inside].sum() + 29e-3
    ratio = out.flux * out.wave ** 2 / summed  # fλ ∝ fν / λ²
    assert np.ptp(ratio) / np.median(ratio) < 1e-5  # the line spectrum is preserved, so is the RV
    assert np.allclose(out.sigma * out.wave ** 2 / np.median(ratio), 0.01 * np.sqrt(29), rtol=1e-5)  # quadrature
    moved = io.read_spectrum(path, cube=config.CubeConfig(position=(5., 5.), radius=1.))
    assert moved.meta['extraction']['n_spaxels'] == 5 and moved.meta['extraction']['centre_source'] == \
        'input.cube.position'
    with pytest.raises(ValueError, match='outside'):
        io.read_spectrum(path, cube=config.CubeConfig(position=(30., 5.)))
    with pytest.raises(ValueError, match='PIXAR_SR'):
        io.read_spectrum(_cube(tmp_path / 'nopix.fits', pixar=None)[0])


def test_cube_has_no_slit_term():
    path_cfg = config.Config()
    spec = io.Spectrum(WAVE, np.ones_like(WAVE), np.ones_like(WAVE), 2500., {'no_slit': 'IFU cube extraction'})
    term = wcorr.star_sky_term({'slit': None}, spec, None, path_cfg, 'cube')
    assert term['value'] == 0. and term['term'] == 'none: IFU cube extraction (no slit)' and not term['flags']


# ----------------------------------------------------------------------------------------------------------------------
# configuration, resolution
# ----------------------------------------------------------------------------------------------------------------------

def test_objects_and_cube_config():
    cfg = config.loads('[input]\nobjects = ["1229_9", 3]\n[input.cube]\nposition = [8.0, 8.0]\nradius = 2.5\n')
    assert cfg.input.objects == ['1229_9', 3] and cfg.input.cube.position == (8., 8.) and cfg.input.cube.radius == 2.5
    assert config.loads(config.dumps(cfg)) == cfg
    for bad, why in (('objects = "every"', 'input.objects'), ('objects = []', 'selects nothing'),
                     ('hdu = 2\nobjects = "all"', 'set only one')):
        with pytest.raises(config.ConfigError, match=why):
            config.loads(f'[input]\n{bad}\n')
    with pytest.raises(config.ConfigError, match='radius'):
        config.loads('[input.cube]\nradius = 0.0\n')


def test_advice_when_r_is_too_low():
    assert config.resolution_too_low(None) is None and config.resolution_too_low(config.R_MIN_RV) is None
    why = config.resolution_too_low(100.)
    assert 'below 500' in why and '2998 km/s' in why
    low = config.method_advice('low', 100.)
    assert all(a.reason.startswith(why) for a in low.values())
    for regime in ('low', 'mid', 'high'):  # the recommendations, and so default_for, do not change
        plain = config.method_advice(regime)
        assert {m: (a.recommended, a.options) for m, a in config.method_advice(regime, 100.).items()} == \
            {m: (a.recommended, a.options) for m, a in plain.items()}
        assert config.method_advice(regime, 3000.) == plain


def test_jwst_resolution_rows():
    prism = {'instrument_names': ('NIRSPEC', 'JWST_NIRSPEC_PRISM')}
    assert io.implausible_resolution(118., prism) is None
    assert 'Jakobsen' in io.implausible_resolution(2000., prism)
    assert io._slit_meta({'TELESCOP': 'JWST', 'INSTRUME': 'NIRSPEC', 'GRATING': 'G395H'})['instrument_names'] == \
        ('NIRSPEC', 'JWST_NIRSPEC_G395H')


# ----------------------------------------------------------------------------------------------------------------------
# the committed real examples (examples/jwst_*)
# ----------------------------------------------------------------------------------------------------------------------

def test_example_jwst_mos():
    a, b = sorted((EXAMPLES / 'jwst_mos').glob('*_x1d.fits'))
    assert [e.name for e in io.list_spectra(a)] == ['1229_9', '1229_1', '1229_2', '1229_3']
    entries = io.list_spectra(b)
    assert [(e.name, e.usable) for e in entries] == [('1229_13', False), ('1229_6', True), ('1229_7', True)]
    spectra = io.read_spectra(a)
    assert [s.meta['object'] for s in spectra] == ['1229_9', '1229_1', '1229_2', '1229_3']
    assert all(s.resolution < config.R_MIN_RV and not s.meta['resolution_implausible'] for s in spectra)


def test_example_jwst_ifu():
    (path,) = (EXAMPLES / 'jwst_ifu').glob('*_s3d_cutout.fits.gz')
    (spec,) = io.read_spectra(path)
    ext = spec.meta['extraction']
    assert (ext['x'], ext['y'], ext['n_spaxels']) == (8., 8., 29)
    assert 1320. < spec.resolution < 3600. and not spec.meta['resolution_implausible']
    assert spec.meta['frame_in'] == 'barycentric' and spec.meta['frame_source'] == 'header:SPECSYS'
    assert np.nanmedian(spec.flux / spec.sigma) > 100.
