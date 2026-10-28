"""Tests for rvfitter.io: synthetic files in every supported format, medium handling and the local test data."""
import logging
import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from rvfitter import io
from rvfitter.medium import air_to_vac

WAVE = np.linspace(7000., 9000., 400)
FLUX = 1. + 0.1 * np.sin(WAVE / 50.)
IVAR = np.full_like(WAVE, 400.)
TESTDATA = os.environ.get('RVFITTER_TESTDATA')


def _spec1d(path: Path, header: dict | None = None, ext_header: dict | None = None, vel: bool = False) -> Path:
    cols = [fits.Column(name=n, format='D', array=a) for n, a in (
        ('OPT_WAVE', WAVE), ('OPT_FLAM', FLUX), ('OPT_FLAM_IVAR', IVAR), ('BOX_WAVE', WAVE), ('BOX_FLAM', FLUX * 2),
        ('BOX_FLAM_IVAR', IVAR))]
    tab = fits.BinTableHDU.from_columns(cols)
    prim = fits.PrimaryHDU()
    prim.header.update({'RA': 266.43, 'DEC': -16.64, 'MJD': 57238.0, 'AIRMASS': 1.5, 'TELESCOP': 'GTC',
                        'EXPTIME': 750., 'TARGET': 'My  Star'})
    prim.header.update(header or {})
    if vel:
        tab.header['VEL_TYPE'] = 'barycentric'
        tab.header['VEL_CORR'] = 0.99993
    tab.header.update(ext_header or {})
    fits.HDUList([prim, tab]).writeto(path)
    return path


def test_pypeit_spec1d(tmp_path):
    spec = io.read_spectrum(_spec1d(tmp_path / 's.fits', vel=True))
    np.testing.assert_allclose(spec.wave, WAVE)
    np.testing.assert_allclose(spec.flux, FLUX)
    np.testing.assert_allclose(spec.sigma, 1. / 20.)
    m = spec.meta
    assert m['object'] == 'My  Star' and m['site'] == 'GTC' and m['airmass'] == 1.5
    assert m['ra'] == pytest.approx(266.43) and m['dec'] == pytest.approx(-16.64)
    assert m['mjd_mid'] == pytest.approx(57238.0 + 375. / 86400.)
    assert m['frame_in'] == 'barycentric' and m['vel_corr'] == pytest.approx(0.99993)
    f2 = 0.99993 ** 2
    assert m['vel_corr_kms'] == pytest.approx(299792.458 * (f2 - 1) / (f2 + 1))
    assert m['medium_in'] == 'vacuum' and m['medium_source'] == 'default:pypeit_spec1d'
    assert spec.resolution is None


def test_pypeit_hdu_selection_and_box(tmp_path):
    path = _spec1d(tmp_path / 's.fits')
    with fits.open(path, mode='update') as h:
        h[1].data['OPT_WAVE'][:] = 0.  # unusable optimal extraction -> BOX
    spec = io.read_spectrum(path)
    np.testing.assert_allclose(spec.flux, FLUX * 2)
    assert io.read_spectrum(path, hdu=1).flux.shape == spec.flux.shape
    with pytest.raises(ValueError):
        io.read_spectrum(path, hdu=7)


def test_sigma_nan_where_ivar_nonpositive(tmp_path):
    path = _spec1d(tmp_path / 's.fits')
    with fits.open(path, mode='update') as h:
        h[1].data['OPT_FLAM_IVAR'][[3, 4]] = [0., -1.]
    spec = io.read_spectrum(path)
    assert np.isnan(spec.sigma[[3, 4]]).all() and np.isfinite(spec.sigma[5:]).all()


def test_pypeit_header_medium_overrides_default(tmp_path):
    path = _spec1d(tmp_path / 's.fits', header={'AIRORVAC': 'air'})
    spec = io.read_spectrum(path)
    assert spec.meta['medium_in'] == 'air'
    np.testing.assert_allclose(spec.wave, air_to_vac(WAVE), rtol=1e-12)


def test_explicit_medium_conflict_warns(tmp_path, caplog):
    path = _spec1d(tmp_path / 's.fits', header={'AIRORVAC': 'vac'})
    with caplog.at_level(logging.WARNING, logger='rvfitter.io'):
        spec = io.read_spectrum(path, medium='air')
    assert 'header' in caplog.text and spec.meta['medium_source'] == 'user'
    assert spec.wave[0] > WAVE[0]


def test_coadd_table(tmp_path):
    mask = np.ones(len(WAVE), dtype=bool)
    mask[:10] = False
    cols = [fits.Column(name=n, format='D', array=a) for n, a in (('wave', WAVE), ('flux', FLUX), ('ivar', IVAR))]
    cols.append(fits.Column(name='mask', format='L', array=mask))
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(tmp_path / 'c.fits')
    spec = io.read_spectrum(tmp_path / 'c.fits')
    assert len(spec.wave) == len(WAVE) - 10
    assert spec.meta['medium_in'] == 'vacuum' and spec.meta['reader'] == 'pypeit_coadd'


def _image(path, header, data=FLUX):
    hdu = fits.PrimaryHDU(np.asarray(data, dtype='f8'))
    hdu.header.update(header)
    hdu.writeto(path)
    return path


def test_wcs_linear_needs_medium(tmp_path):
    path = _image(tmp_path / 'i.fits', {'CRVAL1': 7000., 'CDELT1': 5., 'CRPIX1': 1., 'OBJECT': 'X'})
    with pytest.raises(ValueError, match="medium='air'"):
        io.read_spectrum(path)
    spec = io.read_spectrum(path, medium='vacuum')
    np.testing.assert_allclose(spec.wave, 7000. + 5. * np.arange(len(FLUX)))
    assert np.isnan(spec.sigma).all() and spec.meta['object'] == 'X'
    air = io.read_spectrum(path, medium='air')
    assert air.wave[0] > spec.wave[0]


def test_wcs_cd11_crpix_and_ctype_medium(tmp_path):
    path = _image(tmp_path / 'i.fits', {'CRVAL1': 7000., 'CD1_1': 5., 'CRPIX1': 11., 'CTYPE1': 'AWAV'})
    spec = io.read_spectrum(path)  # AWAV -> air, found automatically
    assert spec.meta['medium_in'] == 'air' and spec.meta['medium_source'] == 'header:CTYPE1'
    np.testing.assert_allclose(spec.wave, air_to_vac(7000. + 5. * (np.arange(len(FLUX)) - 10.)))
    path2 = _image(tmp_path / 'j.fits', {'CRVAL1': 7000., 'CDELT1': 5., 'CTYPE1': 'WAVE'})
    assert io.read_spectrum(path2).meta['medium_in'] == 'vacuum'


def test_wcs_log_dcflag_and_ctype_log(tmp_path):
    n = len(FLUX)
    path = _image(tmp_path / 'i.fits', {'CRVAL1': np.log10(7000.), 'CDELT1': 1e-5, 'DC-FLAG': 1, 'AIRORVAC': 'vac'})
    spec = io.read_spectrum(path)
    np.testing.assert_allclose(spec.wave, 10 ** (np.log10(7000.) + 1e-5 * np.arange(n)))
    path2 = _image(tmp_path / 'j.fits', {'CRVAL1': 7000., 'CDELT1': 0.5, 'CTYPE1': 'WAVE-LOG'})
    spec2 = io.read_spectrum(path2)
    np.testing.assert_allclose(spec2.wave, 7000. * np.exp(0.5 * np.arange(n) / 7000.))
    assert np.all(np.diff(np.log(spec2.wave)) > 0)


@pytest.mark.parametrize('nrow', [2, 3])
@pytest.mark.parametrize('transpose', [False, True])
def test_spextool_array_microns(tmp_path, nrow, transpose):
    rows = [WAVE / 1e4, FLUX, IVAR * 0 + 0.05][:nrow]
    data = np.array(rows)
    path = _image(tmp_path / 'sp.fits', {'XUNITS': 'um', 'YUNITS': 'erg', 'AIRORVAC': 'vac'},
                  data.T if transpose else data)
    spec = io.read_spectrum(path)
    np.testing.assert_allclose(spec.wave, WAVE)
    if nrow == 3:
        np.testing.assert_allclose(spec.sigma, 0.05)
    else:
        assert np.isnan(spec.sigma).all()


def test_binary_table_case_insensitive(tmp_path):
    cols = [fits.Column(name='Wavelength', format='D', unit='nm', array=WAVE / 10.),
            fits.Column(name='FLAM', format='D', array=FLUX),
            fits.Column(name='Sigma', format='D', array=IVAR * 0 + 0.1)]
    hdr = fits.Header({'WAVE_MED': 'VACUUM'})
    tab = fits.BinTableHDU.from_columns(cols, header=hdr)
    fits.HDUList([fits.PrimaryHDU(), tab]).writeto(tmp_path / 't.fits')
    spec = io.read_spectrum(tmp_path / 't.fits')
    np.testing.assert_allclose(spec.wave, WAVE)
    np.testing.assert_allclose(spec.sigma, 0.1)
    assert spec.meta['reader'] == 'fits_generic'


def test_table_ivar_and_columns_override(tmp_path):
    cols = [fits.Column(name='lam_obs', format='D', array=WAVE), fits.Column(name='f', format='D', array=FLUX),
            fits.Column(name='iv', format='D', array=IVAR)]
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]).writeto(tmp_path / 't.fits')
    with pytest.raises(ValueError):
        io.read_spectrum(tmp_path / 't.fits', medium='vacuum')
    cmap = {'wave': 'lam_obs', 'flux': 'f', 'ivar': 'iv'}
    spec = io.read_spectrum(tmp_path / 't.fits', medium='vacuum', columns=cmap)
    np.testing.assert_allclose(spec.sigma, 0.05)


def test_cleaning(tmp_path):
    wave = np.array([5., 3., 4., 4., np.nan, 6., 7.]) * 1e3
    flux = np.array([1., 2., 3., 9., 4., np.nan, 6.])
    sig = np.array([.1, .2, .3, .9, .4, .5, .6])
    path = tmp_path / 'm.txt'
    np.savetxt(path, np.c_[wave, flux, sig], header='AIRORVAC = vac')
    spec = io.read_spectrum(path)
    np.testing.assert_array_equal(spec.wave, [3000., 4000., 5000., 7000.])
    np.testing.assert_array_equal(spec.flux, [2., 3., 1., 6.])  # first duplicate kept
    np.testing.assert_allclose(spec.sigma, [.2, .3, .1, .6])


def test_ascii_variants(tmp_path):
    # comma-separated with header row and micron unit in column name
    body = '\n'.join(f'{w/1e4},{f},0.1' for w, f in zip(WAVE, FLUX, strict=True))
    (tmp_path / 'a.csv').write_text('wave_um,flux,err\n' + body)
    with pytest.raises(ValueError, match='air or vacuum'):
        io.read_spectrum(tmp_path / 'a.csv')
    spec = io.read_spectrum(tmp_path / 'a.csv', medium='vacuum')
    np.testing.assert_allclose(spec.wave, WAVE)
    np.testing.assert_allclose(spec.sigma, 0.1)
    # tab separated, two columns, comments with key = value and medium hint
    lines = ['# OBJECT = TestObj', '# wavelength in Angstrom', '# medium: air', '# MJD = 58000.5', '# EXPTIME = 100',
             '# RESOLUTION = 3000'] + [f'{w}\t{f}' for w, f in zip(WAVE, FLUX, strict=True)]
    (tmp_path / 'b.tsv').write_text('\n'.join(lines))
    spec = io.read_spectrum(tmp_path / 'b.tsv')
    assert spec.meta['medium_in'] == 'air' and spec.meta['object'] == 'TestObj'
    assert spec.resolution == 3000.
    assert spec.meta['mjd_mid'] == pytest.approx(58000.5 + 50. / 86400.)
    assert np.isnan(spec.sigma).all()
    np.testing.assert_allclose(spec.wave, air_to_vac(WAVE), rtol=1e-12)
    # a comment word alone is not enough to infer the medium
    body = '\n'.join(f'{w} {f}' for w, f in zip(WAVE, FLUX, strict=True))
    (tmp_path / 'd.txt').write_text('# spectrum in vacuum\n' + body)
    with pytest.raises(ValueError, match='air or vacuum'):
        io.read_spectrum(tmp_path / 'd.txt')
    # ivar column by name, nm units
    (tmp_path / 'c.dat').write_text('# wave[nm] flux ivar\n# medium = vacuum\n' +
                                    '\n'.join(f'{w/10} {f} 100' for w, f in zip(WAVE, FLUX, strict=True)))
    spec = io.read_spectrum(tmp_path / 'c.dat')
    np.testing.assert_allclose(spec.wave, WAVE)
    np.testing.assert_allclose(spec.sigma, 0.1)


def test_resolution_header_and_override(tmp_path):
    path = _spec1d(tmp_path / 's.fits', header={'SPECRES': 2500.})
    assert io.read_spectrum(path).resolution == 2500.
    assert io.read_spectrum(path, resolution=4000.).resolution == 4000.
    with pytest.raises(ValueError):
        io.read_spectrum(path, resolution=-1.)


def test_mjd_mid_keywords(tmp_path):
    path = _spec1d(tmp_path / 's.fits', header={'MJD_MID': 55555.5})
    assert io.read_spectrum(path).meta['mjd_mid'] == 55555.5
    path = _spec1d(tmp_path / 't.fits', header={'MJD': None, 'DATE-OBS': '2015-08-04T00:32:33', 'EXPTIME': 0.})
    meta = io.read_spectrum(path).meta
    assert meta['mjd_mid'] == pytest.approx(57238.0226042, abs=1e-6) and 'DATE-OBS' in meta['mjd_source']


def test_sexagesimal_coordinates_and_airmass(tmp_path):
    path = _spec1d(tmp_path / 's.fits', header={'RA': '17:45:43.08', 'DEC': '-16:38:09.3', 'AIRMASS': None,
                                                 'AMSTART': 1.2, 'AMEND': 1.4, 'OBSERVAT': 'LCO'})
    m = io.read_spectrum(path).meta
    assert m['ra'] == pytest.approx(266.4295, abs=1e-4) and m['dec'] == pytest.approx(-16.6359, abs=1e-4)
    assert m['airmass'] == pytest.approx(1.3) and m['site'] == 'LCO' and m['telescope'] == 'GTC'


def test_bad_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.read_spectrum(tmp_path / 'nope.fits')
    (tmp_path / 'junk.txt').write_text('hello world\nthis is not a spectrum\n')
    with pytest.raises(ValueError, match='Could not read'):
        io.read_spectrum(tmp_path / 'junk.txt', medium='vacuum')
    with pytest.raises(ValueError, match="medium must be"):
        io.read_spectrum(tmp_path / 'junk.txt', medium='mud')


def test_register_reader_priority(tmp_path):
    (tmp_path / 'x.special').write_text('0 0\n')

    def sniff(path, hdu=None, columns=None):
        return path.suffix == '.special'

    def read(path, hdu=None, columns=None):
        return io.RawSpectrum(WAVE, FLUX, np.full_like(WAVE, np.nan), {'OBJECT': 'special'}, 'vacuum', 'test',
                              reader='special')
    io.register_reader('special', sniff, read, first=True)
    try:
        spec = io.read_spectrum(tmp_path / 'x.special')
        assert spec.meta['object'] == 'special' and io.readers()[0] == 'special'
    finally:
        io._READERS[:] = [r for r in io._READERS if r.name != 'special']


def test_specutils_fallback(tmp_path):
    """A FITS file the built-in readers do not recognise is handed to specutils (which may also refuse it)"""
    hdu = fits.PrimaryHDU(np.zeros((5, 5, 5)))
    hdu.writeto(tmp_path / 'cube.fits')
    with pytest.raises(ValueError, match='specutils'):
        io.read_spectrum(tmp_path / 'cube.fits', medium='vacuum')


def _multi_spec1d(path: Path, snrs: list[float], names: list[str], objtypes: list[str], target: str = 'My Star',
                  maskdef: list[str] | None = None) -> Path:
    """A PypeIt spec1d with one object table per S/N (flux 1, ivar = S/N²)"""
    prim = fits.PrimaryHDU()
    prim.header.update({'TARGET': target, 'TELESCOP': 'GTC'})
    hdus = [prim]
    for k, (snr, name, ot) in enumerate(zip(snrs, names, objtypes, strict=True)):
        cols = [fits.Column(name=n, format='D', array=a) for n, a in (
            ('OPT_WAVE', WAVE), ('OPT_FLAM', np.ones_like(WAVE) * (k + 1)),
            ('OPT_FLAM_IVAR', np.full_like(WAVE, (snr / (k + 1)) ** 2)))]
        tab = fits.BinTableHDU.from_columns(cols, name=name)
        tab.header.update({'NAME': name, 'OBJTYPE': ot})
        if maskdef:
            tab.header['MASKDEF_OBJNAME'] = maskdef[k]
        hdus.append(tab)
    fits.HDUList(hdus).writeto(path)
    return path


def test_pypeit_multi_object_picks_highest_snr(tmp_path, caplog):
    path = _multi_spec1d(tmp_path / 'm.fits', [17., 86.], ['SPAT0168-SLIT0457-DET02', 'SPAT0240-SLIT0457-DET02'],
                         ['science', 'science'])
    caplog.set_level(logging.INFO, logger='rvfitter')
    spec = io.read_spectrum(path)
    assert spec.meta['hdu'] == 2 and spec.meta['extname'] == 'SPAT0240-SLIT0457-DET02' and spec.meta['n_objects'] == 2
    assert 'highest-S/N science object' in caplog.text and 'HDU 1 SPAT0168' in caplog.text
    assert io.read_spectrum(path, hdu=1).meta['hdu'] == 1  # input.hdu overrides


def test_pypeit_multi_object_prefers_science_and_warns_when_close(tmp_path, caplog):
    path = _multi_spec1d(tmp_path / 'm.fits', [200., 120., 100.], ['A', 'B', 'C'], ['standard', 'science', 'science'])
    caplog.set_level(logging.INFO, logger='rvfitter')
    spec = io.read_spectrum(path)
    assert spec.meta['hdu'] == 2  # the brightest table is not a science object
    assert 'less than twice as bright' in caplog.text


def test_pypeit_multi_object_matches_target_name(tmp_path):
    path = _multi_spec1d(tmp_path / 'm.fits', [100., 10.], ['SPAT0100', 'SPAT0200'], ['science', 'science'],
                         target='J1745-1640', maskdef=['other star', 'J1745-1640'])
    assert io.read_spectrum(path).meta['hdu'] == 2  # the name wins over S/N


@pytest.mark.data
@pytest.mark.skipif(not TESTDATA, reason='RVFITTER_TESTDATA not set')
class TestLocalData:
    base = Path(TESTDATA or '.') / 'pypeit/R2500I/GTC54-15A0/OB0055'

    def test_spec1d_target_is_the_bright_trace(self):
        """J1745's spec1d holds SPAT0168 (S/N 17) first and the target SPAT0240 (S/N 86) second"""
        path = next((self.base / 'Science').glob('spec1d_*DENIS*.fits'))
        spec = io.read_spectrum(path)
        assert spec.meta['hdu'] == 2 and spec.meta['extname'].startswith('SPAT0240')
        assert np.nanmedian(spec.flux / spec.sigma) > 50.
        assert io.read_spectrum(path, hdu=1).meta['extname'].startswith('SPAT0168')

    def test_tellcorr_empty_header(self):
        path = self.base / 'Science/DENISJ174534.6-164053_tellcorr.fits'
        # Decision: wave/flux/ivar tables are PypeIt coadd/tellcorr products and are vacuum by default
        spec = io.read_spectrum(path)
        assert spec.meta['medium_in'] == 'vacuum' and spec.meta['medium_source'] == 'default:pypeit_coadd'
        assert len(spec.wave) > 1000 and np.all(np.diff(spec.wave) > 0)
        assert 7000 < spec.wave[0] < 8000 and 9500 < spec.wave[-1] < 11000
        assert np.isfinite(spec.sigma).any()
        air = io.read_spectrum(path, medium='air')
        assert np.all(air.wave > spec.wave)

    @pytest.mark.parametrize('sub, vtype', [('nopipecorr', None), ('yespipecorr', 'barycentric'),
                                            ('yespipecorrhelio', 'heliocentric')])
    def test_spec1d_vel_keywords(self, sub, vtype):
        files = sorted((self.base / 'sciencebk' / sub).glob('spec1d_*DENIS*.fits'))
        assert files
        spec = io.read_spectrum(files[0])
        m = spec.meta
        assert m['object'] == 'DENISJ174534.6-164053' and m['site'] == 'GTC'
        assert m['ra'] == pytest.approx(266.4295, abs=1e-3) and m['airmass'] == pytest.approx(1.8807, abs=1e-3)
        assert m['mjd_mid'] == pytest.approx(57238.0226151791 + 375. / 86400., abs=1e-8)
        assert m['frame_in'] == vtype
        if vtype is None:
            assert m['vel_corr'] is None and 'VEL_CORR' not in m['velocity_keywords']
        else:
            assert m['vel_corr'] == pytest.approx(0.99993, abs=1e-5)
        assert np.isfinite(spec.sigma).sum() > 1000


# ---- PypeIt masks and arc-line range ---------------------------------------------------------------------------
def test_pypeit_opt_mask_drops_masked_pixels(tmp_path):
    path = _spec1d(tmp_path / 's.fits')
    bad = np.zeros(len(WAVE), bool)
    bad[[10, 11, 300]] = True
    with fits.open(path, mode='update') as h:
        t = h[1].data
        cols = [fits.Column(name=n, format='D', array=t[n]) for n in t.columns.names]
        cols += [fits.Column(name='OPT_MASK', format='L', array=~bad),
                 fits.Column(name='BOX_MASK', format='L', array=np.ones(len(WAVE), bool))]
        t['OPT_FLAM'][bad] = -2.4e4  # a wild flux with a finite ivar, as PypeIt leaves it
        new = fits.BinTableHDU.from_columns(cols)
        new.data['OPT_FLAM'][:] = t['OPT_FLAM']
        h[1] = new
    spec = io.read_spectrum(path)
    assert len(spec.wave) == len(WAVE) - 3
    assert not np.any(np.isin(spec.wave, WAVE[bad])) and spec.flux.min() > 0
    # box extraction uses BOX_MASK, which is all good here
    with fits.open(path, mode='update') as h:
        h[1].data['OPT_WAVE'][:] = 0.
    assert len(io.read_spectrum(path).wave) == len(WAVE)


def _wvcalib(path: Path, waves=(7500., 8000., 9000.), spat_id=0, extra=None) -> Path:
    exts = [fits.PrimaryHDU()]
    for sid, w in [(spat_id, waves), *(extra or [])]:
        col = fits.Column(name='wave_fit', format=f'{len(w)}D', array=np.array([w]))
        exts.append(fits.BinTableHDU.from_columns([col], name=f'SPAT_ID-{sid}_WAVEFIT'))
        exts.append(fits.BinTableHDU.from_columns([fits.Column(name='xval', format='D', array=np.zeros(1))],
                                                  name=f'SPAT_ID-{sid}_PYPEITFIT'))  # ignored
    fits.HDUList(exts).writeto(path)
    return path


def test_arc_range_from_calibrations_dir(tmp_path):
    (tmp_path / 'Science').mkdir()
    (tmp_path / 'Calibrations').mkdir()
    _wvcalib(tmp_path / 'Calibrations' / 'WaveCalib_A_0_DET01.fits')
    spec = io.read_spectrum(_spec1d(tmp_path / 'Science' / 's.fits'))
    assert spec.meta['arc_range'] == (7500., 9000.)
    assert spec.meta['arc_file'].endswith('WaveCalib_A_0_DET01.fits')


def test_arc_range_absent_without_calibration(tmp_path):
    assert 'arc_range' not in io.read_spectrum(_spec1d(tmp_path / 's.fits')).meta


def test_arc_range_same_directory_slit_match_and_air(tmp_path):
    _wvcalib(tmp_path / 'wvcalib.fits', waves=(7000., 7100.), spat_id=168, extra=[(240, (7600., 9400.))])
    path = _spec1d(tmp_path / 's.fits')
    with fits.open(path, mode='update') as h:
        h[1].name = 'SPAT0240-SLIT0457-DET02'
    assert io.read_spectrum(path).meta['arc_range'] == (7600., 9400.)  # the slit's own lines only
    # PypeIt arc solutions are vacuum whatever the spectrum's medium, so they are never converted
    assert io.read_spectrum(path, medium='air').meta['arc_range'] == (7600., 9400.)


@pytest.mark.data
@pytest.mark.skipif(not TESTDATA, reason='RVFITTER_TESTDATA not set')
def test_j1745_arc_range_and_mask():
    base = Path(TESTDATA) / 'pypeit/R2500I/GTC54-15A0/OB0055'
    path = next((base / 'Science').glob('spec1d_*DENIS*.fits'))
    spec = io.read_spectrum(path)
    lo, hi = spec.meta['arc_range']
    assert lo == pytest.approx(7437.416, abs=1e-3) and hi == pytest.approx(9925.919, abs=1e-3)
    assert spec.meta['arc_file'].endswith('OB0055/wvcalib.fits')
    assert spec.flux.min() > -100.  # HDU 2's OPT_MASK removes the pixel at -7e7 (flux/sigma -24745)
    nopipe = next((base / 'sciencebk/nopipecorr').glob('spec1d_*DENIS*.fits'))
    assert io.read_spectrum(nopipe).meta['arc_range'] == (lo, hi)  # found three levels up
    tell = io.read_spectrum(base / 'Science/DENISJ174534.6-164053_tellcorr.fits')
    assert tell.meta['arc_range'] == (lo, hi)
