from astropy import log as astropy_log
from astropy.io.fits import getdata
from astropy.modeling import Fittable1DModel
from astropy.nddata import StdDevUncertainty
import astropy.units as u
from matplotlib import rcParams
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from copy import copy
import json
import os
from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import extract_region
from specutils.utils.wcs_utils import vac_to_air
from typing import Union, Optional, List, Dict, Tuple


class Quantiser:
    """
    Underlying RV fitting class
    """

    def __init__(self, wunit: u.Quantity, funit: u.Quantity, rvunit: u.Quantity, spec: Spectrum1D):
        """
        When initialising the line centering programme

        Parameters
        ----------
        wunit
            The wavelength unit
        funit
            The flux unit
        rvunit
            The radial velocity unit
        spec
            The spectrum of the object
        """
        self.wunit = wunit
        self.funit = funit
        self.rvunit = rvunit
        self.c1 = None
        self.c2 = None
        self.c3 = None
        self.c4 = None
        self.r1 = None
        self.r2 = None
        self.iscut = False
        self.rescale = True
        self.best_rmsdiqr = np.inf
        self.snr = 0
        self.spec = spec
        self.sub_spec = self.spec
        self.cont = None
        self.contwindow = self.getcontwindow()
        self.linewindow = self.getlinewindow()

    def __assertwavelength__(self, value: Optional[Union[float, u.Quantity]]) -> Optional[u.Quantity]:
        if isinstance(value, float) or isinstance(value, int):
            value *= self.wunit
        elif not self.__assertquantity__(value, True):
            pass
        return value

    def __assertflux__(self, value: Optional[Union[float, u.Quantity]]) -> Optional[u.Quantity]:
        if isinstance(value, float) or isinstance(value, int):
            value *= self.funit
        elif not self.__assertquantity__(value, True):
            pass
        return value

    def __assertrv__(self, value: Optional[Union[float, u.Quantity]]) -> Optional[u.Quantity]:
        if isinstance(value, float) or isinstance(value, int):
            value *= self.rvunit
        elif not self.__assertquantity__(value, True):
            pass
        return value

    @staticmethod
    def __assertquantity__(value: Optional[u.Quantity], optional: bool):
        if optional and value is not None and not isinstance(value, u.Quantity):
            raise AttributeError(f'Value {value} must be numeric or a quantity (or None)')
        elif not optional and not (isinstance(value, u.CompositeUnit) or isinstance(value, u.Unit) or
                                   isinstance(value, u.IrreducibleUnit)):
            print(value, type(value))
            raise AttributeError(f'Value {value} must be a unit')
        return True

    @staticmethod
    def __assertmodel__(value: Optional[Fittable1DModel]) -> Optional[Fittable1DModel]:
        if value is not None and not isinstance(value, Fittable1DModel):
            raise AttributeError(f'{value} needs to be a Fittable1DModel (astropy)')
        return value

    @property
    def wunit(self) -> u.Unit:
        return self._wunit

    @wunit.setter
    def wunit(self, value):
        if not self.__assertquantity__(value, False):
            raise AttributeError('wunit must be an astropy unit')
        self._wunit = value

    @property
    def funit(self) -> u.CompositeUnit:
        return self._funit

    @funit.setter
    def funit(self, value):
        if not self.__assertquantity__(value, False):
            raise AttributeError('funit must be an astropy unit')
        self._funit = value

    @property
    def rvunit(self) -> u.CompositeUnit:
        return self._rvunit

    @rvunit.setter
    def rvunit(self, value):
        if not self.__assertquantity__(value, False):
            raise AttributeError('rvunit must be an astropy unit')
        self._rvunit = value

    @property
    def c1(self) -> Optional[u.Quantity]:
        return self._c1

    @c1.setter
    def c1(self, value: Union[float, u.Quantity]):
        self._c1 = self.__assertwavelength__(value)

    @property
    def c2(self) -> Optional[u.Quantity]:
        return self._c2

    @c2.setter
    def c2(self, value: Union[float, u.Quantity]):
        self._c2 = self.__assertwavelength__(value)

    @property
    def c3(self) -> Optional[u.Quantity]:
        return self._c3

    @c3.setter
    def c3(self, value: Union[float, u.Quantity]):
        self._c3 = self.__assertwavelength__(value)

    @property
    def c4(self) -> Optional[u.Quantity]:
        return self._c4

    @c4.setter
    def c4(self, value: Union[float, u.Quantity]):
        self._c4 = self.__assertwavelength__(value)

    @property
    def r1(self) -> Optional[u.Quantity]:
        return self._r1

    @r1.setter
    def r1(self, value: Union[float, u.Quantity]):
        self._r1 = self.__assertwavelength__(value)

    @property
    def r2(self) -> Optional[u.Quantity]:
        return self._r2

    @r2.setter
    def r2(self, value: Union[float, u.Quantity]):
        self._r2 = self.__assertwavelength__(value)

    @property
    def cont(self) -> Optional[Fittable1DModel]:
        return self._cont

    @cont.setter
    def cont(self, value):
        self._cont = self.__assertmodel__(value)

    def __checkcontpoints__(self) -> bool:
        if any([cpoint is None for cpoint in (self.c1, self.c2, self.c3, self.c4)]):
            return False
        return True

    def __checkregionpoints__(self) -> bool:
        if any([rpoint is None for rpoint in (self.r1, self.r2)]):
            return False
        return True

    def getcontwindow(self) -> Optional[Tuple[List[u.Quantity], List[u.Quantity]]]:
        """
        Retrieving the continuum window

        Returns
        -------
        _
            The collection of continuum points
        """
        if not self.__checkcontpoints__():
            return None
        return [self.c1, self.c2], [self.c3, self.c4]

    def getlinewindow(self) -> Optional[SpectralRegion]:
        """
        Retrieving the region with the spectral line

        Returns
        -------
        _
            The spectral region for the line
        """
        if not self.__checkregionpoints__():
            return None
        else:
            return SpectralRegion(self.r1, self.r2)

    def cutspec(self, spec: Spectrum1D) -> Spectrum1D:
        """
        Cutting the spectrum to the left and right-most points

        Parameters
        ----------
        spec
            The spectrum to be cut
        Returns
        -------
        spec
            The spectrum cut
        """
        x1, x2 = self.c1, self.c4
        spec = copy(spec)
        if not any([xpoint is None for xpoint in (x1, x2)]):
            xreg = SpectralRegion(x1, x2)
            sub_spec = extract_region(spec, xreg)
            self.iscut = True
            return sub_spec
        return spec

    @staticmethod
    def poly_cutter(wave: np.ndarray, flux: np.ndarray, fluxerr: np.ndarray = None,
                    polycoeff: int = 5, bign: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creating a interpolation of a given order

        Parameters
        ----------
        wave
            The array of wavelengths
        flux
            The array of fluxes
        fluxerr
            The array of flux errors
        polycoeff
            The coefficient to use when interpolating
        bign
            The number by which to multiply the number of data points by

        Returns
        -------
        x, y, yerr
            The array of wavelengths
            The array of fluxes
            The array of flux errors
        """
        if bign is None:
            bign = len(wave) * 10
        x = np.linspace(np.min(wave), np.max(wave), bign)
        p = interp1d(wave, flux, kind=polycoeff)
        y = p(x)
        if fluxerr is not None:
            perr = interp1d(wave, fluxerr, kind=polycoeff)
            yerr = perr(x)
        else:
            yerr = np.empty_like(flux)
        return x, y, yerr

    def __updatewindows__(self):
        """
        Updating the windows for cutting spectra by the continuum and line window
        """
        self.sub_spec = self.cutspec(self.spec)
        self.contwindow = self.getcontwindow()
        self.linewindow = self.getlinewindow()


def inv_rv_calc(shift: float, wave: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Inverse RV shift

    Parameters
    ----------
    shift
        The shift in RV
    wave
        The wavelength values to shift

    Returns
    -------
    _
        The shifted wavelengths
    """
    c = 299792458 / 1e3
    cair = c / 1.000276
    return shift * wave / cair


def json_handle(jf: Union[os.PathLike, str, bytes],
                d: Dict[str, Dict[str, List[Union[float, str, bool]]]] = None) \
        -> Dict[str, Dict[str, List[Union[float, str, bool]]]]:
    """
    Handling a .json, either saving or loading

    Parameters
    ----------
    jf
        The filename of the .json
    d
        The dictionary of data to be saved or loaded to the json

    Returns
    -------
    d
        The dictionary of data to be saved or loaded to the json
    """
    if not os.path.exists(jf):
        d = {}
    if d is None:
        with open(jf, 'r') as jd:
            d = json.load(jd)
    else:
        with open(jf, 'w') as jd:
            json.dump(d, jd)
    return d


def sigma_clipper(wave: np.ndarray, flux: np.ndarray, fluxerr: np.ndarray,
                  sigma: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sigma clip the spectra by the sigma from the median of entire flux array

    Parameters
    ----------
    wave
        Wavelength array
    flux
        Flux array
    fluxerr
        Flux error array
    sigma
        Number of sigma to determine as outliers

    Returns
    -------
    wave
        Wavelength array, cut
    flux
        Flux array, cut
    fluxerr
        Flux error array, cut
    """
    med = np.median(flux)
    std = np.std(flux)
    boolcut: np.ndarray = (flux > med - sigma * std) & (flux < med + sigma * std)

    if len(flux[boolcut]) < int(0.95 * len(flux)):  # 95% of original data
        return wave, flux, fluxerr

    wave = wave[boolcut]
    flux = flux[boolcut]
    fluxerr = fluxerr[boolcut]
    return wave, flux, fluxerr


def normaliser(x: np.ndarray, *args, xmin: float = 8100, xmax: float = 8200):
    """
    Normalising a flux from a given wavelength

    Parameters
    ----------
    x
        Wavelength array
    args
        The arguments to be normalised
    xmin
        The minimum wavelength
    xmax
        The maximum wavelength

    Returns
    -------
    out
        The list of arrays starting with wavelength
    """
    boolcut: np.ndarray = (x > xmin) & (x < xmax)
    args = list(args)
    if np.any([len(x) != len(arg) for arg in args]):
        raise IndexError('check input shapes')
    normval = np.nanmedian(args[0][boolcut])
    for i, val in enumerate(args):
        args[i] /= normval
    out = [x, ] + args
    return out


def freader(f: str, **kwargs) -> Spectrum1D:
    """
    Reading a file out into a spectrum

    Parameters
    ----------
    f
        The filename
    kwargs
        Extra parameters

    Returns
    -------
    spec
        The spectrum from the filename
    """
    wavearr: Optional[np.ndarray] = kwargs.get('wavearr', None)  # the wavelength array to interpolate for
    wunit = kwargs.get('wunit', u.AA)
    funit = kwargs.get('funit', u.erg / u.cm ** 2 / wunit / u.s)
    if f.endswith('txt'):
        try:
            wave, flux = np.loadtxt(f, unpack=True, usecols=(0, 1))  # load file
        except (OSError, FileNotFoundError) as e:
            raise (e, 'Cannot find given file in: ', f)
        except ValueError:
            wave, flux = np.loadtxt(f, unpack=True, usecols=(0, 1), skiprows=1)  # load file
        if wavearr is not None:
            flux = np.interp(wavearr, wave, flux)
            wave = wavearr
        fluxerr = np.zeros_like(flux)
    else:  # fits
        target = getdata(f)
        try:
            wave = target.wave
            wave, uniquebool = np.unique(wave, return_index=True)  # check there aren't duplicate wavelengths
            wave = np.array(vac_to_air(wave * wunit, method='Edlen1953') / wunit)
            flux = target.flux[uniquebool]
            fluxivar = target.ivar[uniquebool]
            fluxerr = np.divide(1., fluxivar, where=~np.isclose(fluxivar, 0))
        except AttributeError:
            wave = target[0]
            wave: np.ndarray = np.array(vac_to_air(wave * wunit, method='Edlen1953') / wunit)
            flux = target[1]
            fluxerr = target[2]
            boolcut = (~np.isnan(wave)) & (~np.isnan(flux)) & (~np.isnan(fluxerr))
            wave = wave[boolcut]
            flux = flux[boolcut]
            fluxerr = fluxerr[boolcut]
    wave, flux, fluxerr = sigma_clipper(wave, flux, fluxerr, sigma=5)
    wave, flux, fluxerr = normaliser(wave, flux, fluxerr, xmin=np.min(wave), xmax=np.max(wave))
    unc = StdDevUncertainty(fluxerr, unit=funit)
    spec = Spectrum1D(flux * funit, wave * wunit,
                      uncertainty=unc)
    return spec


def rmsdiqr_check(observed: np.ndarray, expected: np.ndarray, best: float) -> Tuple[float, bool]:
    """
    Calculate the rmsdiqr of two distributions

    Parameters
    ----------
    observed
        Observed values
    expected
        Expected values
    best
        Current best value

    Returns
    -------
    chi
        Chisquare value
    significant
        Switch whether value is significant
    """
    rmsd = np.sqrt(np.sum((observed - expected) ** 2) / len(observed))
    iqr = np.subtract(*np.percentile(observed, [75, 25]))
    rmsdiqr = rmsd / iqr
    if rmsdiqr <= best:
        significant = True
    else:
        significant = False
    return rmsdiqr, significant


def get_snr_and_rvunc(spec: Spectrum1D, continuum_region: SpectralRegion,
                      fitting_region: SpectralRegion, fwhm: u.Quantity) -> Tuple[float, float]:
    """
    Finds the SNR for a given region and the minimum RV uncertainty based on that

    Parameters
    ----------
    spec
        The spectrum object
    continuum_region
        The continuum region not used in the line fitting
    fitting_region
        The region used in line fitting
    fwhm
        FWHM of the fit

    Returns
    -------
    snr_value
        Signal to noise ratio
    rvunc_min
        Minimum rv uncertainty from resolution
    """
    line_reg: Spectrum1D = extract_region(spec, fitting_region)
    n_pix = len(line_reg.spectral_axis.value)
    cont_reg: Spectrum1D = extract_region(spec, continuum_region, return_single_spectrum=True)
    fwhm_value = fwhm.value
    resolution = np.nanmean(line_reg.spectral_axis.value) / fwhm_value
    snr_value = np.mean(cont_reg.flux.value) / np.std(cont_reg.flux.value)
    rvunc_min = 3e5 / (resolution * np.sqrt(n_pix * snr_value))
    return snr_value, rvunc_min


def logging_rvcalc(s: str = '', perm: str = 'a'):
    """
    Logging the information to a file

    Parameters
    ----------
    s
        The string being saved
    perm
        The permission of the file opening
    """
    if not os.path.exists('calculating.log'):
        perm = 'w'
    with open('calculating.log', perm) as f:
        f.write(s + '\n')
    return


def spec_unpack(spec: Spectrum1D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpacking a spectrum object into wavelength, flux and flux error

    Parameters
    ----------
    spec
        The spectrum to be unpacked

    Returns
    -------
    wave, flux, fluxerr
        The arrays of wavelength, flux and flux error
    """
    wave = copy(spec.spectral_axis.value)
    flux = copy(spec.flux.value)
    fluxerr = copy(spec.uncertainty.quantity.value)
    return wave, flux, fluxerr


def stephens(s: Union[pd.Series, float]) -> Union[np.ndarray, float]:
    """
    Stephens relation for converting to teff

    Parameters
    ----------
    s
        The spectral type number for the Stephens relation

    Returns
    -------
    teff
        The effective temperature converted
    """
    teff = 4400.9 - 467.26 * s + 54.67 * s ** 2 - 4.4727 * s ** 3 + 0.17667 * s ** 4 - 0.0025492 * s ** 5
    teff = np.where((1200 < teff) & (teff < 4000), teff, np.nan)
    return teff


astropy_log.setLevel('ERROR')
dpi = 200  # 200-300 as per guidelines
maxpix = 670  # max pixels of plot
width = maxpix / dpi  # max allowed with
rcParams.update({'axes.labelsize': 'large', 'axes.titlesize': 'large',  # the size of labels and title
                 'xtick.labelsize': 'large', 'ytick.labelsize': 'large',  # the size of the axes ticks
                 'legend.fontsize': 'large', 'legend.frameon': False,  # legend font size, no frame
                 'legend.facecolor': 'none', 'legend.handletextpad': 0.25,
                 # legend no background colour, separation from label to point
                 'font.serif': ['Computer Modern', 'Helvetica', 'Arial',  # default fonts to try and use
                                'Tahoma', 'Lucida Grande', 'DejaVu Sans'],
                 'font.family': 'serif',  # use serif fonts
                 'mathtext.fontset': 'cm', 'mathtext.default': 'regular',  # if in math mode, use these
                 'figure.figsize': [width, 0.7 * width], 'figure.dpi': dpi,
                 # the figure size in inches and dots per inch
                 'lines.linewidth': .75,  # width of plotted lines
                 'xtick.top': True, 'ytick.right': True,  # ticks on right and top of plot
                 'xtick.minor.visible': True, 'ytick.minor.visible': True,  # show minor ticks
                 'text.usetex': True})  # process text with LaTeX instead of matplotlib math mode
