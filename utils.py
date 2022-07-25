from astropy.io.fits import getdata
from astropy.modeling import Fittable1DModel
from astropy.nddata import StdDevUncertainty
import astropy.units as u
from matplotlib import rcParams
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from copy import copy
import logging
import json
import os
from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import extract_region
from specutils.utils.wcs_utils import vac_to_air
from typing import Union, Optional, List, Dict, Tuple


class Quantiser:

    def __init__(self, wunit: u.Quantity, funit: u.Quantity, rvunit: u.Quantity, spec: Spectrum1D):
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
        if not self.__checkcontpoints__():
            return None
        return [self.c1, self.c2], [self.c3, self.c4]

    def getlinewindow(self) -> Optional[SpectralRegion]:
        if not self.__checkregionpoints__():
            return None
        else:
            return SpectralRegion(self.r1, self.r2)

    def cutspec(self, spec: Spectrum1D):
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
        self.sub_spec = self.cutspec(self.spec)
        self.contwindow = self.getcontwindow()
        self.linewindow = self.getlinewindow()


def inv_rv_calc(shift: float, wave: np.ndarray) -> np.ndarray:
    c = 299792458 / 1e3
    cair = c / 1.000276
    return shift * wave / cair


def json_handle(jf: Union[os.PathLike, str, bytes],
                d: Dict[str, Dict[str, List[Union[float, str, bool]]]] = None) \
        -> Dict[str, Dict[str, List[Union[float, str, bool]]]]:
    if not os.path.exists(jf):
        d = {}
    if d is None:
        with open(jf, 'r') as jd:
            d = json.load(jd)
    else:
        with open(jf, 'w') as jd:
            json.dump(d, jd)
    return d


def normaliser(x: np.ndarray, *args, xmin: float = 8100, xmax: float = 8200):
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
    if f.endswith('txt'):
        try:
            wave, flux = np.loadtxt(f, unpack=True, usecols=(0, 1))  # load file
        except (OSError, FileNotFoundError) as e:
            raise (e, 'Cannot find given file in: ', f)
        except ValueError:
            wave, flux = np.loadtxt(f, unpack=True, usecols=(0, 1), skiprows=1)  # load file
        fluxerr = np.zeros_like(flux)
    else:  # fits
        target = getdata(f)
        wave = target.wave[1:]
        wave = np.array(vac_to_air(wave * u.AA, method='Edlen1953') / u.AA)
        flux = target.flux[1:]
        fluxerr = np.divide(1., target.ivar[1:], where=~np.isclose(target.ivar[1:], 0))
    wunit = kwargs.get('wunit', u.AA)
    funit = kwargs.get('funit', u.erg / u.cm ** 2 / u.Angstrom / u.s)
    wave, flux, fluxerr = normaliser(wave, flux, fluxerr, xmin=np.min(wave), xmax=np.max(wave))
    unc = StdDevUncertainty(fluxerr, unit=funit)
    spec = Spectrum1D(flux * funit, wave * wunit,
                      uncertainty=unc)
    return spec


def logging_rvcalc(s: str = '', perm: str = 'a'):
    with open('calculating.log', perm) as f:
        f.write(s + '\n')
    return


def spec_unpack(spec: Spectrum1D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wave = copy(spec.spectral_axis.value)
    flux = copy(spec.flux.value)
    fluxerr = copy(spec.uncertainty.quantity.value)
    return wave, flux, fluxerr


def stephens(s: Union[pd.Series, float]) -> np.ndarray:
    teff = 4400.9 - 467.26 * s + 54.67 * s ** 2 - 4.4727 * s ** 3 + 0.17667 * s ** 4 - 0.0025492 * s ** 5
    teff = np.where((1200 < teff) & (teff < 4000), teff, np.nan)
    return teff


logging.getLogger().setLevel(logging.ERROR)
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
