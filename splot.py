from astropy.modeling import Fittable1DModel
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from specutils import SpectralRegion, Spectrum1D
from specutils.manipulation import extract_region
from specutils.fitting import fit_lines, fit_continuum

import logging
from typing import Union, Optional, Tuple, List, Dict


class Quantiser:

    def __init__(self, wunit: u.Quantity, funit: u.Quantity, rvunit: u.Quantity):
        self.wunit = wunit
        self.funit = funit
        self.rvunit = rvunit
        self.c1 = None
        self.c2 = None
        self.c3 = None
        self.c4 = None
        self.r1 = None
        self.r2 = None

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
        elif not optional and not (isinstance(value, u.CompositeUnit) or isinstance(value, u.Unit)):
            raise AttributeError(f'Value {value} must be a unit')
        return True

    @staticmethod
    def __assertmodel__(value: Optional[Fittable1DModel]) -> Optional[Fittable1DModel]:
        if value is not None and not isinstance(value, Fittable1DModel):
            raise AttributeError(f'{value} needs to be a Fittable1DModel (astropy)')
        return value

    @property
    def wunit(self):
        return self._wunit

    @wunit.setter
    def wunit(self, value: u.Unit) -> u.Unit:
        if not self.__assertquantity__(value, False):
            raise AttributeError('wunit must be an astropy unit')
        self._wunit = value

    @property
    def funit(self):
        return self._funit

    @funit.setter
    def funit(self, value: u.Unit) -> u.Unit:
        if not self.__assertquantity__(value, False):
            raise AttributeError('funit must be an astropy unit')
        self._funit = value

    @property
    def rvunit(self):
        return self._rvunit

    @rvunit.setter
    def rvunit(self, value: u.Unit) -> u.Unit:
        if not self.__assertquantity__(value, False):
            raise AttributeError('rvunit must be an astropy unit')
        self._rvunit = value

    @property
    def c1(self):
        return self._c1

    @c1.setter
    def c1(self, value: Union[float, u.Quantity]) -> u.Quantity:
        self._c1 = self.__assertwavelength__(value)

    @property
    def c2(self):
        return self._c2

    @c2.setter
    def c2(self, value: Union[float, u.Quantity]) -> u.Quantity:
        self._c2 = self.__assertwavelength__(value)

    @property
    def c3(self):
        return self._c3

    @c3.setter
    def c3(self, value: Union[float, u.Quantity]) -> u.Quantity:
        self._c3 = self.__assertwavelength__(value)

    @property
    def c4(self):
        return self._c4

    @c4.setter
    def c4(self, value: Union[float, u.Quantity]) -> u.Quantity:
        self._c4 = self.__assertwavelength__(value)

    @property
    def r1(self):
        return self._r1

    @r1.setter
    def r1(self, value: Union[float, u.Quantity]) -> u.Quantity:
        self._r1 = self.__assertwavelength__(value)

    @property
    def r2(self):
        return self._r2

    @r2.setter
    def r2(self, value: Union[float, u.Quantity]) -> u.Quantity:
        self._r2 = self.__assertwavelength__(value)


class Splot(Quantiser):

    def __init__(self, spec: Spectrum1D, labline: u.Quantity, spec_index: str, ax: plt.Axes = None, **kwargs):
        wunit = kwargs.get('wunit', u.AA)
        funit = kwargs.get('funit', u.erg / u.cm ** 2 / u.Angstrom / u.s)
        rvunit = kwargs.get('rvunit', u.km / u.s)
        self.spec_index = spec_index
        super().__init__(wunit, funit, rvunit)
        self.kwargs = kwargs
        self.c1 = kwargs.get('c1', self.c1)
        self.c2 = kwargs.get('c2', self.c2)
        self.r1 = kwargs.get('r1', self.r1)
        self.r2 = kwargs.get('r2', self.r2)
        self.c3 = kwargs.get('c3', self.c3)
        self.c4 = kwargs.get('c4', self.c4)
        self.spec = spec
        self.sub_spec = self.spec
        self.sub_speccorr = self.sub_spec
        self.ax = ax
        self.labline = self.__assertwavelength__(labline)
        self.mu = self.__assertwavelength__(kwargs.get('mu', self.labline))
        self.std = self.__assertwavelength__(kwargs.get('std', 2))
        self.fwhm_G = self.__assertwavelength__(kwargs.get('fwhm_G', self.__stdtofwhm__()))
        self.fwhm_L = self.__assertwavelength__(kwargs.get('fwhm_L', self.__stdtofwhm__()))
        self.fwhm_V = self.__assertwavelength__(kwargs.get('fwhm_V', self.__stdtofwhm__()))
        self.x_0 = self.__assertwavelength__(kwargs.get('x_0', self.labline))
        self.amplitude = self.__assertflux__(kwargs.get('amplitude', 0))
        self.line_profile = kwargs.get('line_profile', 'gaussian')
        self.working_profile = self.line_profile
        self.cont = kwargs.get('cont', None)
        self.fitted_profile = kwargs.get('fitted_profile', None)
        self.linewindow = self.getlinewindow()
        self.contwindow = self.getcontwindow()
        self.fitter = LevMarLSQFitter(calc_uncertainties=True)
        self.shift = self.__assertwavelength__(np.nan)
        self.shifterr = self.__assertwavelength__(np.nan)
        self.rv = self.__assertrv__(np.nan)
        self.rverr = self.__assertrv__(np.nan)
        self.cov: np.ndarray = np.eye(3)
        self.iscut = False
        self.contfound = False
        self.profilefound = False
        self.linewidth = None
        self.lineedges = None
        self.rescale = True
        self.use = kwargs.get('use', True)
        return

    def reset(self):
        self.__init__(self.spec, self.labline, self.spec_index, self.ax, **self.kwargs)

    @property
    def line_profile(self):
        return self._line_profile

    @line_profile.setter
    def line_profile(self, value: str) -> str:
        if not isinstance(value, str):
            raise AttributeError('line_profile must be a string')
        if value not in ('gaussian', 'lorentzian', 'voigt'):
            raise AttributeError('line_profile must be one of: gaussian, lorentzian or voigt')
        self._line_profile = value

    @property
    def cont(self):
        return self._cont

    @cont.setter
    def cont(self, value) -> Optional[Fittable1DModel]:
        self._cont = self.__assertmodel__(value)

    @property
    def fitted_profile(self):
        return self._fitted_profile

    @fitted_profile.setter
    def fitted_profile(self, value) -> Optional[Fittable1DModel]:
        self._fitted_profile = self.__assertmodel__(value)

    def __str__(self):
        s = """
Interactive plotting routine help:
? - Prints this help menu
q - Quit routine (lines are rejected by default)
r - Resets back to default
1 - Selects left hand edge of spectra; anything further left is cut
2 - Selects right most edge of left hand continuum box
3 - Selects left most edge of line being fitted
4 - Selects right most edge of line being fitted
5 - Selects left most edge of right hand continuum box
6 - Selects right hand edge of spectra; anything further right is cut
7 - Fits a Gaussian profile
8 - Fits a Lorentzian profile
9 - Fits a Voigt profile
a - Marks initial guess of amplitude (default = 0)
x - Marks initial guess of line center (default = laboratory line)
y - Accept this line fitting
n - Reject this line fitting
b - Go back to previous line
        """
        return s

    def __stdtofwhm__(self,) -> u.Quantity:
        return self.std * (2 * np.sqrt(2 * np.log(2)))

    def __checkcontpoints__(self) -> bool:
        if any([cpoint is None for cpoint in (self.c1, self.c2, self.c3, self.c4)]):
            return False
        return True

    def __checkregionpoints__(self) -> bool:
        if any([rpoint is None for rpoint in (self.r1, self.r2)]):
            return False
        return True

    def getcontwindow(self) -> Optional[Tuple[SpectralRegion]]:
        if not self.__checkcontpoints__():
            return None
        return [self.c1, self.c2], [self.c3, self.c4]

    def getlinewindow(self) -> Optional[SpectralRegion]:
        if not self.__checkregionpoints__():
            return None
        else:
            return SpectralRegion(self.r1, self.r2)

    def cutspec(self):
        x1, x2 = self.c1, self.c4
        if not any([xpoint is None for xpoint in (x1, x2)]):
            xreg = SpectralRegion(x1, x2)
            self.sub_spec = extract_region(self.spec, xreg)
            self.iscut = True
            self.rescale = True

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
        self.cutspec()
        self.contwindow = self.getcontwindow()
        self.linewindow = self.getlinewindow()

    def getcont(self):
        self.__updatewindows__()
        if self.contwindow is None:
            return
        if self.cont is None:
            self.cont = models.Linear1D(0.01)
        self.cont = fit_continuum(self.spec, self.cont, window=self.contwindow, exclude_regions=self.linewindow)
        sub_fluxcorr = self.sub_spec.flux - self.cont(self.sub_spec.spectral_axis)
        self.sub_speccorr = Spectrum1D(flux=sub_fluxcorr, spectral_axis=self.sub_spec.spectral_axis,
                                       uncertainty=self.sub_spec.uncertainty)
        self.contfound = True
        self.rescale = True

    def __fitready__(self):
        self.getcont()
        self.fitter = LevMarLSQFitter(calc_uncertainties=True)

    def getshift(self, ind: int):
        self.shift = self.x_0 - self.labline
        self.cov = self.fitter.fit_info['param_cov']
        errs = np.sqrt(np.diag(self.cov))
        self.shifterr = errs[ind] * self.wunit

    def __rvcalc__(self, shift: u.Quantity) -> u.Quantity:
        c = 299792458 / 1e3 * self.rvunit
        cair = c / 1.000276
        return cair * shift / self.labline

    def getrv(self):
        self.rv = self.__rvcalc__(self.shift)
        rvup = self.__rvcalc__(self.shift + self.shifterr) - self.rv
        rvdown = self.rv - self.__rvcalc__(self.shift - self.shifterr)
        self.rverr = (rvup + rvdown) / 2

    def getlinewidth(self):
        left = self.x_0 - self.std * 2
        right = self.x_0 + self.std * 2
        self.lineedges = [left, right]
        self.linewidth = self.__rvcalc__(right - left)

    def fit_line_wrap(self, *args, **kwargs):
        try:
            fitted = fit_lines(*args, **kwargs)
            if self.fitter.fit_info['param_cov'] is None:
                raise ValueError('Could not solve')
        except Exception as e:
            print(f'Fit failed: {repr(e)}')
            if self.ax is not None:
                self.ax.text(0.5, 0.5, f'Fit failed: {repr(e)}', transform=self.ax.transAxes,
                             horizontalalignment='center')
            self.profilefound = False
            self.rescale = False
            return None
        self.rescale = True
        return fitted

    def gaussian_fit(self):
        g_init = models.Gaussian1D(self.amplitude, self.mu, self.std)
        g_fit = self.fit_line_wrap(self.sub_speccorr, g_init, fitter=self.fitter,
                                   window=self.linewindow, exclude_regions=self.contwindow)
        if g_fit is None:
            return False
        self.fitted_profile = g_fit
        self.amplitude = self.fitted_profile.amplitude
        self.mu = self.x_0 = self.fitted_profile.mean
        self.std = self.fitted_profile.stddev
        self.fwhm_G = self.__stdtofwhm__()
        self.getshift(1)
        return True

    def lorentz_fit(self):
        l_init = models.Lorentz1D(self.amplitude, self.x_0, self.fwhm_L)
        l_fit = self.fit_line_wrap(self.sub_speccorr, l_init, fitter=self.fitter,
                                   window=self.linewindow, exclude_regions=self.contwindow)
        if l_fit is None:
            return False
        self.fitted_profile = l_fit
        self.amplitude = self.fitted_profile.amplitude
        self.mu = self.x_0 = self.fitted_profile.x_0
        self.fwhm_L = self.fitted_profile.fwhm
        self.std = self.fwhm_L / 2.
        self.getshift(1)
        return True

    def voigt_fit(self):
        v_init = models.Voigt1D(self.x_0, self.amplitude, self.fwhm_L, self.fwhm_G)
        v_fit = self.fit_line_wrap(self.sub_speccorr, v_init, fitter=self.fitter,
                                   window=self.linewindow, exclude_regions=self.contwindow)
        if v_fit is None:
            return False
        self.fitted_profile = v_fit
        self.mu = self.x_0 = self.fitted_profile.x_0
        self.amplitude = self.fitted_profile.amplitude_L
        self.fwhm_L = self.fitted_profile.fwhm_L
        self.fwhm_G = self.fitted_profile.fwhm_G
        self.fwhm_V = self.fwhm_L / 2 + np.sqrt(self.fwhm_L ** 2 / 4 + self.fwhm_G ** 2)
        self.std = self.fwhm_V / 2.
        self.getshift(0)
        return True

    def fit_profile(self):
        self.__fitready__()
        if not self.contfound or self.linewindow is None:
            return
        self.contwindow = [SpectralRegion(*x, ) for x in self.contwindow]
        worked = False
        if self.line_profile == 'gaussian':
            worked = self.gaussian_fit()
        elif self.line_profile == 'lorentzian':
            worked = self.lorentz_fit()
        elif self.line_profile == 'voigt':
            worked = self.voigt_fit()
        if worked:
            self.getrv()
            self.getlinewidth()
            self.profilefound = True
            self.working_profile = self.line_profile

    def plotter(self):
        self.fit_profile()
        if self.iscut:
            spec = self.sub_spec
        else:
            spec = self.spec
        wave, flux, fluxerr = spec_unpack(spec)
        if self.rescale and not self.contfound:
            self.ax.set_xlim(np.min(wave), np.max(wave))
            self.ax.set_ylim(*np.array([np.floor(np.min(flux)),
                                        np.ceil(np.max(flux))]), )
            self.rescale = False
        if self.iscut:
            self.ax.errorbar(wave, flux, yerr=fluxerr, marker='s', lw=0, elinewidth=1.5, c='black',
                             ms=4, mfc='white', mec='black', label='Data', barsabove=True)
        else:
            self.ax.plot(wave, flux, 'k', label='Data')
        self.ax.axvline(self.labline.value, color='grey', ls='--', label='Lab')
        if self.profilefound:
            self.ax.set_title('\t' * 2 + f'{self.spec_index.capitalize()}: {self.rv.value:.1f} km/s')
        else:
            self.ax.set_title('\t' * 2 + f'{self.spec_index.capitalize()}')
        if not self.iscut:
            return
        fitx, fity, fityerr = self.poly_cutter(wave, flux, fluxerr, 5)
        self.ax.plot(fitx, fity, 'b-')
        if self.rescale and not self.profilefound:
            self.ax.set_xlim(np.min(wave), np.max(wave))
            self.ax.set_ylim(*np.array([np.floor(np.min(fity)),
                                        np.ceil(np.max(fity))]), )
        if not self.contfound or not self.use:
            return
        ls = '-'
        if self.working_profile == 'lorentzian':
            ls = '--'
        elif self.working_profile == 'voigt':
            ls = '-.'
        contyval = self.cont(fitx * u.AA).value
        self.ax.plot(fitx, contyval, c='k', ls=ls)
        self.ax.fill_betweenx([np.min(fity), np.max(fity)], self.c1.value, self.c2.value,
                              color='grey', alpha=0.25)
        self.ax.fill_betweenx([np.min(fity), np.max(fity)], self.c3.value, self.c4.value,
                              color='grey', alpha=0.25)

        if not self.profilefound:
            return
        fityval = (self.fitted_profile(fitx * u.AA) + self.cont(fitx * u.AA)).value
        self.ax.plot(fitx, fityval, c='orange', ls=ls)
        self.ax.axvline(self.x_0.value, color='black', ls='-', label='Measured')
        if self.rescale:
            self.ax.fill_betweenx([np.min(fity), np.max(fity)], self.r1.value, self.r2.value,
                                  color='grey', alpha=0.5)
            self.ax.set_ylim(0.1 * np.floor(np.min(fityval) / 0.1), 0.1 * np.ceil(np.max(fityval) / 0.1))
            self.ax.set_xlim(self.x_0.value - 2 * self.std.value, self.x_0.value + 2 * self.std.value)


def interactive_fit(spec: Spectrum1D, spec_indices: Dict[str, float], **kwargs) -> Tuple[List[str], List[Splot]]:
    def keypress(e):
        global curr_pos
        obj = objlist[curr_pos]
        if e.key == 'y':
            goodinds[curr_pos] = True
            curr_pos += 1
        elif e.key == 'n':
            goodinds[curr_pos] = False
            curr_pos += 1
        elif e.key == 'b':
            curr_pos -= 1 if curr_pos - 1 >= 0 else curr_pos
        elif e.key == 'r':
            obj.reset()
        elif e.key == '1':
            obj.c1 = e.xdata
        elif e.key == '2':
            obj.c2 = e.xdata
        elif e.key == '3':
            obj.r1 = e.xdata
        elif e.key == '4':
            obj.r2 = e.xdata
        elif e.key == '5':
            obj.c3 = e.xdata
        elif e.key == '6':
            obj.c4 = e.xdata
        elif e.key == '7':
            obj.line_profile = 'gaussian'
        elif e.key == '8':
            obj.line_profile = 'lorentzian'
        elif e.key == '9':
            obj.line_profile = 'voigt'
        elif e.key == 'a':
            obj.amplitude = e.ydata
        elif e.key == 'x':
            obj.x_0 = obj.mu = e.xdata
        elif e.key == '?':
            print(obj)
        elif e.key == 'q':
            plt.close()
        else:
            return
        if curr_pos == len(useset):
            plt.close()
            return
        for artist in plt.gca().lines + plt.gca().collections + plt.gca().texts:
            artist.remove()
        objlist[curr_pos].plotter()
        fig.canvas.draw()
        return

    global curr_pos
    curr_pos = 0

    fig, ax = plt.subplots(figsize=(8, 5))
    fig: plt.Figure = fig
    ax: plt.Axes = ax
    fig.canvas.mpl_connect('key_press_event', keypress)
    useset = np.fromiter(spec_indices.keys(), dtype='<U8')
    ax.set_xlabel(r'Wavelength [$\AA$]')
    ax.set_ylabel('Normalised Flux')
    curr_pos = 0
    goodinds = np.zeros(len(useset), dtype=bool)
    objlist = np.empty_like(goodinds, dtype=object)
    for i, (spec_index, labline) in enumerate(spec_indices.items()):
        objlist[i] = Splot(spec, labline, spec_index, ax, **kwargs)
    objlist[0].plotter()
    plt.show()
    outset = useset[goodinds].tolist()
    return outset, objlist


def spec_unpack(spec: Spectrum1D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wave = spec.spectral_axis.value
    flux = spec.flux.value
    fluxerr = spec.uncertainty.quantity.value
    return wave, flux, fluxerr


logging.getLogger().setLevel(logging.ERROR)
