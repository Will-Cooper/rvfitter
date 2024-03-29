from astropy.convolution import convolve, Gaussian1DKernel
import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import Sequence

from utils import *

curr_pos = 0
rcParams['keymap.back'].remove('left')
rcParams['keymap.forward'].remove('right')


class Xcorr(Quantiser):
    """
    The programme for cross correlation
    """

    def __init__(self, spec: Spectrum1D, labline: Union[float, u.Quantity],
                 spec_index: str, ax: plt.Axes = None, **kwargs):
        """
        When initialising the cross correlation programme

        Parameters
        ----------
        spec
            The spectrum of the object
        labline
            The wavelength of the spectral index
        spec_index
            The name of the spectral index
        ax
            The axis being plotted on
        kwargs
            Extra fitting parameters, see:
            wunit, funit, rvunit, templatedir, waverms, rv, rvstep, teff, grav, met, smoothlevel, use, c1, c4
        """
        self.kwargs = kwargs
        wunit = kwargs.get('wunit', u.AA)  # the unit for wavelengths
        funit = kwargs.get('funit', u.erg / u.cm ** 2 / wunit / u.s)  # flux unit
        rvunit = kwargs.get('rvunit', u.km / u.s)  # RV unit
        self.spec_index = spec_index
        super().__init__(wunit, funit, rvunit, spec)
        self.templatedir = kwargs.get('templatedir', 'bt-settl-cifist/useful/')  # the template directory
        self.templatedf = self.get_template_converter()
        self.spec = copy(spec)
        self.sub_spec = copy(self.spec)
        self.sub_speccorr = copy(self.sub_spec)
        self.ax = ax
        self.labline = self.__assertwavelength__(labline)
        if self.labline > (2 * u.um):  # NIST values greater than 2 microns are in vacuum, need air
            self.labline = vac_to_air(self.labline, method='Edlen1953')
        self.waverms = self.__assertwavelength__(kwargs.get('waverms', 0))  # wavelength rms
        self.rv = self.__assertrv__(kwargs.get('rv', 0))  # RV
        self.rverr = self.__assertrv__(5)
        self.rvstep = self.__assertrv__(kwargs.get('rvstep', 10))  # RV step size
        self.teffunit = u.K
        self.gravunit = u.dex
        self.metunit = u.dex
        self.teff = kwargs.get('teff', 2000) * self.teffunit  # teff
        self.grav = kwargs.get('grav', 5.) * self.gravunit  # gravity
        self.met = kwargs.get('met', 0.) * self.metunit  # metallicity
        self.templatefname = ''
        self.temp_spec = copy(spec)
        self.sub_temp_spec = copy(self.temp_spec)
        self.sub_temp_speccorr = copy(self.sub_temp_spec)
        self.smoothlevel = kwargs.get('smoothlevel', 1)  # smoothing level
        self.gottemplate = False
        self.tempchanged = True
        self.templates_query()
        if not self.gottemplate:
            raise IndexError('Failed to initialise with default teff/ grav/ met')
        self.contfound = False
        self.profilefound = False
        self.use = kwargs.get('use', True)  # whether to use this spectral line or not
        self.conttemplate = None
        self.c1 = kwargs.get('c1', self.spec.spectral_axis.min())  # the left most spectral boundary
        self.c4 = kwargs.get('c4', self.spec.spectral_axis.max())  # the right most spectral boundary
        self.linewindow = self.getlinewindow()
        self.contwindow = self.getcontwindow()
        self.iscut = False
        return

    def reset(self):
        """
        Resetting the data
        """
        self.__init__(copy(self.spec), self.labline, self.spec_index, self.ax, **self.kwargs)

    def __assertteff__(self, value: Optional[Union[float, u.Quantity]]) -> Optional[u.Quantity]:
        if isinstance(value, float) or isinstance(value, int):
            value *= self.teffunit
        elif not self.__assertquantity__(value, True):
            pass
        return value

    def __assertgrav__(self, value: Optional[Union[float, u.Quantity]]) -> Optional[u.Quantity]:
        if isinstance(value, float) or isinstance(value, int):
            value *= self.gravunit
        elif not self.__assertquantity__(value, True):
            pass
        return value

    def __assertmet__(self, value: Optional[Union[float, u.Quantity]]) -> Optional[u.Quantity]:
        if isinstance(value, float) or isinstance(value, int):
            value *= self.metunit
        elif not self.__assertquantity__(value, True):
            pass
        return value

    @property
    def teffunit(self):
        return self._teffunit

    @teffunit.setter
    def teffunit(self, value):
        if not self.__assertquantity__(value, False):
            raise AttributeError('teffunit must be an astropy unit')
        self._teffunit = value

    @property
    def gravunit(self):
        return self._gravunit

    @gravunit.setter
    def gravunit(self, value):
        if not self.__assertquantity__(value, False):
            raise AttributeError('gravunit must be an astropy unit')
        self._gravunit = value

    @property
    def metunit(self):
        return self._metunit

    @metunit.setter
    def metunit(self, value):
        if not self.__assertquantity__(value, False):
            raise AttributeError('metunit must be an astropy unit')
        self._metunit = value

    @property
    def rvstep(self) -> u.Quantity:
        return self._rvstep

    @rvstep.setter
    def rvstep(self, value):
        self._rvstep = self.__assertrv__(value)

    @property
    def teff(self) -> u.Quantity:
        return self._teff

    @teff.setter
    def teff(self, value):
        self._teff = self.__assertteff__(value)

    @property
    def grav(self) -> u.Quantity:
        return self._grav

    @grav.setter
    def grav(self, value):
        self._grav = self.__assertgrav__(value)

    @property
    def met(self) -> u.Quantity:
        return self._met

    @met.setter
    def met(self, value):
        self._met = self.__assertmet__(value)

    @property
    def conttemplate(self) -> Optional[Fittable1DModel]:
        return self._conttemplate

    @conttemplate.setter
    def conttemplate(self, value):
        self._conttemplate = self.__assertmodel__(value)

    def __str__(self):
        s = """
Interactive plotting routine help:
? - Prints this help menu
q - Quit routine (lines are rejected by default)
r - Resets back to default
1 - Selects left hand edge of spectra; anything further left is cut
2 - Selects right hand edge of spectra; anything further right is cut
3 - Decrease smoothing level of template by 1 sigma
4 - Increase smoothing level of template by 1 sigma
5 - Decrease metallicity by 0.5
6 - Increase metallicity by 0.5
7 - Change RV in steps of 5 km/s
8 - Change RV in steps of 10 km/s
9 - Change RV in steps of 100 km/s
right - Increase RV
left - Decrease RV
up - Increase Teff by 100K
down - Decrease Teff by 100K
+ - Increase log g by 0.5
- - Decrease log g by 0.5
y - Accept this line fitting
n - Reject this line fitting
b - Go back to previous line
        """
        return s

    @staticmethod
    def get_template_converter() -> pd.DataFrame:
        """
        Converting the .json of templates into a dataframe

        Returns
        -------
        df
            The dataframe of the template lookup
        """
        jdname = 'template_lookup.json'
        if not os.path.exists(jdname):
            raise FileNotFoundError(f'Need lookup file: {jdname}')
        with open(jdname, 'r') as jd:
            d = json.load(jd)
        df = pd.DataFrame.from_dict(d, 'index', columns=('teff', 'logg', 'met'))
        return df

    def templates_query(self):
        """
        Querying the templates with a given teff, gravity and metallicity
        """
        if self.tempchanged:
            try:
                fname = self.templatedf.loc[(self.templatedf.teff == self.teff.value) &
                                            (self.templatedf.logg == self.grav.value) &
                                            (self.templatedf.met == self.met.value)].iloc[0].name
                fname = self.templatedir + fname
                wave = copy(self.spec.spectral_axis)
                kwargs = dict(wunit=u.AA)
                kwargs['wavearr'] = wave.to(u.AA).value
                temp_spec = freader(fname, **kwargs)
                temp_spec = Spectrum1D(temp_spec.flux, temp_spec.spectral_axis.to(self.wunit),
                                       uncertainty=temp_spec.uncertainty)
                temp_spec = self.cutspec(temp_spec)
            except (IndexError, FileNotFoundError, OSError):
                self.gottemplate = False
                return
            self.templatefname = fname
            self.temp_spec = temp_spec
        self.gottemplate = True
        return

    def shiftsmooth(self, temp_spec: Spectrum1D, wavearr: np.ndarray) -> Spectrum1D:
        """
        Shifting and smoothing the spectrum

        Parameters
        ----------
        temp_spec
            The input spectrum
        wavearr
            Wavelength array to be interpolated to

        Returns
        -------
        temp_spec
            The shifted input spectrum
        """
        temp_spec.radial_velocity = self.rv
        wavetemp, fluxtemp, fluxtemperr = spec_unpack(temp_spec)
        fluxsmooth = convolve(fluxtemp, Gaussian1DKernel(self.smoothlevel))
        fluxsmooth = np.interp(wavearr, wavetemp, fluxsmooth)
        fluxerrsmooth = np.interp(wavearr, wavetemp, fluxtemperr)
        temp_spec = Spectrum1D(fluxsmooth * self.funit, wavearr * self.wunit,
                               uncertainty=StdDevUncertainty(fluxerrsmooth, unit=self.funit))
        self.rverr = self.rvstep / 2 + inv_rv_calc(self.waverms.to(self.wunit).value, self.labline.value) * self.rvunit
        return temp_spec

    def __updatewindows__(self):
        """
        Cutting the spectrum
        """
        super().__updatewindows__()
        self.sub_temp_spec = self.cutspec(self.temp_spec)
        if self.c1 == self.spec.spectral_axis.min() or self.c4 == self.spec.spectral_axis.max():
            self.iscut = False
        else:
            self.iscut = True

    def normalise(self):
        """
        Normalising the spectra
        """
        wave, flux, fluxerr = spec_unpack(self.sub_spec)
        wavetemp, fluxtemp, fluxtemperr = spec_unpack(self.sub_temp_spec)
        wave, flux, fluxerr = normaliser(wave, flux, fluxerr, xmin=self.c1.value, xmax=self.c4.value)
        wavetemp, fluxtemp, fluxtemperr = normaliser(wavetemp, fluxtemp, fluxtemperr,
                                                     xmin=self.c1.value, xmax=self.c4.value)
        self.sub_speccorr = Spectrum1D(flux * self.funit, wave * self.wunit,
                                       uncertainty=StdDevUncertainty(fluxerr, unit=self.funit))
        self.sub_temp_speccorr = Spectrum1D(fluxtemp * self.funit, wavetemp * self.wunit,
                                            uncertainty=StdDevUncertainty(fluxtemperr, unit=self.funit))

    def __fitready__(self):
        """
        A check being made before any plotting
        """
        try:
            self.templates_query()
            if not self.gottemplate:
                raise ValueError('Out of template range')
            self.__updatewindows__()
            self.sub_temp_spec = self.shiftsmooth(self.sub_temp_spec, self.sub_spec.wavelength.to(self.wunit).value)
            self.normalise()
        except Exception as e:
            print(f'Fit failed: {repr(e)}')
            if self.ax is not None:
                self.ax.text(0.5, 0.5, f'Fit failed: {repr(e)}', transform=self.ax.transAxes,
                             horizontalalignment='center')
            self.rescale = False
            self.gottemplate = False
            return
        self.gottemplate = True

    def plotter(self):
        """
        Plotting routine
        """
        self.__fitready__()
        handles, labels = [], []
        if self.iscut:
            spec = self.sub_speccorr
            tempspec = self.sub_temp_speccorr
        else:
            spec = self.spec
            tempspec = self.temp_spec
        wave, flux, fluxerr = spec_unpack(spec)
        wavetemp, fluxtemp = spec_unpack(tempspec)[:2]
        if self.rescale:
            xroundpoint, yroundpoint = 5, 1
            self.ax.set_ylim(yroundpoint * np.floor(np.min(flux) / yroundpoint),
                             yroundpoint * np.ceil(np.max(flux) / yroundpoint))
            self.ax.set_xlim(self.c1.value, self.c4.value)
            self.rescale = False
        if self.iscut:
            ebar = self.ax.errorbar(wave, flux, yerr=fluxerr, marker='s', lw=0, elinewidth=1.5, c='black',
                                    ms=4, mfc='white', mec='black', barsabove=True)
            fitx, fity, fityerr = self.poly_cutter(wave, flux, fluxerr, 5)
            splineplot = self.ax.plot(fitx, fity, 'b-')
            handles.extend([ebar, splineplot[0]])
            labels.extend(['Data Points', 'Data Spline'])
            rmsdiqr, sig = rmsdiqr_check(flux, fluxtemp, self.best_rmsdiqr)
            if sig:
                self.best_rmsdiqr = rmsdiqr
                sigcol = 'green'
            else:
                sigcol = 'red'
            self.ax.text(0.05, 0.95,  f'RMSDIQR = {rmsdiqr:.2f}', transform=self.ax.transAxes, zorder=6,
                         verticalalignment='top', c=sigcol, bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'))
        else:
            p = self.ax.plot(wave, flux, 'k')
            handles.extend(p)
            labels.append('Data')
        self.ax.axvline(self.labline.value, color='grey', ls='--')
        self.ax.axvline(self.labline.value + inv_rv_calc(self.rv.value, self.labline.value),
                        color='black')
        spectitle = self.spec_index.capitalize().replace('1', '\,\\textsc{i}')
        self.ax.set_title('\t' * 2 + f'{spectitle}: RV={self.rv.value:.1f}\,km\,s$^{{-1}}$\n'
                                     f'T={self.teff.value}\,K, $\log g$={self.grav.value}\,dex, '
                                     f'$\\vert$Fe/H$\\vert$={self.met.value}\,dex')
        if not self.use:
            return
        ls = '-'
        templateplot = self.ax.plot(wavetemp, fluxtemp, c='orange', ls=ls)
        handles.extend(templateplot)
        labels.append('Template')
        leg = self.ax.legend(handles, labels)
        leg.set_draggable(True)


def manual_xcorr_fit(spec: Spectrum1D, spec_indices: Dict[str, float], **kwargs) -> Tuple[List[str], Sequence[Xcorr]]:
    """
    Manually fitting the cross correlation for each object

    Parameters
    ----------
    spec
        The spectrum of the object
    spec_indices
        The dictionary of indices
    kwargs
        The fit parameters passed on to the fitting

    Returns
    -------
    useset, objlist
        The list of lines used
        The list of all of the fits
    """
    def keypress(e):
        global curr_pos
        obj = objlist[curr_pos]
        if e.key == 'y':
            goodinds[curr_pos] = True
            curr_pos += 1
            if curr_pos < len(useset):
                objkwargs = obj.kwargs
                objkwargs['teff'] = copy(obj.teff.value)
                objkwargs['grav'] = copy(obj.grav.value)
                objkwargs['met'] = copy(obj.met.value)
                objkwargs['rv'] = copy(obj.rv.value)
                objkwargs['rvstep'] = copy(obj.rvstep.value)
                objlist[curr_pos].kwargs = objkwargs
                objlist[curr_pos].reset()
        elif e.key == 'n':
            goodinds[curr_pos] = False
            curr_pos += 1
            if curr_pos < len(useset):
                objkwargs = obj.kwargs
                objkwargs['teff'] = copy(obj.teff.value)
                objkwargs['grav'] = copy(obj.grav.value)
                objkwargs['met'] = copy(obj.met.value)
                objkwargs['rv'] = copy(obj.rv.value)
                objkwargs['rvstep'] = copy(obj.rvstep.value)
                objlist[curr_pos].kwargs = objkwargs
                objlist[curr_pos].reset()
        elif e.key == 'b':
            curr_pos -= 1 if curr_pos - 1 >= 0 else curr_pos
        elif e.key == 'r':
            obj.reset()
        elif e.key == '1':
            obj.c1 = e.xdata
            obj.rescale = True
        elif e.key == '2':
            obj.c4 = e.xdata
            obj.rescale = True
        elif e.key == '3':
            obj.smoothlevel = obj.smoothlevel - 1 or 1
        elif e.key == '4':
            obj.smoothlevel += 1
        elif e.key == '5':
            obj.met -= 0.5 * u.dex
        elif e.key == '6':
            obj.met += 0.5 * u.dex
        elif e.key == '7':
            obj.rvstep = 5 * rvunit
        elif e.key == '8':
            obj.rvstep = 10 * rvunit
        elif e.key == '9':
            obj.rvstep = 100 * rvunit
        elif e.key == '-':
            obj.grav -= 0.5 * u.dex
        elif e.key in ('+', '='):
            obj.grav += 0.5 * u.dex
        elif e.key == 'up':
            obj.teff += 100 * u.K
        elif e.key == 'down':
            obj.teff -= 100 * u.K
        elif e.key == 'right':
            obj.rv += obj.rvstep
        elif e.key == 'left':
            obj.rv -= obj.rvstep
        elif e.key == '?':
            print(obj)
        elif e.key == 'q':
            plt.close(2)
            return
        else:
            return
        if curr_pos == len(useset):
            plt.close(2)
            return
        if e.key in ('up', 'down', '-', '+', '=', '5', '6'):
            obj.tempchanged = True
        else:
            obj.tempchanged = False
        for artist in plt.gca().lines + plt.gca().collections + plt.gca().texts:
            artist.remove()
        if e.key != 'q':
            try:
                objlist[curr_pos].plotter()
            except Exception as e:
                print(e)
        fig.canvas.draw()
        return

    global curr_pos
    curr_pos = 0
    rvunit = kwargs.get('rvunit', u.km / u.s)
    fig, ax = plt.subplots(figsize=(8, 5), num=2)
    fig: plt.Figure = fig
    ax: plt.Axes = ax
    fig.canvas.mpl_connect('key_press_event', keypress)
    useset = np.fromiter(spec_indices.keys(), dtype='<U8')
    ax.set_xlabel(f'Wavelength\,[{spec.spectral_axis.unit.to_string()}]')
    ax.set_ylabel('Normalised Flux [$F_{\lambda}$]')
    curr_pos = 0
    goodinds = np.zeros(len(useset), dtype=bool)
    objlist = np.empty_like(goodinds, dtype=object)
    for i, (spec_index, labline) in tqdm(enumerate(spec_indices.items()), total=len(spec_indices),
                                         desc='Prepping Cross Correlation', leave=False):
        objlist[i] = Xcorr(copy(spec), labline, spec_index, ax, **kwargs)
    objlist[0].plotter()
    plt.show()
    outset: list = useset[goodinds].tolist()
    return outset, objlist
