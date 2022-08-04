from astropy.convolution import convolve, Gaussian1DKernel
import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import Sequence

from utils import *

curr_pos = 0
rcParams['keymap.back'].remove('left')
rcParams['keymap.forward'].remove('right')


class Xcorr(Quantiser):

    def __init__(self, spec: Spectrum1D, labline: Union[float, u.Quantity],
                 spec_index: str, ax: plt.Axes = None, **kwargs):
        self.kwargs = kwargs
        wunit = kwargs.get('wunit', u.AA)
        funit = kwargs.get('funit', u.erg / u.cm ** 2 / u.Angstrom / u.s)
        rvunit = kwargs.get('rvunit', u.km / u.s)
        self.spec_index = spec_index
        super().__init__(wunit, funit, rvunit, spec)
        self.templatedir = kwargs.get('templatedir', 'bt-settl-cifist/useful/')
        self.templatedf = self.get_template_converter()
        self.spec = copy(spec)
        self.sub_spec = copy(self.spec)
        self.sub_speccorr = copy(self.sub_spec)
        self.ax = ax
        self.labline = self.__assertwavelength__(labline)
        self.rv = self.__assertrv__(kwargs.get('rv', 0))
        self.rverr = self.__assertrv__(5)
        self.rvstep = self.__assertrv__(kwargs.get('rvstep', 10))
        self.teffunit = u.K
        self.gravunit = u.dex
        self.metunit = u.dex
        self.teff = kwargs.get('teff', 2000) * self.teffunit
        self.grav = kwargs.get('grav', 5.) * self.gravunit
        self.met = kwargs.get('met', 0.) * self.metunit
        self.templatefname = ''
        self.temp_spec = copy(spec)
        self.sub_temp_spec = copy(self.temp_spec)
        self.sub_temp_speccorr = copy(self.sub_temp_spec)
        self.smoothlevel = kwargs.get('smoothlevel', 1)
        self.gottemplate = False
        self.tempchanged = True
        self.templates_query()
        if not self.gottemplate:
            raise IndexError('Failed to initialise with default teff/ grav/ met')
        self.contfound = False
        self.profilefound = False
        self.use = kwargs.get('use', True)
        self.conttemplate = None
        self.c1 = kwargs.get('c1', self.spec.spectral_axis.min())
        self.c4 = kwargs.get('c4', self.spec.spectral_axis.max())
        self.linewindow = self.getlinewindow()
        self.contwindow = self.getcontwindow()
        self.iscut = False
        return

    def reset(self):
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
        jdname = 'template_lookup.json'
        if not os.path.exists(jdname):
            raise FileNotFoundError(f'Need lookup file: {jdname}')
        with open(jdname, 'r') as jd:
            d = json.load(jd)
        df = pd.DataFrame.from_dict(d, 'index', columns=('teff', 'logg', 'met'))
        return df

    def templates_query(self):
        if self.tempchanged:
            try:
                fname = self.templatedf.loc[(self.templatedf.teff == self.teff.value) &
                                            (self.templatedf.logg == self.grav.value) &
                                            (self.templatedf.met == self.met.value)].iloc[0].name
                fname = self.templatedir + fname
                kwargs = self.kwargs
                kwargs['wavearr'] = self.spec.spectral_axis.value
                temp_spec = self.cutspec(freader(fname, **self.kwargs))
            except (IndexError, FileNotFoundError, OSError):
                self.gottemplate = False
                return
            self.templatefname = fname
            self.temp_spec = temp_spec
        self.gottemplate = True
        return

    def shiftsmooth(self, temp_spec: Spectrum1D):
        temp_spec.radial_velocity = self.rv
        wavetemp, fluxtemp, fluxtemperr = spec_unpack(temp_spec)
        fluxsmooth = convolve(fluxtemp, Gaussian1DKernel(self.smoothlevel))
        temp_spec = Spectrum1D(fluxsmooth * self.funit, wavetemp * self.wunit,
                               uncertainty=StdDevUncertainty(fluxtemperr, unit=self.funit))
        self.rverr = self.rvstep / 2
        return temp_spec

    def __updatewindows__(self):
        super().__updatewindows__()
        self.sub_temp_spec = self.cutspec(self.temp_spec)
        if self.c1 == self.spec.spectral_axis.min() or self.c4 == self.spec.spectral_axis.max():
            self.iscut = False
        else:
            self.iscut = True

    def normalise(self):
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
        try:
            self.templates_query()
            if not self.gottemplate:
                raise ValueError('Out of template range')
            self.__updatewindows__()
            self.sub_temp_spec = self.shiftsmooth(self.sub_temp_spec)
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
            self.ax.set_xlim(np.min(wave), np.max(wave))
            self.ax.set_ylim(*np.array([np.floor(np.min(flux)),
                                        np.ceil(np.max(flux))]), )
            self.rescale = False
        if self.iscut:
            ebar = self.ax.errorbar(wave, flux, yerr=fluxerr, marker='s', lw=0, elinewidth=1.5, c='black',
                                    ms=4, mfc='white', mec='black', barsabove=True)
            fitx, fity, fityerr = self.poly_cutter(wave, flux, fluxerr, 5)
            splineplot = self.ax.plot(fitx, fity, 'b-')
            handles.extend([ebar, splineplot[0]])
            labels.extend(['Data Points', 'Data Spline'])
        else:
            p = self.ax.plot(wave, flux, 'k')
            handles.extend(p)
            labels.append('Data')
        self.ax.axvline(self.labline.value, color='grey', ls='--')
        self.ax.axvline(self.labline.value + inv_rv_calc(self.rv.value, self.labline.value),
                        color='black', ls='--')
        self.ax.set_title('\t' * 2 + f'{self.spec_index.capitalize()}: {self.teff.value}\,K, '
                                     f'{self.grav.value}\,dex, {self.met.value}\,dex, {self.rv.value:.1f}\,km/s')
        if not self.use:
            return
        ls = '-'
        templateplot = self.ax.plot(wavetemp, fluxtemp, c='orange', ls=ls)
        handles.extend(templateplot)
        labels.append('Template')
        self.ax.legend(handles, labels)


def manual_xcorr_fit(spec: Spectrum1D, spec_indices: Dict[str, float], **kwargs) -> Tuple[List[str], Sequence[Xcorr]]:
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
    ax.set_xlabel(r'Wavelength [$\AA$]')
    ax.set_ylabel('Normalised Flux')
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
