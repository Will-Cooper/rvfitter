from astropy.modeling import models
import matplotlib.pyplot as plt
import pandas as pd
from specutils.fitting import fit_continuum

from utils import *

curr_pos = 0


class Xcorr(Quantiser):

    def __init__(self, spec: Spectrum1D, labline: Union[float, u.Quantity],
                 spec_index: str, ax: plt.Axes = None, **kwargs):
        wunit = kwargs.get('wunit', u.AA)
        funit = kwargs.get('funit', u.erg / u.cm ** 2 / u.Angstrom / u.s)
        rvunit = kwargs.get('rvunit', u.km / u.s)
        self.spec_index = spec_index
        super().__init__(wunit, funit, rvunit, spec)
        self.kwargs = kwargs
        self.templatedir = kwargs.get('templatedir', 'bt_spectra/useful/')
        self.templatedf = self.get_template_converter()
        self.spec = spec
        self.sub_spec = self.spec
        self.sub_speccorr = self.sub_spec
        self.temp_spec = None
        self.sub_temp_spec = self.temp_spec
        self.sub_temp_speccorr = self.sub_temp_spec
        self.ax = ax
        self.labline = self.__assertwavelength__(labline)
        self.rv = self.__assertrv__(np.nan)
        self.rverr = self.__assertrv__(np.nan)
        self.teffunit = self.__assertquantity__(u.K, False)
        self.gravunit = self.__assertquantity__(u.dex, False)
        self.metunit = self.__assertquantity__(u.dex, False)
        self.teff: u.Quantity = kwargs.get('teff', 2000) * self.teffunit
        self.grav: u.Quantity = kwargs.get('grav', 5.) * self.gravunit
        self.met: u.Quantity = kwargs.get('met', 0.) * self.metunit
        self.templatefname = ''
        self.gottemplate = self.templates_query()
        if not self.gottemplate:
            raise IndexError('Failed to initialise with default teff/ grav/ met')
        self.contfound = False
        self.profilefound = False
        self.use = kwargs.get('use', True)
        self.linewindow = self.getlinewindow()
        self.contwindow = self.getcontwindow()
        return

    def reset(self):
        self.__init__(self.spec, self.labline, self.spec_index, self.ax, **self.kwargs)

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
        try:
            fname = self.templatedf.loc[(self.templatedf.teff == self.teff.value) &
                                        (self.templatedf.logg == self.grav.value) &
                                        (self.templatedf.met == self.met.value)].iloc[0].name
            fname = self.templatedir + fname
            temp_spec = freader(fname, **self.kwargs)
        except (IndexError, FileNotFoundError, OSError):
            return False
        self.templatefname = fname
        self.temp_spec = temp_spec
        return True

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

    def plotter(self):
        if self.iscut:
            spec = self.sub_spec
            tempspec = self.sub_temp_spec
        else:
            spec = self.spec
            tempspec = self.temp_spec
        wave, flux, fluxerr = spec_unpack(spec)
        wavetemp, fluxtemp = spec_unpack(tempspec)[:2]
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
        contyval = self.cont(fitx * self.wunit).value
        self.ax.plot(fitx, contyval, c='k', ls=ls)
        self.ax.fill_betweenx([np.min(fity), np.max(fity)], self.c1.value, self.c2.value,
                              color='grey', alpha=0.25)
        self.ax.fill_betweenx([np.min(fity), np.max(fity)], self.c3.value, self.c4.value,
                              color='grey', alpha=0.25)
        fityval = np.interp(fitx, wavetemp, fluxtemp)
        if not self.profilefound:
            return
        self.ax.plot(fitx, fityval, c='orange', ls=ls)
        if self.rescale:
            self.ax.fill_betweenx([np.min(fity), np.max(fity)], self.r1.value, self.r2.value,
                                  color='grey', alpha=0.5)
            self.ax.set_ylim(0.1 * np.floor(np.min(fityval) / 0.1), 0.1 * np.ceil(np.max(fityval) / 0.1))
            self.ax.set_xlim(self.c2.value, self.c3.value)


def interactive_fit(spec: Spectrum1D, spec_indices: Dict[str, float], **kwargs) -> Tuple[List[str], np.ndarray[Xcorr]]:
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
        objlist[i] = Xcorr(spec, labline, spec_index, ax, **kwargs)
    objlist[0].plotter()
    plt.show()
    outset: list = useset[goodinds].tolist()
    return outset, objlist


def spec_unpack(spec: Spectrum1D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wave = spec.spectral_axis.value
    flux = spec.flux.value
    fluxerr = spec.uncertainty.quantity.value
    return wave, flux, fluxerr
