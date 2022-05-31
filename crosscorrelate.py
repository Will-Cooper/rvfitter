import pandas as pd

from xcorr import *


def fitparams(useset: List[str], objlist: List[Xcorr]) -> Dict[str, List[Union[float, str, bool]]]:
    dobj = {}
    for obj in objlist:
        key = obj.spec_index
        paramlist = []
        for val in (obj.teff, obj.grav, obj.met):
            if not isinstance(val, str) and val is not None:
                val = float(val.value)
            paramlist.append(val)
        paramlist.append(True if key in useset else False)
        dobj[key] = paramlist
    return dobj


def crosscorrelatepyast(spec_indices: dict, useset: list, lcfit: float, tname: str,
                        goodfits: dict, hires: bool = True, f: str = '') -> Tuple[float, float, np.ndarray, np.ndarray]:
    fig, ax = plt.subplots(num=5, figsize=(4, 3))
    mucorr, correrr = -9999., -9999.
    xcorr, xerr = np.full(len(useset), np.nan), np.full(len(useset), np.nan)
    fullf = f'bt_spectra/useful/{f}.txt'
    wavetemp, fluxtemp = np.loadtxt(fullf, unpack=True, usecols=(0, 1))
    j = -1
    for spec_index, pos in tqdm(spec_indices.items(), total=len(spec_indices.keys()),
                                desc='Fitting Cross Correlation', leave=False):
        if spec_index not in useset:
            continue
        j += 1
        sub_spec: Spectrum1D = goodfits[spec_index][1]
        waveo, fluxo = spec_unpack(sub_spec)[:2]
        wavet = waveo.copy()
        fluxt: np.ndarray = np.interp(waveo, wavetemp, convolve(fluxtemp, Gaussian1DKernel(1)))
        fluxo = (fluxo - np.min(fluxo)) / (np.max(fluxo) - np.min(fluxo))
        fluxt = (fluxt - np.min(fluxt)) / (np.max(fluxt) - np.min(fluxt))
        if hires:
            drv, corrstat = crosscorrRV(wavet, fluxt, waveo, fluxo, -200, 200, 5, skipedge=5)
        else:
            drv, corrstat = crosscorrRV(wavet, fluxt, waveo, fluxo, -200, 200, 5, skipedge=1)
        corrstat = (corrstat - np.min(corrstat)) / (np.max(corrstat) - np.min(corrstat))
        xfit = np.linspace(-200, 200, 401)
        fitter = LevMarLSQFitter(calc_uncertainties=True)
        fake_spec = Spectrum1D(corrstat * u.dimensionless_unscaled, drv * u.km / u.s)
        g_init = models.Gaussian1D(1, 0, 10)
        g_fit = fit_lines(fake_spec, g_init, fitter=fitter)
        yfit = g_fit(xfit * u.km / u.s).value
        ymax = np.max(yfit)
        fitmax = g_fit.mean.value
        xfit2sig = xfit[yfit > .995]
        try:
            fiterr = np.mean(np.abs(np.subtract(xfit2sig[[-1, 0]], fitmax)))
        except IndexError:
            fitmax, fiterr = np.nan, np.nan
        else:
            fitplot = ax.plot(xfit, yfit / ymax, label=spec_index.capitalize())
            ax.axvline(fitmax, color=fitplot[-1].get_color(), ls='--')
        xcorr[j] = fitmax
        xerr[j] = fiterr
    ax.axvline(lcfit, color='black', ls='--', label='Line Center')
    xcorrcut = xcorr[~np.isnan(xcorr)]
    xerrcut = xerr[~np.isnan(xerr)]
    if len(xcorrcut):
        mucorr, stdcorr = ss.norm.fit(xcorrcut)
        if len(xcorrcut) > 1:
            correrr = stdcorr / np.sqrt(len(xcorrcut))
        else:
            correrr = xerrcut[0]
        ax.axvline(mucorr, color='gray', ls='--', label='Mean')
        ax.legend(frameon=True, framealpha=1, facecolor='white')
        ax.set_xlabel('RV [km/s]')
        ax.set_ylabel('Normalised Cross Correlation Stat')
        if hires:
            fig.savefig(f'crosscorrplots/{tname}_R2500I_xcorr.pdf', bbox_inches='tight')
        else:
            fig.savefig(f'crosscorrplots/{tname}_R300R_xcorr.pdf', bbox_inches='tight')
    else:
        plt.clf()
    return mucorr, correrr, xcorr, xerr


def kastcorrelate(spec: Spectrum1D, df: pd.DataFrame, ind: int, spec_indices: dict, useset: list,
                  goodfits: dict, hires: bool) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    tname = df.loc[df['index'] == ind].shortname.iloc[0]
    wave, flux, fluxerr = spec_unpack(spec)
    sp = kast.Spectrum(instr='GTC Osiris', name=tname, wave=wave, flux=flux, unc=fluxerr)
    cols = [s.split('/')[-1].strip('.txt') for s in glob.glob('bt_spectra/useful/lte*txt')]
    dftemps = pd.DataFrame(columns=cols)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    # teff iterating
    bestf = ''
    bestdist: float = np.inf
    gravdict = {4.5: 'dashed', 5.0: 'solid', 5.5: 'dotted'}
    for i, f in tqdm(enumerate(cols), total=len(cols), desc='Convolving templates', leave=False):
        fullf = f'bt_spectra/useful/{f}.txt'
        aps: list = f.split('lte0')[-1].split('-')
        grav = float(aps[1])
        if grav not in gravdict or ~np.isclose(grav, 5):
            continue
        wavetemp, fluxtemp = np.loadtxt(fullf, unpack=True, usecols=(0, 1))
        fluxsmooth: np.ndarray = np.interp(wave, wavetemp, convolve(fluxtemp, Gaussian1DKernel(1)))
        dftemps[f] = fluxsmooth
        sptemp = kast.Spectrum(instr='BT-Settl', name=f, wave=wave, flux=fluxsmooth)
        thisdist = kast.compareSpectra_simple(sp, sptemp, fit_range=[8000, 8500])[0]
        if thisdist < bestdist:
            bestdist = thisdist
            bestf = f

    # gravity iterating
    gravlocation = np.mean([spec_indices['na1-a'], spec_indices['na1-b']])
    gravregions = [gravlocation - 25, gravlocation + 25]
    for grav in gravdict.keys():
        gravstr = str(grav)
        aps = bestf.split('lte0')[-1].split('-')
        gravlast: str = aps[1]
        f: str = bestf.replace(gravlast, gravstr)
        fullf = f'bt_spectra/useful/{f}.txt'
        try:
            wavetemp, fluxtemp = np.loadtxt(fullf, unpack=True, usecols=(0, 1))
            fluxsmooth: np.ndarray = np.interp(wave, wavetemp, convolve(fluxtemp, Gaussian1DKernel(1)))
        except (OSError, FileNotFoundError):
            continue
        dftemps[f] = fluxsmooth
        sptemp = kast.Spectrum(instr='BT-Settl', name=f, wave=wave, flux=fluxsmooth)
        thisdist = kast.compareSpectra_simple(sp, sptemp, fit_range=gravregions)[0]
        if thisdist > bestdist:
            bestdist = thisdist
            bestf = f
    aps = bestf.split('lte0')[-1].split('-')
    grav = float(aps[1])
    teff = float(aps[0]) * 100
    fluxsmooth = dftemps[bestf].values

    # rv calc
    lcfit = df.loc[df['index'] == ind].thisrvlc.iloc[0]
    rvshift, gerr, xcorr, xerr = crosscorrelatepyast(spec_indices, useset,
                                                     lcfit, tname, goodfits, hires, bestf)
    # plotting
    keepplot = True
    if keepplot:
        wavetempplot, fluxtempplot = normaliser(wave - inv_rv_calc(rvshift, wave), fluxsmooth)
        waveplot, fluxplot = normaliser(wave, flux)
        ax.plot(wavetempplot, fluxtempplot,
                label=f'{rvshift:.1f} km/s Shifted BT-Settl {int(teff)}K {grav:.1f} dex Template', color='blue')
        ax.plot(waveplot, fluxplot, label=tname, color='black')
        axsod: plt.Axes = fig.add_axes([0.135, 0.75, 0.1, 0.1])
        axsod.plot(wavetempplot, fluxtempplot, color='blue')
        axsod.plot(waveplot, fluxplot, color='black')
        fluxsod = fluxplot[(waveplot > 8100) & (waveplot < 8200)]
        # axsod.set_title('Na~I Doublet')
        axsod.set_xlim(gravregions[0], gravregions[1])
        axsod.set_ylim(0.95 * np.min(fluxsod), 1.05 * np.max(fluxsod))
        axsod.set_yticks([])
        ax.legend(loc='lower right')
        ax.set_yscale('log')
        ax.set_xlabel(r'Wavelength [$\AA$]')
        ax.set_ylabel(r'Log Normalised Flux [$F_{\lambda}$]')
        if hires:
            fig.savefig(f'templateplots/{tname}_R2500I_template.pdf', bbox_inches='tight')
        else:
            fig.savefig(f'templateplots/{tname}_R300R_template.pdf', bbox_inches='tight')
    else:
        plt.clf()

    # saving
    df.loc[df['shortname'] == tname, 'xcorrteff'] = teff
    df.loc[df['shortname'] == tname, 'xcorrgrav'] = grav
    logging_rvcalc(f'Teff {teff:0g}K, Grav {grav:.1f} logg')
    df.loc[df['shortname'] == tname, 'thisuncorrvxcorr'] = rvshift
    df.loc[df['shortname'] == tname, 'thisrvxcorrerr'] = gerr
    corrected_rvshift = round(rvshift, 1)
    logging_rvcalc(f'Cross correlated corrected RV {corrected_rvshift:.1f} +/- {gerr:.1f} km/s')
    df.loc[df['shortname'] == tname, 'thisrvxcorr'] = corrected_rvshift
    return df, xcorr, xerr


def crosscorrelate(df: pd.DataFrame):
    return df
