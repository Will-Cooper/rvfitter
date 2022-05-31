from astropy.convolution import convolve, Gaussian1DKernel
from astropy.io.fits import getdata
from astropy.nddata import StdDevUncertainty
import pandas as pd
from PyAstronomy.pyasl import crosscorrRV
from matplotlib import rcParams
from matplotlib.ticker import AutoMinorLocator
import scipy.stats as ss
from specutils.utils.wcs_utils import vac_to_air
from splat import Spectrum, measureIndexSet
from tqdm import tqdm

import argparse
from collections import defaultdict
import glob
import json
from os import PathLike
from warnings import simplefilter

import kastredux.core as kast
from splot import *


def inv_rv_calc(shift: float, wave: np.ndarray) -> np.ndarray:
    c = 299792458 / 1e3
    cair = c / 1.000276
    return shift * wave / cair


def tabquery(fname: str, df: pd.DataFrame) -> Optional[pd.Series]:
    fnameonly = fname.split('/')[-1]
    if 'R2500I' in fname:
        thisres = 'R2500I'
        tgtname = fnameonly[: fnameonly.find('_tellcorr.fits')]
    elif 'R300R' in fname:
        thisres = 'R0300R'
        tgtname = fnameonly[: fnameonly.find('.fits')]
    else:
        thisres = ''
        tgtname = fnameonly
    try:
        row: pd.Series = df[(df.target == tgtname) & (df.resolution == thisres)].copy().iloc[0]
    except IndexError:  # file not in table (i.e. standard)
        return None
    return row


def get_ew(wave: np.ndarray, flux: np.ndarray, wavebounds: np.ndarray, cont: Fittable1DModel):
    regbool = (wave > wavebounds[0]) & (wave < wavebounds[1])
    regwave = wave[regbool]
    regflux = flux[regbool]
    continuum = cont(regwave * u.AA).value
    contcorr = np.divide(regflux, continuum)
    ew = np.trapz(np.ones(len(regwave)) - contcorr, regwave)
    return ew


def get_indices(ind: int, sp: Spectrum, dflines: pd.DataFrame) -> pd.DataFrame:
    for ref in ('kirkpatrick', 'martin'):
        measured: dict = measureIndexSet(sp, ref=ref)
        for key in measured:
            val = measured[key][0]
            if not np.isnan(val):
                dflines.loc[dflines['index'] == ind, key] = round(val, 1)
    return dflines


def json_handle(jf: PathLike[str],
                d: Dict[str, List[Union[float, str]]] = None) -> Dict[str, Dict[str, List[Union[float, str]]]]:
    if d is None:
        with open(jf, 'r') as jd:
            d = json.load(jd)
    else:
        with open(jf, 'w') as jd:
            json.dump(d, jd)
    return d


def fitparams(useset: List[str], objlist: List[Splot]) -> Dict[str, List[Union[float, str]]]:
    dobj = {}
    for obj in objlist:
        key = obj.spec_index
        paramlist = []
        for val in (obj.c1, obj.c2, obj.r1, obj.r2, obj.c3, obj.c4, obj.mu, obj.std, obj.amplitude,
                    obj.x_0, obj.fwhm_G, obj.fwhm_L, obj.fwhm_V, obj.line_profile):
            if not isinstance(val, str) and val is not None:
                val = float(val.value)
            paramlist.append(val)
        paramlist.append(True if key in useset else False)
        dobj[key] = paramlist
    return dobj


def interactive_loop(spec: Spectrum1D, spec_indices: Dict[str, float],
                     fname: str) -> Tuple[List[str], List[Splot]]:
    dout = json_handle('lines_used.json')
    args = (spec, spec_indices)
    outset, objlist = interactive_fit(*args)
    dobj = fitparams(outset, objlist)
    dout[fname] = dobj
    json_handle('lines_used.json', dout)
    return outset, objlist


def find_lc_values(spec: Spectrum1D, fname: str, useset: list, spec_indices: Dict[str, float],
                   objlist: List[Splot], df: pd.DataFrame, dflines: pd.DataFrame, ind: int,
                   hires: bool) -> Tuple[pd.DataFrame, pd.DataFrame, dict,
                                         np.ndarray, np.ndarray, list]:
    fig, axs = plt.subplots(4, 2, figsize=(8, 4))
    fig: plt.Figure = fig
    axs: np.ndarray = axs.flatten()
    uncor_rv_list, uncor_err_list = np.full(len(useset), np.nan), np.full(len(useset), np.nan)
    uncor_rv, uncor_err = -9999., -9999.
    wave, flux, fluxerr = spec_unpack(spec)
    sp = Spectrum(flux=flux, noise=fluxerr, wave=wave,
                  wave_unit=u.Angstrom, flux_unit=(u.erg / u.cm ** 2 / u.Angstrom / u.s))
    linewidths = defaultdict(float)
    ews = defaultdict(float)
    goodfits = defaultdict(list)
    j = -1
    for i, spec_index in tqdm(enumerate(spec_indices), total=len(spec_indices.keys()),
                              desc='Fitting Line Centers', leave=False):
        ax: plt.Axes = axs[i]
        obj = objlist[i]
        obj.ax = ax
        obj.plotter()
        if spec_index not in useset:
            continue
        j += 1
        goodfits[spec_index] = [obj.fitted_profile, obj.sub_spec, obj.cont]
        logging_rvcalc(f'{spec_index.capitalize()} -- {obj.line_profile.capitalize()} Profile'
                       f' with {obj.std.value:.1f}A sigma.')
        uncor_rv_list[j] = obj.rv.value
        uncor_err_list[j] = obj.rverr.value
        if obj.linewidth is not None:
            lineedges = [obj.lineedges[0].value, obj.lineedges[1].value]
            linewidths[spec_index] = obj.linewidth.value
            ews[spec_index] = get_ew(obj.sub_spec.spectral_axis.value, obj.sub_spec.flux.value,
                                     lineedges, obj.cont)

    tname = df.loc[df['index'] == ind].shortname.iloc[0]
    uncor_rv_list_cut = uncor_rv_list[~np.isnan(uncor_rv_list)]
    uncor_err_list_cut = uncor_err_list[~np.isnan(uncor_err_list)]
    if len(uncor_rv_list_cut):
        uncor_rv, uncor_std = ss.norm.fit(uncor_rv_list_cut)
        if len(uncor_rv_list_cut) > 1:
            uncor_err = uncor_std / np.sqrt(len(uncor_rv_list_cut))
        else:
            uncor_err = uncor_err_list_cut[0]
        df.loc[df['shortname'] == tname, 'thisuncorlc'] = round(uncor_rv, 1)
        df.loc[df['shortname'] == tname, 'thisrvlcerr'] = round(uncor_err, 1)
    else:
        logging_rvcalc('Empty RV list for line centre calculation!')
    cor_rv = round(uncor_rv, 1)
    logging_rvcalc(f'RV Line Centre Corrected = {cor_rv:.1f} +/- {uncor_err:.1f}km/s')
    df.loc[df['shortname'] == tname, 'thisrvlc'] = round(cor_rv, 1)

    dflines = get_indices(ind, sp, dflines)
    fig.supxlabel(r'RV Shift [km/s]')
    fig.supylabel(r'Normalised Flux [$F_{\lambda}$]')
    fig.subplots_adjust(hspace=0.95)
    fname = 'rvplots/' + fname.split('/')[-1]
    if hires:
        fname = '.'.join(fname.split('.')[:-1]) + 'R2500I_rv.pdf'
    else:
        fname = '.'.join(fname.split('.')[:-1]) + 'R300R_rv.pdf'
    plt.savefig(fname, bbox_inches='tight')
    for key in linewidths:
        dflines.loc[dflines['index'] == ind, key.replace('-', '_') + '_width'] = round(linewidths[key], 1)
    for key in ews:
        dflines.loc[dflines['index'] == ind, key.replace('-', '_') + '_ew'] = round(ews[key], 1)
    return df, dflines, goodfits, uncor_rv_list, uncor_err_list, useset


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


def normaliser(x: np.ndarray, *args, xmin: float = 8100, xmax: float = 8200):
    boolcut: np.ndarray = (x > xmin) & (x < xmax)
    args = list(args)
    if np.any([len(x) != len(arg) for arg in args]):
        raise IndexError('check input shapes')
    normval: float = np.nanmedian(args[0][boolcut])
    for i, val in enumerate(args):
        args[i] /= normval
    out = [x, ] + args
    return out


def freader(f: str) -> Spectrum1D:
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
    wunit = u.AA
    funit = u.erg / u.cm ** 2 / u.Angstrom / u.s
    wave, flux, fluxerr = normaliser(wave, flux, fluxerr)
    spec = Spectrum1D(flux * funit, wave * wunit,
                      uncertainty=StdDevUncertainty(fluxerr, unit=funit))
    return spec


def logging_rvcalc(s: str = '', perm: str = 'a'):
    with open('calculating.log', perm) as f:
        f.write(s + '\n')
    return


def chekres(fname: str) -> bool:
    if 'R300R' in fname:
        hires = False
    else:
        hires = True
    return hires


def adoptedrv(df: pd.DataFrame, ind: int, hires: bool, lcvals: np.ndarray, lcerr: np.ndarray,
              xcorr: np.ndarray, xerr: np.ndarray, spec_indices: dict, useset: list) -> pd.DataFrame:
    fig: plt.Figure = plt.figure(figsize=(4, 3))
    axlines: plt.Axes = fig.add_axes([0.1, 0.4, 0.8, 0.5])
    axpdf: plt.Axes = fig.add_axes([0.1, 0.1, 0.8, 0.3])
    allindices = np.array(list(spec_indices.keys()))
    indicesplot = [specindex.capitalize() + r' $\lambda$'
                   + f'{int(pos)}' + r'$\AA$' for specindex, pos in spec_indices.items()]
    useset = np.array(useset)
    ypos = np.arange(len(allindices)) + 1
    indicesbool = np.isin(allindices, useset)
    lcplot = np.full_like(ypos, np.nan, dtype=float)
    lcploterr = np.full_like(ypos, np.nan, dtype=float)
    xplot = np.full_like(ypos, np.nan, dtype=float)
    xploterr = np.full_like(ypos, np.nan, dtype=float)
    lcplot[indicesbool] = lcvals
    lcploterr[indicesbool] = lcerr
    xplot[indicesbool] = xcorr
    xploterr[indicesbool] = xerr
    xcorr = xcorr[~np.isnan(xcorr)]
    xerr = xerr[~np.isnan(xerr)]
    lcvals = lcvals[~np.isnan(lcvals)]
    lcerr = lcerr[~np.isnan(lcerr)]
    locx, scalex = ss.norm.fit(xcorr)
    loclc, scalelc = ss.norm.fit(lcvals)
    if len(lcvals) == 1:
        scalelc = lcerr[0]
    if len(xcorr) == 1:
        scalex = xerr[0]
    minlc, maxlc = np.min(lcvals - lcerr), np.max(lcvals + lcerr)
    minxc, maxxc = np.min(xcorr - xerr), np.max(xcorr + xerr)
    minboth, maxboth = np.min([minlc, minxc]), np.max([maxlc, maxxc])
    minpos = np.floor(minboth / 5) * 5
    maxpos = np.ceil(maxboth / 5) * 5
    pdfxpoints = np.linspace(minpos, maxpos, int(maxpos - minpos + 1))
    xcorrpdf = ss.norm.pdf(pdfxpoints, loc=locx, scale=scalex)
    lcpdf = ss.norm.pdf(pdfxpoints, loc=loclc, scale=scalelc)
    posteriorpdf = xcorrpdf * lcpdf
    locpost = (locx * scalelc ** 2 + loclc * scalex ** 2) / (scalex ** 2 + scalelc ** 2)
    stdpost = np.sqrt(scalex ** 2 * scalelc ** 2 / (scalex ** 2 + scalelc ** 2))
    errpost = stdpost / np.sqrt(len(xcorr) + len(lcvals))

    xcorrpdf /= np.max(xcorrpdf)
    lcpdf /= np.max(lcpdf)
    posteriorpdf /= np.max(posteriorpdf)

    axlines.errorbar(xplot, ypos - .1, xerr=xploterr, marker='s', ms=6, color='orange', lw=0, elinewidth=1.5)
    axlines.errorbar(lcplot, ypos + .1, xerr=lcploterr, marker='d', ms=6, color='blue', lw=0, elinewidth=1.5)
    axlines.set_yticks(ypos)
    axlines.set_yticklabels(indicesplot, fontsize='medium')
    axlines.set_ylim(0, ypos[-1] + 1)
    axlines.set_xticks([])
    axlines.yaxis.set_minor_locator(AutoMinorLocator(1))
    axlines.set_xlim(minpos, maxpos)
    axlines.set_ylabel('Spectral Feature', fontsize='medium')

    axpdf.plot(pdfxpoints, xcorrpdf, color='orange', label=rf'${locx:.1f}\pm{scalex / np.sqrt(len(xcorr)):.1f}$ km/s')
    [ax.axvline(locx, color='orange', ls='--', lw=0.75) for ax in (axpdf, axlines)]
    axpdf.plot(pdfxpoints, lcpdf, color='blue', label=rf'${loclc:.1f}\pm{scalelc / np.sqrt(len(lcvals)):.1f}$ km/s')
    [ax.axvline(loclc, color='blue', ls='--', lw=0.75) for ax in (axpdf, axlines)]
    axpdf.plot(pdfxpoints, posteriorpdf, color='black', label=rf'${locpost:.1f}\pm{errpost:.1f}$ km/s')
    [ax.axvline(locpost, color='black', ls='--', lw=0.75) for ax in (axpdf, axlines)]
    axpdf.set_yticks([0, 0.5, 1])
    axpdf.set_ylim(-0.1, 1.1)
    axpdf.set_xlim(minpos, maxpos)
    axpdf.set_xlabel('RV [km/s]', fontsize='medium')
    axpdf.set_ylabel('Normalised PDF', fontsize='medium')

    tname = df.loc[df['index'] == ind].shortname.iloc[0]
    if hires:
        fig.savefig(f'adoptedrvplots/{tname}_R2500I_adoptedrv.pdf', bbox_inches='tight')
    else:
        fig.savefig(f'adoptedrvplots/{tname}_R300R_adoptedrv.pdf', bbox_inches='tight')

    df.loc[df['shortname'] == tname, 'thisrv'] = locpost
    df.loc[df['shortname'] == tname, 'thisrverr'] = errpost
    logging_rvcalc(f'Adopted RV {locpost:.1f} +/- {errpost:.1f} km/s')
    return df


def load_fitinfo(spec: Spectrum1D, spec_indices: Dict[str, float],
                 fname: str, repeat: bool) -> Tuple[List[str], List[Splot]]:
    d = json_handle('lines_used.json')
    if repeat or fname not in d.keys():
        useset, objlist = interactive_loop(spec, spec_indices, fname)
    else:
        dobj = d[fname]
        useset, objlist = [], []
        # obj.c1, obj.c2, obj.r1, obj.r2, obj.c3, obj.c4, obj.mu, obj.std, obj.amplitude,
        # obj.x_0, obj.fwhm_G, obj.fwhm_L, obj.fwhm_V, obj.line_profile
        paramlist = ('c1', 'c2', 'r1', 'r2', 'c3', 'c4', 'mu', 'std', 'amplitude',
                     'x_0', 'fwhm_G', 'fwhm_L', 'fwhm_V', 'line_profile', 'use')
        for spec_index, objinfo in tqdm(dobj.items(), total=len(dobj.keys()),
                                        desc='Loading Fits', leave=False):
            if objinfo[-1]:
                useset.append(spec_index)
            labline = spec_indices[spec_index]
            kwargs = {param: objinfo[i] for i, param in enumerate(paramlist)}
            obj = Splot(spec, labline, spec_index, **kwargs)
            objlist.append(obj)
    return useset, objlist


def main(fname, spec_indices, df, dflines, repeat):
    spec = freader(fname)
    hires = chekres(fname)
    row = tabquery(fname, df)
    if row is None:
        ind = None
    else:
        ind = row['index']
    if ind is None:
        return df, dflines
    logging_rvcalc('\n' + fname.split('/')[-1].strip('_tellcorr.fits'))
    useset, objlist = load_fitinfo(spec, spec_indices, fname, repeat)
    if not len(useset):
        return df, dflines
    dfout, dflines, goodfits, lcvals, lcerr, useset = find_lc_values(spec, fname, useset, spec_indices, objlist,
                                                                     df, dflines, ind, hires)
    dfout, xcorr, xerr = kastcorrelate(spec, dfout, ind, spec_indices, useset,
                                       goodfits, hires)
    if len(lcvals) and len(xcorr) and len(lcvals) == len(xcorr):
        dfout = adoptedrv(dfout, ind, hires, lcvals, lcerr, xcorr, xerr, spec_indices, useset)
    return dfout, dflines


if __name__ == '__main__':
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
    _spec_indices = {'k1-a': 7664.8991, 'k1-b': 7698.9645,
                     'rb1-a': 7800.27, 'rb1-b': 7947.60,
                     'na1-a': 8183.256, 'na1-b': 8194.824,
                     'cs1-a': 8521.13, 'cs1-b': 8943.47}
    allinds = list(_spec_indices.keys())
    simplefilter('ignore', np.RankWarning)  # a warning about poorly fitting polynomial, ignore
    tabname = 'Master_info_correct_cm_edr3.csv'
    myargs = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    myargs.add_argument('-f', '--file-name', required=True, help='File to be plotted')
    myargs.add_argument('-r', '--repeat', action='store_true', default=False)
    sysargs = myargs.parse_args()
    _fname: str = sysargs.file_name
    _repeat: bool = sysargs.repeat
    _df: pd.DataFrame = pd.read_csv(tabname)
    _dflines: pd.DataFrame = pd.read_csv(tabname.replace('.csv', '_lines.csv'))
    _df.rename(columns={col: col.lower() for col in _df.columns}, inplace=True)
    _df, _dflines = main(_fname, _spec_indices, _df, _dflines, _repeat)
    _dflines.to_csv(tabname.replace('.csv', '_lines.csv'), index=False)
