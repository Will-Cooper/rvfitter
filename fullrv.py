from astropy.io.fits import getheader
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import scipy.stats as ss
from splat import Spectrum, measureIndexSet

import argparse
from collections import OrderedDict
import sys
from typing import Sequence
from warnings import simplefilter

sys.path.insert(0, 'rvfitter/')
from utils import *
from linecentering import linecentering
from crosscorrelate import crosscorrelate


def tabquery(fname: str, df: pd.DataFrame) -> Optional[pd.Series]:
    fnameonly = fname.split('/')[-1]
    if 'R2500I' in fname:
        thisres = 'R2500I'
        if 'tellcorr' in fnameonly:
            tgtname = fnameonly[: fnameonly.find('_tellcorr.fits')]
        else:
            tgtname = fnameonly[: fnameonly.find('.fits')]
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


def get_indices(tname: str, colname: str, fname: str, dflines: pd.DataFrame) -> pd.DataFrame:
    spec = freader(fname)
    wave, flux, fluxerr = spec_unpack(spec)
    sp = Spectrum(wave=wave, flux=flux, wave_unit=spec.spectral_axis.unit, flux_unit=spec.flux.unit)
    for ref in ('kirkpatrick', 'martin'):
        measured: dict = measureIndexSet(sp, ref=ref)
        for key in measured:
            val = measured[key][0]
            if not np.isnan(val):
                dflines.loc[dflines[colname] == tname, key] = round(val, 1)
    return dflines


def chekres(fname: str) -> bool:
    if 'R300R' in fname:
        hires = False
    else:
        hires = True
    return hires


def adoptedrv(df: pd.DataFrame, colname: str, tname: str, hires: bool, lcvals: Sequence[float], lcerr: Sequence[float],
              xcorr: Sequence[float], xerr: Sequence[float], spec_indices: dict) -> pd.DataFrame:
    fig: plt.Figure = plt.figure(figsize=(4, 3), num=5)
    axlines: plt.Axes = fig.add_axes([0.1, 0.4, 0.8, 0.5])
    axpdf: plt.Axes = fig.add_axes([0.1, 0.1, 0.8, 0.3])
    allindices = np.array(list(spec_indices.keys()))
    indicesplot = np.array([specindex.capitalize() + r' $\lambda$'
                           + f'{int(pos)}' + u.AA.to_string(u.format.Latex)
                            for specindex, pos in spec_indices.items()])
    ypos = np.arange(len(allindices)) + 1
    lcplot = copy(lcvals)
    lcploterr = copy(lcerr)
    xplot = copy(xcorr)
    xploterr = copy(xerr)
    xcorr = xcorr[~np.isnan(xcorr)]
    xerr = xerr[~np.isnan(xerr)]
    lcvals = lcvals[~np.isnan(lcvals)]
    lcerr = lcerr[~np.isnan(lcerr)]
    if not len(lcvals) or not len(xcorr):
        plt.close(5)
        return df
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
    axpdf.set_ylabel('PDF', fontsize='medium')

    if hires:
        fig.savefig(f'adoptedrvplots/{tname}_R2500I_adoptedrv.pdf', bbox_inches='tight')
    else:
        fig.savefig(f'adoptedrvplots/{tname}_R300R_adoptedrv.pdf', bbox_inches='tight')

    df.loc[df[colname] == tname, 'thisrv'] = locpost
    df.loc[df[colname] == tname, 'thisrverr'] = errpost
    logging_rvcalc(f'Adopted RV {locpost:.1f} +/- {errpost:.1f} km/s')
    return df


def errorappend(df: pd.DataFrame, tname: str, colname: str):
    teffstd = df.loc[df[colname] == tname, 'xcorrtefferr'].iloc[0]
    gravstd = df.loc[df[colname] == tname, 'xcorrgraverr'].iloc[0]
    if not np.isnan(df.loc[df[colname] == tname, 'xcorrtefferr'].iloc[0]):
        df.loc[df[colname] == tname, 'xcorrtefferr'] = np.sqrt(teffstd ** 2 + 50 ** 2)
        df.loc[df[colname] == tname, 'xcorrgraverr'] = np.sqrt(gravstd ** 2 + 0.25 ** 2)
    return df


def getwaverms(fname: str) -> float:
    wvcalibf = fname[:fname.find('Science')] + 'wvcalib.fits'
    head: Dict[str, float] = getheader(wvcalibf, 2)
    disp: float = head['CEN_DISP']  # central dispersion in wave/ pix
    rms: float = head['RMS']  # rms in pixels
    return disp * rms


def main(fname, spec_indices, df, dflines, repeat):
    spec_indices = OrderedDict(spec_indices)
    hires = chekres(fname)
    if hires:
        fappend = 'R2500I'
    else:
        fappend = 'R300R'
    row = tabquery(fname, df)
    if row is not None:
        ind = row['index']
    else:
        return df, dflines
    colname = 'shortname'
    tname = df.loc[df['index'] == ind, colname].iloc[0]
    expectedteff = stephens(df.loc[df['index'] == ind].kasttypenum.iloc[0] - 60)
    if np.isnan(expectedteff):
        expectedteff = 2000
    else:
        expectedteff = 100 * round(expectedteff / 100)
        if 1200 > expectedteff > 4000:
            expectedteff = 2000
    logging_rvcalc(f'\n{tname}')
    dfout = df
    waverms = getwaverms(fname)
    if hires:
        dfout, lcvals, lcerr = linecentering(fname, spec_indices, dfout, repeat, tname, colname, fappend,
                                             waverms=waverms)
    else:
        lcvals, lcerr = np.full(len(spec_indices), np.nan), np.full(len(spec_indices), np.nan)
    dfout, xcorr, xerr = crosscorrelate(fname, spec_indices, dfout, repeat, tname, colname, fappend,
                                        teff=expectedteff, dorv=hires, waverms=waverms)
    dfout = errorappend(dfout, tname, colname)
    dfout = adoptedrv(dfout, colname, tname, hires, lcvals, lcerr, xcorr, xerr, spec_indices)
    dflines = get_indices(tname, colname, fname, dflines)
    return dfout, dflines


if __name__ == '__main__':
    _spec_indices = {'k1-a': 7664.8991, 'k1-b': 7698.9646,
                     'rb1-a': 7800.268, 'rb1-b': 7947.603,
                     'na1-a': 8183.2556, 'na1-b': 8194.8240,
                     'cs1-a': 8521.13165, 'cs1-b': 8943.47424}  # air
    allinds = list(_spec_indices.keys())
    simplefilter('ignore', np.RankWarning)  # a warning about poorly fitting polynomial, ignore
    tabname = 'gtc_fullinfo.csv'
    myargs = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    myargs.add_argument('-f', '--file-name', required=True, help='File to be plotted')
    myargs.add_argument('-r', '--repeat', action='store_true', default=False)
    sysargs = myargs.parse_args()
    _fname: str = sysargs.file_name
    _repeat: bool = sysargs.repeat
    _df: pd.DataFrame = pd.read_csv(tabname)
    _dflines: pd.DataFrame = pd.read_csv('spectral_indices.csv')
    _df.rename(columns={col: col.lower() for col in _df.columns}, inplace=True)
    _df, _dflines = main(_fname, _spec_indices, _df, _dflines, _repeat)
    _df.to_csv(tabname, index=False)
    _dflines.to_csv('spectral_indices.csv', index=False)
