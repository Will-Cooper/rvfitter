from astropy.io.fits import getheader
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import scipy.stats as ss

import argparse
from collections import OrderedDict
import io
import sys
from typing import Sequence
from warnings import simplefilter

sys.path.insert(0, 'rvfitter/')
original_stdout = sys.stdout
sys.stdout = io.StringIO()
from splat import Spectrum, measureIndexSet
sys.stdout = original_stdout
from utils import *
from linecentering import linecentering
from crosscorrelate import crosscorrelate


def tabquery(fname: str, df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Queries the dataframe to extract that row

    Parameters
    ----------
    fname
        The full filename
    df
        The DataFrame containing the data

    Returns
    -------
    row
        The series for a given object
    """
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
    """
    Retrieve the spectral indices for a spectrum

    Parameters
    ----------
    tname
        The name of the object to compare with the column
    colname
        The column to check the object name
    fname
        The full filename
    dflines
        The dataframe to insert the spectral indices

    Returns
    -------
    dflines
        The dataframe to insert the spectral indices
    """
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
    """
    Checks the resolution of the spectra from the filename

    Parameters
    ----------
    fname
        The full filename

    Returns
    -------
    hires
        Switch if spectra is high resolution or not
    """
    if 'R300R' in fname:
        hires = False
    else:
        hires = True
    return hires


def adoptedrv(df: pd.DataFrame, colname: str, tname: str, lcvals: Sequence[float], lcerr: Sequence[float],
              xcorr: Sequence[float], xerr: Sequence[float], spec_indices: dict, systematic: float = 0.,
              outname: Optional[str] = None, wunit: u.Unit = u.AA) -> pd.DataFrame:
    """
    The method for creating an adopted radial velocity

    Parameters
    ----------
    df
        The dataframe to insert the adopted RV
    colname
        The column name in the dataframe to find the correct object
    tname
        The name of the object to check in the column
    lcvals
        The array of line centered RVs
    lcerr
        The array of line centered RV errors
    xcorr
        The array of cross correlated RVs
    xerr
        The array of cross correlated RV errors
    spec_indices
        Dictionary of spectral indices to central wavelength
    systematic
        Systematic RV error to be appended in quadrature
    outname
        The file name of the figure inside adoptedrvplots/ (default: <tname>_adoptedrv.pdf)
    wunit
        The unit of the spec_indices wavelengths, used for the line labels

    Returns
    -------
    df
        The dataframe to enter the adopted RV
    """
    fig: plt.Figure = plt.figure(figsize=(4, 3))
    axlines: plt.Axes = fig.add_axes([0.1, 0.4, 0.8, 0.5])
    axpdf: plt.Axes = fig.add_axes([0.1, 0.1, 0.8, 0.3])
    allindices = np.array(list(spec_indices.keys()))
    posfmt = (lambda pos: f'{int(pos)}') if wunit == u.AA else (lambda pos: f'{pos:.2f}')
    indicesplot = np.array([specindex.capitalize().replace('1', '\,\\textsc{i}') + r' $\lambda$'
                           + posfmt(pos) + '\,' + wunit.to_string(u.format.Latex)
                            for specindex, pos in spec_indices.items()])
    ypos = np.arange(len(allindices)) + 1
    if not all(len(arr) == len(allindices) for arr in (lcvals, lcerr, xcorr, xerr)):
        raise IndexError('the per-line arrays must hold one slot per spectral index, in spec_indices order')
    lcplot = copy(lcvals)
    lcploterr = copy(lcerr)
    xplot = copy(xcorr)
    xploterr = copy(xerr)
    # joint masks, so a value is only ever used together with its own error
    lcgood = np.isfinite(lcplot) & np.isfinite(lcploterr)
    xgood = np.isfinite(xplot) & np.isfinite(xploterr)
    if not np.any(lcgood) and not np.any(xgood):
        plt.close(fig)
        return df
    # each method, for reporting and plotting: weighted mean with the chi^2-inflated error
    loclc, scalelc = combine_rvs(lcplot, lcploterr)
    locx, scalex = combine_rvs(xplot, xploterr)
    # adopted: hierarchical line x method model over every finite (method, line) measurement, marginalising
    # offsets shared by both methods on one line (tau_line) and offsets of a whole method (tau_meth).
    # The per-line arrays hold one fixed slot per line of spec_indices (NaN if rejected), so the line names
    # pair the two methods' measurements of the same line
    hier = hierarchical_combine(np.concatenate([lcplot, xplot]), np.concatenate([lcploterr, xploterr]),
                                np.concatenate([allindices, allindices]),
                                ['lc'] * len(allindices) + ['xcorr'] * len(allindices))
    # the posterior of mu has power-law tails when few methods (or lines) constrain a tau under its half-Cauchy
    # prior, so its SD grows without limit with the tau grid; the median and the central 68 per cent interval
    # are stable, so they define the adopted value and its error
    locpost, scalepost = hier['median'], (hier['q84'] - hier['q16']) / 2
    errpost = np.sqrt(scalepost ** 2 + systematic ** 2)
    logging_rvcalc(f'Hierarchical model ({hier["n"]} measurements, {hier["nlines"]} lines, '
                   f'{hier["nmethods"]} methods; half-Cauchy tau scale {hier["tau_scale"]:.2f} km/s): '
                   f'mu mean {hier["mean"]:.2f} +/- SD {hier["sd"]:.2f} km/s, median {hier["median"]:.2f} '
                   f'[16%: {hier["q16"]:.2f}, 84%: {hier["q84"]:.2f}]; '
                   f'tau_line = {hier["tau_line"]:.2f} km/s, tau_meth = {hier["tau_meth"]:.2f} km/s')
    if hier['nmethods'] == 1:
        logging_rvcalc('WARNING: only one method measured this object, so the between-method systematic is not'
                       ' estimable (tau_meth fixed at 0)')
    lo = np.concatenate([lcplot - lcploterr, xplot - xploterr, [locpost - 3 * errpost]])
    hi = np.concatenate([lcplot + lcploterr, xplot + xploterr, [locpost + 3 * errpost]])
    minpos = np.floor(np.nanmin(lo) / 5) * 5
    maxpos = np.ceil(np.nanmax(hi) / 5) * 5
    pdfxpoints = np.linspace(minpos, maxpos, int(maxpos - minpos + 1) * 10)
    # the actual posterior of mu (a mixture of Gaussians), broadened by the systematic as in its label
    posteriorpdf = mixture_pdf(pdfxpoints, hier['means'], hier['sds'], hier['weights'], extra_sd=systematic)
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

    for loc, scale, colour in ((locx, scalex, 'orange'), (loclc, scalelc, 'blue')):
        if not np.isfinite(loc):  # this method measured no line
            continue
        methodpdf = ss.norm.pdf(pdfxpoints, loc=loc, scale=scale)
        axpdf.plot(pdfxpoints, methodpdf / np.max(methodpdf), color=colour,
                   label=rf'${loc:.1f}\pm{scale:.1f}$\,km\,s$^{{-1}}$')
        [ax.axvline(loc, color=colour, ls='--', lw=0.75) for ax in (axpdf, axlines)]
    axpdf.plot(pdfxpoints, posteriorpdf, color='black',
               label=rf'${locpost:.1f}\pm{errpost:.1f}$\,km\,s$^{{-1}}$')
    [ax.axvline(locpost, color='black', ls='--', lw=0.75) for ax in (axpdf, axlines)]
    axpdf.set_yticks([0, 0.5, 1])
    axpdf.set_ylim(-0.1, 1.1)
    axpdf.set_xlim(minpos, maxpos)
    axpdf.set_xlabel(f'RV\,[km\,s$^{{-1}}$]', fontsize='medium')
    axpdf.set_ylabel('PDF', fontsize='medium')

    os.makedirs('adoptedrvplots', exist_ok=True)
    fig.savefig(f'adoptedrvplots/{outname or f"{tname}_adoptedrv.pdf"}', bbox_inches='tight')
    plt.close(fig)

    df.loc[df[colname] == tname, 'thisrv'] = locpost
    df.loc[df[colname] == tname, 'thisrverr'] = errpost
    df.loc[df[colname] == tname, 'rv_adopted_nmethods'] = hier['nmethods']
    df.loc[df[colname] == tname, 'rv_adopted_nlines'] = hier['nlines']
    logging_rvcalc(f'Adopted RV {locpost:.1f} +/- {errpost:.1f} km/s from {hier["nmethods"]} method(s)'
                   f' and {hier["nlines"]} line(s)')
    return df


def errorappend(df: pd.DataFrame, tname: str, colname: str) -> pd.DataFrame:
    """
    Editing the errors on Teff and log g to account for grid model size

    Parameters
    ----------
    df
        The dataframe of all data
    tname
        The name of the target
    colname
        The column name that contains the target name

    Returns
    -------
    df
        The dataframe of all objects
    """
    teffstd = df.loc[df[colname] == tname, 'xcorrtefferr'].iloc[0]
    gravstd = df.loc[df[colname] == tname, 'xcorrgraverr'].iloc[0]
    if not np.isnan(df.loc[df[colname] == tname, 'xcorrtefferr'].iloc[0]):
        df.loc[df[colname] == tname, 'xcorrtefferr'] = np.sqrt(teffstd ** 2 + 50 ** 2)
        df.loc[df[colname] == tname, 'xcorrgraverr'] = np.sqrt(gravstd ** 2 + 0.25 ** 2)
    return df


def getwaverms(fname: str) -> float:
    """
    Finding the wavelength rms

    Parameters
    ----------
    fname
        The full file name

    Returns
    -------
    _
        The wavelength rms
    """
    wvcalibf = fname[:fname.find('Science')] + 'wvcalib.fits'
    head: Dict[str, float] = getheader(wvcalibf, 2)
    disp: float = head['CEN_DISP']  # central dispersion in wave/ pix
    rms: float = head['RMS']  # rms in pixels
    return disp * rms


def main(fname: str, spec_indices: Dict[str, float], df: pd.DataFrame, dflines: pd.DataFrame,
         repeat: bool, systematic: float):
    """
    The main control module

    Parameters
    ----------
    fname
        The full filename
    spec_indices
        The dictionary of spectral indices
    df
        The dataframe with all of the data
    dflines
        The dataframe of the spectral lines
    repeat
        Switch to repeat or not
    systematic
        Systematic RV uncertainty to add on

    Returns
    -------
    dfout, dflines
        The dataframe of all data
        The dataframe of spectral lines
    """
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
    expectedteff = expected_teff(stephens(df.loc[df['index'] == ind].kasttypenum.iloc[0] - 60))
    logging_rvcalc(f'\n{tname}')
    dfout = df
    waverms = getwaverms(fname)
    if hires:  # checking high resolution
        dfout, lcvals, lcerr = linecentering(fname, spec_indices, dfout, repeat, tname, colname, fappend,
                                             waverms=waverms)  # perform the line centering
    else:  # don't do line centering for low resolution
        lcvals, lcerr = np.full(len(spec_indices), np.nan), np.full(len(spec_indices), np.nan)
    dfout, xcorr, xerr = crosscorrelate(fname, spec_indices, dfout, repeat, tname, colname, fappend,
                                        teff=expectedteff, dorv=hires, waverms=waverms)  # cross correlation
    dfout = errorappend(dfout, tname, colname)
    dfout = adoptedrv(dfout, colname, tname, lcvals, lcerr, xcorr, xerr, spec_indices, systematic=systematic,
                      outname=f'{tname}_{fappend}_adoptedrv.pdf')
    dflines = get_indices(tname, colname, fname, dflines)
    return dfout, dflines


def build_parser() -> argparse.ArgumentParser:
    """
    The command line parser

    Returns
    -------
    myargs
        The argument parser
    """
    myargs = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    myargs.add_argument('-f', '--file-name', required=True, help='File to be plotted')
    myargs.add_argument('-r', '--repeat', action='store_true', default=False, help='Repeat manual measurements?')
    myargs.add_argument('-s', '--systematic', type=float, default=0., help='Systematic RV uncertainty [km/s]')
    return myargs


if __name__ == '__main__':
    _spec_indices = {'k1-a': 7664.8991, 'k1-b': 7698.9646,
                     'rb1-a': 7800.268, 'rb1-b': 7947.603,
                     'na1-a': 8183.2556, 'na1-b': 8194.8240,
                     'cs1-a': 8521.13165, 'cs1-b': 8943.47424}  # air
    allinds = list(_spec_indices.keys())
    simplefilter('ignore', np.RankWarning)  # a warning about poorly fitting polynomial, ignore
    tabname = 'gtc_fullinfo.csv'
    sysargs = build_parser().parse_args()
    set_plot_style()
    _fname: str = sysargs.file_name
    _repeat: bool = sysargs.repeat
    _systematic: float = sysargs.systematic
    _df: pd.DataFrame = pd.read_csv(tabname)
    _dflines: pd.DataFrame = pd.read_csv('spectral_indices.csv')
    _df.rename(columns={col: col.lower() for col in _df.columns}, inplace=True)
    _df, _dflines = main(_fname, _spec_indices, _df, _dflines, _repeat, _systematic)
    _df.to_csv(tabname, index=False)
    _dflines.to_csv('spectral_indices.csv', index=False)
