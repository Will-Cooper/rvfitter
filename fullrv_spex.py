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
from fullrv import adoptedrv


def spex_target_name(fname: str) -> str:
    """
    The target name from a SpeX file name of the form <mode>_<target>[_...].fits

    Parameters
    ----------
    fname
        The full filename

    Returns
    -------
    _
        The target name
    """
    fnameonly = fname.split('/')[-1]
    if fnameonly.endswith('.fits'):  # remove the suffix only (str.strip would remove any of its characters)
        fnameonly = fnameonly[:-len('.fits')]
    return fnameonly.split('_')[1]


def tabquery(fname: str, colname: str, df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Queries the dataframe to extract that row

    Parameters
    ----------
    fname
        The full filename
    colname
        The name of the column with the target name
    df
        The DataFrame containing the data

    Returns
    -------
    row
        The series for a given object
    """
    tgtname = spex_target_name(fname)
    try:
        row: pd.Series = df[df[colname] == tgtname].copy().iloc[0]
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
    spec = freader(fname, wunit=u.um)
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
    if 'prism' in fname:
        hires = False
    else:
        hires = True
    return hires


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
    wunit = u.um
    funit = u.erg / u.cm ** 2 / wunit / u.s
    colname = 'Short Name'
    row = tabquery(fname, colname, df)
    tname = row[colname]
    logging_rvcalc(f'\n{tname}')
    hires = chekres(fname)
    expectedteff = expected_teff(row['Teff (K)'])
    dfout = df
    # line centering
    if hires:
        dfout, lcvals, lcerr = linecentering(fname, spec_indices, dfout, repeat, tname, colname,
                                             wunit=wunit, funit=funit, nrows=4, waverms=waverms)
    else:
        lcvals, lcerr = np.full(len(spec_indices), np.nan), np.full(len(spec_indices), np.nan)
    # cross correlation
    dfout, xcorr, xerr = crosscorrelate(fname, spec_indices, dfout, repeat, tname, colname,
                                        wunit=wunit, funit=funit, teff=expectedteff, dorv=hires,
                                        templatedir='bt-settl-cifist/useful_gemma/', nrows=4, waverms=waverms)
    try:
        dfout = errorappend(dfout, tname, colname)
    except KeyError:
        pass
    dfout = adoptedrv(dfout, colname, tname, lcvals, lcerr, xcorr, xerr, spec_indices, systematic=systematic,
                      outname=f'{tname}_adoptedrv.pdf', wunit=wunit)
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
    _spec_indices = {
        'na1-a': 8183.2556, 'na1-b': 8194.8240,  # continuum a challenge
        # 'cs1-a': 8521.13, 'cs1-b': 8943.47,  # often too noisy
        'na1-c': 11381.45, 'na1-d': 11403.78,
        'k1-a': 11690.219, 'k1-b': 11772.838,
        'k1-c': 12432.274, 'k1-d': 12522.144,
        # 'na1-e': 22062.42, 'na1-f': 22089.69  # too low signal
    }  # air unless > 20000A
    _spec_indices = {spec: val / 1e4 for spec, val in _spec_indices.items()}
    allinds = list(_spec_indices.keys())
    simplefilter('ignore', np.RankWarning)  # a warning about poorly fitting polynomial, ignore
    tabname = 'CandidateInfo.csv'
    waverms = 0.075 * u.Angstrom
    sysargs = build_parser().parse_args()
    _fname: str = sysargs.file_name
    _repeat: bool = sysargs.repeat
    _systematic: float = sysargs.systematic
    _df: pd.DataFrame = pd.read_csv(tabname)
    _dflines: pd.DataFrame = pd.read_csv('targets_lines.csv')
    _df, _dflines = main(_fname, _spec_indices, _df, _dflines, _repeat, _systematic)
    _df.to_csv(tabname, index=False)
    _dflines.to_csv('targets_lines.csv', index=False)
