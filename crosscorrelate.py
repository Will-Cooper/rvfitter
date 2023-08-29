import scipy.stats as ss

from xcorr import *


jsonname = 'xcorr_used.json'


def load_fitinfo(spec: Spectrum1D, spec_indices: Dict[str, float],
                 fname: str, repeat: bool, **kwargs) -> Tuple[List[str], List[Xcorr]]:
    """
    Loading the information of the cross correlation fits

    Parameters
    ----------
    spec
        The spectrum of the object
    spec_indices
        The dictionary of spectral indices
    fname
        The full file name
    repeat
        Switch to repeat the process or not
    kwargs
        Dictionary of information to pass to the fitting procedure

    Returns
    -------
    useset, objlist
        The list of lines used
        The list of all of the fits
    """
    d = json_handle(jsonname)
    if repeat or fname not in d.keys():
        useset, objlist = interactive_loop(spec, spec_indices, fname, **kwargs)
    else:
        dobj = d[fname]
        useset, objlist = [], []
        paramlist = ('c1', 'c4', 'teff', 'grav', 'met', 'rv', 'smoothlevel', 'use')
        for spec_index, objinfo in tqdm(dobj.items(), total=len(dobj.keys()),
                                        desc='Loading Fits', leave=False):
            if objinfo[-1]:
                useset.append(spec_index)
            labline = spec_indices[spec_index]
            kwargs = {param: objinfo[i] for i, param in enumerate(paramlist)}
            obj = Xcorr(spec, labline, spec_index, **kwargs)
            objlist.append(obj)
    return useset, objlist


def interactive_loop(spec: Spectrum1D, spec_indices: Dict[str, float],
                     fname: str, **kwargs) -> Tuple[List[str], Sequence[Xcorr]]:
    """
    Interactively fitting the cross correlation

    Parameters
    ----------
    spec
        The spectrum of the object
    spec_indices
        The dictionary of the spectral indices
    fname
        The full filename
    kwargs
        The fit parameters to pass to the fitting routine

    Returns
    -------
    useset, objlist
        The list of lines used
        The list of all of the fits
    """
    dout = json_handle(jsonname)
    args = (spec, spec_indices)
    outset, objlist = manual_xcorr_fit(*args, **kwargs)
    dobj = fitparams(outset, objlist)
    dout[fname] = dobj
    json_handle(jsonname, dout)
    return outset, objlist


def fitparams(useset: List[str], objlist: Sequence[Xcorr]) -> Dict[str, List[Union[float, str, bool]]]:
    """
    Saving the fitted parameters

    Parameters
    ----------
    useset
        List of spectral lines used
    objlist
        The list of line fits

    Returns
    -------
    dobj
        The dictionary of object fit parameters
    """
    dobj = {}
    for obj in objlist:  # for every object
        key = obj.spec_index
        paramlist = []
        for val in (obj.c1, obj.c4, obj.teff,
                    obj.grav, obj.met, obj.rv, obj.smoothlevel):
            if not (isinstance(val, str) or isinstance(val, float) or isinstance(val, int)) and val is not None:
                val = float(val.value)
            paramlist.append(val)
        paramlist.append(True if key in useset else False)
        dobj[key] = paramlist
    return dobj


def auto_xcorr_fit(useset: list, spec_indices: Dict[str, float], objlist: List[Xcorr], df: pd.DataFrame,
                   tname: str, colname: str,
                   fappend: str = '', **kwargs) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Automatically cross correlating each spectral line

    Parameters
    ----------
    useset
        The list of objects to use
    spec_indices
        The dictionary of spectral indices
    objlist
        The list of object fit parameters
    df
        The dataframe to insert the data within
    tname
        The target name
    colname
        The column to check for the target names
    fappend
        What to append to a filename when saving
    kwargs
        The fit parameters

    Returns
    -------
    df, rv_list, err_list
        The dataframe of all of the data
        The list of radial velocities for each spectral line
        The list of RV errors from each spectral line
    """
    nrows = kwargs.get('nrows', 4)
    ncols = kwargs.get('ncols', 2)
    dorv = kwargs.get('dorv', True)
    fig, axs = plt.subplots(nrows, ncols, figsize=(8, 4), num=4)
    fig: plt.Figure = fig
    axs: np.ndarray = axs.flatten()
    wunit: u.Unit = kwargs.get('wunit', u.AA)
    allindices = np.array(list(spec_indices.keys()))
    rv_list, err_list = np.full(len(allindices), np.nan), np.full(len(allindices), np.nan)
    teff_list, grav_list, met_list = np.full(len(allindices), np.nan), np.full(len(allindices), np.nan),\
        np.full(len(allindices), np.nan)
    rv, err = np.nan, np.nan
    teff, grav, met = np.nan, np.nan, np.nan
    j = -1
    for i, spec_index in tqdm(enumerate(spec_indices), total=len(spec_indices.keys()),
                              desc='Fitting Cross Correlation', leave=False):  # every spectral index
        ax: plt.Axes = axs[i]
        obj = objlist[i]
        obj.ax = ax
        obj.plotter()
        yroundpoint = 0.5
        if all([isinstance(objcoord, u.Quantity) for objcoord in (obj.c1, obj.c4)]):
            specreg: Spectrum1D = extract_region(obj.sub_speccorr, SpectralRegion(obj.c1, obj.c4))
            miny, maxy = specreg.flux.min().value, specreg.flux.max().value
            roundmin = np.floor(miny / yroundpoint) * yroundpoint
            roundmax = np.ceil(maxy / yroundpoint) * yroundpoint
            if roundmax - maxy < 0.1:
                roundmax += yroundpoint / 2
            ax.set_ylim(roundmin, roundmax)
            ax.set_xlim(obj.c1.value, obj.c4.value)
        ax.legend([], [])
        if spec_index not in useset:
            continue
        j += 1
        teffobj = int(obj.teff.value)
        logging_rvcalc(f'{spec_index.capitalize()} -- {teffobj}K, {obj.grav.value:.1f} log g,'
                       f' {obj.met.value:.1f} [Fe/H]; {obj.rv.value:.1f} km/s')
        if dorv:
            rv_list[j] = obj.rv.value
            err_list[j] = obj.rverr.value
        teff_list[j] = teffobj
        grav_list[j] = obj.grav.value
        met_list[j] = obj.met.value

    rv_list_cut = rv_list[~np.isnan(rv_list)]
    err_list_cut = err_list[~np.isnan(err_list)]
    teff_list_cut = teff_list[~np.isnan(teff_list)]
    grav_list_cut = grav_list[~np.isnan(grav_list)]
    met_list_cut = met_list[~np.isnan(met_list)]
    if len(rv_list_cut):  # if there is at least a line used in the cross correlation
        rv, std = ss.norm.fit(rv_list_cut)
        if len(rv_list_cut) > 1:
            err = std / np.sqrt(len(rv_list_cut))
        else:
            err = err_list_cut[0]
        df.loc[df[colname] == tname, 'thisrvxcorrerr'] = round(err, 1)
    else:
        logging_rvcalc('Empty RV list for cross correlation calculation!')
    if len(teff_list_cut):
        teff, teffstd = ss.norm.fit(teff_list_cut)
        grav, gravstd = ss.norm.fit(grav_list_cut)
        met, metstd = ss.norm.fit(met_list_cut)
    else:
        teffstd, gravstd, metstd = np.nan, np.nan, np.nan
    logging_rvcalc(f'RV Cross Correlation = {rv:.1f} +/- {err:.1f}km/s')
    df.loc[df[colname] == tname, 'thisrvxcorr'] = round(rv, 1)
    df.loc[df[colname] == tname, 'xcorrteff'] = teff
    df.loc[df[colname] == tname, 'xcorrgrav'] = grav
    df.loc[df[colname] == tname, 'xcorrmet'] = met
    df.loc[df[colname] == tname, 'xcorrtefferr'] = teffstd
    df.loc[df[colname] == tname, 'xcorrgraverr'] = gravstd
    df.loc[df[colname] == tname, 'xcorrmeterr'] = metstd
    fig.supxlabel(fr'Wavelength\,[{wunit.to_string(u.format.Latex)}]')
    fig.supylabel(r'Normalised Flux\,[$F_{\lambda}$]')
    fig.subplots_adjust(hspace=2.5, wspace=0.15)
    if not os.path.exists('xcorrplots'):
        os.mkdir('xcorrplots')
    fname = f'xcorrplots/{tname}{"_" + fappend}_xcorr.pdf'
    plt.savefig(fname, bbox_inches='tight')
    plt.close(4)
    return df, rv_list, err_list


def crosscorrelate(fname: str, spec_indices: Dict[str, float], df: pd.DataFrame,
                   repeat: bool, tname: str,
                   colname: str, fappend: str = '', **kwargs) -> Tuple[pd.DataFrame, Sequence[float], Sequence[float]]:
    """
    The main cross correlation programme

    Parameters
    ----------
    fname
        The full filename
    spec_indices
        The dictionary of spectral indices
    df
        The dataframe of all of the data
    repeat
        Switch to repeat the manual fitting or not
    tname
        The name of the target
    colname
        The column name to check for the target
    fappend
        What to append to a filename when saving
    kwargs
        Additional fit parameter information to pass to the fitting routine

    Returns
    -------
    dfout, xcorr, xerr
        The dataframe with appended RV fits
        The list of cross correlated RVs for each spectral line
        The list of cross correlated RV errors from each spectral line
    """
    spec = freader(fname)
    logging_rvcalc(f'{tname}: Cross Correlation')
    useset, objlist = load_fitinfo(spec, spec_indices, fname, repeat, **kwargs)
    if not len(useset):  # if no successful fits, return the same lists but empty
        allindices = np.array(list(spec_indices.keys()))
        xcorr, xerr = np.full(len(allindices), np.nan), np.full(len(allindices), np.nan)
        return df, xcorr, xerr
    dfout, xcorr, xerr = auto_xcorr_fit(useset, spec_indices, objlist, df, tname, colname, fappend, **kwargs)
    return dfout, xcorr, xerr
