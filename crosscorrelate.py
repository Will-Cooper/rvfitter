import scipy.stats as ss

from xcorr import *


jsonname = 'xcorr_used.json'


def load_fitinfo(spec: Spectrum1D, spec_indices: Dict[str, float],
                 fname: str, repeat: bool, **kwargs) -> Tuple[List[str], List[Xcorr]]:
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
    dout = json_handle(jsonname)
    args = (spec, spec_indices)
    outset, objlist = manual_xcorr_fit(*args, **kwargs)
    dobj = fitparams(outset, objlist)
    dout[fname] = dobj
    json_handle(jsonname, dout)
    return outset, objlist


def fitparams(useset: List[str], objlist: Sequence[Xcorr]) -> Dict[str, List[Union[float, str, bool]]]:
    dobj = {}
    for obj in objlist:
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
    nrows = kwargs.get('nrows', 4)
    ncols = kwargs.get('ncols', 2)
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
                              desc='Fitting Cross Correlation', leave=False):
        ax: plt.Axes = axs[i]
        obj = objlist[i]
        obj.ax = ax
        obj.plotter()
        ax.set_ylim(*np.array([np.floor(np.min(obj.sub_speccorr.flux.value)),
                               np.ceil(np.max(obj.sub_speccorr.flux.value))]), )
        ax.set_yticks([])
        ax.legend([], [])
        if spec_index not in useset:
            continue
        j += 1
        teffobj = int(obj.teff.value)
        logging_rvcalc(f'{spec_index.capitalize()} -- {teffobj}K, {obj.grav.value:.1f} log g,'
                       f' {obj.met.value:.1f} [Fe/H]; {obj.rv.value:.1f} km/s')
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
    if len(rv_list_cut):
        rv, std = ss.norm.fit(rv_list_cut)
        teff, _ = ss.norm.fit(teff_list_cut)
        grav, _ = ss.norm.fit(grav_list_cut)
        met, _ = ss.norm.fit(met_list_cut)
        if len(rv_list_cut) > 1:
            err = std / np.sqrt(len(rv_list_cut))
        else:
            err = err_list_cut[0]
        df.loc[df[colname] == tname, 'thisrvxcorrerr'] = round(err, 1)
    else:
        logging_rvcalc('Empty RV list for cross correlation calculation!')
    logging_rvcalc(f'RV Cross Correlation = {rv:.1f} +/- {err:.1f}km/s')
    df.loc[df[colname] == tname, 'thisrvxcorr'] = round(rv, 1)
    df.loc[df[colname] == tname, 'xcorrteff'] = teff
    df.loc[df[colname] == tname, 'xcorrgrav'] = grav
    df.loc[df[colname] == tname, 'xcorrmet'] = met

    fig.supxlabel(r'Wavelength [' + wunit.to_string(u.format.Latex) + ']')
    fig.supylabel(r'Normalised Flux [$F_{\lambda}$]')
    fig.subplots_adjust(hspace=1)
    if not os.path.exists('xcorrplots'):
        os.mkdir('xcorrplots')
    fname = f'xcorrplots/{tname}{"_" + fappend}_xcorr.pdf'
    plt.savefig(fname, bbox_inches='tight')
    plt.close(4)
    return df, rv_list, err_list


def crosscorrelate(fname: str, spec_indices: Dict[str, float], df: pd.DataFrame,
                   repeat: bool, tname: str,
                   colname: str, fappend: str = '', **kwargs) -> Tuple[pd.DataFrame, Sequence[float], Sequence[float]]:
    spec = freader(fname)
    logging_rvcalc(f'{tname}: Cross Correlation')
    useset, objlist = load_fitinfo(spec, spec_indices, fname, repeat, **kwargs)
    if not len(useset):
        allindices = np.array(list(spec_indices.keys()))
        xcorr, xerr = np.full(len(allindices), np.nan), np.full(len(allindices), np.nan)
        return df, xcorr, xerr
    dfout, xcorr, xerr = auto_xcorr_fit(useset, spec_indices, objlist, df, tname, colname, fappend, **kwargs)
    return dfout, xcorr, xerr
