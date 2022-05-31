import pandas as pd
import scipy.stats as ss
from tqdm import tqdm

from splot import *


jsonname = 'lines_used.json'


def load_fitinfo(spec: Spectrum1D, spec_indices: Dict[str, float],
                 fname: str, repeat: bool, **kwargs) -> Tuple[List[str], List[Splot]]:
    d = json_handle(jsonname)
    if repeat or fname not in d.keys():
        useset, objlist = interactive_loop(spec, spec_indices, fname, **kwargs)
    else:
        dobj = d[fname]
        useset, objlist = [], []
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


def interactive_loop(spec: Spectrum1D, spec_indices: Dict[str, float],
                     fname: str, **kwargs) -> Tuple[List[str], List[Splot]]:
    dout = json_handle(jsonname)
    args = (spec, spec_indices)
    outset, objlist = manual_lc_fit(*args, **kwargs)
    dobj = fitparams(outset, objlist)
    dout[fname] = dobj
    json_handle(jsonname, dout)
    return outset, objlist


def fitparams(useset: List[str], objlist: List[Splot]) -> Dict[str, List[Union[float, str, bool]]]:
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


def auto_lc_fit(useset: list, spec_indices: Dict[str, float], objlist: List[Splot], df: pd.DataFrame,
                tname: str, colname: str,
                fappend: str = '', **kwargs) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    nrows = kwargs.get('nrows', 4)
    ncols = kwargs.get('ncols', 2)
    fig, axs = plt.subplots(nrows, ncols, figsize=(8, 4))
    fig: plt.Figure = fig
    axs: np.ndarray = axs.flatten()
    wunit: u.Unit = kwargs.get('wunit', u.AA)
    rv_list, err_list = np.full(len(useset), np.nan), np.full(len(useset), np.nan)
    rv, err = np.nan, np.nan
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
        logging_rvcalc(f'{spec_index.capitalize()} -- {obj.line_profile.capitalize()} Profile'
                       f' with {obj.std.value:.1f}A sigma; {obj.rv.value} km/s.')
        rv_list[j] = obj.rv.value
        err_list[j] = obj.rverr.value

    rv_list_cut = rv_list[~np.isnan(rv_list)]
    err_list_cut = err_list[~np.isnan(err_list)]
    if len(rv_list_cut):
        rv, std = ss.norm.fit(rv_list_cut)
        if len(rv_list_cut) > 1:
            err = std / np.sqrt(len(rv_list_cut))
        else:
            err = err_list_cut[0]
        df.loc[df[colname] == tname, 'thisrvlcerr'] = round(err, 1)
    else:
        logging_rvcalc('Empty RV list for line centre calculation!')
    logging_rvcalc(f'RV Line Centre = {rv:.1f} +/- {err:.1f}km/s')
    df.loc[df[colname] == tname, 'thisrvlc'] = round(rv, 1)

    fig.supxlabel(r'Wavelength [' + wunit.to_string(u.format.Latex) + ']')
    fig.supylabel(r'Normalised Flux [$F_{\lambda}$]')
    fig.subplots_adjust(hspace=0.95)
    if not os.path.exists('lcplots'):
        os.mkdir('lcplots')
    fname = f'lcplots/{tname}{"_" + fappend}.pdf'
    plt.savefig(fname, bbox_inches='tight')
    return df, rv_list, err_list


def linecentering(fname: str, spec_indices: Dict[str, float], df: pd.DataFrame,
                  repeat: bool, tname: str, colname: str, fappend: str = '', **kwargs) -> pd.DataFrame:
    spec = freader(fname, **kwargs)
    logging_rvcalc(f'{tname}: Line Center')
    useset, objlist = load_fitinfo(spec, spec_indices, fname, repeat, **kwargs)
    if not len(useset):
        return df
    dfout, lcvals, lcerr = auto_lc_fit(useset, spec_indices, objlist, df, tname, colname, fappend, **kwargs)
    return dfout
