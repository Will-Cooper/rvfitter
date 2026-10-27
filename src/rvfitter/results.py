"""Results tables and writers (CSV, JSON, TSV, HTML) plus per-line details."""
import html
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .types import ObjectResult

__all__ = ['FLAGS', 'EXCLUDE_FLAGS', 'METHOD_ORDER', 'flag_names', 'to_table', 'details_table', 'write',
           'summary_lines']

log = logging.getLogger(__name__)

#: Bit flags used in ``Measurement.flag`` and the ``rv_<method>_flag`` columns (combined with bitwise OR):
#: fit_failed (the fit failed or gave a non-finite error), few_regions (too few usable pixels or regions),
#: error_floor (the error was raised to a floor), template_grid_edge (template requested outside the grid; the
#: nearest point was used), telluric_contaminated, not_converged (optimum at the edge of the search range or an
#: unconverged sampler), poor_fit (reduced χ² > 3; the error has been scaled by √(χ²/ν), but check the
#: template) and outlier (a gross outlier among the method's regions)
FLAGS: dict[str, int] = {
    'fit_failed': 1,
    'few_regions': 2,
    'error_floor': 4,
    'template_grid_edge': 8,
    'telluric_contaminated': 16,
    'not_converged': 32,
    'poor_fit': 64,
    'outlier': 128,
}

#: Flags that exclude a measurement from a method's combination and from the adopted RV
EXCLUDE_FLAGS = FLAGS['fit_failed'] | FLAGS['few_regions'] | FLAGS['not_converged'] | FLAGS['outlier']

#: Fixed order of the per-method column blocks
METHOD_ORDER = ('linecentre', 'ccf', 'template_fit', 'forward_mcmc', 'lsd', 'telluric_ref')

_FORMATS = {'.csv': 'csv', '.json': 'json', '.tsv': 'tsv', '.html': 'html', '.htm': 'html'}
_NAN = float('nan')


def flag_names(flag: int) -> list[str]:
    """Names of the bits set in ``flag``"""
    flag = int(flag)
    return [name for name, bit in FLAGS.items() if flag & bit]


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return _NAN


def _method_flag(mres) -> int:
    flag = 0
    for meas in mres.measurements:
        flag |= int(meas.flag)
    return flag


def _quad(*terms: float) -> float:
    return float(math.sqrt(sum(0.0 if not np.isfinite(t) else t * t for t in terms)))


def _columns() -> list[str]:
    cols = ['object', 'n_spectra', 'files']
    for m in METHOD_ORDER:
        cols += [f'rv_{m}', f'rv_{m}_err', f'rv_{m}_flag', f'n_{m}']
    cols += ['rv_adopted', 'rv_adopted_err', 'rv_adopted_q16', 'rv_adopted_q84', 'tau_line', 'tau_method',
             'n_methods', 'n_lines', 'rv_mean', 'rv_spread', 'sys_template', 'sys_wavezp', 'rv_total_err',
             'bary_kms', 'bary_source', 'frame', 'binary_dbic', 'binary_flag', 'config_hash', 'version']
    return cols


def to_table(results: Sequence[ObjectResult]) -> pd.DataFrame:
    """
    One row per object, with a fixed column order

    Methods that were not run are NaN (``<NA>`` for the integer flag and count columns).
    """
    rows = []
    for res in results:
        row: dict[str, Any] = {'object': res.object, 'n_spectra': len(res.files),
                               'files': ';'.join(map(str, res.files))}
        rvs, errs = [], []
        for m in METHOD_ORDER:
            mres = res.methods.get(m)
            if mres is None:
                row.update({f'rv_{m}': _NAN, f'rv_{m}_err': _NAN, f'rv_{m}_flag': pd.NA, f'n_{m}': pd.NA})
                continue
            row[f'rv_{m}'] = _num(mres.rv)
            row[f'rv_{m}_err'] = _num(mres.rv_err)
            row[f'rv_{m}_flag'] = _method_flag(mres)
            row[f'n_{m}'] = int(mres.n_regions)
            if np.isfinite(mres.rv):
                rvs.append(float(mres.rv))
                errs.append(_num(mres.rv_err))
        ad = res.adopted or {}
        row['rv_adopted'] = _num(ad.get('median', ad.get('rv')))
        row['rv_adopted_err'] = _num(ad.get('err'))
        row['rv_adopted_q16'] = _num(ad.get('q16'))
        row['rv_adopted_q84'] = _num(ad.get('q84'))
        row['tau_line'] = _num(ad.get('tau_line'))
        row['tau_method'] = _num(ad.get('tau_method'))
        row['n_methods'] = int(ad.get('n_methods', len(res.methods)))
        row['n_lines'] = int(ad.get('n_lines', 0))
        rv_arr, err_arr = np.array(rvs), np.array(errs)
        good = np.isfinite(err_arr) & (err_arr > 0)
        if good.any():
            w = 1.0 / err_arr[good] ** 2
            row['rv_mean'] = float(np.sum(w * rv_arr[good]) / np.sum(w))
        else:
            row['rv_mean'] = float(rv_arr.mean()) if len(rv_arr) else _NAN
        row['rv_spread'] = float(np.std(rv_arr, ddof=1)) if len(rv_arr) > 1 else _NAN
        sysd = res.systematics or {}
        row['sys_template'] = _num(sysd.get('template_spread'))
        row['sys_wavezp'] = _num(sysd.get('wavelength_zero_point'))
        row['rv_total_err'] = (_quad(row['rv_adopted_err'], row['sys_template'], row['sys_wavezp'],
                                       _num(sysd.get('floor')))
                               if np.isfinite(row['rv_adopted_err']) else _NAN)
        bary = res.barycentric or {}
        row['bary_kms'] = _num(bary.get('applied_kms'))
        row['bary_source'] = str(bary.get('source', ''))
        row['frame'] = str(bary.get('frame', ''))
        row['binary_dbic'] = _num((res.binary or {}).get('dbic'))
        row['binary_flag'] = bool((res.binary or {}).get('flag', False))
        row['config_hash'] = res.config_hash
        row['version'] = res.version
        rows.append(row)
    df = pd.DataFrame(rows, columns=_columns())
    for m in METHOD_ORDER:
        df[f'rv_{m}_flag'] = df[f'rv_{m}_flag'].astype('Int64')
        df[f'n_{m}'] = df[f'n_{m}'].astype('Int64')
    return df


def details_table(results: Sequence[ObjectResult]) -> pd.DataFrame:
    """One row per ``Measurement`` (method, region, RV, error, flag and scalar diagnostics)"""
    rows = []
    for res in results:
        for m, mres in res.methods.items():
            for meas in mres.measurements:
                row = {'object': res.object, 'method': m, 'region': meas.region, 'rv': _num(meas.rv),
                       'rv_err': _num(meas.rv_err), 'flag': int(meas.flag)}
                for k, v in (meas.diagnostics or {}).items():
                    if isinstance(v, (int, float, str, bool, np.number)):
                        row[f'diag_{k}'] = v
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=['object', 'method', 'region', 'rv', 'rv_err', 'flag'])
    return pd.DataFrame(rows)


def _fmt(value: Any) -> str:
    if value is None or value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
        return ''
    if isinstance(value, (float, np.floating)):
        return f'{value:.4g}' if abs(value) < 1e-3 else f'{value:.3f}'
    return str(value)


_HTML_CSS = """
body{font-family:Helvetica,Arial,sans-serif;margin:1.5em;color:#222;background:#fff}
h1{font-size:1.3em}
table{border-collapse:collapse;font-size:0.85em}
th,td{border:1px solid #ccc;padding:3px 8px;text-align:right;white-space:nowrap}
td.l,th.l{text-align:left}
th{background:#004488;color:#fff;cursor:pointer;position:sticky;top:0}
th:hover{background:#0066bb}
tr:nth-child(even){background:#f4f6f9}
a{color:#004488}
@media (prefers-color-scheme: dark){body{background:#181818;color:#ddd}tr:nth-child(even){background:#222}
td,th{border-color:#444}a{color:#DDAA33}}
"""

_HTML_JS = """
document.querySelectorAll('th').forEach(function(th,i){th.addEventListener('click',function(){
var t=th.closest('table'),b=t.tBodies[0],rows=Array.from(b.rows),asc=th.dataset.asc!=='1';
th.dataset.asc=asc?'1':'0';
rows.sort(function(x,y){var a=x.cells[i].textContent,c=y.cells[i].textContent,
na=parseFloat(a),nc=parseFloat(c);
if(a===''&&c==='')return 0;if(a==='')return 1;if(c==='')return -1;
var r=(!isNaN(na)&&!isNaN(nc))?na-nc:a.localeCompare(c);return asc?r:-r;});
rows.forEach(function(r){b.appendChild(r);});});});
"""


def _to_html(df: pd.DataFrame, plots: Mapping[str, Any] | None) -> str:
    plots = plots or {}
    left = {c: not (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])) for c in df.columns}
    head = ''.join(f'<th class="{"l" if left[c] else ""}">{html.escape(c)}</th>' for c in df.columns)
    if plots:
        head += '<th class="l">plots</th>'
    body = []
    for _, row in df.iterrows():
        cells = ''.join(f'<td class="{"l" if left[c] else ""}">{html.escape(_fmt(row[c]))}</td>' for c in df.columns)
        if plots:
            paths = plots.get(row['object'], ())
            if isinstance(paths, (str, Path)):
                paths = [paths]
            links = ' '.join(f'<a href="{html.escape(str(p), quote=True)}">{html.escape(Path(str(p)).name)}</a>'
                             for p in paths)
            cells += f'<td class="l">{links}</td>'
        body.append(f'<tr>{cells}</tr>')
    return (f'<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><title>rvfitter results</title>'
            f'<style>{_HTML_CSS}</style></head><body><h1>rvfitter results</h1>'
            f'<p>Click a column heading to sort.</p><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table><script>{_HTML_JS}</script></body></html>\n')


def _resolve_formats(path: Path, formats: str | Sequence[str] | None) -> list[tuple[str, Path]]:
    if formats is None:
        fmt = _FORMATS.get(path.suffix.lower())
        if fmt is None:
            return [('csv', path.with_name(path.name + '.csv'))]
        return [(fmt, path)]
    if isinstance(formats, str):
        formats = [f for f in formats.split(',') if f]
    out = []
    for f in formats:
        f = f.strip().lower().lstrip('.')
        if f == 'htm':
            f = 'html'
        if f not in ('csv', 'json', 'tsv', 'html'):
            raise ValueError(f'unknown output format {f!r}')
        out.append((f, path.with_name((path.stem if path.suffix.lower() in _FORMATS else path.name) + f'.{f}')))
    return out


def write(results: Sequence[ObjectResult], path: str | Path, formats: str | Sequence[str] | None = None, *,
          config: Any = None, dump: Callable[[Any, Path], Any] | None = None,
          plots: Mapping[str, Any] | None = None) -> list[Path]:
    """
    Write the results table, the details table and (optionally) the config used

    Parameters
    ----------
    results
        Object results
    path
        Output path; the format is inferred from its extension unless ``formats`` is given
    formats
        One or more of csv, json, tsv, html (list or comma-separated); each is written to ``path`` with that suffix
    config, dump
        If both are given, ``dump(config, <stem>_config.toml)`` is called (e.g. ``rvfitter.config.dump``)
    plots
        Mapping of object name to plot path(s), linked from the HTML table

    Returns
    -------
    list of the files written
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = to_table(results)
    written = []
    for fmt, out in _resolve_formats(path, formats):
        if fmt == 'csv':
            df.to_csv(out, index=False)
        elif fmt == 'tsv':
            df.to_csv(out, index=False, sep='\t')
        elif fmt == 'json':
            out.write_text(df.to_json(orient='records', indent=2) + '\n')
        else:
            out.write_text(_to_html(df, plots))
        written.append(out)
    stem = path.with_suffix('') if path.suffix.lower() in _FORMATS else path
    det = Path(f'{stem}_details.csv')
    details_table(results).to_csv(det, index=False)
    written.append(det)
    if config is not None and dump is not None:
        cfgpath = Path(f'{stem}_config.toml')
        dump(config, cfgpath)
        written.append(cfgpath)
    for w in written:
        log.info('Wrote %s', w)
    return written


def summary_lines(results: Sequence[ObjectResult]) -> list[str]:
    """Aligned plain-text summary of the adopted RVs, for the log"""
    df = to_table(results)
    head = ['object', 'n_spec', 'RV_adopted [km/s]', 'err', 'total_err', 'methods', 'bary [km/s]', 'binary']

    def f(x):
        return f'{x:.2f}' if np.isfinite(x) else '--'

    rows = [[str(r['object']), str(r['n_spectra']), f(r['rv_adopted']), f(r['rv_adopted_err']),
             f(r['rv_total_err']), str(r['n_methods']), f(r['bary_kms']), 'yes' if r['binary_flag'] else 'no']
            for _, r in df.iterrows()]
    widths = [max(len(h), *(len(x[i]) for x in rows)) if rows else len(h) for i, h in enumerate(head)]

    def line(cells):
        pairs = enumerate(zip(cells, widths, strict=True))
        return '  '.join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in pairs)

    return [line(head)] + [line(x) for x in rows]
