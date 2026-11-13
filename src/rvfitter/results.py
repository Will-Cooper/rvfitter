"""Results tables and writers (CSV, JSON, TSV, HTML) plus per-line details."""
import functools
import html
import logging
import math
import operator
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import METHOD_NAMES, OUTPUT_FORMATS, dump
from .types import ObjectResult

__all__ = ['FLAGS', 'EXCLUDE_FLAGS', 'TEMPLATE_COLUMNS', 'flag_names', 'to_table', 'details_table',
           'write', 'summary_lines']

log = logging.getLogger(__name__)

#: Bit flags used in ``Measurement.flag`` and the ``rv_<method>_flag`` columns (combined with bitwise OR):
#: fit_failed (the fit failed or gave a non-finite error), few_regions (too few usable pixels or regions),
#: error_floor (the error was raised to a floor), template_grid_edge (template requested outside the grid; the
#: nearest point was used), telluric_contaminated, not_converged (optimum at the edge of the search range or an
#: unconverged sampler), poor_fit (reduced χ² > 3; the error has been scaled by √(χ²/ν), but check the
#: template), outlier (a gross outlier among the method's regions) and template_param_edge (the best template is
#: at the edge of the searched parameter range or of the grid; see the logged guidance) and template_grid_hole (a
#: requested template lies in a hole of the grid, i.e. a missing corner file: the nearest point was used)
FLAGS: dict[str, int] = {
    'fit_failed': 1,
    'few_regions': 2,
    'error_floor': 4,
    'template_grid_edge': 8,
    'telluric_contaminated': 16,
    'not_converged': 32,
    'poor_fit': 64,
    'outlier': 128,
    'template_param_edge': 256,
    'template_grid_hole': 512,
}

#: Flags that exclude a measurement from a method's combination and from the adopted RV
EXCLUDE_FLAGS = FLAGS['fit_failed'] | FLAGS['few_regions'] | FLAGS['not_converged'] | FLAGS['outlier']

_FORMATS = {f'.{f}': f for f in OUTPUT_FORMATS} | {'.htm': 'html'}
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


def _count(value: Any) -> int:
    n = _num(value)
    return int(n) if np.isfinite(n) else 0


#: Best-template columns, taken from the first template method present in TEMPLATE_METHOD_ORDER
TEMPLATE_COLUMNS = ('template_method', 'template_grid', 'template_teff', 'template_logg', 'template_met',
                    'template_alpha', 'template_chi2', 'template_at_edge')
TEMPLATE_METHOD_ORDER = ('template_fit', 'forward_mcmc', 'ccf', 'lsd')


def _template_row(res: ObjectResult) -> dict[str, Any]:
    """The best-template columns of one object (NaN/empty if no template method ran)"""
    row: dict[str, Any] = {c: _NAN for c in TEMPLATE_COLUMNS}
    row.update(template_method='', template_grid='', template_at_edge=False)
    for m in TEMPLATE_METHOD_ORDER:
        mres = res.methods.get(m)
        summ = next((x.diagnostics.get('template_summary') for x in (mres.measurements if mres else [])
                     if (x.diagnostics or {}).get('template_summary')), None)
        if summ:
            row['template_method'] = m
            row['template_grid'] = str(summ.get('template_grid', ''))
            for a in ('teff', 'logg', 'met', 'alpha'):
                row[f'template_{a}'] = _num(summ.get(f'template_{a}'))
            row['template_chi2'] = _num(summ.get('template_chi2'))
            row['template_at_edge'] = bool(summ.get('template_at_edge', False))
            break
    return row


def _quad(*terms: float) -> float:
    return float(math.sqrt(sum(0.0 if not np.isfinite(t) else t * t for t in terms)))


# Output columns in order. A plain name is computed in `to_table`; a tuple (column, source, key, converter, default)
# is read from ``source`` ('adopted', 'systematics', 'epochs', 'barycentric' or 'binary') of the ObjectResult.
_SPEC: tuple[str | tuple, ...] = (
    'object', 'n_spectra', 'files',
    *(c for m in METHOD_NAMES for c in (f'rv_{m}', f'rv_{m}_err', f'rv_{m}_flag', f'n_{m}')),
    'rv_adopted',
    ('rv_adopted_err', 'adopted', 'err', _num, None), ('rv_adopted_q16', 'adopted', 'q16', _num, None),
    ('rv_adopted_q84', 'adopted', 'q84', _num, None), ('tau_line', 'adopted', 'tau_line', _num, None),
    ('tau_method', 'adopted', 'tau_method', _num, None), 'n_methods', ('n_lines', 'adopted', 'n_lines', int, 0),
    'rv_mean', 'rv_spread',
    ('sys_template', 'systematics', 'template_spread', _num, None),
    ('sys_wavezp', 'systematics', 'wavelength_zero_point', _num, None),
    # the star's absolute zero point (slit centring): the applied star - sky constant's error, else the slit-centring
    # prior (never a measured offset's value); NaN when unconstrained (see star_sky_term)
    ('sys_star_sky', 'systematics', 'star_sky_systematic', _num, None),
    'rv_total_err', *TEMPLATE_COLUMNS,
    ('zp_sky_kms', 'systematics', 'zp_sky_kms', _num, None), ('zp_sky_err', 'systematics', 'zp_sky_err', _num, None),
    ('zp_sky_slope', 'systematics', 'zp_sky_slope', _num, None),
    ('zp_sky_model_spread', 'systematics', 'zp_sky_model_spread', _num, None),
    ('zp_tell_kms', 'systematics', 'zp_tell_kms', _num, None),
    ('zp_tell_err', 'systematics', 'zp_tell_err', _num, None),
    ('zp_star_sky', 'systematics', 'zp_star_sky', _num, None),
    ('zp_star_sky_err', 'systematics', 'zp_star_sky_err', _num, None),
    ('zp_star_sky_n', 'systematics', 'zp_star_sky_n', _count, None),
    # the across-slit DAR term s·D(λ) of the star - sky fit: s (known from the slit PA, or fitted) and its error
    ('zp_dar_s', 'systematics', 'zp_dar_s', _num, None), ('zp_dar_s_err', 'systematics', 'zp_dar_s_err', _num, None),
    ('star_sky_term', 'systematics', 'star_sky_term', str, ''),  # what sys_star_sky is
    ('slit_prior_kms', 'systematics', 'slit_prior', _num, None),  # the computed slit-centring prior, used or not
    ('wavelength_correction', 'barycentric', 'wavelength_correction_mode', str, ''),  # the configured mode
    ('wavelength_correction_applied', 'barycentric', 'wavelength_correction', str, 'none'),  # what was applied
    ('zp_flags', 'barycentric', 'zp_flags', int, 0),  # pipeline.ZP_FLAGS bits
    ('rv_epoch_pvalue', 'epochs', 'pvalue', _num, None), ('rv_epoch_variable', 'epochs', 'variable', bool, False),
    ('n_epochs', 'epochs', 'n', int, 1),
    ('bary_kms', 'barycentric', 'applied_kms', _num, None), ('bary_source', 'barycentric', 'source', str, ''),
    ('frame', 'barycentric', 'frame', str, ''),
    ('binary_dbic', 'binary', 'dbic', _num, None), ('binary_flag', 'binary', 'flag', bool, False),
    'refitted', 'config_hash', 'version',
)
_COLUMNS = [c if isinstance(c, str) else c[0] for c in _SPEC]


def to_table(results: Sequence[ObjectResult]) -> pd.DataFrame:
    """
    One row per object, with a fixed column order

    Methods that were not run are NaN (``<NA>`` for the integer flag and count columns).
    """
    rows = []
    for res in results:
        ad = res.adopted or {}
        src = {'adopted': ad, 'systematics': res.systematics or {}, 'epochs': ad.get('epochs') or {},
               'barycentric': res.barycentric or {}, 'binary': res.binary or {}}
        row: dict[str, Any] = {'object': res.object, 'n_spectra': len(res.files),
                               'files': ';'.join(map(str, res.files)), 'refitted': bool(res.refitted),
                               'config_hash': res.config_hash, 'version': res.version}
        rvs, errs = [], []
        for m in METHOD_NAMES:
            mres = res.methods.get(m)
            if mres is None:
                row.update({f'rv_{m}': _NAN, f'rv_{m}_err': _NAN, f'rv_{m}_flag': pd.NA, f'n_{m}': pd.NA})
                continue
            row[f'rv_{m}'] = _num(mres.rv)
            row[f'rv_{m}_err'] = _num(mres.rv_err)
            row[f'rv_{m}_flag'] = functools.reduce(operator.or_, (int(x.flag) for x in mres.measurements), 0)
            row[f'n_{m}'] = int(mres.n_regions)
            if np.isfinite(mres.rv):
                rvs.append(float(mres.rv))
                errs.append(_num(mres.rv_err))
        for c in _SPEC:
            if not isinstance(c, str):
                col, section, key, conv, default = c
                row[col] = conv(src[section].get(key, default))
        row['rv_adopted'] = _num(ad.get('median', ad.get('rv')))
        row['n_methods'] = int(ad.get('n_methods', len(res.methods)))
        rv_arr, err_arr = np.array(rvs), np.array(errs)
        good = np.isfinite(err_arr) & (err_arr > 0)
        if good.any():
            w = 1.0 / err_arr[good] ** 2
            row['rv_mean'] = float(np.sum(w * rv_arr[good]) / np.sum(w))
        else:
            row['rv_mean'] = float(rv_arr.mean()) if len(rv_arr) else _NAN
        row['rv_spread'] = float(np.std(rv_arr, ddof=1)) if len(rv_arr) > 1 else _NAN
        sysd = src['systematics']
        row['rv_total_err'] = (_quad(row['rv_adopted_err'], row['sys_template'], row['sys_wavezp'],
                                     row['sys_star_sky'], _num(sysd.get('floor')))
                               if np.isfinite(row['rv_adopted_err']) else _NAN)
        row.update(_template_row(res))
        rows.append(row)
    df = pd.DataFrame(rows, columns=_COLUMNS)
    for m in METHOD_NAMES:
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


_HTML_CSS = ('table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:2px 6px;white-space:nowrap}'
             'th{cursor:pointer}')
_HTML_JS = """
document.querySelectorAll('th').forEach((th, i) => th.onclick = () => {
  const b = th.closest('table').tBodies[0], s = th.dataset.asc = th.dataset.asc === '1' ? '-1' : '1';
  [...b.rows].sort((x, y) => {
    const a = x.cells[i].textContent, c = y.cells[i].textContent;
    return s * (a !== '' && c !== '' && !isNaN(a) && !isNaN(c) ? a - c : a.localeCompare(c));
  }).forEach(r => b.append(r));
});
"""


def _to_html(df: pd.DataFrame, plots: Mapping[str, Any] | None) -> str:
    plots = plots or {}
    head = ''.join(f'<th>{html.escape(c)}</th>' for c in df.columns) + ('<th>plots</th>' if plots else '')
    body = []
    for _, row in df.iterrows():
        cells = ''.join(f'<td>{html.escape(_fmt(row[c]))}</td>' for c in df.columns)
        if plots:
            paths = plots.get(row['object'], ())
            if isinstance(paths, (str, Path)):
                paths = [paths]
            links = ' '.join(f'<a href="{html.escape(str(p), quote=True)}">{html.escape(Path(str(p)).name)}</a>'
                             for p in paths)
            cells += f'<td>{links}</td>'
        body.append(f'<tr>{cells}</tr>')
    return (f'<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><title>rvfitter results</title>'
            f'<style>{_HTML_CSS}</style></head><body><p>Click a column heading to sort.</p>'
            f'<table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table><script>{_HTML_JS}</script></body></html>\n')


def _stem(path: Path) -> Path:
    return path.with_suffix('') if path.suffix.lower() in _FORMATS else path


def _resolve_formats(path: Path, formats: str | Sequence[str] | None) -> list[tuple[str, Path]]:
    if formats is None:
        fmt = _FORMATS.get(path.suffix.lower())
        return [(fmt, path)] if fmt else [('csv', Path(f'{path}.csv'))]
    if isinstance(formats, str):
        formats = [f for f in formats.split(',') if f]
    out = []
    for f in formats:
        f = f.strip().lower().lstrip('.')
        f = 'html' if f == 'htm' else f
        if f not in OUTPUT_FORMATS:
            raise ValueError(f'unknown output format {f!r}')
        out.append((f, Path(f'{_stem(path)}.{f}')))
    return out


def write(results: Sequence[ObjectResult], path: str | Path, formats: str | Sequence[str] | None = None, *,
          config: Any = None, plots: Mapping[str, Any] | None = None) -> list[Path]:
    """
    Write the results and details tables and, if ``config`` is given, the config used; return the files written.

    The format is inferred from the extension of ``path`` unless ``formats`` (csv, json, tsv, html; a list or
    comma-separated) is given. ``plots`` maps an object name to plot path(s), linked from the HTML table.
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
    stem = _stem(path)
    det = Path(f'{stem}_details.csv')
    details_table(results).to_csv(det, index=False)
    written.append(det)
    if config is not None:
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
