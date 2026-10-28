import json
import re

import numpy as np
import pandas as pd
import pytest

from rvfitter import results
from rvfitter.types import Measurement, MethodResult, ObjectResult


def make_result(name='J1745', rv=27.4, with_lsd=False):
    ms = {}
    for m, off in (('linecentre', 0.5), ('ccf', -0.3)):
        meas = [Measurement(m, 'K I 7701', rv + off, 2.0, 0, {'chi2': 1.1, 'note': 'x', 'arr': [1, 2]}),
                Measurement(m, 'Na I 8200', rv + off + 0.2, 3.0, 4)]
        ms[m] = MethodResult(m, rv + off, 1.7, 2, meas)
    if with_lsd:
        ms['lsd'] = MethodResult('lsd', rv, 1.0, 1, [Measurement('lsd', 'all', rv, 1.0, 32)])
    return ObjectResult(
        object=name, files=['a.fits', 'b.fits'], methods=ms,
        adopted={'median': rv, 'err': 1.5, 'q16': rv - 1.5, 'q84': rv + 1.5, 'tau_line': 0.4, 'tau_method': 0.2,
                 'n_methods': 2, 'n_lines': 4},
        systematics={'template_spread': 0.8, 'wavelength_zero_point': 0.6},
        barycentric={'applied_kms': -12.3, 'source': 'header', 'frame': 'barycentric'},
        binary={'dbic': 3.2, 'flag': False}, config_hash='abc123', version='2.0.0')


def test_schema():
    df = results.to_table([make_result(), make_result('B', with_lsd=True)])
    cols = list(df.columns)
    assert cols[:3] == ['object', 'n_spectra', 'files']
    assert cols[3:7] == ['rv_linecentre', 'rv_linecentre_err', 'rv_linecentre_flag', 'n_linecentre']
    assert cols[-2:] == ['config_hash', 'version']
    order = [cols.index(f'rv_{m}') for m in results.METHOD_ORDER]
    assert order == sorted(order)
    assert len(df) == 2
    assert np.isnan(df.loc[0, 'rv_template_fit']) and pd.isna(df.loc[0, 'rv_lsd_flag'])
    assert df.loc[0, 'rv_linecentre_flag'] == 4
    assert df.loc[1, 'rv_lsd_flag'] == 32
    assert df.loc[0, 'rv_total_err'] == pytest.approx(np.sqrt(1.5**2 + 0.8**2 + 0.6**2))
    assert df.loc[0, 'rv_spread'] == pytest.approx(np.std([27.9, 27.1], ddof=1))
    assert df.loc[0, 'rv_mean'] == pytest.approx(27.5)
    assert df.loc[0, 'files'] == 'a.fits;b.fits'


def test_details():
    d = results.details_table([make_result()])
    assert len(d) == 4
    assert 'diag_chi2' in d.columns and 'diag_arr' not in d.columns


def test_flags():
    assert results.FLAGS['fit_failed'] == 1 and results.FLAGS['not_converged'] == 32
    assert results.flag_names(5) == ['fit_failed', 'error_floor']


@pytest.mark.parametrize('ext', ['csv', 'tsv', 'json', 'html'])
def test_write_roundtrip(tmp_path, ext):
    res = [make_result(), make_result('B', with_lsd=True)]
    written = results.write(res, tmp_path / f'out.{ext}', config={'a': 1},
                            dump=lambda c, p: p.write_text('a = 1\n'))
    assert tmp_path / f'out.{ext}' in written
    assert (tmp_path / 'out_details.csv').exists() and (tmp_path / 'out_config.toml').exists()
    ref = results.to_table(res)
    if ext == 'csv':
        back = pd.read_csv(tmp_path / 'out.csv')
    elif ext == 'tsv':
        back = pd.read_csv(tmp_path / 'out.tsv', sep='\t')
    elif ext == 'json':
        text = (tmp_path / 'out.json').read_text()
        assert 'NaN' not in text
        back = pd.DataFrame(json.loads(text))
    else:
        assert (tmp_path / 'out.html').read_text().count('<tr>') == 3
        return
    assert list(back.columns) == list(ref.columns)
    assert back['object'].tolist() == ref['object'].tolist()
    for c in ('rv_adopted', 'rv_ccf', 'rv_lsd', 'rv_total_err'):
        np.testing.assert_allclose(back[c].astype(float), ref[c].astype(float), equal_nan=True)
    assert back['rv_lsd_flag'].isna().tolist() == [True, False]


def test_multiple_formats(tmp_path):
    w = results.write([make_result()], tmp_path / 'r', formats='csv,json,html')
    assert {p.suffix for p in w} == {'.csv', '.json', '.html'}


def test_unknown_format(tmp_path):
    with pytest.raises(ValueError):
        results.write([make_result()], tmp_path / 'r', formats='xlsx')


def test_html_self_contained(tmp_path):
    results.write([make_result()], tmp_path / 'o.html', plots={'J1745': ['plots/J1745.png']})
    text = (tmp_path / 'o.html').read_text()
    assert not re.search(r'(https?:)?//', text)
    assert 'src=' not in text and 'plots/J1745.png' in text and '<script>' in text


def test_empty_results(tmp_path):
    results.write([], tmp_path / 'e.csv')
    assert results.summary_lines([])[0].startswith('object')


def test_summary_lines():
    lines = results.summary_lines([make_result(), make_result('longer_name')])
    assert len(lines) == 3 and len({len(x) for x in lines}) == 1


def test_dotted_stem_and_floor(tmp_path):
    res = make_result()
    res.systematics['floor'] = 0.5
    w = results.write([res], tmp_path / 'run.v2.csv')
    assert (tmp_path / 'run.v2_details.csv') in w
    assert results.to_table([res]).loc[0, 'rv_total_err'] == pytest.approx(np.sqrt(1.5**2 + 0.8**2 + 0.6**2 + 0.5**2))
    w = results.write([res], tmp_path / 'run.v2', formats='json')
    assert (tmp_path / 'run.v2.json') in w


def test_refitted_column():
    plain, refit = make_result(), make_result('B')
    refit.refitted = True
    df = results.to_table([plain, refit])
    assert df['refitted'].tolist() == [False, True] and df['refitted'].dtype == bool
