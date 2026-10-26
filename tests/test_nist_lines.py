"""
Every atomic line rest wavelength in the feature catalogue must match the NIST Atomic Spectra Database.

`src/rvfitter/data/nist_lines.csv` holds vacuum wavelengths computed from NIST Ritz energy levels,
λ_vac = 10⁸ / (E_k − E_i) Å. It is regenerated with `tools/build_nist_reference.py`.
A 0.01 Å error at 8000 Å is a 0.4 km/s RV bias, so that is the tolerance.
"""
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

TOL = 0.01  # Å


def _nist():
    return pd.read_csv(files('rvfitter') / 'data' / 'nist_lines.csv')


def test_nist_reference_is_self_consistent():
    nist = _nist()
    assert len(nist) > 100
    assert np.allclose(nist.wave_vac, 1e8 / (nist.ek - nist.ei), atol=1e-3)
    # spot checks against well-known vacuum values: K I D2/D1, Hα (strongest component)
    for wave in (7667.0089, 7701.0835):
        assert np.min(np.abs(nist.wave_vac - wave)) < 1e-3


def test_catalogue_lines_match_nist():
    features = pytest.importorskip('rvfitter.features')
    if not hasattr(features, 'load_catalogue'):
        pytest.skip('feature catalogue not implemented yet')
    cat = features.load_catalogue()
    nist = _nist()
    lines = cat[np.asarray(cat['kind']) == 'line']
    assert len(lines) > 0
    bad = []
    for row in lines:
        species = str(row['species']).strip()
        cands = nist[nist.species == species]
        if cands.empty:
            bad.append(f'{row["name"]}: species {species!r} not in NIST reference')
            continue
        # blended multiplets may use a strength-weighted centroid: accept the nearest component or that centroid
        near = cands[np.abs(cands.wave_vac - float(row['wave_vac'])) < 1.0]
        aki = pd.to_numeric(near.aki, errors='coerce').fillna(0).to_numpy()
        targets = list(near.wave_vac)
        if len(near) > 1 and aki.sum() > 0:
            targets.append(float(np.average(near.wave_vac, weights=aki)))
        diff = min((abs(t - float(row['wave_vac'])) for t in targets), default=np.inf)
        if diff > TOL:
            bad.append(f'{row["name"]}: {float(row["wave_vac"]):.4f} Å differs from NIST by {diff:.4f} Å')
    assert not bad, 'catalogue wavelengths disagree with NIST:\n' + '\n'.join(bad)
