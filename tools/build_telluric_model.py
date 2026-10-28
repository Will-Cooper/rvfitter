"""
Build rvfitter's packaged telluric transmission model (`src/rvfitter/data/telluric_hitran.npz`) from HITRAN.

Usage: python tools/build_telluric_model.py OUT.npz CACHE_DIR

Line parameters (vacuum wavenumbers, intensities at 296 K, air- and self-broadening, temperature exponents and
air pressure shifts) are fetched from the HITRANonline line-by-line API (the HAPI endpoint, https://hitran.org/lbl/api)
for H2O, CO2, O2 and CH4, cached in CACHE_DIR, and integrated through a layered model atmosphere:

- temperature: the US Standard Atmosphere 1976; pressure from hydrostatic equilibrium; site at 2400 m (a typical
  high observatory: La Palma 2.3-2.4 km, Paranal 2.6 km);
- dry gases (O2 20.95 per cent, CO2 420 ppm, CH4 1.9 ppm by volume), well mixed, in 24 layers equally spaced in
  ln p from the site to 1 hPa;
- H2O with an exponential number-density profile of scale height 1.5 km above the site, normalised to a precipitable
  water vapour (PWV) of 1 mm, in 12 layers up to 15 km;
- Voigt profiles (Doppler + Lorentz with the HITRAN temperature exponent) including the air pressure shift; line
  intensities scaled to the layer temperature with partition sums Q ∝ T (O2, CO2) or T^1.5 (H2O, CH4);
- line wings cut where the reference optical depth falls below 1e-3, and at 25 cm⁻¹ at most; no continuum
  absorption (MT_CKD), Rayleigh scattering or aerosols, which are smooth and absorbed by any continuum fit.

The optical depths are computed at the zenith on a grid uniform in ln λ (vacuum λ = 1e8/ν Å) at four times the
stored sampling, and stored per pixel of R_s = λ/Δλ = 300 000 as τ_eff = −ln⟨exp(−τ)⟩ (equivalent-width preserving
at the reference conditions), for the dry gases (airmass 1) and for H2O (airmass 1, PWV 1 mm). The model is then
T(λ; X, w) = exp(−X·(τ_dry + w·τ_H2O)), valid for spectral resolutions up to about 100 000.

Licence: HITRAN states no redistribution licence; the data are free of charge, and the citation policy
(https://hitran.org/citepolicy, read 2026-10-28) requires citing the database edition (HITRAN2024: Gordon et al.
2026, JQSRT 353, 109807; HITRAN2020: Gordon et al. 2022, JQSRT 277, 107949). We ship only this derived, smoothed
transmission model, not the line parameters, with the citation in its metadata. The API client approach follows
HAPI (MIT licence; Kochanov et al. 2016, JQSRT 177, 15).
"""
import hashlib
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from scipy.special import voigt_profile

OUT, CACHE = Path(sys.argv[1]), Path(sys.argv[2])
CACHE.mkdir(parents=True, exist_ok=True)

WMIN, WMAX = 6000., 25000.  # vacuum Å
R_STORE, OVERSAMPLE = 300_000, 4
SITE_KM = 2.4
PWV_REF_MM = 1.
H2O_SCALE_KM = 1.5
API = 'https://hitran.org/lbl/api?numin={numin}&numax={numax}&iso_ids_list={ids}'
# global isotopologue ids (HITRAN): H2O 1-4, CO2 7-9, O2 36-38, CH4 32-33
MOLECULES = {
    'H2O': dict(ids=(1, 2, 3, 4), mass=18.01, qexp=1.5, smin=1e-27),
    'CO2': dict(ids=(7, 8, 9), mass=44.0, qexp=1.0, smin=1e-27, vmr=420e-6),
    'O2': dict(ids=(36, 37, 38), mass=32.0, qexp=1.0, smin=1e-29, vmr=0.2095),
    'CH4': dict(ids=(32, 33), mass=16.04, qexp=1.5, smin=1e-25, vmr=1.9e-6),
}
C2 = 1.4387769  # second radiation constant [cm K]
KB, AMU, C_MS, G0 = 1.380649e-23, 1.66053907e-27, 299792458., 9.80665
T_REF = 296.
M_AIR = 28.9647e-3 / 6.02214076e23  # kg per molecule


def fetch(name: str, ids: tuple[int, ...]) -> tuple[Path, str]:
    nu_lo, nu_hi = 1e8 / WMAX - 30, 1e8 / WMIN + 30
    path = CACHE / f'{name}_{nu_lo:.0f}_{nu_hi:.0f}.par'
    if not path.exists():
        parts = []
        edges = np.arange(np.floor(nu_lo), nu_hi + 1000., 1000.)
        for a, b in zip(edges[:-1], edges[1:], strict=True):  # 1000 cm⁻¹ chunks [a, b), three tries each
            url = API.format(numin=f'{a:.0f}', numax=f'{b - 1e-6:.6f}', ids=','.join(map(str, ids)))
            for attempt in range(3):
                try:
                    print(f'fetching {url}', file=sys.stderr)
                    req = urllib.request.Request(url, headers={'User-Agent': 'rvfitter-build/2'})
                    with urllib.request.urlopen(req, timeout=600) as r:
                        try:
                            parts.append(r.read())
                        except http.client.IncompleteRead as exc:
                            # the API's Content-Length counts CRLF but it sends LF: accept the body when the
                            # shortfall is exactly one byte per complete 160-character record
                            data = exc.partial
                            lines = data.splitlines()
                            if exc.expected != len(lines) or any(len(x) != 160 for x in lines):
                                raise
                            parts.append(data)
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code == 404:  # the API answers 404 when the interval holds no lines
                        print('  no lines', file=sys.stderr)
                        break
                    print(f'  failed ({exc}); retrying', file=sys.stderr)
                    time.sleep(5 * (attempt + 1))
                except Exception as exc:  # noqa: BLE001
                    print(f'  failed ({exc}); retrying', file=sys.stderr)
                    time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError(f'could not fetch {url}')
        tmp = path.with_suffix('.tmp')
        tmp.write_bytes(b''.join(p if p.endswith(b'\n') or not p else p + b'\n' for p in parts))
        tmp.rename(path)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def parse(path: Path) -> dict[str, np.ndarray]:
    """HITRAN 160-character records"""
    cols = dict(nu=(3, 15), S=(15, 25), g_air=(35, 40), g_self=(40, 45), elow=(45, 55), n_air=(55, 59),
                d_air=(59, 67))
    out = {k: [] for k in cols}
    for line in path.read_text().splitlines():
        if len(line) < 67:
            continue
        for k, (a, b) in cols.items():
            out[k].append(float(line[a:b]))
    return {k: np.asarray(v) for k, v in out.items()}


def us1976(z_km: np.ndarray) -> np.ndarray:
    """US Standard Atmosphere 1976 temperature [K] below 51 km"""
    z = np.asarray(z_km, dtype=float)
    return np.select([z < 11, z < 20, z < 32, z < 47], [288.15 - 6.5 * z, 216.65 + 0 * z, 216.65 + (z - 20),
                                                        228.65 + 2.8 * (z - 32)], 270.65 + 0 * z)


def pressure_profile() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fine z [km], p [Pa], T [K] by hydrostatic integration from sea level"""
    z = np.linspace(0, 60, 60001)
    t = us1976(z)
    dz = np.diff(z) * 1e3
    lnp = np.concatenate([[0.], -np.cumsum(M_AIR * G0 / (KB * 0.5 * (t[1:] + t[:-1])) * dz)])
    return z, 101325. * np.exp(lnp), t


def layers():
    """Dry and H2O layers: (p [atm], T [K], air column [cm⁻²] or H2O column [cm⁻²])"""
    z, p, t = pressure_profile()
    p_site = np.interp(SITE_KM, z, p)
    edges = np.exp(np.linspace(np.log(p_site), np.log(100.), 25))
    dry = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        n_air = (a - b) / (M_AIR * G0) / 1e4  # molecules cm⁻²
        pm = np.sqrt(a * b)
        zm = float(np.interp(-np.log(pm), -np.log(p), z))  # -ln p increases with z
        dry.append((pm / 101325., float(us1976(zm)), n_air))
    # H2O: n ∝ exp(−(z − site)/H); 1 mm PWV = 0.1 g cm⁻² = 0.1/(18.015/N_A) molecules cm⁻²
    n_tot = PWV_REF_MM * 0.1 / (18.015 / 6.02214076e23)
    zed = np.linspace(SITE_KM, 15., 13)
    frac = np.diff(-np.exp(-(zed - SITE_KM) / H2O_SCALE_KM))
    frac /= frac.sum()
    wet = []
    for a, b, f in zip(zed[:-1], zed[1:], frac, strict=True):
        zm = 0.5 * (a + b)
        wet.append((float(np.interp(zm, z, p)) / 101325., float(us1976(zm)), n_tot * f))
    return dry, wet


def tau_species(lines: dict, mol: dict, lay: list, column_scale: float, nu_grid: np.ndarray) -> np.ndarray:
    """Zenith optical depth on `nu_grid` (descending wavenumber order is handled by sorting)"""
    order = np.argsort(nu_grid)
    nu_s = nu_grid[order]
    tau = np.zeros(len(nu_s))
    keep = lines['S'] >= mol['smin']
    nu0, s0, ga, gs, el, na, da = (lines[k][keep] for k in ('nu', 'S', 'g_air', 'g_self', 'elow', 'n_air', 'd_air'))
    el = np.where(el < 0, 0., el)
    p = np.array([x[0] for x in lay])
    t = np.array([x[1] for x in lay])
    col = np.array([x[2] for x in lay]) * column_scale
    q = (t / T_REF) ** mol['qexp']
    n_ref = col.sum()
    print(f"  {len(nu0)} lines, {len(lay)} layers, column {n_ref:.3g} cm-2", file=sys.stderr)
    for i in range(len(nu0)):
        # per-layer line parameters
        s_t = (s0[i] / q * np.exp(-C2 * el[i] * (1 / t - 1 / T_REF))
               * (1 - np.exp(-C2 * nu0[i] / t)) / (1 - np.exp(-C2 * nu0[i] / T_REF)))
        gam = ga[i] * p * (T_REF / t) ** na[i]  # Lorentz HWHM [cm⁻¹]
        sig = nu0[i] / C_MS * np.sqrt(KB * t / (mol['mass'] * AMU))  # Doppler σ [cm⁻¹]
        centre = nu0[i] + da[i] * p
        strength = float(np.sum(s_t * col))
        g0 = float(np.max(gam))
        half = np.sqrt(strength * g0 / (np.pi * 1e-3)) if g0 > 0 else 0.
        half = min(max(half, 6 * float(sig.max()), 4 * g0, 0.05), 25.)
        lo, hi = np.searchsorted(nu_s, [nu0[i] - half, nu0[i] + half])
        if hi <= lo:
            continue
        x = nu_s[lo:hi][None, :] - centre[:, None]
        tau[lo:hi] += np.sum((s_t * col)[:, None] * voigt_profile(x, sig[:, None], gam[:, None]), axis=0)
    out = np.empty_like(tau)
    out[order] = tau
    return out


def main() -> None:
    t0 = time.time()
    dry, wet = layers()
    dl = 1. / (R_STORE * OVERSAMPLE)
    n_store = int(np.floor(np.log(WMAX / WMIN) * R_STORE))
    lnw = np.log(WMIN) + (np.arange(n_store * OVERSAMPLE) + 0.5) * dl
    nu_grid = 1e8 / np.exp(lnw)
    taus, prov = {}, {}
    for name, mol in MOLECULES.items():
        path, sha = fetch(name, mol['ids'])
        lines = parse(path)
        print(f'{name}: {len(lines["nu"])} lines in the file', file=sys.stderr)
        if name == 'H2O':
            taus[name] = tau_species(lines, mol, wet, 1., nu_grid)
        else:
            taus[name] = tau_species(lines, mol, dry, mol['vmr'], nu_grid)
        prov[name] = dict(file=path.name, sha256=sha, n_lines_file=int(len(lines['nu'])),
                          n_lines_used=int(np.sum(lines['S'] >= mol['smin'])), s_min=mol['smin'],
                          isotopologue_ids=list(mol['ids']), vmr=mol.get('vmr'))
    dry_tau = taus['O2'] + taus['CO2'] + taus['CH4']

    def store(tau):  # equivalent-width preserving average of the transmission over each stored pixel
        tr = np.exp(-tau).reshape(n_store, OVERSAMPLE).mean(axis=1)
        return np.clip(-np.log(np.clip(tr, 1e-300, 1.)), 0., 6e4).astype(np.float16)

    meta = dict(
        description='Zenith telluric optical depths from HITRAN line-by-line data through a layered model '
                    'atmosphere: T = exp(-airmass*(tau_dry + pwv_mm*tau_h2o)); vacuum wavelengths; '
                    'stored pixels are uniform in ln(lambda): ln(lambda_i) = lnlam0 + (i + 0.5)*dlnlam',
        source='HITRANonline line-by-line API (https://hitran.org/lbl/api), fetched 2026-10-28',
        citation='Gordon et al. 2026, JQSRT 353, 109807 (HITRAN2024); Gordon et al. 2022, JQSRT 277, 107949 '
                 '(HITRAN2020)',
        licence='HITRAN data are free of charge and must be cited (https://hitran.org/citepolicy); no '
                'redistribution licence is stated. Only this derived transmission model is shipped.',
        builder='tools/build_telluric_model.py', site_km=SITE_KM, atmosphere='US Standard 1976',
        h2o_scale_km=H2O_SCALE_KM, pwv_ref_mm=PWV_REF_MM, r_store=R_STORE, oversample=OVERSAMPLE,
        wmin=WMIN, wmax=WMAX, molecules=prov,
        omitted='O3, N2O, CO, collision-induced and continuum absorption, Rayleigh scattering, aerosols')
    np.savez_compressed(OUT, lnlam0=np.log(WMIN), dlnlam=1. / R_STORE, n=n_store, tau_dry=store(dry_tau),
                        tau_h2o=store(taus['H2O']), meta=json.dumps(meta))
    print(f'wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB) in {time.time() - t0:.0f} s', file=sys.stderr)


if __name__ == '__main__':
    main()
