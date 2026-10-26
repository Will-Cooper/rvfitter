# Phase A: J1745−1640 benchmark through the legacy bug fixes

Here, we record how each Phase A legacy fix moves the radial velocities (RVs) of the benchmark object J1745−1640 (DENIS J174534.6−164053, L1.5; GTC/OSIRIS R2500I, OB0055).
Every row was produced by the harness `run_legacy_j1745.py`, which calls the legacy `fullrv.main()` non-interactively (`repeat=False`) on `pypeit/R2500I/GTC54-15A0/OB0055/Science/DENISJ174534.6-164053_tellcorr.fits`.
It uses the by-hand fits cached in `lines_used.json` and `xcorr_used.json` from the original working directory, so the line windows, profiles and by-eye cross-correlation shifts are identical in every row.
The line-centring profiles are refitted in every run, whereas the per-line cross-correlation shifts are the cached by-eye values; their errors and all combinations are recomputed.
The stored `thisrv*` columns are blanked before each run, so every number is freshly computed.

Reference values: Cooper et al. (2024) line centring 36.2 ± 4.4 km/s, by-eye cross-correlation 28.8 ± 4.7 km/s, adopted 32.7 ± 6.5 km/s; literature (Burgasser et al. 2015, MagE) 26.0 ± 2.0 km/s.
All values are in km/s.
Each method is compared with the adopted benchmark (26.2 ± 2.3 km/s; see `docs/methods.md`) through z = |RV − 26.2| / √(2.3² + σ_RV²), computed from the unrounded mean and the quoted error.
Agreement is good for z ≤ 1 and acceptable for z ≤ 2.

| Unit | Line centring | z_LC | Cross-correlation | z_XC | Adopted | z_adopted | Note |
|---|---|---|---|---|---|---|---|
| A0 (main @ 6df73ec, before fixes) | 36.2 ± 4.4 (36.163 unrounded) | 2.01 | 28.8 ± 4.7 (28.750 unrounded) | 0.49 | 32.71 ± 6.47 | 0.95 | Reproduces the paper exactly (adopted 32.709357 ± 6.469662, as stored in `gtc_fullinfo.csv`). |
| A1 flux errors | 36.2 ± 4.4 (36.163) | 2.01 | 28.8 ± 4.7 (28.750) | 0.49 | 32.71 ± 6.47 | 0.95 | Bit-identical to A0. The flux errors weight none of the fits (the astropy fitters are unweighted and the SNR estimate uses the continuum scatter), and the only two ivar = 0 pixels are the first and last (7318 and 10145 Å, vacuum, as in the raw PypeIt file), outside every window. Only the plotted error bars change (σ = 1/√ivar instead of 1/ivar). |
| A2 Doppler constant | 36.2 ± 4.4 (36.173) | 2.01 | 28.8 ± 4.7 (28.750) | 0.49 | 32.71 ± 6.47 (32.714 ± 6.471) | 0.95 | Line centring now uses c instead of c/n_air, so every per-line RV and error grows by a factor of 1.000276 (+0.01 km/s; e.g. Na I b 45.23 → 45.25). The cross-correlation is unchanged, because specutils already shifts the template with the vacuum c (`radial_velocity` setter); `inv_rv_calc` only enters the xcorr error and plot marker. |

## A0 per-line values

| Line | λ_air [Å] | LC RV | LC err | Xcorr RV | Xcorr err |
|---|---|---|---|---|---|
| K I a | 7664.90 | 30.01 | 18.53 | 30.0 | 5.0 |
| K I b | 7698.96 | 27.79 | 10.40 | 20.0 | 5.0 |
| Rb I a | 7800.27 | 35.21 | 7.24 | 35.0 | 5.0 |
| Rb I b | 7947.60 | 20.44 | 9.23 | 25.0 | 5.0 |
| Na I a | 8183.26 | 19.67 | 6.99 | 25.0 | 5.0 |
| Na I b | 8194.82 | 45.23 | 6.03 | 35.0 | 5.0 |
| Cs I a | 8521.13 | 47.94 | 4.51 | 55.0 | 5.0 |
| Cs I b | 8943.47 | 11.06 | 11.42 | 5.0 | 5.0 |

All eight lines are accepted by both methods for this object, so the compressed-index labelling bug (A4) does not affect J1745−1640.
The wavelength RMS from `wvcalib.fits` is 0.077 Å (0.0552 pix × 1.392 Å/pix).


## Wavelength medium of the BT-Settl templates

We checked empirically which medium the BT-Settl-CIFIST templates are in, because the legacy `freader` converts the (vacuum) PypeIt spectra to air but leaves the `.txt` templates untouched.
We located the line minima (parabola through the deepest pixels) in the raw `lte020.0-5.0-0.0a+0.0.BT-Settl.spec.7.dat.txt` and in the trimmed `useful/lte020.0-5.0.txt`, and compared them with the NIST wavelengths (air, and converted to vacuum with Edlén 1953).

| Line | NIST air [Å] | NIST vacuum [Å] | Raw minimum [Å] | `useful/` minimum [Å] | Minimum − air [Å] | Minimum − vacuum [Å] |
|---|---|---|---|---|---|---|
| K I a | 7664.899 | 7667.009 | 7664.911 | 7664.911 | +0.012 | −2.097 |
| K I b | 7698.965 | 7701.084 | 7698.967 | 7698.967 | +0.002 | −2.117 |
| Rb I a | 7800.268 | 7802.414 | 7800.259 | 7800.259 | −0.009 | −2.155 |
| Rb I b | 7947.603 | 7949.789 | 7947.595 | 7947.595 | −0.008 | −2.194 |
| Na I a | 8183.256 | 8185.505 | 8183.251 | 8183.251 | −0.005 | −2.255 |
| Na I b | 8194.824 | 8197.077 | 8194.814 | 8194.814 | −0.010 | −2.263 |
| Cs I a | 8521.132 | 8523.473 | 8521.119 | 8521.119 | −0.013 | −2.354 |
| Cs I b | 8943.474 | 8945.930 | 8943.473 | 8943.473 | −0.001 | −2.457 |

Both files are in air: every minimum lies within 0.013 Å (0.5 km/s) of the NIST air wavelength and 2.1–2.5 Å (≈82 km/s) below the vacuum one.
The `useful/` file is a straight trim of the raw file (4000–11000 Å; every one of its wavelengths appears exactly in the raw file), so no conversion was applied when it was made.
As a result, the legacy cross-correlation compared an air-converted spectrum with an air template, which is consistent, and the paper's 28.8 km/s carries no 80 km/s medium offset.
This follows the PHOENIX convention (vacuum below 2000 Å, air above), so the medium must be declared per grid in v2 rather than assumed; templates downloaded from SVO must be checked in the same way.
