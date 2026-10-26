# Phase A: J1745−1640 benchmark through the legacy bug fixes

Here, we record how each Phase A legacy fix moves the radial velocities (RVs) of the benchmark object J1745−1640 (DENIS J174534.6−164053, L1.5; GTC/OSIRIS R2500I, OB0055).
Every row was produced by the harness `run_legacy_j1745.py`, which calls the legacy `fullrv.main()` non-interactively (`repeat=False`) on `pypeit/R2500I/GTC54-15A0/OB0055/Science/DENISJ174534.6-164053_tellcorr.fits`.
It uses the by-hand fits cached in `lines_used.json` and `xcorr_used.json` from the original working directory, so the line windows, profiles and by-eye cross-correlation shifts are identical in every row.
The stored `thisrv*` columns are blanked before each run, so every number is freshly computed.

Reference values: Cooper et al. (2024) line centring 36.2 ± 4.4 km/s, by-eye cross-correlation 28.8 ± 4.7 km/s, adopted 32.7 ± 6.5 km/s; literature (Burgasser et al. 2015, MagE) 26.0 ± 2.0 km/s.
All values are in km/s.

| Unit | Line centring | Cross-correlation | Adopted | Note |
|---|---|---|---|---|
| A0 (main @ 6df73ec, before fixes) | 36.2 ± 4.4 (36.163 unrounded) | 28.8 ± 4.7 (28.750 unrounded) | 32.71 ± 6.47 | Reproduces the paper exactly (adopted 32.709357 ± 6.469662, as stored in `gtc_fullinfo.csv`). |

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
