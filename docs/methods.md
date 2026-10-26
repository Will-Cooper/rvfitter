# Methods and benchmark

This document maps each radial-velocity (RV) technique in [`literature.md`](literature.md) to its implementation in rvfitter v2, and records results on the primary benchmark.
The method sections are filled in as each technique lands.

## Benchmark: J1745−1640

DENIS J174534.6−164053 (Gaia DR3 4123874907297370240; L1.5; *G*\,=\,18.42 mag; parallax 50.9 mas).
It was chosen because it was the test object of Cooper et al. (2024), who validated their RVs against independent IRAF/IDL measurements of it.

### Published radial velocities

| Value [km/s] | Method | Instrument | Reference | Status |
|---|---|---|---|---|
| 28 ± 9 | Line centres (K I, Rb I, Na I, Cs I, Hα) | Magellan/MagE, *R*~4100, 2009 May 31 | Burgasser et al. 2015, ApJS 220, 18 (Table 6) | Same spectrum as below |
| 26.2 ± 1.1 | Cross-correlation against MagE RV standards | as above | Burgasser et al. 2015 | Same spectrum |
| **26.2 ± 2.3** | Adopted: CCF plus 2 km/s systematic in quadrature | as above | Burgasser et al. 2015 | **Independent** |
| 26 ± 2 | Rounded re-quote | SIMBAD | — | Re-quote |
| 36.2 ± 4.4 | Line centring | GTC/OSIRIS R2500I, 2015 Aug 4 | Cooper et al. 2024, MNRAS 534, 695 | Same spectrum as below |
| 28.8 ± 4.7 | By-eye template shift ("cross-correlation") | as above | Cooper et al. 2024 | Same spectrum |
| **32.7 ± 6.5** | Adopted | as above | Cooper et al. 2024 | **Independent** |
| 35.1 ± 4.3, 29.0 ± 3.8 | IDL Gaussian and IRAF Splot Voigt line centres | GTC/OSIRIS (IRAF reduction) | Cooper et al. 2024, appendix | Same data |

No RV was found in Gaia DR3, APOGEE DR17 (a weak check) or Hsu et al. (2024).
SIMBAD's attribution of 26 ± 2 to Hsu et al. (2021) appears to be a compilation re-quote; Hsu et al. (2021) is a T-dwarf study.

### Combination of independent measurements

The two independent measurements are 26.2 ± 2.3 and 32.7 ± 6.5 km/s.
Their inverse-variance weighted mean is 26.9 ± 2.2 km/s.
They differ by 6.5 ± 6.9 km/s (0.9σ; χ² = 0.88 for one degree of freedom), so there is no evidence of variability or binarity.

### Adopted benchmark and acceptance criterion

We adopt **26.2 ± 2.3 km/s** (Burgasser et al. 2015).
This is because it is the only precise measurement with a standard-calibrated error, and the combined value barely differs from it.
A method passes if its RV lies within ±5 km/s (≈2σ) of 26.2 km/s **and** its quoted error is consistent with the offset.
We do not tune any method towards this number: the criterion is checked, not fitted.

The original Cooper et al. (2024) values sit at:
- line centring 36.2 ± 4.4: +10.0 km/s (2.0σ);
- by-eye cross-correlation 28.8 ± 4.7: +2.6 km/s (0.5σ);
- adopted 32.7 ± 6.5: +6.5 km/s (0.9σ).

The high line-centring value matches the large scatter that Burgasser et al. (2015) also found from line centres (28 ± 9 km/s).

### Results by method

| Method | v1 (paper) | v1 after Phase A fixes | v2 | Offset from benchmark |
|---|---|---|---|---|
| Line centring | 36.2 ± 4.4 | | | |
| Cross-correlation | 28.8 ± 4.7 (by eye) | | | |
| Template fit | — | | | |
| Forward model (MCMC) | — | | | |
| LSD | — | | | |
| Telluric-referenced | — | | | |
| Adopted | 32.7 ± 6.5 | | | |
