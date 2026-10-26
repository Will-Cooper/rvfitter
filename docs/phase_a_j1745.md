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
| A2 Doppler constant | 36.2 ± 4.4 (36.173) | 2.01 | 28.8 ± 4.7 (28.750) | 0.49 | 32.71 ± 6.47 (32.714 ± 6.471) | 0.95 | Line centring now uses c instead of c/n_air, so every per-line RV and error grows by a factor of 1.000276 (+0.01 km/s; e.g. Na I b 45.23 → 45.25). The cross-correlation is unchanged (to <1e-6 km/s), because specutils already shifts the template with the vacuum c (`radial_velocity` setter); `inv_rv_calc` only enters the xcorr error and plot marker. |
| A3 error propagation | 36.71 ± 4.65 | 2.02 | 28.66 ± 5.21 | 0.43 | 28.95 ± 5.45 (hierarchical median and central 68 per cent half-width) | 0.46 | Per-line errors, per-method errors and the adopted combination are all redone; see "A3 in detail" below. |
| A4 per-line bookkeeping | 36.71 ± 4.65 | 2.02 | 28.66 ± 5.21 | 0.43 | 28.95 ± 5.45 | 0.46 | Bit-identical to A3, as expected: all eight lines are accepted by both methods and the cached fits are stored in `spec_indices` order, so neither the compressed slots nor the list-position matching affected J1745−1640. Rejected lines now stay NaN in their own slot, each fit is matched to its line by name, and `adoptedrv` checks that every per-line array has one slot per line. The Reviewer's A3 follow-ups are folded in here: `hierarchical_combine` now uses the Woodbury identity and the matrix determinant lemma (memory ∝ ngrid × r², with r = number of lines + methods, instead of ngrid² × n²), which reproduces μ, its SD and quantiles to ≤1e-8 km/s; the τ medians now use the mid-point cumulative distribution, which moves them by about half a grid cell (τ_line 12.55 → 12.89 km/s, τ_meth 1.72 → 1.76 km/s) without changing μ. |
| A5 driver fixes | 36.71 ± 4.65 | 2.02 | 28.66 ± 5.21 | 0.43 | 28.95 ± 5.45 | 0.46 | Bit-identical to A4. The Teff range check now works (`utils.expected_teff`: rounded to 100 K, 2000 K if NaN or outside 1200–4000 K); for this object `stephens` already gives a valid value. `--systematic` is parsed as a float in both drivers (`build_parser`). The SpeX target name removes the `.fits` suffix rather than any of its characters. Templates are chosen by nearest Teff, then log g, then [M/H] among the files that exist (`utils.nearest_template`), and any substitution is logged and recorded; J1745−1640 uses exact grid points, so nothing changed. `fullrv_spex` now calls the shared `fullrv.adoptedrv` (new `outname` and `wunit` arguments; `hires` removed), so SpeX objects also get the A3 hierarchical combination. An object measured by only one method now gets an adopted RV: a factor with a single level (one method or one line) has its τ fixed at 0 because its offset cannot be separated from μ, a warning says that the between-method systematic is not estimable, and the table records `rv_adopted_nmethods` and `rv_adopted_nlines`. |

## A3 in detail

**Per-line errors.**
For line centring, the wavelength rms (0.077 Å) is now added to the fit error in quadrature rather than linearly, so the per-line errors drop by 1.3–2.7 km/s (e.g. Cs I a 4.51 → 3.25 km/s).
The weights therefore shift towards the well-fitted Cs I a and Na I b lines, which moves the line-centring mean from 36.17 to 36.71 km/s.
For the cross-correlation, the per-line error is now √((Δv/√12)² + (c·0.077 Å/λ)²) = 3.9–4.2 km/s for the step Δv = 10 km/s, instead of the 5.0 km/s of the paper.
The first term is the standard deviation of a uniform quantisation error over one step, and the second is the wavelength rms as a velocity (the old wavelength term was ≈0.002 km/s, a unit error).
The SNR-based floor `rvunc_min` = c/(R·SNR·√Npix), with R = λ/FWHM of the fitted profile, is computed and stored for each line but is not applied to the errors.

**Per-method combination.**
Each method is the inverse-variance mean of its lines, with the formal error 1/√Σw inflated by √(χ²/(N−1)) when the lines scatter more than their errors allow (never deflated).
For J1745−1640 the scatter dominates for both methods (χ²/(N−1) ≫ 1): line centring 36.71 ± 4.65 km/s and cross-correlation 28.66 ± 5.21 km/s.

**Adopted RV: hierarchical line × method model.**
We model every per-line measurement as v_{m,i} = μ + δ_i + β_m + ε_{m,i}, with a line offset δ_i ~ N(0, τ_line²) shared by both methods, a method offset β_m ~ N(0, τ_meth²) shared by all lines and the measurement error ε_{m,i} ~ N(0, σ_{m,i}²).
We marginalise δ and β analytically, take a flat prior on μ and half-Cauchy priors on both τ, with a scale equal to the median per-measurement error (4.16 km/s here), and integrate the τ on a fixed 200 × 200 grid (`utils.hierarchical_combine`).
Because this default scale is taken from the data, the prior is a mild form of empirical Bayes; the sensitivity to it is quantified below.
The posterior of μ is then a mixture of Gaussians, which is what the adopted-RV figure now shows as the black curve.
With only two methods, τ_meth is constrained by a single difference, so the posterior of μ has power-law tails and its standard deviation grows without limit as the τ grid is extended (6.75, 7.14, 7.95 and 8.69 km/s for grids reaching 10, 100, 1000 and 10000 times the prior scale), whereas the quantiles do not move.
As a result, we adopt the posterior median and half the central 68 per cent interval: 28.95 ± 5.45 km/s (16 and 84 per cent quantiles 23.48 and 34.39 km/s; posterior mean 28.94 km/s).
The posterior medians of the scatter terms are τ_line = 12.6 km/s and τ_meth = 1.7 km/s.
The line-to-line scatter therefore dominates: both methods share the same line offsets (e.g. Cs I a is high, 48 and 55 km/s, and Cs I b low, 11 and 5 km/s), so the lines are weighted almost equally and μ lies close to the unweighted mean of the per-line averages (29.2 km/s).
The two methods themselves agree within the model (small τ_meth), so the line-centring mean is high mainly because its errors give most weight to Cs I a and Na I b.

Prior sensitivity (not tuned):

| Half-Cauchy scale | Median μ | 16–84 per cent | Half-width | τ_line | τ_meth | z |
|---|---|---|---|---|---|---|
| ×0.5 (2.08 km/s) | 28.95 | 23.82–34.03 | 5.11 | 12.4 | 1.1 | 0.49 |
| ×1 (4.16 km/s, default) | 28.95 | 23.48–34.39 | 5.45 | 12.6 | 1.7 | 0.46 |
| ×2 (8.32 km/s) | 28.95 | 22.83–35.03 | 6.10 | 12.9 | 2.6 | 0.42 |

The median does not depend on the prior scale; the half-width changes by −6 and +12 per cent, through τ_meth.

Comparison of combinations on the same A3 per-line values:

| Combination | Adopted [km/s] | z |
|---|---|---|
| Two method means, independent (ρ = 0) | 33.14 ± 3.47 | 1.67 |
| Two method means, GLS with ρ = 0.89 from the paired lines | 36.66 ± 4.65 | 2.01 |
| All 16 measurements pooled, χ²-inflated inverse variance | 31.58 ± 3.57 | 1.27 |
| Hierarchical line × method (adopted) | 28.95 ± 5.45 | 0.46 |

The independent and pooled combinations ignore the shared line offsets and so understate the error; GLS accounts for the correlation but, with ρσ_XC/σ_LC ≈ 1, puts almost all the weight on line centring.

**Choices changed on their merits rather than kept for continuity with the paper:**
1. The by-eye cross-correlation quantisation term is Δv/√12 (the standard deviation of a uniform error over one step), not Δv/2.
2. Each method's error is the formal error inflated by √(χ²/(N−1)), not the weighted standard deviation divided by √N.
3. The adopted RV comes from the hierarchical model (median and central 68 per cent interval), not the inverse-variance mean of the two methods weighted by their line scatter and divided by √2.
4. The method RVs and errors are stored unrounded in the table (they were rounded to 0.1 km/s).

Still as in the paper, and left for Phase C: the Voigt FWHM approximation fL/2 + √(fL²/4 + fG²) (it only enters `rvunc_min` and the plotted line width), and `rvunc_min` is not applied as a floor.

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
We located the line minima (parabola through the deepest pixels) in the raw `rvfitter-testdata/templates/bt-settl-cifist/raw/lte020.0-5.0-0.0a+0.0.BT-Settl.spec.7.dat.txt` and in the trimmed `rvfitter-testdata/templates/bt-settl-cifist/useful/lte020.0-5.0.txt`, and compared them with the NIST wavelengths (air, and converted to vacuum with Edlén 1953).

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
This follows the convention of the Allard BT-Settl `spec.7` files (vacuum below 2000 Å, air above).
Other grids differ: the Göttingen PHOENIX-ACES library (Husser et al. 2013) is in vacuum, for example.
As a result, the medium must be declared per grid in v2 rather than assumed, and templates downloaded from SVO must be checked in the same way.
