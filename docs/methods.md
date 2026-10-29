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
We compare each method with the benchmark through the difference divided by the two errors added in quadrature:
z = |RV − 26.2| / √(2.3² + σ_RV²).
Agreement is good for z ≤ 1 and acceptable for z ≤ 2; anything above 2 is investigated and explained individually.
Because a large quoted error would make z small trivially, error honesty is checked separately: the injection–recovery tests require calibrated uncertainties (pull distribution width 0.8–1.2).
We do not tune any method towards this number: the criterion is checked, not fitted.

On this criterion, the original Cooper et al. (2024) values sit at:
- line centring 36.2 ± 4.4: +10.0 / √(2.3² + 4.4²) = 10.0 / 5.0, z = 2.0;
- by-eye cross-correlation 28.8 ± 4.7: +2.6 / 5.2, z = 0.5;
- adopted 32.7 ± 6.5: +6.5 / 6.9, z = 0.9, i.e. good agreement.

The high line-centring value matches the large scatter that Burgasser et al. (2015) also found from line centres (28 ± 9 km/s).

### Results by method

| Method | v1 (paper) | v1 after Phase A fixes | v2 | z (quadrature) |
|---|---|---|---|---|
| Line centring | 36.2 ± 4.4 | | | |
| Cross-correlation | 28.8 ± 4.7 (by eye) | | | |
| Template fit | — | | | |
| Forward model (MCMC) | — | | | |
| LSD | — | | | |
| Telluric-referenced | — | | | |
| Adopted | 32.7 ± 6.5 | | | |

## Barycentric correction of the GTC spectra

The GTC/OSIRIS spectra used by Cooper et al. (2024) were corrected to the heliocentric frame by `PypeIt` 1.7.0 during reduction.
rvfitter v2 recomputes that correction for J1745−1640 from the header coordinates and mid-exposure time, at the Roque de los Muchachos site.
It finds −20.851 km/s heliocentric (−20.858 km/s barycentric).
`PypeIt` applied −21.006 km/s (`VEL_CORR` = 0.9999299356).

The 0.155 km/s difference traces to the GTC headers: they record `LON-OBS` = +17.877°, but GTC lies at 17.89° **west**.
Every GTC RV in the paper sample therefore probably carries a sub-0.2 km/s offset from this.
That is well below the quoted uncertainties, but it is systematic, so it is recorded here.
The helio–barycentric difference for this spectrum is only 0.007 km/s.

## Wavelength zero point of the GTC spectra (sky lines)

The arc-lamp solution of the J1745−1640 R2500I spectrum has an RMS of only 0.055 pixel, but that measures internal consistency, not the absolute zero point.
We measured the zero point independently from the OH sky emission in the `PypeIt` spec1d sky spectrum (`OPT_COUNTS_SKY`/`BOX_COUNTS_SKY`, observatory frame).
We cross-correlated it per segment with a vacuum model sky and checked it against Gaussian fits to individual lines, using vacuum wavelengths from Rousselot et al. (2000).

- Over 7700–8960 Å, where all the alkali lines lie, the observed wavelengths are too red by **+19 ± 5 km/s (+0.54 ± 0.14 Å at 8500 Å)**. The result is consistent across apertures and extraction types.
- The offset depends on wavelength: from +5 km/s near 7950 Å to +25–33 km/s near 8400–8450 Å, and back to +3 km/s at 8946 Å. This is a distortion of the wavelength solution, not a flexure shift, which is why `PypeIt`'s global flexure correlation found ≈0.
- Correcting each line by its local sky offset reduces the line-to-line scatter from 11.3 to 3.2 km/s. This test does not use the benchmark. Most of the apparent line-to-line scatter in this spectrum is therefore wavelength calibration, not template physics.
- The sky fills the slit uniformly, whereas the star may be off-centre, so the sky cannot measure the star's own constant offset. That requires the telluric absorption lines, which share the star's light path.

The same distortion probably affects other GTC/OSIRIS R2500I spectra in Cooper et al. (2024), and its size varies by line, so the paper's line-centre RVs may carry offsets of up to a few tens of km/s per line.

### The star's own zero point from the telluric lines

The audit behind this section is summarised in `docs/calibration_audit.md`.
We also measured the star's zero point from the telluric absorption in the non-telluric-corrected spec1d of J1745−1640 (the target trace, SPAT0240).
We fitted each telluric segment with a HITRAN transmission model (built locally; vacuum) multiplied by a BT-Settl 2100/5.0 template, with the stellar and telluric velocities both free.
Each node's error includes the spread over the neighbouring templates (±100 K).

- The 9000–9900 Å H2O nodes give −22 to −55 km/s, and the weak H2O 8200 band gives −6 km/s.
- We did not use the O2 A band. In L dwarfs it is filled by the pressure-broadened K I wings, and an injection test showed template-dependent biases of +5 to +31 km/s there. The catalogue therefore marks O2 A as a zero-point reference only above 2800 K.
- The sky curve itself has no segment inside the H2O bands, because the absorption there eats the OH emission, so four of the five nodes lie in the curve's unmeasured gap. We therefore measure the sky directly in each node's own window instead (`sky_zeropoint.measure_windows`; the absorption weakens the OH lines there but does not move them). These values are used only for this comparison, never in the curve.
- Four nodes have a usable sky in the same window (the 9416–9656 Å sky correlates at only 0.56). They give a star − sky offset of **−23.8 ± 3.6 km/s** (H2O 8200: −17.3 ± 10.6; H2O 9300: −24.9 ± 4.9, −23.1 ± 13.8 and −24.5 ± 7.1 km/s; χ² = 0.4 for 3 degrees of freedom, p = 0.93). The star therefore sees the sky's distortion plus a constant, as the audit found for both stars of that night (`docs/calibration_audit.md`). This is probably slit centring: the star sits bluer than the uniformly illuminated sky.
- The telluric curve alone cannot correct the alkali lines: Rb I 7802, Rb I 7949 and Cs I 8523 lie 290–430 Å from the nearest telluric node.

### What rvfitter v2 does

We fixed the following error budget before running the benchmark, and did not change it afterwards.
An offset is either applied, with its error carried as a systematic, or unknown, with a physically motivated prior width; a measured offset's value is never added to the error budget.

By default (`wavelength_correction = "sky-if-needed"`), rvfitter measures the sky zero point whenever a sky spectrum is present and applies its shape only when three tests on the calibration data pass:
(a) at least three measured sky segments, with every region more than half covered by them;
(b) a significant offset (|mean Δv| > 3σ, with σ combining the curve and arc-RMS errors) or a significant variation between segments (χ² p < 0.001);
(c) a correction larger than 0.05 resolution elements.
The stellar line scatter is never used.
When the sky is applied, the star's telluric zero point is measured too, and the star − sky constant is applied with the sky (`wavelength_correction_applied = "sky+star"`) when at least two telluric nodes have the sky measured in the same window; O2 A is not a node below 2800 K.
When the nodes disagree (χ² p ≤ 0.01), the constant's error is inflated by the Birge factor and flagged (`star_sky_inconsistent`).
The error budget then has three zero-point terms:
- `sys_wavezp`, the uncertainty of the applied curve at the measurements' wavelengths (or, when nothing is applied, only the uncertainty of the measured curve, never its value; a WARNING is logged when a significant offset is left unapplied);
- `sys_star_sky`, the error of the applied star − sky constant;
- otherwise, `sys_star_sky` is a slit-centring prior. The star's profile (a Gaussian of FWHM F) is taken to sit anywhere within ±max((W − F)/2, 0.1 arcsec) of the slit centre, clipped by the slit edges, and σ is the RMS centroid of the transmitted light. This is (W − F)/√12 for F ≪ W, and it stays finite and small when the seeing overfills the slit. It is converted to a velocity through the plate scale along the dispersion and the dispersion of the wavelength solution (`calibration.slit_prior = "auto"`; a number in km/s, or `"none"`). W comes from the header (`SLITW`, else 'N.NNarcsec' in `MASKNAME` or `DECKER`), F from `PypeIt`'s trace FWHM, and the plate scale from the header or `data/instruments.csv` (OSIRIS 0.127 arcsec per unbinned pixel; GTC OSIRIS manual and `PypeIt`). Without these inputs, a WARNING says that the absolute zero point is unconstrained, the term is NaN and `zp_flags` has `zp_unconstrained`. `calibration.zero_point_known = true` (fibre, IFU or absolute calibration) removes the term.

The sky segments now carry a synthetic bias test: noise-free model skies with the line width off by −20 or +25 per cent, or the line amplitudes perturbed by ±30 per cent, are measured like the data, and the largest |Δv| is added in quadrature to the segment's error (a segment biased by more than 0.1 resolution elements is excluded).
On J1745−1640 the short blue-edge segment next to O2 A has the largest bias (12 km/s against a statistical error of 5 km/s) and is down-weighted; none is excluded.
`io` sets non-finite sky pixels and those above 10³ times the median |sky| to NaN.

For J1745−1640, all three sky tests pass: 6 segments with every region 100 per cent measured, a mean offset of +14.9 ± 3.6 km/s (4.1σ), a segment variation with χ² = 53 for 5 degrees of freedom (p = 3 × 10⁻¹⁰), and a largest correction of 33.5 km/s at the regions against a floor of 5.8 km/s.
The sky and the star − sky constant (−23.8 ± 3.6 km/s, four nodes) are therefore both applied.
The slit prior, reported in every mode, is 18.0 km/s: W = 1.23 arcsec (`DECKER` = 'LongSlits1.23Arcsec'), F = 3.06 binned pixels × 0.254 arcsec = 0.78 arcsec (target trace, HDU 2), σ = 0.092 arcsec = 0.36 pixel at 1.39 Å per pixel. This is close to the ≈20 km/s per-object scatter of the paper's RVs about the literature (`docs/calibration_audit.md`).

Table: J1745−1640 (CCF + template fit; feature windows; R 2602; BT-Settl; `sky_source` and `telluric_source` = the Science spec1d). The benchmark is 26.2 ± 2.3 km/s, and z = |RV − 26.2|/√(2.3² + σ²). σ_total = σ_stat ⊕ sys_template ⊕ sys_wavezp ⊕ sys_star_sky (the floor is 0).

| Mode | Applied | Adopted RV [km/s] | σ_stat (z) | σ_total (z) | τ_line | sys_template | sys_wavezp | sys_star_sky |
|---|---|---|---|---|---|---|---|---|
| none | none | 37.73 | 5.03 (2.08) | 22.23 (0.52) | 8.3 | 7.0 | 9.7 | 18.0 (slit prior) |
| sky | sky | 18.95 | 3.32 (1.80) | 22.12 (0.33) | 2.0 | 7.6 | 9.7 | 18.0 (slit prior) |
| sky-if-needed (default) | sky | 18.95 | 3.32 (1.80) | 22.12 (0.33) | 2.0 | 7.6 | 9.7 | 18.0 (slit prior) |
| auto (opt-in) | sky + star (−23.8) | 42.86 | 3.25 (4.18) | 13.16 (1.25) | 1.9 | 7.5 | 9.7 | 3.6 (measured) |
| telluric | refused (coverage) | 37.73 | 5.03 (2.08) | 22.23 (0.52) | 8.3 | 7.0 | 9.7 | 18.0 (slit prior) |

Correcting the sky shape collapses the line-to-line scatter (τ_line falls from 8.3 to 1.9 km/s), a test that does not use the benchmark.
The modes without a measured constant now carry the 18.0 km/s slit prior instead of the measured offset's value, so their total errors (22.1–22.2 km/s) are honest but large.
The default applies the measured constant and lands 16.7 km/s above the benchmark, at z = 1.25 with the total error but z = 4.2 with the statistical error alone.
The two methods agree (CCF 43.2 and template fit 42.5 km/s), so the offset is common to both.
We note three possible causes, none of which we tested against the benchmark: the H2O 9300 nodes, three of the four, lie 540–1260 Å redward of Cs I 8523, the reddest alkali line, so the constant assumes that the star − sky difference is the same at 7800–8500 Å, which only the 8200 Å node (−17.3 ± 10.6 km/s) tests directly; the BT-Settl template in the telluric fit may bias the nodes (a smooth star gave a constant 3.5 km/s less negative in the audit); and the literature value itself, from a different instrument and epoch, has its own zero point.
In conclusion, on this spectrum the wavelength calibration, not the RV method, limits the absolute RV, and the budget now says so without ever counting a measured offset as an error.

### Why the star − sky constant is opt-in

On J1745−1640 the telluric nodes measure the star 23.8 ± 3.6 km/s bluer than the sky (4 nodes, χ² 0.4/3), and applying it moves the RV from 18.9 to 42.9 km/s, further from Burgasser et al. (2015).
The physics of the correction is sound, but it has been tested on one object only, and three of the four nodes lie 540–1260 Å redward of the bluest alkali line used.
We therefore keep it opt-in (`wavelength_correction = "auto"`) until it is validated on a larger sample with independent RVs, and report the measured constant (`zp_star_sky`) in every mode.
In the default, the star's slit-centring term enters only through the slit prior (18.0 km/s for J1745).
