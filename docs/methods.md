# Methods and benchmark

This document records the radial-velocity (RV) techniques in rvfitter, the results on the primary benchmark, the wavelength zero-point analysis and the validation on published spectra.

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

| Method | Previous release | Previous release, bug-fixed | rvfitter | z (quadrature) |
|---|---|---|---|---|
| Line centring (model-free) | 36.2 ± 4.4 | | `sky-if-needed` (opt-in): 13.55 ± 3.62 (σ_total 19.42); `none` (default): 34.99 ± 5.55 (σ_total 22.83) | sky-if-needed 2.95 (0.65 total); none 1.46 (0.38 total) |
| Cross-correlation | 28.8 ± 4.7 (by eye) | | | |
| Template fit | — | | | |
| Forward model (MCMC) | — | | `sky-if-needed` (opt-in): 18.83 ± 2.45 (σ_total 20.62); `none` (default): 37.58 ± 4.85 (σ_total 21.04); vsini 39–88 km/s and Teff 1810–1970 K per window (not physical at R 2600) | sky-if-needed 2.19 (0.36 total); none 2.12 (0.54 total) |
| LSD | — | | `sky-if-needed` (opt-in): 10.61 ± 10.63 (σ_total 22.26); `none` (default): 33.57 ± 10.94 (σ_total 22.40); template mask (BT-Settl 2000/5.0, ~2720 lines), χ²/ν ≈ 94 inflates the error; vsini unresolved (< 126 km/s) | sky-if-needed 1.43 (0.70 total); none 0.66 (0.33 total) |
| Telluric-referenced | — | | | |
| Adopted | 32.7 ± 6.5 | | | |

The single- versus two-template binary test (`methods.binary`, template-fit windows, `none`) gives ΔBIC = 6.7 for J1745−1640, below the threshold of 10, so it is not flagged. The two-component solution (64 and −15 km/s, flux ratio 0.53) absorbs template mismatch (χ²/ν = 16 for the single fit), not a second star.

## Barycentric correction of the GTC spectra

The GTC/OSIRIS spectra used by Cooper et al. (2024) were corrected to the heliocentric frame by `PypeIt` 1.7.0 during reduction.
rvfitter recomputes that correction for J1745−1640 from the header coordinates and mid-exposure time, at the Roque de los Muchachos site.
It finds −20.851 km/s heliocentric (−20.858 km/s barycentric).
`PypeIt` applied −21.006 km/s (`VEL_CORR` = 0.9999299356).

The 0.155 km/s difference traces to the GTC headers: they record `LON-OBS` = +17.877°, but GTC lies at 17.89° **west**.
Every GTC/OSIRIS spectrum reduced by `PypeIt` with these headers therefore probably carries a sub-0.2 km/s offset from this.
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

The same distortion probably affects other GTC/OSIRIS R2500I spectra reduced this way, and its size varies by line, so uncorrected line-centre RVs from such spectra may carry offsets of up to a few tens of km/s per line.

### The star's own zero point from the telluric lines

We also measured the star's zero point from the telluric absorption in the non-telluric-corrected spec1d of J1745−1640 (the target trace, SPAT0240).
We fitted each telluric segment with a HITRAN transmission model (built locally; vacuum) multiplied by a BT-Settl 2100/5.0 template, with the stellar and telluric velocities both free.
Each node's error includes the spread over the neighbouring templates (±100 K).

- The 9000–9900 Å H2O nodes give −22 to −55 km/s, and the weak H2O 8200 band gives −6 km/s.
- We did not use the O2 A band. In L dwarfs it is filled by the pressure-broadened K I wings, and an injection test showed template-dependent biases of +5 to +31 km/s there. The catalogue therefore marks O2 A as a zero-point reference only above 2800 K.
- The sky curve itself has no segment inside the H2O bands, because the absorption there eats the OH emission, so four of the five nodes lie in the curve's unmeasured gap. We therefore measure the sky directly in each node's own window instead (`sky_zeropoint.measure_windows`; the absorption weakens the OH lines there but does not move them). These values are used only for this comparison, never in the curve.
- Four nodes have a usable sky in the same window (the 9416–9656 Å sky correlates at only 0.56). They give a star − sky offset of **−23.8 ± 3.6 km/s** (H2O 8200: −17.3 ± 10.6; H2O 9300: −24.9 ± 4.9, −23.1 ± 13.8 and −24.5 ± 7.1 km/s; χ² = 0.4 for 3 degrees of freedom, p = 0.93). The star therefore sees the sky's distortion plus a constant, as the audit found for both stars of that night. This is probably slit centring: the star sits bluer than the uniformly illuminated sky.
- The telluric curve alone cannot correct the alkali lines: Rb I 7802, Rb I 7949 and Cs I 8523 lie 290–430 Å from the nearest telluric node.

### What rvfitter does

We fixed the following error budget before running the benchmark, and did not change it afterwards.
An offset is either applied, with its error carried as a systematic, or unknown, with a physically motivated prior width; a measured offset's value is never added to the error budget.

rvfitter measures the sky zero point whenever a sky spectrum is present, in every mode. The default, `none`, only reports it, because the sky correction did not improve the agreement with the literature on the GTC sample ([Sky correction on the GTC sample](#sky-correction-on-the-gtc-sample)). With the opt-in `wavelength_correction = "sky-if-needed"`, it applies the sky shape only when three tests on the calibration data pass:
(a) at least three measured sky segments, with every region more than half covered by them;
(b) a significant offset (|mean Δv| > 3σ, with σ combining the curve and arc-RMS errors) or a significant variation between segments (χ² p < 0.001);
(c) a correction larger than 0.05 resolution elements.
The stellar line scatter is never used.
In `auto` (and whenever `calibration.telluric_source` is given), the star's telluric zero point is measured too, and in `auto` the star − sky constant is applied whenever at least two telluric nodes have the sky measured in the same window: with the sky curve when the three tests pass (`wavelength_correction_applied = "sky+star"`), and alone otherwise (`"star"`, e.g. on a spectrum already tied to the sky in the reduction, where the tests find nothing to correct).
O2 A is not a node below 2800 K (its `teff_min` in the feature catalogue), because the pressure-broadened K I 7667/7701 wings fill it, so for L dwarfs the constant rests on the H2O nodes alone.
The star model of the telluric fit is the template nearest the priors from the first configured grid; the fit is repeated with the prior template of every other configured grid (`templates.grid`) and with a smooth star, and the spread is added to the constant's error.
Until 2.2.0 every grid in the template cache joined this spread, whether configured or not, so the error of the constant depended on what happened to be downloaded (e.g. on GTC54-15A0/OB0019, 6.9, 6.3 and 4.2 km/s with BT-Settl alone against 9.2, 12.2 and 7.4 km/s with Sonora Bobcat also cached); it now uses the configured grids only.
When the nodes disagree (χ² p ≤ 0.01), the constant's error is inflated by the Birge factor and flagged (`star_sky_inconsistent`).
The error budget then has three zero-point terms:
- `sys_wavezp`, the uncertainty of the applied curve at the measurements' wavelengths (or, when nothing is applied, only the uncertainty of the measured curve, never its value; a WARNING is logged when a significant offset is left unapplied), or, for wavelengths already tied to the sky in the reduction, the error of that tie (`calibration.zero_point_tied`, a number or a header card), which replaces both the measured-curve and the arc terms because the zero point in use is the tie's; the curve is then still measured and reported as a check, and `wavezp_term` says which term was used. On 75 tied GTC/OSIRIS R2500I exposures (tie error 3.0 km/s at 8200 Å), the re-measured curve's term was 15.4 km/s (median), the largest in `rv_total_err`;
- `sys_star_sky`, the error of the applied star − sky constant;
- otherwise, `sys_star_sky` is a slit-centring prior. The star's profile (a Gaussian of FWHM F) is taken to sit anywhere within ±max((W − F)/2, 0.1 arcsec) of the slit centre, clipped by the slit edges, and σ is the RMS centroid of the transmitted light. This is (W − F)/√12 for F ≪ W, and it stays finite and small when the seeing overfills the slit. It is converted to a velocity through the plate scale along the dispersion and the dispersion of the wavelength solution (`calibration.slit_prior = "auto"`; a number in km/s, `"header:KEY"` for a per-spectrum value such as the error of an offset applied in the reduction, or `"none"`). When the reduction already applied the star − sky offset (`calibration.star_sky_applied`, true or a header card), `auto` does not apply its measured constant a second time and the term is `slit_prior` as given. W comes from the header (`SLITW`, else 'N.NNarcsec' in `MASKNAME` or `DECKER`), F from `PypeIt`'s trace FWHM, and the plate scale from the header or `data/instruments.csv`, matched on `PYP_SPEC`, then `INSTRUME` and `INSTR` (OSIRIS 0.127 arcsec per unbinned pixel, GTC OSIRIS manual and `PypeIt`; uSpeX 0.10, IRTF SpeX page; ARCoIRIS 0.417, CTIO overview; the NIR arrays are unbinned). A file with no arc RMS (every Spextool file) gets `calibration.wave_rms_kms` if set, else the instrument's typical value from the same table, with a WARNING and the `wave_rms_assumed` flag; no instrument has a documented typical value yet, so without the option that term, and `rv_total_err`, stay NaN. W, the plate scale and the binning may also come from the raw frames' headers (`input.raw_header_dir`; a coadd without its spec1d files is matched to its raw spectrum frames by `OBJECT`). Without a trace FWHM, F is the typical seeing `calibration.seeing_fwhm` (0.8 arcsec by default), with a WARNING and the `seeing_assumed` flag. Without W, a WARNING says that the absolute zero point is unconstrained, the term is NaN and `zp_flags` has `zp_unconstrained`. `calibration.zero_point_known = true` (fibre, IFU or absolute calibration) sets the term to 0. A spectrum extracted from an IFU cube has no slit: the term is 0, and the slit inputs, the DAR across the slit and the star − sky offset are not computed (both logged). An unknown (NaN) term makes `rv_total_err` NaN, with a WARNING and the `total_err_undefined` flag; it is never counted as 0.

The sky segments now carry a synthetic bias test: noise-free model skies with the line width off by −20 or +25 per cent, or the line amplitudes perturbed by ±30 per cent, are measured like the data, and the largest |Δv| is added in quadrature to the segment's error (a segment biased by more than 0.1 resolution elements is excluded).
On J1745−1640 the short blue-edge segment next to O2 A has the largest bias (12 km/s against a statistical error of 5 km/s) and is down-weighted; none is excluded.
`io` sets non-finite sky pixels and those above 10³ times the median |sky| to NaN.

For J1745−1640, all three sky tests pass: 6 segments with every region 100 per cent measured, a mean offset of +14.9 ± 3.6 km/s (4.1σ), a segment variation with χ² = 53 for 5 degrees of freedom (p = 3 × 10⁻¹⁰), and a largest correction of 33.5 km/s at the regions against a floor of 5.8 km/s.
The sky is therefore applied; the star − sky constant (−23.7 ± 4.3 km/s from four nodes, its error including the 2.3 km/s spread over the telluric-fit star models) is applied with it only in `auto`.
The slit prior, reported in every mode, is 18.0 km/s: W = 1.23 arcsec (`DECKER` = 'LongSlits1.23Arcsec'), F = 3.06 binned pixels × 0.254 arcsec = 0.78 arcsec (target trace, HDU 2), σ = 0.092 arcsec = 0.36 pixel at 1.39 Å per pixel. This is close to the ≈20 km/s per-object scatter of the previous release's RVs about the literature.

### Differential atmospheric refraction across the slit

The sky fills the slit, whilst the star's image is displaced towards the zenith by the atmospheric refraction R(λ), which is larger in the blue.
The component of that displacement across the slit moves the star's light along the dispersion, so the star − sky zero point is not a constant but a + s·D(λ).
Here, D(λ) = c g [R(λ) − R(λ_ref)] d(λ)/(s_spec λ) is the across-slit differential refraction as a velocity (d the dispersion, s_spec the plate scale along it, g the gain of the slit-clipped centroid, 0.72 for J1745−1640, and λ_ref the mean node wavelength), and s = sin(PA_slit − q) is the signed sine of the angle between the slit and the parallactic angle q (`calibration/dar.py`).
R(λ) comes from ERFA's refraction model with the pressure, temperature and humidity of the raw header (otherwise a standard atmosphere, with a WARNING), and q and the zenith distance come from astropy, sampled over the exposure.
The reduced spec1d keeps none of the slit-angle keywords, so the raw-frame headers are read through `input.raw_header` (a path or glob) or `input.raw_header_dir` (matched by the frame number in `PypeIt`'s FILENAME).

For GTC/OSIRIS, the raw header's PA is the sky position angle (east of north) of the detector +Y axis, which is the dispersion direction with the wavelength increasing along +Y, and the long slit lies along X, at PA + 90°.
We took this from the raw header's own WCS (its CD matrix puts +Y at PA, and IPA = IPA0 + PA), from the OSIRIS User Manual v3.0 (§1.1.6, §6 and §9.2) and from `PypeIt`'s `gtc_osiris` spectrograph (no spectral flip).
We note that the manual's sentence "IPA = 150.54 − P.A." has the opposite sign; we followed the header, with which the slit (PA + 90°) lies close to the parallactic angle at acquisition for the targets and standards in the test data, as observers request (within 0.3–5° for J1745−1640, 2MUCD20444, GD 153 and G 191-B2B, and up to ≈20° near the zenith, where q changes quickly).
The convention was never inferred from an RV.

When the PA is known, s is fixed at its exposure mean and its error is the drift over the exposure, and only a is fitted from the telluric nodes.
When it is unknown, s is fitted with a uniform prior in [−1, 1] (WARNING, `zp_flags` bit `dar_pa_unknown`), and its posterior width enters `sys_star_sky` through σ²(a + s·D) = σ_a² + D²σ_s² + 2D cov(a, s) at the measurements' wavelengths.
In every mode the slit prior gains a DAR term in quadrature, the RMS of s·(D − mean D) over the measurements' wavelengths: (s² + σ_s²)^½ times that RMS of D for a known PA, and 1/√3 of it otherwise.
The results report `zp_dar_s` and `zp_dar_s_err` next to `zp_star_sky` (the fitted a).

For J1745−1640 (frame 0000809229), the slit was at PA 36.1° against a parallactic angle of 37.9–40.2° during the exposure at a zenith distance of 58°, so s = −0.052 ± 0.016.
D(λ) spans +27 to −9 km/s per unit s between 7700 and 9800 Å, so the DAR moves the star by less than 1.5 km/s here, but it would reach 25 km/s for a slit perpendicular to the parallactic angle.
Without the raw header, the four nodes give s = +0.20 ± 0.44, consistent with the known value but far less precise.

### J1745−1640 after the task-15 fixes

Since the table below was first made, four changes were made, all decided before the benchmark was looked at:
the DAR term above;
the sky applied at each region is measured in a window centred on that region, grown from the region's width until the synthetic bias test passes and its error is no larger than the curve's (otherwise the curve is used, with a WARNING and the bit `sky_window_fallback`; on J1745−1640 this happens for the Na I doublet, where the packaged OH list has few lines);
the star − sky constant carries the spread over the telluric-fit star models (BT-Settl −23.7, SAND −23.0 and a smooth star −19.4 km/s; 2.3 km/s; SAND entered because it was in the cache, which no longer counts unless SAND is configured);
and the tellcorr file inherits `VEL_CORR` from its spec1d, so the curves are evaluated in the observatory frame.

Table: J1745−1640 (CCF + template fit; feature windows; R 2602; BT-Settl; `sky_source` and `telluric_source` = the Science spec1d; `input.raw_header_dir` = the raw headers). The benchmark is 26.2 ± 2.3 km/s, and z = |RV − 26.2|/√(2.3² + σ²). σ_total = σ_stat ⊕ sys_template ⊕ sys_wavezp ⊕ sys_star_sky (the floor is 0). In every mode zp_star_sky = −23.7 ± 4.3 km/s, zp_dar_s = −0.052 ± 0.016 and the slit prior is 18.0 km/s (its DAR share is below 0.5 km/s).

| Mode | Applied | Adopted RV [km/s] | σ_stat (z) | σ_total (z) | τ_line | sys_template | sys_wavezp | sys_star_sky |
|---|---|---|---|---|---|---|---|---|
| none (default) | none | 37.73 | 5.03 (2.08) | 22.21 (0.52) | 8.3 | 7.0 | 9.7 | 18.0 (slit prior ⊕ DAR) |
| sky-if-needed | sky | 16.26 | 3.50 (2.37) | 21.09 (0.47) | 2.7 | 8.7 | 5.6 | 18.0 (slit prior ⊕ DAR) |
| auto (opt-in), PA known | sky + star (a = −23.7, s = −0.052) | 41.18 | 3.34 (3.70) | 11.75 (1.25) | 2.7 | 8.8 | 5.6 | 4.3 (measured) |
| auto (opt-in), PA unknown | sky + star (a = −23.9, s = +0.20 ± 0.44) | 36.62 | 3.38 (2.55) | 14.05 (0.73) | 2.4 | 8.0 | 5.6 | 9.5 (measured) |

The per-region sky windows lower `sys_wavezp` from 9.7 to 5.6 km/s and τ_line stays small (2.4–2.7 km/s), whilst `sky-if-needed` moves from 18.95 to 16.26 km/s.
With the PA known, the DAR barely changes the constant, and `auto` lands 15.0 km/s above the benchmark (z = 1.25 with the total error, 3.7 with the statistical error alone), much as before (42.86 km/s).
With the PA unknown, the fitted s takes up some of the slope between the 8200 Å node and the 9300 Å nodes and the RV falls to 36.6 km/s; this is not evidence for either value of s, since the known s is well inside the fitted posterior.
We therefore do not regard the DAR as the cause of the `auto` offset on this spectrum: the slit was close to the parallactic angle.

Line centring is model-free by default: its RVs come from the NIST rest wavelengths alone, with no template, so it is the template-independent cross-check of CCF and the template fit, whose agreement would otherwise rest on the same models.
The opt-in `shape_correction` subtracts, line by line, the RV that the same profile fit measures on the noise-free template nearest the priors (same R, pixels and windows), and adds the spread over the ±1 Teff-step neighbours to the error; the method is then template-dependent (INFO, and `linecentre_shape_correction` in the results).
Cs I 8946 is no longer used for line centring: its catalogue caveat now says it is blended at R ≈ 2600 (BT-Settl fits of it range from −54 to +3 km/s).

Table: J1745−1640 line centring (Voigt; R 2602; same inputs as above), with the default (model-free) and with `shape_correction`.

| Mode | shape_correction | RV [km/s] | σ_stat (z) | σ_total (z) | τ_line | sys_wavezp | sys_star_sky |
|---|---|---|---|---|---|---|---|
| none | off (default) | 34.99 | 5.55 (1.46) | 22.83 (0.38) | 11.0 | 12.7 | 18.1 |
| none | on | 36.60 | 6.26 (1.56) | 23.01 (0.45) | 12.8 | 12.7 | 18.1 |
| sky-if-needed | off (default) | 13.55 | 3.62 (2.95) | 19.42 (0.65) | 6.4 | 6.2 | 18.0 |
| sky-if-needed | on | 15.54 | 4.45 (2.13) | 19.59 (0.54) | 8.4 | 6.2 | 18.0 |

The shape correction moves the method by +1.6 to +2.0 km/s and increases τ_line, so on this spectrum the profile bias is not what separates line centring from the template methods.

Earlier table (before the task-15 fixes, kept for comparison):

Table: J1745−1640 (CCF + template fit; feature windows; R 2602; BT-Settl; `sky_source` and `telluric_source` = the Science spec1d). The benchmark is 26.2 ± 2.3 km/s, and z = |RV − 26.2|/√(2.3² + σ²). σ_total = σ_stat ⊕ sys_template ⊕ sys_wavezp ⊕ sys_star_sky (the floor is 0).

| Mode | Applied | Adopted RV [km/s] | σ_stat (z) | σ_total (z) | τ_line | sys_template | sys_wavezp | sys_star_sky |
|---|---|---|---|---|---|---|---|---|
| none (default) | none | 37.73 | 5.03 (2.08) | 22.23 (0.52) | 8.3 | 7.0 | 9.7 | 18.0 (slit prior) |
| sky | sky | 18.95 | 3.32 (1.80) | 22.12 (0.33) | 2.0 | 7.6 | 9.7 | 18.0 (slit prior) |
| sky-if-needed | sky | 18.95 | 3.32 (1.80) | 22.12 (0.33) | 2.0 | 7.6 | 9.7 | 18.0 (slit prior) |
| auto (opt-in) | sky + star (−23.8) | 42.86 | 3.25 (4.18) | 13.16 (1.25) | 1.9 | 7.5 | 9.7 | 3.6 (measured) |
| telluric | refused (coverage) | 37.73 | 5.03 (2.08) | 22.23 (0.52) | 8.3 | 7.0 | 9.7 | 18.0 (slit prior) |

Correcting the sky shape collapses the line-to-line scatter (τ_line falls from 8.3 to 1.9 km/s), a test that does not use the benchmark.
The modes without a measured constant now carry the 18.0 km/s slit prior instead of the measured offset's value, so their total errors (22.1–22.2 km/s) are honest but large.
`auto` applies the measured constant and lands 16.7 km/s above the benchmark, at z = 1.25 with the total error but z = 4.2 with the statistical error alone.
The two methods agree (CCF 43.2 and template fit 42.5 km/s), so the offset is common to both.
We note three possible causes, none of which we tested against the benchmark: the H2O 9300 nodes, three of the four, lie 540–1260 Å redward of Cs I 8523, the reddest alkali line, so the constant assumes that the star − sky difference is the same at 7800–8500 Å, which only the 8200 Å node (−17.3 ± 10.6 km/s) tests directly; the BT-Settl template in the telluric fit may bias the nodes (a smooth star gave a constant 3.5 km/s less negative in the audit); and the literature value itself, from a different instrument and epoch, has its own zero point.
In conclusion, on this spectrum the wavelength calibration, not the RV method, limits the absolute RV, and the budget now says so without ever counting a measured offset as an error.

### Why the star − sky constant is opt-in

On J1745−1640 the telluric nodes measure the star 23.8 ± 3.6 km/s bluer than the sky (4 nodes, χ² 0.4/3), and applying it moves the RV from 18.9 to 42.9 km/s, further from Burgasser et al. (2015).
The physics of the correction is sound, but it has been tested on one object only, and three of the four nodes lie 540–1260 Å redward of the bluest alkali line used.
We therefore kept it opt-in (`wavelength_correction = "auto"`) until it was validated on a larger sample with independent RVs ([below](#the-star--sky-constant-on-sky-tied-spectra)), and report the measured constant (`zp_star_sky`) in every mode.
In the other modes, the star's slit-centring term enters only through the slit prior (18.0 km/s for J1745).

### The star − sky constant on sky-tied spectra

Here, we discuss the validation of the constant on a larger sample, from an independent re-reduction of the GTC/OSIRIS R2500I sample of Cooper et al. (2024) with `PypeIt` 2.0.1, in which each exposure's wavelength scale was tied to its own OH sky lines before rvfitter was run.
On these spectra the sky tests find nothing to correct, so before this version `auto` fell back to the slit-centring prior and the measured constant was never applied.
The constant was measured for 73 of the 75 exposures: median −8.5 km/s, 16th–84th percentiles −24.0 to +9.3 km/s, and a median error of 7.4 km/s.
It is consistent between exposures of the same object (e.g. J1745−1640: −15.4 ± 4.3 and −11.1 ± 4.4 km/s).
Against literature RVs (N = 18), not applying it gave a median offset of −15.9 km/s and a robust scatter of 31.7 km/s; applying it gave about −2.5 and 14.3 km/s.
As a result, `auto` now applies a measured constant (with its error, `sys_star_sky`) whenever it is measured, with or without the sky curve.
We note that on the original reductions, which were not tied to the sky, `auto` was not better than `none` ([Sky correction on the GTC sample](#sky-correction-on-the-gtc-sample): median +1.7 km/s and robust scatter 32.7 km/s, against −5.4 and 29.8 km/s), probably because there the arc distortion, not the slit centring, dominates.
We therefore keep `none` as the default, and recommend `auto` for spectra whose wavelength scale is tied to the sky.

### Regions on unmeasured parts of the sky curve

A region that lies on an unmeasured part of the sky curve (held flat, ramped or extrapolated over more than half its width) is dropped from every method when the sky correction is applied, with a WARNING naming it. The correction is refused (`sky_unsafe`) only when such regions are the majority.
This was decided after line centring added K I 7667, which lies in the O2 A gap of J1745's sky: under the earlier rule that single line refused the correction for every method, silently returning J1745's sky-if-needed result to the uncorrected values. With the current rule, J1745's line centring moves from 34.0 (uncorrected) to 15.5 km/s, consistent with the sky-corrected CCF and template fit (≈19 km/s); whether the sky correction itself brings the star closer to its true RV is tested on the GTC sample ([Sky correction on the GTC sample](#sky-correction-on-the-gtc-sample)).

## Sky and star zero points across the slit

We audited the GTC/OSIRIS R2500I spectra of J1745−1640 and Ross 640 (night of 2015 August 3) to set the zero-point error budget.
Four independent measurements of the J1745−1640 sky agree: cross-correlation with the `PypeIt` OH model, cross-correlation with its Paranal sky, Gaussian centroids of 23 vetted lines, and `PypeIt`'s `spec_flex_shift` run in 250 Å windows.
The sky is +5 to +38 km/s too red over 7700–8950 Å and −17 to −46 km/s too blue over 9000–9450 Å.
`PypeIt`'s single global flexure shift (≈0 km/s) is the average of this two-signed distortion, so it cannot reveal it.
Both stars follow the sky's shape plus a constant that depends on where the star sits in the slit, and the constant varies by over 100 km/s between traces in the same slit (the other J1745−1640 trace gives +59 km/s and Ross 640 +140 km/s).
Consequently, an offset is either applied, with its error carried as a systematic, or unknown, with a prior width; a measured offset's value is never added to the error budget.

## Validation on published spectra

We compared rvfitter with independent literature values and with the previous, script-based release (kept in the git history and the Zenodo archive of earlier versions) for two sets of published spectra.
The analysis plan and the rule for the default wavelength-correction mode were fixed before any comparison, and nothing was tuned to it.
The optical set is 56 late-M and L dwarfs observed with GTC/OSIRIS R2500I (Cooper et al. 2024), as telluric-corrected `PypeIt` coadds, run with `template_fit`, `ccf` and `linecentre` and the BT-Settl-CIFIST grid (wavelength correction `none`; only J1745−1640 among them has a sky spectrum).
The near-infrared set is 44 ultracool dwarfs from IRTF/SpeX SXD and Blanco/ARCoIRIS (Cheng et al. 2025), run with `linecentre` only, because our local BT-Settl-CIFIST grid had been cut to 4000–11 000 Å (a downloaded grid covers the near infrared; [settings.md](settings.md#choosing-a-model-grid)); Spextool files have no sky, so no wavelength correction was applied in any mode; their 8 SpeX prism spectra (R ≈ 80–220) are refused, as intended.
The pull is ΔRV/√(σ₁² + σ₂²), and a robust pull width of 0.8–1.2 (1.4826 × MAD) counts as honest errors.

| Comparison | N | Weighted mean Δ [km/s] | Robust scatter [km/s] | Robust pull width | \|pull\| < 2 |
|---|---|---|---|---|---|
| Optical: rvfitter − literature | 19 | −1.7 ± 6.5 | – | 1.21 ± 0.20 | 89% |
| Optical: previous release − literature | 17 | −2.7 ± 1.8 | 15.7 | 1.33 | 76% |
| Optical: rvfitter − previous release, adopted | 47 | +3.5 ± 3.8 | 9.6 | – | – |
| Optical: rvfitter `ccf` − previous by-eye cross-correlation | 48 | +7.3 ± 1.8 | 21.8 | 1.24 | 79% |
| Optical: rvfitter `linecentre` − previous line centring | 48 | −0.6 ± 1.9 | 13.5 | 0.98 | 92% |
| Near-infrared: rvfitter − literature (σ_stat) | 5 | −4.4 ± 2.5 | 12.1 | 0.84 | 80% |
| Near-infrared: rvfitter − previous release, adopted (σ_stat) | 35 | −3.2 ± 2.9 | 7.2 | – | 100% |
| Near-infrared: rvfitter `linecentre` − previous line centring (σ_stat) | 35 | −1.0 ± 3.9 | 9.6 | – | 100% |

- The errors are close to honest against the literature (pull width 1.21 ± 0.20); they include a slit-centring term that the previous release lacked.
- `rv_total_err` is undefined for the near-infrared spectra, because no wavelength-calibration accuracy could be sourced for SpeX or ARCoIRIS; setting `calibration.wave_rms_kms` gives one.
- The largest outliers are 2MASS J1331+3407 (−97 km/s; the Rb I 7802 line lies where that wavelength solution is distorted) and 2MASS J0940+2946 (+55 km/s; the previous release was also high, so the cause is probably the zero point or the low-resolution literature value).
- Several optical coadds have solutions distorted blueward of about 7950 Å; a leave-one-out line-outlier test removes the affected lines.
- Only J1745−1640 has a sky spectrum among these coadds, so this set alone cannot choose a wavelength-correction mode; the GTC sample below does.

### Sky correction on the GTC sample

Here, we present the test that set the default wavelength-correction mode.
We reran the GTC/OSIRIS sample with each spectrum's sky (its `PypeIt` spec1d) in the three modes `none`, `sky-if-needed` and `auto`, and compared each with the literature RVs, under a decision rule that we fixed before looking at the results.
The table gives the comparison with the literature for the 20 objects that have a literature RV.

| Mode | N | Median Δ [km/s] | Robust σ [km/s] | Pull width (total error) |
|---|---|---|---|---|
| `none` | 20 | −5.4 | 29.8 | 1.20 |
| `sky-if-needed` | 20 | −8.7 | 22.5 | 1.19 |
| `auto` | 20 | +1.7 | 32.7 | 1.31 |

The sky correction lowered the robust scatter from 29.8 to 22.5 km/s, but it did not change the pull width, and the physical test did not support it.
If the sky distortion were what moves each RV away from the literature, the uncorrected offset (`none` − literature) would follow each object's sky Δv at its RV lines.
For 19 objects, the weighted slope is 1.52 ± 0.34, but with χ²/dof = 5.3 and no significant rank correlation (Spearman p = 0.17, Pearson p = 0.27), so the apparent slope rests on a few objects.
On the nine literature objects where `sky-if-needed` applied the sky, the robust scatter fell from 19.4 to 16.6 km/s, whilst the rms rose from 29.3 to 29.7 km/s.
The sky distortion shapes agree within a night (correlation r ≥ 0.9 on 10 of 14 nights) but vary between nights, so a correction from one night's sky cannot be carried to another.
We could not test `auto`, because the sample has only H₂O telluric nodes and no O₂ A band.
On J1745−1640 the adopted RV (template fit, CCF and line centring) was 36.6 km/s with `none`, 15.2 km/s with `sky-if-needed` and 40.5 km/s with `auto`, against the sample's literature value of 26.0 ± 2.0 km/s, so even the benchmark does not favour one mode.
As a result, `none` is the default, and `sky-if-needed` and `auto` are opt-in; we recommend `sky-if-needed` only as a check when a sky spectrum is present, since it moves RVs by up to tens of km/s, and `auto` for spectra already tied to the sky ([The star − sky constant on sky-tied spectra](#the-star--sky-constant-on-sky-tied-spectra)).
The sky zero point is still measured and reported in every mode (`zp_sky_kms`), and the Diagnostics tab and the `wavelength_shift_kms` column give how much an applied correction moved each RV.

## References

- Blake, Charbonneau & White 2010, ApJ 723, 684 (arXiv:1008.3874)
- Burgasser et al. 2015, ApJS 220, 18
- Cheng et al. 2025 (near-infrared test spectra)
- Cooper et al. 2024, MNRAS 534, 695 (arXiv:2409.03706)
- Hsu et al. 2021, ApJS 257, 45 (arXiv:2107.01222)
- Phan-Bao et al. 2008, MNRAS 383, 831 (discovery of J1745−1640)
- Rousselot et al. 2000, A&A 354, 1134 (OH line list)
