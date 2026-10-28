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

We also measured the star's zero point from the telluric absorption in the non-telluric-corrected spec1d of J1745−1640 (the target trace, SPAT0240).
We fitted each telluric segment with a HITRAN transmission model (built locally; vacuum) multiplied by a BT-Settl 2100/5.0 template, with the stellar and telluric velocities both free.
Each node's error includes the spread over the neighbouring templates (±100 K).

- The 9000–9900 Å H2O nodes give −22 to −55 km/s, and the weak H2O 8200 band gives −6 km/s.
- We did not use the O2 A band. In L dwarfs it is filled by the pressure-broadened K I wings, and an injection test showed template-dependent biases of +5 to +31 km/s there. The catalogue therefore marks O2 A as a zero-point reference only above 2800 K.
- Only the 8200 Å node lies inside the range where the sky is measured (sky segments 7451–7592 and 7702–8952 Å, within the arc-line coverage of 7437–9926 Å). The other four lie in the sky's unmeasured gap, where the sky curve is only an interpolation, so they cannot be compared with it.
- The single valid comparison gives a star − sky offset of **−19.6 ± 7.3 km/s** (one node; the error includes the 2.3 km/s spread over sky-curve models). This is probably slit centring: the star sits bluer than the uniformly illuminated sky. We note that one node cannot test whether the star sees the sky's distortion plus a constant.
- The telluric curve alone cannot correct the alkali lines: Rb I 7802, Rb I 7949 and Cs I 8523 lie 290–430 Å from the nearest telluric node.

### What rvfitter v2 does

By default (`wavelength_correction = "sky-if-needed"`), rvfitter measures the sky zero point whenever a sky spectrum is present and applies its shape only when three tests on the calibration data pass:
(a) at least three measured sky segments, with every region more than half covered by them;
(b) a significant offset (|mean Δv| > 3σ, with σ combining the curve and arc-RMS errors) or a significant variation between segments (χ² p < 0.001);
(c) a correction larger than 0.05 resolution elements.
The stellar line scatter is never used.
The star − sky offset is always reported (`zp_star_sky`, its error and `zp_star_sky_n`), and when it is not applied it is added to the total error as `sys_star_sky`.
"auto" (the sky shape plus the star − sky constant) is opt-in.
The telluric model is never shipped: users build it from HITRAN with `rvfitter templates telluric --build`.

For J1745−1640, all three tests pass: 6 segments with every region 100 per cent measured, a mean offset of +12.1 ± 3.1 km/s (3.9σ), a segment variation with χ² = 176 for 5 degrees of freedom (p = 4 × 10⁻³⁶), and a largest correction of 33.6 km/s at the regions against a floor of 5.8 km/s.
The sky correction is therefore applied.

Table: J1745−1640 (CCF + template fit; feature windows; R 2602; BT-Settl). The benchmark is 26.2 ± 2.3 km/s, and z = |RV − 26.2|/√(2.3² + σ²).

| Mode | Applied | Adopted RV [km/s] | σ_stat (z) | σ_total (z) | τ_line | sys_wavezp | sys_star_sky |
|---|---|---|---|---|---|---|---|
| none | none | 37.73 | 5.03 (2.08) | 31.78 (0.36) | 8.3 | 22.3 | 21.0 |
| sky-if-needed (default) | sky | 18.88 | 3.32 (1.81) | 24.72 (0.29) | 2.0 | 10.2 | 21.0 |
| sky | sky | 18.88 | 3.32 (1.81) | 24.72 (0.29) | 2.0 | 10.2 | 21.0 |
| auto | sky + (−19.6) | 38.86 | 3.24 (3.18) | 14.97 (0.84) | 2.0 | 12.5 | — |
| telluric | refused (coverage) | 37.73 | 5.03 (2.08) | 31.78 (0.36) | 8.3 | 22.3 | 21.0 |

Correcting the sky shape collapses the line-to-line scatter (τ_line falls from 8.3 to 2.0 km/s), a test that does not use the benchmark.
The star's absolute zero point stays uncertain at the ±20 km/s level, and the total errors include this.
"auto" moves the RV by the one-node constant, which is why it is not the default.
In conclusion, on this spectrum the wavelength calibration, not the RV method, limits the absolute RV.
