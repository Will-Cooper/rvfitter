# Sample validation

Here, we compare rvfitter v2 with literature radial velocities and with the previous (v1) rvfitter results, for two samples.
The analysis plan and the rule for choosing the default wavelength-correction mode were fixed before any comparison was made.
Nothing was tuned to the comparison.

## Samples and set-up
- **GTC/OSIRIS R2500I:** 56 targets from Cooper et al. (2024).
  - Default methods: `template_fit`, `ccf` and model-free `linecentre`.
  - Local BT-Settl-CIFIST grid; `teff_prior` from the user's tables; R = 2602.
  - Telluric-corrected PypeIt coadds, already heliocentric.
  - Only DENIS J1745−1640 has a sky spectrum, so the three correction modes can differ only for it, and the figures below are for mode `none`.
- **NIR (GUCDS VI, Cooper et al., arXiv:2503.08514):** 31 IRTF/SpeX SXD (R ≈ 2000) and 13 Blanco/ARCoIRIS (R ≈ 3500) spectra, read natively from Spextool files.
  - Line centring only: the optical template grid does not cover the near-infrared.
  - The 8 SpeX prism spectra (R ≈ 80–220) are refused, as intended (minimum R of 1000).
- **Literature:**
  - SIMBAD, Gaia DR3 and the user's compilation for GTC;
  - for the NIR sample, Table 5 of GUCDS VI (Jönsson et al. 2020; Kiman et al. 2019);
  - the previous v1 values are from `legacy/gtc_fullinfo.csv` and GUCDS VI Table 6.
- **Statistics:** pull = ΔRV/√(σ_v2² + σ_ref²). The robust width is 1.4826 × MAD; 0.8–1.2 counts as honest errors.

## Results

| Comparison | N | Weighted mean Δ [km/s] | Robust scatter [km/s] | Robust pull width | \|pull\| < 2 |
|---|---|---|---|---|---|
| GTC v2 − literature | 19 | −1.7 ± 6.5 | – | 1.21 ± 0.20 | 89% |
| GTC v1 − literature | 17 | −2.7 ± 1.8 | 15.7 | 1.33 | 76% |
| GTC v2 − v1, adopted | 47 | +3.5 ± 3.8 | 9.6 | – | – |
| GTC v2 `ccf` − v1 by-eye cross-correlation | 48 | +7.3 ± 1.8 | 21.8 | 1.24 | 79% |
| GTC v2 `linecentre` − v1 line centring | 48 | −0.6 ± 1.9 | 13.5 | 0.98 | 92% |
| NIR v2 − literature (σ_stat) | 5 | −4.4 ± 2.5 | 12.1 | 0.84 | 80% |
| NIR v2 − v1, adopted (σ_stat) | 35 | −3.2 ± 2.9 | 7.2 | – | 100% |
| NIR v2 `linecentre` − v1 line centring (σ_stat) | 35 | −1.0 ± 3.9 | 9.6 | – | 100% |

- **Default mode:** the sample cannot choose a wavelength-correction mode, because only one object has a sky spectrum. Under the pre-registered rule, `sky-if-needed` stays the default on physical grounds, and `auto` stays opt-in.
- **Errors:** v2's errors are close to honest against the literature (pull width 1.21 ± 0.20).
  - Before the fixes, the slit-centring term was silently dropped and the width was 1.48.
  - The v1 errors did not include slit centring. The paper's RVs probably carry per-object zero-point errors of about 20 km/s.
- **NIR total errors:** `rv_total_err` is NaN for the NIR spectra, because no wavelength-calibration accuracy could be sourced for SpeX or ARCoIRIS. Set `calibration.wave_rms_kms` to obtain one. The NIR comparisons therefore use the statistical errors.
- **Outliers against the literature:**
  - 2MASS J1331+3407 (−97 km/s): an Rb I 7802 failure in a coadd whose wavelength solution is distorted below about 7950 Å;
  - 2MASS J0940+2946 (+55 km/s): v1 was also high, so the cause is probably the zero point or the SDSS literature value.
- **GTC wavelength solutions:** some GTC coadds have distorted solutions blueward of about 7950 Å, outside the arc lines. A leave-one-out line-outlier test removes the affected lines, but those spectra should be re-reduced.
