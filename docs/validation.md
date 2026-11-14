# Validation on published spectra

Here, we test rvfitter on two sets of previously published spectra. We compare its radial velocities with (a) independent literature values and (b) the values that the earlier, script-based rvfitter (v1) gave for the same spectra.
The analysis plan, and the rule for choosing the default wavelength-correction mode, were fixed before any comparison was made, and nothing was tuned to the comparison.
The data are test sets only; rvfitter does not depend on them.

## Test data
- **GTC/OSIRIS R2500I (optical):** 56 late-M and L dwarfs (Cooper et al. 2024), as telluric-corrected `PypeIt` coadds.
  - Default methods: `template_fit`, `ccf` and model-free `linecentre`, with the BT-Settl-CIFIST grid.
  - Only DENIS J1745−1640 has a sky spectrum, so the three wavelength-correction modes can differ only for it; the figures below are for mode `none`.
- **IRTF/SpeX SXD and Blanco/ARCoIRIS (near-infrared):** 44 ultracool dwarfs (Cheng et al. 2025), read from `Spextool` files.
  - Model-free `linecentre` only: the optical template grid does not cover the near-infrared.
  - Their 8 SpeX prism spectra (R ≈ 80–220) are refused, as intended (minimum R of 1000).
- **Literature:**
  - optical: SIMBAD, Gaia DR3 and a compilation of published values;
  - near-infrared: the literature values compiled by Cheng et al. (2025), from Jönsson et al. (2020) and Kiman et al. (2019).
- **v1 values:** optical from the original catalogue (`legacy/`); near-infrared from Cheng et al. (2025), Table 6.
- **Statistics:** pull = ΔRV/√(σ_v2² + σ_ref²). The robust width is 1.4826 × MAD; 0.8–1.2 counts as honest errors.

## Results

| Comparison | N | Weighted mean Δ [km/s] | Robust scatter [km/s] | Robust pull width | \|pull\| < 2 |
|---|---|---|---|---|---|
| Optical: v2 − literature | 19 | −1.7 ± 6.5 | – | 1.21 ± 0.20 | 89% |
| Optical: v1 − literature | 17 | −2.7 ± 1.8 | 15.7 | 1.33 | 76% |
| Optical: v2 − v1, adopted | 47 | +3.5 ± 3.8 | 9.6 | – | – |
| Optical: v2 `ccf` − v1 by-eye cross-correlation | 48 | +7.3 ± 1.8 | 21.8 | 1.24 | 79% |
| Optical: v2 `linecentre` − v1 line centring | 48 | −0.6 ± 1.9 | 13.5 | 0.98 | 92% |
| Near-infrared: v2 − literature (σ_stat) | 5 | −4.4 ± 2.5 | 12.1 | 0.84 | 80% |
| Near-infrared: v2 − v1, adopted (σ_stat) | 35 | −3.2 ± 2.9 | 7.2 | – | 100% |
| Near-infrared: v2 `linecentre` − v1 line centring (σ_stat) | 35 | −1.0 ± 3.9 | 9.6 | – | 100% |

- **Default mode:** one object with a sky spectrum cannot choose a wavelength-correction mode, so under the pre-registered rule `sky-if-needed` stays the default on physical grounds, and `auto` stays opt-in.
- **Errors:** v2's errors are close to honest against the literature (pull width 1.21 ± 0.20). They include a slit-centring term that v1 did not, so v1's errors are smaller than its scatter about the literature.
- **Near-infrared total errors:** `rv_total_err` is NaN there, because no wavelength-calibration accuracy could be sourced for SpeX or ARCoIRIS. Setting `calibration.wave_rms_kms` gives one. The near-infrared comparisons therefore use the statistical errors.
- **Outliers against the literature:**
  - 2MASS J1331+3407 (−97 km/s): the Rb I 7802 line falls where that spectrum's wavelength solution is distorted, below about 7950 Å;
  - 2MASS J0940+2946 (+55 km/s): v1 was also high, so the cause is probably the zero point or the low-resolution literature value.
- **Distorted wavelength solutions:** several of the optical coadds have solutions distorted blueward of about 7950 Å, outside the arc lines. A leave-one-out line-outlier test removes the affected lines.
