# Example spectra

Three ultracool dwarfs to try rvfitter on: one optical, two near-infrared. Each folder holds the spectra and a ready-made
config, `rvfitter.toml`, set to model-free line centring, so no template download is needed and each run takes a few
seconds. Results, the config actually used, a log and figures are written to `output/` in that folder.

| Folder | Object | Instrument | Literature RV [km/s] | rvfitter (line centring) [km/s] |
|---|---|---|---|---|
| `gtc_osiris/` | DENIS J174534.6−164053 (L1.5) | GTC/OSIRIS R2500I, optical (Cooper et al. 2024) | 26.2 ± 2.3 (Burgasser et al. 2015) | 13.6 ± 3.6 |
| `near_infrared/` | J0808+3157 | IRTF/SpeX SXD, near-infrared (Cheng et al. 2025) | 31.1 ± 0.1 (Jönsson et al. 2020) | 33.6 ± 6.7 |
| `near_infrared/` | J1536+0646 | Blanco/ARCoIRIS, near-infrared (Cheng et al. 2025) | 41.7 ± 13.7 (Kiman et al. 2019) | 31.8 ± 5.4 |

## Command line

```bash
rvfitter run examples/gtc_osiris/rvfitter.toml
rvfitter run examples/near_infrared/rvfitter.toml
```

## GUI (or the standalone app)

1. File > Load config… and choose `rvfitter.toml` in one of the folders.
2. Open the spectra in that folder (Open spectra > Folder…).
3. Press Measure RVs. The adopted RV appears next to each object; the Results and Diagnostics tabs show the fits.

## Things to try
- **Template methods:** download a grid once, e.g. `rvfitter templates download --grid bt-settl-cifist --teff 1900 2300`, then add `"ccf"` and `"template_fit"` to `[methods] enabled`. The optical example has the grid's coverage; the near-infrared ones need a near-infrared grid (Sonora Bobcat or SAND, from Zenodo).
- **Wavelength zero point:** the optical example measures the sky (OH) lines in its `spec1d` file and corrects the wavelength scale (`calibration.wavelength_correction`). Set it to `"none"` to compare. J1745−1640 is a known hard case: its GTC wavelength solution is distorted, and no correction mode reproduces the literature value (see `docs/calibration_audit.md`).
- **Total error for the near-infrared spectra:** SpeX and ARCoIRIS files do not record a wavelength-calibration accuracy, so `rv_total_err` is left undefined. Add `[calibration] wave_rms_kms = 5.0` (your estimate) to get one.
