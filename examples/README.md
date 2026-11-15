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

1. Open the spectra in that folder (Open spectra > Folder…). The Setup boxes fill themselves from the data.
2. Optional: File > Open settings… and choose `rvfitter.toml`, to use the example's own settings (line centring only, no template download).
3. Press Measure RVs. The adopted RV appears next to each object; the Results and Diagnostics tabs show the fits.

## JWST format tests: IFU cube and multi-object spectra

Two folders test the JWST readers; neither has a literature RV to compare with.

- `jwst_ifu/`: a 17 × 17 spaxel cut-out (wavelength planes 0–1846, SCI, ERR and DQ; WCS kept valid; gzip, 3.4 MB) of the NIRSpec IFU G395H/F290LP cube `jw01386013001_03104_00001_nrs1_s3d.fits` of the planetary-mass companion VHS 1256−1257 b, from the JWST Early Release Science programme 1386 (PI S. Hinkley; Miles et al. 2023, ApJL 946, L6). rvfitter extracts it in a circular aperture of radius 3 spaxels (29 spaxels) on the brightest spaxel of the median-collapsed cube, x = 8, y = 8 (x = 19, y = 33 in the full cube, the centre the JWST pipeline used), and gets 95 per cent of the pipeline's point-source flux. R ≈ 2090 is estimated from the pixel spacing (inside the documented G395H range, 1320–3600). The wavelengths are already barycentric (`SPECSYS = 'BARYCENT'`), and the cube has no slit, so no slit-centring term. VHS 1256 b has no published RV: this is a format test. The example's `ccf` and `template_fit` need a template grid covering 2.9–4.1 µm, e.g. `rvfitter templates download --grid bt-settl-cifist --teff 1100 1300 --logg 4 5`.
- `jwst_mos/`: the two NIRSpec MOS PRISM/CLEAR x1d files `jw01229004001_03101_00001_nrs2_x1d.fits` (four sources) and `jw01229004001_05101_00001_nrs1_x1d.fits` (three) of brown-dwarf candidates in IC 348, from the guaranteed-time programme 1229 (PI C. Alves de Oliveira; Luhman et al. 2024, AJ 167, 19). Each EXTRACT1D extension becomes its own object, named by its `SRCNAME`; 1229_13 has no valid pixel and is skipped with a WARNING. These test the reader only: at R ≈ 120 (PRISM, R 30–330) one resolution element is about 2500 km/s, so no RV can be constrained, and the run says so.

Both were downloaded from the Mikulski Archive for Space Telescopes (MAST). Please acknowledge them as requested by MAST: "This work is based on observations made with the NASA/ESA/CSA James Webb Space Telescope. The data were obtained from the Mikulski Archive for Space Telescopes at the Space Telescope Science Institute, which is operated by the Association of Universities for Research in Astronomy, Inc., under NASA contract NAS 5-03127 for JWST", and cite the programme papers above.

```bash
rvfitter run examples/jwst_ifu/rvfitter.toml
rvfitter run examples/jwst_mos/rvfitter.toml
```

In the GUI, open either folder (Open spectra > Folder…):

- `jwst_mos/`: each x1d file is listed with its sources (name, S/N) as tick boxes; 1229_13 is greyed and unticked, and hovering over it says why. Untick a source to leave it out, or right-click for Tick all usable / Target only; Measure RVs measures exactly the ticked sources. The Setup tab says in gold that R ≈ 120 is too low for an RV.
- `jwst_ifu/`: the Setup tab gains a row, IFU extraction: brightest spaxel (8, 8), radius 3 spaxels. Choose… shows the collapsed cube with the aperture: click to move its centre, set the radius, or go back to the brightest spaxel; the extracted spectrum is previewed, and OK applies the choice.

## Things to try
- **Template methods:** download a grid once, e.g. `rvfitter templates download --grid bt-settl-cifist --teff 1900 2300`, then add `"ccf"` and `"template_fit"` to `[methods] enabled`. The optical example has the grid's coverage; the near-infrared ones need a near-infrared grid (Sonora Bobcat or SAND, from Zenodo).
- **Wavelength zero point:** the optical example measures the sky (OH) lines in its `spec1d` file and corrects the wavelength scale (`calibration.wavelength_correction`). Set it to `"none"` to compare. J1745−1640 is a known hard case: its GTC wavelength solution is distorted, and no correction mode reproduces the literature value (see `docs/methods.md`).
- **Total error for the near-infrared spectra:** SpeX and ARCoIRIS files do not record a wavelength-calibration accuracy, so `rv_total_err` is left undefined. Add `[calibration] wave_rms_kms = 5.0` (your estimate) to get one.
