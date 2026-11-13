# Migrating from the script-based rvfitter (v1) to v2

v1 (the `splot`, `xcorr` and `fullrv*` scripts) is frozen in `legacy/`.
v2 replaces the interactive scripts with a config-driven pipeline: the same measurements run unattended, and the GUI covers the interactive fitting.
Install: [install.md](install.md).

## Drivers and options

| v1 | v2 |
|---|---|
| `python fullrv.py -f spec.fits` (GTC) / `fullrv_spex.py` (SpeX) | `rvfitter measure spec.fits --out results.csv` (also a folder, `.zip` or tarball) |
| hard-coded `_spec_indices` dict (K I, Rb I, Na I, Cs I lines, air) | lines come from `data/features.ecsv` via `rvfitter.regions.select`; add, replace or remove with `[regions.overrides]` in the config, or in the GUI region editor |
| `linecentering` / `splot` (interactive Gaussian, Lorentzian, Voigt fit) | method `linecentre` (`[methods.options.linecentre]`: `profile`, `joint`, `shape_correction`); edit single fits in `rvfitter gui` |
| `crosscorrelate` / `xcorr` (by eye) | method `ccf`, plus `template_fit` and optional `forward_mcmc` |
| `-s SYSTEMATIC` | `combine.systematic_floor` [km/s], added in quadrature |
| `-r` (repeat manual measurements) | re-run `rvfitter run config.toml`; refit one object in the GUI |
| `waverms = 0.075 Å` (SpeX, fixed) | no fixed value: the zero-point terms are measured and carried in the error (`calibration.*`, below) |
| `gtc_fullinfo.csv`, `CandidateInfo.csv`, `spectral_indices.csv`, `targets_lines.csv` | `results.csv`, `results_details.csv`, `results_config.toml` |
| `splat` templates | `rvfitter templates download` (BT-Settl, Sonora, SAND, Lowz); `splat` is no longer needed |
| air wavelengths in the code | everything is vacuum internally; set `input.medium` (`auto`, `air`, `vacuum`) |
| barycentric correction done by hand | `velocity_frame.correction` (`auto`, `apply`, `skip`) and `frame` |

Starting a v2 config from a spectrum: `rvfitter init spec.fits -o rvfitter.toml`, edit, then `rvfitter run rvfitter.toml`.
Option names: `rvfitter measure --help`, or the dataclasses in `rvfitter.config`.

## What changed in the numbers, and why

The v1 benchmark value for J1745−1640 was 36.2 ± 4.4 km/s from line centring, against the literature 26.2 ± 2.3 km/s.
v2 defaults give 13.55 ± 3.62 km/s (line centring, `sky-if-needed`) and, with no zero-point correction, 34.99 ± 5.55 km/s (full table: [methods.md](methods.md#results-by-method)).
The reasons:

- **Wavelength zero point.** The GTC/OSIRIS arc solution is distorted by tens of km/s across the spectrum (RMS 0.1 Å measures only internal consistency).
  v2 measures the sky zero point from OH lines and applies it when safe (`calibration.wavelength_correction = "sky-if-needed"`); the star minus sky constant is opt-in (`"auto"`).
  Its error enters `sys_wavezp` and `sys_star_sky`, so `rv_total_err` is larger and more honest than v1's error.
  See [calibration_audit.md](calibration_audit.md).
- **Slit centring.** The star's position in the slit sets an absolute zero-point error (18 km/s for the benchmark), included in the error budget and reported in every mode.
- **Differential atmospheric refraction** across the slit is modelled when the slit position angle is known.
- **Model-free line centring** is the default; the v1 template-calibrated shape correction is opt-in (`shape_correction = true`).
- **Combination.** v1 averaged the lines of one method; v2 uses a hierarchical random-effects combination over regions, methods and epochs, so scatter between regions (`tau_line`) and methods (`tau_method`) widens the error.
- **Errors are not tuned to the benchmark**: the error budget was fixed before the benchmark ([methods.md](methods.md)).

## Converting `lines_used.json` and `xcorr_used.json`

There is no converter: none exists in the repository (no `examples/gtc_cooper2024/`), and the files are not read by v2.
Each v1 file holds, per spectrum file, the lines you chose and fitted by hand (`c1`, `c2`, `c3`, `c4` continuum windows, `mu`, `std`, `amplitude`, `line_profile`, `use`).
To reproduce those choices, express them as region overrides in the config:

```toml
[[regions.overrides.linecentre]]
name = "k1-a"          # one entry per line you used; vacuum Å
kind = "line"
wmin = 7664.0
wmax = 7666.5
centre = 7665.0        # rest wavelength
continuum = [[7650.0, 7660.0], [7670.0, 7680.0]]   # continuum windows

[[regions.overrides.linecentre]]
name = "na1-a"         # a line you set use = false for
remove = true
```

Use the `use` flags to decide which lines to `remove`, and the continuum windows `c1`..`c4` and the fit window `r1`..`r2` for `wmin`, `wmax` and `continuum`.
Convert air wavelengths to vacuum first (`rvfitter.medium`).
Applying such overrides per object needs one config per object, or the GUI region editor.
