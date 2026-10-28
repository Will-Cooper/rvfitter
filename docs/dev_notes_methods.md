# Developer notes: methods package (M)

## Handoff (state at unit5-fix)

**Modules (public API):**
- `broadening`: `doppler_factor`, `shift_wave`, `doppler_shift`, `log_wave_grid`, `rebin` (flux-conserving), `resample`, `gaussian_kernel`, `rotation_kernel` (Gray), `convolve`, `broaden_loglam`, `instrumental(spec, R | R(λ) | table)`, `rotational(spec, vsini, ε)`.
- `combine`: `weighted_mean` (Birge), `combine_regions(meas, details=)` (random effects), `hierarchical(...)` (Woodbury, half-Cauchy, grid), `hierarchical_from_methods(methods, exclude_flags=)`, `mixture_pdf`.
- `templates`: `GridMeta` (axes teff/logg/met/alpha + extra; coverage, upstream, medium air/vacuum/auto), `TemplateGrid` (`get`, `broadened`, `select`, `nearest`, `bounds`; multilinear N-D), `KNOWN_GRIDS`, `verify_medium`, `measure_medium`, `medium_check_lines` (NIST), `download(source=auto|svo|zenodo)`, `available`, `cli_main` (`download | list | verify | telluric [--build]`).
- `grid_sources`: Zenodo records for Bobcat, Diamondback, Elf Owl and SAND; zip members fetched by HTTP range, tar.gz streamed with MD5.
- `methods/_common`: options, regions, `Prepared`/`prepare`, `Template.at` (pixel-integrated), `candidates`/`open_grid`/`build_template`, `template_summary` (best params, edges, guidance), `segment`, `finish` (random effects + outlier rejection), flags.
- `methods/ccf`, `methods/template_fit`: `measure(spectrum, regions, cfg, *, executor, rng)`. `template_fit` also has `measure_resolution` and `estimate_resolution`. `methods.REGISTRY[name]` is callable and has `.measure`.
- `calibration/sky_zeropoint`: `measure(..., valid_range=)` (OH sky, local per-span Δv(λ)), `SkyZeroPoint.measured_fraction`/`measured_ranges`, `apply` (any curve: sky or telluric, plus a constant), `applied_dv`, `curve_at(curve_dict, λ)`, `oh_lines`, `model_sky`.
- `calibration/telluric`: `model()` (HITRAN, vacuum, 6000–25 000 Å, τ_dry and τ_H2O; read from `cache_path()`, never shipped; raises `TelluricModelUnavailable`), `available`, `cache_path` (`$RVFITTER_TELLURIC_MODEL` overrides), `reload`, `transmission(wave, airmass, pwv, resolution=, dv=)`, `mask(wave, threshold)`.
- `calibration/telluric_build`: `build(out=None, *, line_cache=, force=, get=)` (HITRANonline → cache; provenance, citation and terms in the metadata and a JSON sidecar). `tools/build_telluric_model.py` is a thin wrapper.
- `calibration/telluric_zeropoint`: `measure(spectrum, regions, resolution=, template=, teff=, spread_templates=)` → `TelluricZeroPoint` (unpacks as (zp, err); `dv(λ)`), `star_minus_sky(tell, sky)` (measured sky only; `n_nodes`), `frame_factor`.
- `methods/telluric_ref`: v_star − v_tell per telluric segment (`prior_template`, `neighbour_templates`).
- `io.read_spectrum`: multi-object PypeIt spec1d → the name-matched table, else the highest-S/N science object (logged; `input.hdu` overrides; `meta['hdu'|'extname'|'n_objects']`).
- `systematics`: `template_spread(_details)` (templates, grids, window modes), `wavelength_zero_point(spectrum, cfg, *, wave=)` (max of the arc RMS and the zero-point term at the measurements' wavelengths).
- `pipeline` (shared): `wavelength_correction(raw, spec, cfg, name, regions=)` (sky-if-needed, sky, auto, telluric, none), `ZP_FLAGS`, `zp_flag_names`, `epoch_labels`, `refit_object`, `suggested_refits`, `apply_patch`, `config_for_object`, `object_seed`.
- `plotting`, `methods/{linecentre,forward,lsd,binary}`: not started.

**Decisions in force:**
- Per method, regions are combined by random effects (`hierarchical` with one method; Birge is kept in the diagnostics). Across methods, the hierarchical line × method model gives the adopted RV.
- Several epochs: regions → epoch (each epoch measured and combined on its own), then random effects over the epochs, for each method and for the adopted RV (`tau_epoch`). Regions are still labelled `<epoch>:<region>`. The χ² variability test uses each epoch's statistical error ⊕ its sys_wavezp (`rv_epoch_pvalue`; variable when p < 0.001; the statistical-only p is `adopted['epochs']['pvalue_stat']`). Zero-point terms over epochs: offsets averaged, errors and systematics as RMS, flags OR-ed, per-epoch values in `barycentric['wavelength_correction_epochs']`.
- `results.FLAGS`: fit_failed 1, few_regions 2, error_floor 4, template_grid_edge 8, telluric_contaminated 16, not_converged 32, poor_fit 64, outlier 128, template_param_edge 256, template_grid_hole 512. `EXCLUDE_FLAGS` = 1 | 2 | 32 | 128. Object-level `zp_flags` (`pipeline.ZP_FLAGS`): sky_applied 1, telluric_applied 2, star_sky_applied 4, sky_not_needed 8, sky_unsafe 16, telluric_unavailable 32, telluric_refused 64, star_sky_unmeasured 128, no_sky 256.
- Template-method windows come from the catalogue's `model_fidelity` (`regions.template_windows = "features"` by default at R ≤ 5000; "full" for every span). H2O stays masked for spans even when telluric-corrected.
- `calibration.wavelength_correction` defaults to **"sky-if-needed"** (user/orchestrator decision): the sky curve is applied only when (a) ≥ 3 sky segments are measured and every region (before correction) is > 50 per cent inside measured segments, (b) |weighted mean Δv| > 3σ (curve ⊕ model spread ⊕ arc RMS) or the segment χ² gives p < 0.001, and (c) the largest correction at the regions exceeds 0.05 resolution elements. Calibration data only. Otherwise the zero point is reported and counted in sys_wavezp. "sky" always applies; "auto" (sky shape + star − sky constant) is opt-in; "telluric" is refused when a region is > 250 Å from every telluric node or only O2 nodes exist; "none" measures only. Columns: `wavelength_correction` (mode), `wavelength_correction_applied`, `zp_flags`.
- `sys_wavezp` at the measurements' wavelengths (RMS over them): an applied curve's own error σ(λ) ⊕ `zp_sky_model_spread` (⊕ the constant's error in "auto"); an unapplied curve's √(Δv² + σ²) (the sky's if measured, else the telluric curve's). `sys_star_sky` = |zp_star_sky| ⊕ its error whenever the constant is measured but not applied (all modes except auto/telluric) and enters `rv_total_err`.
- `zp_star_sky` uses only telluric nodes inside used sky segments; NaN (flag star_sky_unmeasured) when none. O2 A is not a zero-point node below 2800 K (catalogue `teff_min`). Node errors include the ±1 Teff-step template spread.
- `calibration.wave_valid_range` (e.g. the arc-line range): sky segments centred outside it are not used.
- The HITRAN-derived model is never shipped or committed (`.gitignore`); it is built into the user cache. Tests use `tests/telluric_synth.py` via `tests/conftest.py`; `real_telluric_model` skips without the cache.
- Seeds are `SeedSequence([seed, blake2b(object name)])`. `[[refits]]` patches apply only to their object.
- Nothing is ever tuned to the J1745 benchmark; choices are fixed before looking.

**J1745 (feature windows, R 2602, BT-Settl, sky from the Science spec1d, tellurics from the nopipecorr spec1d (target trace), wave_valid_range = arc lines 7437–9926 Å), unit5-fix.** z = |RV − 26.2|/√(2.3² + σ²):

| Mode | Applied | Adopted [km/s] | σ_stat (z) | σ_total (z) | τ_line | sys_wavezp | sys_star_sky |
|---|---|---|---|---|---|---|---|
| none | none | 37.73 | 5.03 (2.08) | 31.78 (0.36) | 8.3 | 22.3 | 21.0 |
| sky-if-needed | sky | 18.88 | 3.32 (1.81) | 24.72 (0.29) | 2.0 | 10.2 | 21.0 |
| sky | sky | 18.88 | 3.32 (1.81) | 24.72 (0.29) | 2.0 | 10.2 | 21.0 |
| auto | sky + (−19.6) | 38.86 | 3.24 (3.18) | 14.97 (0.84) | 2.0 | 12.5 | — |
| telluric | refused | 37.73 | 5.03 (2.08) | 31.78 (0.36) | 8.3 | 22.3 | 21.0 |

zp_star_sky = −19.6 ± 7.3 km/s (1 node: H2O 8200). sky-if-needed tests: 6 segments, regions 100 per cent measured, +12.1 ± 3.1 km/s (3.9σ; no arc RMS in the tellcorr header), segment χ² 176/5 (p = 4 × 10⁻³⁶), max correction 33.6 km/s against a 5.8 km/s floor → applied.

**Open items, in order:**
1. Zenodo fixes (reviewer-p, `review_unit3b_fix.md`):
   - write to a temporary file and rename only after its checksum passes; remove partial downloads on error;
   - three retries with backoff; `_RangeFile` must require a 206 response and check Content-Range;
   - record the licence, record/DOI and citation in the metadata and manifest, and print them;
   - a `refitted` results column;
   - mocked tests.
2. Unit 5 left-overs:
   - `io` does not apply PypeIt's `OPT_MASK` (J1745 HDU 2 has a pixel at flux/σ = −24 745; the telluric fit clips it);
   - `docs/architecture.md` (shared) should list `wavelength_zero_point(spectrum, cfg, *, wave=None)` and `telluric_zeropoint.measure(..., teff=, spread_templates=)`;
   - the arc-line range is not read automatically from `wvcalib.fits` (set `calibration.wave_valid_range`);
   - `template_fit`'s `telluric_model` option could use `calibration.telluric`.
3. Unit 4: `methods/linecentre.py`.
4. Unit 6: `methods/forward.py` (emcee, `fit_epochs`).
5. Unit 7: LSD, binary, systematics, `inject`, `forecast`, `plotting`.

**Gotchas:**
- Run with `PYTHONPATH=<wt>/src`. Test-data tests need `RVFITTER_TESTDATA=/home/wcooper/Documents/rvfitter-testdata` and are marked `data`; the J1745 files are under `pypeit/R2500I/GTC54-15A0/OB0055/`.
- The real telluric model must be built once (`rvfitter templates telluric --build`; with `--line-cache <dir>` an existing HITRAN line cache is reused offline, ~20 s). The build from the unit-5 line files reproduces the former packaged npz bit for bit.
- Test markers:
  - `slow` covers the injection tests and J1745-like (about 6 minutes);
  - `network` (always also `slow`) is skipped offline, and on HTTP 403/timeout from SVO, which currently refuses this host; it now includes the real HITRAN build;
  - CI runs `not data and not gui and not slow and not network`.
- Injection tests calibrate per window (`segment_res=0`). The method-level random effects are conservative (pull ≈ 0.75–0.9).
- The local BT-Settl grid has no metadata file: open it with `grid='bt-settl-cifist'` (medium air, verified).
- `Spectrum.sky` is optional. The tellcorr coadd has no sky: set `calibration.sky_source` to its spec1d. For a multi-object spec1d, every object's sky is measured and the best (most segments) is used and logged; on J1745 only HDU 1's sky is measurable (HDU 2's fails), so HDU 1's is used.
- Config `methods.options` keys are validated against each method's `DEFAULTS`; unknown keys raise.
- In this worktree, chained shell commands with `cd` and heredocs are sometimes refused: write scripts to `tmp/<name>/` and run them.
- The fitted H2O scaling on J1745 (a_H2O ≈ 21 at airmass 1.88) means PWV ≈ 11 mm, high for La Palma. It is probably real absorption, but the H2O τ profile (one 1.5 km scale height) is then extrapolated.


Here, we record the design decisions, validation and J1745−1640 results of the methods package (`broadening`, `combine`, `templates`, `calibration/*`, `methods/*`, `systematics`, `plotting`).

## Unit 1: broadening and combination

### `broadening.py`
- `doppler_shift` uses the relativistic factor √((1 + β)/(1 − β)) with the vacuum c; by default it only scales the wavelength axis, and `onto_original_grid=True` resamples back onto the input grid.
- `rebin` is the flux-conserving resampler used everywhere: the mean of the piecewise-linear interpolant over each output pixel, computed from its exact cumulative integral. The integral is conserved to machine precision (tested), upsampling reduces to linear interpolation, downsampling to pixel integration, and any output pixel touching a non-finite input is NaN.
- `instrumental(spec, R)` accepts a constant R, a callable R(λ) or a (λ, R) table. In all cases the spectrum is resampled onto a grid uniform in u(λ) = ∫R/λ dλ, on which the Gaussian line-spread function has FWHM 1, so one fixed kernel handles a variable LSF; for a constant R this is the usual ln λ grid. `resolution_native` removes an existing resolution in quadrature.
- `rotational(spec, vsini, ε)` uses the Gray (2005) kernel, averaged over pixels, on a ln λ grid with a step of at most vsini/10.
- `broaden_loglam` is the array-level fast path for the fitting loops (flux already on a uniform ln λ grid).
- Convolutions pad with the edge values, so the ends do not droop.

### `combine.py`
- `weighted_mean` and `combine_regions` (Birge inflation, never deflation) port `combine_rvs`.
- `hierarchical` ports `hierarchical_combine` with the Woodbury identity, single-level τ fixed at 0 (now with a warning), half-Cauchy priors and grid quadrature. Keys follow the `ObjectResult.adopted` contract: `median`, `err` (half of q84 − q16), `q16`, `q84`, `mean`, `sd`, `tau_line`, `tau_method`, `n_lines`, `n_methods`, plus `n`, `tau_scale` and the mixture components.
- `hierarchical_from_methods` builds the line × method problem from `MethodResult`s, with the region name as the line label; flagged measurements are excluded by default.
- `tests/test_combine.py` extracts the legacy functions from `legacy/utils.py` with `ast` (so its heavy imports are not needed) and checks seven cases, including fixed τ, a single method and a 3 × 8 problem: μ, σ, quantiles and τ medians agree to ≤1e-8 km/s.

### Review fixes (4cb965b)
- `instrumental` defaults `resolution_native` to `spec.resolution`, broadens by the quadrature difference and raises if the target is sharper; `np.inf` treats the input as unresolved.
- `convolve` dilates the NaN mask by the kernel half-width, so interpolated fill never reaches a returned pixel.
- Internal grids sample ≥10 pixels per kernel FWHM. Coarse inputs are upsampled with a cubic spline and point-sampled back (the earlier rebin round trip broadened the LSF by 1–11 per cent at 3–6 pix/FWHM); the output FWHM is now within 1 per cent of the expectation at 3, 5 and 10 pix/FWHM (tested).
- The Gaussian kernel is pixel-integrated below σ = 1 pixel; a dense-covariance brute-force test checks `hierarchical` against the mathematics rather than the legacy copy.

## Unit 2: templates (030e694)
- `TemplateGrid(directory, grid=None, meta=None, verify=True)` reads `rvfitter_grid.toml` (name, medium, flux units, filename regex with `teff`/`logg`/`met` groups, `teff_scale`, `met_default`, `resolution_native`); without it, a known grid name supplies the defaults with a warning (the read-only test grid is opened this way).
- Wavelengths are converted to vacuum on reading (`rvfitter.medium`); files are parsed with the pandas C reader (≈0.05 s per 280 000-line file) and kept in an `lru_cache` keyed on path, mtime and size.
- `get()` interpolates trilinearly in flux; outside the grid, or where a corner is missing, it uses the nearest point and sets `meta['outside_grid']`. `broadened()` caches the LSF-broadened grid points (keyed on file, R, grid and output grid) and interpolates after broadening, which is equivalent because both are linear.
- `verify_medium()` locates the minima of the strongest Na I, K I, Rb I and Cs I lines (A_ki ≥ 2.5e7 s⁻¹, from `data/nist_lines.csv`; no typed wavelengths) and assigns each to the nearer of the NIST air and vacuum values. On the local BT-Settl-CIFIST grid all eight lines agree with air (median offset −0.009 Å, maximum 0.03 Å; vacuum offsets −2.1 to −2.4 Å), confirming that the Lyon files are air. A grid whose declared medium disagrees raises `MediumMismatchError` on first use.
- `download()` queries the SVO SSAP service, filters the rows to the requested ranges, downloads into the platformdirs cache (`$RVFITTER_CACHE` overrides), and keeps a `manifest.json` of SHA-256 checksums; files whose checksum matches are not fetched again. SVO was unreachable from the development machine (timeout), so the VOTable parsing is generic (access-reference utype, or a `Spectrum`/`access_url` column) and the tests mock the network; the SVO model names for Sonora (`sonora_bobcat`, `sonora_elf_owl`) still need checking online.

## Unit 3: CCF and template fit

### Shared (`methods/_common.py`)
- Options come from `cfg.methods.options[method]` merged over each method's `DEFAULTS`; unknown keys raise. R comes from the option, then `spectrum.resolution`, then `cfg.input.resolution`, and is otherwise an error.
- Windows are all regions except `telluric`/`continuum` kinds (configurable); `telluric` regions are masked.
- Template candidates: the option `template`/`templates` (rest-frame Spectrum or (wave, flux)), else `cfg.templates` (local_dir, grid, teff_prior ± 300 K, ranges). Selection is by the lowest total χ² over the windows, evaluated per candidate with `executor.map`.
- Seeds: `child_seeds(rng, n)` spawns a fixed number of task seeds, so Monte Carlo and bootstrap results do not depend on the executor (tested with a thread pool).
- Flag bits come from `results.FLAGS`: fit_failed 1, few_regions 2 (too few pixels), template_grid_edge 8 (outside grid), not_converged 32 (optimum at the edge of the RV search). Two bits are not yet in `results.FLAGS` and are requested: poor_fit 64 (χ²/ν > 3; the error is scaled) and outlier 128. Bits 1, 2, 32 and 128 exclude a measurement from the method's Birge combination.
- `methods.REGISTRY[name]` is a lazy, picklable callable; `REGISTRY[name].measure(...)` (as the pipeline calls it) and `REGISTRY[name](...)` are equivalent.
- **Segments (plan technique 16).** Windows longer than `segment_res` = 150 resolution elements (≈450 Å at R 2500) are split into near-equal segments in ln λ, each a separate measurement, so segment-to-segment scatter enters the Birge and hierarchical combinations. This was added at integration, when `regions.select(telluric_corrected=True)` returned a single 7324–10 137 Å window for J1745 and both methods fell to z > 2 (see below); 150 was fixed a priori (about five segments over a typical optical spectrum), not tuned.
- **Gross outliers.** Among at least three usable measurements, one with |rv − median| > 5√(s² + σ²) (s = 1.4826 MAD) is flagged `outlier` and excluded. This catches only failures (a fit on a wrong optimum), not ordinary scatter.
- **Unknown R.** `resolution_of` takes the method option, then `cfg.input.resolution`, then `spectrum.resolution`; if none is set, it calls `estimate_resolution` (template-fit LSF; coarse χ² profile over R = 300–100 000, refined over ×/÷1.5), caches the result per spectrum and logs a warning. `estimate_resolution` is exported from `rvfitter.methods`. A telluric-line estimate will be added with the telluric model in unit 5.
- `templates` config: `local_dir`, else the downloaded grid in the cache (`rvfitter templates download`); `templates.medium` overrides the declared medium (still verified).
- `template_fit` accepts `telluric_model` (written by `config.default_for` for the NIR regimes); it logs a warning until unit 5 implements it.

### `ccf.py`
- Data are rebinned onto a ln λ grid at native sampling, divided by a normalised-Gaussian-convolution continuum (FWHM 25 resolution elements by default) and correlated with the equally high-passed, R-broadened template over an RV grid (default −500 to 500 km/s, step ≤5 km/s). The peak is located by a parabola through its top (±max(2 steps, 0.2 HWHM)). A Gaussian fit over the upper half of the peak was the first default, but the slow injection test at R = 20 000, SNR 30 and vsini 30 km/s showed a bias of −0.18 ± 0.02 km/s (−0.8σ) for it and −0.003 ± 0.024 km/s for the parabola: the CCF of a real line list is asymmetric, and noise moves the half-maximum boundary of the fitted region. Noiseless spectra show no bias with either, so the Gaussian is kept only as an option.
- Error: the standard deviation over Monte Carlo noise trials (100 by default) multiplied by √max(1, χ²/ν) of the scaled template against the normalised data. This scaling was added after J1745 showed χ²/ν ≈ 50–130 from template mismatch, where the pure noise error (0.6–0.8 km/s) was far below the window-to-window difference. It changes errors only, follows the same rule as the template fit, and leaves the synthetic pulls unchanged (χ²/ν ≈ 1).
- The Zucker (2003) ML error is stored as a diagnostic. On synthetic spectra it is ≈25 per cent smaller than the Monte Carlo scatter, probably because of the resampling and high-pass, so it is not used as the error.

### `template_fit.py`
- The model is P(x)·T(λ/D(v)) on the observed pixels (no data resampling); P (degree 3) is solved linearly at each trial, v is found by a coarse scan and bounded refinement, and vsini optionally by L-BFGS-B from several starts.
- Error: covariance 2H⁻¹ of the profiled χ², scaled by √(χ²/ν); optional pixel bootstrap (tested to agree with the covariance error within 0.7–1.4).
- `measure_resolution()` profiles χ² over R (parabola in ln R).

### Injection–recovery (seeded; tests/injection.py)

| Test | N | bias/σ | pull width |
|---|---|---|---|
| CCF, R 2500, SNR 30 (3 segments) | 40 | passes (< 0.2) | passes (0.8–1.2) |
| Template fit, R 2500, SNR 30 (3 segments) | 60 | passes | passes |

A trial run with 30 realisations gave a CCF bias of 0.13σ and pull width 1.04, and a template-fit bias of 0.09σ and pull width 1.03. The slow tests (R 2500/6000/20 000, SNR 15–40, vsini 0–30 km/s, N = 120; the R 20 000 case over 8000–8500 Å to keep it to a minute) are marked `slow`, and all six pass.

## J1745−1640 benchmark

Benchmark 26.2 ± 2.3 km/s (Burgasser et al. 2015); z = |RV − 26.2| / √(2.3² + σ²).
Input: `Science/DENISJ174534.6-164053_tellcorr.fits`, read with `io.read_spectrum(medium='vacuum')` (PypeIt vacuum and already heliocentric, so the barycentric step is skipped). The file carries no resolution.
Templates: the local BT-Settl-CIFIST grid, Teff prior 2100 K ± 300 K (21 candidates), chosen by χ². Seed 42.

### Resolution
`estimate_resolution` gives **R = 2274 ± 149** on the `regions.select` window (whole spectrum, `telluric_corrected=True`). On the earlier hand-made W1/W2 windows with the telluric bands masked, `measure_resolution` gave R = 1940 ± 160. Both are below the nominal 2500; the difference between them probably reflects template mismatch in the line depths, which differs between regions, so the estimate is uncertain at the ≈15 per cent level.

### Current benchmark (`regions.select`, telluric_corrected, segments; tests/test_benchmark_j1745_methods.py)

| Method | R | RV [km/s] | z | Per segment [km/s] | Template |
|---|---|---|---|---|---|
| CCF | 2274 (estimated) | 32.65 ± 13.59 | 0.47 | 7324–7816: 22.8 ± 14.9; 7816–8341: 35.5 ± 4.2; 8341–8901: 49.6 ± 3.8; 8901–9499: −9.7 ± 11.5; 9499–10137: −44.6 ± 8.8 | 2100/5.5 |
| Template fit | 2274 | 23.67 ± 18.51 | 0.14 | −249.2 ± 32.3; 32.0 ± 4.1; 49.2 ± 5.5; −13.0 ± 10.1; −42.2 ± 9.5 | 1900/5.5 |
| CCF | 2500 (nominal) | 32.27 ± 11.63 | 0.51 | 7324–7732: 176 ± 126; 7732–8162: 29.4 ± 5.9; 8162–8616: 47.6 ± 3.5; 8616–9096: 34.3 ± 4.7; 9096–9602: −52.1 ± 13.1; 9602–10137: −37.8 ± 9.5 | 2100/5.5 |
| Template fit | 2500 | 32.93 ± 11.66 | 0.57 | −109.9 ± 75.7; 26.0 ± 4.9; 47.3 ± 3.0; 35.6 ± 7.0; −51.9 ± 10.9; −33.4 ± 10.6 | 2000/5.0 |

The CLI run `rvfitter measure <tellcorr> --config j1745.toml --bary skip --telluric-corrected --medium vacuum` writes one results row with finite `rv_ccf`, `rv_template_fit` and `rv_adopted` (tested).

### Investigation: the single-window result (z > 2)
Before segmentation, `regions.select(telluric_corrected=True)` gave one unmasked 7324–10 137 Å window, and the methods returned 13.5 ± 4.2 (CCF, z = 2.6) and 6.4 ± 5.3 km/s (template fit, z = 3.4).
Fitting 350 Å slices showed why. The 7700–9100 Å slices agree (25–63 km/s, mostly 30–48 km/s), whereas:
- every slice redward of ≈9100 Å, i.e. the H2O band and its correction residuals plus the FeH Wing–Ford band, gives −10 to −64 km/s;
- the short TiO/VO stretch blueward of the O2 A band gives ±100–530 km/s with large errors (a poorly matched molecular band with no sharp features).

A single window lets the long, high-weight H2O region pull the answer towards the telluric frame, and its error (4–5 km/s) hides the disagreement. With segments, the disagreement inflates the Birge error to 12–19 km/s, and the result is consistent with the benchmark.
We therefore recommend to the regions package that the strong H2O band (≈9270–9680 Å) stay masked for the fitting methods even on telluric-corrected spectra; residuals of the correction are not random noise.

### History (hand-made regions, before integration)
W1 7400–8600 Å and W2 8600–10 000 Å with the O2 A (7590–7710 Å) and H2O (9270–9680 Å) bands masked, R = 2500: CCF 35.57 ± 12.09 (z = 0.76; W1 44.1, W2 18.4), template fit 26.05 ± 9.49 (z = 0.02; W1 33.9, W2 14.5). At the then-measured R = 1938: CCF 34.52 ± 11.50, template fit 23.95 ± 8.54.
In every configuration, the blue-to-red trend (≈45 km/s near 8200–8600 Å and below 0 redward of 9100 Å) is the same line-to-line scatter that the Phase A hierarchical combination found (τ_line ≈ 13 km/s). It probably comes from BT-Settl mismatch, which differs between the TiO/VO/CrH and FeH/H2O regions, and from telluric-correction residuals.
The two methods choose different templates, because the CCF's χ² is computed after the high-pass whereas the template fit's χ² includes the continuum polynomial.

## Diagnosis: why the J1745 ccf/template_fit errors are 13–25 km/s

Here, we investigate why the J1745−1640 errors are an order of magnitude larger than the 2–3 km/s that MagE reaches at R ≈ 4100, or the ≈5 km/s of the paper's own methods on this spectrum.
Set-up: `regions.select(telluric_corrected=True)` with the H2O bands now masked (spans 7342–7592, 7702–8952 and 9903–10 136 Å), R = 2274 (`estimate_resolution`), segments of 150 resolution elements, seed 42 (script `tmp/m/diag_j1745.py`, not in the repository).
The median SNR is 87 per pixel (89 in 7700–8950 Å), so photon noise is not the limit.

### (a) Per-segment RVs and errors
| Segment [Å] | CCF [km/s] | CCF stat (MC) | ×√(χ²/ν) | Template fit [km/s] | TF stat (unscaled cov.) | ×√(χ²/ν) |
|---|---|---|---|---|---|---|
| 7702–8098 | 23.0 ± 8.1 | 2.10 | 3.8 | 19.5 ± 7.2 | 1.43 | 5.0 |
| 8098–8514 | 46.9 ± 4.0 | 0.99 | 4.0 | 41.6 ± 4.9 | 1.01 | 4.9 |
| 8514–8952 | 41.6 ± 4.4 | 0.68 | 6.5 | 43.4 ± 5.9 | 0.68 | 8.8 |
| 9903–10 136 | −74.8 ± 11.9 | 1.75 | 6.8 | −73.3 ± 7.2 | 1.10 | 6.5 |
| 7342–7592 | −454 (outlier) | 14.2 | 1.5 | −454 (outlier) | 6.3 | 1.8 |
| **Combined (Birge)** | **35.9 ± 15.5** | formal 0.52 | Birge ratio 37 (stat) / 5.7 (scaled) | **17.9 ± 24.9** | formal 0.47 | 54 / 8.2 |

The statistical (noise) errors are 0.7–2 km/s per segment, and 0.5 km/s combined.
Two inflations act on top of them: the √(χ²/ν) scaling (χ²/ν = 15–77 from template mismatch), and the Birge ratio of the segment scatter, which is still 5.7–8.2 even with the scaled errors.
The large final errors therefore come from the segments disagreeing, above all the red 9903–10 136 Å segment at −75 km/s, and not from the noise.

### (b) Injection–recovery on a J1745-like synthetic
BT-Settl 2000/5.0 was broadened to R = 2274, shifted to 26.2 km/s and put on J1745's pixel grid with its σ, masks and segments. Its continuum is the J1745/model ratio smoothed with a 20 000 km/s Gaussian; a 3000 km/s smoothing left continuum structure that neither the polynomial nor the filter follows, giving χ²/ν ≈ 60. N = 30 (15 for the grid search):

| Template used | Method | Mean | Scatter | Mean error | Pull mean / SD | χ²/ν (seg. 1) |
|---|---|---|---|---|---|---|
| Matched (2000/5.0) | CCF | 26.15 | 0.40 | 0.47 | −0.11 / 0.80 | 3.1 |
| Matched | Template fit | 26.18 | 0.32 | 0.39 | −0.05 / 0.82 | 0.96 |
| Mismatched (1800/5.5) | CCF | 27.21 | 0.63 | 1.75 | 0.60 / 0.39 | 48 |
| Mismatched | Template fit | 29.08 | 0.44 | 3.33 | 0.87 / 0.09 | 91 |

At J1745's R, SNR and coverage, the achievable statistical precision is therefore 0.3–0.5 km/s with calibrated pulls. A wrong grid template biases the RV by 1–3 km/s; its χ² scaling then over-inflates the error relative to the scatter (pull SD 0.1–0.4) but roughly covers the bias.
The real data scatter by 10–100 km/s between segments, 30–200 times the noise. The excess is therefore not noise: it is template mismatch and/or something in the data.

**CCF fix found here.** In the first run, the matched-template CCF had χ²/ν = 15 and a +0.6 km/s bias. The cause was that the data were high-passed within their window and mask, whereas the template was high-passed over its own full range, so the two continua differed near window edges and on the steep L-dwarf band slopes. The CCF now shifts the raw template onto the data grid for each trial velocity and applies exactly the data's filter (same kernel and pixel mask) to every row. This removed the bias and brought χ²/ν to 3.1 and the pull SD to 0.80.

### (c) Sensitivity (real data)
- **Template** (pinned; CCF / template fit, km/s): Teff 1800–1900 K gives 31–34 / 15–30 at every log g; 2000–2200 K gives 33–48 / 18–44. Where the red segment is rejected as an outlier (e.g. 2000/4.5, 2200/5.0), the errors fall to 4–10 km/s, and the values (37–48 km/s) are those of the 7700–8950 Å window alone.
- **CCF high-pass FWHM** of 10–100 resolution elements: 34.8–35.9 km/s; the segments move by ≤5 km/s. The filter is not the problem.
- **Segment length** (pre-fix CCF): unsegmented 25 ± 54 (CCF) and −28 ± 94 km/s (template fit); 75 resolution elements gives six segments at 13–65 km/s in the main window plus the red segment at −73/−85. The main-window segments scatter by ≈10–15 km/s at any length.
- **Template-fit polynomial degree** 1/3/5: 12.9/17.9/30.9 km/s (the segments themselves move by ≤5 km/s; the combined value moves through the weights).

### (d) Statistical versus systematic terms, and what drives the error
- The statistical term is small: 0.5–2 km/s per segment and about 0.5 km/s combined, consistent with the synthetic.
- The error is driven by:
  - the red 9903–10 136 Å segment (−73 to −86 km/s for every template, filter and segmentation);
  - a 10–15 km/s scatter between segments in 7700–8950 Å, whose mean is ≈40 km/s;
  - χ²/ν = 15–80.
- The red segment is the same with every template, so it is probably not template physics alone. It sits at the red edge of the detector (fringing, sparse arc lines, so possibly a λ-solution extrapolation error; 75 km/s ≈ 2.5 Å) inside the FeH Wing–Ford band, which BT-Settl reproduces poorly.
- The main window's ≈40 km/s is ≈14 km/s above the benchmark, like the paper's line centring (36 km/s). A λ zero-point offset of ≈0.4 Å at 8500 Å would do this; unit 5's telluric zero-point (O2 A and H2O lines in the uncorrected `sciencebk` spectrum) tests it directly.

**Recommendation.**
- Report per-segment errors as statistical only (MC noise or unscaled covariance), without the √(χ²/ν) factor. At present the mismatch is counted twice: once by the χ² scaling of each segment, then again by the Birge inflation of the segment scatter.
- Let the segment-to-segment scatter (Birge at method level; τ_line in the hierarchical adopted RV) carry the mismatch.
- Report `template_spread` (the RV spread over the grid templates within the prior) as a separate systematic, as the report recommends (statistical, template-spread and λ zero-point terms separately, plus their total).
- With this, J1745's statistical error would be ≈0.5 km/s, and the per-method errors (Birge) would still show the segment disagreement.
- The decision is the orchestrator's; it is not implemented yet.

## Review of units 2–3: changes

### Blocker: CCF continuum handling (34133d6 and follow-up)
- **Same filter for data and template** (34133d6): each shifted template row is high-passed on the data's pixels with the data's kernel and mask.
- **No resampling of the data**: the CCF now works on the observed pixels. Within a window the velocity step is near-constant, and data and template rows pass through the same filter, so a uniform ln λ grid is unnecessary.
- **Filter narrower than the window**: the high-pass FWHM is capped at a third of each window, since a filter wider than a 60–75 Å feature window cannot remove the continuum slope inside it.
- **Relative slope removed**: after a first CCF, the data are divided by a degree-1 polynomial fitted to data/template at the peak (`flatten_degree`), and the CCF is redone. A continuum slope that the template lacks otherwise leaves a slope × line term after the high-pass: noiseless J1745-like injections gave −0.1 to −0.55 km/s per feature window, and a flat continuum removed it.
- **Edge-clipped features dropped**: `regions` drops a feature window when a feature lies within 3 resolution elements of an edge clipped by a telluric mask. On J1745 this is Cs I 8946, at the H2O band, which kept a −0.3 km/s noiseless bias.
- **Slow test** (`test_j1745_like_injection`, N = 80): BT-Settl 2000/5.0 at R = 2300, injected into J1745's pixels, σ, masks and feature windows, with the exact template.

  | Code | Method | Bias [σ] | Pull SD |
  |---|---|---|---|
  | Current | CCF | −0.10 | 0.90 |
  | Current | Template fit | −0.18 | 0.88 |
  | Pre-34133d6 (from 1dd9378) | CCF | −1.58 (−0.97 km/s) | 0.38 |

  The test therefore catches the blocker.

### Feature windows (catalogue-driven)
- `features.ecsv` has a new `model_fidelity` column: good for atomic lines and CO bandheads; poor for TiO, VO, CrH, FeH, H2O and CH4; n/a for tellurics. The catalogue description cites deep-research-report.md ("Synthetic versus empirical templates", "Synthetic-template mismatch") and Hsu et al. (2021).
- `regions.template_windows = "features"` (default) or `"full"`. With "features", for ccf, template_fit and forward_mcmc at R ≤ 5000, the windows are ±10 resolution elements around the good features, merged where they overlap and clipped by the telluric masks. If none is usable, the full spans are used with a warning. The choice and its reason are logged at INFO.
- `systematics.template_spread` reports the full-minus-features difference (`delta_modes`) alongside the SD over grid templates. The total is the SD over every variant.

### Nits
- template_fit scales by √max(1, χ²/ν), as the CCF does.
- `Template.at` averages the shifted template over each observed pixel by default, matching pixel-integrated data. The toy `synth.py` spectra are point-sampled, which leaves a 0.07σ (0.13 km/s) template-fit bias at N = 200. Pixel-integrating `synth.make_spectrum` (a P0 file) is suggested.
- The `_onto_internal` docstring notes the spline's flux non-conservation for unresolved features.
- Injection tests calibrate the statistical error on single windows (`segment_res=0`). With segments, the Birge factor max(1, √(χ²/(N − 1))) is ≥1 by construction (≈1.15 for three segments of pure noise), so those errors are conservative; this is tested separately (pull SD > 0.65).
- The J1745 segment scatter reflects the window choice and model mismatch. Given that scatter, the quoted errors are honest (the reviewer's independent CCF reproduced it).

### J1745 after the review fixes
R from `estimate_resolution` is now **2602 ± 198** (feature windows and pixel-integrated templates), consistent with the nominal 2500. Seed 42, telluric_corrected, H2O masked.

| Windows | Method | R | RV [km/s] | z | Per window [km/s] | Template |
|---|---|---|---|---|---|---|
| features | CCF | 2602 | 41.46 ± 6.14 | 2.33 | Cs I 8523: 51.4 ± 2.9; Na I: 33.9 ± 3.8; Rb I 7802: 38.6 ± 5.9; Rb I 7949: 20.8 ± 6.0 | 2000/5.5 |
| features | Template fit | 2602 | 43.19 ± 5.21 | 2.98 | 50.9 ± 3.7; 36.8 ± 5.6; 38.2 ± 9.2; 26.0 ± 7.7 | 2000/5.0 |
| features | CCF | 2500 | 41.40 ± 6.03 | 2.35 | 49.9; 32.5; 38.8; 19.3 | 1800/5.5 |
| features | Template fit | 2500 | 44.23 ± 4.82 | 3.38 | 50.1; 36.7; 38.7; 25.2 | 2000/5.0 |
| full | CCF | 2602 | 35.38 ± 15.62 | 0.58 | 24.8, 46.1, 41.2, −72.2 (red edge), −456 (outlier) | 2100/5.5 |
| full | Template fit | 2602 | 19.17 ± 24.31 | 0.29 | 19.0, 41.5, 42.8, −70.7, −456 (outlier) | 2000/5.0 |

The full − features differences (−6 km/s for the CCF and −24 km/s for the template fit) enter the template systematic.
**Feature windows disagree with the benchmark at z = 2.3–3.4. This is investigated, not tuned away.**
- The per-line values are stable across method, template (1800–2200 K, log g 4.5–5.5) and R (2274–2814): Cs I 8523 ≈ 50 km/s, Na I ≈ 35 km/s, Rb I 7802 ≈ 38 km/s and Rb I 7949 ≈ 22 km/s.
- Cs I 8523 has the smallest error and so dominates the weighted mean; the unweighted mean of the four lines is ≈37 km/s. The Cooper et al. (2024) line centring found the same pattern on this spectrum (Cs I a 48–55 km/s; line-centring mean 36 km/s).
- With the same feature windows, the hierarchical adopted RV will weight the lines more equally through τ_line.
- The remaining offset of the atomic lines from 26.2 km/s was attributed to a wavelength zero-point offset. The OH sky lines measured it (reviewer, then `calibration.sky_zeropoint`; unit 3c): +0.54 ± 0.14 Å (≈+19 km/s) at 8500 Å in the observatory frame, varying with wavelength (+5 to +30 km/s within 7700–8960 Å, and ≈−73 km/s at the red edge). It is therefore not the uniform ≈0.3 Å first guessed.
- The reviewer's independent check gave ≈30 km/s with the same idea. It differed in including the K I lines (24 km/s; masked here, because the O2 A band stays masked for template spans) and Cs I 8946.

## Unit 3b: grid parameters, edges and extensible grids

### (c) N-dimensional grids
- `GridMeta` has the standard axes teff, logg, met and alpha, plus named `extra_axes` (e.g. Elf Owl `co`, `kzz`; Diamondback `fsed`), with `extra_defaults`. It also carries a `coverage` table (the documented extent of the full grid, used only for guidance) and a `verified` marker. `medium` may be `"auto"`: it is measured from the NIST alkali minima on first use, and `download()` records the measured value in the metadata.
- `TemplateGrid` interpolates multilinearly over every axis (2^d corners; any missing corner falls back to the nearest point, with `outside_grid` set).
- `select()` takes every teff and log g within their ranges. It holds met, alpha and the extra axes at the value nearest their prior or default unless a range is set, so solar-only grids behave exactly as before and the candidate count does not multiply silently. `bounds()` is now `{axis: (min, max)}`.
- `KNOWN_GRIDS` entries:
  - bt-settl-cifist: air, verified;
  - bt-settl-agss2009 ([M/H] −4 to +0.5, [α/M] 0 to 0.4);
  - sand (Alvarado et al. 2024; [M/H] −2.4 to +0.3, [α/M] −0.2 to +0.6);
  - sonora-bobcat;
  - sonora-diamondback (fsed);
  - sonora-elfowl (C/O, Kzz);
  - lowz (Meisner et al. 2021; [M/H] −2.5 to +1, C/O, Kzz; may not be on SVO, so it can be used as a local directory with its own GridMeta).

  Every grid apart from CIFIST is marked `verified=False` with medium `"auto"`. SVO was unreachable, so the model ids are best guesses: `rvfitter templates list --available` and the `network`-marked slow test check them live.
- `download()` writes generic file names (`t2000_g5.00_m-0.50_a+0.20[_co…][_kzz…][_fsed…].txt`), queries every requested axis, and keeps the SHA-256 manifest. The CLI takes `--teff/--logg/--met/--alpha MIN MAX` and `--range AXIS MIN MAX`.
- `templates.grid` may be a list. Candidates are drawn from every grid, the best is chosen by χ² across all of them, and `template_spread` reports the SD over grids (`sd_grids`). `templates.local_dir` may be a mapping grid → directory. New config keys: `alpha_range`, `alpha_prior` and `ranges` (extra axes).

### (a) Best-template output
- Each template method stores `template_summary` in its diagnostics: the best grid and label, `template_<teff|logg|met|alpha>` (NaN where the grid has no such axis, i.e. it is neither encoded in the file names nor has more than one value), `template_chi2` and `template_at_edge`.
- It also stores a χ²-weighted mean ± SD per axis over the candidates of the best grid, with weights exp(−Δχ²/(2s)) and s = max(1, χ²_min/ν), so that a mismatch-dominated fit does not give spuriously sharp parameters. The scalar keys reach the details table.
- The results table gains `template_method`, `template_grid`, `template_teff`, `template_logg`, `template_met`, `template_alpha`, `template_chi2` and `template_at_edge`, taken from the first of template_fit, forward_mcmc, ccf or lsd that ran.
- The best template and the χ²-weighted parameters are logged at INFO.

### (b) Edges
- If the best candidate is at the low or high end of a searched axis with at least two values, the new flag `template_param_edge` (256) is set; it is informational, not an exclusion. A WARNING then gives the axis and direction and one of:
  - "widen `templates.<axis>_range` to [a, b]" when the grid extends further;
  - when the grid itself ends there, a KNOWN_GRIDS grid whose coverage extends beyond it and which covers the current teff/logg, with the exact `rvfitter templates download --grid … --met lo hi` command and the config change (`templates.grid = […, new]` and the new range).
- The same guidance is available as data (`cfg_patch`, `download` kwargs, `command`) for a GUI. `outside_grid` stays separate.

### (d) Reproducible refits
- `pipeline.refit_object(result, cfg, cfg_patch, …) -> (ObjectResult, new_cfg)` deep-merges the patch into a copy of the config, validates it, appends `{'object', 'patch'}` to `new_cfg.refits` (a new top-level config list, written to the TOML as `[[refits]]`), and remeasures the object from its files.
- The seed is that of a run on the object alone, so `rvfitter run` with the written TOML reproduces it.
- `pipeline.suggested_refits(result)` lists the edge actions for a GUI "Download grid & refit" button: call `templates.download(**action['download'])` if it is not None, then `refit_object(result, cfg, action['cfg_patch'])`.
- `pipeline.apply_patch(cfg, patch)` is exposed too.

### Tests
`tests/test_grids_nd.py`:
- a synthetic 4D grid plus an extra axis (exact multilinear interpolation over 32 corners), and the holding of unranged axes;
- the auto medium;
- edge guidance for "widen" and for "download" (command, patch and message), and no edge inside the grid;
- the edge flag set by template_fit, the results and details columns, and `suggested_refits`;
- `refit_object` (patch applied, original untouched, recorded and round-tripped in the TOML, deterministic seed);
- a live SVO check (`slow`, `network`, skipped offline).

The new `network` marker is registered in pyproject.toml.


## Unit 3c: random-effects combination, sky zero point, wavelength correction

### Per-method combination
- `combine.combine_regions` is now a random-effects model, `hierarchical` with one method: regions are the lines, τ is free under a half-Cauchy prior, and the result is the median and half the 16–84 per cent width.
- When τ ≫ σ_i the regions are weighted almost equally. Inverse-variance weights with a Birge factor instead let one region with a small error set the answer (J1745: Cs I 8523).
- The Birge value is kept for comparison in each measurement's diagnostics (`method_birge_rv`, `method_birge_err`), together with `method_tau`.
- With few regions the random-effects error is conservative, because τ is marginalised. The injection tests therefore calibrate the statistical errors per window (pull SD 0.8–1.2) and check the method level only as conservative (0.6–1.2).

### OH sky zero point (`calibration/sky_zeropoint.py`)
- **Data:** `Spectrum.sky` is a new optional field, kept through `cut` and `resample`. io reads it from PypeIt spec1d `OPT/BOX_COUNTS_SKY`, and from a `sky` column in generic FITS tables and ASCII files.
- **OH list:** `data/oh_sky_vacuum.csv` is PypeIt's `OH_MODS_lines.dat`, generated by its `wavemodel.py` from the Rousselot et al. (2000) OH list, in vacuum. It is redistributed under PypeIt's BSD-3-Clause licence, with the notice in `data/PYPEIT_LICENSE.txt`. We checked it against PypeIt's vetted single vacuum sky lines: median −0.003 Å, MAD 0.016 Å over 36 lines. `paranal_sky.fits` was not used, because its provenance ("SDSS code") is unclear.
- **Method:**
  1. Undo the header velocity correction (PypeIt `VEL_CORR`), so the sky is in the observatory frame.
  2. Subtract a running 20th percentile, leaving the emission lines.
  3. Measure the sky FWHM by maximising the correlation peak over lags.
  4. Cross-correlate 250 Å segments with the OH model, Hann-tapered so that lines cut by the segment edges do not bias the lag.
  5. Reject segments that overlap telluric bands of catalogue depth ≥ 0.3 by more than 20 per cent, have fewer than three strong lines, or correlate below 0.7.
  6. Fit Δv(λ) as a polynomial (degree 0–3 chosen by BIC, leaving at least one degree of freedom), with the covariance scaled by max(1, χ²/ν). Δv is held constant beyond the span of the used segments, with no extrapolation.
- **Synthetic test:** a linear Δv curve is recovered to ≤2 km/s, in both frames.
- **J1745:** the Science spec1d sky gives +19 to +30 km/s between 7800 and 8900 Å (the exact numbers depend on the segmentation), and −73 km/s in the 9888–10 146 Å segment. That is the same value as the stars' red-edge segment (−70 to −86 km/s), which confirms that it was a distortion of the wavelength solution.

### Wavelength correction in the pipeline
- `calibration.wavelength_correction = "auto" | "sky" | "telluric" | "none"` (default `"auto"`), plus `calibration.sky_source` (a file whose sky shares the input's wavelength solution, e.g. the spec1d of a coadd), `sky_segment_width` and `sky_degree`.
- The zero point is measured on the input spectrum and applied once to the wavelengths, after the barycentric step and before regions and methods. Any method sees only corrected wavelengths.
- Every measurement records `zp_applied_kms` and `rv_uncorrected` = rv + Δv(λ of its region) in its diagnostics (first order) and so in the details table.
- The results gain `zp_sky_kms`, `zp_sky_err`, `zp_sky_slope` and `wavelength_correction`. `systematics.wavelength_zero_point` is the larger of the arc-RMS term (a lower bound) and the sky zero-point error.
- The telluric constant comes with unit 5. Until then `"telluric"` logs a warning and applies nothing, and `"auto"` applies the sky alone.

### CCF continuum
- Feature windows now carry continuum windows: the outer `cont_fwhm` (3) resolution elements at each end.
- CCF windows of ≤100 Å with continuum windows use an explicit linear continuum, fitted σ-weighted to those pixels. Each shifted template row is fitted with the same pixels and weights, so data and template stay normalised identically. The relative-slope flattening is then unnecessary and is skipped.
- Diagnostics record `continuum_mode`, `filter_fwhm` (high-pass windows) and `flatten_degree`.
- The J1745-like slow test passes, with per-window pulls calibrated.

### J1745: no correction vs sky correction
Through `pipeline.run`: tellcorr input, sky from the Science spec1d, R = 2602, feature windows, heliocentric (barycentric step skipped). All choices were fixed before the comparison.

| Correction | CCF | Template fit | Adopted (hierarchical) | τ_line | z (adopted) |
|---|---|---|---|---|---|
| none | 36.75 ± 4.32 (z 2.16; τ 6.0; Birge 37.1) | 39.25 ± 5.87 (z 2.07; τ 8.9; Birge 42.5) | 37.85 ± 4.96 | 7.9 | 2.13 |
| sky | 18.17 ± 3.34 (z 1.98; τ 2.9) | 19.07 ± 3.41 (z 1.73; τ 2.9) | 18.63 ± 3.35 | 2.2 | 1.86 |

- **Benchmark-free test:** the line-to-line scatter falls from τ_line = 7.9 to 2.2 km/s, and the two methods agree to 0.9 km/s. The per-line values move by 10–30 km/s, by the local Δv.
- With the sky correction, the total error is 6.5 km/s (sys_wavezp 3.8 and sys_template 4.2).
- The corrected value is 7.6 km/s below the benchmark. The sky cannot see the star's own offset from slit centring; only the telluric absorption lines, which share the starlight's path, can. That term (unit 5) is the next test, and no correction is chosen by its effect on 26.2.
- Per-line values with the sky correction (CCF): Cs I 8523 19.6, Na I 12.9, Rb I 7802 24.6 and Rb I 7949 14.5 km/s (uncorrected 49.6, 36.1, 34.4 and 24.2 km/s).

## Unit 3b-fix: grid holes, reproducible refits, Zenodo grid source

- **Holes:** a request inside the grid's range whose multilinear corners include a missing file falls back to the nearest point with the new flag `template_grid_hole` (512). The missing corner files are named in a WARNING. We do not renormalise over the corners that exist, because that would bias the template. `outside_grid` / `template_grid_edge` now means only a request outside an axis's range.
- **Seeds:** each object's RNG is `SeedSequence([cfg.seed, blake2b(name)])` (`pipeline.object_seed`), independent of the other inputs and their order.
- **Refits:** a refit is recorded as `{'object', 'patch'}` without changing the global values. `pipeline.run` applies each patch only to its object (`config_for_object`), and `Config.hash()` excludes the refit list, so it describes each object's configuration.
- **End-to-end test** (`tests/test_refit_e2e.py`): two synthetic objects are run, the second is refitted with a wider `met_range`, the TOML is dumped and rerun from scratch, and the CSV tables are byte-identical. The unpatched object is unchanged, and the patched object alone gives the same row.
- **Download defaults:** `templates.download` holds an unranged met/alpha at the grid default (SAND: the near-solar chemistry [M/H] +0.1, [α/M] +0.15, since its chemistries are specific pairs). All requests carry an rvfitter User-Agent.
- **SVO** now answers HTTP 403 Forbidden from this host (curl too), so its model ids remain unverified. The network test skips on 403 or a timeout.
- **Zenodo** (`grid_sources.py`; `templates.download(source="auto" | "svo" | "zenodo")` and CLI `--source`; auto tries SVO first, then Zenodo). The records were verified live on 2026-10-27; all are CC-BY-4.0:

  | Grid | Record(s) | Archive | Format |
  |---|---|---|---|
  | Sonora Bobcat | 5063476 | `spectra_m{met}.tar.gz`, `…_co{0.5,1.5}_g1000nc.tar.gz` | text, μm, F_ν, g [MKS] |
  | Sonora Diamondback | 12735103 | `spectra.zip` (1445 members) | text, μm, W/m²/m; f_sed (nc = 0) |
  | Sonora Elf Owl (Paper IV) | 15150865, 15150874, 15150881 | `teff_{lo}_{hi}.tar.gz` (≈9.5 GB each) | netCDF-4 (h5py), μm, erg/s/cm²/cm |
  | SAND | 11582126 | `SAND.zip` (5181 "recommended" FITS) | FITS table, Å, erg/s/cm²/Å, AIRORVAC = VAC |

  - zip archives are read remotely with HTTP range requests, so only the selected members are transferred, each checked by its CRC-32.
  - tar.gz archives are streamed once, with only the selected members converted, and the whole stream is checked against Zenodo's MD5 (a mismatch deletes the converted files).
  - Member names are checked as in `archives` (unsafe ones are skipped with a warning), and output names are generated from the parsed parameters. Each member is capped at 500 MB.
  - The spectra keep their native sampling and resolution (Å, F_λ). The medium is the header's (SAND) or measured on opening; both live tests gave vacuum.
  - Live tests: SAND (three models at 1900–2100 K in 4 s, declared vacuum confirmed) and Bobcat (two models from a 108 MB stream in 40 s, MD5 verified, measured vacuum).
- **Upstream URLs** are now on every grid. LOWZ is on Harvard Dataverse, and BT-Settl CIFIST and AGSS2009 are on the Lyon server (which times out from here); these are manual downloads, opened with a GridMeta, and `download` says so.
- SED_Tools / NJM was not used: it serves only R ≈ 1000 BOSZ and BT-Cond spectra, standardised for SED work, and disables SSL verification on failure.
- **CI:** `-m "not data and not gui and not slow and not network"`.


## Unit 3c-fix

- **Epochs (the blocker):** each epoch is measured separately, and its regions are relabelled `<epoch>:<region>` (the file stem, else the MJD, else e1, e2, …). Epoch disagreement therefore enters the random-effects (method) and hierarchical (adopted) scatter.
  - Two synthetic epochs at 10 and 30 km/s (σ ≈ 1) now give an error above 5 km/s and are flagged variable (`tests/test_epochs.py`); before, they gave 20 ± 0.7.
  - The per-epoch method and adopted RVs are in each measurement's diagnostics. `adopted['epochs']` holds the χ² test, and the results gain `rv_epoch_pvalue`, `rv_epoch_variable` and `n_epochs`.
- **Sky curve:**
  - Segments are cut from the telluric-free spans, so the blue edge starts after the O2 A band.
  - Each segment is correlated on the data's own pixels against the shifted OH model. Rebinning had biased few-line segments by several km/s (7952–8202 Å on the synthetic: 6.5 against a true 11.1 km/s), and the Zucker N now counts native pixels, so the errors are realistic.
  - Δv(λ) is piecewise linear within each span and held to the span boundaries. It ramps only inside unmeasured gaps (continuous, so the wavelengths stay monotonic; `apply` refuses a non-monotonic result), and its error grows with distance.
  - The model spread (interpolation, linear, constant over the main span) is included in `zp_sky_err`.
- **J1745 sky segments (native pixels):**

  | Segment [Å] | Δv [km/s] |
  |---|---|
  | 7451–7592 | −131.6 ± 4.9 (blue edge) |
  | 7702–7952 | +14.0 ± 3.4 |
  | 7952–8202 | +6.8 ± 2.0 |
  | 8202–8452 | +17.5 ± 1.7 |
  | 8452–8702 | +37.9 ± 8.7 (correlation 0.72) |
  | 8702–8952 | +22.3 ± 1.5 |
  | 9903–10 146 | −103.5 ± 3.2 (red edge) |

  The main span gives +20.1 ± 8.2 km/s (model spread 2.3). The earlier −73 and −95 km/s at the red edge came from the old grid-based segmentations (250 Å from 7311 Å, and the reviewer's defaults); the native-pixel value is −103.5. Both array edges lie beyond the arc lines (7437–9926 Å).
- **Default `wavelength_correction = "none"`** (coordinator decision): the zero point is measured and reported, and an INFO line asks the user to check it when |Δv| > 2σ.
- **Nits:**
  - `sky_source` coverage warning;
  - README credit for the PypeIt and NIST data;
  - these notes now state which segmentation produced which numbers.


## Unit 5: telluric zero point and telluric_ref

### Transmission model (`calibration/telluric.py`; built into the user cache since unit5-fix, never shipped)
- `tools/build_telluric_model.py` fetches HITRAN line-by-line data (H2O, CO2, O2, CH4; vacuum wavenumbers) from the HITRANonline API (fetched 2026-10-28; the API's Content-Length counts CRLF but it sends LF, and it answers 404 for empty intervals, both handled), and integrates them through a US Standard 1976 atmosphere above a 2.4 km site: dry gases in 24 layers, H2O with a 1.5 km scale height normalised to 1 mm PWV. It uses Voigt profiles with the air pressure shift, and wings cut at τ < 10⁻³ or 25 cm⁻¹.
- The zenith τ_dry and τ_H2O are stored at R_s = 300 000 (computed at 1.2 M, equivalent-width-preserving averages), 6000–25 000 Å. T = exp(−X(τ_dry + w·τ_H2O)).
- **Vacuum check:** at R = 20 000, the O2 A and B blue edges lie 1.0–1.3 Å redward of the Fraunhofer A/B lines in vacuum, and 2.8–3.4 Å from the air values (tested). An independent ESO SkyCalc/LBLRTM transmission (splat resources; vacuum; airmass 3, R = 60 000) agrees within 0.2 km/s in O2 B, O2 A, H2O 8200 and both halves of H2O 9300 (cross-correlation peaks 0.96–0.997).
- **Licence:** HITRAN states no redistribution licence; it is free and must be cited (citepolicy, read 2026-10-28). Superseded in unit5-fix: the model is no longer shipped, but built on demand into the user cache (`rvfitter templates telluric --build`).
- Omitted: O3, N2O, collision-induced and continuum absorption, Rayleigh and aerosols (smooth; absorbed by the continuum polynomial).

### Zero point (`calibration/telluric_zeropoint.py`)
- Observatory frame: `VEL_CORR` and the pipeline's barycentric shift (now recorded as `meta['barycentric_applied_kms']`) are undone.
- Segments: the catalogue's telluric bands, cut into ≤250 Å pieces in ln λ and padded by 15 resolution elements. A segment is used if a species is deeper than 5 per cent in the model.
- Model: P₂(λ) · [T(λ/D(v); a_dry, a_H2O) ⊗ LSF], averaged over the contiguous pixels (masked pixels are dropped from χ², not from the pixel edges). The deep species' τ scalings are free. One round of 5σ (robust) clipping is applied.
  - Without a template, catalogue stellar lines are masked (±5 resolution elements).
  - With a template (default in the pipeline when templates are configured: the grid point nearest the priors), the star is modelled as S(λ/D(v_star)) with v_star free, and nothing is masked.
- Errors come from the χ² curvature of the velocity profile (scalings re-optimised; a quadratic surface for (v, v_star)), multiplied by √max(1, χ²/ν). Δv(λ) is piecewise linear between the segment nodes (one span), and `zp_kms` is the Birge-inflated weighted mean.
- Without a known R, R is estimated from the telluric lines: 2555 for a synthetic R = 2500, and 2212 on J1745 (the methods use 2602; both are within the earlier ±15 per cent spread).
- **Tests:** a constant + linear distortion is recovered per segment within 3.5σ, in both frames. The R estimate, the star − sky constant (12 km/s injected), the pipeline modes and telluric_ref (RV 30 km/s through −21 km/s VEL_CORR and the distortion) are tested.
  - Slow pull test, N = 30 at SNR 60: passes for a constant zero point and for constant + distortion.
  - With a distortion, a segment measures the information-weighted mean of Δv over its pixels ((dT/dλ)² weights), not Δv at its centre. Against the centre, the pull SD was 1.95, because the 9300 Å segments' information is lopsided. The same caveat applies when the telluric nodes are compared with the sky curve, at the level of a few km/s for J1745's slopes.

### Pipeline
- `calibration.telluric_source` (+ `telluric_segment_width`, `telluric_template`) takes a non-corrected file. Its object is matched to the input by the high-pass flux correlation minus |ln flux ratio| outside the telluric bands. On J1745 and Ross 640 this picks HDU 2 (scores 0.21 against −1.93, and 0.93 against −0.44).
- `star_minus_sky`: the weighted mean of Δv_tell − Δv_sky(λ) over the telluric nodes, with the sky curve's distance-inflated error, so nodes in unmeasured sky gaps carry little weight. It is Birge-inflated, and its χ² tests "same shape + constant".
- New results columns: `zp_tell_kms`, `zp_tell_err`, `zp_star_sky` and `zp_star_sky_err`.
- `sys_wavezp` is:
  - the sky error ⊕ the constant's error for "auto";
  - the telluric curve's error for "telluric";
  - otherwise the unapplied telluric offset when it was measured, else the sky's.

### J1745: star (tellurics) against sky
Telluric fits on nopipecorr spec1d HDU 2 (the target), with the 2100/5.0 template, R 2602 and airmass 1.88. The sky comes from the Science spec1d (HDU 1).
All choices (segments, padding, degree, depth threshold, the template rule, and "auto" = sky shape + weighted constant) were fixed before any RV was computed.

| Telluric segment [Å] | Δv_tell [km/s] (χ²/ν) | Δv_sky(λ) [km/s] | Sky there |
|---|---|---|---|
| O2 A 7592–7702 | −44.4 ± 11.5 (56) | −59 ± 13 | ramp between the blue-edge node (−132) and 7827 Å |
| H2O 8122–8352 | −6.0 ± 5.5 (66) | +13.6 ± 4.1 | measured span |
| H2O 8952–9181 | −39.7 ± 3.8 (126) | +7 ± 13 | unmeasured gap |
| H2O 9181–9416 | −55.3 ± 2.2 (122) | −23 ± 25 | gap |
| H2O 9416–9656 | −47.0 ± 1.4 (52) | −55 ± 27 | gap |
| H2O 9656–9903 | −21.5 ± 3.2 (44) | −87 ± 15 | gap |

- **zp_star_sky = −10.2 ± 14.2 km/s** (χ² 37.4 for 5 dof). The one segment inside the sky's measured span gives −19.6 ± 6.9 km/s.
- A smooth-star fit (no template; stellar lines masked; R 2213 from the tellurics) gives −39, −5, −34, −54, −44 and −17 km/s, i.e. the same within 1–6 km/s. Template mismatch does not drive Δv_tell.
- **The star does not see "the sky's distortion plus a constant".** Over 8950–9900 Å the tellurics are nearly flat (−22 to −55 km/s), where the sky's ramp, which is only an interpolation across its gap, runs from +22 to −103. Where both are measured (8200 Å), the star is about 20 km/s bluer than the sky. The fits' χ²/ν of 44–126 at S/N ≈ 90 show that the model is imperfect (one PWV profile, a Gaussian LSF, BT-Settl mismatch), so the errors are Birge-inflated.
- Control: Ross 640 (DZ white dwarf, 2.52″ slit, HDU 2; smooth model) gives Δv_tell from −35 to −88 km/s and a star − sky of −26 ± 14 km/s (−43 ± 15 at 8200 Å). Its own sky gives +18 km/s.
  - The first object of each spec1d (SPAT0168 in J1745's file) gives +17 to +91 km/s.
  - Different objects in the same slit therefore have very different zero points, as off-centre point sources should. The slit-centring term is real and large here (1″ ≈ 190 km/s at R2500I with 2 × 2 binning).

**RVs (stat. z; methods ccf + template_fit, adopted = hierarchical):**

| wavelength_correction | Adopted [km/s] | z | Total error |
|---|---|---|---|
| none | 37.73 ± 5.03 | 2.08 | 45.3 (sys_wavezp 44.4 = unapplied telluric offset) |
| sky | 18.88 ± 3.32 | 1.81 | 9.5 |
| telluric | 58.86 ± 5.52 | 5.46 | 10.5 |
| auto | 29.14 ± 3.29 | 0.73 | 17.1 |

- "telluric" alone is not usable for these lines. Inside 7700–8950 Å, where every alkali line lies, there is only one telluric node (the weak 8200 band), so the curve there is an interpolation between −44, −6 and −40 km/s. This is a coverage failure of the telluric-only correction, not evidence about the star.
- "auto" keeps τ_line = 2.0 and lands at z 0.73. It depends on the constant, though: with only the in-span 8200 Å node (−19.6) it would move to ≈38.5 km/s. The ±14 km/s on the constant (in the total error) is therefore the honest uncertainty of the star's absolute zero point here.
- telluric_ref gives 47.5 ± 7.7 km/s (z 2.65). The Na I (8200) segment alone gives 38.2 ± 9.4; the 9300 Å segments rely on FeH/H2O stellar features that BT-Settl models poorly (56–63 km/s); O2 A gives 30.7 ± 50.5.

**Verdict:** the sky correction is valid for the *shape* of the wavelength solution within its measured span (τ_line collapses whether or not a constant is added). It is not valid for the star's absolute zero point: both stars observed that night sit 10–40 km/s bluer than the sky, and the telluric constant is needed. On J1745 that constant is only weakly measured (±14 km/s), because the strong telluric bands lie outside the alkali-line span. The default stays "none"; "auto" is the best-motivated correction, with an error dominated by the star − sky term.


## Unit 5 review fixes (unit5-fix; review_3cfix_unit5.md)

All choices below were fixed before the J1745 benchmark was run, and none was changed afterwards.

### Blocker 1: the star − sky constant
- `star_minus_sky` now compares only telluric nodes that fall inside a used (measured) sky segment. It never uses a sky value that is interpolated across a gap, held flat beyond the last node or ramped towards an edge node.
- With no such node it returns NaN, logs why and sets the `star_sky_unmeasured` flag.
- The sky's curve-model spread is added to the error, and `n_nodes` is reported (`zp_star_sky_n`).
- `calibration.wave_valid_range` drops sky segments centred outside the arc coverage. For J1745 we set it to the arc lines in `wvcalib.fits` (7437–9926 Å), which removes the red-edge node (−103.5 km/s).
- The constant is always reported. When it is not applied, |zp_star_sky| ⊕ its error enters `rv_total_err` as `sys_star_sky`; in "auto" its error is in `sys_wavezp`.
- J1745: **−19.6 ± 7.3 km/s from one node** (H2O 8200; the four 9000–9900 Å nodes lie in the sky gap). Before the fix it was −10.2 ± 14.2 from six nodes.

### Blocker 2: licence
- `src/rvfitter/data/telluric_hitran.npz` is deleted (a normal commit; history is not rewritten) and ignored.
- `calibration.telluric_build.build` builds the model into the platformdirs cache. It is run with `rvfitter templates telluric --build [--line-cache DIR] [--out PATH] [--force]`, and `rvfitter templates telluric` alone shows the status and provenance.
- The metadata and a JSON sidecar record the source, the fetch date and SHA-256 of each line file, the output SHA-256, the citation and the terms (free of charge; must be cited; not to be redistributed).
- Without the model, `telluric_zeropoint.measure` returns None and `telluric_ref` returns a fit_failed measurement. Both log the build command, and the pipeline sets the `telluric_unavailable` flag. Nothing crashes.
- Tests:
  - `tests/telluric_synth.py`, a synthetic model with the real format and band layout, autoused via `tests/conftest.py`;
  - an offline build with a faked HTTP layer, which checks provenance, line-cache reuse and an identical rebuild;
  - a network + slow test of the real build, with the vacuum (Fraunhofer A/B) checks;
  - a cache test that skips without the model.

### Decision 1 (amended): "sky-if-needed" is the default
The tests and their thresholds are listed in the Handoff.
They use only the sky segments, their errors and model spread, the arc RMS, the resolution and the region positions (selected before correction). They never use the stellar line scatter.
The decision and every test value are logged at INFO, and recorded in `barycentric['sky_tests']`, `wavelength_correction_applied` and `zp_flags`.
For J1745 all three tests pass, so the sky correction is applied.

### Should-fix
1. **O2 A for cool targets.** The catalogue's O2 A row has `teff_min` = 2800 K (its caveat names the K I wings and the injection biases). `telluric_zeropoint.measure(teff=)` and `telluric_ref` skip catalogue bands whose Teff range excludes the prior. Each node's error now includes the SD of Δv over the template and its ±1 Teff-step neighbours (`err_template`; `neighbour_templates`).
2. **"telluric" coverage and error.** The curve is refused (with the `telluric_refused` flag) when a region lies more than 250 Å from every node, or when only O2 nodes exist. Its error is σ_tell(λ) at the measurements' wavelengths. J1745: refused, because Rb I 7802, Rb I 7949 and Cs I 8523 lie 434, 286 and 287 Å from the nearest node.
3. **Within-epoch correlation.** Epochs are now combined in two levels: regions → epoch, then random effects over the epochs' RVs, both per method (`_combine_epochs`) and for the adopted RV (`_adopted_over_epochs`, `tau_epoch`). Two epochs at 10 and 30 km/s now give an error above 5 km/s with 1 or 8 regions per epoch (tested). The variability χ² includes each epoch's sys_wavezp, and the statistical-only p-value is reported too.
4. **Multi-epoch zero point.** The zero-point terms are combined over epochs (mean offsets, RMS errors and systematics), and the per-epoch values are recorded.
5. **io HDU selection.** See the Handoff. J1745 now reads SPAT0240 (S/N 86); this is tested on the data and on synthetic files (name match, highest S/N, science preferred, a warning when the S/N ratio is below two). `sky_source` uses the best-measured object's sky and records its HDU (`sky_zeropoint['source_hdu']`). `_matching_object` reuses the same table reader and logs failures at debug level.
6. **"none" with a telluric source.** The unapplied term is now evaluated at the measurements' wavelengths. It uses the sky curve when one is measured (the star − sky offset is then `sys_star_sky`), else the telluric curve.

### Nits
- The telluric `span_bounds` now use the used segments only.
- `zp_kms` is logged and documented as the node-weighted mean, which is not "the" zero point.
- The previous Handoff table's "none" row (total error 23.4, sys_wavezp 21.7) was computed **without** `telluric_source`, and the Unit 5 table's row (45.3, 44.4) with it. Both are superseded by the table above.
- The PWV note is in the Gotchas.
- `_matching_object`'s per-HDU failures are logged at debug level.

### J1745 telluric segments (unit5-fix; target trace, 2100/5.0 template ± 100 K)
The telluric curve runs from −55.3 to −6.0 km/s over 8236–9779 Å (5 of 6 segments used; O2 A skipped at 2100 K).
Its node-weighted mean is −43.9 ± 5.9 km/s (Birge 5.5), but this is only a summary.
The RVs per mode are in the Handoff table.
"auto" moves the RV to 38.9 km/s with the one-node constant; this is why "auto" stays opt-in.
