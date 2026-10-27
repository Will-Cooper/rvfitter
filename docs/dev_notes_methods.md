# Developer notes: methods package (M)

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
- The remaining offset of the atomic lines from 26.2 km/s (≈10 km/s, i.e. ≈0.3 Å at 8500 Å) would be explained by a wavelength zero-point offset in the GTC spectrum, even though the arc RMS is small (0.055 pixel). Unit 5's telluric zero point (O2 A and H2O in the uncorrected `sciencebk` coadd) measures this directly and is the next test.
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
