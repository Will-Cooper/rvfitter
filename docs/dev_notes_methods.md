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
- Flag bits (to be mirrored by `results.FLAGS`): 1 edge, 2 outside grid, 4 fit failed, 8 few pixels, 16 poor fit (χ²/ν > 3). Bits 1, 4 and 8 exclude a measurement from the method's Birge combination.
- `methods.REGISTRY[name]` is a lazy, picklable callable with the `measure` signature.

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
| CCF, R 2500, SNR 30 | 40 | see test (passes) | 0.8–1.2 |
| Template fit, R 2500, SNR 30 | 60 | see test (passes) | 0.8–1.2 |

A trial run with 30 realisations gave a CCF bias of 0.13σ and pull width 1.04, and a template-fit bias of 0.09σ and pull width 1.03. The slow tests (R 2500/6000/20 000, SNR 15–40, vsini 0–30 km/s, N = 120; the R 20 000 case over 8000–8500 Å to keep it to a minute) are marked `slow`, and all six pass.

## J1745−1640 benchmark

Benchmark 26.2 ± 2.3 km/s (Burgasser et al. 2015); z = |RV − 26.2| / √(2.3² + σ²).
Input: `Science/DENISJ174534.6-164053_tellcorr.fits` (PypeIt vacuum, already heliocentric, so no barycentric step).
Regions (by hand until the regions package merges, fixed before any fit): W1 7400–8600 Å and W2 8600–10 000 Å, with the O2 A band (7590–7710 Å) and the strongest H2O band (9270–9680 Å) masked, i.e. the plan's R ≲ 3000 optical default.
Templates: the local BT-Settl-CIFIST grid, teff prior 2100 K ± 300 K (21 candidates), chosen by χ².
Seed 42; the CCF uses 200 Monte Carlo trials.

| Method | R | RV [km/s] | z | Per window (W1, W2) | Template | Notes |
|---|---|---|---|---|---|---|
| CCF | 2500 (nominal) | 35.57 ± 12.09 | 0.76 | 44.10 ± 2.98, 18.43 ± 4.22 | 2100/5.5 | Birge-inflated; χ²/ν flag on both windows |
| Template fit | 2500 | 26.05 ± 9.49 | 0.02 | 33.85 ± 5.46, 14.49 ± 6.65 | 1800/5.0 | χ²/ν = 58 and 132 |
| CCF | 1938 (measured) | 34.52 ± 11.50 | 0.71 | 43.18 ± 3.16, 19.24 ± 4.19 | 2100/5.5 | |
| Template fit | 1938 | 23.95 ± 8.54 | 0.25 | 31.34 ± 6.45, 14.08 ± 7.45 | 1800/5.0 | |

The resolution measured by profiling the template-fit χ² is R = 1940 ± 160, lower than the nominal 2500. This may be partly real (seeing-limited slit width) and partly template mismatch in the line depths, so we keep the nominal R as the default and give the measured R as a sensitivity test; it moves each method by ≤2.1 km/s (1.0 km/s for the CCF).
Both methods agree with the benchmark (z < 1), but the errors (≈10 km/s) are dominated by a real difference between the two windows: both methods find W1 high (34–43 km/s) and W2 low (14–22 km/s). This probably reflects BT-Settl mismatch, which differs between the TiO/VO/CrH region and the FeH/H2O region, plus residuals of the telluric correction near the H2O bands. It is the same line-to-line scatter that the hierarchical combination found in Phase A (τ_line ≈ 13 km/s), and it should shrink with the catalogue's narrower, feature-based windows from the regions package.
The two methods choose different templates (2100/5.5 vs 1800/5.0), because the CCF's χ² is computed after the high-pass whereas the template fit's χ² includes the continuum polynomial.
