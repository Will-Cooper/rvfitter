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

## J1745−1640 benchmark

Benchmark 26.2 ± 2.3 km/s (Burgasser et al. 2015); z = |RV − 26.2| / √(2.3² + σ²).

| Method | RV [km/s] | z | Notes |
|---|---|---|---|
