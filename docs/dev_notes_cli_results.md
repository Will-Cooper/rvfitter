# Dev notes: results, logging_setup, cli, pipeline (P2)

- Config attributes, combine/systematics signatures and CLI entry points follow the interface addendum in docs/architecture.md.
- `pipeline.run` keywords n_procs, medium, methods, bary override the config; CLI options default to None so the config wins.
- Several epochs of one object: each is measured separately, then `combine.combine_regions` over all measurements per method.
- Logging: `rvfitter` logger has propagate=False; third-party loggers are only quietened if they already exist (creating `astropy` early breaks astropy's logger class).
- `input.telluric_corrected` (CLI `--telluric-corrected`) relaxes telluric masking in `regions.select` for the O2 bands of line regions only (strictness x 0.25). H2O and every other band stay masked, for line regions and for ccf/fit spans: residuals of a correction are not noise (on J1745−1640 the unmasked H2O region pulled the template RVs to ≈ −45 km/s; see dev_notes_methods.md). `--no-plots` sets `output.plots = false`.
- `results.FLAGS`: fit_failed 1, few_regions 2, error_floor 4, template_grid_edge 8, telluric_contaminated 16, not_converged 32, poor_fit 64 (χ²/ν > 3; error scaled), outlier 128 (gross outlier among a method's regions). `results.EXCLUDE_FLAGS` (fit_failed, few_regions, not_converged, outlier) excludes a measurement from its method's combination and from the adopted RV.
- The adopted RV is `combine.hierarchical_from_methods` over every method's per-region measurements, leaving out non-finite and `EXCLUDE_FLAGS` measurements; each exclusion is logged at INFO and listed in `adopted['excluded']`.
- `pipeline.run` raises `NoMethodsError` ("no RV methods available: ...") if no enabled method is registered; the CLI logs it once and exits 1.
- archives: the no-filter tar fallback suppresses the Python 3.12 DeprecationWarning (members are validated manually).
- `tests/test_package.py` placeholder test was changed to a usage-error check.
