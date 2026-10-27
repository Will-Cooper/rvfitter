# Dev notes: results, logging_setup, cli, pipeline (P2)

- Config attributes, combine/systematics signatures and CLI entry points follow the interface addendum in docs/architecture.md.
- `pipeline.run` keywords n_procs, medium, methods, bary override the config; CLI options default to None so the config wins.
- Several epochs of one object: each is measured separately, then `combine.combine_regions` over all measurements per method.
- Logging: `rvfitter` logger has propagate=False; third-party loggers are only quietened if they already exist (creating `astropy` early breaks astropy's logger class).
- `input.telluric_corrected` (CLI `--telluric-corrected`) relaxes telluric masking in `regions.select`: line regions use strictness x 0.25, and ccf/fit regions mask only opaque bands (effective depth >= 0.9). `--no-plots` sets `output.plots = false`.
- `pipeline.run` raises `NoMethodsError` ("no RV methods available: ...") if no enabled method is registered; the CLI logs it once and exits 1.
- archives: the no-filter tar fallback suppresses the Python 3.12 DeprecationWarning (members are validated manually).
- `tests/test_package.py` placeholder test was changed to a usage-error check.
