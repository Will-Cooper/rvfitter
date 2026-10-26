# Dev notes: results, logging_setup, cli, pipeline (P2)

- Config attributes, combine/systematics signatures and CLI entry points follow the interface addendum in docs/architecture.md.
- `pipeline.run` keywords n_procs, medium, methods, bary override the config; CLI options default to None so the config wins.
- Several epochs of one object: each is measured separately, then `combine.combine_regions` over all measurements per method.
- Logging: `rvfitter` logger has propagate=False; third-party loggers are only quietened if they already exist (creating `astropy` early breaks astropy's logger class).
- `tests/test_package.py` placeholder test was changed to a usage-error check.
