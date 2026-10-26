# Dev notes: results, logging_setup, cli, pipeline (P2)

## Assumptions about other packages (to check at merge)
- `config.Config` attributes read with `getattr` defaults: `seed` (0), `methods` (list of names, or name -> options with `enabled`), `objects` (grouping override), `teff`, `resolution`, `bary`/`barycentric.mode`/`.frame`, `inputs`/`input.path` and `output` (only for `rvfitter run`), `hash()`.
- `pipeline.run` gained optional keywords `medium`, `methods`, `bary` (the CLI overrides) beyond the contract.
- `combine.hierarchical(values, errors, line_ids, method_ids)` is called with numpy arrays and two lists; it must return `median, err, q16, q84, tau_line, tau_method, n_methods, n_lines`.
- `systematics.wavelength_zero_point(spectrum)` and `systematics.template_spread(method_results)` are called with these guessed arguments, both returning floats; failures are logged at DEBUG and the term is left NaN.
- Several epochs of one object: each is measured separately, then `combine.combine_regions` over all measurements per method.
- Lazy CLI dispatch: `rvfitter.inject.main(argv)`, `rvfitter.forecast.main(argv)`, `rvfitter.templates.cli_main(argv)` (argv = `download ...` / `list`), `rvfitter.gui.app.main()`. A missing module or attribute gives "not yet available" and exit 2.
- Logging: `rvfitter` logger has `propagate=False`; third-party loggers are only quietened if they already exist (creating `astropy` early breaks astropy's own logger class).
- `tests/test_package.py::test_cli_placeholder_exits_2` was changed to a usage-error check.
