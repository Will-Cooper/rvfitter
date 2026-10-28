# rvfitter v2 architecture and interface contract

This document fixes the data types and function signatures that the modules of `src/rvfitter/` share.
Each Phase C work package codes against these interfaces, so packages can be developed in parallel.
Changing an interface here requires the orchestrator's agreement.

## Conventions

- **Units:**
  - Wavelengths are stored internally as plain `float64` arrays in **vacuum Ångström**.
  - Velocities are in km/s.
  - Astropy `Quantity` is used only at I/O boundaries.
- **Speed of light:** `rvfitter.constants.C_KMS = 299792.458`. Doppler: v = c·Δλ/λ. Shifting a spectrum by velocity v uses the relativistic factor λ' = λ·√((1+β)/(1−β)).
- **Medium:**
  - Conversions are done only by `rvfitter.medium.air_to_vac(wave, method)` and `vac_to_air(wave, method)`.
  - `method` is one of `"ciddor"` (the default), `"edlen"` or `"morton"`.
- **No hard-coded wavelength windows** anywhere except the feature catalogue (`data/features.ecsv`).
- **Logging:**
  - Every module does `log = logging.getLogger(__name__)`.
  - There are no `print` calls.
  - INFO is for user-facing progress, DEBUG for detail.
- **Randomness:** functions that use random numbers take an `rng: numpy.random.Generator` argument. Seeds are derived per task with `SeedSequence.spawn`. No global random state.
- **Parallelism:** functions that can use several processes take `executor: concurrent.futures.Executor | None`. With `None` they run serially. Results must be identical either way.
- **Matplotlib:**
  - Only `plotting.py` imports pyplot, and it uses the object-oriented API.
  - No rcParams are changed at import. The style is applied with `plotting.style_context()`.

## Core data types (`rvfitter.types`)

```python
@dataclass(frozen=True)
class Spectrum:
    wave: np.ndarray            # vacuum Å, strictly increasing, finite
    flux: np.ndarray            # arbitrary units
    sigma: np.ndarray           # 1σ flux error (np.nan if unknown)
    resolution: float | None    # R = λ/Δλ (FWHM); None = unknown
    meta: Mapping[str, Any]     # header-derived: object, ra, dec, mjd_mid, site, exptime, airmass, medium_in, frame_in, vel_corr ...
    source: str                 # file path or archive member

@dataclass(frozen=True)
class Region:
    name: str                   # e.g. "K I 7701" or "CCF 7600-8000"
    kind: str                   # "line" | "band" | "ccf" | "telluric" | "continuum"
    wmin: float; wmax: float    # vacuum Å
    centre: float | None        # vacuum rest wavelength for lines
    continuum: tuple[tuple[float, float], ...]  # continuum windows, vacuum Å
    reason: str                 # why it was selected (logged and written to the config)

@dataclass
class Measurement:              # one method on one region/order
    method: str                 # "linecentre" | "ccf" | "template_fit" | "forward_mcmc" | "lsd" | "telluric_ref"
    region: str
    rv: float; rv_err: float    # km/s, statistical
    flag: int                   # 0 = ok; bit flags defined in results.FLAGS
    diagnostics: dict           # method-specific (chi2, peak height, template, vsini, samples path ...)

@dataclass
class MethodResult:
    method: str
    rv: float; rv_err: float    # combined over regions (combine.combine_regions)
    n_regions: int
    measurements: list[Measurement]

@dataclass
class ObjectResult:
    object: str
    files: list[str]
    methods: dict[str, MethodResult]
    adopted: dict               # hierarchical: median, err (half 16–84), q16, q84, mean, sd, tau_line, tau_method, n_methods, n_lines
    systematics: dict           # {"template_spread": .., "wavelength_zero_point": .., "total": ..}
    barycentric: dict           # {"applied_kms": .., "source": "header"|"computed"|"skipped", "frame": ..}
    binary: dict                # {"dbic": .., "flag": bool}
    config_hash: str
    version: str
```

## Module interfaces

| Module | Public functions / classes | Package |
|---|---|---|
| `constants`, `medium`, `types` | as above | P0 (orchestrator) |
| `io` | `read_spectrum(path, *, medium="auto", medium_method="ciddor", hdu=None, columns=None, resolution=None) -> Spectrum`; `register_reader(name, sniff, read)`; `find_wvcalib(path)`, `arc_line_range(wvcalib, extname=None)` | P1 |
| `archives` | `collect_inputs(path) -> list[Path]` (file, directory, zip or tar; safe extraction to a temporary directory); `group_by_object(spectra, override=None) -> dict[str, list[Spectrum]]` | P1 |
| `config` | `Config` (dataclass tree), `load(path) -> Config`, `dump(cfg, path)`, `default_for(spectrum) -> Config` (regime defaults by R and band), `Config.hash()` | P1 |
| `results` | `to_table(results: list[ObjectResult]) -> pandas.DataFrame` (one row per object; `rv_<method>`, `rv_<method>_err`, `rv_<method>_flag`, `rv_adopted*`, `rv_mean`, `rv_spread`, barycentric columns); `details_table(...)`; `write(results, path, formats)` (csv, json, tsv, html) | P2 |
| `logging_setup` | `configure(verbosity, logfile) -> None` (console handler to stdout, file handler at DEBUG) | P2 |
| `cli` | `main(argv=None)`; subcommands init, run, measure, inject, forecast, templates, gui, doctor | P2 |
| `pipeline` | `run(cfg, inputs=None, *, n_procs, medium, methods, bary, on_object, cancel) -> list[ObjectResult]` (objects processed sequentially, a process pool within each object); `measure_object(spectra, cfg, executor, rng, *, name, methods, bary, cancel) -> ObjectResult`; `wavelength_correction(raw, spec, cfg, name, regions) -> (Spectrum, dict)`; `suggested_refits(result) -> list[dict]`; `refit_object(result, cfg, cfg_patch, *, inputs, n_procs, medium, methods, bary) -> (ObjectResult, Config)`; `apply_patch(cfg, cfg_patch) -> Config`; `config_for_object(cfg, name)`; `object_seed(seed, name)` | P2 → M |
| `barycentric` | `correction(spectrum, mode="auto", frame="barycentric", overrides=None) -> (kms, source)`; `apply(spectrum, kms) -> Spectrum` | P3 |
| `features` | `load_catalogue() -> Table`; `Feature` records with vacuum λ, species, kind, teff_range, r_min, caveats, ref | P3 |
| `regions` | `select(spectrum, *, teff=None, methods, overrides=None) -> dict[method, list[Region]]` | P3 |
| `gui.app` | `main()`; calls only `config`, `archives`, `pipeline`, `results`, `plotting` and `templates` (through `gui.backend`) | P4 |
| `templates` | `TemplateGrid` (`.get(teff, logg, met, ...) -> Spectrum`, N-D multilinear), `KNOWN_GRIDS`, `download(grid, teff_range, logg_range, met_range, alpha_range, *, ranges, directory, fetch, verify, source="auto") -> TemplateGrid`, `verify_medium(grid)`, `cli_main(argv)`; cache via platformdirs | M |
| `grid_sources` | Zenodo grids (Sonora Bobcat, Diamondback, Elf Owl; SAND): `ZENODO_GRIDS`, `download_zenodo(grid, want, directory, *, axes, write)`, `provenance(grid)`, `record_files(record)`; used by `templates.download` | M |
| `broadening` | `instrumental(spec, R)`, `rotational(spec, vsini, epsilon)`, `doppler_shift(spec, v)` | M |
| `calibration.telluric` | `model()`, `available()`, `cache_path()`, `transmission(wave, airmass=1., pwv=, *, resolution=None, dv=0.) -> ndarray`, `mask(wave, threshold=0.05, *, airmass, pwv, resolution) -> bool ndarray`; the HITRAN-derived model is built into the user cache (`templates telluric --build`), never shipped; raises `TelluricModelUnavailable` without it | M |
| `calibration.sky_zeropoint` | `measure(spectrum, *, sky=None, seg_width=250., ..., valid_range=None) -> SkyZeroPoint \| None` (OH sky lines, local Δv(λ)); `apply(spectrum, zp=None, *, constant_kms=0., constant_err=nan)`; `applied_dv(meta, wave)`, `curve_at(curve, wave)` | M |
| `calibration.telluric_zeropoint` | `measure(spectrum, regions=None, *, resolution, airmass, pwv, seg_width, ..., template=None, teff=None, spread_templates=()) -> TelluricZeroPoint \| None` (unpacks as `(zp, err)`; `.dv(λ)`); `star_minus_sky(tell, sky) -> dict` | M |
| `methods.telluric_ref` | `measure(spectrum, regions, cfg, *, executor, rng)`: v_star − v_tell per telluric segment; `prior_template`, `neighbour_templates` | M |
| `methods.*` | each method: `measure(spectrum, regions, cfg, *, executor, rng) -> MethodResult` | M |
| `combine` | `combine_regions(measurements) -> (rv, err)` (Birge), `hierarchical(values, errors, line_ids, method_ids, ...) -> dict` (ported from legacy, Woodbury) | M |
| `systematics` | `template_spread(spectrum, regions, cfg, *, executor, rng)`, `template_spread_details(...)`, `wavelength_zero_point(spectrum, cfg, *, wave=None)` | M |
| `plotting` | `style_context()`, `plot_line_fits`, `plot_ccf`, `plot_adopted`, `plot_corner` | M (+P4 embeds) |

**Packages:**
- P1–P4 are the Sonnet plumbing packages, each in its own worktree.
- M is the Opus methods package.
- P0 (`constants`, `medium`, `types`) is written by the orchestrator first, so that everyone shares it.

## Testing conventions

- Unit tests live in `tests/test_<module>.py`.
- Tests that need the local test data use `@pytest.mark.data` and the `RVFITTER_TESTDATA` environment variable, and are skipped if it is unset.
- GUI tests use `@pytest.mark.gui` and pytest-qt with `QT_QPA_PLATFORM=offscreen`.
- Synthetic spectra come from `tests/synth.py`: `make_spectrum(rv, R, snr, vsini, wave_range, medium, rng)`, which builds a toy line forest. It is owned by P0.

## Interface addendum (agreed after P2, binding for P1, P2 and M)

**Config attribute names** (P1 provides; P2 and P4 read):
- `cfg.seed: int`
- `cfg.input`: `.paths: list[str]`, `.medium` ("auto" | "air" | "vacuum"), `.medium_method`, `.hdu`, `.resolution: float | None`
- `cfg.grouping.override: dict[str, str] | None` (file → object)
- `cfg.velocity_frame`: `.correction` ("auto" | "apply" | "skip"), `.frame`, `.overrides: dict`
- `cfg.templates`: `.grid`, `.local_dir`, `.medium`, `.teff_prior: float | None`, `.logg_prior`, `.met_prior`, `.teff_range`, `.logg_range`, `.met_range`
- `cfg.methods`: `.enabled: list[str]`, `.options: dict[str, dict]`
- `cfg.regions`: `.overrides: dict[str, list[dict]]`, `.min_snr`, `.telluric_strictness`
- `cfg.combine`: `.tau_scale: float | None`, `.systematic_floor: float`
- `cfg.parallel.n_procs: int`
- `cfg.output`: `.path`, `.formats: list[str]`, `.plots: bool`, `.medium`
- `cfg.hash() -> str`

**Functions:**
- `combine.hierarchical(values, errors, line_ids, method_ids, *, tau_scale=None, ngrid=200) -> dict`, with keys `median`, `err`, `q16`, `q84`, `mean`, `sd`, `tau_line`, `tau_method`, `n_methods`, `n_lines`.
- `systematics.wavelength_zero_point(spectrum, cfg) -> float`: km/s, NaN if unknown.
- `systematics.template_spread(spectrum, regions, cfg, *, executor=None, rng) -> float`: km/s.
- `pipeline.run(cfg, inputs, *, n_procs, medium=None, methods=None, bary=None)`. The keywords override the config (CLI use).

**CLI entry points:** `rvfitter.inject.main(argv)` and `rvfitter.forecast.main(argv)` (new modules, owned by M), `rvfitter.templates.cli_main(argv)`, `rvfitter.gui.app.main()`.

### Addendum 2 (after P2 review and P4)
- **Region overrides:** `cfg.regions.overrides: dict[str, list[dict]]`, keyed by **method name**. Each entry is one of:
  - `{"name", "kind", "wmin", "wmax"[, "centre", "continuum"]}`, which adds or replaces the region of that name;
  - `{"name", "remove": true}`, which drops it.

  `regions.select(..., overrides=cfg.regions.overrides)` applies them. The GUI region editor writes this structure and never re-implements the selection.
- **pipeline.run** gains `on_object: Callable[[ObjectResult], None] | None = None` (called after each object) and `cancel: threading.Event | None = None` (checked between objects and between methods; on cancel, return the results so far). Pass `cfg.regions.overrides` to `regions.select` and `cfg.combine.tau_scale` to `combine.hierarchical`. Add `cfg.combine.systematic_floor` in quadrature to `rv_total_err`.
- **Plotting:** `plotting.plot_line_fits`, `plot_ccf`, `plot_corner` and `plot_adopted` have the signature `fn(result: ObjectResult) -> matplotlib.figure.Figure`. Methods must therefore store in `Measurement.diagnostics` what their plots need (arrays as lists or ndarrays).
- **Output:** the GUI uses `cfg.output.path` and `cfg.output.formats`, as named in the first addendum.

### Addendum 3 (current signatures; refits, zero points and templates)
- **Keywords**
  - `systematics.wavelength_zero_point(spectrum, cfg, *, wave=None)`: `wave` is the array of wavelengths at which the RVs were measured. The zero-point term is the RMS of the curve's Δv over them (the arc-RMS term alone if `wave` is None).
  - `calibration.telluric_zeropoint.measure(..., teff=None, spread_templates=())`: `teff` selects the catalogue's telluric nodes by `teff_min`/`teff_max` (O2 A is not a node below 2800 K); `spread_templates` are alternative stellar templates, and the spread of the node velocities between them is added to each node's error.
  - `io.read_spectrum(..., medium_method=)` selects the air to vacuum formula (`cfg.input.medium_method`).
- **PypeIt input**
  - Pixels masked in `OPT_MASK` (`BOX_MASK` for box extraction) are dropped.
  - `io.find_wvcalib` looks for `wvcalib*.fits` or `WaveCalib*.fits` beside the file, in `Calibrations/`, and up to three parents.
  - `Spectrum.meta['arc_range']` is the span of the `wave_fit` arc lines in vacuum Å (converted like the spectrum if it was read as air), with `meta['arc_file']`.
  - `calibration.wave_valid_range`, when unset, takes this value (INFO-logged); a configured value always wins.
- **Calibration modules**
  - `sky_zeropoint` measures Δv(λ) per segment of telluric-free OH lines; `valid_range` drops segments centred outside the arc range.
  - `telluric` and `telluric_zeropoint` give the star's own Δv(λ) from the telluric bands. `star_minus_sky` gives the slit-centring constant `zp_star_sky`.
  - `pipeline.wavelength_correction` chooses what to apply from `cfg.calibration.wavelength_correction` (`sky-if-needed` | `sky` | `auto` | `telluric` | `none`) and returns the zero-point record stored in `ObjectResult.barycentric` (`wavelength_correction*`, `zp_flags`; see `pipeline.ZP_FLAGS`).
  - `methods.telluric_ref` uses the telluric curve as a velocity reference: one RV per telluric segment.
- **`grid_sources`:** each `ZENODO_GRIDS[name]` is a `ZenodoGrid` (record ids, file selector, name parser, reader, DOI, citation). `download_zenodo` streams the files by HTTP range or tar.gz, checks MD5/CRC, writes through a `.part` file and rolls back on error. `templates.download(source="auto")` tries SVO, then Zenodo.
- **Refits**
  - `pipeline.suggested_refits(result) -> list[dict]` reads the edge guidance (`template_summary['edges']`, behind the flags `template_param_edge`, `template_grid_edge` and `template_grid_hole`) of the measurements. Each dict has `method`, `axis`, `direction`, `action` (`widen` | `download` | `none`), `message`, `cfg_patch`, `download` (keyword arguments for `templates.download`, or None) and `command` (the CLI equivalent).
  - `pipeline.apply_patch(cfg, cfg_patch) -> Config` deep-merges a patch into a copy of `cfg` and validates it.
  - `pipeline.refit_object(result, cfg, cfg_patch, *, inputs=None, n_procs=1, ...) -> (ObjectResult, Config)` re-measures one object. The returned Config has `{"object", "patch"}` appended to `refits` (written as `[[refits]]` by `config.dump`), and `run` applies it to that object only (`config_for_object`). The seed is `object_seed(seed, name)`, so the written TOML reproduces the refit. The new result has `refitted = True`.
  - The GUI ("Download grid & refit", through `gui.backend.Backend.suggested_refits` and `.refit`) runs `templates.download(**s["download"])` when set, then `refit_object`, in a worker thread. It replaces the object's row and keeps the new Config as the session config, so "Save analysis" writes the `[[refits]]`.
