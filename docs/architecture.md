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
| `io` | `read_spectrum(path, *, medium="auto", hdu=None, columns=None, resolution=None) -> Spectrum`; `register_reader(name, sniff, read)` | P1 |
| `archives` | `collect_inputs(path) -> list[Path]` (file, directory, zip or tar; safe extraction to a temporary directory); `group_by_object(spectra, override=None) -> dict[str, list[Spectrum]]` | P1 |
| `config` | `Config` (dataclass tree), `load(path) -> Config`, `dump(cfg, path)`, `default_for(spectrum) -> Config` (regime defaults by R and band), `Config.hash()` | P1 |
| `results` | `to_table(results: list[ObjectResult]) -> pandas.DataFrame` (one row per object; `rv_<method>`, `rv_<method>_err`, `rv_<method>_flag`, `rv_adopted*`, `rv_mean`, `rv_spread`, barycentric columns); `details_table(...)`; `write(results, path, formats)` (csv, json, tsv, html) | P2 |
| `logging_setup` | `configure(verbosity, logfile) -> None` (console handler to stdout, file handler at DEBUG) | P2 |
| `cli` | `main(argv=None)`; subcommands init, run, measure, inject, forecast, templates, gui, doctor | P2 |
| `pipeline` | `run(cfg, inputs, *, n_procs) -> list[ObjectResult]` (objects processed sequentially, a process pool within each object); `measure_object(spectra, cfg, executor, rng) -> ObjectResult` | P2 skeleton → M |
| `barycentric` | `correction(spectrum, mode="auto", frame="barycentric", overrides=None) -> (kms, source)`; `apply(spectrum, kms) -> Spectrum` | P3 |
| `features` | `load_catalogue() -> Table`; `Feature` records with vacuum λ, species, kind, teff_range, r_min, caveats, ref | P3 |
| `regions` | `select(spectrum, *, teff=None, methods, overrides=None) -> dict[method, list[Region]]` | P3 |
| `gui.app` | `main()`; calls only `config`, `archives`, `pipeline`, `results` and `plotting` | P4 |
| `templates` | `TemplateGrid` (`.get(teff, logg, met) -> Spectrum`, interpolated), `download(grid, ranges)`, `verify_medium(grid)`, cache via platformdirs | M |
| `broadening` | `instrumental(spec, R)`, `rotational(spec, vsini, epsilon)`, `doppler_shift(spec, v)` | M |
| `calibration.telluric` | `transmission(wave, airmass, pwv)`, `mask(wave, threshold)` | M |
| `calibration.telluric_zeropoint` | `measure(spectrum, regions) -> (dv, err)` | M |
| `methods.*` | each method: `measure(spectrum, regions, cfg, *, executor, rng) -> MethodResult` | M |
| `combine` | `combine_regions(measurements) -> (rv, err)` (Birge), `hierarchical(values, errors, line_ids, method_ids, ...) -> dict` (ported from legacy, Woodbury) | M |
| `systematics` | `template_spread(...)`, `wavelength_zero_point(spectrum)` | M |
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
