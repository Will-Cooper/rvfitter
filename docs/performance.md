# Performance

Benchmark: DENIS J1745−1640 (R2500I, tellcorr coadd), default methods (`template_fit`, `ccf`, `linecentre`), R 2602, local BT-Settl grid, `velocity_frame.correction = "skip"`, `sky_source` and `telluric_source` = the spec1d, `wavelength_correction` `none` and `sky-if-needed`.
Machine: Intel Core Ultra 7 155H (22 logical cores), 15 GB RAM, Linux, Python 3.12.
Wall time per run (one object, both modes shown as `none` / `sky-if-needed`); peak RSS is per process.
Timings are noisy (a few per cent) on a shared machine; the first run of a fresh process is "cold", a repeat in the same process "warm".

| Methods | n_procs | Before: cold [s] | Before: warm [s] | After: cold [s] | After: warm [s] | Peak RSS [MB] |
|---|---|---|---|---|---|---|
| default | 1 | 11.6 / 11.2 | 10.5 / 11.2 | 10.9 / 10.2 | 9.6 / 10.4 | ${\approx}$440 |
| default | 8 | 13.6 / 13.8 | 13.1 / 13.6 | 13.7 / 13.1 | 12.2 / 13.7 | ${\approx}$440 per process |
| `lsd` only | 1 | | | 9.0 / 8.4 | | ${\approx}$430 |
| `lsd` only | 8 | | | 11.3 / 10.8 | | ${\approx}$430 per process |
| `forward_mcmc` only | 1 | | | 228 / 217 | | ${\approx}$430 |
| `forward_mcmc` only | 8 | | | 82 / 80 | | ${\approx}$430 per process |

## What was optimised

The profile at `n_procs = 1` (cumulative): the wavelength-zero-point step (`wavelength_correction.correct`, mostly the telluric fits) 63 per cent, the template systematics 25 per cent, and the template and CCF fits the rest.
`Template.at` recomputed the cumulative flux integral over the whole template on every call (56 000 calls); it is now computed once per template (`functools.cached_property`).
The results are bit-identical (maximum absolute difference 0 in every RV, error and systematic, for both modes and for `n_procs` 1, 4 and 8).
Tried and rejected (no measurable gain): slicing or vectorising `model_sky` (the cost is per-call overhead across 48 000 calls of the Python line loop), and a larger template-file cache (77 reads of 33 files cost 2 s, but the saving was inside the noise).
The remaining time is spread over many small numpy calls, so further speed-ups need algorithmic changes (e.g. vectorising the sky correlation over lags), not caching.

## Choosing `n_procs`

Parallelism is within an object (one task per region, Monte Carlo chunk or template variant), and workers are spawned for each object, each importing the scientific stack and starting with cold template caches (${\approx}$2 s).
For a single object with a handful of windows, as here, `n_procs = 1` is as fast or faster; the wavelength-correction step is serial.
Use more processes for objects with many regions or epochs, for `forward_mcmc` (2.7 times faster at 8) and for large batches; beyond the number of regions or chains there is no gain.
The automatic setting (no `--n-procs`, `parallel.n_procs` unset) follows this: serial, unless `forward_mcmc` is enabled, in which case `cpu_count() − 1`.
Results do not depend on `n_procs` (seeds are per task): `n_procs` 1, 4 and 8 give identical numbers.
Each worker holds its own template cache (${\approx}$440 MB for this grid), so memory scales with `n_procs`.

## Opt-in methods

`lsd` run alone takes ${\approx}$9 s, about the same as the default methods, because the fixed wavelength-correction step dominates; it is cheap.
`forward_mcmc` costs ${\approx}$220 s at `n_procs = 1` and ${\approx}$80 s at 8 for this object (it samples the template parameters and RV jointly); enable it only for objects where the standard methods disagree or a posterior is needed.
