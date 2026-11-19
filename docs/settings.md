# Settings guide

Here, we explain every setting that rvfitter shows, in the GUI's **Advanced settings…** and in `rvfitter.toml`, and why the defaults and our recommendations are what they are.
Each entry gives what the setting does, its units and default, when to change it, and the reason.
The settings are grouped as in the TOML file (`[input]`, `[templates]`, …); the heading of each entry is its dotted path, which is also the TOML key.
[Which settings for my data?](#which-settings-for-my-data) gives worked recipes per data type, and [Choosing a model grid](#choosing-a-model-grid) compares the template grids.
The measurements, the benchmark and the zero-point analysis behind many of the recommendations are in [methods.md](methods.md); we link to it rather than repeat it.

We note three things that apply throughout.
rvfitter works internally in vacuum wavelengths, in km/s, and in the frame chosen in [`velocity_frame.frame`](#velocity_frame-frame).
An empty value in the GUI is an unset value in the TOML file (TOML has no null), and it means "automatic", as described in each entry.
When a spectrum is opened, the GUI's Setup tab fills the resolving power, medium, methods, velocity frame and wavelength correction from the data, so most users need only check those.

<a id="two-settings"></a>
> **Two settings that move RVs by tens of km/s.** Check both for every new data set; the GUI's Setup tab says what each will do to the selected spectrum, and Measure RVs never runs on a guessed frame.
>
> - **Velocity frame** ([`velocity_frame.correction`](#velocity_frame-correction); default `auto`). If a header keyword says the spectrum is already corrected (`VEL_TYPE`, `SPECSYS`, `VHELIO`, JWST products), nothing is applied; else rvfitter computes the barycentric correction from RA/Dec, mid-exposure time and site; else the frame is unknown, and the GUI asks you whether the spectrum is already corrected (the command line warns and flags `frame_unknown`). A wrong answer shifts every RV by up to ±30 km/s: the Earth's orbital speed, 29.8 km/s, times the cosine of the target's ecliptic latitude. Check: the Setup tab's status line, or `bary_source` and `bary_kms` in the results; if your reduction corrected the wavelengths without saying so in the header, set `skip`.
> - **Wavelength zero point** ([`calibration.wavelength_correction`](#calibration-wavelength_correction); default `none`). The sky (OH) zero point is measured whenever a sky spectrum is present, but by default not applied, because on the GTC sample of 20 objects with literature RVs the sky correction gave no significant improvement ([methods.md](methods.md#sky-correction-on-the-gtc-sample)). `sky-if-needed` applies it where the arc solution is distorted; that is typically tens of km/s (on the GTC benchmark J1745−1640 it moved the RV by −21 km/s). Check: `zp_sky_kms` (the measured sky offset) and, after an opt-in run, `wavelength_shift_kms` (how much the correction moved the RV), also shown in the Diagnostics tab; compare a run with and without it.

## Contents

- [General](#general): [`seed`](#seed)
- [Input](#input): paths, medium, resolution, telluric correction, raw headers, multi-object files, [IFU cubes](#input-cube)
- [Grouping](#grouping)
- [Velocity frame](#velocity_frame)
- [Templates](#templates)
- [Methods](#methods)
- [Regions](#regions)
- [Combine](#combine)
- [Calibration](#calibration)
- [Parallel](#parallel)
- [Output](#output)
- [Which settings for my data?](#which-settings-for-my-data)
- [Choosing a model grid](#choosing-a-model-grid)

<a id="general"></a>
## General

<a id="seed"></a>
### `seed`

The random seed for the Monte Carlo errors, the bootstrap and the samplers.
Integer ≥ 0; default 0.
The same seed gives the same results for any number of processes, because each object and method draws from its own stream derived from the seed.
We recommend leaving it fixed, so that a result can be reproduced from its saved `results_config.toml`.
Change it only to check that a result does not depend on the random draws: a change of more than a small fraction of the quoted error would mean that the number of Monte Carlo trials is too small for that spectrum.

<a id="input"></a>
## Input

Which spectra are read and how their wavelengths, resolution and contents are interpreted.

<a id="input-paths"></a>
### `input.paths`

The spectra (files, folders or globs) that `rvfitter run` reads.
List of strings; default empty.
Relative paths are relative to the folder of the TOML file.
It is not used in the GUI, where spectra are opened or dropped instead.

<a id="input-medium"></a>
### `input.medium`

The medium of the wavelengths in the files: `auto`, `air` or `vacuum`; default `auto`, which reads each header.
Setting `air` or `vacuum` forces it, and a conflict with the header is logged.
Change it when a header has no medium keyword or states the wrong one.
This matters because the air–vacuum difference is (n − 1)c ≈ 82 km/s in the optical and near-infrared: a spectrum read in the wrong medium is shifted by about that much against the vacuum laboratory wavelengths and templates.
JWST products are always read as vacuum.

<a id="input-medium_method"></a>
### `input.medium_method`

The air-to-vacuum formula: `ciddor` (Ciddor 1996, standard dry air), `edlen` (Edlén 1953) or `morton` (Morton 2000); default `ciddor`.
It is used only for spectra in air, and wavelengths below 2000 Å are always taken as vacuum (the IAU convention).
We recommend the default.
The three formulae differ by less than 0.01 km/s between 7000 and 22 000 Å, far below any RV error rvfitter can reach, so the choice only matters for matching another pipeline exactly.

<a id="input-hdu"></a>
### `input.hdu`

The FITS extension to read, as a number or a name; default unset, which chooses it automatically (for a `PypeIt` spec1d, the object named after the target, else the highest-S/N science object, with a WARNING if the runner-up is more than half as bright).
Set it when the automatic choice picks the wrong trace.
It cannot be combined with [`input.objects`](#input-objects), because both choose the spectra.

<a id="input-resolution"></a>
### `input.resolution`

The resolving power R = λ/Δλ applied to every spectrum, overriding the header.
Dimensionless, > 1; default unset, which uses each spectrum's own R: its header, else an estimate from the pixel spacing that assumes Nyquist sampling (two pixels per resolution element), with a WARNING.
We recommend setting it whenever the header lacks R, because the estimate is only as good as that sampling assumption, and R sets almost everything downstream:
- the regime, and so the default methods ([below](#methods-enabled)): low below 4500, mid from 4500 to 20 000, high above 20 000;
- the line-spread function that the templates are broadened to, so a wrong R leaves a template too narrow or too broad, which biases the template fit and, when it is free, vsini;
- the widths of the region and continuum windows, in resolution elements.

Below R 500 no RV is constrained, because one resolution element, c/R, exceeds 600 km/s; even a centroid good to a twentieth of it (30 km/s at high S/N) is as large as the RVs of nearby stars and brown dwarfs.
The run then warns, and any value measured is flagged `resolution_too_low`.
In practice, `linecentre`, `ccf`, `template_fit`, `forward_mcmc` and `lsd` already refuse below R 1000, the smallest `r_min` of the catalogue features they measure (an ERROR naming the method, recorded as that method's failure; the other objects of a batch still run), because at R 1000 one resolution element, 300 km/s, is wider than the alkali line cores.
When R is unknown, the low-resolution defaults are used, with a WARNING.

<a id="input-telluric_corrected"></a>
### `input.telluric_corrected`

Whether the spectra are already telluric-corrected.
True or false; default false.
When true, the telluric masking of the regions is relaxed, since the bands no longer absorb.
Set it for telluric-corrected coadds and for all space data (JWST), which has no telluric absorption.
We note that the star's own zero point from its telluric lines then needs the uncorrected file, through [`calibration.telluric_source`](#calibration-telluric_source), and that `telluric_ref` cannot run on corrected spectra.
We do not relax the masking automatically, because a correction leaves residuals whose size the file does not record.

<a id="input-raw_header"></a>
### `input.raw_header`

The raw frame headers (a path or glob), for the slit position angle that the reduction drops; default unset.
The angle gives the differential atmospheric refraction across the slit, which moves the star's light along the dispersion relative to the sky (see [methods.md](methods.md#differential-atmospheric-refraction-across-the-slit)).
Without it the angle is fitted with a wide prior (flag `dar_pa_unknown`), which is honest but much less precise.
It matters most at large zenith distance and for a slit far from the parallactic angle: for J1745−1640 the effect is below 1.5 km/s, but it would reach 25 km/s for a slit perpendicular to the parallactic angle.

<a id="input-raw_header_dir"></a>
### `input.raw_header_dir`

A folder searched for the raw frames named in the reduced files (`PypeIt` FILENAME); default unset.
It serves the same purpose as [`input.raw_header`](#input-raw_header), and also supplies the slit width, plate scale and binning when the reduced file lacks them.
A coadd without its spec1d files is matched to its raw frames by `OBJECT`.

<a id="input-objects"></a>
### `input.objects`

Which spectra of a multi-object file (`PypeIt` spec1d, JWST x1d, multi-extension FITS) to measure, each as its own object.
One of `auto`, `target`, `all`, or a list of entry names or HDU numbers (file-qualified selectors such as `file.fits[3]` are allowed); default `auto`.
`auto` takes the entry named after the header target if exactly one is; otherwise every usable entry when the entries carry their own source names (a MOS file); otherwise the target.
`all` and `auto` skip entries with a median S/N per pixel below 3, with a WARNING, because at that S/N even the strongest bands are barely above the noise.
We recommend `auto`; use a list to measure a subset, or `all` to measure every trace of a long slit.

<a id="input-cube"></a>
### IFU cubes (`input.cube`)

A spectrum is extracted from an IFU cube by summing a circular aperture on the median-collapsed image.
Spaxels valid in fewer than half of the wavelength planes are ignored when collapsing, so that edge spaxels with a few valid planes cannot become the brightest.

<a id="input-cube-position"></a>
### `input.cube.position`

The aperture centre (x, y), in 0-based pixels of the collapsed image; default unset, the brightest spaxel.
Set it when the target is not the brightest source in the field, e.g. a faint companion next to its host, or click on the collapsed image (Setup > IFU extraction > Choose…).

<a id="input-cube-radius"></a>
### `input.cube.radius`

The aperture radius, in spaxels; default unset, which means 3 spaxels.
Three spaxels (0.3 arcsec on the NIRSpec IFU) hold 95 per cent of the flux of the JWST pipeline's point-source extraction of VHS 1256−1257 b ([examples/README.md](../examples/README.md#jwst-format-tests-ifu-cube-and-multi-object-spectra)).
A larger radius gains little flux and adds noise and, near a host star, its light; a smaller one loses signal.
We recommend the default for point sources on NIRSpec, and scaling it with the point-spread function in spaxels for other instruments.

<a id="grouping"></a>
## Grouping

<a id="grouping-override"></a>
### `grouping.override`

Which object each spectrum belongs to, when the header object names or file names group them wrongly.
A table from file path, name or stem, or header object name, to object name, e.g. `{"spec_0012.fits" = "J1745-1640"}`; default unset (automatic).
Spectra of one object are measured together, and several epochs are combined with a variability test, so a wrong grouping either merges two objects or splits one object's epochs.
Spectra named by detector position rather than by source (e.g. `SPAT0240` traces) are never merged across files, because the same position in two files need not be the same source.

<a id="velocity_frame"></a>
## Velocity frame

<a id="velocity_frame-correction"></a>
### `velocity_frame.correction`

The barycentric correction: `auto`, `apply` or `skip`; default `auto`.
`auto` skips a spectrum that is already corrected (e.g. `PypeIt` `VEL_TYPE`, FITS `SPECSYS`, IRAF `VHELIO`, JWST products), else computes the correction from the header coordinates, mid-exposure time and site, else uses an ESO BERV keyword with a WARNING; otherwise the frame is unknown, and rvfitter never guesses: the GUI asks whether the spectrum is already corrected and records the answer (`skip` with its frame, or `apply`), whilst the command line applies nothing, logs a WARNING with the bound for that target and sets the flag `frame_unknown` (`zp_flags`; `bary_source = "unknown"`).
The bound is the Earth's orbital speed, 29.8 km/s, times the cosine of the target's ecliptic latitude (±30 km/s without coordinates).
`apply` always computes it, and `skip` never corrects.
We recommend `auto`.
Set `skip` for spectra that a pipeline has corrected without recording it, because applying the correction twice shifts the RV by up to 30 km/s; we note that no Spextool velocity keyword could be verified, so SpeX spectra are treated as uncorrected.

<a id="velocity_frame-frame"></a>
### `velocity_frame.frame`

The frame of the reported RVs: `barycentric` or `heliocentric`; default `barycentric`.
We recommend the barycentric frame because the Solar System barycentre is the inertial reference, and it is the frame of Gaia and most modern RVs.
The two frames differ by at most about 0.013 km/s (0.007 km/s for J1745−1640), so the choice is only bookkeeping, e.g. to match a pipeline that already corrected to the heliocentre.

<a id="velocity_frame-overrides"></a>
### `velocity_frame.overrides`

Values used instead of the header: `ra`, `dec` [deg], `mjd` (mid-exposure), `site` (an astropy site name) or `site_lat` and `site_lon` [deg] together; default empty.
Use it when the header is wrong or incomplete.
The overrides enter only a computed correction (`auto` or `apply`); with `correction = "skip"` they have no effect.
For example, GTC/OSIRIS headers record `LON-OBS` = +17.877°, whereas GTC lies at 17.89° west, which shifts the correction by 0.155 km/s ([methods.md](methods.md#barycentric-correction-of-the-gtc-spectra)); `{site = "lapalma"}` avoids it.

<a id="templates"></a>
## Templates

The model-atmosphere grid used by `template_fit`, `ccf`, `lsd`, `forward_mcmc` and, for the stellar model, by the telluric fits, and the part of its parameter space that is searched.
`linecentre` uses no templates unless its `shape_correction` option is on.

<a id="templates-grid"></a>
### `templates.grid`

The grid name, or a list of names; default `bt-settl-cifist`.
Templates are downloaded and cached on first use.
With several grids, all are searched together and their spread enters the error budget as a systematic.
We recommend BT-Settl-CIFIST for late-M and L dwarfs in the optical, because it covers 1200–7000 K at solar metallicity, its medium has been verified against the NIST alkali lines, and it is the grid of the validation on 56 GTC/OSIRIS spectra ([methods.md](methods.md#validation-on-published-spectra)).
For cooler, metal-poor or near-infrared targets, see [Choosing a model grid](#choosing-a-model-grid).
Listing two physically different grids is the most direct way to see how much the RV depends on the model physics.

<a id="templates-local_dir"></a>
### `templates.local_dir`

A folder with your own template grid, or a table `{grid = folder}` for several grids; default unset (the download cache).
A folder needs either an `rvfitter_grid.toml` metadata file or a known grid's file names.
Use it for a grid that is already converted (e.g. a cache folder copied from another machine), for a grid that cannot be downloaded (e.g. LOWZ) or for your own models; it is read as it is.
For the original files as distributed (e.g. `spectra.zip`), see the next two sections.

<a id="templates-download"></a>
### Getting a grid: download

A grid that is not in the cache is downloaded when you press Measure RVs ("Download and measure"), or with `rvfitter templates download --grid G --teff MIN MAX --logg MIN MAX`.
The BT-Settl grids come from SVO, one spectrum per request, and the Sonora grids and SAND from their Zenodo records; the files are checked, converted and kept in the cache (`RVFITTER_CACHE`), with a `manifest.json` of their source and checksums.

<a id="templates-from-files"></a>
### Getting a grid: original files you already have

If you already have a grid's original files, as distributed on Zenodo, rvfitter converts them instead of downloading them: choose "Use files I already have…" at Measure RVs and pick the folder, or run `rvfitter templates download --grid G --from DIR` with the same ranges.
For the BT-Settl grids, `--from DIR` takes two-column spectra (wavelength [Å], flux) under the names that SVO and Lyon use (e.g. `lte020.0-5.0-0.0a+0.0.BT-Settl.7.dat.txt`), or a folder in rvfitter's own format (another cache, with its `rvfitter_grid.toml`), and merges the points within the ranges into the cached grid of that name: the points already there are kept, the manifest records the folder, the folder's medium is checked and must match the cached grid's, and a point that is already cached with different values stops the merge before anything is written.
The folder (searched with its subfolders) holds either the record's archives (`spectra_m+0.0.tar.gz` for Sonora Bobcat, `spectra.zip` for Diamondback, `teff_<min>_<max>.tar.gz` for Elf Owl, `SAND.zip`) or the files extracted from them (e.g. `sp_t1000g316nc_m0.0.gz`, `spectra_logzz_4.0_teff_1000.0_grav_316.0_mh_0.0_co_1.0.nc`); the extracted files are used when present, because they are quicker to read.
The conversion, the medium check and the manifest (which records the folder) are those of a download, and the result goes into the cache in the same way, so the folder itself is never changed.
Every requested Teff/log g point (in steps of 100 K and 0.5 dex within the ranges) must be in the folder; otherwise nothing is written and the error names the missing points.
The BT-Settl grids and LOWZ cannot be converted this way: download BT-Settl, and use [`templates.local_dir`](#templates-local_dir) for LOWZ.

<a id="templates-medium"></a>
### `templates.medium`

The medium of the template wavelengths, `air` or `vacuum`; default unset, taken from the grid's metadata.
Every grid's declared medium is checked against the strong Na I, K I, Rb I and Cs I lines, and a mismatch stops the run.
Set it only for a local grid without metadata, for the same 82 km/s reason as [`input.medium`](#input-medium).

<a id="templates-teff_range"></a>
### `templates.teff_range`

The Teff range searched, [min, max] in K; default unset, which means the [Teff prior](#templates-teff_prior) ± 300 K, or the whole grid without a prior.
Narrow it when the spectral type is well known, to save time and avoid a poor template being chosen by noise; widen it when the type is uncertain.
Give one value (e.g. `teff_range = 3000`) to fit only that Teff; it is saved as [3000, 3000].
The ±300 K window is a compromise: it spans several grid steps (100 K) to allow for the scatter of the Teff–spectral-type relation, without letting the search reach templates whose molecular bands differ grossly from the target's.

<a id="templates-logg_range"></a>
### `templates.logg_range`

The log g range searched, [min, max] in dex (cgs); default unset, the whole grid.
Restrict it to 4.5–5.5 dex for field brown dwarfs, or lower for young, low-gravity objects.
Give one value when the gravity is known (e.g. `logg_range = 5.0` for a field dwarf): only that log g is downloaded and fitted, which gives fewer templates and a faster run, and it is saved as [5.0, 5.0].
If no grid point equals the value, the run stops with an error naming the nearest values (for local grids and SVO; a Zenodo download does not list them), so that a typing error does not silently select another gravity.
At low R, log g and the template's RV are only weakly correlated, so this mainly saves time and keeps the chosen template physical.

<a id="templates-met_range"></a>
### `templates.met_range`

The metallicity range searched, [min, max] in dex; default unset, which holds [M/H] at the grid point nearest the [metallicity prior](#templates-met_prior), or the grid default.
We hold it fixed by default because each free axis multiplies the number of candidates, and at low R metallicity is poorly constrained by the RV windows.
Free it for subdwarfs, together with a grid that varies it (SAND, BT-Settl-AGSS2009, LOWZ).
A single value (e.g. `met_range = -1.0`) selects only that grid value, with the same error as for [`templates.logg_range`](#templates-logg_range) if it is absent.

<a id="templates-alpha_range"></a>
### `templates.alpha_range`

The alpha-enhancement range searched, [min, max] in dex; default unset, held at the grid point nearest the [alpha prior](#templates-alpha_prior), or the grid default.
The reasoning, and the single-value form, are as for [`templates.met_range`](#templates-met_range).

<a id="templates-ranges"></a>
### `templates.ranges`

Ranges of any other grid axes, e.g. `{co = [0.5, 1.0]}` for C/O, `kzz` for vertical mixing or `fsed` for the cloud sedimentation efficiency; default empty.
An axis without a range is held at its default.
A single value (e.g. `{co = 0.5}`) selects only that value, as for the other ranges.
Free these only when the spectrum's band shapes clearly depend on them (e.g. T dwarfs with LOWZ or Elf Owl), since every free axis multiplies the run time.

<a id="templates-teff_prior"></a>
### `templates.teff_prior`

The expected Teff, in K (e.g. from the spectral type); default unset.
It chooses the features (each catalogue feature has a Teff range in which it is useful; e.g. Rb I 1300–2400 K, TiO 1900–3500 K) and centres the template search.
We recommend always setting it, e.g. ~2100 K for an L1.
Without it, every feature in the catalogue is considered and the whole grid is searched, which is slower and can admit features that are absent or blended at the target's Teff.
A prior (Teff, or a set log g, [M/H] or [α/M] prior) more than one grid step beyond a configured grid that is available stops the run before anything is read, with an error naming the grid, its range and the request; within one step, the nearest grid point is used and the measurement flagged `template_grid_edge`.
This is because a fit at the edge template is biased, not merely less precise: with a cache holding only BT-Settl-CIFIST 2600–4500 K, every L dwarf (1700–2100 K) was fitted with the 2600 K template, which shifted the whole sample by about +14 km/s.

<a id="templates-logg_prior"></a>
### `templates.logg_prior`

The expected log g, in dex; default unset, which means 5.0 dex.
It is used when no grid point lies in the ranges and for the template of the telluric fit.
The default suits field brown dwarfs and M dwarfs; lower it for young objects.

<a id="templates-met_prior"></a>
### `templates.met_prior`

The expected metallicity, in dex; default unset (the grid default).
[M/H] is held at the nearest grid point when no [metallicity range](#templates-met_range) is set.

<a id="templates-alpha_prior"></a>
### `templates.alpha_prior`

The expected alpha enhancement, in dex; default unset (the grid default).
It is held at the nearest grid point when no [alpha range](#templates-alpha_range) is set.

<a id="methods"></a>
## Methods

<a id="methods-enabled"></a>
### `methods.enabled`

Which RV techniques run; each is a separate measurement, and they are combined into the adopted RV (see [`combine`](#combine)).
A list from `template_fit`, `ccf`, `linecentre`, `telluric_ref`, `forward_mcmc` and `lsd`; default `["template_fit", "ccf", "linecentre"]`.
When a spectrum is opened, the defaults follow its resolving power:

| Regime | R | Default methods | Why |
|---|---|---|---|
| low | < 4500 | `template_fit`, `ccf`, `linecentre` | The template fit is the most precise RV at low R, with vsini held at 0 because it is unresolved; the CCF gives one RV per window as a check; line centring is the model-free check against laboratory (NIST) wavelengths. |
| mid | 4500–20 000 | `template_fit` (vsini free), `ccf` | vsini is resolved above R 4500, so it is fitted; line centring is optional because the template methods are more precise here. |
| high | > 20 000 | `template_fit` (vsini free), `telluric_ref` | The telluric lines are resolved, so the star can be measured against them, free of the wavelength-calibration and slit-centring offsets; the CCF becomes an optional cross-check. |

We recommend keeping at least one model-free method (`linecentre`) or one that does not rely on the wavelength solution (`telluric_ref`) next to the template methods.
This is because the template fit and the CCF share the same models, so their agreement alone does not test the models.
`telluric_ref` needs resolved telluric lines, the telluric model (`rvfitter templates telluric --build`) and spectra that are not telluric-corrected, so it is not recommended below R 20 000.
`forward_mcmc` is slow (MCMC, on all but one CPU core) and at low R its vsini and Teff are not physical (on J1745−1640 at R 2600 it gave vsini 39–88 km/s per window); it is worth running at high R, especially for several epochs fitted together.
`lsd` needs the many resolved lines of a high-R spectrum to build its mean line profile; at low R its error is large (χ²/ν ≈ 94 on J1745−1640).

<a id="methods-options"></a>
### `methods.options`

Keyword options per method, as a table keyed by method name; default empty (each method's defaults).
Unknown keys stop the run with the list of valid ones.
The options most worth knowing are:
- `template_fit`: `fit_vsini` (default false at low R, true above R 4500, as set by the regime defaults), `vsini_max` (150 km/s), `poly_degree` (3), `bootstrap` (0, covariance errors);
- `ccf`, `template_fit`, `forward_mcmc`: `rv_min`, `rv_max` (the search range, −500 to +500 km/s), `teff_window` (300 K), `resolution` (an R for this method only);
- `linecentre`: `profile` (`voigt`, `gaussian` or `lorentzian`; default `voigt`), `joint` (one RV for every line; default false), `shape_correction` (default false);
- `lsd`: `mask` (`template` or `catalogue`);
- `forward_mcmc`: `n_walkers` (32), `n_burn` (1000), `n_steps` (4000), `telluric` (false).

We keep `shape_correction` off by default because it makes line centring template-dependent, which removes its value as the model-free check; on J1745−1640 it moved the RV by only +1.6 to +2.0 km/s ([methods.md](methods.md#j17451640-after-the-task-15-fixes)).
Widen `rv_min`/`rv_max` only for objects expected beyond ±500 km/s, since a wider search admits more spurious peaks.

<a id="methods-binary"></a>
### `methods.binary`

Whether to test for an unresolved binary, by comparing one template with two shifted copies (results `binary_dbic`, `binary_flag`).
True or false; default false.
The object is flagged when ΔBIC > 10, "very strong" evidence on the Kass & Raftery (1995) scale.
We note that template mismatch can mimic a second component: on J1745−1640, ΔBIC = 6.7 came from a two-component fit absorbing the mismatch (χ²/ν = 16), not from a second star ([methods.md](methods.md#results-by-method)).
A flag is therefore a reason to look more closely, not a detection.

<a id="regions"></a>
## Regions

The spectral regions (lines, bands, windows) that each method measures are selected from a literature-backed feature catalogue by resolving power, S/N, telluric contamination and the Teff prior.

<a id="regions-overrides"></a>
### `regions.overrides`

Per-method additions, replacements or removals of regions, in vacuum Å: `{name, kind, wmin, wmax[, centre, continuum]}` adds or replaces a region, and `{name, remove = true}` drops it; default empty.
We recommend editing them graphically on the Regions tab.
Remove a region when its diagnostics show a blend, a bad pixel or a distorted wavelength solution (e.g. Rb I 7802 for 2MASS J1331+3407, [methods.md](methods.md#validation-on-published-spectra)), rather than lowering the global thresholds.

<a id="regions-min_snr"></a>
### `regions.min_snr`

Features whose local S/N per pixel is below this are dropped.
Default 5.
It applies to the selection of regions and to the window-mode check of the systematics, which share one set of options.
Lower it for faint spectra, at the cost of noisier regions; the combination down-weights noisy regions, but a region at very low S/N can still pull a profile fit onto noise.
The S/N is a robust estimate from second differences of the flux, so smooth spectral structure does not count as noise.

<a id="regions-telluric_strictness"></a>
### `regions.telluric_strictness`

A scale for the airmass-dependent threshold at which telluric bands are masked: a band is masked when its airmass-scaled depth exceeds 0.15 divided by this value.
Default 1; larger masks more.
Raise it for high-airmass or humid nights, lower it for good conditions or partially corrected spectra.
It reaches the systematics' window-mode check as well as the region selection.
Telluric absorption is the main reason a region's RV can be pulled towards the observatory frame, which is why we mask by default rather than model it.

<a id="regions-max_regions"></a>
### `regions.max_regions`

The maximum number of regions kept per method, ranked by their forecast RV information content.
Default unset, which means no limit; set a number (at least 1) to keep only the best.
When set, the ranking keeps the regions that carry most of the RV information; the rest add little precision and more opportunities for mismatch.

<a id="regions-continuum_fwhm"></a>
### `regions.continuum_fwhm`

How far the continuum windows extend around a feature, in resolution elements.
Default 3.
A wider window gives a better-determined continuum but is more likely to include neighbouring features, which matters most in the crowded spectra of late-M and L dwarfs.

<a id="regions-template_windows"></a>
### `regions.template_windows`

The windows for `ccf`, `template_fit` and `forward_mcmc` below R 4500 (the edge where vsini is freed): `features` (±10 resolution elements around well-modelled atomic lines and CO bandheads) or `full` (every non-telluric span); default `features`.
From R 4500 up the full spans are always used.
We recommend `features` because the synthetic grids reproduce the molecular bands (TiO, VO, CrH, FeH, H2O) of late-M and L dwarfs poorly, which shifts the template RVs band by band; the atomic lines and bandheads are modelled well.
Use `full` for warmer stars (above ~3650 K), which have no catalogue features at their Teff (see [Stars](#stars-fgk-and-m-dwarfs)).
The other mode is always run as a check, and its difference enters the template spread, `sys_template`.

<a id="combine"></a>
## Combine

The adopted RV comes from a hierarchical model of all regions and methods, v = μ + δ_line + β_method + ε, in which the line-to-line scatter τ_line and the method-to-method scatter τ_method are marginalised under half-Cauchy priors.
The adopted value is the posterior median of μ, and its error is half the central 68 per cent interval.
This is because the region-to-region scatter is usually well above the formal errors (template mismatch, wavelength distortions), and a plain weighted mean would then let one region set the answer and understate the error.

<a id="combine-tau_scale"></a>
### `combine.tau_scale`

The scale of the half-Cauchy priors on τ_line and τ_method, in km/s; default unset, which means the median measurement error.
The half-Cauchy prior is weakly informative: it allows scatter much larger than its scale, whilst not favouring it.
Tying the scale to the median error is a mild empirical-Bayes choice that makes the prior scale with the data's own precision.
Change it only with a reason, e.g. a known instrument-level scatter.

<a id="combine-systematic_floor"></a>
### `combine.systematic_floor`

A floor added in quadrature to the adopted error, in km/s; default 0.
Use it for a known instrument floor that rvfitter does not model, e.g. a published long-term stability.
We keep it at 0 because the measured systematics (`sys_template`, `sys_wavezp`, `sys_star_sky`) are already in the total error, and a guessed floor would hide whether those are honest.

<a id="calibration"></a>
## Calibration

The corrections of the wavelength solution that are applied before any method runs, and the zero-point terms of the error budget.
The rule throughout is that an offset is either applied, with its error carried as a systematic, or unknown, with a physically motivated prior width; a measured offset's value is never added to the error.
The full analysis is in [methods.md](methods.md#wavelength-zero-point-of-the-gtc-spectra-sky-lines).

<a id="calibration-wavelength_correction"></a>
### `calibration.wavelength_correction`

The wavelength zero-point correction:
- `none` (default): measure and report only;
- `sky-if-needed` (opt-in): the OH sky curve Δv(λ), applied only when it is safe and significant;
- `auto` (opt-in): the star − sky constant from the star's telluric lines whenever it is measured, with the sky curve when `sky-if-needed` would apply it, and alone otherwise (`wavelength_correction_applied = "star"`, e.g. on a spectrum already tied to the sky in the reduction);
- `sky`: the sky curve alone, whenever it can be measured;
- `telluric`: the star's telluric curve alone.

We made `none` the default because on the GTC sample of 20 objects with literature RVs the sky correction gave no significant improvement ([methods.md](methods.md#sky-correction-on-the-gtc-sample)).
`sky-if-needed` applies the sky curve only when three tests on the calibration data pass: at least three measured sky segments covering more than half of every region; a significant offset (|mean Δv| > 3σ) or variation between segments (χ² p < 0.001); and a correction larger than 0.05 resolution elements.
The stellar line scatter is never used, so the decision does not depend on the RVs being measured.
We offer it because arc-lamp solutions can be distorted by tens of km/s with a small arc RMS: on the GTC/OSIRIS spectrum of J1745−1640 the sky shows +5 to +38 km/s over 7700–8950 Å, and correcting it reduced the line-to-line scatter from 11.3 to 3.2 km/s.
`auto` needs a spectrum with its telluric absorption (a `PypeIt` spec1d, or `calibration.telluric_source`) and a sky; its constant's error is `sys_star_sky`.
On 75 GTC/OSIRIS R2500I exposures whose wavelengths were already tied to the OH sky lines in the reduction, applying the constant brought the RVs of the 18 objects with literature values from a median offset of −15.9 to about −2.5 km/s, and the robust scatter from 31.7 to 14.3 km/s; on the original reductions, which were not tied to the sky, `auto` was not better than `none` ([methods.md](methods.md#the-star--sky-constant-on-sky-tied-spectra)).
We therefore keep `none` as the default, and recommend `auto` for spectra whose wavelength scale is tied to the sky.
The O2 A band is not a telluric node below 2800 K (the `teff_min` of O2 A in the feature catalogue), because the pressure-broadened K I 7667/7701 wings fill it; for L dwarfs the star − sky constant therefore rests on the H2O nodes alone.
`telluric` is refused when the regions lie more than 250 Å from every usable telluric node, as for the optical alkali lines; the WARNING names each such region, its nearest node and the distance (e.g. `Rb I 7802: 429 Å from H2O 8200 (node at 8231 Å)`).
A spectrum without a sky (most Spextool and JWST files) is left uncorrected in every mode.
[Wavelength-correction mode](#recipe-correction-mode) gives our recommendation and what the validation rests on.

<a id="calibration-slit_prior"></a>
### `calibration.slit_prior`

The slit-centring error, used when the star − sky constant is not applied: `auto`, a velocity in km/s, or `none`; default `auto`.
`auto` takes the star to sit anywhere within the slit, which gives σ ≈ (W − F)/√12 for a slit width W and trace FWHM F, converted to a velocity through the plate scale and dispersion (18.0 km/s for J1745−1640).
This is because the sky fills the slit uniformly, whilst the star may be off-centre, so the sky cannot measure the star's own offset.
`none` sets the term to 0 with a WARNING and the flag `zp_unconstrained`, so the total error then omits the slit centring and is understated.
Give a number when the acquisition accuracy is known better than this uniform prior.

<a id="calibration-seeing_fwhm"></a>
### `calibration.seeing_fwhm`

The seeing FWHM assumed for the `auto` slit prior when the file has no trace FWHM, in arcsec.
Default 0.8 arcsec (flag `seeing_assumed`).
Set it to the seeing of your night: a better seeing (smaller F) gives a larger slit prior, because the star can then wander further in the slit.

<a id="calibration-wave_rms_kms"></a>
### `calibration.wave_rms_kms`

The wavelength-calibration accuracy assumed when the file records no arc RMS, in km/s; default unset (flag `wave_rms_assumed` when used).
Unset, the instrument's typical value from `data/instruments.csv` is used, but no instrument has a documented typical value yet, so the arc term is unknown for files without an arc RMS.
`sys_wavezp` is the larger of the arc term and the zero-point term of a measured sky or telluric curve, so it, and `rv_total_err`, is NaN (flag `total_err_undefined`) only when both are unknown.
For `PypeIt` files (e.g. GTC/OSIRIS), rvfitter reads the arc RMS from the `WaveCalib` file in the spectrum's folder or up to three parents (each also as `Calibrations/`); without it, a measured sky curve still gives the zero-point term.
Spextool and JWST files record neither an arc RMS nor a sky, so they need this setting for a total error.
Set it to your own estimate, converted to a velocity as c Δλ/λ: e.g. the RMS of the arc-line residuals of your wavelength solution, the RMS offset of sky or telluric line centroids from their rest wavelengths over your range, or, for JWST, the documented calibration accuracy of your mode, which we have not sourced.
We do not guess a value, because the published numbers are the arc-fit RMS of one reduction rather than a typical accuracy; an arc RMS is also a lower bound, since a well-fitted arc can hide a distortion of the whole solution, as the sky lines of J1745−1640 show ([`calibration.wavelength_correction`](#calibration-wavelength_correction)).

<a id="calibration-zero_point_known"></a>
### `calibration.zero_point_known`

Whether the absolute zero point is known, as for fibre data or an absolute calibration: then there is no slit-centring term.
True or false; default false.
A spectrum extracted from an IFU cube is recognised as slitless automatically, so this need not be set for cubes.
Set it only when the instrument really has no slit-position dependence; otherwise the total error is understated.

<a id="calibration-sky_source"></a>
### `calibration.sky_source`

A file whose sky spectrum shares the input's wavelength solution (e.g. the `PypeIt` spec1d of a telluric-corrected coadd); default unset.
Use it when the input itself carries no sky, as for telluric-corrected files.

<a id="calibration-sky_segment_width"></a>
### `calibration.sky_segment_width`

The maximum width of the sky segments, in Å; Δv(λ) is interpolated between them.
Default 250 Å.
Narrower segments follow a distortion more closely but hold fewer OH lines each, so their errors grow; at 250 Å the GTC/OSIRIS sky still has enough lines per segment, and the distortion there varies over several hundred Å ([methods.md](methods.md#sky-and-star-zero-points-across-the-slit)).

<a id="calibration-telluric_source"></a>
### `calibration.telluric_source`

An uncorrected file with the input's wavelength solution, for the star's own zero point from its telluric lines; default unset.
It is needed when the input is telluric-corrected and the star − sky constant (`auto`) or the `telluric` mode is wanted.

<a id="calibration-telluric_segment_width"></a>
### `calibration.telluric_segment_width`

The maximum width of the telluric-band segments, in Å.
Default 250 Å.
The trade-off is as for [`calibration.sky_segment_width`](#calibration-sky_segment_width).

<a id="calibration-telluric_template"></a>
### `calibration.telluric_template`

Whether the star is modelled with the template nearest the priors in the telluric fit.
True or false; default true.
We recommend it because the stellar lines and bands overlap the telluric bands in cool dwarfs, so fitting the telluric lines against a smooth star biases the nodes (a smooth star gave a constant 3.5–4.3 km/s less negative on J1745−1640; [methods.md](methods.md#j17451640-after-the-task-15-fixes)).
Untick it only when no template covers the telluric bands.

<a id="calibration-wave_valid_range"></a>
### `calibration.wave_valid_range`

The vacuum wavelength range, [min, max] in Å in the observatory frame, where the wavelength solution is constrained, e.g. by the arc lines; default unset, which uses the arc-line range found next to a `PypeIt` file.
Sky segments outside it are not used.
Set it when you know the arc lines cover less than the spectrum, since a solution extrapolated beyond its arcs can be distorted.

<a id="parallel"></a>
## Parallel

<a id="parallel-n_procs"></a>
### `parallel.n_procs`

The number of worker processes within each object; default unset (automatic): serial, or all but one core when `forward_mcmc` runs.
Results do not depend on it, because the random streams and the work units are fixed in advance.
A pool only pays off for the MCMC; for the other methods, the start-up of the processes costs more than it saves.

<a id="output"></a>
## Output

<a id="output-path"></a>
### `output.path`

The file the results are written to after each run; each format gets its own extension.
Default `results.csv`; relative paths are relative to the TOML file.
In the GUI, empty means saving only with Export.
The settings actually used are saved next to it as `results_config.toml`, so any result can be reproduced.

<a id="output-formats"></a>
### `output.formats`

The results file formats: any of `csv`, `json`, `tsv` and `html`; default `["csv"]`.

<a id="output-plots"></a>
### `output.plots`

Whether `rvfitter run` writes the diagnostic plots next to the results.
True or false; default true.
The GUI shows them in the Diagnostics tab instead.
We recommend keeping them, because the fits of discrepant regions are the quickest way to find a blend or a distorted wavelength solution.

<a id="output-plot_style"></a>
### `output.plot_style`

The style of the written plots: `quick` (bundled Computer Modern fonts, no LaTeX needed) or `paper` (text typeset by LaTeX; needs `latex` and `dvipng`); default `quick`.

<a id="output-medium"></a>
### `output.medium`

The medium of the wavelengths in the reports and plots: `vacuum` or `air`; default `vacuum`.
It changes only the labels: all measurements are made in vacuum.
Choose `air` to compare with line lists or figures in air.

<a id="which-settings-for-my-data"></a>
## Which settings for my data?

In this Section, we give a starting point for common kinds of data:

| Data | Recipe |
|---|---|
| Optical, R < 4500, late-M and L dwarfs (GTC/OSIRIS) | [Optical at low R](#recipe-optical-low-r) |
| Near-infrared, R < 4500 (SpeX SXD, ARCoIRIS) | [Near-infrared at low R](#recipe-nir-low-r) |
| R ≳ 4500, echelle, fibre-fed | [Mid and high R](#recipe-high-r) |
| JWST NIRSpec IFU cube | [IFU cubes](#recipe-ifu) |
| JWST NIRSpec MOS, several sources per file | [Multi-object spectra](#recipe-mos) |
| M dwarfs and FGK stars | [Stars](#stars-fgk-and-m-dwarfs) |
| Galaxies | [Galaxies](#recipe-galaxies) (not supported) |
| Which features a band offers at a given Teff | [Features by band](#features-by-band) |
| Which wavelength correction to use | [Wavelength-correction mode](#recipe-correction-mode) |

In every case, set [`input.resolution`](#input-resolution) if the header lacks it and a [Teff prior](#templates-teff_prior) from the spectral type, and look at the diagnostics before trusting an RV.
The input must be one merged 1D spectrum per object (or an IFU cube, or a multi-object x1d file): rvfitter has no reader for per-order or 2D echelle files, except Spextool's order stack, which it merges (where orders overlap, the bluer order's pixels are kept).
For other echelle data, merge the orders into one 1D spectrum with your reduction pipeline's order-merging or coadd step first, and trim the low-S/N order edges or weight the overlaps by their inverse variance, because a noisy order edge spliced in unweighted adds spurious features at the order joins.

To see what precision to expect before observing or measuring, run `rvfitter forecast --snr <S/N per pixel> --resolution <R> --wave-range <min> <max>` (or `--spectrum <file>` for its pixels and R), with `--teff` and `--logg` or the config's priors and a downloaded grid.
It writes the photon-noise RV error (Bouchy et al. 2001) per feature window, per band and in total, and the windows that hold 90 per cent of the information, to `forecast.csv`.
This is a lower limit: it has no zero-point term, and at low R the zero point dominates (on J1745−1640 the statistical errors were 3–5 km/s, against a total of about 20 km/s; [methods.md](methods.md#results-by-method)), whilst the validation scatter against the literature is in [methods.md](methods.md#validation-on-published-spectra).

<a id="recipe-correction-mode"></a>
### Wavelength-correction mode

We recommend the default, `none` ([`calibration.wavelength_correction`](#calibration-wavelength_correction)).
This was decided by a test on the GTC sample fixed before the results were seen: against 20 literature RVs, `sky-if-needed` lowered the robust scatter from 29.8 to 22.5 km/s but left the pull width unchanged (1.20 against 1.19), and the uncorrected offsets did not follow each object's sky Δv significantly (Spearman p = 0.17; [methods.md](methods.md#sky-correction-on-the-gtc-sample)).
For spectra whose wavelength scale was already tied to the OH sky lines in the reduction, we recommend `auto`, which then applies the star − sky constant alone ([methods.md](methods.md#the-star--sky-constant-on-sky-tied-spectra)).
Without a sky spectrum (every Spextool and JWST file, and coadds without a `sky_source`) every mode applies nothing.
With a sky, `sky-if-needed` is a useful check: on J1745−1640 it reduced the line-to-line scatter from 11.3 to 3.2 km/s but moved line centring from 35.0 to 13.6 km/s, against 26.2 ± 2.3 km/s in the literature.
No mode agrees with the literature within the statistical errors for every method, whilst every mode lies within 1.3 of its total errors ([methods.md](methods.md#results-by-method)), so the zero point of that spectrum, not the mode, limits it.
When you have a sky, we therefore suggest comparing a run with `sky-if-needed` against the default, and treating a difference larger than the statistical errors as a zero-point problem of that spectrum.

<a id="recipe-optical-low-r"></a>
### Brown dwarfs in the optical at low R (e.g. GTC/OSIRIS R2500I)

This is the case rvfitter was built and validated on (56 late-M and L dwarfs, R ≈ 2600; [methods.md](methods.md#validation-on-published-spectra)).

```toml
[input]
paths = ["DENISJ174534.6-164053_tellcorr.fits"]
telluric_corrected = true          # PypeIt coadd
resolution = 2602.0
raw_header_dir = "raw"             # slit angle for the DAR, slit width and binning

[velocity_frame]
correction = "skip"               # PypeIt corrected it, but the coadd does not record it
frame = "heliocentric"            # the label of the reported RVs: PypeIt's frame

[templates]
grid = "bt-settl-cifist"
teff_prior = 2100.0               # L1-L2

[methods]
enabled = ["template_fit", "ccf", "linecentre"]

[calibration]
sky_source = "spec1d_....fits"    # the spec1d of the same exposure: zp_sky_kms is reported
# wavelength_correction = "sky-if-needed"  # opt-in: apply the sky zero point (moves J1745's RV by -21 km/s)
# telluric_source = "spec1d_....fits"  # only for wavelength_correction = "auto"
```

The low-R defaults apply: vsini fixed at 0, feature windows, and no wavelength correction (`none`; the sky zero point is measured and reported).
`PypeIt` corrected this coadd to the heliocentre, but its headers carry no `VEL_TYPE` (nor coordinates and time), so under `auto` its frame is unknown and the GUI asks; `skip` says so in the settings.
`skip` applies no correction, and `frame` then only labels the output: here the RVs are heliocentric, as `PypeIt` made them.
Leaving `frame = "barycentric"` would mislabel them by at most about 0.013 km/s ([`velocity_frame.frame`](#velocity_frame-frame)).
With `skip`, a [`velocity_frame.overrides`](#velocity_frame-overrides) site (e.g. `{site = "lapalma"}` for the wrong GTC header longitude) has no effect, so the 0.155 km/s offset of `PypeIt`'s own correction stays in the RVs, well below their errors ([methods.md](methods.md#barycentric-correction-of-the-gtc-spectra)).
For a file with `VEL_TYPE` (e.g. a `PypeIt` spec1d), `auto` recognises the correction and only warns if the frames differ.
We expect the zero point, not the method, to limit the absolute RV here: on J1745−1640 the slit prior alone is 18.0 km/s, against statistical errors of 3–5 km/s.
`regions.template_windows = "features"` matters most in this regime, because of the poorly modelled TiO, VO, CrH and FeH bands.

<a id="recipe-nir-low-r"></a>
### Near-infrared at low R (IRTF/SpeX SXD, Blanco/ARCoIRIS)

This recipe is for SpeX SXD (R ≈ 2000 with the 0.3 arcsec slit) and ARCoIRIS (R ≈ 3500, fixed slit) spectra in Spextool format (R from the instrument pages cited in `data/instruments.csv`; other SpeX slits give other R, which we have not sourced, so set [`input.resolution`](#input-resolution)).

```toml
[input]
paths = ["sxd_J0808+3157.fits", "xtc_J1536+0646.fits"]

[methods]
enabled = ["template_fit", "ccf", "linecentre"]   # the low-R defaults; needs a downloaded grid

[calibration]
wave_rms_kms = 5.0                # your estimate; otherwise rv_total_err is undefined
```

We recommend the low-R defaults, as in the optical: every grid whose files we have checked covers 0.6–15 µm or more, the J, H and K bands included ([Choosing a model grid](#choosing-a-model-grid)).
We note candidly that the near-infrared validation ran line centring only, because our local BT-Settl-CIFIST grid had been cut to 4000–11 000 Å before rvfitter read it, so the template methods are not yet validated in this band.
Without a grid, `enabled = ["linecentre"]` (the setting of `examples/near_infrared/`) runs on the J-band lines with no download.
At R ≈ 2000 (SXD) the features that survive are K I (from R 1000), Na I 11385, 11407 (from R 1500) and the CO bandheads (from R 1000); ARCoIRIS at R ≈ 3500 adds Al I 13127, 13154 (from R 3000); Fe I needs R ≥ 4000 ([Features by band](#features-by-band) gives the Teff ranges).
Spextool files record no arc RMS and no sky, so no sky correction is possible and the calibration term needs [`calibration.wave_rms_kms`](#calibration-wave_rms_kms); the `auto` slit prior is computed from the slit width (`SLTW_ARC`) and the plate scale of `data/instruments.csv`, with the assumed [`calibration.seeing_fwhm`](#calibration-seeing_fwhm), because Spextool records no trace FWHM.
Below R 1000 the catalogue offers no near-infrared feature (the lowest `r_min` there is 1000), so line centring gets no regions and the template methods fall back to the full spans, each with a WARNING; all three methods then refuse the spectrum (R below their minimum of 1000, an ERROR per method), and no RV is reported for it.
This holds, for example, for a T dwarf at R 750 and for SpeX prism spectra (R ≈ 80–220), as intended: such R cannot constrain an RV, so the only thing to do is check that the R used is right (set [`input.resolution`](#input-resolution) if the header or the pixel-spacing estimate is wrong), not raise it to get past the limit.
rvfitter treats every Spextool file as not velocity-corrected, because no Spextool velocity keyword could be verified, and computes the barycentric correction from the header (`bary_source = "computed"` in the results); if yours were already corrected, set `velocity_frame.correction = "skip"`.

<a id="recipe-high-r"></a>
### Mid and high R (R ≳ 4500), echelle and fibre data

Above R 4500 vsini is resolved, so `template_fit` fits it (`fit_vsini = true`, set by the regime defaults), and from R 4500 up the full non-telluric spans are used, because many resolved lines average down the mismatch of individual bands.
Above R 20 000 we recommend `telluric_ref` for spectra that are not telluric-corrected, since the telluric lines tie the RV to the atmosphere rather than to the arc solution and the slit position; build the telluric model first (`rvfitter templates telluric --build`).
At high R, `forward_mcmc` (several epochs fitted together) and `lsd` (RV and vsini from the mean line profile) become worthwhile.
Echelle orders must be merged into one 1D spectrum first (see above).
For a fibre feed, set [`calibration.zero_point_known`](#calibration-zero_point_known) to true, because there is no slit; for a slit echelle, leave it false, so that the slit-centring prior enters the error.
The telluric model covers 6000–25 000 Å (vacuum), and the catalogue's telluric bands start at O2 B (6862 Å), so `telluric_ref` and the telluric zero point have no reference in bluer orders.
`telluric_ref` uses the catalogue's telluric bands that the spectrum covers and whose R limit it reaches (O2 from R 5000, H2O and CO2 from R 10 000, CH4 2.3 µm from R 20 000), whatever the star's Teff, so it serves FGK stars as well; it needs a spectrum that is not telluric-corrected.
We have not validated rvfitter above R 4500: the benchmark and both validation sets are at R ≈ 2000–3500.

<a id="recipe-ifu"></a>
### IFU cubes (e.g. JWST NIRSpec IFU)

A spectrum is extracted in a circular aperture of radius 3 spaxels on the brightest spaxel ([`input.cube`](#input-cube)).

```toml
[input]
paths = ["jw01386013001_03104_00001_nrs1_s3d.fits"]
telluric_corrected = true         # space data: no telluric absorption

[input.cube]
# position = [19.0, 33.0]         # e.g. for a companion that is not the brightest source

[templates]
teff_prior = 1200.0

[methods]
enabled = ["ccf", "template_fit"]
```

JWST wavelengths are vacuum and already barycentric (the pipeline's `assign_wcs` step), so `auto` applies no further correction.
A cube has no slit, so the slit-centring term, the DAR and the star − sky constant are not computed.
JWST files record no arc RMS and no instrument in `data/instruments.csv` has a typical calibration accuracy, so without [`calibration.wave_rms_kms`](#calibration-wave_rms_kms) `sys_wavezp`, and so `rv_total_err`, is NaN (flag `total_err_undefined`); we recommend setting it to your own estimate of the NIRSpec wavelength-calibration accuracy (e.g. from the JWST documentation), which we have not sourced.
The catalogue has no feature at 2.9–5 µm ([Features by band](#features-by-band)), so the template methods fall back to the full spans (WARNING) and line centring has no lines.
VHS 1256 b is an L/T object at about 1100–1400 K, where the grid advice ranks `sonora-elfowl` first below 1300 K and `bt-settl-cifist` first above it; all the Sonora grids reach 4.1 µm.
We recommend `sonora-elfowl` for such an object; our example run used `sonora-bobcat` at 1000–1400 K only because that download was already in our cache, and its best template sat at the 1400 K edge, so we would extend the range upwards before trusting the result.
The resolving power is not in the headers, so rvfitter estimates it from the pixel spacing (R ≈ 2090 for this cube, assuming 2.5 pixels per resolution element; flag `resolution_estimated`) and checks it against the documented range of the grating in `data/instruments.csv` (Jakobsen et al. 2022: PRISM 30–330, the M gratings 500–1340, the H gratings 1320–3600), widened by a factor 1.5.
For G395H our only sourced value is that range, R 1320–3600 for the H gratings as a whole (the abstract of Jakobsen et al. 2022), which does not give R at 2.9–4.1 µm; the estimate of 2090 lies within it even without the widening.
R varies along each grating, so set [`input.resolution`](#input-resolution) to the value at your wavelengths when you know it, e.g. from the dispersion curve the JWST documentation gives for the grating.
G395H is therefore in the low-R regime (below R 4500), so the vsini cannot be measured, and the absolute zero point rests on the JWST wavelength calibration, which rvfitter cannot check without sky lines.

<a id="recipe-mos"></a>
### Multi-object spectra (e.g. JWST NIRSpec MOS PRISM)

Each source of an x1d file becomes its own object, named by its `SRCNAME`, with [`input.objects = "auto"`](#input-objects); sources with S/N below 3 per pixel are skipped.
The PRISM, however, has R ≈ 30–330, so one resolution element is about 2500 km/s at R 120, and no RV can be constrained: the run warns, and every method refuses the spectrum, because it is below R 1000 ([`input.resolution`](#input-resolution)).
This is a property of the data, not a setting to work around.
The NIRSpec gratings (R 500–1340 and 1320–3600) may give RVs, with the same caveats as the IFU about R and `wave_rms_kms`.
A MOS shutter is a slit, but rvfitter has no plate scale for it, so it cannot compute the slit-centring prior: `sys_star_sky`, and so `rv_total_err`, is NaN (WARNING, flag `zp_unconstrained`) unless you set [`calibration.slit_prior`](#calibration-slit_prior) to a velocity in km/s.
To estimate it, take the uncertainty of the source's position across the shutter along the dispersion, Δx in detector pixels, and convert it as σ_v ≈ c Δx D/λ, with D the dispersion in Å per pixel at λ (the spacing of the x1d wavelengths).
The x1d headers record the planned position in the shutter (`SRCXPOS`, `SRCYPOS`, 'Source position in slit') but not the pointing error, and rvfitter has no sourced shutter size in pixels, so take both from the JWST documentation of your mode; the uniform form of the `auto` prior, (W − F)/√12 with W the shutter width and F the trace FWHM, assumes a source anywhere in the shutter, which suits an unknown position.
We do not recommend `zero_point_known = true` here, because it would drop the term rather than estimate it.

All sources of one file share one configuration.
To give a source its own settings, e.g. a Teff prior, add a `[[refits]]` entry for it by its object name, which `rvfitter run` applies to that object only:

```toml
[[refits]]
object = "1229_9"
patch = {templates = {teff_prior = 1500.0}}
```

Alternatively, run groups of sources separately with different configs, unticking the others in the GUI.

<a id="stars-fgk-and-m-dwarfs"></a>
### Stars (FGK and M dwarfs)

rvfitter can measure stars, but its feature catalogue was built for ultracool dwarfs: its features cover Teff 400–3650 K and 6565–23 835 Å ([Features by band](#features-by-band)).
For M dwarfs (2400–3650 K), use the defaults with `bt-settl-cifist` at solar metallicity (`bt-settl-agss2009` is ranked second there), or `sand` for metal-poor M subdwarfs (it varies [M/H] from −2.4 to +0.3 dex, with alpha enhancement), and free [`templates.met_range`](#templates-met_range) for those.
For FGK stars, no catalogue feature applies at their Teff, so line centring has no lines and the feature windows are empty.
We therefore recommend `template_fit` and `ccf` with `regions.template_windows = "full"` (the setting advice gives `full` above 3650 K, the upper Teff limit of the well-modelled features), and `bt-settl-agss2009` (400–70 000 K, [M/H] −4 to +0.5 dex, [α/M] 0 to 0.4 dex) with a metallicity range, or `bt-settl-cifist` up to 7000 K for solar-metallicity stars.
For a field dwarf of known gravity, `logg_range = 5.0` fits only that gravity and needs fewer templates.
A single value must be a grid point: for a log g between grid points (e.g. 4.4 dex) the run stops with an error naming the nearest grid values, so give one of them or a range that brackets it (e.g. `logg_range = [4.0, 4.5]`).
For spectra that are not telluric-corrected at R ≥ 5000, `telluric_ref` also works for FGK stars ([Mid and high R](#recipe-high-r)).
We note that we have not validated rvfitter on FGK stars, and that dedicated stellar pipelines with line masks tuned for them (e.g. the instrument's own cross-correlation pipeline at high R) are probably more precise there.

<a id="recipe-galaxies"></a>
### Galaxies

rvfitter is not designed for galaxies, and we do not recommend it for them.
It has no galaxy or stellar-population templates, and it does not treat velocity dispersion: its only broadening is a stellar rotation kernel, whereas a galaxy's lines are broadened by the line-of-sight velocity distribution of its stars.
It also has no redshift treatment: it applies the non-relativistic Doppler shift of a nearby source, searches only ±500 km/s by default, and its feature catalogue and resolution regimes assume stellar and substellar spectra.
A galaxy spectrum would therefore give a biased velocity with an error that does not describe it.
For galaxy kinematics we suggest `pPXF` (Cappellari & Emsellem 2004; Cappellari 2017), which fits the velocity and dispersion with stellar-population templates; for redshifts, template-based redshift fitters such as `Redrock` (from DESI) are designed for the task.

<a id="features-by-band"></a>
### Features by band

The table lists the non-telluric features of the catalogue (`src/rvfitter/data/features.ecsv`), with the Teff range [K] and the minimum R at which each is used.
Line centring uses the atomic lines; the template methods use windows around the well-modelled features (atomic lines and CO bandheads) below R 4500, and fall back to the full non-telluric spans, with a WARNING, when none applies; the poorly modelled bands enter only through the full spans.

| Band | Atomic lines (line centring and feature windows) | Well-modelled bandheads | Poorly modelled bands |
|---|---|---|---|
| Optical (6500–10 000 Å) | Hα 6565 (1500–3500, R 1000); Li I 6710 (1500–3000, R 2000); K I 7667, 7701 (1200–3000, R 1000); Rb I 7802, 7949 (1300–2400, R 1500); Na I 8185, 8197 (1300–3200, R 1000); Cs I 8523, 8946 (1200–2300, R 1500) | – | TiO 7054, 7591, 8434 (1900–3500); VO 7334, 7851 (1700–3000); CrH 8613 (1300–2600); FeH 8694, 9899 (1300–3200) |
| J (1.1–1.35 µm) | Na I 11385, 11407 (1500–3200, R 1500); K I 11693, 11776, 12436, 12526 (1000–3650, R 1000); Fe I 11886, 11887, 11976 (1800–3500, R 4000); Al I 13127, 13154 (1800–3200, R 3000) | – | – |
| H (1.4–1.8 µm) | – | – | H2O 1.4 µm (500–3000); CH4 1.6 µm (400–1500) |
| K (1.9–2.4 µm) | – | CO 22935, 23227, 23535 (1200–3500, R 1000) | H2O 1.9 µm (500–3000); CH4 2.2 µm (400–1500) |
| L and M (2.9–5 µm) | – | – | – |

For example, an M5 dwarf (about 3000 K) in the near infrared gets the J-band K I (R ≥ 1000), Na I (R ≥ 1500), Al I (R ≥ 3000) and Fe I (R ≥ 4000) lines and the K-band CO bandheads (R ≥ 1000), so at R 2000 only the K I and Na I lines and the CO bandheads survive; a T5 dwarf (about 1100 K) at J, H and K gets only the four J-band K I lines; and an L or T dwarf at 2.9–4.1 µm gets no catalogue feature, so line centring cannot run and the template methods use the full spans.

<a id="choosing-a-model-grid"></a>
## Choosing a model grid

The table lists the grids rvfitter knows (`templates.KNOWN_GRIDS`).
The coverage is the grid's parameter range; except for BT-Settl-CIFIST, it has not yet been checked against a live download, so a particular download may cover less.
All grids are converted to vacuum on reading, and their medium is verified against the alkali lines.

| Grid | Teff [K] | log g [dex] | [M/H] [dex] | Other axes | Physics | When to use | Reference |
|---|---|---|---|---|---|---|---|
| `bt-settl-cifist` (default) | 1200–7000 | 2.5–5.5 | 0 | – | PHOENIX with the BT-Settl cloud model, CIFIST solar abundances | Late-M and L dwarfs, M dwarfs, solar-metallicity stars to 7000 K; validated here | Allard et al. 2012; Caffau et al. 2011 |
| `bt-settl-agss2009` | 400–70 000 | −0.5–6.0 | −4.0–+0.5 | [α/M] 0–0.4 | BT-Settl with Asplund et al. (2009) abundances | Non-solar metallicity, warmer stars | Allard et al. 2012 |
| `sand` | 700–4000 | 2.0–6.0 | −2.4–+0.3 | [α/M] −0.05–0.4 | PHOENIX-based, metal-poor and alpha-enhanced (specific [M/H], [α/M] pairs) | M, L and T subdwarfs | Alvarado et al. 2024, RNAAS 8, 134 |
| `sonora-bobcat` | 200–2400 | 3.0–5.5 | −0.5–+0.5 | C/O 0.5–1.5 | Cloud-free, chemical equilibrium | T and Y dwarfs; late-L dwarfs as a cloud-free comparison | Marley et al. 2021 |
| `sonora-diamondback` | 900–2400 | 3.5–5.5 | −0.5–+0.5 | f_sed 0–8 (0 cloud-free) | Silicate clouds of variable sedimentation efficiency | Cloudy L and L/T transition dwarfs | Morley et al. 2024 |
| `sonora-elfowl` | 275–2400 | 3.25–5.5 | −1.0–+1.0 | C/O 0.22–1.14, log K_zz 2–9 | Cloud-free, disequilibrium chemistry from vertical mixing | T and Y dwarfs, where CH4, CO and NH3 are out of equilibrium | Mukherjee et al. 2024; Wogan et al. 2025 |
| `lowz` | 500–1600 | 3.5–5.25 | −2.5–+1.0 | C/O 0.1–0.85, log K_zz −1–10 | Cloud-free, low-metallicity, with mixing | Metal-poor T dwarfs; download manually and use [`templates.local_dir`](#templates-local_dir) | Meisner et al. 2021 |

rvfitter keeps the full wavelength range of every file it downloads (neither the SVO nor the Zenodo route cuts it).
The wavelength coverage below was read from a file of each grid where we could, and from the Zenodo record otherwise; the bands are optical (0.4–1 µm), J, H, K and L/M (2.9–5 µm).

| Grid | Wavelength [µm] | Bands | Native resolving power | Checked on |
|---|---|---|---|---|
| `bt-settl-cifist` | 0–1000 | all | – | an SVO spectrum (2000 K, log g 5.0) |
| `bt-settl-agss2009` | not checked (the SVO query timed out) | – | – | – |
| `sand` | 0.0001–1000 | all | – | a Zenodo file (1600 K, log g 5.0) |
| `sonora-bobcat` | 0.4–50 | all | 6000–200 000 | the 27 cached files (Zenodo) |
| `sonora-diamondback` | 0.3–250 | all | ≈50 000 at 0.3–0.8 µm, ≈200 000 at 0.8–2 µm, ≈30 000 at 2–20 µm | a Zenodo file (1300 K) |
| `sonora-elfowl` | 0.6–15 | all (optical from 0.6 µm) | – | the Zenodo record's description |
| `lowz` | not checked (manual download) | – | – | – |

A local grid can cover less: our local BT-Settl-CIFIST grid (the one the benchmark tests read) had been cut to 4000–11 000 Å before rvfitter read it, so it covers neither the J, H and K bands nor 2.9–5 µm; check the range of any grid you prepared yourself.
At high R, we note that the Sonora resolving powers above (from their Zenodo records) set a limit: at 2–20 µm their spectra resolve no better than R ≈ 30 000.

We recommend choosing the grid by the physics that dominates the RV windows, rather than by the widest coverage.
This is because a template RV is set by the positions and shapes of the lines it matches, so a grid that mismodels the dominant opacity (clouds in L dwarfs, CH4 in T dwarfs, metallicity in subdwarfs) shifts the RV even when its Teff is right.
The setting advice ranks the grids by Teff, using the Teff prior (or the best-fit Teff of the last run), and keeps those whose coverage contains it:

| Teff [K] | Grids, best first |
|---|---|
| ≥ 4000 | `bt-settl-agss2009`, `bt-settl-cifist` |
| 2000–4000 | `bt-settl-cifist`, `bt-settl-agss2009`, `sonora-diamondback`, `sand` |
| 1300–2000 | `bt-settl-cifist`, `sonora-diamondback`, `sonora-elfowl` |
| < 1300 | `sonora-elfowl`, `sonora-bobcat`, `sonora-diamondback` |

A [metallicity prior](#templates-met_prior) of −0.5 dex or lower puts `sand`, `lowz` and `bt-settl-agss2009` first.
The reasons are those above.
Above 4000 K the clouds and molecular bands that distinguish the other grids are absent, and only `bt-settl-agss2009` varies [M/H] and [α/M] there.
Between 2000 and 4000 K the CIFIST grid is validated here on M and L dwarfs, and the others serve non-solar metallicity (`bt-settl-agss2009`, `sand`) or cloud comparison (`sonora-diamondback`).
Between 1300 and 2000 K, in the L dwarfs, clouds shape the spectrum, so the cloudy grids come first.
Below 1300 K, in the T dwarfs, CH4, CO and NH3 are out of equilibrium and clouds matter less, so the cloud-free disequilibrium grid `sonora-elfowl` comes first.

When in doubt, list two grids in [`templates.grid`](#templates-grid): their spread enters the error as a systematic.

## References

- Allard F., Homeier D., Freytag B., 2012, Phil. Trans. R. Soc. A, 370, 2765 (BT-Settl)
- Alvarado E. et al., 2024, RNAAS, 8, 134 (SAND)
- Asplund M. et al., 2009, ARA&A, 47, 481
- Bouchy F., Pepe F., Queloz D., 2001, A&A, 374, 733
- Caffau E. et al., 2011, Sol. Phys., 268, 255
- Cappellari M., Emsellem E., 2004, PASP, 116, 138; Cappellari M., 2017, MNRAS, 466, 798 (pPXF)
- Ciddor P. E., 1996, Appl. Opt., 35, 1566
- Edlén B., 1953, J. Opt. Soc. Am., 43, 339
- Jakobsen P. et al., 2022, A&A, 661, A80 (NIRSpec)
- Kass R. E., Raftery A. E., 1995, J. Am. Stat. Assoc., 90, 773
- Marley M. S. et al., 2021, ApJ, 920, 85 (Sonora Bobcat)
- Meisner A. M. et al., 2021, ApJ, 915, 120 (LOWZ)
- Morley C. V. et al., 2024 (Sonora Diamondback)
- Morton D. C., 2000, ApJS, 130, 403
- Mukherjee S. et al., 2024; Wogan N. F. et al., 2025 (Sonora Elf Owl)
- The references of the benchmark and validation are in [methods.md](methods.md#references).
