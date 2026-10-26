# Dev notes: io, archives, config (P1)

## io.py
- Registry order: `pypeit_spec1d`, `pypeit_coadd`, `fits_generic`, `ascii`, then `specutils` fallback.
  `register_reader(name, sniff, read, first=False)`; `sniff(path, hdu, columns) -> bool`, `read(...) -> RawSpectrum` (Å, file medium).
  If a sniffed reader's `read` fails the next one is tried; errors are collected in the final `ValueError`.
- **Medium decisions** (never silent):
  - explicit `medium` wins (warning if it contradicts the header);
  - `auto` reads `AIRORVAC`, `WAVE_MED`-style keys, `VACUUM`, then `CTYPE1` ('AWAV' → air, 'WAVE' → vacuum);
  - PypeIt spec1d and the `wave/flux/ivar` coadd/tellcorr tables (lower-case column names) fall back to **vacuum**, logged at INFO and recorded as `meta['medium_source'] = 'default:pypeit_*'`.
    This is how the GTC `*_tellcorr.fits` files with empty headers are read.
  - Any other file without a keyword raises `ValueError` asking for `medium='air'|'vacuum'`.
- `meta` keys added beyond the contract: `medium_source`, `reader`, `mjd_source`, `telescope`, `site_lat`/`site_lon` (if in header), and `velocity_keywords` (dict of VHELIO/BCV/BERV/VEL_TYPE/VEL_CORR… found), for the barycentric module's "already corrected" detection.
- `meta['vel_corr']` is PypeIt's raw `VEL_CORR`, a multiplicative factor already applied to the wavelengths (about 0.99993 for J1745); it is not in km/s.
  `meta['frame_in']` is the lower-cased `VEL_TYPE` (None if absent).
- `mjd_mid`: `MJD_MID` if present, else the start-of-exposure keywords (`MJD-OBS`, `MJD`, `DATE-OBS`) plus `EXPTIME`/2.
- RA/DEC: floats are degrees; sexagesimal strings are hours (RA) and degrees (Dec).
- Units without a declared unit are guessed from the median wavelength (µm < 100, nm < 3000, else Å) with a warning.
- 1D WCS log-linear: `DC-FLAG = 1` means log10 (IRAF); `CTYPE1` ending `-LOG` follows FITS WCS paper III.
- ASCII: `# KEY = value` comment lines form a pseudo-header (OBJECT, MJD, AIRORVAC, RESOLUTION…); column names and units are taken from a header row or comment; words "vacuum"/"air" in comments are used as a medium hint only if exactly one occurs.
- `columns` = `{'wave': name|index, 'flux': …, 'err'|'ivar'|'var': …}`.

## archives.py
- `open_inputs(path, max_bytes=2 GB, max_members=10 000)` is the context manager; `collect_inputs` keeps archive extractions alive until interpreter exit (use `open_inputs` when possible).
- Zip: names, symlink/special mode bits, and bytes actually written are checked (declared sizes can lie). Tar: single pass with `filter='data'` plus explicit link/device rejection. Needs Python >= 3.11.4.
- `load_spectra(files, **read_kwargs)` reads files, logs and skips unreadable ones, and raises if none could be read.
- `group_by_object(spectra, override)`: override keys are tried against source path, file name, stem, header object name.

## config.py
- Sections: input, velocity_frame, templates, methods, regions, combine, parallel, output (+ `seed`, `schema_version`).
  Unknown keys are errors with "did you mean" hints; all semantic errors are listed together.
- `Config.hash()` hashes the whole canonical JSON, **including** `input.paths` and `output`. If the orchestrator wants hashes that ignore where data and outputs live, exclude those two sections in `hash()`.
- `VelocityFrameConfig.overrides()` returns keys `ra, dec, mjd (mid-exposure), site, site_lat, site_lon`.
- Option keys invented in `default_for` for the Opus package to honour or rename: `template_fit.telluric_model`, `forward_mcmc.fit_vsini`, `telluric_model`, `fit_wave_offset`, `per_order`. Region thresholds `min_snr`, `telluric_threshold`, `max_regions`, `continuum_fwhm` are for P3.
- Regime edges: R < 4500 low, 4500–20000 mid, > 20000 high. Bands: optical (ends below 11000 Å), nir (starts at or above 9000 Å), else mixed. These only select default methods, not regions.
- `systematic_floor` defaults to 0 (the plan gives anchors, not a default).
