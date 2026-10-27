# Dev notes: barycentric, features, regions (P3)

- `barycentric.correction` returns the km/s to add (RV_frame = RV_obs + v). `apply` scales wavelengths by sqrt((1+b)/(1-b)). PypeIt `VEL_CORR` is that factor, so v = c (f^2-1)/(f^2+1) (J1745: -21.0055 km/s heliocentric).
- PypeIt's GTC header has `LON-OBS = +17.877`, but GTC is at 17.89 deg WEST. PypeIt's correction therefore uses a wrong longitude and differs by ~0.15 km/s from the one computed with the built-in site table (`GTC`). The barycentric file matches our calculation to <1 m/s only when its lat/lon are passed verbatim. Recommend a config option or note for GTC reproduction.
- Spextool: no velocity keyword could be verified; SpeX spectra are treated as uncorrected (UNVERIFIED).
- Offline site table values are rounded public coordinates (~0.01 deg).
- `features.ecsv` has an extra column `depth` (approximate telluric absorption depth at R~2000, airmass 1; UNSURE, order of magnitude) used by the strictness thresholds. Atomic lines come from `nist_lines.csv`; band limits, Teff ranges and r_min are approximate and cite Kirkpatrick 1999, McLean 2003, Cushing 2005 (not re-verified against the papers).
- `regions.select` has one extra keyword, `options: SelectOptions`. H I (Halpha) is excluded from automatic line selection (emission).
- Methods: `linecentre` gets lines, `telluric_ref` gets telluric bands (needs R >= r_min), all others get non-telluric spans.
