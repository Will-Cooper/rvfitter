# Measuring brown-dwarf radial velocities: literature review

This document reviews [`deep-research-report.md`](deep-research-report.md) (the primary source) against rvfitter.
Items not covered by that report are marked *[outside report]*.
The report cites works by author and year only, so references below follow the same convention; full bibcodes are given where we have verified them.

## Technique inventory

Key: **core** = measures RV; **cal** = calibration or correction step; **err** = error or combination scheme.

| # | Technique | What it does | Applies when | Precision quoted | References | Role |
|---|---|---|---|---|---|---|
| 1 | Cross-correlation (CCF) against an empirical or synthetic template | Correlate over a velocity grid. Fit the peak with a Gaussian or parabola. Mask tellurics, detector defects and fringing. | Reconnaissance and relative RVs. Weak with spectral-type, rotation or binary mismatch. | Not quoted alone | Early optical/NIR surveys | core |
| 2 | Direct template matching | Minimise pixel residuals between the data and a shifted, broadened template with a continuum polynomial. Keeps pixel weights; masking is easy. | An empirical template needs a known RV and a close type, gravity, [M/H] and vsini match. | Not quoted separately | Blake et al. 2010 | core |
| 3 | Pixel-level forward model (MCMC) | Model = Doppler-shifted atmosphere × telluric × (gas cell) ⊗ LSF × continuum. Jointly fit Teff, logg, vsini, RV, continuum, wavelength terms, LSF and telluric strength. Reject outliers and refit. Several epochs can share one model with a per-epoch RV. | Preferred for L/T dwarfs at R ≈ 3000–50 000. | Median 0.5 km/s, 37 T dwarfs (Hsu 2021). Median 0.4 km/s, 258 M6–L2 dwarfs, APOGEE (Hsu 2024). 5 km/s floor at JWST R~3000 (Hsu 2026). | Hsu 2021, 2024, 2026; Bean 2010 | core |
| 4 | Telluric-referenced RV (CH4 2.3 µm, O2 as fiducial) | Uses telluric lines in the same beam as the wavelength zero-point. | Slit spectrographs, K band, R ≈ 25 000–50 000. | ~300 m/s at R~50k. ~200 m/s for slowly rotating L dwarfs. 50 m/s for a bright, slow M standard. | Blake 2007, 2010 | core + cal |
| 5 | Least-squares deconvolution (LSD) | Solves for a mean line profile. | Best for line profiles, rotation and Doppler imaging (e.g. Luhman 16B). Less clean for absolute RV because of differing pressure broadening. | — | Crossfield 2014 | core (niche) |
| 6 | Single vs double template | Tests for unresolved binarity. | Apparent RV variability or composite line profiles. | — | — | diagnostic |
| 7 | Line centring on individual lines | The report discourages relying on single pressure-broadened alkali lines (K I wings span tens to hundreds of km/s). It recommends broad multi-feature fitting, framed as joint inference. | Lines that are not strongly pressure-broadened. | — | McLean 2007 | core (disfavoured) |
| 8 | Template choice: synthetic vs empirical | Synthetic templates give an absolute scale but carry opacity-driven RV offsets. Empirical templates transfer their own zero-point. | Always | BT-Settl vs Sonora: several tenths of km/s on average, more in some mid/late-T J-band cases | Hsu 2021 | core + err |
| 9 | LSF and vsini | Convolve the template with the instrument LSF and rotation. Fit them jointly, with a variable LSF where needed. | Always. A median vsini of 17 km/s is comparable to one resolution element at R = 22 500. | — | Hsu 2024; Bean 2010 | core |
| 10 | Telluric treatment | Mask opaque regions. Forward-model moderate absorption with H2O/CH4 column, shift and LSF as nuisance parameters. Avoid blind division by a telluric standard. | Ground-based NIR | 0.1–0.6 km/s wavelength residuals from tellurics | Hsu 2021; Figueira 2010/11 | cal |
| 11 | Telluric-line zero-point | Telluric lines as the wavelength reference. | Older slit spectrographs | O2: ~10 m/s raw, 1–2 m/s modelled (HARPS) | Figueira | cal |
| 12 | Arc lamps (ThAr, U-Ne) | Initial wavelength solution. | Optical/NIR | — | Instrument notes | cal (reduction) |
| 13 | Gas cells (I2, NH3) | Common-path calibration with a simultaneous LSF. | Rare for brown dwarfs | NH3: ~5 m/s over months on late-M dwarfs | Bean 2010 | cal (reduction) |
| 14 | Fabry–Pérot / laser comb | Tracks instrument drift. | CARMENES, SPIRou, CRIRES+ | SPIRou ~2 m/s; iSHELL ~10 m/s (bright stars) | Instrument pages | cal (reduction) |
| 15 | Barycentric correction | Reproducible relativistic correction, using mid-exposure time and coordinates. | Always | Coherent zero-point error if omitted | — | cal |
| 16 | Per-order RVs, then combination | Compare J vs K bands and individual orders. Inflate errors when the scatter exceeds the formal errors. Weighted mean. | Multi-order data | Modelling offsets of a few hundred m/s to km/s in T dwarfs | Hsu 2021 | err |
| 17 | Template-spread systematic | Compare grids, standards and orders, and include the spread in the error. | Always | See #8 | Hsu 2021 | err |
| 18 | Injection–recovery | Convolve with vsini and LSF, add noise and tellurics, and run the real code. | Planning and validation | — | — | err / validation |
| 19 | Epoch strategy and activity | Multi-epoch shared model. Bisector and line-profile checks, photometry, multi-band RVs. | Variable L/T dwarfs | LP 944-20: ~3.5 km/s optical variability vs 0.36 km/s rms in the NIR | Martín 2006 | err |
| 20 | Band/order selection | Choose regions by joint brown-dwarf and calibration information content, not by flux. | Planning | — | Reiners 2010; Hsu 2021 | cal |

### Instrument precision anchors (from the report)

| Instrument | R | Precision | Reference |
|---|---|---|---|
| MagE | ~4100 | 2–3 km/s (85 late-M/L dwarfs) | Burgasser 2015 |
| NIRSPEC | ~20 000 | ~2 km/s | Prato 2015 |
| PHOENIX | ~50 000 | ~0.3 km/s | — |
| UVES | ~40 000 | 40–670 m/s | — |

## Gap analysis against rvfitter v1

**Missing:**
- An actual CCF (#1) and pixel-level template matching (#2).
- A forward model (#3).
- LSF and vsini fitting (#9).
- Telluric masking or modelling (#10, #11).
- LSD (#5).
- Barycentric correction (#15).
- A binarity test (#6).
- Scatter inflation (#16).
- A template-spread systematic (#17).
- Injection–recovery (#18).
- Activity diagnostics (#19).
- Band selection (#20).

**Present but wrong or weak:**
- **v1 "cross-correlation":** the user nudges the template RV by eye. It is neither reproducible nor weighted, and its error (rvstep/2 + λ RMS) is not statistical. The λ RMS is a fair ingredient of a systematic term, but should not be the error itself.
- **Air wavelengths with c/n_air in the Doppler formula** *[outside report]*. The fix is to convert to vacuum and use c.
- **Single-line alkali centring:** acceptable at R~2000 *[outside report]* with core-only windows, a systematic floor and a scatter-inflated weighted mean.

**Present and fine:**
- Gaussian/Lorentzian/Voigt profile options.
- Weighted-mean combination, once scatter inflation is added.
- BT-Settl templates (standard but model-dependent).
- Resolution-matched template smoothing.

## Recommended defaults per regime (our synthesis)

| Regime | Primary | Secondary / notes |
|---|---|---|
| Optical R~2000 (e.g. GTC/OSIRIS R2500I) | Template fit + CCF with vsini fixed (or 0) | Line centring on line cores with a scatter-inflated error. λ-solution floor. O2 A-band telluric mask. |
| NIR R~2000 (SpeX) | Forward model or template fit across J/K | Tellurics dominate: mask or model them. |
| R~6000 | Forward model with free RV and vsini | — |
| R>20 000 | Full forward model (RV, vsini, Teff, logg, LSF, telluric scale/shift, continuum, λ offset) | Per-order RVs with inflation. Multi-epoch shared model. |

In all regimes:
- Apply the barycentric correction (unless the spectrum is already corrected).
- Report precision and accuracy separately.
- Report the statistical, template-spread and λ zero-point terms separately, with a quadrature total.

## Validation strategy

- Repeated observations of a stable standard test repeatability. Objects with independent RVs test the absolute zero-point.
- Run injection–recovery on the actual target type with the real code.
- Compare different atmosphere grids (BT-Settl vs Sonora) and individual orders/bands.
- Plot RV against airmass and PWV.
- Compare with literature catalogues: Blake 2010, Burgasser 2015, Prato 2015, Hsu 2021, 2024. Gaia, APOGEE and Cooper 2024 overlaps are *[outside report]*.
- **Primary benchmark for rvfitter: J1745−1640** (DENIS J174534.6−164053, L1.5). See `docs/methods.md` for the compiled literature RVs and the acceptance criteria.

## Verified references

- Blake, Charbonneau & White 2010, ApJ 723, 684 (arXiv:1008.3874)
- Burgasser et al. 2015, ApJS 220, 18
- Hsu et al. 2021, ApJS 257, 45 (arXiv:2107.01222)
- Cooper et al. 2024, MNRAS 534, 695 (arXiv:2409.03706)
- Phan-Bao et al. 2008, MNRAS 383, 831 (discovery of J1745−1640)

Other references are as named in the report and have not been independently verified here.
