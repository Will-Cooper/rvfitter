# RV Fitting Processes
This codebase is designed for determining radial
velocities.
It's been written broadly such that one can use
the two packages: `splot` and `xcorr`.
The point of this is to recreate older codes such as
IDL's `Gaussfit.pro` or IRAF's `splot` in python.

## Splot
Line center fitting.
The control module of `linecentering.linecentering`
can be imported to automatically or manually fit line
centers.

## Xcorr
Cross correlation.
The control module of `crosscorrelate.crosscorrelate`
can be imported to automatically or manually fit cross
correlate.

### Example
An example script is given, showing the use of `crosscorrelate`
and `linecentering`, in `fullrv`.
It's the only script given set up for a specific use-case.

## Installing
Clone this repository locally with
```bash
git clone https://github.com/Will-Cooper/rvfitter.git
conda env create -f environment.yml
```
The `fullrv` code requires the `splat` package,
follow the installation instructions
[there](https://github.com/aburgasser/splat).

### Citing
[![DOI](https://zenodo.org/badge/498288692.svg)](https://zenodo.org/badge/latestdoi/498288692)
