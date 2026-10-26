from astropy import log as astropy_log
from astropy.io.fits import getdata
from astropy.modeling import Fittable1DModel
from astropy.nddata import StdDevUncertainty
import astropy.units as u
from matplotlib import rcParams
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from copy import copy
import json
import os
from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import extract_region
from specutils.utils.wcs_utils import vac_to_air
from typing import Union, Optional, List, Dict, Tuple, Sequence


class Quantiser:
    """
    Underlying RV fitting class
    """

    def __init__(self, wunit: u.Quantity, funit: u.Quantity, rvunit: u.Quantity, spec: Spectrum1D):
        """
        When initialising the line centering programme

        Parameters
        ----------
        wunit
            The wavelength unit
        funit
            The flux unit
        rvunit
            The radial velocity unit
        spec
            The spectrum of the object
        """
        self.wunit = wunit
        self.funit = funit
        self.rvunit = rvunit
        self.c1 = None
        self.c2 = None
        self.c3 = None
        self.c4 = None
        self.r1 = None
        self.r2 = None
        self.iscut = False
        self.rescale = True
        self.best_rmsdiqr = np.inf
        self.snr = 0
        self.rvunc_min = np.nan
        self.spec = spec
        self.sub_spec = self.spec
        self.cont = None
        self.contwindow = self.getcontwindow()
        self.linewindow = self.getlinewindow()

    def __assertwavelength__(self, value: Optional[Union[float, u.Quantity]]) -> Optional[u.Quantity]:
        if isinstance(value, float) or isinstance(value, int):
            value *= self.wunit
        elif not self.__assertquantity__(value, True):
            pass
        return value

    def __assertflux__(self, value: Optional[Union[float, u.Quantity]]) -> Optional[u.Quantity]:
        if isinstance(value, float) or isinstance(value, int):
            value *= self.funit
        elif not self.__assertquantity__(value, True):
            pass
        return value

    def __assertrv__(self, value: Optional[Union[float, u.Quantity]]) -> Optional[u.Quantity]:
        if isinstance(value, float) or isinstance(value, int):
            value *= self.rvunit
        elif not self.__assertquantity__(value, True):
            pass
        return value

    @staticmethod
    def __assertquantity__(value: Optional[u.Quantity], optional: bool):
        if optional and value is not None and not isinstance(value, u.Quantity):
            raise AttributeError(f'Value {value} must be numeric or a quantity (or None)')
        elif not optional and not (isinstance(value, u.CompositeUnit) or isinstance(value, u.Unit) or
                                   isinstance(value, u.IrreducibleUnit)):
            print(value, type(value))
            raise AttributeError(f'Value {value} must be a unit')
        return True

    @staticmethod
    def __assertmodel__(value: Optional[Fittable1DModel]) -> Optional[Fittable1DModel]:
        if value is not None and not isinstance(value, Fittable1DModel):
            raise AttributeError(f'{value} needs to be a Fittable1DModel (astropy)')
        return value

    @property
    def wunit(self) -> u.Unit:
        return self._wunit

    @wunit.setter
    def wunit(self, value):
        if not self.__assertquantity__(value, False):
            raise AttributeError('wunit must be an astropy unit')
        self._wunit = value

    @property
    def funit(self) -> u.CompositeUnit:
        return self._funit

    @funit.setter
    def funit(self, value):
        if not self.__assertquantity__(value, False):
            raise AttributeError('funit must be an astropy unit')
        self._funit = value

    @property
    def rvunit(self) -> u.CompositeUnit:
        return self._rvunit

    @rvunit.setter
    def rvunit(self, value):
        if not self.__assertquantity__(value, False):
            raise AttributeError('rvunit must be an astropy unit')
        self._rvunit = value

    @property
    def c1(self) -> Optional[u.Quantity]:
        return self._c1

    @c1.setter
    def c1(self, value: Union[float, u.Quantity]):
        self._c1 = self.__assertwavelength__(value)

    @property
    def c2(self) -> Optional[u.Quantity]:
        return self._c2

    @c2.setter
    def c2(self, value: Union[float, u.Quantity]):
        self._c2 = self.__assertwavelength__(value)

    @property
    def c3(self) -> Optional[u.Quantity]:
        return self._c3

    @c3.setter
    def c3(self, value: Union[float, u.Quantity]):
        self._c3 = self.__assertwavelength__(value)

    @property
    def c4(self) -> Optional[u.Quantity]:
        return self._c4

    @c4.setter
    def c4(self, value: Union[float, u.Quantity]):
        self._c4 = self.__assertwavelength__(value)

    @property
    def r1(self) -> Optional[u.Quantity]:
        return self._r1

    @r1.setter
    def r1(self, value: Union[float, u.Quantity]):
        self._r1 = self.__assertwavelength__(value)

    @property
    def r2(self) -> Optional[u.Quantity]:
        return self._r2

    @r2.setter
    def r2(self, value: Union[float, u.Quantity]):
        self._r2 = self.__assertwavelength__(value)

    @property
    def cont(self) -> Optional[Fittable1DModel]:
        return self._cont

    @cont.setter
    def cont(self, value):
        self._cont = self.__assertmodel__(value)

    def __checkcontpoints__(self) -> bool:
        if any([cpoint is None for cpoint in (self.c1, self.c2, self.c3, self.c4)]):
            return False
        return True

    def __checkregionpoints__(self) -> bool:
        if any([rpoint is None for rpoint in (self.r1, self.r2)]):
            return False
        return True

    def getcontwindow(self) -> Optional[Tuple[List[u.Quantity], List[u.Quantity]]]:
        """
        Retrieving the continuum window

        Returns
        -------
        _
            The collection of continuum points
        """
        if not self.__checkcontpoints__():
            return None
        return [self.c1, self.c2], [self.c3, self.c4]

    def getlinewindow(self) -> Optional[SpectralRegion]:
        """
        Retrieving the region with the spectral line

        Returns
        -------
        _
            The spectral region for the line
        """
        if not self.__checkregionpoints__():
            return None
        else:
            return SpectralRegion(self.r1, self.r2)

    def cutspec(self, spec: Spectrum1D) -> Spectrum1D:
        """
        Cutting the spectrum to the left and right-most points

        Parameters
        ----------
        spec
            The spectrum to be cut
        Returns
        -------
        spec
            The spectrum cut
        """
        x1, x2 = self.c1, self.c4
        spec = copy(spec)
        if not any([xpoint is None for xpoint in (x1, x2)]):
            xreg = SpectralRegion(x1, x2)
            sub_spec = extract_region(spec, xreg)
            self.iscut = True
            return sub_spec
        return spec

    @staticmethod
    def poly_cutter(wave: np.ndarray, flux: np.ndarray, fluxerr: np.ndarray = None,
                    polycoeff: int = 5, bign: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creating a interpolation of a given order

        Parameters
        ----------
        wave
            The array of wavelengths
        flux
            The array of fluxes
        fluxerr
            The array of flux errors
        polycoeff
            The coefficient to use when interpolating
        bign
            The number by which to multiply the number of data points by

        Returns
        -------
        x, y, yerr
            The array of wavelengths
            The array of fluxes
            The array of flux errors
        """
        if bign is None:
            bign = len(wave) * 10
        x = np.linspace(np.min(wave), np.max(wave), bign)
        p = interp1d(wave, flux, kind=polycoeff)
        y = p(x)
        if fluxerr is not None:
            perr = interp1d(wave, fluxerr, kind=polycoeff)
            yerr = perr(x)
        else:
            yerr = np.empty_like(flux)
        return x, y, yerr

    def __updatewindows__(self):
        """
        Updating the windows for cutting spectra by the continuum and line window
        """
        self.sub_spec = self.cutspec(self.spec)
        self.contwindow = self.getcontwindow()
        self.linewindow = self.getlinewindow()


def inv_rv_calc(shift: float, wave: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Inverse RV shift

    Parameters
    ----------
    shift
        The shift in RV
    wave
        The wavelength values to shift

    Returns
    -------
    _
        The shifted wavelengths
    """
    c = 299792458 / 1e3  # vacuum speed of light in km/s, valid for air or vacuum wavelengths
    return shift * wave / c


def wave_to_rv(shift: Union[float, np.ndarray], wave: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Converting a wavelength shift (or uncertainty) into a velocity, c * shift / wave

    Parameters
    ----------
    shift
        The wavelength shift
    wave
        The wavelength at which the shift is measured, in the same unit as shift

    Returns
    -------
    _
        The velocity in km/s
    """
    c = 299792458 / 1e3  # vacuum speed of light in km/s
    return c * shift / wave


def json_handle(jf: Union[os.PathLike, str, bytes],
                d: Dict[str, Dict[str, List[Union[float, str, bool]]]] = None) \
        -> Dict[str, Dict[str, List[Union[float, str, bool]]]]:
    """
    Handling a .json, either saving or loading

    Parameters
    ----------
    jf
        The filename of the .json
    d
        The dictionary of data to be saved or loaded to the json

    Returns
    -------
    d
        The dictionary of data to be saved or loaded to the json
    """
    if not os.path.exists(jf):
        d = {}
    if d is None:
        with open(jf, 'r') as jd:
            d = json.load(jd)
    else:
        with open(jf, 'w') as jd:
            json.dump(d, jd)
    return d


def sigma_clipper(wave: np.ndarray, flux: np.ndarray, fluxerr: np.ndarray,
                  sigma: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sigma clip the spectra by the sigma from the median of entire flux array

    Parameters
    ----------
    wave
        Wavelength array
    flux
        Flux array
    fluxerr
        Flux error array
    sigma
        Number of sigma to determine as outliers

    Returns
    -------
    wave
        Wavelength array, cut
    flux
        Flux array, cut
    fluxerr
        Flux error array, cut
    """
    med = np.median(flux)
    std = np.std(flux)
    boolcut: np.ndarray = (flux > med - sigma * std) & (flux < med + sigma * std)

    if len(flux[boolcut]) < int(0.95 * len(flux)):  # 95% of original data
        return wave, flux, fluxerr

    wave = wave[boolcut]
    flux = flux[boolcut]
    fluxerr = fluxerr[boolcut]
    return wave, flux, fluxerr


def normaliser(x: np.ndarray, *args, xmin: float = 8100, xmax: float = 8200):
    """
    Normalising a flux from a given wavelength

    Parameters
    ----------
    x
        Wavelength array
    args
        The arguments to be normalised
    xmin
        The minimum wavelength
    xmax
        The maximum wavelength

    Returns
    -------
    out
        The list of arrays starting with wavelength (the inputs are not modified)

    Notes
    -----
    The arguments are treated as plain numpy arrays: ``np.asarray`` strips any astropy Quantity units,
    so the normalised arrays are returned as unitless ndarrays.
    """
    boolcut: np.ndarray = (x > xmin) & (x < xmax)
    if np.any([len(x) != len(arg) for arg in args]):
        raise IndexError('check input shapes')
    normval = np.nanmedian(np.asarray(args[0])[boolcut])
    # build new arrays rather than dividing in place, so the caller's arrays are left untouched
    out = [x, ] + [np.asarray(arg) / normval for arg in args]
    return out


def ivar_to_sigma(ivar: np.ndarray) -> np.ndarray:
    """
    Converting an inverse variance array into a standard deviation array

    Parameters
    ----------
    ivar
        The inverse variance array

    Returns
    -------
    sigma
        The standard deviation, 1 / sqrt(ivar), with NaN wherever ivar is not finite and positive
    """
    ivar = np.asarray(ivar, dtype=float)
    good = np.isfinite(ivar) & (ivar > 0)
    sigma = np.full_like(ivar, np.nan)
    np.divide(1., np.sqrt(ivar, out=np.zeros_like(ivar), where=good), out=sigma, where=good)
    return sigma


def freader(f: str, **kwargs) -> Spectrum1D:
    """
    Reading a file out into a spectrum

    Parameters
    ----------
    f
        The filename
    kwargs
        Extra parameters

    Returns
    -------
    spec
        The spectrum from the filename
    """
    wavearr: Optional[np.ndarray] = kwargs.get('wavearr', None)  # the wavelength array to interpolate for
    wunit = kwargs.get('wunit', u.AA)
    funit = kwargs.get('funit', u.erg / u.cm ** 2 / wunit / u.s)
    if f.endswith('txt'):
        try:
            wave, flux = np.loadtxt(f, unpack=True, usecols=(0, 1))  # load file
        except (OSError, FileNotFoundError) as e:
            raise FileNotFoundError(f'Cannot find given file in: {f}') from e
        except ValueError:
            wave, flux = np.loadtxt(f, unpack=True, usecols=(0, 1), skiprows=1)  # load file
        if wavearr is not None:
            flux = np.interp(wavearr, wave, flux)
            wave = wavearr
        fluxerr = np.zeros_like(flux)
    else:  # fits
        target = getdata(f)
        try:
            wave = target.wave
            wave, uniquebool = np.unique(wave, return_index=True)  # check there aren't duplicate wavelengths
            wave = np.array(vac_to_air(wave * wunit, method='Edlen1953') / wunit)
            flux = target.flux[uniquebool]
            fluxivar = target.ivar[uniquebool]
            fluxerr = ivar_to_sigma(fluxivar)  # sigma = 1 / sqrt(ivar); NaN where ivar <= 0
        except AttributeError:
            wave = target[0]
            wave: np.ndarray = np.array(vac_to_air(wave * wunit, method='Edlen1953') / wunit)
            flux = target[1]
            fluxerr = target[2]
    # remove pixels without a usable flux or error (e.g. ivar <= 0, masked by the pipeline), for every input type
    boolcut = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(fluxerr)
    wave = wave[boolcut]
    flux = flux[boolcut]
    fluxerr = fluxerr[boolcut]
    wave, flux, fluxerr = sigma_clipper(wave, flux, fluxerr, sigma=5)
    wave, flux, fluxerr = normaliser(wave, flux, fluxerr, xmin=np.min(wave), xmax=np.max(wave))
    unc = StdDevUncertainty(fluxerr, unit=funit)
    spec = Spectrum1D(flux * funit, wave * wunit,
                      uncertainty=unc)
    return spec


def rmsdiqr_check(observed: np.ndarray, expected: np.ndarray, best: float) -> Tuple[float, bool]:
    """
    Calculate the rmsdiqr of two distributions

    Parameters
    ----------
    observed
        Observed values
    expected
        Expected values
    best
        Current best value

    Returns
    -------
    chi
        Chisquare value
    significant
        Switch whether value is significant
    """
    rmsd = np.sqrt(np.sum((observed - expected) ** 2) / len(observed))
    iqr = np.subtract(*np.percentile(observed, [75, 25]))
    rmsdiqr = rmsd / iqr
    if rmsdiqr <= best:
        significant = True
    else:
        significant = False
    return rmsdiqr, significant


def get_snr_and_rvunc(spec: Spectrum1D, continuum_region: SpectralRegion,
                      fitting_region: SpectralRegion, fwhm: u.Quantity) -> Tuple[float, float]:
    """
    Finds the SNR for a given region and the minimum RV uncertainty based on that

    Parameters
    ----------
    spec
        The spectrum object
    continuum_region
        The continuum region not used in the line fitting
    fitting_region
        The region used in line fitting
    fwhm
        FWHM of the fitted (working) profile, in the same unit as the spectral axis

    Returns
    -------
    snr_value
        Signal to noise ratio
    rvunc_min
        Minimum rv uncertainty from resolution, sigma = c / (R * SNR * sqrt(Npix)) with R = lambda / FWHM;
        NaN if the FWHM is not positive and finite
    """
    line_reg: Spectrum1D = extract_region(spec, fitting_region)
    n_pix = len(line_reg.spectral_axis.value)
    cont_reg: Spectrum1D = extract_region(spec, continuum_region, return_single_spectrum=True)
    if getattr(fwhm, 'unit', None) is not None and hasattr(fwhm, 'quantity'):  # astropy model Parameter
        fwhm = fwhm.quantity
    fwhm_value = fwhm.to(spec.spectral_axis.unit).value if isinstance(fwhm, u.Quantity) else float(fwhm)
    snr_value = np.mean(cont_reg.flux.value) / np.std(cont_reg.flux.value)
    if not np.isfinite(fwhm_value) or fwhm_value <= 0 or n_pix < 1:
        return snr_value, np.nan
    resolution = np.nanmean(line_reg.spectral_axis.value) / fwhm_value
    c = 299792458 / 1e3  # km/s
    rvunc_min = c / (resolution * snr_value * np.sqrt(n_pix))
    return snr_value, rvunc_min


def combine_rvs(rvs: np.ndarray, errs: np.ndarray) -> Tuple[float, float]:
    """
    Combining several RV measurements into an inverse-variance weighted mean and its uncertainty

    Parameters
    ----------
    rvs
        The radial velocities (NaN entries are ignored)
    errs
        The uncertainties on those radial velocities

    Returns
    -------
    rv, err
        The weighted mean RV
        Its uncertainty: the measurement's own error for a single value, otherwise the formal error
        1 / sqrt(sum(w)) inflated by sqrt(chi^2 / (N - 1)) when the values scatter more than their errors allow
        (chi^2 = sum(w * (v - mean)^2)), and never deflated below the formal error
    """
    rvs, errs = np.asarray(rvs, dtype=float), np.asarray(errs, dtype=float)
    good = np.isfinite(rvs) & np.isfinite(errs) & (errs > 0)
    rvs, errs = rvs[good], errs[good]
    if not len(rvs):
        return np.nan, np.nan
    if len(rvs) == 1:
        return float(rvs[0]), float(errs[0])
    weights = 1.0 / errs ** 2
    rv = np.average(rvs, weights=weights)
    formal = 1.0 / np.sqrt(np.sum(weights))
    chisq = np.sum(weights * (rvs - rv) ** 2)
    return float(rv), float(formal * max(1., np.sqrt(chisq / (len(rvs) - 1))))


def _half_cauchy_logpdf(tau: np.ndarray, scale: float) -> np.ndarray:
    """
    The log probability density of a half-Cauchy distribution of the given scale, for tau >= 0
    """
    return np.log(2. / (np.pi * scale)) - np.log1p((tau / scale) ** 2)


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """
    Trapezoidal quadrature weights for an increasing, possibly non-uniform, grid
    """
    w = np.zeros_like(x)
    dx = np.diff(x)
    w[:-1] += dx / 2
    w[1:] += dx / 2
    return w


def mixture_pdf(x: np.ndarray, means: np.ndarray, sds: np.ndarray, weights: np.ndarray,
                extra_sd: float = 0.) -> np.ndarray:
    """
    The probability density of a mixture of Gaussians

    Parameters
    ----------
    x
        Where to evaluate the density
    means, sds, weights
        The component means, standard deviations and (normalised) weights
    extra_sd
        An extra standard deviation added in quadrature to every component (e.g. a systematic)

    Returns
    -------
    _
        The density at x
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    sds = np.sqrt(sds ** 2 + extra_sd ** 2)
    keep = weights > 1e-12 * np.max(weights)
    m, s, w = means[keep], sds[keep], weights[keep]
    z = (x[:, None] - m[None, :]) / s[None, :]
    return np.sum(w * np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * s), axis=1)


def hierarchical_combine(values: np.ndarray, errors: np.ndarray, line_ids: Sequence, method_ids: Sequence,
                         tau_scale: Optional[float] = None, ngrid: int = 200, tau_max_factor: float = 10.,
                         tau_line: Optional[float] = None, tau_meth: Optional[float] = None) -> dict:
    """
    Combining RV measurements of several lines by several methods with a hierarchical line x method model

    The model is v_{m,i} = mu + delta_i + beta_m + eps_{m,i}, with delta_i ~ N(0, tau_line^2) shared by all
    methods measuring line i, beta_m ~ N(0, tau_meth^2) shared by all lines measured by method m, and
    eps_{m,i} ~ N(0, sigma_{m,i}^2).
    Marginalising delta and beta analytically gives v ~ N(mu 1, C), with
    C = diag(sigma^2) + tau_line^2 [same line] + tau_meth^2 [same method].
    For fixed (tau_line, tau_meth) and a flat prior on mu, the posterior of mu is Gaussian with mean
    1^T C^-1 v / 1^T C^-1 1 and variance 1 / 1^T C^-1 1, and the marginal likelihood of the taus is
    p(v | tau) ∝ |C|^-1/2 (1^T C^-1 1)^-1/2 exp(-(v^T C^-1 v - (1^T C^-1 v)^2 / 1^T C^-1 1) / 2).
    Both taus have half-Cauchy priors and are integrated deterministically on a fixed grid (tau = 0 plus
    ngrid - 1 log-spaced points from 1e-3 times the scale up to tau_max, trapezoidal weights), so the
    posterior of mu is a mixture of Gaussians. There is no random number generation.
    tau_max is the larger of tau_max_factor times the scale and three times the full range of the values, so
    the grid covers the likelihood whatever prior scale is chosen.

    Parameters
    ----------
    values
        The RV measurements (non-finite entries, or those with non-finite or non-positive errors, are dropped)
    errors
        Their uncertainties
    line_ids
        The line each measurement belongs to (any hashable labels)
    method_ids
        The method each measurement comes from (any hashable labels)
    tau_scale
        The scale of the half-Cauchy priors on both taus; by default the median measurement uncertainty
    ngrid
        The number of grid points per tau axis (including tau = 0)
    tau_max_factor
        The upper end of the tau grid is at least this many times tau_scale (the prior is truncated there)
    tau_line, tau_meth
        If given, that tau is fixed at this value instead of being integrated over

    Returns
    -------
    result
        A dictionary with the posterior mean and sd of mu ('mean', 'sd'), its 'median', 'q16' and 'q84',
        the posterior medians of the taus ('tau_line', 'tau_meth'), 'tau_scale', the number of measurements,
        lines and methods ('n', 'nlines', 'nmethods') and the mixture components ('means', 'sds', 'weights')
    """
    from scipy.optimize import brentq
    from scipy.special import ndtr

    values, errors = np.asarray(values, dtype=float), np.asarray(errors, dtype=float)
    line_ids, method_ids = np.asarray(line_ids, dtype=object), np.asarray(method_ids, dtype=object)
    good = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    values, errors, line_ids, method_ids = values[good], errors[good], line_ids[good], method_ids[good]
    n = len(values)
    out = dict(mean=np.nan, sd=np.nan, median=np.nan, q16=np.nan, q84=np.nan, tau_line=np.nan,
               tau_meth=np.nan, tau_scale=np.nan, n=n, nlines=len(set(line_ids)),
               nmethods=len(set(method_ids)), means=np.array([]), sds=np.array([]), weights=np.array([]))
    if not n:
        return out
    if tau_scale is None:
        tau_scale = float(np.median(errors))
    out['tau_scale'] = tau_scale
    tau_max = max(tau_max_factor * tau_scale, 3. * float(np.ptp(values)))

    def taugrid(fixed):
        if fixed is not None:
            return np.array([float(fixed)]), np.zeros(1), np.ones(1)
        grid = np.concatenate([[0.], np.logspace(np.log10(1e-3 * tau_scale),
                                                 np.log10(tau_max), ngrid - 1)])
        return grid, _half_cauchy_logpdf(grid, tau_scale), _trapezoid_weights(grid)

    tl, logprior_l, quad_l = taugrid(tau_line)
    tm, logprior_m, quad_m = taugrid(tau_meth)
    sameline = (line_ids[:, None] == line_ids[None, :]).astype(float)
    samemeth = (method_ids[:, None] == method_ids[None, :]).astype(float)
    cov = (np.diag(errors ** 2)[None, None] + (tl ** 2)[:, None, None, None] * sameline[None, None]
           + (tm ** 2)[None, :, None, None] * samemeth[None, None])  # (nl, nm, n, n)
    rhs = np.broadcast_to(np.stack([np.ones(n), values], axis=1), cov.shape[:2] + (n, 2))
    sol = np.linalg.solve(cov, rhs)  # columns: C^-1 1, C^-1 v
    a = sol[..., 0].sum(axis=-1)  # 1^T C^-1 1
    b = sol[..., 1].sum(axis=-1)  # 1^T C^-1 v
    vcv = np.einsum('i,...i->...', values, sol[..., 1])  # v^T C^-1 v
    _, logdet = np.linalg.slogdet(cov)
    loglike = -0.5 * (logdet + np.log(a) + vcv - b ** 2 / a)
    with np.errstate(divide='ignore'):  # zero quadrature weight at the end of a grid
        logpost = (loglike + (logprior_l + np.log(quad_l))[:, None] + (logprior_m + np.log(quad_m))[None, :])
    weights = np.exp(logpost - np.max(logpost))
    weights /= weights.sum()
    means, sds = b / a, np.sqrt(1. / a)

    mean = float(np.sum(weights * means))
    sd = float(np.sqrt(np.sum(weights * (sds ** 2 + means ** 2)) - mean ** 2))
    keep = weights > 1e-12 * weights.max()
    km, ks, kw = means[keep], sds[keep], weights[keep] / weights[keep].sum()

    def quantile(q):
        lo, hi = np.min(km - 10 * ks), np.max(km + 10 * ks)
        return brentq(lambda x: np.sum(kw * ndtr((x - km) / ks)) - q, lo, hi, xtol=1e-8)

    def tau_median(grid, marginal):
        if len(grid) == 1:
            return float(grid[0])
        cdf = np.cumsum(marginal)
        return float(np.interp(0.5, cdf, grid))

    out.update(mean=mean, sd=sd, median=quantile(0.5), q16=quantile(0.15865525393145707),
               q84=quantile(0.8413447460685429), tau_line=tau_median(tl, weights.sum(axis=1)),
               tau_meth=tau_median(tm, weights.sum(axis=0)), means=means.ravel(), sds=sds.ravel(),
               weights=weights.ravel())
    return out


def logging_rvcalc(s: str = '', perm: str = 'a'):
    """
    Logging the information to a file

    Parameters
    ----------
    s
        The string being saved
    perm
        The permission of the file opening
    """
    if not os.path.exists('calculating.log'):
        perm = 'w'
    with open('calculating.log', perm) as f:
        f.write(s + '\n')
    return


def spec_unpack(spec: Spectrum1D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpacking a spectrum object into wavelength, flux and flux error

    Parameters
    ----------
    spec
        The spectrum to be unpacked

    Returns
    -------
    wave, flux, fluxerr
        The arrays of wavelength, flux and flux error
    """
    wave = copy(spec.spectral_axis.value)
    flux = copy(spec.flux.value)
    fluxerr = copy(spec.uncertainty.quantity.value)
    return wave, flux, fluxerr


def stephens(s: Union[pd.Series, float]) -> Union[np.ndarray, float]:
    """
    Stephens relation for converting to teff

    Parameters
    ----------
    s
        The spectral type number for the Stephens relation

    Returns
    -------
    teff
        The effective temperature converted
    """
    teff = 4400.9 - 467.26 * s + 54.67 * s ** 2 - 4.4727 * s ** 3 + 0.17667 * s ** 4 - 0.0025492 * s ** 5
    teff = np.where((1200 < teff) & (teff < 4000), teff, np.nan)
    return teff


astropy_log.setLevel('ERROR')
dpi = 200  # 200-300 as per guidelines
maxpix = 670  # max pixels of plot
width = maxpix / dpi  # max allowed with
rcParams.update({'axes.labelsize': 'large', 'axes.titlesize': 'large',  # the size of labels and title
                 'xtick.labelsize': 'large', 'ytick.labelsize': 'large',  # the size of the axes ticks
                 'legend.fontsize': 'large', 'legend.frameon': False,  # legend font size, no frame
                 'legend.facecolor': 'none', 'legend.handletextpad': 0.25,
                 # legend no background colour, separation from label to point
                 'font.serif': ['Computer Modern', 'Helvetica', 'Arial',  # default fonts to try and use
                                'Tahoma', 'Lucida Grande', 'DejaVu Sans'],
                 'font.family': 'serif',  # use serif fonts
                 'mathtext.fontset': 'cm', 'mathtext.default': 'regular',  # if in math mode, use these
                 'figure.figsize': [width, 0.7 * width], 'figure.dpi': dpi,
                 # the figure size in inches and dots per inch
                 'lines.linewidth': .75,  # width of plotted lines
                 'xtick.top': True, 'ytick.right': True,  # ticks on right and top of plot
                 'xtick.minor.visible': True, 'ytick.minor.visible': True,  # show minor ticks
                 'text.usetex': True})  # process text with LaTeX instead of matplotlib math mode
