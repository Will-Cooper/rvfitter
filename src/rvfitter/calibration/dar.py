"""
Atmospheric differential refraction (DAR) across the slit, as a velocity.

The atmosphere displaces the star's image towards the zenith by the refraction R(λ), larger in the blue, whilst
the sky fills the slit and is not displaced. The component across the slit moves the star's light along the
dispersion, so the star − sky zero point varies with wavelength as

    Δv(λ) = a + s · D(λ),   D(λ) = c · g · [R(λ) − R(λ_ref)] / s_spec · d(λ) / λ,

with s = sin(PA_slit − q) the signed sine of the angle between the slit and the parallactic angle q (both east of
north), PA_slit = PA_disp + 90° where PA_disp is the sky PA of the direction of increasing wavelength across the
slit, s_spec the plate scale along the dispersion [arcsec/pix], d(λ) the dispersion [Å/pix] and g = dc/dx the
gain of the clipped profile's centroid at the slit centre (`slit.centroid_gain`; 1 for a star much smaller than
the slit). R(λ) is ERFA's refraction model (`erfa.refco`, Hohenkerk & Sinclair) for the site's pressure,
temperature and humidity, and λ_ref is a reference wavelength (the constant a absorbs the choice).

PA_disp comes from the raw frame's header through `DISPERSION_PA` (per instrument), and q, the zenith distance and
their change over the exposure from astropy (hour angle, declination and the site). s is the mean over the
exposure of s · tan z, divided by tan z at mid-exposure, and its error the RMS of that product's drift.
"""
import logging
from collections.abc import Callable, Sequence
from typing import Any

import erfa
import numpy as np

from ..constants import C_KMS
from . import slit as slit_mod

__all__ = ['DISPERSION_PA', 'refraction_arcsec', 'parallactic_deg', 'geometry', 'dar_kms']

log = logging.getLogger(__name__)

#: Standard atmosphere when the header has no weather: temperature [°C] and relative humidity [0-1]; the pressure
#: comes from the site altitude (scale height 8.4 km). DAR scales with pressure / temperature, so these set D(λ)
#: to ≈ 5 per cent
STANDARD_TEMP_C = 10.
STANDARD_RH = 0.5
_SCALE_HEIGHT_M = 8400.


def _osiris_dispersion_pa(h: dict) -> float:
    """
    GTC/OSIRIS (Nasmyth, before OSIRIS+): the raw header's PA is the sky PA (east of north) of the detector's +Y
    axis, which is the dispersion direction with the wavelength increasing along +Y, and the long slit lies along X
    """
    # Sources: (1) the raw header's own WCS: the CD matrix puts the +Y axis at PA = header PA (e.g. J1745's frame
    # 0000809229: CD1_2 = -5.71e-5, CD2_2 = +4.16e-5 deg/pix -> PA -53.92 deg; Ross 640's 0000809220 -> +4.99 deg),
    # and IPA = IPA0 + PA; (2) the OSIRIS User Manual v3.0 §1.1.6 and §9.2 (IPA0 ≈ 150.54 deg gives North up, East
    # left) and §6 (the spectral direction is the detector's vertical, the spatial offsets ΔX); (3) PypeIt's
    # gtc_osiris.py (GTCOSIRISSpectrograph: specaxis = 0, specflip = False, i.e. the wavelength increases with
    # the raw row number Y). The manual's §1.1.6 sentence "IPA = 150.54 - P.A." has the opposite sign of the
    # header relation; the header is preferred because PA + 90 deg (the slit) matches the parallactic angle at
    # acquisition for every GTC54-15A0 / GTC8-15ITP target and standard in the test data, as observers request.
    return float(h['PA'])


#: Instrument (raw INSTRUME, lower case) → sky PA [deg, east of north] of increasing wavelength across the slit
DISPERSION_PA: dict[str, Callable[[dict], float]] = {'osiris': _osiris_dispersion_pa}


def refraction_arcsec(wave: np.ndarray | float, zenith_deg: float, pressure_hpa: float, temp_c: float, rh: float
                      ) -> np.ndarray:
    """Refraction [arcsec] at vacuum wavelength `wave` [Å] and zenith distance `zenith_deg` (ERFA refco)"""
    refa, refb = erfa.refco(pressure_hpa, temp_c, rh, np.asarray(wave, float) / 1e4)
    t = np.tan(np.radians(zenith_deg))
    return np.degrees(refa * t + refb * t ** 3) * 3600.


def parallactic_deg(ha_rad: np.ndarray, dec_rad: float, lat_rad: float) -> np.ndarray:
    """The parallactic angle q [deg, east of north: the PA of the zenith seen from the star]"""
    return np.degrees(np.arctan2(np.sin(ha_rad), np.tan(lat_rad) * np.cos(dec_rad) - np.sin(dec_rad) * np.cos(ha_rad)))


def _weather(h: dict, meta: dict, loc: Any) -> tuple[float, float, float, str]:
    """(pressure [hPa], temperature [°C], relative humidity [0-1], source) from the raw header, else standard"""
    p, t, rh = (h.get(k) for k in ('PRESSURE', 'TAMBIENT', 'HUMIDITY'))
    if all(v is not None and np.isfinite(float(v)) for v in (p, t, rh)):
        return float(p), float(t), float(rh) / 100., 'raw header PRESSURE, TAMBIENT, HUMIDITY'
    alt = float(loc.height.to_value('m')) if loc is not None else 0.
    p = 1013.25 * np.exp(-alt / _SCALE_HEIGHT_M)
    log.warning('%s: no weather in the raw header: DAR uses a standard atmosphere (%.0f hPa at %.0f m, %.0f °C, '
                'humidity %.0f per cent)', meta.get('object', ''), p, alt, STANDARD_TEMP_C, 100. * STANDARD_RH)
    return p, STANDARD_TEMP_C, STANDARD_RH, f'standard atmosphere at {alt:.0f} m'


def geometry(raw_headers: Sequence[dict], meta: dict) -> dict[str, Any]:
    """
    The DAR geometry of a spectrum: {'s', 's_err', 'pa_known', 'zenith_deg', 'pressure_hpa', 'temp_c', 'rh',
    'q_deg', 'pa_disp_deg', 'source'}

    One entry per exposure in `raw_headers` (primary-header cards); without them, or for an instrument missing from
    `DISPERSION_PA`, the PA is unknown (`pa_known` False, s NaN) and the zenith distance comes from meta['airmass'].
    'zenith_deg' is NaN when neither is available.
    """
    from astropy import units as u
    from astropy.coordinates import AltAz, SkyCoord
    from astropy.time import Time

    from ..barycentric import _location
    nan = float('nan')
    loc = _location(meta)
    out: dict[str, Any] = dict(s=nan, s_err=nan, pa_known=False, zenith_deg=nan, pressure_hpa=nan, temp_c=nan,
                               rh=nan, q_deg=[], pa_disp_deg=None, source='')
    airmass = meta.get('airmass')
    usable = [h for h in raw_headers if str(h.get('INSTRUME', '')).strip().lower() in DISPERSION_PA]
    if raw_headers and not usable:
        log.warning('%s: no slit position-angle convention for instrument %s: the slit PA is treated as unknown',
                    meta.get('object', ''), sorted({str(h.get('INSTRUME')) for h in raw_headers}))
    if usable and loc is not None:
        prod, tanz_mid, qs, pas = [], [], [], []
        for h in usable:
            c = SkyCoord(float(h['RADEG']), float(h['DECDEG']), unit='deg')
            t0 = Time(float(h['MJD-OBS']), format='mjd', scale='utc')
            times = t0 + np.linspace(0., float(h['EXPTIME']), 5) * u.s
            ha = (times.sidereal_time('apparent', longitude=loc.lon) - c.ra).wrap_at(180 * u.deg).rad
            q = parallactic_deg(ha, c.dec.rad, loc.lat.rad)
            z = 90. - c.transform_to(AltAz(obstime=times, location=loc)).alt.deg
            pa = DISPERSION_PA[str(h['INSTRUME']).strip().lower()](h)
            s = np.sin(np.radians(pa + 90. - q))
            prod.append(s * np.tan(np.radians(z)))
            tanz_mid.append(np.tan(np.radians(z[2])))
            qs.append([float(x) for x in q])
            pas.append(pa)
        prod = np.concatenate(prod)
        tz = float(np.mean(tanz_mid))
        out.update(s=float(np.mean(prod) / tz), s_err=float(np.std(prod) / tz), pa_known=True,
                   zenith_deg=float(np.degrees(np.arctan(tz))), q_deg=qs, pa_disp_deg=pas,
                   source=f'{len(usable)} raw header(s): PA, RADEG/DECDEG, MJD-OBS, EXPTIME; q and z from astropy')
        p, t, rh, wsrc = _weather(usable[0], meta, loc)
    else:
        if airmass is not None and np.isfinite(airmass) and airmass >= 1.:
            out['zenith_deg'] = float(np.degrees(np.arccos(1. / airmass)))
            out['source'] = 'airmass from the spectrum header; slit PA unknown'
        p, t, rh, wsrc = _weather(usable[0] if usable else {}, meta, loc)
    out.update(pressure_hpa=p, temp_c=t, rh=rh, source=f"{out['source']}; weather: {wsrc}")
    return out


def dar_kms(geo: dict, inp: slit_mod.SlitInputs, wave: np.ndarray | float, ref_wave: float) -> np.ndarray:
    """
    D(λ) [km/s per unit s]: the across-slit DAR between `wave` and `ref_wave` [Å, observatory frame] as a
    velocity; NaN where the zenith distance, the plate scale or the dispersion is unknown
    """
    lam = np.atleast_1d(np.asarray(wave, float))
    if not np.isfinite(geo['zenith_deg']) or inp is None or inp.scale_spec is None or inp.disp is None:
        return np.full(len(lam), np.nan)
    r = refraction_arcsec(np.append(lam, ref_wave), geo['zenith_deg'], geo['pressure_hpa'], geo['temp_c'], geo['rh'])
    gain = (slit_mod.centroid_gain(inp.width_arcsec, inp.fwhm_arcsec)
            if inp.width_arcsec and inp.fwhm_arcsec else 1.)
    d = np.interp(lam, inp.disp_wave, inp.disp)
    return C_KMS * gain * (r[:-1] - r[-1]) / inp.scale_spec * d / lam
