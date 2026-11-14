"""A synthetic ObjectResult with the diagnostics each plot needs."""
import numpy as np

from rvfitter.types import Measurement, MethodResult, ObjectResult


def synthetic_result() -> ObjectResult:
    rng = np.random.default_rng(1)
    w = np.linspace(7690., 7712., 80)
    y = 1 - 0.4 * np.exp(-0.5 * ((w - 7701.) / 1.2) ** 2)
    lc = Measurement('linecentre', 'K I 7701', 10., 1., 0, dict(
        wave=w, data=y + rng.normal(0, 0.02, w.size), model=y, sigma=np.full(w.size, 0.02), lines={'K I 7701': {}},
        rest_wave=[7701.], fit_wave=[7701.3], continuum=((7690., 7694.), (7708., 7712.))))
    v = np.linspace(-100., 120., 111)
    c = 0.8 * np.exp(-0.5 * ((v - 10.) / 15.) ** 2)
    peak = dict(rv=10., height=0.8, curvature=-0.8 / 15. ** 2, width=15., edge=False, ok=True)
    cc = Measurement('ccf', 'CCF 7600-8000', 11., 2., 0, dict(rv_grid=v, ccf=c, peak=peak))
    fw = Measurement('forward_mcmc', 'K I 7701', 10.5, 0.8, 0, dict(
        samples=rng.normal([10.5, 5.], [0.8, 1.], (400, 2)), sample_names=['rv_0', 'vsini']))
    sky = dict(node_wave=[7000., 7500., 8000.], node_dv=[1., 0.5, 0.2], node_err=[0.3, 0.3, 0.3],
               node_span=[0, 0, 0], span_bounds={0: [6900., 8100.]}, inflate=0.)
    return ObjectResult(
        'J0000', ['a.fits'],
        methods={'linecentre': MethodResult('linecentre', 10., 1., 1, [lc]),
                 'ccf': MethodResult('ccf', 11., 2., 1, [cc]),
                 'forward_mcmc': MethodResult('forward_mcmc', 10.5, 0.8, 1, [fw])},
        adopted={'median': 10.4, 'err': 0.9, 'q16': 9.5, 'q84': 11.3},
        barycentric={'wavelength_curves': {'sky_zeropoint': sky, 'telluric_zeropoint': None,
                                           'wavelength_correction': dict(source='sky', constant_kms=0., sky=sky,
                                                                         telluric=None)}})
