"""Small numerical helpers shared by the calibration and the methods."""
import numpy as np

__all__ = ['parabola_vertex', 'rms', 'weighted_mean_chi2']


def parabola_vertex(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    (x0, y0, y'') of the least-squares parabola through (x, y): its vertex and second derivative 2a.

    x0 and y0 are ±inf or NaN when the fit is a line (a = 0); callers check the curvature's sign first.
    """
    a, b, c = np.polyfit(x, y, 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        x0 = -b / (2 * a)
        y0 = c - b * b / (4 * a)
    return float(x0), float(y0), float(2 * a)


def weighted_mean_chi2(v: np.ndarray, err: np.ndarray) -> tuple[float, float, float]:
    """Inverse-variance weighted mean, its formal error 1/√Σw and χ² = Σw(v − mean)² about it."""
    w = 1. / np.asarray(err, float) ** 2
    v = np.asarray(v, float)
    mean = float(np.sum(w * v) / np.sum(w))
    return mean, float(1. / np.sqrt(np.sum(w))), float(np.sum(w * (v - mean) ** 2))


def rms(x) -> float:
    """Root mean square of the finite values; NaN if there are none."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v * v))) if len(v) else float('nan')
