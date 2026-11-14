"""
Diagnostic figures, each ``plot_*(result: ObjectResult) -> Figure``, in one consistent style.

The figures are built with the object-oriented API (``Figure`` directly, no pyplot state), so they work in any
thread and backend; the caller saves or embeds them. The style (`style_context`) is applied only inside a ``with``
block, never at import, and its text is frozen onto the figure so that it still draws correctly afterwards.
"""
from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import matplotlib as mpl
import numpy as np
from cycler import cycler
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import blended_transform_factory

from . import features
from .calibration import sky_zeropoint
from .results import EXCLUDE_FLAGS
from .types import Measurement, ObjectResult

log = logging.getLogger(__name__)

__all__ = ['style_context', 'plot_line_fits', 'plot_ccf', 'plot_adopted', 'plot_corner',
           'plot_wavelength_correction', 'make_all', 'save_all']

COLOURS = ('#000000', '#004488', '#DDAA33')  # black, blue, gold; then the same three dashed
GREY = '0.45'
_ANG = r'$\mathrm{\AA}$'  # cmr10 has no Å glyph; mathtext draws it
# display names of the methods
METHOD_LABELS = {'ccf': 'CCF', 'template_fit': 'Template fit', 'linecentre': 'Line centring',
                 'forward_mcmc': 'Forward model', 'lsd': 'LSD', 'telluric_ref': 'Telluric reference'}


_PAPER: ContextVar[bool] = ContextVar('rvfitter_paper_plots', default=False)


def _rc(paper: bool) -> dict:
    # quick: bundled Computer Modern, pinned on each text by `_freeze`, so a figure draws the same wherever it is drawn.
    # paper: LaTeX typesets the text; it reads the font from rcParams at draw time, so paper figures must be saved
    # inside `style_context(paper=True)` (as `save_all` does)
    if paper and not (shutil.which('latex') and shutil.which('dvipng')):
        raise RuntimeError('output.plot_style = "paper" needs a LaTeX installation (latex and dvipng on PATH); '
                           'install TeX Live/MiKTeX or use plot_style = "quick"')
    fonts = ({'text.usetex': True, 'font.serif': ['Computer Modern Roman']} if paper else
             {'text.usetex': False, 'font.serif': ['cmr10'], 'axes.formatter.use_mathtext': True,
              'axes.unicode_minus': False})
    return {
        **fonts, 'font.family': 'serif', 'mathtext.fontset': 'cm',
        'axes.prop_cycle': cycler(color=COLOURS * 2, linestyle=['-'] * 3 + ['--'] * 3),
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True, 'image.cmap': 'viridis',
        'figure.constrained_layout.use': False, 'savefig.bbox': 'tight',
    }


@contextmanager
def style_context(paper: bool | None = None) -> Iterator[None]:
    """The rvfitter figure style inside a ``with`` block only; `paper` (default: the enclosing choice) uses LaTeX"""
    paper = _PAPER.get() if paper is None else paper
    token = _PAPER.set(paper)
    try:
        with mpl.rc_context(_rc(paper)):
            yield
    finally:
        _PAPER.reset(token)


def _break_gaps(w: np.ndarray, *ys: np.ndarray) -> tuple[np.ndarray, ...]:
    """Insert NaN where the pixel spacing jumps (masked intervals), so lines are not drawn across them"""
    dw = np.diff(w)
    at = np.flatnonzero(dw > 1.5 * np.median(dw)) + 1
    return (np.insert(w, at, np.nan), *(np.insert(y, at, np.nan) for y in ys))


def _ang() -> str:
    return r'\AA' if mpl.rcParams['text.usetex'] else _ANG


def _t(s: str) -> str:
    """Free text (region and parameter names) escaped for LaTeX when it typesets the figure"""
    if not mpl.rcParams['text.usetex']:
        return s
    return s.replace('\\', '').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&').replace('#', r'\#')


def _kms() -> str:
    return r'km s$^{-1}$'


def _freeze(fig: Figure) -> Figure:
    """Pin the style's fonts on every text (tick labels included), so the figure draws as built outside the style"""
    fig.draw_without_rendering()  # creates the tick labels while the style is active
    family, math = mpl.rcParams['font.serif'], mpl.rcParams['mathtext.fontset']
    for t in fig.findobj(Text):
        t.set_fontfamily(family)
        t.set_math_fontfamily(math)
    return fig


def _styled(fn: Callable[[ObjectResult], Figure]) -> Callable[[ObjectResult], Figure]:
    def wrapper(result: ObjectResult) -> Figure:
        with style_context():
            fig = fn(result)
            return fig if mpl.rcParams['text.usetex'] else _freeze(fig)
    wrapper.__name__, wrapper.__doc__ = fn.__name__, fn.__doc__
    return wrapper


def _measurements(result: ObjectResult, method: str, key: str) -> list[Measurement]:
    res = result.methods.get(method)
    return [m for m in (res.measurements if res else []) if key in m.diagnostics]


def _need(result: ObjectResult, method: str, key: str) -> list[Measurement]:
    ms = _measurements(result, method, key)
    if not ms:
        raise ValueError(f'{result.object}: no {method} measurements with {key!r} diagnostics to plot')
    return ms


def _grid(n: int, ncols: int = 3, **kw) -> tuple[Figure, np.ndarray]:
    ncols = min(n, ncols)
    nrows = -(-n // ncols)
    fig = Figure(figsize=(3.2 * ncols, 2.6 * nrows))
    axes = fig.subplots(nrows, ncols, squeeze=False, gridspec_kw=dict(wspace=0, hspace=0.35), **kw)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes


def _top_label(ax, x: float, text: str, line: bool = True) -> None:
    """A grey feature label along the top of the axes, with a dashed line for an atomic line"""
    if line:
        ax.axvline(x, color=GREY, ls='--', lw=0.8, zorder=0)
    ax.text(x, 1.02, _t(text), color=GREY, ha='center', va='bottom', fontsize=7,
            transform=blended_transform_factory(ax.transData, ax.transAxes))


def _bracket(ax, lo: float, hi: float, text: str) -> None:
    """A grey bracket marker along the top of the axes for a band (text '⊕' for a telluric band)"""
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot([lo, lo, hi, hi], [1.0, 1.03, 1.03, 1.0], color=GREY, lw=0.8, transform=tr, clip_on=False)
    ax.text(0.5 * (lo + hi), 1.04, text, color=GREY, ha='center', va='bottom', fontsize=7, transform=tr)


def _rv_text(ax, m: Measurement) -> None:
    ax.text(0.03, 0.04, rf'${m.rv:.1f}\pm{m.rv_err:.1f}$ {_kms()}', transform=ax.transAxes, fontsize=7)


# ---------------------------------------------------------------------------------------------- line fits
@_styled
def plot_line_fits(result: ObjectResult) -> Figure:
    """Line-centring fits: one panel per window with data, fit, continuum windows and the fitted line centres"""
    ms = _need(result, 'linecentre', 'model')
    fig, axes = _grid(len(ms), sharey='row')
    for ax, m in zip(axes.flat, ms, strict=False):
        d = m.diagnostics
        w, y = np.asarray(d['wave']), np.asarray(d['data'])
        norm = np.nanmedian(y) or 1.
        for lo, hi in d.get('continuum', ()):
            ax.axvspan(lo, hi, color=COLOURS[2], alpha=0.2, lw=0)
        wg, yg, mg = _break_gaps(w, y / norm, np.asarray(d['model']) / norm)
        ax.step(wg, yg, where='mid', color=COLOURS[0], lw=0.8)
        ax.plot(wg, mg, color=COLOURS[1], lw=1.2)
        for name, rest, fit in zip(d['lines'], d['rest_wave'], d['fit_wave'], strict=True):
            ax.axvline(fit, color=COLOURS[2], ls='--', lw=1)
            _top_label(ax, rest, name)
        ax.set_xlim(w.min(), w.max())
        _rv_text(ax, m)
    for ax in axes[-1]:
        ax.set_xlabel(f'Wavelength [{_ang()}]')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'Normalised Flux [$F_\lambda$]')
    for ax in axes[:, 1:].flat:
        ax.tick_params(labelleft=False)
    return fig


# ---------------------------------------------------------------------------------------------------- CCF
@_styled
def plot_ccf(result: ObjectResult) -> Figure:
    """Cross-correlation functions: one panel per window with the parabolic peak fit and the measured RV"""
    ms = _need(result, 'ccf', 'ccf')
    fig, axes = _grid(len(ms), sharey='row')
    for ax, m in zip(axes.flat, ms, strict=False):
        d = m.diagnostics
        v, c, peak = np.asarray(d['rv_grid']), np.asarray(d['ccf']), d['peak']
        ax.plot(v, c, color=COLOURS[0], lw=1)
        if peak['ok']:
            x = np.linspace(peak['rv'] - 1.5 * peak['width'], peak['rv'] + 1.5 * peak['width'], 100)
            ax.plot(x, peak['height'] + 0.5 * peak['curvature'] * (x - peak['rv']) ** 2, color=COLOURS[1], lw=1.2)
            ax.axvline(peak['rv'], color=COLOURS[2], ls='--', lw=1)
        _top_label(ax, np.nanmean(v), m.region, line=False)
        _rv_text(ax, m)
    allc = np.concatenate([np.asarray(m.diagnostics['ccf'])[np.isfinite(m.diagnostics['ccf'])] for m in ms])
    for ax in axes[:, 0]:
        ax.set_ylim(allc.min() - 0.05 * np.ptp(allc), allc.max() + 0.1 * np.ptp(allc))
    for ax in axes[-1]:
        ax.set_xlabel(f'RV [{_kms()}]')
    for ax in axes[:, 0]:
        ax.set_ylabel('CCF')
    for ax in axes[:, 1:].flat:
        ax.tick_params(labelleft=False)
    return fig


# ----------------------------------------------------------------------------------------------- adopted
@_styled
def plot_adopted(result: ObjectResult) -> Figure:
    """Per-line RVs of every method with errors, and the adopted RV as a grey band (16th to 84th percentile)"""
    methods = {n: [m for m in r.measurements if np.isfinite(m.rv)] for n, r in result.methods.items()}
    methods = {n: ms for n, ms in methods.items() if ms}
    if not methods:
        raise ValueError(f'{result.object}: no finite measurements to plot')
    lines = list(dict.fromkeys(m.region for ms in methods.values() for m in ms))
    fig = Figure(figsize=(4.5, 0.3 * len(lines) * len(methods) ** 0.5 + 1.2), layout='tight')
    ax = fig.add_subplot()
    step = 0.6 / len(methods)
    cycle = list(mpl.rcParams['axes.prop_cycle'])
    for k, (name, ms) in enumerate(methods.items()):
        sty, mark = cycle[k % len(cycle)], 'os'[(k % len(cycle)) >= len(COLOURS)]
        off = (k - 0.5 * (len(methods) - 1)) * step
        y = np.array([lines.index(m.region) + off for m in ms])
        rv, err = np.array([m.rv for m in ms]), np.array([m.rv_err for m in ms])
        used = np.array([not m.flag & EXCLUDE_FLAGS for m in ms])
        kw = dict(color=sty['color'], ecolor=sty['color'], ls='none', capsize=0, elinewidth=1.2, ms=4)
        ax.errorbar(rv[used], y[used], xerr=err[used], fmt=mark, label=METHOD_LABELS.get(name, name), **kw)
        ax.errorbar(rv[~used], y[~used], xerr=err[~used], fmt=mark, mfc='none', **kw)
    ad = result.adopted
    if ad.get('median') is not None and np.isfinite(ad['median']):
        ax.axvspan(ad.get('q16', ad['median'] - ad['err']), ad.get('q84', ad['median'] + ad['err']),
                   color=GREY, alpha=0.25, lw=0)
        ax.axvline(ad['median'], color=GREY, ls='--', lw=1)
        ax.set_title(rf'Adopted RV $={ad["median"]:.2f}\pm{ad["err"]:.2f}$ {_kms()}', fontsize=9)
    ax.set_yticks(range(len(lines)), [_t(x) for x in lines])
    ax.set_ylim(-0.7, len(lines) - 0.3)
    ax.invert_yaxis()
    ax.set_xlabel(f'RV [{_kms()}]')
    ax.legend(fontsize=7, frameon=False)
    return fig


# ------------------------------------------------------------------------------------------------ corner
@_styled
def plot_corner(result: ObjectResult) -> Figure:
    """Corner plot of the ``forward_mcmc`` posterior samples of the first window (needs the `corner` package)"""
    try:
        import corner
    except ImportError as err:
        raise ImportError('plot_corner needs the `corner` package: install it with `pip install corner`') from err
    m = _need(result, 'forward_mcmc', 'samples')[0]
    d = m.diagnostics
    fig = Figure(figsize=(6, 6))
    corner.corner(np.asarray(d['samples']), labels=[_t(str(n)) for n in d['sample_names']], fig=fig,
                  quantiles=[0.16, 0.5, 0.84], show_titles=True, color=COLOURS[0])
    return fig


# ------------------------------------------------------------------------------- wavelength correction
@_styled
def plot_wavelength_correction(result: ObjectResult) -> Figure:
    """Sky Δv(λ) per segment, the applied correction curve and the telluric nodes (if measured)"""
    cur = result.barycentric.get('wavelength_curves') or {}
    sky, tell, applied = cur.get('sky_zeropoint'), cur.get('telluric_zeropoint'), cur.get('wavelength_correction')
    if not (sky or tell):
        raise ValueError(f'{result.object}: no wavelength-correction curves were measured')
    fig = Figure(figsize=(6, 3), layout='tight')
    ax = fig.add_subplot()
    waves = []
    for curve, col, label in ((sky, COLOURS[0], 'Sky segments'), (tell, COLOURS[2], 'Telluric nodes')):
        if curve and len(curve['node_wave']):
            ax.errorbar(curve['node_wave'], curve['node_dv'], yerr=curve['node_err'], fmt='o', ms=3, color=col,
                        lw=1, label=label)
            waves += list(curve['node_wave'])
    if applied and applied.get('source') not in (None, 'none'):
        w = np.linspace(min(waves), max(waves), 500)
        dv = np.full_like(w, applied['constant_kms'])
        for key in ('sky', 'telluric'):
            if applied.get(key):
                dv = dv + sky_zeropoint.curve_at(applied[key], w)[0]
        ax.plot(w, dv, color=COLOURS[1], lw=1.2, label=f'Applied ({applied["source"]})')
    ax.axhline(0, color=GREY, ls=':', lw=0.8)
    for f in features.features():
        if f.kind == 'telluric' and min(waves) < f.wmax and f.wmin < max(waves):
            _bracket(ax, f.wmin, f.wmax, r'$\oplus$')
    ax.set_xlabel(f'Wavelength [{_ang()}]')
    ax.set_ylabel(f'$\\Delta v$ [{_kms()}]')
    ax.legend(fontsize=7, frameon=False)
    return fig


# ------------------------------------------------------------------------------------------------ driver
_AVAILABLE = {
    'line_fits': (plot_line_fits, lambda r: _measurements(r, 'linecentre', 'model')),
    'ccf': (plot_ccf, lambda r: _measurements(r, 'ccf', 'ccf')),
    'adopted': (plot_adopted, lambda r: bool(r.methods)),
    'corner': (plot_corner, lambda r: _measurements(r, 'forward_mcmc', 'samples')),
    'wavelength_correction': (plot_wavelength_correction, lambda r: (r.barycentric.get('wavelength_curves') or {})
                              .get('sky_zeropoint') or (r.barycentric.get('wavelength_curves') or {})
                              .get('telluric_zeropoint')),
}


def make_all(result: ObjectResult) -> dict[str, Figure]:
    """Every figure the result has data for, keyed by name (``corner`` is skipped, with a warning, if not installed)"""
    out = {}
    for name, (fn, has) in _AVAILABLE.items():
        if not has(result):
            continue
        try:
            out[name] = fn(result)
        except ImportError as err:
            log.warning('%s: %s figure skipped: %s', result.object, name, err)
    return out


def save_all(results: list[ObjectResult], directory: Path, style: str = 'quick') -> list[Path]:
    """Write every figure of every result as ``<directory>/<object>_<figure>.pdf``; `style` 'quick' or 'paper'"""
    if style not in ('quick', 'paper'):
        raise ValueError(f"plot style must be 'quick' or 'paper', not {style!r}")
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with style_context(paper=style == 'paper'):  # built and saved in the style (LaTeX reads it at save time)
        for res in results:
            for name, fig in make_all(res).items():
                path = directory / f'{res.object}_{name}.pdf'
                fig.savefig(path)
                paths.append(path)
    log.info('Wrote %d figures to %s', len(paths), directory)
    return paths
