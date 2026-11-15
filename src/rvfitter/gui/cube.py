"""The IFU extraction dialog: the collapsed cube with the aperture, and a preview of the extracted spectrum."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import plotting
from ..io import CUBE_RADIUS_DEFAULT

APERTURE_COLOUR = '#DDAA33'  # gold, against the viridis image


def _ticks(ax) -> None:
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True, bottom=True, left=True)


class CubeDialog(QDialog):
    """
    Click the collapsed image to centre the aperture, set its radius, or go back to the brightest spaxel; the
    spectrum extracted with each choice is previewed (``preview(position, radius)`` returns it). After OK,
    ``position`` (None: the brightest spaxel) and ``radius`` [spaxels] are the choice.
    """

    def __init__(self, image: np.ndarray, brightest: tuple[int, int], cube: Any,
                 preview: Callable[[tuple[float, float] | None, float], Any], name: str,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f'IFU extraction: {name}')
        self.image, self.brightest, self._preview = image, brightest, preview
        self.position: tuple[float, float] | None = None if cube.position is None else tuple(cube.position)
        with plotting.style_context(paper=False):
            self.figure = Figure(figsize=(9, 3.6), layout='constrained')
            self.ax_img, self.ax_spec = self.figure.subplots(1, 2, width_ratios=[1, 1.8])
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect('button_press_event', self._clicked)
        self.radius_box = QDoubleSpinBox()
        self.radius_box.setRange(0.5, max(image.shape) / 2)
        self.radius_box.setSingleStep(0.5)
        self.radius_box.setSuffix(' spaxels')
        self.radius_box.setValue(CUBE_RADIUS_DEFAULT if cube.radius is None else cube.radius)
        self.radius_box.valueChanged.connect(self._update)
        self.reset_btn = QPushButton('Brightest spaxel')
        self.reset_btn.setToolTip(f'Centre the aperture on the brightest spaxel {brightest}.')
        self.reset_btn.clicked.connect(self.reset)
        self.info = QLabel()
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        bar = QHBoxLayout()
        for w in (QLabel('Radius'), self.radius_box, self.reset_btn, self.info):
            bar.addWidget(w)
        bar.addStretch(1)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel('Click the image to centre the extraction aperture.'))
        lay.addWidget(self.canvas, 1)
        lay.addLayout(bar)
        lay.addWidget(buttons)
        self.resize(960, 480)
        self._update()

    @property
    def radius(self) -> float:
        return self.radius_box.value()

    def centre(self) -> tuple[float, float]:
        return self.brightest if self.position is None else self.position

    def reset(self) -> None:
        self.position = None
        self._update()

    def _clicked(self, event) -> None:
        if event.inaxes is self.ax_img and event.xdata is not None:
            self.set_position((float(round(event.xdata)), float(round(event.ydata))))

    def set_position(self, position: tuple[float, float]) -> None:
        self.position = None if position == self.brightest else position
        self._update()

    def _update(self) -> None:
        """Redraw the image with the aperture and the spectrum it extracts"""
        x, y = self.centre()
        where = f'brightest spaxel ({x}, {y})' if self.position is None else f'spaxel ({x:g}, {y:g})'
        self.info.setText(f'Centre: {where}')
        with plotting.style_context(paper=False):
            ax = self.ax_img
            ax.clear()
            finite = self.image[np.isfinite(self.image)]
            lo, hi = np.percentile(finite, [1, 99.5]) if finite.size else (0, 1)
            ax.imshow(self.image, origin='lower', cmap='viridis', vmin=lo, vmax=hi, interpolation='nearest')
            ax.add_patch(Circle((x, y), self.radius, fill=False, color=APERTURE_COLOUR, lw=1.5))
            ax.plot(*self.brightest, '+', color='white', ms=8)
            ax.set_xlabel('x [spaxel]')
            ax.set_ylabel('y [spaxel]')
            _ticks(ax)
            ax = self.ax_spec
            ax.clear()
            try:
                spec = self._preview(self.position, self.radius)
            except (OSError, ValueError) as exc:  # e.g. an aperture with no valid spaxel: say so, keep the dialog
                ax.text(0.5, 0.5, f'No spectrum: {exc}', transform=ax.transAxes, ha='center', wrap=True)
            else:
                ax.step(spec.wave, spec.flux / np.nanmedian(spec.flux), where='mid', color='k', lw=0.6)
            ax.set_xlabel('Wavelength [Å]')
            ax.set_ylabel(r'Normalised Flux [$F_\lambda$]')
            _ticks(ax)
        self.canvas.draw_idle()
