"""Diagnostics of the selected object: its barycentric correction."""
from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget


class DiagnosticsPanel(QWidget):
    """The barycentric correction applied to the selected object."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        box = QGroupBox('Barycentric correction')
        box.setToolTip('The velocity correction applied to this object, where it came from and its frame.')
        self.bary = QLabel()
        self.bary.setWordWrap(True)
        QVBoxLayout(box).addWidget(self.bary)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(box)
        lay.addStretch(1)
        self.clear()

    def clear(self) -> None:
        self.bary.setText('No object selected.')

    @staticmethod
    def bary_text(result: Any) -> str:
        b = result.barycentric
        if not b:
            return 'No barycentric information recorded.'
        kms = b.get('applied_kms')
        value = 'n/a' if kms is None else f'{kms:+.3f} km/s'
        return f'Value applied: {value}    Source: {b.get("source", "?")}    Frame: {b.get("frame", "?")}'

    def show_result(self, result: Any) -> None:
        self.bary.setText(f'{result.object}:  {self.bary_text(result)}')
