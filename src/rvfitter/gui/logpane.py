"""Thread-safe live log pane."""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QTextCharFormat
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

COLOURS = {
    logging.DEBUG: '#888888',
    logging.INFO: '#222222',
    logging.WARNING: '#b36b00',
    logging.ERROR: '#cc0000',
    logging.CRITICAL: '#990000',
}


class _Emitter(QObject):
    record = Signal(int, str)


class QtLogHandler(logging.Handler):
    """A logging handler that emits a Qt signal, so records from worker threads reach the GUI safely."""

    def __init__(self, level: int = logging.DEBUG):
        super().__init__(level)
        self.emitter = _Emitter()
        self.setFormatter(logging.Formatter('%(asctime)s %(levelname)-7s %(name)s: %(message)s', '%H:%M:%S'))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.emitter.record.emit(record.levelno, self.format(record))
        except RuntimeError:  # emitter already deleted at shutdown
            pass


class LogPane(QWidget):
    """Coloured, level-filtered log view fed by a :class:`QtLogHandler`."""

    def __init__(self, handler: QtLogHandler, parent: QWidget | None = None):
        super().__init__(parent)
        self.handler = handler
        self._records: list[tuple[int, str]] = []
        self.text = QPlainTextEdit(readOnly=True, maximumBlockCount=20000)
        self.level = QComboBox()
        for name in ('DEBUG', 'INFO', 'WARNING', 'ERROR'):
            self.level.addItem(name, getattr(logging, name))
        self.level.setCurrentText('INFO')
        self.level.setToolTip('Show only messages at or above this level.')
        clear = QPushButton('Clear')
        clear.clicked.connect(self.clear)
        bar = QHBoxLayout()
        bar.addWidget(self.level)
        bar.addStretch(1)
        bar.addWidget(clear)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(bar)
        lay.addWidget(self.text)
        handler.emitter.record.connect(self.add_record)
        self.level.currentIndexChanged.connect(self._refilter)

    def add_record(self, levelno: int, message: str) -> None:
        self._records.append((levelno, message))
        if levelno >= self.level.currentData():
            self._append(levelno, message)

    def _append(self, levelno: int, message: str) -> None:
        fmt = QTextCharFormat()
        colour = COLOURS[max((k for k in COLOURS if k <= levelno), default=logging.DEBUG)]
        fmt.setForeground(QColor(colour))
        cursor = self.text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(message + '\n', fmt)
        self.text.setTextCursor(cursor)
        self.text.ensureCursorVisible()

    def _refilter(self) -> None:
        self.text.clear()
        for lvl, msg in self._records[-5000:]:
            if lvl >= self.level.currentData():
                self._append(lvl, msg)

    def clear(self) -> None:
        self._records.clear()
        self.text.clear()

    def plain_text(self) -> str:
        return self.text.toPlainText()
