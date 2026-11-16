"""Thread-safe live log pane."""
from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont, QPalette, QTextCharFormat
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import theme


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
    """Collapsible, level-filtered log view fed by a :class:`QtLogHandler`; the header counts warnings."""

    def __init__(self, handler: QtLogHandler, parent: QWidget | None = None):
        super().__init__(parent)
        self.handler = handler
        self._records: list[tuple[int, str]] = []
        self.warnings = 0
        self.text = QPlainTextEdit(readOnly=True, maximumBlockCount=20000)
        self.toggle = QToolButton(text='Log', checkable=True, autoRaise=True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        font = self.toggle.font()
        font.setBold(True)
        self.toggle.setFont(font)
        self.toggle.setToolTip('Show or hide the log.')
        self.warn_label = QLabel()
        self.warn_label.setToolTip('Warnings and errors logged so far.')
        self.level = QComboBox()
        for name in ('DEBUG', 'INFO', 'WARNING', 'ERROR'):
            self.level.addItem(name, getattr(logging, name))
        self.level.setCurrentText('INFO')
        self.level.setToolTip('Show only messages at or above this level.')
        self.show_label = QLabel('Show:')
        self.clear_btn = QPushButton('Clear log')
        self.clear_btn.setToolTip('Remove all messages from the log.')
        self.clear_btn.clicked.connect(self.clear)
        self.header = QWidget()
        bar = QHBoxLayout(self.header)
        bar.setContentsMargins(0, 0, 0, 0)
        bar.addWidget(self.toggle)
        bar.addWidget(self.warn_label)
        bar.addStretch(1)
        self.controls = [self.show_label, self.level, self.clear_btn]
        for w in self.controls:
            bar.addWidget(w)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.header)
        lay.addWidget(self.text)
        handler.emitter.record.connect(self.add_record)
        self.level.currentIndexChanged.connect(self._refilter)
        self.toggle.toggled.connect(self._expand)
        self._expand(False)
        self._show_warnings()

    def _expand(self, on: bool) -> None:
        self.text.setVisible(on)
        for w in self.controls:
            w.setVisible(on)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.setMaximumHeight(16777215 if on else self.header.sizeHint().height())

    def _show_warnings(self) -> None:
        n = self.warnings
        self.warn_label.setText(f'{n} warning{"" if n == 1 else "s"}' if n else '')
        pal = self.warn_label.palette()
        pal.setColor(QPalette.ColorRole.WindowText, theme.colour('flagged', pal))
        self.warn_label.setPalette(pal)

    def add_record(self, levelno: int, message: str) -> None:
        self._records.append((levelno, message))
        if levelno >= logging.WARNING:
            self.warnings += 1
            self._show_warnings()
        if levelno >= self.level.currentData():
            self._append(levelno, message)

    def _append(self, levelno: int, message: str) -> None:
        fmt = QTextCharFormat()  # palette colours only: dim for DEBUG, bold for WARNING and above
        if levelno < logging.INFO:
            fmt.setForeground(self.palette().placeholderText())
        elif levelno >= logging.WARNING:
            fmt.setFontWeight(QFont.Weight.Bold)
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
        for f in self.handler.filters:  # a cleared pane shows repeated messages again
            if hasattr(f, 'reset'):
                f.reset()
        self._records.clear()
        self.text.clear()
        self.warnings = 0
        self._show_warnings()

    def plain_text(self) -> str:
        return self.text.toPlainText()
