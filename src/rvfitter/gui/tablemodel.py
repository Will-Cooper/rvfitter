"""Qt model over a growing pandas DataFrame."""
from __future__ import annotations

import math

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QBrush, QColor


class PandasModel(QAbstractTableModel):
    """Read-only table model whose rows can be appended one object at a time."""

    def __init__(self, df: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self.df = df if df is not None else pd.DataFrame()
        self.notes: dict[int, str] = {}  # row -> guidance text (marked and shown as a tooltip)

    def rowCount(self, parent=None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.df)

    def columnCount(self, parent=None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.df.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.df.iat[index.row(), index.column()]
        note = self.notes.get(index.row())
        if role == Qt.ItemDataRole.ToolTipRole and note:
            return note
        if role == Qt.ItemDataRole.BackgroundRole and note:
            return QBrush(QColor('#fff3cd'))
        if role == Qt.ItemDataRole.ForegroundRole and note:
            return QBrush(QColor('#000000'))
        if role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0 and note:
                return f'\u26a0 {value}'
            if isinstance(value, float):
                return '' if math.isnan(value) else f'{value:.4g}'
            return str(value)
        if role == Qt.ItemDataRole.TextAlignmentRole and isinstance(value, int | float):
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self.df.columns[section])
        return str(section + 1)

    def append(self, rows: pd.DataFrame) -> None:
        """Append rows, adding any new columns."""
        if rows is None or len(rows) == 0:
            return
        rows = rows.reset_index(drop=True)
        if len(self.df) == 0 or list(rows.columns) != list(self.df.columns):
            self.beginResetModel()
            self.df = pd.concat([self.df, rows], ignore_index=True)
            self.endResetModel()
            return
        n = len(self.df)
        self.beginInsertRows(QModelIndex(), n, n + len(rows) - 1)
        self.df = pd.concat([self.df, rows], ignore_index=True)
        self.endInsertRows()

    def replace(self, row: int, rows: pd.DataFrame) -> None:
        """Replace one row (adding any new columns)."""
        rows = rows.reset_index(drop=True)
        if list(rows.columns) != list(self.df.columns):
            self.beginResetModel()
            keep = self.df.drop(index=row).reset_index(drop=True)
            top, bottom = keep.iloc[:row], keep.iloc[row:]
            self.df = pd.concat([top, rows, bottom], ignore_index=True)
            self.endResetModel()
            return
        self.df.iloc[row] = rows.iloc[0]
        self.dataChanged.emit(self.index(row, 0), self.index(row, len(self.df.columns) - 1))

    def set_note(self, row: int, text: str | None) -> None:
        """Mark (or, with ``None``, unmark) a row with guidance text."""
        if text:
            self.notes[row] = text
        else:
            self.notes.pop(row, None)
        if 0 <= row < len(self.df):
            self.dataChanged.emit(self.index(row, 0), self.index(row, max(0, len(self.df.columns) - 1)))

    def clear(self) -> None:
        self.beginResetModel()
        self.notes = {}
        self.df = pd.DataFrame()
        self.endResetModel()
