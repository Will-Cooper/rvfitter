"""Qt model over a growing pandas DataFrame."""
from __future__ import annotations

import math

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt


class PandasModel(QAbstractTableModel):
    """Read-only table model whose rows can be appended one object at a time."""

    def __init__(self, df: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self.df = df if df is not None else pd.DataFrame()

    def rowCount(self, parent=None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.df)

    def columnCount(self, parent=None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.df.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self.df.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
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

    def clear(self) -> None:
        self.beginResetModel()
        self.df = pd.DataFrame()
        self.endResetModel()
