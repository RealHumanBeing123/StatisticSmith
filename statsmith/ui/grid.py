"""Spreadsheet grid: a virtual table that reads only the rows on screen (works for tens of millions of rows)."""
from __future__ import annotations

import datetime as _dt
from collections import OrderedDict

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QFontMetrics, QKeySequence, QGuiApplication
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QMenu, QTableView

from ..data.engine import ColumnInfo, Engine, qid
from .workers import run_async

PAGE = 200
MAX_CACHED_PAGES = 150
MAX_COPY_CELLS = 5_000_000


def fmt_value(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:
            return ""
        return f"{v:.10g}"
    if isinstance(v, (_dt.datetime, _dt.date, _dt.time)):
        return str(v)
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    return str(v)


class DatasetModel(QAbstractTableModel):
    edited = Signal(str, int, str, str)          # dataset, row (0-based), column, new text

    def __init__(self, engine: Engine, name: str, parent=None):
        super().__init__(parent)
        self.engine, self.name = engine, name
        self._cur = engine.con.cursor()          # private cursor: scrolling never waits for a long analysis
        self._cache: OrderedDict[int, list] = OrderedDict()
        self._ds = engine.datasets[name]
        self._retry = QTimer(self)
        self._retry.setSingleShot(True)
        self._retry.timeout.connect(self._emit_all)

    # ------------------------------------------------------------ structure
    @property
    def ds(self):
        return self._ds

    def reload(self):
        self.beginResetModel()
        self._ds = self.engine.datasets[self.name]
        self._cache.clear()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._ds.n_rows

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._ds.columns)

    def column_info(self, c: int) -> ColumnInfo:
        return self._ds.columns[c]

    # ------------------------------------------------------------ paging
    def _page(self, p: int):
        rows = self._cache.get(p)
        if rows is not None:
            self._cache.move_to_end(p)
            return rows
        try:
            rows = self._cur.execute(f"SELECT * FROM {qid(self.name)} LIMIT {PAGE} OFFSET {p * PAGE}").fetchall()
        except Exception:
            self._retry.start(400)
            return None
        self._cache[p] = rows
        while len(self._cache) > MAX_CACHED_PAGES:
            self._cache.popitem(last=False)
        return rows

    def _emit_all(self):
        if self.rowCount() and self.columnCount():
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))

    def value(self, r: int, c: int):
        page = self._page(r // PAGE)
        if page is None:
            return None
        i = r % PAGE
        return page[i][c] if i < len(page) else None

    # ------------------------------------------------------------ Qt model API
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            return fmt_value(self.value(index.row(), index.column()))
        if role == Qt.TextAlignmentRole:
            return int(Qt.AlignRight | Qt.AlignVCenter) if self._ds.columns[index.column()].is_numeric else int(Qt.AlignLeft | Qt.AlignVCenter)
        if role == Qt.ToolTipRole:
            col = self._ds.columns[index.column()]
            v = self.value(index.row(), index.column())
            if col.value_labels and v is not None:
                lab = col.value_labels.get(v) or col.value_labels.get(str(v)) or (col.value_labels.get(int(v)) if isinstance(v, float) and v == int(v) else None)
                if lab:
                    return f"{fmt_value(v)} = {lab}"
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal:
            col = self._ds.columns[section]
            if role == Qt.DisplayRole:
                return col.name
            if role == Qt.ToolTipRole:
                return f"{col.name}  [{col.sql_type}]" + (f"\n{col.label}" if col.label else "")
        elif role == Qt.DisplayRole:
            return str(section + 1)
        return None

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole or not index.isValid():
            return False
        col = self._ds.columns[index.column()]
        text = str(value).strip()
        try:
            if text == "":
                v = None
            elif col.is_numeric:
                v = float(text) if col.sql_type.upper() in ("DOUBLE", "FLOAT", "DECIMAL", "REAL") or "." in text or "e" in text.lower() else int(text)
            else:
                v = text
            self.engine.update_cell(self.name, index.row(), col.name, v)
        except Exception as e:  # noqa: BLE001
            self.editError = str(e).splitlines()[0]
            return False
        self._cache.pop(index.row() // PAGE, None)
        self.dataChanged.emit(index, index)
        self.edited.emit(self.name, index.row(), col.name, text)
        return True


class SheetView(QTableView):
    """Excel-like selection: click headers for whole rows/columns, drag for ranges, shift/ctrl-click to extend."""
    selection_text = Signal(str)
    selection_changed_cols = Signal(list)

    def __init__(self, model: DatasetModel, row_height: int = 22, parent=None):
        super().__init__(parent)
        self.setModel(model)
        self.dmodel = model
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setAlternatingRowColors(True)
        self.setCornerButtonEnabled(True)               # click the corner to select everything
        vh, hh = self.verticalHeader(), self.horizontalHeader()
        vh.setDefaultSectionSize(row_height)
        vh.setSectionResizeMode(QHeaderView.Fixed)
        vh.setSectionsClickable(True)
        hh.setSectionsClickable(True)
        hh.setDefaultSectionSize(110)
        hh.setMinimumSectionSize(40)
        hh.setSectionResizeMode(QHeaderView.Interactive)
        hh.setHighlightSections(True)
        vh.setHighlightSections(True)
        hh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.resize_row_header()
        self._token = 0
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(180)
        self._timer.timeout.connect(self._compute_stats)
        self.selectionModel().selectionChanged.connect(lambda *_: self._timer.start())
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        model.modelReset.connect(self.resize_row_header)

    def resize_row_header(self):
        fm = QFontMetrics(self.font())
        self.verticalHeader().setFixedWidth(max(40, fm.horizontalAdvance("9" * (len(str(max(self.dmodel.rowCount(), 1))) + 1)) + 14))

    # ------------------------------------------------------------ selection helpers
    def selected_ranges(self):
        """[(top, bottom, left, right)] of the current selection (0-based, inclusive)."""
        return [(r.top(), r.bottom(), r.left(), r.right()) for r in self.selectionModel().selection()]

    def selected_columns(self) -> list[str]:
        cols = set()
        for _, _, l, r in self.selected_ranges():
            cols.update(range(l, r + 1))
        return [self.dmodel.column_info(c).name for c in sorted(cols)]

    def current_column_name(self) -> str | None:
        i = self.currentIndex()
        return self.dmodel.column_info(i.column()).name if i.isValid() else None

    def _compute_stats(self):
        rngs = self.selected_ranges()
        cols = self.selected_columns()
        self.selection_changed_cols.emit(cols)
        if not rngs:
            self.selection_text.emit("")
            return
        ncell = sum((b - t + 1) * (r - l + 1) for t, b, l, r in rngs)
        nrows = len({(t, b) for t, b, *_ in rngs}) and sum(b - t + 1 for t, b, *_ in rngs)
        self._token += 1
        token = self._token
        eng, name, ds = self.dmodel.engine, self.dmodel.name, self.dmodel.ds
        head = f"{ncell:,} cell{'s' if ncell != 1 else ''}" if len(rngs) > 1 or ncell < 2 else \
            f"{rngs[0][1] - rngs[0][0] + 1:,} rows × {rngs[0][3] - rngs[0][2] + 1:,} cols"
        if ncell == 1:
            self.selection_text.emit(head)
            return

        def work():
            cur = eng.con.cursor()
            n = s = 0.0
            lo, hi = float("inf"), float("-inf")
            cnt_all = 0
            for t, b, l, r in rngs:
                for c in range(l, r + 1):
                    col = ds.columns[c]
                    if not col.is_numeric:
                        v = cur.execute(f"SELECT count(x) FROM (SELECT {qid(col.name)} x FROM {qid(name)} LIMIT {b - t + 1} OFFSET {t})").fetchone()[0]
                        cnt_all += v
                        continue
                    row = cur.execute(f"SELECT count(x), sum(x), min(x), max(x) FROM (SELECT TRY_CAST({qid(col.name)} AS DOUBLE) x "
                                      f"FROM {qid(name)} LIMIT {b - t + 1} OFFSET {t})").fetchone()
                    if row[0]:
                        n += row[0]; s += row[1]; lo = min(lo, row[2]); hi = max(hi, row[3])
                    cnt_all += row[0]
            cur.close()
            return n, s, lo, hi, cnt_all

        def done(res):
            if token != self._token:
                return
            n, s, lo, hi, cnt = res
            txt = head + f"   |   non-missing {int(cnt):,}"
            if n:
                txt += f"   |   numeric n={int(n):,}  sum={s:.6g}  mean={s / n:.6g}  min={lo:.6g}  max={hi:.6g}"
            self.selection_text.emit(txt)

        self.selection_text.emit(head + "   |   …")
        run_async(work, done, lambda *_: None, parent=self)

    # ------------------------------------------------------------ clipboard
    def keyPressEvent(self, e):
        if e.matches(QKeySequence.Copy):
            self.copy_selection()
            return
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace) and not self.state() == QAbstractItemView.EditingState:
            n = sum((b - t + 1) * (r - l + 1) for t, b, l, r in self.selected_ranges())
            if 0 < n <= 2000:
                for idx in self.selectionModel().selectedIndexes():
                    self.dmodel.setData(idx, "")
            return
        super().keyPressEvent(e)

    def copy_selection(self):
        rngs = self.selected_ranges()
        if not rngs:
            return
        t = min(r[0] for r in rngs); b = max(r[1] for r in rngs)
        l = min(r[2] for r in rngs); r_ = max(r[3] for r in rngs)
        if (b - t + 1) * (r_ - l + 1) > MAX_COPY_CELLS:
            b = t + MAX_COPY_CELLS // (r_ - l + 1) - 1
        names = [self.dmodel.column_info(c).name for c in range(l, r_ + 1)]
        rows = self.dmodel.engine.fetch(self.dmodel.name, t, b - t + 1, names)
        whole_cols = (b - t + 1) == self.dmodel.rowCount() or len(rows) > 1
        lines = ["\t".join(names)] if (t == 0 and len(rows) > 1) or whole_cols and False else []
        lines += ["\t".join(fmt_value(v) for v in row) for row in rows]
        QGuiApplication.clipboard().setText("\n".join(lines))

    def _menu(self, pos: QPoint):
        m = QMenu(self)
        m.addAction("Copy", self.copy_selection)
        m.addAction("Select all", self.selectAll)
        m.exec(self.viewport().mapToGlobal(pos))
