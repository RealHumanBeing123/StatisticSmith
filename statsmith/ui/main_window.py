"""Main window: spreadsheet tabs, Analysis / Graph menus, terminal, script, settings, projects."""
from __future__ import annotations

import datetime as _dt
import json
import shutil
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QInputDialog, QLabel, QMainWindow, QMenu, QMessageBox, QProgressBar,
                               QProgressDialog, QStackedWidget, QTabWidget, QToolBar, QToolButton, QWidget)

from .. import APP_NAME, __version__
from ..charts import CATEGORIES, REGISTRY as CHARTS, ChartStyle, specs_by_category
from ..codegen import Recorder
from ..data import Engine, ImportCancelled, build_codebook_text, has_codebook, human_bytes, qid
from ..data.engine import SUPPORTED
from ..runner import MAX_CHART_ROWS, load_roles, run_analysis
from ..settings import Settings
from ..stats import REGISTRY as ANALYSES
from ..stats import base as stats_base
from ..stats.base import CATEGORY_ORDER, by_category
from ..terminal import Terminal
from . import theme
from .dialogs import (ChoiceDialog, CodebookDialog, ScriptWindow, SettingsDialog, TextWindow, VariableDialog)
from .graph_window import GraphWindow
from .grid import DatasetModel, SheetView
from .terminal_dock import TerminalDock
from .workers import run_async

FILTER = ("Data files (*.csv *.tsv *.txt *.tab *.xlsx *.xlsm *.xls *.ods *.dta *.sav *.sas7bdat *.rds *.rdata *.rda "
          "*.parquet *.json *.feather);;CSV / text (*.csv *.tsv *.txt *.tab);;Excel (*.xlsx *.xlsm *.xls *.ods);;"
          "Stata (*.dta);;R data (*.rds *.rdata *.rda);;All files (*)")
PROJECT_EXT = ".ssproj"


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings | None = None, engine: Engine | None = None):
        super().__init__()
        self.settings = settings or Settings.load()
        stats_base.DEFAULT_DECIMALS[0] = self.settings.decimals
        self.engine = engine or Engine(self.settings.ram_gb, self.settings.cores)
        self.recorder = Recorder(seed=self.settings.seed)
        self.terminal = Terminal(self.engine, self.recorder)
        self.sheets: dict[str, tuple[DatasetModel, SheetView]] = {}
        self.modified: set[str] = set()
        self.results_log: list[str] = []
        self.windows: list = []
        self.script_win: ScriptWindow | None = None
        self.project_path: str | None = None
        self.setWindowTitle(APP_NAME)
        self.resize(1360, 820)
        self.setAcceptDrops(True)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.South)
        self.tabs.setTabsClosable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.tabCloseRequested.connect(self.close_dataset)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.stack = QStackedWidget()          # tabs + empty page live here; setCentralWidget would delete the old one
        self.setCentralWidget(self.stack)
        self.empty = QLabel("<h2>Statistic Smith</h2><p>Open a data file (File → Open data…) or drag it here.<br>"
                            "CSV, Excel, Stata .dta, R .rds/.RData, SPSS, SAS, Parquet and JSON are supported.</p>")
        self.empty.setAlignment(Qt.AlignCenter)
        self.stack.addWidget(self.empty)
        self.stack.addWidget(self.tabs)

        self.dock = TerminalDock(self.terminal, self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.dock)
        self.dock.hide()
        self.dock.dataset_changed.connect(self._terminal_dataset)
        self.dock.switch_dataset.connect(self._focus_dataset)
        self.dock.figure_requested.connect(self._terminal_figure)
        self.dock.recorded.connect(self._recorded)

        self._build_status()
        self._build_menus()
        self._tab_changed()
        self._show_empty(True)

    # ===================================================================== construction
    def _build_status(self):
        sb = self.statusBar()
        self.msg = QLabel("Ready")
        self.sel = QLabel("")
        self.res = QLabel("")
        self.bar = QProgressBar()
        self.bar.setMaximumWidth(180)
        self.bar.hide()
        sb.addWidget(self.msg, 1)
        sb.addWidget(self.sel, 3)
        sb.addPermanentWidget(self.bar)
        sb.addPermanentWidget(self.res)
        self._update_res_label()

    def _update_res_label(self):
        self.res.setText(f"  RAM limit {self.settings.ram_gb:g} GB · {self.settings.cores} cores  ")

    def _act(self, menu, text, fn=None, shortcut=None, checkable=False, tip=""):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.setCheckable(checkable)
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        if fn:
            a.triggered.connect(lambda *_: fn())
        if menu is not None:
            menu.addAction(a)
        return a

    def _build_menus(self):
        mb = self.menuBar()
        # ---- File
        m = mb.addMenu("&File")
        self._act(m, "Open data…", self.open_data, "Ctrl+O")
        self.recent_menu = m.addMenu("Open recent")
        self._refresh_recent()
        self._act(m, "Open example dataset (clinical trial)", self.open_example, tip="A small simulated blood-pressure trial, to learn the software")
        m.addSeparator()
        self._act(m, "Open project…", self.open_project, "Ctrl+Shift+O")
        self._act(m, "Save project", self.save_project, "Ctrl+S")
        self._act(m, "Save project as…", lambda: self.save_project(True), "Ctrl+Shift+S")
        m.addSeparator()
        self._act(m, "Export current dataset…", self.export_dataset)
        self._act(m, "Save all results as text file…", self.save_results)
        m.addSeparator()
        self._act(m, "Exit", self.close, "Ctrl+Q")
        # ---- Data
        d = mb.addMenu("&Data")
        self._act(d, "Compute new column…", self.compute_column)
        self._act(d, "Filter rows into new dataset…", self.filter_rows)
        self._act(d, "Sort into new dataset…", self.sort_rows)
        d.addSeparator()
        self._act(d, "Summary of selected columns", self.column_summary, "Ctrl+Shift+D")
        self._act(d, "Dataset information", self.dataset_info)
        self.codebook_act = self._act(d, "Open codebook…", self.open_codebook, "F1")
        # ---- Analysis (dropdown built from the registry; also used by the toolbar button)
        self.analysis_menu = QMenu("&Analysis", self)
        for cat, specs in by_category().items():
            if not specs:
                continue
            sub = self.analysis_menu.addMenu(f"{cat}  ({len(specs)})")
            for s in specs:
                self._act(sub, s.name, lambda s=s: self.run_analysis_dialog(s.id), tip=s.description)
        mb.addMenu(self.analysis_menu)
        # ---- Graph
        self.graph_menu = QMenu("&Graph", self)
        by = specs_by_category()
        for cat in CATEGORIES:
            specs = by.get(cat, [])
            if not specs:
                continue
            sub = self.graph_menu.addMenu(f"{cat}  ({len(specs)})")
            for s in specs:
                self._act(sub, s.name, lambda s=s: self.graph_dialog(s.id), tip=s.description)
        mb.addMenu(self.graph_menu)
        # ---- Tools
        t = mb.addMenu("&Tools")
        self.term_act = self._act(t, "Terminal (SQL / Python / Stata / R)", None, "Ctrl+`", checkable=True)
        self.term_act.toggled.connect(self.dock.setVisible)
        self.dock.visibilityChanged.connect(self.term_act.setChecked)
        self._act(t, "Script (.do / .R / .py)…", self.show_script, "Ctrl+Shift+R")
        self._act(t, "Settings…", self.show_settings, "Ctrl+,")
        # ---- Help
        h = mb.addMenu("&Help")
        self._act(h, "Stats Flashcards…", self.show_flashcards, "F2", tip="Learn which test to use and what p-values, z-scores, standard deviations … mean")
        self._act(h, "Methods & verification notes", self.show_methods)
        self._act(h, "About", self.about)
        # ---- Toolbar (StatCrunch-style dropdown buttons)
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)
        tb.addAction(self._act(None, "Open", self.open_data))
        tb.addAction(self._act(None, "Save", self.save_project))
        tb.addSeparator()
        for label, menu in (("Analysis ▾", self.analysis_menu), ("Graph ▾", self.graph_menu)):
            b = QToolButton()
            b.setText(label)
            b.setMenu(menu)
            b.setPopupMode(QToolButton.InstantPopup)
            b.setStyleSheet("QToolButton { font-weight: bold; }")
            tb.addWidget(b)
        tb.addSeparator()
        tb.addAction(self._act(None, "Summary", self.column_summary, tip="Summary statistics of the selected columns"))
        self.cb_btn = self._act(None, "Codebook", self.open_codebook, tip="Open the codebook attached to this dataset")
        tb.addAction(self.cb_btn)
        tb.addSeparator()
        tb.addAction(self.term_act)
        tb.addAction(self._act(None, "Script", self.show_script))
        tb.addAction(self._act(None, "Settings", self.show_settings))
        self.term_act.setText("Terminal")

    # ===================================================================== state helpers
    def current_name(self) -> str | None:
        w = self.tabs.currentWidget()
        for n, (_, v) in self.sheets.items():
            if v is w:
                return n
        return None

    def current_ds(self):
        n = self.current_name()
        return self.engine.datasets[n] if n else None

    def current_sheet(self) -> SheetView | None:
        n = self.current_name()
        return self.sheets[n][1] if n else None

    def _need_dataset(self) -> bool:
        if not self.current_name():
            QMessageBox.information(self, APP_NAME, "Open a dataset first (File → Open data…).")
            return False
        return True

    def _show_empty(self, show: bool):
        self.stack.setCurrentWidget(self.empty if (show and self.tabs.count() == 0) else self.tabs)

    def flash(self, text: str, ms: int = 6000):
        self.statusBar().showMessage(text, ms)

    def _tab_changed(self, *_):
        n = self.current_name()
        self.terminal.set_active(n)
        self.dock.set_dataset_label(n)
        ds = self.engine.datasets.get(n) if n else None
        ok = bool(ds and has_codebook(ds))
        self.codebook_act.setEnabled(ok)
        self.cb_btn.setEnabled(ok)
        self.cb_btn.setToolTip("Open the codebook attached to this dataset" if ok else "No codebook is attached to this dataset")
        self.sel.setText("")

    def _recorded(self):
        if self.script_win:
            self.script_win.on_recorder_changed()

    # ===================================================================== datasets / tabs
    def add_dataset_tab(self, name: str):
        self._show_empty(False)
        model = DatasetModel(self.engine, name)
        view = SheetView(model, self.settings.grid_row_height)
        view.selection_text.connect(self.sel.setText)
        model.edited.connect(self._cell_edited)
        self.sheets[name] = (model, view)
        i = self.tabs.addTab(view, f"{name}  ({self.engine.datasets[name].n_rows:,} × {len(self.engine.datasets[name].columns)})")
        self.tabs.setCurrentIndex(i)
        self._tab_changed()

    def refresh_dataset(self, name: str):
        if name in self.sheets:
            self.sheets[name][0].reload()
            ds = self.engine.datasets[name]
            self.tabs.setTabText(self.tabs.indexOf(self.sheets[name][1]), f"{name}  ({ds.n_rows:,} × {len(ds.columns)})")
        else:
            self.add_dataset_tab(name)

    def _focus_dataset(self, name: str):
        if name in self.sheets:
            self.tabs.setCurrentWidget(self.sheets[name][1])

    def _terminal_dataset(self, name: str):
        if name in self.engine.datasets:
            self.modified.add(name)
            self.refresh_dataset(name)
        self._recorded()

    def close_dataset(self, idx: int):
        w = self.tabs.widget(idx)
        name = next((n for n, (_, v) in self.sheets.items() if v is w), None)
        if not name:
            return
        if QMessageBox.question(self, "Close dataset", f"Close '{name}'? (The original file is not changed.)") != QMessageBox.Yes:
            return
        self.tabs.removeTab(idx)
        self.sheets.pop(name)
        self.modified.discard(name)
        self.engine.drop_dataset(name)
        w.deleteLater()
        if self.tabs.count() == 0:
            self._show_empty(True)
        self._tab_changed()

    def _cell_edited(self, ds: str, row: int, col: str, text: str):
        self.modified.add(ds)
        c = self.engine.datasets[ds].col(col)
        try:
            float(text)
            lit = text
        except ValueError:
            lit = f'"{text}"'
        self.recorder.transform(f"Edited cell: row {row + 1}, column {col} = {text or '(missing)'}",
                                stata=f"replace {col} = {lit if text else '.'} in {row + 1}",
                                r=f'df[{row + 1}, "{col}"] <- {lit if text else "NA"}',
                                python=f'df.loc[{row}, "{col}"] = {lit if text else "None"}')
        self._recorded()

    # ===================================================================== import
    def open_data(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open data", self.settings.recent_files[0] if self.settings.recent_files else "", FILTER)
        for p in paths:
            self.import_path(p)

    def import_path(self, path: str, name: str | None = None, options: dict | None = None, record: bool = True, done=None):
        p = Path(path)
        ext = p.suffix.lower()
        if ext not in SUPPORTED:
            QMessageBox.warning(self, "Open data", f"Unsupported file type '{ext}'.\nSupported: {', '.join(sorted(SUPPORTED))}")
            return
        options = dict(options or {})
        try:
            if ext in (".xlsx", ".xlsm", ".xls", ".ods") and "sheet" not in options:
                sheets = self.engine.excel_sheets(str(p))
                if len(sheets) > 1:
                    dlg = ChoiceDialog("Excel sheet", "This workbook has several sheets. Which one do you want to open?", sheets, self)
                    if dlg.exec() != QDialog.DialogCode.Accepted:
                        return
                    options["sheet"] = dlg.value()
                elif sheets:
                    options["sheet"] = sheets[0]
            if ext in (".rdata", ".rda") and "object" not in options:
                objs = self.engine.r_objects(str(p))
                if len(objs) > 1:
                    dlg = ChoiceDialog("R object", "This R file contains several data frames. Which one do you want to open?", objs, self)
                    if dlg.exec() != QDialog.DialogCode.Accepted:
                        return
                    options["object"] = dlg.value()
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Open data", str(e))
            return
        size = p.stat().st_size if p.exists() else 0
        pd_ = QProgressDialog(f"Opening {p.name} ({human_bytes(size)})…", "Cancel", 0, 1000, self)
        pd_.setWindowTitle("Importing")
        pd_.setWindowModality(Qt.WindowModal)
        pd_.setMinimumDuration(300)
        pd_.setValue(0)

        def on_prog(f, m):
            pd_.setLabelText(m)
            if f < 0:
                pd_.setRange(0, 0)
            else:
                pd_.setRange(0, 1000)
                pd_.setValue(int(f * 1000))

        def finished(ds):
            pd_.close()
            self.settings.add_recent(str(p))
            self.settings.save()
            self._refresh_recent()
            self.add_dataset_tab(ds.name)
            if record:
                self.recorder.import_file(str(p), options.get("sheet"), ds.name)
                self._recorded()
            msg = f"Imported {ds.n_rows:,} rows × {len(ds.columns)} columns from {p.name}"
            if ds.notes:
                msg += " — " + "; ".join(ds.notes)
            self.flash(msg, 12000)
            if done:
                done(ds)

        def failed(msg, tb):
            pd_.close()
            if "ImportCancelled" in tb:
                self.flash("Import cancelled")
                return
            QMessageBox.critical(self, "Import failed", f"{p.name}\n\n{msg}")

        w = run_async(self.engine.import_file, finished, failed, on_prog, self, with_progress=True, path=str(p), name=name, options=options)
        pd_.canceled.connect(w.cancel.set)

    def open_example(self, kind: str = "dta"):
        """Open the bundled example (Stata version carries variable/value labels, so the codebook works)."""
        from ..app import examples_dir
        f = examples_dir() / f"trial.{kind}"
        if self.engine.datasets.get("trial"):
            self._focus_dataset("trial")
            return
        self.import_path(str(f))

    def welcome(self):
        """First start: explain the basics and load the example data."""
        self.settings.welcomed = True
        self.settings.save()
        QMessageBox.information(self, f"Welcome to {APP_NAME}",
            "<b>Welcome!</b> An example dataset (a simulated blood-pressure trial) will open now so you can look around:"
            "<ul><li>Click a column or row header to select it, or drag across cells to select a block.</li>"
            "<li>With columns selected, open <b>Analysis ▾</b> (e.g. Regression → OLS, Means → t-tests) or <b>Graph ▾</b> "
            "(histogram, box plot, scatter ...) - your selection pre-fills the dialog.</li>"
            "<li><b>Codebook</b> shows the variable labels; <b>Terminal</b> opens SQL / Python / Stata / R; "
            "<b>Script</b> shows the reproducible .do / R / Python file.</li>"
            "<li>Not sure which test to use? <b>Help ▸ Stats Flashcards</b> (F2) teaches which test fits which data, and what p-values, z-scores and standard deviations mean.</li>"
            "<li>Open your own data with <b>File → Open data…</b> or drag a file onto the window. "
            "You can reopen the example any time from the File menu.</li></ul>")
        self.open_example()

    def _refresh_recent(self):
        self.recent_menu.clear()
        for p in self.settings.recent_files[:10]:
            self._act(self.recent_menu, Path(p).name, lambda p=p: self.import_path(p), tip=p)
        self.recent_menu.setEnabled(bool(self.settings.recent_files))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            f = u.toLocalFile()
            if f.endswith(PROJECT_EXT):
                self.open_project(f)
            else:
                self.import_path(f)

    # ===================================================================== data menu
    def _new_dataset_from_query(self, base: str, query: str, desc: str, stata: str, r: str, python: str):
        try:
            ds = self.engine.create_from_query(base, query, source=desc)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Data", str(e).splitlines()[0])
            return
        self.modified.add(ds.name)
        old = self.current_ds()
        if old:                                         # keep labels
            by = {c.name: c for c in old.columns}
            for c in ds.columns:
                if c.name in by:
                    c.label, c.value_labels = by[c.name].label, by[c.name].value_labels
        self.add_dataset_tab(ds.name)
        self.recorder.transform(desc, stata=stata, r=r, python=python)
        self._recorded()

    def compute_column(self):
        if not self._need_dataset():
            return
        ds = self.current_ds()
        name, ok = QInputDialog.getText(self, "Compute column", "New column name:")
        if not ok or not name.strip():
            return
        expr, ok = QInputDialog.getText(self, "Compute column", f"Expression (SQL, e.g.  log(income) / age ):\nColumns: {', '.join(ds.names[:12])}…", text="")
        if not ok or not expr.strip():
            return
        try:
            self.engine.add_column(ds.name, name.strip(), expr)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Compute column", str(e).splitlines()[0])
            return
        self.modified.add(ds.name)
        self.refresh_dataset(ds.name)
        n = name.strip()
        self.recorder.transform(f"Computed column {n} = {expr}   (SQL expression)",
                                stata=f"* NOTE: translate to Stata:  generate {n} = {expr}",
                                r=f"# NOTE: translate to R:  df${n} <- {expr}",
                                python=f'import duckdb\ndf = duckdb.sql("SELECT *, ({expr}) AS \\"{n}\\" FROM df").df()')
        self._recorded()

    def filter_rows(self):
        if not self._need_dataset():
            return
        ds = self.current_ds()
        cond, ok = QInputDialog.getText(self, "Filter rows", "Keep rows where (SQL condition, e.g.  age >= 18 AND sex = 'F'):")
        if not ok or not cond.strip():
            return
        self._new_dataset_from_query(f"{ds.name}_filtered", f"SELECT * FROM {qid(ds.name)} WHERE {cond}",
                                     f"Filtered {ds.name}: {cond}", f"* keep if {cond}   (translate the SQL condition)",
                                     f"# df <- subset(df, {cond})   (translate the SQL condition)",
                                     f'import duckdb\ndf = duckdb.sql("SELECT * FROM df WHERE {cond}").df()')

    def sort_rows(self):
        if not self._need_dataset():
            return
        ds = self.current_ds()
        sh = self.current_sheet()
        sel = sh.selected_columns() if sh else []
        col, ok = QInputDialog.getItem(self, "Sort", "Sort by column:", ds.names, ds.names.index(sel[0]) if sel else 0, False)
        if not ok:
            return
        direction, ok = QInputDialog.getItem(self, "Sort", "Order:", ["ascending", "descending"], 0, False)
        if not ok:
            return
        d = "DESC" if direction == "descending" else "ASC"
        self._new_dataset_from_query(f"{ds.name}_sorted", f"SELECT * FROM {qid(ds.name)} ORDER BY {qid(col)} {d} NULLS LAST",
                                     f"Sorted {ds.name} by {col} {direction}", f"gsort {'-' if d == 'DESC' else '+'}{col}",
                                     f"df <- df[order(df${col}{', decreasing = TRUE' if d == 'DESC' else ''}), ]",
                                     f'df = df.sort_values("{col}", ascending={d == "ASC"})')

    def column_summary(self):
        if not self._need_dataset():
            return
        sh, ds = self.current_sheet(), self.current_ds()
        cols = sh.selected_columns() or ds.names
        num = [c for c in cols if ds.col(c).is_numeric]
        if not num:
            QMessageBox.information(self, "Summary", "Select at least one numeric column.")
            return
        self.run_analysis_dialog("summary", preselect=num, auto=True)

    def dataset_info(self):
        if not self._need_dataset():
            return
        ds = self.current_ds()
        lines = [f"Dataset: {ds.name}", f"Source : {ds.source}", f"Rows   : {ds.n_rows:,}", f"Columns: {len(ds.columns)}", ""]
        w = max(len(c.name) for c in ds.columns)
        for c in ds.columns:
            lines.append(f"  {c.name:<{w}}  {c.sql_type:<12} {c.label}")
        lines += ["", f"Engine: DuckDB, memory limit {self.engine.memory_gb:g} GB, {self.engine.threads} threads"]
        self._show_text(f"Dataset information - {ds.name}", "\n".join(lines))

    def open_codebook(self):
        ds = self.current_ds()
        if not ds or not has_codebook(ds):
            QMessageBox.information(self, "Codebook", "No codebook is attached to this dataset.\n(Stata/SPSS files carry variable and value "
                                    "labels; for other files place e.g. 'mydata.codebook.txt' next to the data file.)")
            return
        dlg = CodebookDialog(build_codebook_text(ds, self.engine), ds.codebook.get("sidecar", ""), self, ds.name)
        self.windows.append(dlg)
        dlg.show()

    # ===================================================================== analysis
    def _show_text(self, title: str, text: str, log: bool = False):
        if log:
            stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.results_log.append(f"[{stamp}]  dataset: {self.current_name()}\n{text}\n")
        w = TextWindow(title, text, self, suggested=f"{title.split(':')[0].replace(' ', '_')}.txt")
        self.windows = [x for x in self.windows if x.isVisible()] + [w]
        w.show()
        w.raise_()

    def run_analysis_dialog(self, aid: str, preselect: list[str] | None = None, auto: bool = False):
        if not self._need_dataset():
            return
        spec = ANALYSES[aid]
        ds, sh = self.current_ds(), self.current_sheet()
        pre = preselect if preselect is not None else (sh.selected_columns() if sh else [])
        dlg = VariableDialog(spec.name, spec.roles, spec.params, ds, pre, spec.description, self)
        dlg.add_learn_button(lambda: self.show_flashcards(f"test:{aid}", True))
        if auto:
            sel0 = dlg.selection()
            if all(sel0.get(r.name) for r in spec.roles if r.required):
                dlg.accept()
        if not auto and dlg.exec() != QDialog.DialogCode.Accepted:
            return
        sel, params, name = dlg.selection(), dlg.values(), ds.name
        self.msg.setText(f"Running {spec.name}…")
        self.bar.setRange(0, 0)
        self.bar.show()

        def done(res):
            self.bar.hide()
            self.msg.setText("Ready")
            self._show_text(res.title, res.to_text(), log=True)
            self._recorded()

        def fail(msg, tb):
            self.bar.hide()
            self.msg.setText("Ready")
            QMessageBox.warning(self, spec.name, msg)

        run_async(run_analysis, done, fail, parent=self, engine=self.engine, dataset=name, analysis_id=aid, selection=sel,
                  params=params, recorder=self.recorder)

    # ===================================================================== graphs
    def graph_dialog(self, cid: str):
        if not self._need_dataset():
            return
        spec = CHARTS[cid]
        ds, sh = self.current_ds(), self.current_sheet()
        dlg = VariableDialog(spec.name, spec.roles, spec.params, ds, sh.selected_columns() if sh else [], spec.description, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._open_graph(cid, dlg.selection(), dlg.values(), ds.name)

    def _open_graph(self, cid: str, selection: dict, params: dict, dataset: str, existing: GraphWindow | None = None):
        spec = CHARTS[cid]
        self.msg.setText("Preparing graph…")
        self.bar.setRange(0, 0)
        self.bar.show()

        def work():
            return load_roles(self.engine, dataset, spec.roles, selection, sample=MAX_CHART_ROWS, seed=self.settings.seed)

        def done(res):
            self.bar.hide()
            self.msg.setText("Ready")
            data, info = res
            if info["sampled"]:
                data.notes.append(f"random sample of {MAX_CHART_ROWS:,} of {info['n_total']:,} rows (seed {self.settings.seed})")
            if existing is not None:
                existing.replace_data(data, selection, params)
                return
            kw = {p.name: p.default for p in spec.params}
            kw.update(params)
            win = GraphWindow(cid, data, kw, ChartStyle.for_theme(self.settings.theme), selection, dataset,
                              on_change_variables=self._change_graph_vars, parent=self)
            self.windows = [w for w in self.windows if w.isVisible()] + [win]
            win.show()
            self.recorder.chart(cid, data.names, spec.name, kw)
            self._recorded()

        def fail(msg, tb):
            self.bar.hide()
            self.msg.setText("Ready")
            QMessageBox.warning(self, spec.name, msg)

        run_async(work, done, fail, parent=self)

    def _change_graph_vars(self, win: GraphWindow):
        ds = self.engine.datasets[win.dataset]
        dlg = VariableDialog(win.spec.name, win.spec.roles, [], ds, [], "", win)
        # pre-fill with current choice
        for r in win.spec.roles:
            cur = win.selection.get(r.name)
            w = dlg.role_w[r.name]
            if r.multi:
                for i in range(w.count()):
                    w.item(i).setSelected(w.item(i).text() in (cur or []))
            elif cur:
                w.setCurrentText(cur)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._open_graph(win.chart_id, dlg.selection(), {}, win.dataset, existing=win)

    def _terminal_figure(self, cid: str, selection: dict, params: dict):
        if cid in CHARTS and self.current_name():
            self._open_graph(cid, selection, params, self.current_name())

    # ===================================================================== settings / script / help
    def show_settings(self):
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        old_theme = self.settings.theme
        self.settings = dlg.apply_to(self.settings)
        self.settings.save()
        theme.apply_theme(QApplication.instance(), self.settings.theme, self.settings.font_size)
        stats_base.DEFAULT_DECIMALS[0] = self.settings.decimals
        self.recorder.seed = self.settings.seed
        try:
            self.engine.set_limits(self.settings.ram_gb, self.settings.cores)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Settings", f"Could not apply resource limits: {e}")
        for _, v in self.sheets.values():
            v.verticalHeader().setDefaultSectionSize(self.settings.grid_row_height)
        self._update_res_label()
        if self.script_win:
            self.script_win.settings = self.settings
        self.flash("Settings saved")

    def show_script(self):
        if self.script_win is None:
            self.script_win = ScriptWindow(self.recorder, self.settings, self)
        self.script_win.regenerate(force=not self.script_win.dirty)
        self.script_win.show()
        self.script_win.raise_()

    def show_flashcards(self, card_id: str | None = None, flipped: bool = False):
        from .flashcards import FlashcardDialog
        if getattr(self, "flash_win", None) is None:
            self.flash_win = FlashcardDialog(self, on_open_analysis=self.run_analysis_dialog)
        self.flash_win.show()
        self.flash_win.raise_()
        self.flash_win.activateWindow()
        if card_id:
            self.flash_win.show_card_by_id(card_id, flipped)

    def show_methods(self):
        doc = Path(__file__).resolve().parents[2] / "docs" / "METHODS.md"
        text = doc.read_text(encoding="utf-8") if doc.exists() else "See docs/METHODS.md in the project folder."
        self._show_text("Methods & verification", text)

    def about(self):
        QMessageBox.about(self, f"About {APP_NAME}", f"<b>{APP_NAME}</b> {__version__}<br>Offline data analysis: spreadsheet, "
                          f"{len(ANALYSES)} statistical analyses, {len(CHARTS)} chart types, SQL/Python/Stata/R terminal, "
                          "reproducible .do/.R/.py scripts.<br>Engine: DuckDB · UI: Qt (PySide6).")

    # ===================================================================== export / projects
    def export_dataset(self):
        if not self._need_dataset():
            return
        ds = self.current_ds()
        path, _ = QFileDialog.getSaveFileName(self, "Export dataset", f"{ds.name}.csv",
                                              "CSV (*.csv);;Parquet (*.parquet);;Excel (*.xlsx);;Stata (*.dta);;JSON (*.json)")
        if not path:
            return
        self.msg.setText("Exporting…")
        self.bar.setRange(0, 0)
        self.bar.show()

        def done(_):
            self.bar.hide(); self.msg.setText("Ready"); self.flash(f"Exported to {path}")

        def fail(msg, tb):
            self.bar.hide(); self.msg.setText("Ready"); QMessageBox.warning(self, "Export", msg)

        run_async(self.engine.export, done, fail, parent=self, name=ds.name, path=path)

    def save_results(self):
        if not self.results_log:
            QMessageBox.information(self, "Results", "No results yet.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save all results", "results.txt", "Text files (*.txt)")
        if path:
            Path(path).write_text(f"{APP_NAME} results log\n\n" + "\n".join(self.results_log), encoding="utf-8")

    def save_project(self, as_new: bool = False):
        if not self.sheets:
            QMessageBox.information(self, "Save project", "Nothing to save yet.")
            return
        path = self.project_path
        if as_new or not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save project", "project" + PROJECT_EXT, f"Statistic Smith project (*{PROJECT_EXT})")
            if not path:
                return
        if not path.endswith(PROJECT_EXT):
            path += PROJECT_EXT
        base = Path(path)
        datadir = base.with_name(base.stem + "_data")
        items = []
        todo = []
        for name in list(self.sheets):
            ds = self.engine.datasets[name]
            entry = {"name": name, "labels": {c.name: c.label for c in ds.columns if c.label}, "codebook": ds.codebook, "file_label": ds.file_label}
            src = Path(ds.source) if ds.source else None
            if name not in self.modified and src and src.exists():
                entry.update(kind="file", path=str(src.resolve()), rel=_rel(src, base.parent))
                sheet = next((s.data.get("sheet") for s in self.recorder.steps if s.kind == "import" and s.data.get("dataset") == name), None)
                entry["sheet"] = sheet
            else:
                datadir.mkdir(exist_ok=True)
                target = datadir / f"{name}.parquet"
                entry.update(kind="parquet", path=str(target.resolve()), rel=_rel(target, base.parent))
                todo.append((name, str(target)))
            items.append(entry)
        proj = {"version": 1, "app": f"{APP_NAME} {__version__}", "saved": _dt.datetime.now().isoformat(timespec="seconds"),
                "active": self.current_name(), "datasets": items, "script": json.loads(self.recorder.to_json()),
                "results": self.results_log, "seed": self.settings.seed}
        self.msg.setText("Saving project…")
        self.bar.setRange(0, 0)
        self.bar.show()

        def work():
            for n, t in todo:
                self.engine.export(n, t)
            base.write_text(json.dumps(proj, indent=1), encoding="utf-8")
            script = None
            if self.settings.do_autosave_with_project:
                lang = self.settings.do_language
                ext = {"stata": "do", "r": "R", "python": "py"}[lang]
                script = base.with_suffix("." + ext)
                if self.script_win and self.script_win.dirty and self.script_win.language == lang:
                    script.write_text(self.script_win.edit.toPlainText(), encoding="utf-8")
                else:
                    self.recorder.project = base.stem
                    self.recorder.save_script(str(script), lang)
            return script

        def done(script):
            self.bar.hide(); self.msg.setText("Ready")
            self.project_path = str(base)
            self.setWindowTitle(f"{APP_NAME} - {base.name}")
            self.modified.clear() if False else None
            self.flash(f"Project saved: {base.name}" + (f"  +  script {script.name}" if script else ""), 10000)

        def fail(msg, tb):
            self.bar.hide(); self.msg.setText("Ready"); QMessageBox.critical(self, "Save project", msg)

        run_async(work, done, fail, parent=self)

    def open_project(self, path: str | None = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open project", "", f"Statistic Smith project (*{PROJECT_EXT})")
        if not path:
            return
        try:
            proj = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Open project", f"Not a valid project file: {e}")
            return
        base = Path(path)
        self.project_path = str(base)
        self.recorder = Recorder.from_json(json.dumps(proj["script"]))
        self.terminal.recorder = self.recorder
        if self.script_win:
            self.script_win.rec = self.recorder
        self.results_log = list(proj.get("results", []))
        queue = list(proj["datasets"])

        def next_one(_=None):
            if not queue:
                a = proj.get("active")
                if a:
                    self._focus_dataset(a)
                self.setWindowTitle(f"{APP_NAME} - {base.name}")
                self._recorded()
                return
            e = queue.pop(0)
            p = Path(e["path"])
            if not p.exists():
                alt = (base.parent / e["rel"]).resolve()
                if alt.exists():
                    p = alt
                else:
                    f, _ = QFileDialog.getOpenFileName(self, f"Locate data file for '{e['name']}'", "", FILTER)
                    if not f:
                        next_one()
                        return
                    p = Path(f)

            def after(ds, e=e):
                for c in ds.columns:
                    c.label = c.label or e.get("labels", {}).get(c.name, "")
                if e["kind"] == "parquet":
                    self.modified.add(ds.name)
                    ds.codebook = e.get("codebook", {}) or ds.codebook
                    ds.file_label = e.get("file_label", "")
                next_one()

            self.import_path(str(p), name=e["name"], options={"sheet": e["sheet"]} if e.get("sheet") else {}, record=False, done=after)

        next_one()

    # ===================================================================== shutdown
    def closeEvent(self, ev):
        try:
            self.settings.save()
            for w in list(self.windows):
                w.close()
            wd = self.engine.workdir
            self.engine.close()
            shutil.rmtree(wd, ignore_errors=True)
        except Exception:
            pass
        super().closeEvent(ev)


def _rel(p: Path, start: Path) -> str:
    try:
        import os
        return os.path.relpath(p, start)
    except Exception:
        return p.name
