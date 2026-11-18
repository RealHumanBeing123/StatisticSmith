"""Headless end-to-end tests of the Qt UI (offscreen platform)."""
import json
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from statsmith.data import Engine
from statsmith.settings import Settings
from statsmith.ui.dialogs import VariableDialog
from statsmith.ui.main_window import MainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def wait(app, cond, timeout=30):
    t = time.time()
    while not cond():
        app.processEvents()
        time.sleep(0.01)
        assert time.time() - t < timeout, "timed out"


@pytest.fixture
def win(app, tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, "path", classmethod(lambda cls: tmp_path / "settings.json"))
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: print("WARN", a[1:]))
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: (_ for _ in ()).throw(AssertionError(a[1:])))
    s = Settings()
    w = MainWindow(s, Engine(1, 2, workdir=str(tmp_path / "w")))
    w.show()
    yield w
    w.close()


@pytest.fixture
def csv(tmp_path):
    rng = np.random.default_rng(3)
    n = 300
    df = pd.DataFrame({"g": rng.choice(["a", "b"], n), "x": rng.normal(size=n), "y": rng.normal(size=n) + 1})
    df.loc[::10, "x"] = np.nan
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    return str(p), df


def accept_dialogs(monkeypatch, selection=None):
    """Make VariableDialog auto-accept with the pre-filled (or given) values."""
    def fake_exec(self):
        if selection:
            for k, v in selection.items():
                w = self.role_w[k]
                if isinstance(v, list):
                    for i in range(w.count()):
                        w.item(i).setSelected(w.item(i).text() in v)
                else:
                    w.setCurrentText(v)
        return QDialog.DialogCode.Accepted
    monkeypatch.setattr(VariableDialog, "exec", fake_exec)


def test_import_grid_and_selection(app, win, csv):
    path, df = csv
    win.import_path(path)
    wait(app, lambda: win.current_name() == "d")
    sh = win.current_sheet()
    assert sh.dmodel.rowCount() == 300 and sh.dmodel.columnCount() == 3
    assert sh.dmodel.data(sh.dmodel.index(1, 1)) == f"{df.x[1]:.10g}"
    sh.selectColumn(1)
    assert sh.selected_columns() == ["x"]
    sh.selectRow(5)
    assert sh.selected_columns() == ["g", "x", "y"]
    sh.clearSelection()
    sh.setRangeSelection = None
    from PySide6.QtCore import QItemSelectionModel, QItemSelection
    m = sh.dmodel
    sh.selectionModel().select(QItemSelection(m.index(2, 1), m.index(9, 2)), QItemSelectionModel.Select)
    assert sh.selected_ranges() == [(2, 9, 1, 2)]
    wait(app, lambda: "numeric n=" in win.sel.text())
    sub = df.iloc[2:10][["x", "y"]]
    assert f"n={int(sub.count().sum())}" in win.sel.text()


def test_cell_edit_and_script(app, win, csv):
    win.import_path(csv[0])
    wait(app, lambda: win.current_name())
    sh = win.current_sheet()
    assert sh.dmodel.setData(sh.dmodel.index(1, 1), "42.5")
    assert win.engine.fetch("d", 1, 1)[0][1] == 42.5
    txt = win.recorder.render("stata")
    assert "replace x = 42.5 in 2" in txt and 'import delimited' in txt


def test_analysis_runs_and_logs(app, win, csv, monkeypatch):
    win.import_path(csv[0])
    wait(app, lambda: win.current_name())
    accept_dialogs(monkeypatch, {"y": "y", "group": "g"})
    n0 = len(win.windows)
    win.run_analysis_dialog("t_welch")
    wait(app, lambda: win.results_log)
    assert "Welch" in win.results_log[0]
    assert any(s.kind == "analysis" for s in win.recorder.steps)
    assert "t.test" in win.recorder.render("r") or "ttest" in win.recorder.render("stata")


def test_every_chart_through_window(app, win, csv, monkeypatch):
    from statsmith.charts import REGISTRY
    win.import_path(csv[0])
    wait(app, lambda: win.current_name())
    accept_dialogs(monkeypatch, {"x": "x", "y": "y"})
    win.graph_dialog("scatter")
    wait(app, lambda: any(w.__class__.__name__ == "GraphWindow" for w in win.windows))
    gw = [w for w in win.windows if w.__class__.__name__ == "GraphWindow"][0]
    gw.w_title.setText("My title")
    gw.redraw()
    assert gw.fig.axes[0].get_title() == "My title"
    assert len(win.graph_menu.actions()) >= 6


def test_menus_follow_article_categories(win):
    names = [a.text().split("  (")[0] for a in win.graph_menu.actions()]
    assert names[:6] == ["Comparison", "Correlation", "Part-to-whole & hierarchical", "Data over time", "Distribution", "Geospatial & other"]
    assert sum(int(a.text().split("(")[1].rstrip(")")) for a in win.graph_menu.actions()) >= 80


def test_terminal_sql_creates_dataset(app, win, csv):
    win.import_path(csv[0])
    wait(app, lambda: win.current_name())
    win.dock.mode.setCurrentIndex(0)
    win.dock.inp.setPlainText("CREATE TABLE big AS SELECT * FROM d WHERE x > 0")
    win.dock.run()
    wait(app, lambda: "big" in win.sheets)
    assert win.engine.datasets["big"].n_rows > 0


def test_project_roundtrip(app, win, csv, tmp_path, monkeypatch):
    win.import_path(csv[0])
    wait(app, lambda: win.current_name())
    sh = win.current_sheet()
    sh.dmodel.setData(sh.dmodel.index(0, 2), "7")            # makes the dataset 'modified' -> stored as parquet
    accept_dialogs(monkeypatch, {"xs": ["y"]})
    win.run_analysis_dialog("summary")
    wait(app, lambda: win.results_log)
    win.project_path = str(tmp_path / "p.ssproj")
    win.save_project()
    wait(app, lambda: (tmp_path / "p.do").exists())
    proj = json.loads((tmp_path / "p.ssproj").read_text())
    assert proj["datasets"][0]["kind"] == "parquet" and (tmp_path / "p_data" / "d.parquet").exists()
    txt = (tmp_path / "p.do").read_text()
    assert "import delimited" in txt and "summarize" in txt
    # reopen in a second window
    w2 = MainWindow(Settings(), Engine(1, 2, workdir=str(tmp_path / "w2")))
    w2.open_project(str(tmp_path / "p.ssproj"))
    wait(app, lambda: w2.current_name() == "d")
    assert w2.sheets["d"][0].value(0, 2) == 7
    assert len(w2.recorder.steps) == len(win.recorder.steps)
    w2.close()


def test_settings_limits_clamped():
    s = Settings(ram_gb=10_000, cores=100_000).clamped()
    import psutil
    assert s.ram_gb <= psutil.virtual_memory().total / 2**30 + 0.01 and s.cores <= psutil.cpu_count()
    assert Settings().ram_gb == 2 and Settings().cores == 4


def test_example_dataset_opens_with_codebook(app, win, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    win.welcome()
    wait(app, lambda: win.current_name() == "trial")
    assert win.settings.welcomed and win.current_sheet().dmodel.rowCount() == 1000
    assert win.codebook_act.isEnabled()


def test_startup_and_closing_all_tabs_keep_widgets_alive(app, win, csv):
    assert win.stack.currentWidget() is win.empty
    win.import_path(csv[0])
    wait(app, lambda: win.current_name())
    assert win.stack.currentWidget() is win.tabs
    win.tabs.removeTab(0); win.sheets.clear(); win._show_empty(True)
    assert win.stack.currentWidget() is win.empty and win.tabs.count() == 0
