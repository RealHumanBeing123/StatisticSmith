"""Application entry point (also used by the PyInstaller build)."""
from __future__ import annotations

import multiprocessing
import sys
import traceback
from pathlib import Path

from . import APP_NAME


def examples_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "examples"


def log_path() -> Path:
    from .settings import config_dir
    return config_dir() / "error.log"


def selftest(paths: list[str]) -> int:
    """Import every given file (default: the bundled examples) with the real engine and report. Used to verify a build."""
    import tempfile
    from .data import Engine
    files = paths or [str(p) for p in sorted(examples_dir().glob("trial.*")) if p.suffix in (".csv", ".dta")]
    out, bad = [], 0
    eng = Engine(1, 2, workdir=tempfile.mkdtemp(prefix="ss_selftest_"))
    for f in files:
        try:
            ds = eng.import_file(f)
            rows = eng.fetch(ds.name, 0, 2)
            out.append(f"OK   {Path(f).name}: {ds.n_rows} rows x {len(ds.columns)} cols, first row {rows[0] if rows else None}")
        except Exception:  # noqa: BLE001
            bad += 1
            out.append(f"FAIL {Path(f).name}\n{traceback.format_exc()}")
    from .stats import REGISTRY
    from .charts import REGISTRY as CH
    out.append(f"{len(REGISTRY)} analyses, {len(CH)} charts registered")
    text = "\n".join(out)
    try:
        log_path().with_name("selftest.txt").write_text(text, encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if sys.stdout:
        print(text)
    return 1 if bad else 0


def gui_selftest(paths: list[str]) -> int:
    """Start the real window, import files through the GUI code path, draw a graph and run a test. Exit code 0 = all good."""
    import time
    from PySide6.QtWidgets import QApplication, QMessageBox
    from .settings import Settings
    from .ui.main_window import MainWindow
    errors: list[str] = []
    QMessageBox.critical = staticmethod(lambda *a, **k: errors.append(str(a[1:])))
    QMessageBox.warning = staticmethod(lambda *a, **k: errors.append(str(a[1:])))
    app = QApplication(sys.argv[:1])
    win = MainWindow(Settings())
    win.show()
    files = paths or [str(examples_dir() / "trial.csv"), str(examples_dir() / "trial.dta")]
    report = []
    for f in files:
        before = set(win.sheets)
        win.import_path(f)
        t = time.time()
        while set(win.sheets) == before and not errors and time.time() - t < 60:
            app.processEvents(); time.sleep(0.01)
        new = set(win.sheets) - before
        if new:
            n = new.pop()
            m = win.sheets[n][0]
            report.append(f"OK   {Path(f).name}: grid shows {m.rowCount()} rows x {m.columnCount()} cols, cell(0,3)={m.data(m.index(0, 3))!r}")
        else:
            report.append(f"FAIL {Path(f).name}: {errors[-1] if errors else 'nothing loaded'}")
    ok = all(r.startswith("OK") for r in report)
    if ok:
        from .runner import load_roles, run_analysis
        from .charts import REGISTRY as CH
        from .charts import render
        name = next(iter(win.sheets))
        res = run_analysis(win.engine, name, "summary", {"xs": ["age", "sbp"]}, {}, win.recorder) if "sbp" in win.engine.datasets[name].names else None
        if res is not None:
            report.append("OK   analysis ran: " + res.title)
            data, _ = load_roles(win.engine, name, CH["histogram"].roles, {"x": "sbp"})
            from matplotlib.figure import Figure
            render("histogram", data, fig=Figure())
            report.append("OK   histogram drawn")
    text = "\n".join(report + errors)
    try:
        log_path().with_name("gui_selftest.txt").write_text(text, encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    if sys.stdout:
        print(text)
    return 0 if ok else 1


def _install_excepthook(app):
    from PySide6.QtWidgets import QMessageBox

    def hook(et, ev, tb):
        text = "".join(traceback.format_exception(et, ev, tb))
        try:
            log_path().write_text(text, encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        try:
            QMessageBox.critical(None, f"{APP_NAME} - unexpected error", f"{ev}\n\nDetails were saved to:\n{log_path()}")
        except Exception:  # noqa: BLE001
            pass
    sys.excepthook = hook


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    argv = list(sys.argv if argv is None else argv)
    if "--selftest" in argv:
        return selftest([a for a in argv[1:] if a != "--selftest"])
    if "--gui-selftest" in argv:
        return gui_selftest([a for a in argv[1:] if not a.startswith("--")])
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from .settings import Settings
    from .ui import theme
    from .ui.main_window import MainWindow
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    _install_excepthook(app)
    s = Settings.load()
    first_run = not s.welcomed
    theme.apply_theme(app, s.theme, s.font_size)
    win = MainWindow(s)
    win.show()
    args = [a for a in argv[1:] if not a.startswith("--")]
    for a in args:
        win.open_project(a) if a.endswith(".ssproj") else win.import_path(a)
    if first_run and not args:
        QTimer.singleShot(300, win.welcome)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
