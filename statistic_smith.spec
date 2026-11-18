# PyInstaller spec: pyinstaller statistic_smith.spec   (run on Windows to build a Windows app)
import os, sys
sys.path.insert(0, os.path.abspath('.'))     # so collect_* can import statsmith (it is not pip-installed)
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hidden = (collect_submodules("statsmith") + collect_submodules("statsmodels.tsa") + collect_submodules("statsmodels.multivariate")
          + ["scipy.special._cdflib", "pyreadstat._readstat_parser", "pyreadstat._readstat_writer", "pyreadstat.worker", "fastexcel", "polars", "pyarrow"])
datas = [("statsmith/data/geo", "statsmith/data/geo"), ("statsmith/data/examples", "statsmith/data/examples")] + collect_data_files("matplotlib") + [("docs", "docs")]
excludes = ["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick", "PySide6.QtQml", "IPython", "pytest"]

a = Analysis(["main.py"], pathex=["."], datas=datas, hiddenimports=hidden, excludes=excludes)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="StatisticSmith", console=False)
coll = COLLECT(exe, a.binaries, a.datas, name="StatisticSmith")
