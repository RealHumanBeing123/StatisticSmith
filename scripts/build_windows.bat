@echo off
REM Build a Windows folder-app:  scripts\build_windows.bat   (needs Python 3.10+ on PATH)
python -m venv .venv || exit /b 1
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-dev.txt || exit /b 1
python -m pytest -q || exit /b 1
pyinstaller --noconfirm statistic_smith.spec || exit /b 1
echo Verifying the built app (imports the bundled CSV and .dta through the real window code)...
dist\StatisticSmith\StatisticSmith.exe --gui-selftest
if errorlevel 1 (
  echo SELF-TEST FAILED. Details:
  type "%APPDATA%\StatisticSmith\gui_selftest.txt"
  exit /b 1
)
type "%APPDATA%\StatisticSmith\gui_selftest.txt"
echo.
echo Built: dist\StatisticSmith\StatisticSmith.exe
