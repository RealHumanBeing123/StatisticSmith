@echo off
REM ============================================================================
REM  Build Statistic Smith for Windows.   Double-click this file or run it from a
REM  Command Prompt.  The window STAYS OPEN at the end, and everything is also
REM  saved to build_log.txt in the project folder so errors can be read.
REM  Needs Python 3.10 - 3.13 (3.12 recommended) from python.org, "Add to PATH" ticked.
REM ============================================================================
setlocal EnableExtensions
cd /d "%~dp0.."
set "LOG=%CD%\build_log.txt"
echo Statistic Smith build  %DATE% %TIME% > "%LOG%"

call :say "Step 1/6: looking for Python..."
set "PY="
for %%V in (3.12 3.13 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  call :say "ERROR: Python was not found. Install Python 3.12 from https://www.python.org/downloads/ and tick 'Add python.exe to PATH'."
  goto :fail
)
%PY% -c "import sys; print('Using Python', sys.version.split()[0]); sys.exit(1 if sys.version_info >= (3,14) else 0)" >> "%LOG%" 2>&1
if errorlevel 1 (
  call :say "WARNING: Python 3.14 or newer detected. Some libraries (PySide6, DuckDB, pandas...) may not support it yet."
  call :say "         If a step below fails, install Python 3.12 and run this file again."
)

call :say "Step 2/6: creating virtual environment (.venv)..."
if not exist ".venv\Scripts\python.exe" (
  %PY% -m venv .venv >> "%LOG%" 2>&1
  if errorlevel 1 ( call :say "ERROR: could not create the virtual environment." & goto :fail )
)
set "VPY=%CD%\.venv\Scripts\python.exe"

call :say "Step 3/6: installing requirements (first time takes a few minutes)..."
"%VPY%" -m pip install --upgrade pip >> "%LOG%" 2>&1
"%VPY%" -m pip install -r requirements-dev.txt >> "%LOG%" 2>&1
if errorlevel 1 ( call :say "ERROR: installing requirements failed (see build_log.txt, last lines below)." & goto :fail )

call :say "Step 4/6: running tests (a failure here will NOT stop the build)..."
set QT_QPA_PLATFORM=offscreen
"%VPY%" -m pytest -q -x >> "%LOG%" 2>&1
if errorlevel 1 ( call :say "WARNING: some tests failed on this computer. Continuing with the build. Please send build_log.txt to the developer." ) else ( call :say "Tests passed." )
set QT_QPA_PLATFORM=

call :say "Step 5/6: building the app with PyInstaller (several minutes)..."
"%VPY%" -m PyInstaller --noconfirm --clean statistic_smith.spec >> "%LOG%" 2>&1
if errorlevel 1 ( call :say "ERROR: PyInstaller failed (see build_log.txt, last lines below)." & goto :fail )
if not exist "dist\StatisticSmith\StatisticSmith.exe" ( call :say "ERROR: build finished but dist\StatisticSmith\StatisticSmith.exe was not created." & goto :fail )

call :say "Step 6/6: self-test of the built app (imports the bundled CSV and .dta)..."
"dist\StatisticSmith\StatisticSmith.exe" --gui-selftest >> "%LOG%" 2>&1
if errorlevel 1 (
  call :say "SELF-TEST FAILED. Report:"
  type "%APPDATA%\StatisticSmith\gui_selftest.txt"
  type "%APPDATA%\StatisticSmith\gui_selftest.txt" >> "%LOG%" 2>&1
  goto :fail
)
type "%APPDATA%\StatisticSmith\gui_selftest.txt"
echo.
call :say "SUCCESS.  Run:  dist\StatisticSmith\StatisticSmith.exe   (keep the whole dist\StatisticSmith folder together)"
echo.
pause
exit /b 0

:say
echo %~1
echo %~1 >> "%LOG%"
exit /b 0

:fail
echo.
echo ---------------- last lines of build_log.txt ----------------
powershell -NoProfile -Command "Get-Content -Tail 25 '%LOG%'"
echo --------------------------------------------------------------
echo The full log is here: %LOG%
echo Please send that file (or a screenshot of the lines above) to get help.
echo.
pause
exit /b 1
