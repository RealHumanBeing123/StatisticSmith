# Statistic Smith

Offline, Windows-friendly desktop data analysis: a StatCrunch-style spreadsheet front end on top of a DuckDB engine that
streams and spills to disk, so multi-gigabyte files import without freezing the window.

* **Spreadsheet grid** - click a column/row header to select it, drag to select (and expand) a rectangle, Ctrl+C copies, cells are editable.
  The status bar shows count / sum / mean / min / max of the selection.
* **Import** - CSV/TSV/TXT, Excel (.xlsx/.xls/.ods), Stata `.dta`, R `.rds`/`.RData`, SPSS `.sav`, SAS `.sas7bdat`, Parquet, JSON, Feather.
  Import runs in a background thread with a progress bar and a Cancel button; rows stay in file order.
* **Analysis menu** - 77 analyses: every test in Wikipedia's *List of statistical tests*, plus regression (OLS, polynomial, logit, Poisson),
  multivariate (PCA, MANOVA, LDA, clustering, ...) and time-series / quality methods. Results open in a pop-up and can be saved as `.txt`.
* **Graph menu** - 90 chart types grouped like the Datylon article (Comparison, Correlation, Part-to-whole & hierarchical, Data over time,
  Distribution, Geospatial & other) plus a Statistical group (Pareto, Q-Q, ECDF, control chart, ...).
* **Graph editor** - titles, axis titles, palettes and per-series colours, fonts and sizes, scales/limits, grid, legend, value labels; export PNG/SVG/PDF.
* **Terminal** - SQL (DuckDB), Python, and documented subsets of Stata and R commands (type `help`).
* **Reproducible scripts** - everything you do is recorded and can be shown as a Stata `.do`, R, or Python script. The script is editable, and is also
  written next to the project when you save it (Settings).
* **Codebook** - Stata/SPSS variable & value labels, or a sidecar file such as `mydata.codebook.txt`, open from the Codebook button.
* **Settings** - light/dark mode, RAM limit (default 2 GB) and CPU cores (default 4), adjustable up to what the machine has, decimals, seed.

## Run from source
```
python -m venv .venv && .venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python main.py [datafile or project.ssproj]
```
Tests: `pip install -r requirements-dev.txt && python -m pytest -q` (set `QT_QPA_PLATFORM=offscreen` on headless machines).

## Build a Windows app
`scripts\build_windows.bat` (or the *windows-build* job in `.github/workflows/ci.yml`) runs PyInstaller with `statistic_smith.spec`
and produces `dist\StatisticSmith\StatisticSmith.exe`.

## Honest limitations
See *Known limits* in [docs/METHODS.md](docs/METHODS.md): the generated Stata/R scripts and the Windows build were not executed in the
development environment (no Stata, R or Windows there); the Python script and all numerical results are tested.
Stata/R terminals are command subsets, not interpreters.

## Data & licences
Code: MIT. Bundled map data: US states (US Census Bureau, via PublicaMundi/MappingAPI), world countries (Natural Earth, public domain,
via johan/world.geo.json), US tile-grid layout (hafen/grid-designer `us_state_grid1`).
