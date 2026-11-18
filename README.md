# Statistic Smith

Offline, Windows-friendly desktop data analysis: a user friendly spreadsheet front end on top of a DuckDB engine that streams and spills to the disk, so multi-gigabyte files can be imported without freezing the window.

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


## Screenshots

<img width="623" height="1041" alt="SS1" src="https://github.com/user-attachments/assets/9880b888-252a-4d38-8343-df8c400b1086" />

<img width="413" height="243" alt="ss2" src="https://github.com/user-attachments/assets/6ac25dca-de78-40d5-bab3-507d950c2306" />

<img width="306" height="152" alt="ss3" src="https://github.com/user-attachments/assets/119d55d9-a746-4fad-935d-2ac06de4dfa5" />

<img width="761" height="456" alt="ss4" src="https://github.com/user-attachments/assets/ea9306fe-fb3f-49c6-bc64-127b81c2ae94" />

<img width="894" height="661" alt="ss5" src="https://github.com/user-attachments/assets/9481befb-95e9-41a8-977a-a717869424c5" />

<img width="1920" height="782" alt="ss6" src="https://github.com/user-attachments/assets/514481ed-5e10-4a68-9044-dada3c083969" />

<img width="380" height="396" alt="ss7" src="https://github.com/user-attachments/assets/403f01f3-ef2e-4308-9eeb-507a16cfce44" />

## Honest limitations
See *Known limits* in [docs/METHODS.md](docs/METHODS.md): the generated Stata/R scripts and the Windows build were not executed in the
development environment (no Stata, R or Windows there); the Python script and all numerical results are tested.
Stata/R terminals are command subsets, not interpreters.

## StatisticSmith versus [P Value Pro (PVP)](https://github.com/RealHumanBeing123/P-Value-Pro):
Currently, the pros and cons of both of these EDA tools are quite different. The UI is drastically different from one to the other. StatisticSmith is made to be more like excel but still maintaining user friendliness and some familiarity with excel. StatisticSmith also has better support for industry standard files like .do files used in STATA. PVP is far more user friendly, but as of the time of making this project PVP's math and graphs are not yet up to par for the repo to be made public. Both StatisticSmith and PVP allow you to use a terminal that accepts SQL and python code as per the industry standards. StatisticSmith unlike PVP has the RAM/CPU core settings, so PVP will automatically choose to use up to 90% of your resources while StatisticSmith let's the user choose how many resources should be allocated. The default settings in StatisticSmith makes things harder to read, meanwhile PVP's default settings are easier to read and has a much more pleasant darkmode. 

## Data & licences
Code: MIT. Bundled map data: US states (US Census Bureau, via PublicaMundi/MappingAPI), world countries (Natural Earth, public domain,
via johan/world.geo.json), US tile-grid layout (hafen/grid-designer `us_state_grid1`).
