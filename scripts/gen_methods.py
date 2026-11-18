"""Regenerate docs/METHODS.md from the registries (analysis references + chart list)."""
from pathlib import Path

from statsmith.charts import specs_by_category
from statsmith.stats import by_category

HEAD = """# Methods & verification

Everything below is generated from the code (`python scripts/gen_methods.py`). Each analysis states the formula or library routine it uses.

## How results are verified

* **Reference values** - `tests/test_stats_reference.py` reproduces published examples (R documentation datasets: sleep, mtcars; Fisher's tea-tasting;
  NIST Grubbs example) and compares to independent SciPy / statsmodels / closed-form calculations.
* **Multivariate & time series** - `tests/test_stats_multivariate.py` checks PCA, Bartlett/KMO, Hotelling T² (equals the t-test for one variable), MANOVA (Wilks' Λ from the
  SSCP definition), LDA, Cronbach's α, canonical correlation (vs statsmodels), Ljung–Box / ACF (vs statsmodels), ADF/KPSS, ARIMA, Mann–Kendall, runs, capability.
* **Charts** - `tests/test_charts_*.py` read the artists that were actually drawn (bar heights, polygon vertices, line data, tile counts, Sankey link widths,
  chord arc angles ...) and compare them with independent numpy computations; a smoke test renders every chart with normal, NaN-containing, tiny and dark-theme data.
  `tests/test_chart_coverage.py` asserts every chart in the Datylon article exists under the article's category.
* **Reproducible scripts** - `tests/test_codegen.py` executes the generated Python script and checks it gives the same numbers as the app, and that every Stata / R / Python
  template resolves with no leftover placeholders.
* **Large data** - `tests/test_engine_terminal.py` imports multi-million-row CSVs, checks file order is preserved, cancellation, memory/thread limits and all file formats.

### Known limits (please read)

* The generated **Stata and R scripts could not be executed in the build environment** (neither Stata nor R was installed). They are template-generated and were checked
  for resolved placeholders and syntax-by-construction only; the Python script *is* executed in the tests. Run the `.do` / `.R` file in Stata / R to confirm - that is its purpose.
* The Windows executable build (`statistic_smith.spec`, GitHub Actions) was prepared but **not run on Windows** by the author of this code.
* The Stata and R terminals implement a documented **subset** of commands (type `help` in the terminal), not full interpreters.
* Randomised methods (k-means, permutation tests, chart sampling) use the seed in Settings; other software can reach different (equally valid) local optima.
* Charts that draw one mark per row use a reproducible random sample of 200,000 rows for speed; all statistics use every row.

"""


def main():
    out = [HEAD, "## Statistical analyses\n"]
    n = 0
    for cat, specs in by_category().items():
        if not specs:
            continue
        out.append(f"### {cat}\n")
        for s in sorted(specs, key=lambda s: s.name):
            n += 1
            out.append(f"* **{s.name}** (`{s.id}`) - {s.reference or s.description}")
        out.append("")
    out.insert(2, f"**{n} analyses and {sum(len(v) for v in specs_by_category().values())} chart types are registered.**\n")
    out.append("## Charts\n")
    for cat, specs in specs_by_category().items():
        out.append(f"### {cat} ({len(specs)})\n")
        out.append(", ".join(s.name for s in specs) + "\n")
    Path(__file__).resolve().parents[1].joinpath("docs", "METHODS.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
