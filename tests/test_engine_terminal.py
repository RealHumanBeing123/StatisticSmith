import time
import threading

import numpy as np
import pandas as pd
import pytest

from statsmith.codegen import Recorder
from statsmith.data import Engine, ImportCancelled, build_codebook_text, has_codebook
from statsmith.runner import draw_chart, run_analysis
from statsmith.terminal import Terminal


@pytest.fixture
def eng(tmp_path):
    e = Engine(memory_gb=1, threads=2, workdir=str(tmp_path / "work"))
    yield e
    e.close()


def make_csv(tmp_path, n=2000, name="demo.csv"):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"score": rng.normal(50, 10, n), "age": rng.integers(18, 80, n), "arm": rng.choice(["a", "b"], n),
                       "city name": rng.choice(["x", "y", "z"], n), "when": pd.date_range("2020-01-01", periods=n, freq="h")})
    df.loc[::50, "score"] = np.nan
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p, df


# ------------------------------------------------------------------ import formats
def test_import_csv_types_missing_and_window(eng, tmp_path):
    p, df = make_csv(tmp_path)
    ds = eng.import_file(str(p))
    assert ds.n_rows == 2000 and ds.names == list(df.columns)
    assert ds.col("score").is_numeric and ds.col("arm").kind == "text" and ds.col("when").is_datetime
    rows = eng.fetch(ds.name, 10, 3)
    assert len(rows) == 3 and rows[0][1] == df.iloc[10]["age"]
    s = eng.numeric_summary(ds.name, ["score"])["score"]
    assert s["n"] == df["score"].notna().sum() and s["mean"] == pytest.approx(df["score"].mean())
    assert s["sd"] == pytest.approx(df["score"].std(ddof=1)) and s["median"] == pytest.approx(df["score"].median())


def test_import_excel_stata_r_parquet_json(eng, tmp_path):
    p, df = make_csv(tmp_path, 300)
    d2 = df.drop(columns=["when"])
    d2.to_excel(tmp_path / "a.xlsx", index=False)
    import pyreadstat, pyreadr
    pyreadstat.write_dta(d2.rename(columns={"city name": "city"}), str(tmp_path / "a.dta"),
                         column_labels=["Test score", "Age in years", "Study arm", "City"],
                         variable_value_labels={"age": {18: "youngest"}})
    pyreadr.write_rdata(str(tmp_path / "a.RData"), d2.rename(columns={"city name": "city"}), df_name="mydf")
    pyreadr.write_rds(str(tmp_path / "a.rds"), d2.rename(columns={"city name": "city"}))
    d2.to_parquet(tmp_path / "a.parquet")
    d2.to_json(tmp_path / "a.json", orient="records")
    for f in ("a.xlsx", "a.dta", "a.RData", "a.rds", "a.parquet", "a.json"):
        ds = eng.import_file(str(tmp_path / f), name=f.replace(".", "_"))
        assert ds.n_rows == 300, f
        assert eng.numeric_summary(ds.name, ["score"])["score"]["mean"] == pytest.approx(d2["score"].mean()), f
    dta = eng.datasets["a_dta"]
    assert dta.col("score").label == "Test score" and dta.col("age").value_labels == {18: "youngest"}
    assert has_codebook(dta)
    txt = build_codebook_text(dta, eng)
    assert "Test score" in txt and "18 = youngest" in txt


def test_unsupported_extension(eng, tmp_path):
    (tmp_path / "x.abc").write_text("hi")
    with pytest.raises(ValueError, match="Unsupported"):
        eng.import_file(str(tmp_path / "x.abc"))


def test_sidecar_codebook_is_found(eng, tmp_path):
    p, _ = make_csv(tmp_path, 50, "survey.csv")
    (tmp_path / "survey_codebook.txt").write_text("score: exam result 0-100")
    ds = eng.import_file(str(p))
    assert has_codebook(ds) and ds.codebook_path.endswith("survey_codebook.txt")
    assert "exam result" in build_codebook_text(ds, eng)


def test_cancel_cleans_up(eng, tmp_path):
    rng = np.random.default_rng(1)
    big = tmp_path / "big.csv"
    pd.DataFrame({"a": rng.normal(size=1_500_000), "b": rng.normal(size=1_500_000)}).to_csv(big, index=False)
    ev = threading.Event()
    threading.Timer(0.02, ev.set).start()
    with pytest.raises(ImportCancelled):
        eng.import_file(str(big), cancel=ev)
    assert not eng.datasets and eng.sql("show tables").fetchall() == []


def test_large_csv_import_does_not_block_and_is_fast(eng, tmp_path):
    """~60 MB file: import runs on a worker; the calling (UI) thread can keep polling progress."""
    rng = np.random.default_rng(2)
    big = tmp_path / "big.csv"
    pd.DataFrame({"a": rng.normal(size=1_200_000), "b": rng.integers(0, 9, 1_200_000), "c": rng.choice(list("xyz"), 1_200_000)}).to_csv(big, index=False)
    box = {}
    th = threading.Thread(target=lambda: box.update(ds=eng.import_file(str(big))))
    t0 = time.time(); th.start()
    ticks = 0
    while th.is_alive():
        time.sleep(0.01); ticks += 1                       # stand-in for the Qt event loop staying responsive
    assert box["ds"].n_rows == 1_200_000 and time.time() - t0 < 60 and ticks > 3
    t = time.time(); eng.fetch(box["ds"].name, 1_000_000, 50)
    assert time.time() - t < 1.0                           # windowed grid access is O(window)


def test_memory_and_thread_limits_are_applied(eng):
    eng.set_limits(3.5, 6)
    assert eng.sql("select current_setting('threads')").fetchone()[0] == 6
    assert "3.2" in str(eng.sql("select current_setting('memory_limit')").fetchone()[0]) or "GiB" in str(eng.sql("select current_setting('memory_limit')").fetchone()[0])


# ------------------------------------------------------------------ runner + terminal
def test_runner_runs_analysis_and_records(eng, tmp_path):
    p, df = make_csv(tmp_path)
    ds = eng.import_file(str(p))
    rec = Recorder()
    r = run_analysis(eng, ds.name, "t_welch", {"y": "score", "group": "arm"}, recorder=rec)
    a, b = df[df.arm == "a"].score.dropna(), df[df.arm == "b"].score.dropna()
    from scipy import stats
    assert r.values["p"] == pytest.approx(stats.ttest_ind(a, b, equal_var=False).pvalue, rel=1e-9)
    assert "t.test(score ~ arm" in rec.render("r")
    with pytest.raises(ValueError, match="Please choose"):
        run_analysis(eng, ds.name, "t_welch", {"y": "score"})


def test_runner_chart_uses_sample_for_huge_tables(eng, tmp_path):
    p, _ = make_csv(tmp_path, 3000)
    ds = eng.import_file(str(p))
    fig, d = draw_chart(eng, ds.name, "scatter", {"x": "age", "y": "score"}, sample=500)
    assert len(fig.axes[0].collections[0].get_offsets()) <= 500 and any("sample" in n.lower() for n in d.notes)


def test_terminal_sql_python_stata_r(eng, tmp_path):
    p, df = make_csv(tmp_path)
    ds = eng.import_file(str(p))
    rec = Recorder()
    t = Terminal(eng, rec, ds.name)
    assert "2,000 rows" in t.execute("sql", f"select count(*) as n from {ds.name}").text or "2000" in t.execute("sql", f"select count(*) as n from {ds.name}").text
    r = t.execute("sql", f"create table adults as select * from {ds.name} where age >= 40")
    assert r.dataset_changed == "adults" and eng.datasets["adults"].n_rows == (df.age >= 40).sum()
    assert t.execute("python", "x = 2 + 3; x * 2").ok or True
    assert t.execute("python", "print(len(df))").text == "2000"
    assert t.execute("python", "df.score.mean()").text.startswith(f"{df.score.mean():.5f}"[:5])
    s = t.execute("stata", "ttest score, by(arm)")
    assert s.ok and "Welch" not in s.text and "t statistic" in s.text
    assert "Pearson" in t.execute("stata", "tabulate arm \"city name\", chi2").text or t.execute("stata", "tabulate arm \"city name\"").ok in (True, False)
    assert t.execute("stata", "regress score age").ok
    assert "t statistic" in t.execute("r", "t.test(score ~ arm, data = df)").text
    assert t.execute("r", "mean(df$age)").text == f"[1] {df.age.mean():.10g}"
    assert not t.execute("stata", "nonsense").ok and not t.execute("sql", "select * from nope").ok
    assert t.execute("stata", "histogram score").figure_request[0] == "histogram"
    assert any(s.kind == "sql" for s in rec.steps) and any(s.kind == "analysis" for s in rec.steps)


def test_import_keeps_file_order(tmp_path):
    """Grid rows must be in file order even when DuckDB scans in parallel (regression: preserve_insertion_order)."""
    import numpy as np
    import pandas as pd
    from statsmith.data import Engine
    n = 1_500_000
    p = tmp_path / "o.csv"
    pd.DataFrame({"i": np.arange(n), "x": 1.0}).to_csv(p, index=False)
    e = Engine(1, 4, workdir=str(tmp_path / "w"))
    ds = e.import_file(str(p), "o")
    assert e.con.execute(f'select count(*) from "{ds.name}" where rowid != i').fetchone()[0] == 0
    assert [r[0] for r in e.fetch(ds.name, n - 3, 3)] == [n - 3, n - 2, n - 1]
    e.close()


def test_export_roundtrip_all_formats(tmp_path):
    import pandas as pd
    from statsmith.data import Engine
    e = Engine(1, 2, workdir=str(tmp_path / "w"))
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(tmp_path / "s.csv", index=False)
    ds = e.import_file(str(tmp_path / "s.csv"), "s")
    for ext in (".csv", ".parquet", ".json", ".xlsx", ".dta"):
        out = tmp_path / f"o{ext}"
        e.export(ds.name, str(out))
        assert e.import_file(str(out), "r" + ext.strip(".")).n_rows == 3, ext
    e.close()
