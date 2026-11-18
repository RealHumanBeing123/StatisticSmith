"""Out-of-core data engine.

Every dataset lives in a DuckDB table (spilled to disk under ``temp_directory`` when it does not fit in
the RAM budget), so a 4 GB CSV never has to fit in Python memory and the UI thread never touches the file:

* :meth:`Engine.import_file` runs on a worker thread, reports progress and can be cancelled.
* The grid asks for *windows* of rows (``fetch``) - O(window), independent of table size.
* Statistics / charts pull only the columns they need (``column``) or run SQL aggregates in the engine.

The RAM / CPU limits from Settings are applied with ``SET memory_limit`` / ``SET threads``.
"""
from __future__ import annotations

import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import duckdb
import numpy as np

SUPPORTED = {
    ".csv": "CSV", ".tsv": "TSV", ".txt": "Delimited text", ".tab": "TSV",
    ".xlsx": "Excel", ".xlsm": "Excel", ".xls": "Excel (legacy)", ".ods": "OpenDocument",
    ".dta": "Stata", ".rds": "R data (single object)", ".rdata": "R workspace", ".rda": "R workspace",
    ".sav": "SPSS", ".sas7bdat": "SAS", ".parquet": "Parquet", ".json": "JSON", ".ndjson": "JSON lines",
    ".feather": "Feather",
}
NUMERIC_TYPES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER",
                 "UBIGINT", "FLOAT", "DOUBLE", "DECIMAL", "REAL")


def qid(name: str) -> str:
    """Quote an SQL identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def qstr(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def safe_table_name(stem: str) -> str:
    s = re.sub(r"\W+", "_", stem).strip("_").lower() or "data"
    return ("t_" + s) if s[0].isdigit() else s


class ImportCancelled(Exception):
    pass


@dataclass
class ColumnInfo:
    name: str
    sql_type: str
    label: str = ""                 # Stata/SPSS variable label
    value_labels: dict = field(default_factory=dict)

    @property
    def is_numeric(self) -> bool:
        t = self.sql_type.upper().split("(")[0]
        return t in NUMERIC_TYPES

    @property
    def is_datetime(self) -> bool:
        return self.sql_type.upper().startswith(("DATE", "TIMESTAMP", "TIME"))

    @property
    def kind(self) -> str:
        return "numeric" if self.is_numeric else "datetime" if self.is_datetime else \
               "boolean" if self.sql_type.upper() == "BOOLEAN" else "text"


@dataclass
class Dataset:
    """Metadata for one table inside the engine."""
    name: str
    columns: list[ColumnInfo]
    n_rows: int
    source: str = ""
    codebook: dict = field(default_factory=dict)    # parsed codebook / metadata (see codebook.py)
    codebook_path: str = ""
    file_label: str = ""
    notes: list[str] = field(default_factory=list)

    def col(self, name: str) -> ColumnInfo:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def numeric_names(self) -> list[str]:
        return [c.name for c in self.columns if c.is_numeric]


Progress = Callable[[float, str], None]   # fraction 0..1 (or -1 unknown), message


class Engine:
    """Owns the DuckDB database, memory / thread limits and all datasets."""

    def __init__(self, memory_gb: float = 2.0, threads: int = 4, workdir: str | None = None):
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="statsmith_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.workdir / "session.duckdb"
        self.con = duckdb.connect(str(self.db_path))
        self._lock = threading.RLock()
        self.datasets: dict[str, Dataset] = {}
        self.memory_gb, self.threads = memory_gb, threads
        self.set_limits(memory_gb, threads)
        self._import_con: duckdb.DuckDBPyConnection | None = None

    # ----------------------------------------------------------------- resources
    def set_limits(self, memory_gb: float, threads: int):
        memory_gb = max(float(memory_gb), 0.25)
        threads = max(int(threads), 1)
        spill = (self.workdir / "spill")
        spill.mkdir(exist_ok=True)
        with self._lock:
            self.con.execute(f"SET memory_limit='{memory_gb:.2f}GB'")
            self.con.execute(f"SET threads={threads}")
            self.con.execute(f"SET temp_directory={qstr(str(spill))}")
            self.con.execute("SET preserve_insertion_order=true")    # grid rows must stay in file order
        self.memory_gb, self.threads = memory_gb, threads
        try:  # numpy / BLAS thread pools follow the same CPU budget
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=threads)
        except Exception:
            pass

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass

    def sql(self, query: str, params: Sequence[Any] | None = None):
        """Run SQL on the engine and return the DuckDB relation result cursor."""
        with self._lock:
            return self.con.execute(query, params or [])

    # ----------------------------------------------------------------- import
    def import_file(self, path: str, name: str | None = None, progress: Progress | None = None,
                    cancel: threading.Event | None = None, options: dict | None = None) -> Dataset:
        """Import ``path`` into a table. Safe to call from a worker thread. Raises ImportCancelled."""
        options = options or {}
        p = Path(path)
        ext = p.suffix.lower()
        if ext not in SUPPORTED:
            raise ValueError(f"Unsupported file type {ext!r}. Supported: {', '.join(sorted(SUPPORTED))}")
        name = self._unique_name(name or safe_table_name(p.stem))
        prog = progress or (lambda f, m: None)
        cancel = cancel or threading.Event()
        prog(0.0, f"Opening {p.name}")
        labels: dict[str, str] = {}
        vlabels: dict[str, dict] = {}
        file_label = ""
        notes: list[str] = []
        if ext in (".csv", ".tsv", ".txt", ".tab"):
            self._import_delimited(p, name, ext, options, prog, cancel)
        elif ext == ".parquet":
            self._exec_cancellable(f"CREATE TABLE {qid(name)} AS SELECT * FROM read_parquet({qstr(str(p))})", prog, cancel,
                                   f"Loading {p.name}")
        elif ext in (".json", ".ndjson"):
            self._exec_cancellable(f"CREATE TABLE {qid(name)} AS SELECT * FROM read_json_auto({qstr(str(p))})", prog, cancel,
                                   f"Loading {p.name}")
        elif ext in (".xlsx", ".xlsm", ".xls", ".ods"):
            self._import_excel(p, name, options, prog, cancel)
        elif ext == ".dta":
            labels, vlabels, file_label = self._import_stata(p, name, options, prog, cancel)
        elif ext in (".sav", ".sas7bdat"):
            labels, vlabels, file_label = self._import_pyreadstat(p, name, ext, options, prog, cancel)
        elif ext in (".rds", ".rdata", ".rda"):
            self._import_r(p, name, options, prog, cancel, notes)
        elif ext == ".feather":
            import polars as pl
            self._register_frame(name, pl.read_ipc(str(p)).to_arrow())
        if cancel.is_set():
            self._drop(name)
            raise ImportCancelled()
        prog(0.97, "Reading column metadata")
        ds = self._describe(name, str(p))
        for c in ds.columns:
            c.label = labels.get(c.name, "")
            c.value_labels = vlabels.get(c.name, {})
        ds.file_label = file_label
        ds.notes = notes
        from .codebook import attach_codebook
        attach_codebook(ds, p)
        self.datasets[name] = ds
        prog(1.0, f"Imported {ds.n_rows:,} rows x {len(ds.columns)} columns")
        return ds

    # -- helpers
    def _unique_name(self, base: str) -> str:
        name, i = base, 2
        while name in self.datasets:
            name = f"{base}_{i}"
            i += 1
        return name

    def _drop(self, name: str):
        with self._lock:
            try:
                self.con.execute(f"DROP TABLE IF EXISTS {qid(name)}")
            except Exception:
                pass

    def _exec_cancellable(self, sql: str, prog: Progress, cancel: threading.Event, msg: str):
        """Run a long statement on a *separate cursor* so the UI/query connection stays responsive,
        poll DuckDB's progress bar, and interrupt on cancel."""
        cur = self.con.cursor()
        err: list[BaseException] = []

        def run():
            try:
                cur.execute(sql)
            except BaseException as e:  # noqa: BLE001
                err.append(e)

        th = threading.Thread(target=run, daemon=True)
        th.start()
        last = 0.0
        while th.is_alive():
            th.join(0.15)
            if cancel.is_set():
                try:
                    cur.interrupt()
                except Exception:
                    pass
                th.join()
                raise ImportCancelled()
            try:
                f = cur.query_progress()
            except Exception:
                f = -1.0
            if f is not None and f >= 0:
                last = max(last, min(f / 100.0 if f > 1 else f, 0.95))
                prog(last * 0.95, msg)
            else:
                prog(-1.0, msg)
        if err:
            if cancel.is_set():
                raise ImportCancelled()
            raise err[0]

    def _import_delimited(self, p: Path, name: str, ext: str, opt: dict, prog: Progress, cancel):
        delim = opt.get("delimiter") or ("\t" if ext in (".tsv", ".tab") else None)
        parts = [f"read_csv({qstr(str(p))}", "header=" + ("true" if opt.get("header", True) else "false"),
                 "sample_size=20000", "ignore_errors=true", "null_padding=true", "parallel=true",
                 "normalize_names=false"]
        if delim:
            parts.append(f"delim={qstr(delim)}")
        if opt.get("encoding"):
            parts.append(f"encoding={qstr(opt['encoding'])}")
        if opt.get("all_varchar"):
            parts.append("all_varchar=true")
        try:
            self._exec_cancellable(f"CREATE TABLE {qid(name)} AS SELECT * FROM {', '.join(parts)})", prog, cancel,
                                   f"Reading {p.name}")
        except ImportCancelled:
            raise
        except Exception:
            # type-sniffing can fail on very messy files: retry as text so the user still gets the data
            self._drop(name)
            parts.append("all_varchar=true")
            self._exec_cancellable(f"CREATE TABLE {qid(name)} AS SELECT * FROM {', '.join(parts)})", prog, cancel,
                                   f"Reading {p.name} (as text)")

    def _register_frame(self, name: str, arrow_table):
        with self._lock:
            cur = self.con.cursor()
            try:
                cur.register("_tmp_in", arrow_table)
                cur.execute(f"CREATE TABLE {qid(name)} AS SELECT * FROM _tmp_in").fetchall()
                cur.unregister("_tmp_in")
            finally:
                cur.close()

    def _import_excel(self, p: Path, name: str, opt: dict, prog: Progress, cancel):
        import polars as pl
        sheet = opt.get("sheet")
        prog(-1.0, f"Reading Excel {p.name} (this can take a while for large workbooks)")
        try:
            kw: dict[str, Any] = {"engine": "calamine"}
            df = pl.read_excel(str(p), sheet_name=sheet if sheet is not None else None, sheet_id=None if sheet else 1, **kw)
        except Exception:
            df = pl.read_excel(str(p), sheet_name=sheet, sheet_id=None if sheet else 1)
        if isinstance(df, dict):
            df = next(iter(df.values()))
        if cancel.is_set():
            raise ImportCancelled()
        self._register_frame(name, df.to_arrow())

    def excel_sheets(self, path: str) -> list[str]:
        try:
            import fastexcel
            return list(fastexcel.read_excel(path).sheet_names)
        except Exception:
            return []

    def _chunked_to_table(self, name: str, reader: Iterable, total: int | None, prog: Progress, cancel, msg: str):
        """Append DataFrame chunks into a table (used for pyreadstat which cannot stream into DuckDB)."""
        first = True
        done = 0
        for chunk in reader:
            if cancel.is_set():
                raise ImportCancelled()
            with self._lock:
                cur = self.con.cursor()
                try:
                    cur.register("_chunk", chunk)
                    if first:
                        cur.execute(f"CREATE TABLE {qid(name)} AS SELECT * FROM _chunk").fetchall()
                        first = False
                    else:
                        cur.execute(f"INSERT INTO {qid(name)} SELECT * FROM _chunk").fetchall()
                    cur.unregister("_chunk")
                finally:
                    cur.close()
            done += len(chunk)
            prog(min(done / total, 0.95) if total else -1.0, f"{msg}: {done:,} rows")
        if first:
            raise ValueError("The file contains no rows")

    def _import_stata(self, p: Path, name: str, opt: dict, prog: Progress, cancel):
        import pyreadstat
        meta_only = pyreadstat.read_dta(str(p), metadataonly=True)[1]
        total = meta_only.number_rows or None
        step = int(opt.get("chunk", 250_000))
        convert_cat = bool(opt.get("apply_value_labels", False))
        reader = (df for df, _ in pyreadstat.read_file_in_chunks(pyreadstat.read_dta, str(p), chunksize=step,
                                                                 apply_value_formats=convert_cat))
        self._chunked_to_table(name, reader, total, prog, cancel, "Reading Stata file")
        return self._pyreadstat_meta(meta_only)

    def _import_pyreadstat(self, p: Path, name: str, ext: str, opt: dict, prog: Progress, cancel):
        import pyreadstat
        fn = pyreadstat.read_sav if ext == ".sav" else pyreadstat.read_sas7bdat
        meta_only = fn(str(p), metadataonly=True)[1]
        reader = (df for df, _ in pyreadstat.read_file_in_chunks(fn, str(p), chunksize=int(opt.get("chunk", 250_000))))
        self._chunked_to_table(name, reader, meta_only.number_rows or None, prog, cancel, "Reading file")
        return self._pyreadstat_meta(meta_only)

    @staticmethod
    def _pyreadstat_meta(meta):
        labels = dict(zip(meta.column_names, meta.column_labels or [""] * len(meta.column_names)))
        labels = {k: (v or "") for k, v in labels.items()}
        vl: dict[str, dict] = {}
        for col, setname in (meta.variable_to_label or {}).items():
            vl[col] = dict((meta.value_labels or {}).get(setname, {}))
        return labels, vl, getattr(meta, "file_label", "") or ""

    def _import_r(self, p: Path, name: str, opt: dict, prog: Progress, cancel, notes: list[str]):
        import pandas as pd
        import pyreadr
        prog(-1.0, f"Reading R data {p.name}")
        res = pyreadr.read_r(str(p))
        frames = {k: v for k, v in res.items() if isinstance(v, pd.DataFrame)}
        if not frames:
            raise ValueError("No data.frame found in this R file (only data frames can be imported)")
        key = opt.get("object") or next(iter(frames))
        if len(frames) > 1:
            notes.append(f"R file contains {len(frames)} data frames ({', '.join(frames)}); imported '{key}'")
        df = frames[key]
        for c in df.columns:                      # R factors -> text so they chart/aggregate naturally
            if str(df[c].dtype) == "category":
                df[c] = df[c].astype(object)
        df = df.reset_index(drop=True)
        self._register_frame(name, df)

    def r_objects(self, path: str) -> list[str]:
        try:
            import pandas as pd
            import pyreadr
            return [k for k, v in pyreadr.read_r(path).items() if isinstance(v, pd.DataFrame)]
        except Exception:
            return []

    # ----------------------------------------------------------------- introspection
    def _describe(self, name: str, source: str) -> Dataset:
        with self._lock:
            cols = self.con.execute(f"DESCRIBE {qid(name)}").fetchall()
            n = self.con.execute(f"SELECT count(*) FROM {qid(name)}").fetchone()[0]
        return Dataset(name, [ColumnInfo(c[0], str(c[1])) for c in cols], int(n), source)

    def create_from_query(self, name: str, query: str, source: str = "query") -> Dataset:
        name = self._unique_name(safe_table_name(name))
        with self._lock:
            self.con.execute(f"CREATE TABLE {qid(name)} AS {query}")
        ds = self._describe(name, source)
        self.datasets[name] = ds
        return ds

    def refresh(self, name: str) -> Dataset:
        old = self.datasets.get(name)
        ds = self._describe(name, old.source if old else "")
        if old:
            by = {c.name: c for c in old.columns}
            for c in ds.columns:
                if c.name in by:
                    c.label, c.value_labels = by[c.name].label, by[c.name].value_labels
            ds.codebook, ds.codebook_path, ds.file_label, ds.notes = old.codebook, old.codebook_path, old.file_label, old.notes
        self.datasets[name] = ds
        return ds

    def drop_dataset(self, name: str):
        self._drop(name)
        self.datasets.pop(name, None)

    # ----------------------------------------------------------------- data access
    def fetch(self, name: str, offset: int, limit: int, columns: Sequence[str] | None = None) -> list[tuple]:
        """A window of raw rows for the spreadsheet grid (``rowid`` order = file order)."""
        cols = ", ".join(qid(c) for c in columns) if columns else "*"
        with self._lock:
            return self.con.execute(
                f"SELECT {cols} FROM {qid(name)} LIMIT {int(limit)} OFFSET {int(offset)}").fetchall()

    def column(self, name: str, col: str, where: str = "", numeric: bool = False, limit: int | None = None) -> np.ndarray:
        """Whole column as a numpy array (float64 for numeric/``numeric=True``, object otherwise)."""
        sel = f"TRY_CAST({qid(col)} AS DOUBLE)" if numeric else qid(col)
        q = f"SELECT {sel} FROM {qid(name)}" + (f" WHERE {where}" if where else "") + (f" LIMIT {int(limit)}" if limit else "")
        with self._lock:
            arr = self.con.execute(q).fetchnumpy()
        v = next(iter(arr.values()))
        if hasattr(v, "filled"):  # masked array (NULLs)
            if numeric or v.dtype.kind in "fiu":
                return v.astype(float).filled(np.nan)
            return np.array([None if m else x for x, m in zip(v.data, np.ma.getmaskarray(v))], dtype=object)
        return v.astype(float) if numeric else v

    def columns_frame(self, name: str, cols: Sequence[str], numeric: Sequence[str] = (), where: str = "",
                      dropna: bool = False, limit: int | None = None) -> dict[str, np.ndarray]:
        """Several columns as {name: ndarray}, optionally complete-case filtered, in one pass."""
        sels = []
        for c in cols:
            sels.append(f"TRY_CAST({qid(c)} AS DOUBLE) AS {qid(c)}" if c in numeric else qid(c))
        conds = [where] if where else []
        if dropna:
            conds += [f"{qid(c)} IS NOT NULL" for c in cols]
        q = f"SELECT {', '.join(sels)} FROM {qid(name)}" + (f" WHERE {' AND '.join(conds)}" if conds else "") + \
            (f" LIMIT {int(limit)}" if limit else "")
        with self._lock:
            raw = self.con.execute(q).fetchnumpy()
        out: dict[str, np.ndarray] = {}
        for c in cols:
            v = raw[c]
            if hasattr(v, "filled"):
                if c in numeric or v.dtype.kind in "fiu":
                    v = v.astype(float).filled(np.nan)
                else:
                    v = np.array([None if m else x for x, m in zip(v.data, np.ma.getmaskarray(v))], dtype=object)
            elif c in numeric:
                v = v.astype(float)
            out[c] = v
        if dropna:  # NaN produced by TRY_CAST
            keep = np.ones(len(next(iter(out.values()))) if out else 0, dtype=bool)
            for c in numeric:
                if c in out:
                    keep &= ~np.isnan(out[c])
            if not keep.all():
                out = {k: v[keep] for k, v in out.items()}
        return out

    def sample_columns(self, name: str, cols: Sequence[str], n: int, numeric: Sequence[str] = (), seed: int = 1):
        """Reproducible random sample of ``n`` rows (reservoir with fixed seed) - for charts on huge tables."""
        sels = [f"TRY_CAST({qid(c)} AS DOUBLE) AS {qid(c)}" if c in numeric else qid(c) for c in cols]
        with self._lock:
            raw = self.con.execute(
                f"SELECT {', '.join(sels)} FROM {qid(name)} USING SAMPLE reservoir({int(n)} ROWS) REPEATABLE ({int(seed)})"
            ).fetchnumpy()
        out = {}
        for c in cols:
            v = raw[c]
            if hasattr(v, "filled"):
                v = v.astype(float).filled(np.nan) if (c in numeric or v.dtype.kind in "fiu") else \
                    np.array([None if m else x for x, m in zip(v.data, np.ma.getmaskarray(v))], dtype=object)
            out[c] = v
        return out

    def count(self, name: str, where: str = "") -> int:
        with self._lock:
            return int(self.con.execute(f"SELECT count(*) FROM {qid(name)}" + (f" WHERE {where}" if where else "")).fetchone()[0])

    def summary(self, name: str, col: str) -> dict:
        """Fast engine-side summary for the status bar / column header tooltip."""
        c = qid(col)
        with self._lock:
            r = self.con.execute(
                f"SELECT count({c}), count(*) - count({c}), count(DISTINCT {c}) FROM {qid(name)}").fetchone()
        return {"n": r[0], "missing": r[1], "distinct": r[2]}

    def numeric_summary(self, name: str, cols: Sequence[str]) -> dict[str, dict]:
        """count/mean/sd/min/q1/median/q3/max for many columns in ONE table scan."""
        if not cols:
            return {}
        parts = []
        for c in cols:
            x = f"TRY_CAST({qid(c)} AS DOUBLE)"
            parts.append(f"count({x}), avg({x}), stddev_samp({x}), min({x}), quantile_cont({x},0.25), "
                         f"median({x}), quantile_cont({x},0.75), max({x}), sum({x})")
        with self._lock:
            row = self.con.execute(f"SELECT {', '.join(parts)} FROM {qid(name)}").fetchone()
        keys = ("n", "mean", "sd", "min", "q1", "median", "q3", "max", "sum")
        return {c: dict(zip(keys, row[i * 9:(i + 1) * 9])) for i, c in enumerate(cols)}

    # ----------------------------------------------------------------- editing
    def update_cell(self, name: str, rowid_offset: int, col: str, value: Any):
        """Edit one cell (row addressed by position in current order)."""
        with self._lock:
            self.con.execute(
                f"UPDATE {qid(name)} SET {qid(col)} = ? WHERE rowid = (SELECT rowid FROM {qid(name)} LIMIT 1 OFFSET {int(rowid_offset)})",
                [value])

    def add_column(self, name: str, new: str, expression: str):
        with self._lock:
            self.con.execute(f"ALTER TABLE {qid(name)} ADD COLUMN {qid(new)} DOUBLE")
            self.con.execute(f"UPDATE {qid(name)} SET {qid(new)} = {expression}")
        self.refresh(name)

    def export(self, name: str, path: str, query: str | None = None):
        ext = Path(path).suffix.lower()
        src = f"({query})" if query else qid(name)
        fmt = {".csv": "(FORMAT CSV, HEADER)", ".parquet": "(FORMAT PARQUET)", ".json": "(FORMAT JSON, ARRAY true)"}.get(ext)
        if fmt is None and ext not in (".xlsx", ".dta"):
            raise ValueError(f"Cannot export to {ext}")
        cur = self.con.cursor()                      # private cursor: never leaves a pending result on the shared connection
        try:
            if fmt:
                cur.execute(f"COPY {src} TO {qstr(path)} {fmt}").fetchall()
            elif ext == ".xlsx":
                import polars as pl
                pl.from_arrow(cur.sql(f"SELECT * FROM {src}").to_arrow_table()).write_excel(path)
            else:
                import pyreadstat
                pyreadstat.write_dta(cur.execute(f"SELECT * FROM {src}").fetchdf(), path)
        finally:
            cur.close()


def human_bytes(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}" if u != "B" else f"{int(n)} B"
        n /= 1024
    return str(n)


def system_resources() -> dict:
    """Detected RAM and CPU for the Settings page limits (no artificial 128 GB / 256-core cap)."""
    import psutil
    return {"ram_gb": psutil.virtual_memory().total / 2**30,
            "logical_cpus": psutil.cpu_count(logical=True) or os.cpu_count() or 1,
            "physical_cpus": psutil.cpu_count(logical=False) or 1}
