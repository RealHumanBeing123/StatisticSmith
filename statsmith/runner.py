"""Glue between the data engine, the analysis/chart registries and the script recorder.

``selection`` is ``{role_name: column_name | [column_names]}`` - exactly what the dialogs produce.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .charts import REGISTRY as CHARTS
from .charts import ChartData, ChartStyle
from .charts import render as render_chart
from .codegen import Recorder
from .data import Engine
from .stats import REGISTRY as ANALYSES
from .stats import Result

MAX_CHART_ROWS = 200_000       # above this, charts that draw one mark per row use a reproducible random sample


def _flat(selection: dict) -> list[str]:
    out: list[str] = []
    for v in selection.values():
        for c in (v if isinstance(v, (list, tuple)) else [v]):
            if c and c not in out:
                out.append(c)
    return out


def load_roles(engine: Engine, dataset: str, roles, selection: dict, *, sample: int | None = None,
               seed: int = 1, where: str = "") -> tuple[ChartData, dict]:
    """Fetch only the selected columns. Numeric roles are cast to float (non-numeric -> NaN)."""
    ds = engine.datasets[dataset]
    numeric_cols: set[str] = set()
    for r in roles:
        if r.kind.startswith("numeric"):
            for c in (selection.get(r.name) if isinstance(selection.get(r.name), (list, tuple)) else [selection.get(r.name)]):
                if c:
                    numeric_cols.add(c)
    cols = _flat(selection)
    for c in cols:
        if c not in ds.names:
            raise KeyError(f"Column {c!r} not found in dataset {dataset!r}")
    info = {"sampled": False, "n_total": ds.n_rows}
    if sample and ds.n_rows > sample and not where:
        raw = engine.sample_columns(dataset, cols, sample, numeric=numeric_cols, seed=seed)
        info["sampled"] = True
    else:
        raw = engine.columns_frame(dataset, cols, numeric=numeric_cols, where=where)
    data, names = {}, {}
    for r in roles:
        sel = selection.get(r.name)
        if sel is None or sel == [] or sel == "":
            continue
        if isinstance(sel, (list, tuple)):
            data[r.name] = [raw[c] for c in sel]
            names[r.name] = list(sel)
        else:
            data[r.name] = raw[sel]
            names[r.name] = sel
    return ChartData(data, names), info


def missing_roles(roles, selection: dict) -> list[str]:
    miss = []
    for r in roles:
        v = selection.get(r.name)
        if r.required and (v is None or v == "" or v == []):
            miss.append(r.label or r.name)
    return miss


def run_analysis(engine: Engine, dataset: str, analysis_id: str, selection: dict, params: dict | None = None,
                 recorder: Recorder | None = None, where: str = "") -> Result:
    spec = ANALYSES[analysis_id]
    miss = missing_roles(spec.roles, selection)
    if miss:
        raise ValueError("Please choose: " + ", ".join(miss))
    d, _ = load_roles(engine, dataset, spec.roles, selection, where=where)
    kw = {p.name: p.default for p in spec.params}
    kw.update(params or {})
    res = spec.fn(d, **kw)
    if recorder is not None:
        ctx: dict[str, Any] = {"n": len(d)}
        for key in ("xs", "ys"):
            if d.has(key) and isinstance(d[key], list):
                ctx["p" if key == "xs" else "q"] = len(d[key])
        for key in ("x", "row", "col", "group"):          # level counts / 2x2 cells some templates need
            if d.has(key):
                try:
                    ctx["k"] = len({str(v) for v in d[key] if v is not None})
                except Exception:
                    pass
        if d.has("row") and d.has("col"):
            import pandas as pd
            t = pd.crosstab(pd.Series(d["row"]), pd.Series(d["col"])).values
            if t.shape == (2, 2):
                ctx.update(dict(zip("abcd", map(int, t.ravel()))))
        recorder.analysis(analysis_id, d.names, kw, ctx)
    return res


def draw_chart(engine: Engine, dataset: str, chart_id: str, selection: dict, params: dict | None = None,
               style: ChartStyle | None = None, fig=None, recorder: Recorder | None = None, record: bool = True,
               sample: int | None = MAX_CHART_ROWS, seed: int = 1):
    spec = CHARTS[chart_id]
    miss = missing_roles(spec.roles, selection)
    if miss:
        raise ValueError("Please choose: " + ", ".join(miss))
    d, info = load_roles(engine, dataset, spec.roles, selection, sample=sample, seed=seed)
    if info["sampled"]:
        d.notes.append(f"Random sample of {sample:,} of {info['n_total']:,} rows (seed {seed}) - statistics in the text output use all rows")
    kw = {p.name: p.default for p in spec.params}
    kw.update(params or {})
    fig = render_chart(chart_id, d, style or ChartStyle(), fig=fig, **kw)
    if recorder is not None and record:
        recorder.chart(chart_id, d.names, spec.name, kw)
    return fig, d
