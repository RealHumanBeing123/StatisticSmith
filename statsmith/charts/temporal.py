"""Data-over-time charts: line, spline, step line, area, stacked area, stream graph, bump chart, bump area,
candlestick, OHLC, Gantt, barcode."""
from __future__ import annotations

import warnings

import numpy as np

from .base import (ChartStyle, Param, Role, chart, empty_message, finish, first_seen, fmt, make_axes, numeric, short)

MAX_SERIES = 30
_AGG = ("mean", "sum", "median", "min", "max", "count", "first", "last")


# ------------------------------------------------------------------ time-axis helpers
def to_axis(a):
    """Column -> (float positions, kind, labels). kind: 'date' | 'num' | 'cat'.

    datetime64 / parseable date strings -> matplotlib date numbers; numeric -> as is;
    anything else -> category codes in order of appearance (labels returned).
    """
    import matplotlib.dates as mdates
    import pandas as pd
    arr = np.asarray(a)
    if arr.dtype.kind == "M":
        return np.asarray(mdates.date2num(arr.astype("datetime64[ms]")), dtype=float), "date", None
    if arr.dtype.kind in "fiub":
        return arr.astype(float), "num", None
    obj = arr.astype(object)
    ok = np.array([v is not None and str(v) not in ("", "nan", "NaT", "None") for v in obj])
    if ok.sum() == 0:
        return np.full(len(obj), np.nan), "num", None
    num = numeric(obj)
    if np.isfinite(num[ok]).all():
        num[~ok] = np.nan
        return num, "num", None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt = pd.to_datetime(pd.Series(obj[ok].astype(str)), errors="coerce")
    if dt.notna().mean() >= 0.9:
        out = np.full(len(obj), np.nan)
        out[ok] = np.asarray(mdates.date2num(dt.dt.to_pydatetime()), dtype=float)
        return out, "date", None
    labs = first_seen([str(v) for v in obj[ok]])
    code = {l: i for i, l in enumerate(labs)}
    out = np.full(len(obj), np.nan)
    out[ok] = [code[str(v)] for v in obj[ok]]
    return out, "cat", labs


def axis_ticks(ax, kind, labs, which="x"):
    import matplotlib.dates as mdates
    axis = ax.xaxis if which == "x" else ax.yaxis
    if kind == "date":
        loc = mdates.AutoDateLocator()
        axis.set_major_locator(loc)
        axis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    elif kind == "cat":
        step = max(1, len(labs) // 12)
        pos = list(range(0, len(labs), step))
        (ax.set_xticks if which == "x" else ax.set_yticks)(pos, [short(labs[i]) for i in pos])


def series_table(d, agg="mean"):
    """-> (xs sorted unique, labels, M[len(xs), n_series] with NaN where a series has no value, kind, xlabels)."""
    import pandas as pd
    x, kind, xl = to_axis(d["x"])
    ys = [numeric(c) for c in d["y"]]
    names = d.label_list("y")
    ok = np.isfinite(x)
    if d.has("group"):
        g = np.asarray(d["group"], dtype=object).astype(str)
        df = pd.DataFrame({"x": x[ok], "g": g[ok], "y": ys[0][ok]})
        labs = first_seen(df["g"].tolist())[:MAX_SERIES]
        df = df[df["g"].isin(labs)]
        piv = df.pivot_table(index="x", columns="g", values="y", aggfunc=agg, dropna=False) if agg != "count" else \
            df.pivot_table(index="x", columns="g", values="y", aggfunc="count")
        piv = piv.reindex(columns=labs)
    else:
        labs = names[:MAX_SERIES]
        df = pd.DataFrame({f"s{i}": ys[i][ok] for i in range(len(labs))})
        df["x"] = x[ok]
        piv = df.groupby("x").agg(agg)
        piv.columns = labs
    piv = piv.sort_index()
    return piv.index.to_numpy(dtype=float), list(piv.columns), piv.to_numpy(dtype=float), kind, xl


_ROLES = [Role("x", "any", "Time / order"), Role("y", "numeric_multi", "Value(s)"),
          Role("group", "categorical", "Series by (uses first value column)", required=False)]
_PARAMS = [Param("agg", "If several rows share a time", "choice", "mean", choices=_AGG)]


def _frame(fig, d, style, agg):
    xs, labs, M, kind, xl = series_table(d, agg)
    if xs.size == 0 or not labs:
        empty_message(fig, style, "No data")
        return None
    ax = make_axes(fig, style)
    return ax, xs, labs, M, kind, xl


def _end(fig, ax, d, style, kind, xl, labs, ylabel=None, legend=None):
    axis_ticks(ax, kind, xl)
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=ylabel or (labs[0] if len(labs) == 1 else ""),
           legend=(len(labs) > 1) if legend is None else legend, grid_axis="both" if kind != "cat" else "y")
    return ax


# ------------------------------------------------------------------ line family
@chart("line", "Line chart", "Data over time", _ROLES,
       _PARAMS + [Param("markers", "Show markers", "bool", False)], aliases=("line graph", "time series plot"))
def line(fig, d, style, agg="mean", markers=False):
    """One line per series; points are joined in time order."""
    f = _frame(fig, d, style, agg)
    if f is None:
        return
    ax, xs, labs, M, kind, xl = f
    for j, lab in enumerate(labs):
        ok = np.isfinite(M[:, j])
        ax.plot(xs[ok], M[ok, j], color=style.color_for(lab, j), lw=style.line_width, marker="o" if markers else None,
                ms=style.marker_size * 0.6, label=lab, alpha=style.alpha)
    return _end(fig, ax, d, style, kind, xl, labs)


@chart("spline", "Spline chart", "Data over time", _ROLES,
       _PARAMS + [Param("markers", "Show original points", "bool", True)], aliases=("smooth line chart", "curved line chart"))
def spline(fig, d, style, agg="mean", markers=True):
    """Smooth monotone cubic (PCHIP) curve through the data; it never overshoots the data points."""
    from scipy.interpolate import PchipInterpolator
    f = _frame(fig, d, style, agg)
    if f is None:
        return
    ax, xs, labs, M, kind, xl = f
    for j, lab in enumerate(labs):
        ok = np.isfinite(M[:, j])
        x, y = xs[ok], M[ok, j]
        c = style.color_for(lab, j)
        if x.size >= 3:
            g = np.linspace(x[0], x[-1], 400)
            ax.plot(g, PchipInterpolator(x, y)(g), color=c, lw=style.line_width, label=lab, alpha=style.alpha)
        else:
            ax.plot(x, y, color=c, lw=style.line_width, label=lab)
        if markers:
            ax.plot(x, y, "o", color=c, ms=style.marker_size * 0.6)
    return _end(fig, ax, d, style, kind, xl, labs)


@chart("step_line", "Step line chart", "Data over time", _ROLES,
       _PARAMS + [Param("where", "Step position", "choice", "post", choices=("post", "pre", "mid"))], aliases=("step chart", "stepped line"))
def step_line(fig, d, style, agg="mean", where="post"):
    """Value holds constant until the next observation (good for prices, states, counters)."""
    f = _frame(fig, d, style, agg)
    if f is None:
        return
    ax, xs, labs, M, kind, xl = f
    for j, lab in enumerate(labs):
        ok = np.isfinite(M[:, j])
        ax.step(xs[ok], M[ok, j], where=where, color=style.color_for(lab, j), lw=style.line_width, label=lab, alpha=style.alpha)
    return _end(fig, ax, d, style, kind, xl, labs)


# ------------------------------------------------------------------ area family
@chart("area", "Area chart", "Data over time", _ROLES, _PARAMS, aliases=("area graph",))
def area(fig, d, style, agg="mean"):
    """Line chart with the region down to zero filled; overlapping series are translucent."""
    f = _frame(fig, d, style, agg)
    if f is None:
        return
    ax, xs, labs, M, kind, xl = f
    for j, lab in enumerate(labs):
        ok = np.isfinite(M[:, j])
        c = style.color_for(lab, j)
        ax.fill_between(xs[ok], 0, M[ok, j], color=c, alpha=style.alpha * (0.5 if len(labs) > 1 else 0.6), label=lab)
        ax.plot(xs[ok], M[ok, j], color=c, lw=style.line_width)
    return _end(fig, ax, d, style, kind, xl, labs)


def _stack(fig, d, style, agg, baseline, pct=False):
    f = _frame(fig, d, style, agg)
    if f is None:
        return None
    ax, xs, labs, M, kind, xl = f
    Z = np.nan_to_num(M)
    if pct:
        tot = Z.sum(1, keepdims=True)
        Z = np.divide(Z, tot, out=np.zeros_like(Z), where=tot != 0) * 100
    ax.stackplot(xs, Z.T, labels=labs, colors=style.series_colors(labs), alpha=style.alpha, baseline=baseline)
    return ax, xs, labs, Z, kind, xl


@chart("stacked_area", "Stacked area chart", "Data over time", _ROLES,
       _PARAMS + [Param("percent", "100% stacked", "bool", False)], aliases=("stacked area graph",))
def stacked_area(fig, d, style, agg="sum", percent=False):
    """Series stacked on top of each other so the top edge is the total."""
    r = _stack(fig, d, style, agg, "zero", percent)
    if r is None:
        return
    ax, xs, labs, Z, kind, xl = r
    return _end(fig, ax, d, style, kind, xl, labs, ylabel="% of total" if percent else "", legend=True)


@chart("stream_graph", "Stream graph", "Data over time", _ROLES,
       _PARAMS + [Param("baseline", "Baseline", "choice", "weighted_wiggle", choices=("wiggle", "weighted_wiggle", "sym"))],
       aliases=("streamgraph", "river chart"))
def stream_graph(fig, d, style, agg="sum", baseline="weighted_wiggle"):
    """Stacked area whose baseline floats to minimise wiggle, giving a flowing 'stream' shape."""
    r = _stack(fig, d, style, agg, baseline)
    if r is None:
        return
    ax, xs, labs, Z, kind, xl = r
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    return _end(fig, ax, d, style, kind, xl, labs, legend=True)


# ------------------------------------------------------------------ bump family
def _ranks(M):
    """Rank 1 = largest value in each row (ties share the best rank). NaN stays NaN."""
    R = np.full(M.shape, np.nan)
    for i in range(M.shape[0]):
        row = M[i]
        ok = np.isfinite(row)
        if ok.any():
            v = row[ok]
            R[i, ok] = 1 + (v[:, None] < v[None, :]).sum(axis=1)
    return R


@chart("bump_chart", "Bump chart", "Data over time", _ROLES, _PARAMS, aliases=("rank chart", "ranking over time"))
def bump_chart(fig, d, style, agg="sum"):
    """Rank (1 = highest value) of each series at every time point."""
    f = _frame(fig, d, style, agg)
    if f is None:
        return
    ax, xs, labs, M, kind, xl = f
    R = _ranks(M)
    for j, lab in enumerate(labs):
        ok = np.isfinite(R[:, j])
        c = style.color_for(lab, j)
        ax.plot(xs[ok], R[ok, j], color=c, lw=style.line_width * 1.5, marker="o", ms=style.marker_size, label=lab, alpha=style.alpha)
        if ok.any():
            ax.text(xs[ok][-1], R[ok, j][-1], f"  {short(lab, 14)}", va="center", fontsize=style.value_label_size, color=c)
    top = int(np.nanmax(R)) if np.isfinite(R).any() else 1
    ax.set_ylim(top + 0.5, 0.5)
    ax.set_yticks(range(1, top + 1))
    _end(fig, ax, d, style, kind, xl, labs, ylabel="Rank (1 = highest)", legend=False)
    ax.margins(x=0.12)
    return ax


@chart("bump_area", "Bump area chart", "Data over time", _ROLES, _PARAMS, aliases=("ranked area chart",))
def bump_area(fig, d, style, agg="sum"):
    """Areas stacked in rank order at every time point; thickness = value, vertical order = rank."""
    f = _frame(fig, d, style, agg)
    if f is None:
        return
    ax, xs, labs, M, kind, xl = f
    Z = np.nan_to_num(M)
    order = np.argsort(-Z, axis=1, kind="stable")                       # highest first, at the bottom
    bottoms = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        acc = 0.0
        for j in order[i]:
            bottoms[i, j] = acc
            acc += Z[i, j]
    for j, lab in enumerate(labs):
        ax.fill_between(xs, bottoms[:, j], bottoms[:, j] + Z[:, j], color=style.color_for(lab, j), alpha=style.alpha, label=lab, lw=0.5,
                        edgecolor=style.background)
    return _end(fig, ax, d, style, kind, xl, labs, legend=True)


# ------------------------------------------------------------------ financial
_OHLC = [Role("x", "any", "Date / time"), Role("open", "numeric", "Open"), Role("high", "numeric", "High"),
         Role("low", "numeric", "Low"), Role("close", "numeric", "Close")]


def _ohlc_data(d):
    x, kind, xl = to_axis(d["x"])
    o, h, l, c = (numeric(d[k]) for k in ("open", "high", "low", "close"))
    ok = np.isfinite(x) & np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
    x, o, h, l, c = x[ok], o[ok], h[ok], l[ok], c[ok]
    order = np.argsort(x, kind="stable")
    x, o, h, l, c = x[order], o[order], h[order], l[order], c[order]
    hi, lo = np.maximum.reduce([o, h, l, c]), np.minimum.reduce([o, h, l, c])     # repair inconsistent rows
    return x, o, hi, lo, c, kind, xl


def _width(x):
    return 0.6 * float(np.median(np.diff(x))) if x.size > 1 and np.median(np.diff(x)) > 0 else 0.6


@chart("candlestick", "Candlestick chart", "Data over time", _OHLC, aliases=("candle chart", "japanese candlestick"))
def candlestick(fig, d, style, **_):
    """Body spans open to close (up colour when close >= open); wick spans low to high."""
    from matplotlib.patches import Rectangle
    x, o, h, l, c, kind, xl = _ohlc_data(d)
    if not x.size:
        return empty_message(fig, style, "No data")
    ax = make_axes(fig, style)
    up, dn = style.color_for("up", 2), style.color_for("down", 4)
    w = _width(x)
    ax.vlines(x, l, h, colors=[up if cc >= oo else dn for oo, cc in zip(o, c)], lw=1.0)
    for xi, oo, cc in zip(x, o, c):
        ax.add_patch(Rectangle((xi - w / 2, min(oo, cc)), w, abs(cc - oo) or 1e-9, color=up if cc >= oo else dn, alpha=style.alpha))
    ax.autoscale_view()
    ax.set_xlim(x[0] - w, x[-1] + w)
    return _end(fig, ax, d, style, kind, xl, ["price"], ylabel="Price", legend=False)


@chart("ohlc", "OHLC chart", "Data over time", _OHLC, aliases=("open-high-low-close", "bar chart (financial)"))
def ohlc(fig, d, style, **_):
    """Vertical line low-high; left tick = open, right tick = close."""
    x, o, h, l, c, kind, xl = _ohlc_data(d)
    if not x.size:
        return empty_message(fig, style, "No data")
    ax = make_axes(fig, style)
    up, dn = style.color_for("up", 2), style.color_for("down", 4)
    w = _width(x) / 2
    for xi, oo, hh, ll, cc in zip(x, o, h, l, c):
        col = up if cc >= oo else dn
        ax.plot([xi, xi], [ll, hh], color=col, lw=style.line_width * 0.7)
        ax.plot([xi - w, xi], [oo, oo], color=col, lw=style.line_width * 0.7)
        ax.plot([xi, xi + w], [cc, cc], color=col, lw=style.line_width * 0.7)
    ax.set_xlim(x[0] - 2 * w, x[-1] + 2 * w)
    return _end(fig, ax, d, style, kind, xl, ["price"], ylabel="Price", legend=False)


# ------------------------------------------------------------------ gantt
@chart("gantt", "Gantt chart", "Data over time",
       [Role("task", "categorical", "Task"), Role("start", "any", "Start"), Role("end", "any", "End"),
        Role("group", "categorical", "Colour by", required=False)],
       [Param("max_tasks", "Max tasks", "int", 40, min=1, max=300)], aliases=("timeline", "project schedule"))
def gantt(fig, d, style, max_tasks=40):
    """One horizontal bar per task from start to end."""
    s, kind, sl = to_axis(d["start"])
    e, kind2, el = to_axis(d["end"])
    t = np.asarray(d["task"], dtype=object).astype(str)
    ok = np.isfinite(s) & np.isfinite(e)
    s, e, t = s[ok], e[ok], t[ok]
    lo, hi = np.minimum(s, e), np.maximum(s, e)
    g = np.asarray(d["group"], dtype=object).astype(str)[ok] if d.has("group") else None
    n = min(len(t), max_tasks)
    if n == 0:
        return empty_message(fig, style, "No data")
    if len(t) > max_tasks:
        d.notes.append(f"showing first {max_tasks} of {len(t)} tasks")
    ax = make_axes(fig, style)
    glabs = first_seen(g.tolist()) if g is not None else []
    for i in range(n):
        lab = g[i] if g is not None else t[i]
        c = style.color_for(lab, glabs.index(lab) if g is not None else 0)
        ax.barh(i, max(hi[i] - lo[i], 1e-9), left=lo[i], height=style.bar_width * 0.8, color=c, alpha=style.alpha)
    ax.set_yticks(range(n), [short(x, 24) for x in t[:n]])
    ax.invert_yaxis()
    if g is not None:
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=style.color_for(l, i), label=l) for i, l in enumerate(glabs)], loc=style.legend_loc)
    k = kind if kind == kind2 else "num"
    axis_ticks(ax, k, sl or el)
    finish(fig, ax, style, xlabel="Time", ylabel="", legend=g is not None, grid_axis="x")
    return ax


# ------------------------------------------------------------------ barcode
@chart("barcode", "Barcode chart", "Data over time",
       [Role("x", "any", "Time / value"), Role("group", "categorical", "Row per group", required=False)],
       [Param("max_groups", "Max rows", "int", 30, min=1, max=100)], aliases=("event plot", "rug timeline"))
def barcode(fig, d, style, max_groups=30):
    """One thin vertical line per observation along the axis, one row per group - shows when events cluster."""
    x, kind, xl = to_axis(d["x"])
    ok = np.isfinite(x)
    if not ok.any():
        return empty_message(fig, style, "No data")
    ax = make_axes(fig, style)
    if d.has("group"):
        g = np.asarray(d["group"], dtype=object).astype(str)
        labs = first_seen(g[ok].tolist())[:max_groups]
        for i, lab in enumerate(labs):
            ax.vlines(x[ok & (g == lab)], i - 0.4, i + 0.4, colors=style.color_for(lab, i), lw=max(style.line_width * 0.4, 0.5), alpha=0.8)
        ax.set_yticks(range(len(labs)), [short(l) for l in labs])
        ax.invert_yaxis()
    else:
        ax.vlines(x[ok], 0, 1, colors=style.color_for("", 0), lw=max(style.line_width * 0.4, 0.5), alpha=0.8)
        ax.set_yticks([])
    ax.set_xlim(np.nanmin(x[ok]) - 0.01 * np.ptp(x[ok]) - 1e-9, np.nanmax(x[ok]) + 0.01 * np.ptp(x[ok]) + 1e-9)
    axis_ticks(ax, kind, xl)
    finish(fig, ax, style, xlabel=d.name("x"), ylabel="", legend=False, grid_axis="x")
    return ax
