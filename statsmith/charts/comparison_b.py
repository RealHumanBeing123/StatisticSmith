"""Comparison charts (part B): parallel coordinates, radar, nightingale rose, waterfall, matrix chart,
small multiples, word cloud, slope chart, table chart, categorical scatter, quadrant chart."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .base import (AGGS, ChartStyle, Param, Role, aggregate, chart, empty_message, finish, first_seen, fmt, make_axes,
                   numeric, short)


def _cats(a):
    return np.asarray(a, dtype=object).astype(str)


def _multi(d, role="vals"):
    cols = [numeric(c) for c in d[role]]
    return cols, d.label_list(role)


# ------------------------------------------------------------------ parallel coordinates
@chart("parallel_coordinates", "Parallel coordinates", "Comparison",
       [Role("vals", "numeric_multi", "Variables (2+)"), Role("group", "categorical", "Colour by", required=False)],
       [Param("max_lines", "Max lines drawn", "int", 2000, min=10, max=50000)])
def parallel_coordinates(fig, d, style, max_lines=2000):
    """One polyline per row across several axes, each axis scaled min-max to 0-1."""
    cols, names = _multi(d)
    if len(cols) < 2:
        return empty_message(fig, style, "Select at least two numeric variables")
    X = np.column_stack(cols)
    ok = np.isfinite(X).all(axis=1)
    g = _cats(d["group"])[ok] if d.has("group") else None
    X = X[ok]
    if X.shape[0] == 0:
        return empty_message(fig, style, "No complete rows")
    lo, hi = X.min(0), X.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    Z = (X - lo) / span
    ax = make_axes(fig, style)
    idx = np.arange(len(Z))
    if len(Z) > max_lines:
        idx = np.random.default_rng(1).choice(len(Z), max_lines, replace=False)
        d.notes.append(f"drew {max_lines:,} of {len(Z):,} lines")
    xs = np.arange(len(names))
    if g is None:
        for i in idx:
            ax.plot(xs, Z[i], color=style.color_for("", 0), alpha=min(1, 0.15 + 20 / max(len(idx), 1)), lw=style.line_width * 0.5)
    else:
        labs = first_seen(g.tolist())
        for j, lab in enumerate(labs):
            sel = [i for i in idx if g[i] == lab]
            for k, i in enumerate(sel):
                ax.plot(xs, Z[i], color=style.color_for(lab, j), alpha=0.35, lw=style.line_width * 0.5, label=lab if k == 0 else None)
    for x in xs:
        ax.axvline(x, color=style.text_color, lw=0.8, alpha=0.6)
        ax.text(x, -0.03, fmt(lo[x], style), ha="center", va="top", fontsize=style.tick_size - 1, color=style.text_color)
        ax.text(x, 1.03, fmt(hi[x], style), ha="center", va="bottom", fontsize=style.tick_size - 1, color=style.text_color)
    ax.set_xticks(xs, [short(n) for n in names])
    ax.set_yticks([])
    finish(fig, ax, style, ylabel="scaled 0 (min) to 1 (max)", grid_axis="y")
    ax.set_ylim(-0.1, 1.1)
    return ax


# ------------------------------------------------------------------ radar
@chart("radar", "Radar / spider chart", "Comparison",
       [Role("vals", "numeric_multi", "Axes (3+)"), Role("group", "categorical", "Series (mean per group)", required=False)],
       [Param("agg", "Aggregate", "choice", "mean", choices=("mean", "median", "sum", "max", "min")),
        Param("fill", "Fill area", "bool", True)], aliases=("spider chart", "web chart"))
def radar(fig, d, style, agg="mean", fill=True):
    """Each variable is a spoke, scaled to its own maximum; one polygon per group."""
    cols, names = _multi(d)
    if len(cols) < 3:
        return empty_message(fig, style, "Select at least three numeric variables")
    f = {"mean": np.nanmean, "median": np.nanmedian, "sum": np.nansum, "max": np.nanmax, "min": np.nanmin}[agg]
    if d.has("group"):
        g = _cats(d["group"])
        labs = first_seen(g.tolist())[:12]
        rows = [[f(c[g == l]) if np.isfinite(c[g == l]).any() else np.nan for c in cols] for l in labs]
    else:
        labs = [agg]
        rows = [[f(c) if np.isfinite(c).any() else np.nan for c in cols]]
    R = np.array(rows, dtype=float)
    top = np.nanmax(np.abs(R), axis=0)
    top[~np.isfinite(top) | (top == 0)] = 1.0
    S = R / top
    ax = make_axes(fig, style, polar=True)
    ang = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
    for i, lab in enumerate(labs):
        v = np.nan_to_num(S[i])
        c = style.color_for(lab, i)
        ax.plot(np.r_[ang, ang[0]], np.r_[v, v[0]], color=c, lw=style.line_width, label=lab)
        if fill:
            ax.fill(np.r_[ang, ang[0]], np.r_[v, v[0]], color=c, alpha=0.2)
    ax.set_xticks(ang, [short(n) for n in names])
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["25%", "50%", "75%", "100%"], fontsize=style.tick_size - 1)
    ax.tick_params(colors=style.text_color, labelsize=style.tick_size)
    ax.grid(style.show_grid, color=style.grid_color)
    d.notes.append("each spoke is scaled to its own maximum across series")
    finish(fig, ax, style, polar=True, legend=len(labs) > 1)
    ax.figure.tight_layout()
    return ax


# ------------------------------------------------------------------ nightingale rose
@chart("nightingale", "Nightingale rose (coxcomb)", "Comparison",
       [Role("x", "categorical", "Category"), Role("y", "numeric", "Value (blank = count)", required=False)],
       [Param("agg", "Aggregate", "choice", "sum", choices=AGGS)], aliases=("coxcomb", "polar area chart"))
def nightingale(fig, d, style, agg="sum"):
    """Equal-angle wedges whose AREA (radius = sqrt(value)) encodes the value."""
    labels, vals = aggregate(d["x"], d.get("y") if d.has("y") else None, agg)
    ok = np.isfinite(vals) & (vals >= 0)
    labels, vals = [l for l, o in zip(labels, ok) if o], vals[ok]
    if not len(vals):
        return empty_message(fig, style, "No data")
    ax = make_axes(fig, style, polar=True)
    n = len(vals)
    w = 2 * np.pi / n
    th = np.arange(n) * w
    r = np.sqrt(vals)
    ax.bar(th, r, width=w * style.bar_width, align="edge", color=style.series_colors(labels), alpha=style.alpha,
           edgecolor=style.background)
    ax.set_xticks(th + w * style.bar_width / 2, [short(l) for l in labels])
    ax.set_yticklabels([])
    ax.grid(style.show_grid, color=style.grid_color)
    if style.show_values:
        for t, rr, v in zip(th + w * style.bar_width / 2, r, vals):
            ax.text(t, rr, fmt(v, style), ha="center", va="bottom", fontsize=style.value_label_size, color=style.text_color)
    finish(fig, ax, style, polar=True, legend=False)
    return ax


# ------------------------------------------------------------------ waterfall
@chart("waterfall", "Waterfall chart", "Comparison",
       [Role("x", "categorical", "Step"), Role("y", "numeric", "Change")],
       [Param("total", "Add final total bar", "bool", True)], aliases=("bridge chart",))
def waterfall(fig, d, style, total=True):
    """Floating bars: each step starts where the previous one ended."""
    labels, vals = aggregate(d["x"], d["y"], "sum")
    vals = np.nan_to_num(vals)
    if not len(vals):
        return empty_message(fig, style, "No data")
    ax = make_axes(fig, style)
    start = np.r_[0.0, np.cumsum(vals)[:-1]]
    pos, neg = style.color_for("increase", 2), style.color_for("decrease", 4)
    cols = [pos if v >= 0 else neg for v in vals]
    pos_i = np.arange(len(vals))
    ax.bar(pos_i, vals, bottom=start, width=style.bar_width, color=cols, alpha=style.alpha)
    ends = start + vals
    for i in range(len(vals) - 1):
        ax.plot([i + style.bar_width / 2, i + 1 - style.bar_width / 2], [ends[i]] * 2, color=style.text_color, lw=0.8, alpha=0.6)
    labs = list(labels)
    if total:
        t = ends[-1]
        ax.bar(len(vals), t, width=style.bar_width, color=style.color_for("total", 0), alpha=style.alpha)
        pos_i = np.r_[pos_i, len(vals)]
        labs.append("Total")
        ends = np.r_[ends, t]
        vals = np.r_[vals, t]
    ax.axhline(0, color=style.text_color, lw=0.8)
    ax.set_xticks(pos_i, [short(l) for l in labs])
    if style.show_values:
        for i, v in enumerate(vals):
            top = max(ends[i], (ends[i] - v) if i < len(vals) - (1 if total else 0) else 0)
            ax.text(i, top, fmt(v, style), ha="center", va="bottom", fontsize=style.value_label_size, color=style.text_color)
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=d.name("y"), legend=False)
    return ax


# ------------------------------------------------------------------ matrix chart (bubble matrix)
@chart("matrix_chart", "Matrix chart", "Comparison",
       [Role("row", "categorical", "Rows"), Role("col", "categorical", "Columns"),
        Role("value", "numeric", "Size (blank = count)", required=False)],
       [Param("agg", "Aggregate", "choice", "sum", choices=AGGS), Param("max_dim", "Max rows/cols", "int", 25, min=2, max=100)],
       aliases=("bubble matrix", "dot matrix"))
def matrix_chart(fig, d, style, agg="sum", max_dim=25):
    """Row x column grid of circles whose area encodes the aggregated value."""
    from .comparison_a import _agg_table
    rows, cols, M = _agg_table(d["row"], d["col"], d.get("value") if d.has("value") else None, agg)
    if not rows or not cols:
        return empty_message(fig, style, "No data")
    rows, cols, M = rows[:max_dim], cols[:max_dim], M[:max_dim, :max_dim]
    ax = make_axes(fig, style)
    Y, X = np.meshgrid(np.arange(len(rows)), np.arange(len(cols)), indexing="ij")
    v = np.nan_to_num(M)
    top = np.nanmax(np.abs(v)) or 1.0
    area = (np.abs(v) / top) * (style.marker_size * 9) ** 2
    ax.scatter(X.ravel(), Y.ravel(), s=area.ravel(), c=[style.color_for("", 0)], alpha=style.alpha)
    ax.set_xticks(range(len(cols)), [short(c) for c in cols])
    ax.set_yticks(range(len(rows)), [short(r) for r in rows])
    ax.set_xlim(-0.7, len(cols) - 0.3)
    ax.set_ylim(len(rows) - 0.3, -0.7)
    if style.show_values:
        for i in range(len(rows)):
            for j in range(len(cols)):
                if v[i, j]:
                    ax.text(j, i, fmt(v[i, j], style), ha="center", va="center", fontsize=style.value_label_size, color=style.text_color)
    finish(fig, ax, style, xlabel=d.name("col"), ylabel=d.name("row"), legend=False, grid_axis="both")
    return ax


# ------------------------------------------------------------------ small multiples
@chart("small_multiples", "Small multiples", "Comparison",
       [Role("x", "numeric", "X"), Role("y", "numeric", "Y"), Role("group", "categorical", "Panel by")],
       [Param("kind", "Panel type", "choice", "scatter", choices=("scatter", "line")),
        Param("max_panels", "Max panels", "int", 12, min=1, max=48),
        Param("max_points", "Max points per panel", "int", 5000, min=100, max=100000)],
       aliases=("trellis", "facet grid", "panel chart"))
def small_multiples(fig, d, style, kind="scatter", max_panels=12, max_points=5000):
    """Same chart repeated for every group, on shared axes so panels are comparable."""
    x, y, g = numeric(d["x"]), numeric(d["y"]), _cats(d["group"])
    ok = np.isfinite(x) & np.isfinite(y)
    labs = first_seen(g[ok].tolist())[:max_panels]
    if not labs:
        return empty_message(fig, style, "No data")
    fig.clear()
    fig.set_size_inches(style.fig_width, style.fig_height, forward=False)
    fig.patch.set_facecolor(style.background)
    n = len(labs)
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    axs = fig.subplots(nrow, ncol, sharex=True, sharey=True, squeeze=False)
    rng = np.random.default_rng(1)
    for k, lab in enumerate(labs):
        ax = axs[k // ncol][k % ncol]
        ax.set_facecolor(style.axes_background)
        sel = np.flatnonzero(ok & (g == lab))
        if sel.size > max_points:
            sel = rng.choice(sel, max_points, replace=False)
        if kind == "line":
            sel = sel[np.argsort(x[sel], kind="stable")]
            ax.plot(x[sel], y[sel], color=style.color_for(lab, k), lw=style.line_width)
        else:
            ax.scatter(x[sel], y[sel], s=style.marker_size ** 2 / 2, color=style.color_for(lab, k), alpha=style.alpha * 0.7)
        ax.set_title(short(lab, 24), fontsize=style.label_size - 1, color=style.text_color)
        ax.tick_params(colors=style.text_color, labelsize=style.tick_size - 1)
        ax.grid(style.show_grid, color=style.grid_color, alpha=0.8)
        for sp in ax.spines.values():
            sp.set_color(style.grid_color)
    for k in range(n, nrow * ncol):
        axs[k // ncol][k % ncol].axis("off")
    if style.title:
        fig.suptitle(style.title, fontsize=style.title_size, color=style.text_color)
    fig.supxlabel(style.xlabel or d.name("x"), fontsize=style.label_size, color=style.text_color)
    fig.supylabel(style.ylabel or d.name("y"), fontsize=style.label_size, color=style.text_color)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return axs[0][0]


# ------------------------------------------------------------------ word cloud
_STOP = set("a an and are as at be but by for from has have he her his i if in into is it its of on or she that the their "
            "them they this to was we were what when which who will with you your not no so than then there these those "
            "been being do does did can could would should".split())


def word_counts(texts, min_len=3, stop=True, top=80):
    c = Counter()
    for t in texts:
        if t is None:
            continue
        for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", str(t).lower()):
            if len(w) >= min_len and not (stop and w in _STOP):
                c[w] += 1
    return c.most_common(top)


@chart("word_cloud", "Word cloud", "Comparison",
       [Role("text", "any", "Text column"), Role("weight", "numeric", "Word weight (optional counts)", required=False)],
       [Param("top", "Max words", "int", 80, min=5, max=300), Param("stopwords", "Drop common words", "bool", True)],
       aliases=("tag cloud",))
def word_cloud(fig, d, style, top=80, stopwords=True):
    """Words sized by frequency, placed on an outward spiral without overlap."""
    t = np.asarray(d["text"], dtype=object)
    if d.has("weight"):
        w = numeric(d["weight"])
        counts = [(str(a), float(b)) for a, b in zip(t, w) if np.isfinite(b) and b > 0]
        agg = Counter()
        for a, b in counts:
            agg[a] += b
        words = agg.most_common(top)
    else:
        words = word_counts(t, stop=stopwords, top=top)
    if not words:
        return empty_message(fig, style, "No words found")
    ax = make_axes(fig, style)
    ax.axis("off")
    W, H = style.fig_width * style.dpi, style.fig_height * style.dpi
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    fig.canvas.draw() if hasattr(fig.canvas, "draw") else None
    mx = words[0][1]
    sizes = [10 + 50 * np.sqrt(c / mx) for _, c in words]
    placed = []
    rng = np.random.default_rng(1)
    for i, ((wd, c), s) in enumerate(zip(words, sizes)):
        px = s * style.dpi / 72
        bw, bh = 0.7 * px * len(wd) + 0.2 * px, 1.15 * px
        for step in range(3000):
            ang = 0.35 * step
            rad = 0.9 * ang
            cx, cy = W / 2 + rad * np.cos(ang) * 1.3, H / 2 + rad * np.sin(ang)
            r = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            if r[0] < 0 or r[1] < 0 or r[2] > W or r[3] > H:
                continue
            if all(r[2] < q[0] or r[0] > q[2] or r[3] < q[1] or r[1] > q[3] for q in placed):
                placed.append(r)
                ax.text(cx, cy, wd, fontsize=s, ha="center", va="center", color=style.color_for(wd, i), alpha=style.alpha)
                break
    if style.title:
        ax.set_title(style.title, fontsize=style.title_size, color=style.text_color)
    if len(placed) < len(words):
        d.notes.append(f"{len(words) - len(placed)} words did not fit")
    return ax


# ------------------------------------------------------------------ slope chart
@chart("slope_chart", "Slope chart", "Comparison",
       [Role("label", "categorical", "Item"), Role("start", "numeric", "Start value"), Role("end", "numeric", "End value")],
       [Param("max_items", "Max items", "int", 25, min=2, max=200)], aliases=("slopegraph",))
def slope_chart(fig, d, style, max_items=25):
    """Two columns of points joined by lines; the slope shows the change per item."""
    lab, a, b = _cats(d["label"]), numeric(d["start"]), numeric(d["end"])
    ok = np.isfinite(a) & np.isfinite(b)
    lab, a, b = lab[ok][:max_items], a[ok][:max_items], b[ok][:max_items]
    if not len(lab):
        return empty_message(fig, style, "No data")
    ax = make_axes(fig, style)
    up, dn = style.color_for("up", 2), style.color_for("down", 4)
    for l, x0, x1 in zip(lab, a, b):
        c = up if x1 >= x0 else dn
        ax.plot([0, 1], [x0, x1], color=c, lw=style.line_width, marker="o", ms=style.marker_size, alpha=style.alpha)
        ax.text(-0.03, x0, f"{short(l, 16)}  {fmt(x0, style)}", ha="right", va="center", fontsize=style.value_label_size, color=style.text_color)
        ax.text(1.03, x1, f"{fmt(x1, style)}  {short(l, 16)}", ha="left", va="center", fontsize=style.value_label_size, color=style.text_color)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1], [d.name("start"), d.name("end")])
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    finish(fig, ax, style, legend=False, grid_axis="y")
    ax.grid(False)
    return ax


# ------------------------------------------------------------------ table chart
@chart("table_chart", "Table chart", "Comparison",
       [Role("vals", "any_multi", "Columns")],
       [Param("rows", "Rows shown", "int", 20, min=1, max=200), Param("heat", "Shade numeric cells", "bool", True)],
       aliases=("table", "heatmap table"))
def table_chart(fig, d, style, rows=20, heat=True):
    """A formatted table; numeric columns can be shaded by value."""
    import matplotlib as mpl
    cols = d["vals"]
    names = d.label_list("vals")
    n = min(rows, len(cols[0]) if cols else 0)
    if n == 0:
        return empty_message(fig, style, "No data")
    ax = make_axes(fig, style)
    ax.axis("off")
    cell, colors = [], []
    cmap = mpl.colormaps[style.cmap]
    norms = []
    for c in cols:
        v = numeric(c[:n])
        isnum = np.isfinite(v).sum() > 0 and np.isfinite(v).sum() == sum(1 for x in c[:n] if x is not None and str(x) != "nan")
        if isnum and np.isfinite(v).any():
            lo, hi = np.nanmin(v), np.nanmax(v)
            norms.append((lo, hi if hi > lo else lo + 1))
        else:
            norms.append(None)
    for i in range(n):
        r, rc = [], []
        for c, nm in zip(cols, norms):
            val = c[i]
            if nm is not None and np.isfinite(numeric([val])[0]):
                f = float(numeric([val])[0])
                r.append(fmt(f, style))
                rc.append(mpl.colors.to_hex(cmap(0.15 + 0.6 * (f - nm[0]) / (nm[1] - nm[0]))) if heat else style.axes_background)
            else:
                r.append(short(val, 20))
                rc.append(style.axes_background)
        cell.append(r)
        colors.append(rc)
    tb = ax.table(cellText=cell, colLabels=[short(x, 18) for x in names], cellColours=colors, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(style.tick_size)
    for (r, c), t in tb.get_celld().items():
        t.set_edgecolor(style.grid_color)
        t.get_text().set_color("#111111" if (r > 0 and heat and norms[c] is not None) else style.text_color)
        if r == 0:
            t.set_facecolor(style.grid_color)
            t.get_text().set_weight("bold")
    if style.title:
        ax.set_title(style.title, fontsize=style.title_size, color=style.text_color)
    return ax


# ------------------------------------------------------------------ categorical scatter
@chart("categorical_scatter", "Categorical scatter plot", "Comparison",
       [Role("x", "categorical", "Category"), Role("y", "numeric", "Value"), Role("group", "categorical", "Colour by", required=False)],
       [Param("max_points", "Max points", "int", 20000, min=100, max=500000), Param("jitter", "Jitter width", "float", 0.25, min=0, max=0.5)],
       aliases=("dot strip",))
def categorical_scatter(fig, d, style, max_points=20000, jitter=0.25):
    """Every observation as a dot above its category (seeded horizontal jitter reduces overlap)."""
    x, y = _cats(d["x"]), numeric(d["y"])
    ok = np.isfinite(y)
    g = _cats(d["group"]) if d.has("group") else None
    idx = np.flatnonzero(ok)
    if idx.size > max_points:
        idx = np.random.default_rng(1).choice(idx, max_points, replace=False)
        d.notes.append(f"drew {max_points:,} of {int(ok.sum()):,} points")
    if not idx.size:
        return empty_message(fig, style, "No data")
    labs = first_seen(x.tolist())
    code = {l: i for i, l in enumerate(labs)}
    px = np.array([code[v] for v in x[idx]], dtype=float) + np.random.default_rng(2).uniform(-jitter, jitter, idx.size)
    ax = make_axes(fig, style)
    if g is None:
        ax.scatter(px, y[idx], s=style.marker_size ** 2, color=style.color_for("", 0), alpha=style.alpha * 0.6)
    else:
        for j, gl in enumerate(first_seen(g[idx].tolist())):
            m = g[idx] == gl
            ax.scatter(px[m], y[idx][m], s=style.marker_size ** 2, color=style.color_for(gl, j), alpha=style.alpha * 0.6, label=gl)
    ax.set_xticks(range(len(labs)), [short(l) for l in labs])
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=d.name("y"))
    return ax


# ------------------------------------------------------------------ quadrant chart
@chart("quadrant_chart", "Quadrant chart", "Comparison",
       [Role("x", "numeric", "X"), Role("y", "numeric", "Y"), Role("label", "categorical", "Label", required=False)],
       [Param("split", "Divide at", "choice", "median", choices=("median", "mean", "zero")),
        Param("max_points", "Max points", "int", 20000, min=100, max=500000)], aliases=("four quadrant chart",))
def quadrant_chart(fig, d, style, split="median", max_points=20000):
    """Scatter divided into four quadrants by the chosen X and Y cut points; each quadrant shows its count."""
    x, y = numeric(d["x"]), numeric(d["y"])
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    lab = _cats(d["label"])[ok] if d.has("label") else None
    if not x.size:
        return empty_message(fig, style, "No data")
    cx, cy = {"median": (np.median(x), np.median(y)), "mean": (x.mean(), y.mean()), "zero": (0.0, 0.0)}[split]
    ax = make_axes(fig, style)
    idx = np.arange(x.size)
    if idx.size > max_points:
        idx = np.random.default_rng(1).choice(idx, max_points, replace=False)
    q = (x[idx] >= cx).astype(int) + 2 * (y[idx] >= cy).astype(int)
    for k in range(4):
        m = q == k
        ax.scatter(x[idx][m], y[idx][m], s=style.marker_size ** 2, color=style.color_for(f"q{k}", k), alpha=style.alpha * 0.7)
    ax.axvline(cx, color=style.text_color, lw=1.0, ls="--")
    ax.axhline(cy, color=style.text_color, lw=1.0, ls="--")
    xl, yl = ax.get_xlim(), ax.get_ylim()
    cnt = [int(((x >= cx) == (k % 2 == 1)).__and__((y >= cy) == (k >= 2)).sum()) for k in range(4)]
    pos = [(xl[0], yl[0], "left", "bottom"), (xl[1], yl[0], "right", "bottom"), (xl[0], yl[1], "left", "top"), (xl[1], yl[1], "right", "top")]
    for k, (px, py, ha, va) in enumerate(pos):
        ax.text(px, py, f" n = {cnt[k]:,} ", ha=ha, va=va, fontsize=style.value_label_size + 1, color=style.text_color, alpha=0.8)
    if lab is not None and len(idx) <= 60:
        for i in idx:
            ax.annotate(short(lab[i], 14), (x[i], y[i]), fontsize=style.value_label_size - 1, color=style.text_color, xytext=(3, 3), textcoords="offset points")
    d.notes.append(f"quadrant counts (BL,BR,TL,TR) = {cnt}; cut at x={cx:.4g}, y={cy:.4g}")
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=d.name("y"), legend=False, grid_axis="both")
    return ax
