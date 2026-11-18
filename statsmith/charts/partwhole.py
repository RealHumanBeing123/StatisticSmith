"""Part-to-whole & hierarchical charts."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

from . import layout as L
from .base import (AGGS, ChartData, Param, Role, aggregate, chart, empty_message, finish, fmt, make_axes, numeric, short)

CAT = Role("x", "categorical", "Category")
VAL = Role("y", "numeric", "Value (blank = count rows)", required=False)
GRP = Role("group", "categorical", "Group / segment")
LEVELS = Role("levels", "any_multi", "Hierarchy levels (outer to inner)")
SUM_AGGS = ("sum", "count")


def _str(a):
    return np.asarray(a, dtype=object).astype(str)


def xtab(d: ChartData, agg="sum", order="appearance"):
    """Cross-tab of x (rows) by group (cols) -> DataFrame of sums / counts, in first-appearance order."""
    x = pd.Series(_str(d["x"]))
    g = pd.Series(_str(d["group"])) if d.has("group") else pd.Series(["(all)"] * len(x))
    y = pd.Series(numeric(d["y"])) if (d.has("y") and agg != "count") else pd.Series(np.ones(len(x)))
    ok = y.notna()
    df = pd.DataFrame({"x": x[ok], "g": g[ok], "y": y[ok]})
    t = df.pivot_table(index="x", columns="g", values="y", aggfunc="sum", fill_value=0.0, sort=False)
    t = t.reindex(index=list(dict.fromkeys(x[ok])), columns=list(dict.fromkeys(g[ok])))
    if order == "value_desc":
        t = t.loc[t.sum(axis=1).sort_values(ascending=False, kind="stable").index]
    elif order == "value_asc":
        t = t.loc[t.sum(axis=1).sort_values(kind="stable").index]
    elif order == "alpha":
        t = t.sort_index()
    return t


def _slices(d, agg, sort, min_pct, max_slices=30):
    labels, vals = aggregate(d["x"], d.get("y") if d.has("y") else None, agg, "value_desc" if sort else "appearance")
    labels, vals = list(labels), np.asarray(vals, float)
    keep = np.isfinite(vals) & (vals > 0)
    labels, vals = [l for l, k in zip(labels, keep) if k], vals[keep]
    if vals.size == 0:
        return labels, vals
    tot = vals.sum()
    small = (vals / tot * 100 < min_pct)
    if len(vals) > max_slices:
        small |= np.arange(len(vals)) >= max_slices - 1
    if small.sum() > 1:
        labels = [l for l, s in zip(labels, small) if not s] + ["Other"]
        vals = np.r_[vals[~small], vals[small].sum()]
    return labels, vals


_ORD = ("appearance", "value_desc", "value_asc", "alpha")
_COMMON = [Param("agg", "Aggregate", "choice", "sum", choices=SUM_AGGS), Param("order", "Order", "choice", "appearance", choices=_ORD)]


# ------------------------------------------------------------------ stacked bar / column
def _stacked(fig, d, style, agg, order, pct, horizontal):
    ax = make_axes(fig, style)
    t = xtab(d, agg, order)
    if t.empty:
        return empty_message(fig, style, "No data")
    M = t.values.astype(float)
    if pct:
        M = M / np.where(M.sum(axis=1, keepdims=True) == 0, 1, M.sum(axis=1, keepdims=True)) * 100
    pos = np.arange(len(t))
    cols = style.series_colors(list(t.columns))
    bottom = np.zeros(len(t))
    for j, g in enumerate(t.columns):
        kw = dict(color=cols[j], alpha=style.alpha, label=g if len(t.columns) > 1 else None, edgecolor=style.axes_background, linewidth=0.6)
        if horizontal:
            ax.barh(pos, M[:, j], left=bottom, height=style.bar_width, **kw)
        else:
            ax.bar(pos, M[:, j], bottom=bottom, width=style.bar_width, **kw)
        if style.show_values:
            for p, b, v in zip(pos, bottom, M[:, j]):
                if v > 0:
                    (ax.text(b + v / 2, p, fmt(v, style), ha="center", va="center", fontsize=style.value_label_size, color=style.text_color) if horizontal
                     else ax.text(p, b + v / 2, fmt(v, style), ha="center", va="center", fontsize=style.value_label_size, color=style.text_color))
        bottom += M[:, j]
    lab = [short(i) for i in t.index]
    if horizontal:
        ax.set_yticks(pos, lab); ax.invert_yaxis()
    else:
        ax.set_xticks(pos, lab)
    meas = "Percent" if pct else ("Count" if (agg == "count" or not d.has("y")) else d.name("y"))
    cat = d.name("x")
    finish(fig, ax, style, xlabel=meas if horizontal else cat, ylabel=cat if horizontal else meas, legend=len(t.columns) > 1,
           grid_axis="x" if horizontal else "y")
    return ax


_STK = [Param("pct", "100% stacked", "bool", False)]


@chart("stacked_bar", "Stacked bar chart", "Part-to-whole & hierarchical", [CAT, Role("group", "categorical", "Stack by", required=False), VAL],
       _COMMON + _STK, aliases=("stacked bar graph", "100% stacked bar"))
def stacked_bar(fig, d, style, agg="sum", order="appearance", pct=False):
    """Horizontal bars divided into segments (one per group); 'pct' makes every bar 100 %."""
    return _stacked(fig, d, style, agg, order, pct, True)


@chart("stacked_column", "Stacked column chart", "Part-to-whole & hierarchical", [CAT, Role("group", "categorical", "Stack by", required=False), VAL],
       _COMMON + _STK, aliases=("stacked column graph", "100% stacked column"))
def stacked_column(fig, d, style, agg="sum", order="appearance", pct=False):
    """Vertical columns divided into segments (one per group); 'pct' makes every column 100 %."""
    return _stacked(fig, d, style, agg, order, pct, False)


# ------------------------------------------------------------------ diverging
def diverging_segments(M: np.ndarray):
    """Likert-style diverging layout for a row-percentage matrix M (rows x k levels).
    Returns (left, width) arrays, same shape as M: segment j spans [left, left+width]; the first k//2 levels go left of 0,
    an odd middle level is split evenly around 0, the rest go right."""
    n, k = M.shape
    left = np.zeros_like(M)
    for i in range(n):
        row = M[i]
        mid = k / 2.0
        neg = row[: k // 2].sum() + (row[k // 2] / 2 if k % 2 else 0.0)
        x = -neg
        for j in range(k):
            left[i, j] = x
            x += row[j]
    return left, M


def _diverging(fig, d, style, agg, order, horizontal, center):
    ax = make_axes(fig, style)
    pos_c, neg_c = style.palette_colors()[0], style.palette_colors()[4]
    if d.has("group"):
        t = xtab(d, agg, order)
        if t.empty or t.shape[1] < 2:
            return empty_message(fig, style, "Need a grouping variable with at least 2 levels")
        M = t.values.astype(float)
        M = M / np.where(M.sum(1, keepdims=True) == 0, 1, M.sum(1, keepdims=True)) * 100
        left, w = diverging_segments(M)
        k = M.shape[1]
        # diverging palette: negative levels red-ish -> neutral grey -> positive blue-ish
        import matplotlib as mpl
        cm = mpl.colormaps[style.diverging_cmap]
        cols = [cm(v) for v in np.linspace(0.05, 0.95, k)]
        cols = cols[::-1] if style.diverging_cmap.endswith("_r") else cols
        pos = np.arange(len(t))
        for j, g in enumerate(t.columns):
            (ax.barh(pos, w[:, j], left=left[:, j], height=style.bar_width, color=cols[j], alpha=style.alpha, label=g, edgecolor=style.axes_background, lw=0.5)
             if horizontal else
             ax.bar(pos, w[:, j], bottom=left[:, j], width=style.bar_width, color=cols[j], alpha=style.alpha, label=g, edgecolor=style.axes_background, lw=0.5))
        (ax.axvline if horizontal else ax.axhline)(0, color=style.text_color, lw=1)
        lab = [short(i) for i in t.index]
        (ax.set_yticks(pos, lab) if horizontal else ax.set_xticks(pos, lab))
        if horizontal:
            ax.invert_yaxis()
        cat, meas = d.name("x"), "Percent of row"
        finish(fig, ax, style, xlabel=meas if horizontal else cat, ylabel=cat if horizontal else meas, legend=True,
               grid_axis="x" if horizontal else "y")
        return ax
    labels, vals = aggregate(d["x"], d.get("y") if d.has("y") else None, "mean" if d.has("y") else "count", "appearance")
    dev = vals - center
    pos = np.arange(len(labels))
    cols = [pos_c if v >= 0 else neg_c for v in dev]
    (ax.barh(pos, dev, left=center, height=style.bar_width, color=cols, alpha=style.alpha) if horizontal else
     ax.bar(pos, dev, bottom=center, width=style.bar_width, color=cols, alpha=style.alpha))
    (ax.axvline if horizontal else ax.axhline)(center, color=style.text_color, lw=1)
    (ax.set_yticks(pos, [short(l) for l in labels]) if horizontal else ax.set_xticks(pos, [short(l) for l in labels]))
    if horizontal:
        ax.invert_yaxis()
    cat = d.name("x"); meas = ("Mean of " + d.name("y")) if d.has("y") else "Count"
    finish(fig, ax, style, xlabel=meas if horizontal else cat, ylabel=cat if horizontal else meas, legend=False,
           grid_axis="x" if horizontal else "y")
    return ax


_DIV = [Param("agg", "Aggregate", "choice", "sum", choices=SUM_AGGS), Param("order", "Order", "choice", "appearance", choices=_ORD),
        Param("center", "Centre value (no group)", "float", 0.0)]
_DIVR = [CAT, Role("group", "categorical", "Response level (ordered; optional)", required=False), VAL]


@chart("diverging_bar", "Diverging bar chart", "Part-to-whole & hierarchical", _DIVR, _DIV, aliases=("diverging stacked bar", "likert chart"))
def diverging_bar(fig, d, style, agg="sum", order="appearance", center=0.0):
    """Bars diverge from a centre line: positive/negative deviation, or - with a level column - a Likert-style
    diverging stacked bar (first half of the levels left of zero, second half right)."""
    return _diverging(fig, d, style, agg, order, True, center)


@chart("diverging_column", "Diverging column chart", "Part-to-whole & hierarchical", _DIVR, _DIV, aliases=("diverging stacked column",))
def diverging_column(fig, d, style, agg="sum", order="appearance", center=0.0):
    """Vertical version of the diverging bar chart."""
    return _diverging(fig, d, style, agg, order, False, center)


@chart("population_pyramid", "Population pyramid", "Part-to-whole & hierarchical",
       [Role("x", "categorical", "Age group / category"), Role("group", "categorical", "Two groups (e.g. sex)"), VAL],
       [Param("agg", "Aggregate", "choice", "sum", choices=SUM_AGGS), Param("order", "Order", "choice", "appearance", choices=_ORD),
        Param("percent", "Show % of total", "bool", False)], aliases=("age-sex pyramid", "age pyramid"))
def population_pyramid(fig, d, style, agg="sum", order="appearance", percent=False):
    """Back-to-back horizontal bars: first group to the left (negative axis), second group to the right."""
    ax = make_axes(fig, style)
    t = xtab(d, agg, order)
    if t.shape[1] < 2:
        return empty_message(fig, style, "Need a grouping variable with 2 levels")
    t = t.iloc[:, :2]
    M = t.values.astype(float)
    if percent:
        M = M / M.sum() * 100
    pos = np.arange(len(t))
    cols = style.series_colors(list(t.columns))
    ax.barh(pos, -M[:, 0], height=style.bar_width, color=cols[0], alpha=style.alpha, label=t.columns[0])
    ax.barh(pos, M[:, 1], height=style.bar_width, color=cols[1], alpha=style.alpha, label=t.columns[1])
    ax.set_yticks(pos, [short(i) for i in t.index])
    ax.axvline(0, color=style.text_color, lw=1)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks, [fmt(abs(v), style).replace(".00", "") for v in ticks])
    finish(fig, ax, style, xlabel="Percent" if percent else "Count", ylabel=d.name("x"), legend=True, grid_axis="x")
    return ax


# ------------------------------------------------------------------ waffle / icon array
def largest_remainder(shares: np.ndarray, total: int) -> np.ndarray:
    """Integers summing to ``total`` that best match ``shares`` (Hamilton / largest-remainder method)."""
    raw = shares / shares.sum() * total
    base = np.floor(raw).astype(int)
    rem = total - base.sum()
    order = np.argsort(-(raw - base), kind="stable")
    base[order[:rem]] += 1
    return base


def _grid_chart(fig, d, style, agg, n, cols_n, shape):
    ax = make_axes(fig, style)
    labels, vals = _slices(d, agg, True, 0.0)
    if not len(vals):
        return empty_message(fig, style, "No data")
    cnt = largest_remainder(vals, n)
    cols = style.series_colors(labels)
    rows = math.ceil(n / cols_n)
    k = 0
    for lab_i, c in enumerate(cnt):
        for _ in range(c):
            r, cc = divmod(k, cols_n)
            x, y = cc, rows - 1 - r
            if shape == "square":
                ax.add_patch(Rectangle((x + 0.05, y + 0.05), 0.9, 0.9, color=cols[lab_i], alpha=style.alpha, lw=0))
            else:
                ax.add_patch(Circle((x + 0.5, y + 0.5), 0.42, color=cols[lab_i], alpha=style.alpha, lw=0))
            k += 1
    for lab_i, lab in enumerate(labels):
        ax.add_patch(Rectangle((0, 0), 0, 0, color=cols[lab_i], label=f"{lab} ({vals[lab_i] / vals.sum() * 100:.1f}%)"))
    ax.set_xlim(0, cols_n); ax.set_ylim(0, rows); ax.set_aspect("equal"); ax.axis("off")
    finish(fig, ax, style, legend=True, axes_decor=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=style.legend_size, frameon=False, labelcolor=style.text_color)
    d.notes.append(f"each {'square' if shape == 'square' else 'icon'} = {100 / n:.3g}% of the total")
    return ax


@chart("waffle", "Waffle chart", "Part-to-whole & hierarchical", [CAT, VAL],
       [Param("agg", "Aggregate", "choice", "sum", choices=SUM_AGGS), Param("n", "Squares", "int", 100, min=10, max=1000),
        Param("columns", "Columns", "int", 10, min=2, max=50)], aliases=("waffle graph", "square pie"))
def waffle(fig, d, style, agg="sum", n=100, columns=10):
    """Grid of squares; each square is an equal share of the total (largest-remainder rounding)."""
    return _grid_chart(fig, d, style, agg, n, columns, "square")


@chart("icon_array", "Icon array", "Part-to-whole & hierarchical", [CAT, VAL],
       [Param("agg", "Aggregate", "choice", "sum", choices=SUM_AGGS), Param("n", "Icons", "int", 100, min=10, max=1000),
        Param("columns", "Columns", "int", 10, min=2, max=50)], aliases=("icon array chart", "unit chart"))
def icon_array(fig, d, style, agg="sum", n=100, columns=10):
    """Grid of round icons; each icon is an equal share of the total (largest-remainder rounding)."""
    return _grid_chart(fig, d, style, agg, n, columns, "circle")


# ------------------------------------------------------------------ pie / donut / semicircle
def _pie(fig, d, style, agg, min_pct, show_pct, start, inner, half):
    ax = make_axes(fig, style)
    labels, vals = _slices(d, agg, True, min_pct)
    if not len(vals):
        return empty_message(fig, style, "No positive values to plot")
    cols = style.series_colors(labels)
    span = 180.0 if half else 360.0
    frac = vals / vals.sum()
    ang = start if not half else 180.0                       # half: sweep from the left (180°) clockwise over the top
    for i, f in enumerate(frac):
        sweep = f * span
        t1, t2 = (ang - sweep, ang) if True else (ang, ang + sweep)
        ax.add_patch(Wedge((0, 0), 1.0, t1, t2, width=(1 - inner) if inner else None, facecolor=cols[i], alpha=style.alpha,
                           edgecolor=style.axes_background, lw=1.2, label=labels[i]))
        mid = math.radians((t1 + t2) / 2)
        if show_pct or style.show_values:
            rr = (1 + inner) / 2 if inner else 0.68
            txt = f"{f * 100:.1f}%" if show_pct else fmt(vals[i], style)
            if f > 0.025:
                ax.text(rr * math.cos(mid), rr * math.sin(mid), txt, ha="center", va="center", fontsize=style.value_label_size + 1,
                        color="white", fontweight="bold")
        ang = t1
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.15 if half else -1.15, 1.15); ax.set_aspect("equal"); ax.axis("off")
    if inner and not half:
        ax.text(0, 0, f"{vals.sum():,.4g}", ha="center", va="center", fontsize=style.title_size, color=style.text_color)
    finish(fig, ax, style, legend=True, axes_decor=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=style.legend_size, frameon=False, labelcolor=style.text_color) if style.show_legend else None
    return ax


_PIE_P = [Param("agg", "Aggregate", "choice", "sum", choices=SUM_AGGS), Param("min_pct", "Group slices below % into 'Other'", "float", 2.0, min=0, max=50),
          Param("show_pct", "Label with %", "bool", True), Param("start", "Start angle (deg)", "float", 90.0, min=0, max=360)]


@chart("pie", "Pie chart", "Part-to-whole & hierarchical", [CAT, VAL], _PIE_P, aliases=("pie graph",))
def pie(fig, d, style, agg="sum", min_pct=2.0, show_pct=True, start=90.0):
    """Angles proportional to each category's share of the total, largest slice first, clockwise."""
    return _pie(fig, d, style, agg, min_pct, show_pct, start, 0.0, False)


@chart("donut", "Donut chart", "Part-to-whole & hierarchical", [CAT, VAL], _PIE_P + [Param("hole", "Hole size", "float", 0.55, min=0.1, max=0.9)],
       aliases=("doughnut chart",))
def donut(fig, d, style, agg="sum", min_pct=2.0, show_pct=True, start=90.0, hole=0.55):
    """Pie chart with a hole; the total is printed in the centre."""
    return _pie(fig, d, style, agg, min_pct, show_pct, start, hole, False)


@chart("semicircle_donut", "Semi-circle donut chart", "Part-to-whole & hierarchical", [CAT, VAL],
       _PIE_P[:3] + [Param("hole", "Hole size", "float", 0.55, min=0.1, max=0.9)], aliases=("half donut", "semi-circle donut", "half pie"))
def semicircle_donut(fig, d, style, agg="sum", min_pct=2.0, show_pct=True, hole=0.55):
    """Donut drawn over 180 degrees (left to right) - good for few categories."""
    return _pie(fig, d, style, agg, min_pct, show_pct, 180.0, hole, True)


# ------------------------------------------------------------------ marimekko
@chart("marimekko", "Marimekko chart", "Part-to-whole & hierarchical", [CAT, GRP, VAL],
       [Param("agg", "Aggregate", "choice", "sum", choices=SUM_AGGS)], aliases=("mekko chart", "mosaic plot", "variable width stacked bar"))
def marimekko(fig, d, style, agg="sum"):
    """Column width = category total, segment height = share of that category - area = joint amount."""
    ax = make_axes(fig, style)
    t = xtab(d, agg)
    tot = t.sum(axis=1).values.astype(float)
    if t.empty or tot.sum() <= 0:
        return empty_message(fig, style, "No data")
    cols = style.series_colors(list(t.columns))
    x0 = 0.0
    W = tot / tot.sum() * 100
    for i, (idx, row) in enumerate(t.iterrows()):
        y0 = 0.0
        for j, g in enumerate(t.columns):
            h = row[g] / tot[i] * 100 if tot[i] else 0
            ax.add_patch(Rectangle((x0, y0), W[i], h, facecolor=cols[j], alpha=style.alpha, edgecolor=style.axes_background, lw=1.0,
                                   label=g if i == 0 else None))
            if style.show_values and h > 6 and W[i] > 4:
                ax.text(x0 + W[i] / 2, y0 + h / 2, f"{h:.0f}%", ha="center", va="center", fontsize=style.value_label_size, color="white")
            y0 += h
        x0 += W[i]
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ctr = np.cumsum(W) - W / 2
    ax.set_xticks(ctr, [short(i, 12) for i in t.index])
    finish(fig, ax, style, xlabel=d.name("x") + " (width = share of total)", ylabel="Share within category (%)", legend=len(t.columns) > 1, grid_axis="y")
    return ax


# ------------------------------------------------------------------ hierarchical charts
def _tree_from(d: ChartData):
    lv = d["levels"]
    cols = [_str(c) for c in lv]
    paths = list(zip(*cols))
    vals = numeric(d["value"]) if d.has("value") else None
    return L.build_tree(paths, vals)


_HIER = [LEVELS, Role("value", "numeric", "Size (blank = count rows)", required=False)]


def _label_color(rgb):
    r, g, b = rgb[:3]
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.6 else "white"


@chart("treemap", "Treemap", "Part-to-whole & hierarchical", _HIER,
       [Param("labels_depth", "Label depth (0 = deepest)", "int", 0, min=0, max=6), Param("pad", "Padding", "float", 0.6, min=0, max=5)],
       aliases=("tree map", "treemap chart", "squarified treemap"))
def treemap(fig, d, style, labels_depth=0, pad=0.6):
    """Nested rectangles (squarified, Bruls et al. 2000): area = value, colour = top-level branch."""
    from matplotlib.colors import to_rgb
    ax = make_axes(fig, style)
    root = _tree_from(d)
    if not root.children:
        return empty_message(fig, style, "No positive values")
    W, H = 100.0, 100.0 * style.fig_height / style.fig_width
    nodes = L.treemap_layout(root, W, H, pad=pad, header=0.0)
    tops = [c.name for c in root.children]
    cols = dict(zip(tops, style.series_colors(tops)))
    maxd = max(n.depth for n in nodes)
    show = labels_depth or maxd
    for n in nodes:
        if n.children and n.depth < maxd:
            ax.add_patch(Rectangle((n.x, n.y), n.w, n.h, fill=False, edgecolor=style.axes_background, lw=1.5))
            continue
        c = cols[n.top]
        ax.add_patch(Rectangle((n.x, n.y), n.w, n.h, facecolor=c, alpha=style.alpha, edgecolor=style.axes_background, lw=1.0))
        if n.w > 6 and n.h > 3.5:
            ax.text(n.x + n.w / 2, n.y + n.h / 2, f"{short(n.name, 14)}\n{n.value:,.4g}", ha="center", va="center",
                    fontsize=max(style.value_label_size - 1, 5), color=_label_color(to_rgb(c)), clip_on=True)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    for t in tops:
        ax.add_patch(Rectangle((0, 0), 0, 0, color=cols[t], label=t))
    finish(fig, ax, style, legend=len(tops) > 1, axes_decor=False)
    if style.show_legend and len(tops) > 1:
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=style.legend_size, frameon=False, labelcolor=style.text_color)
    return ax


@chart("circular_treemap", "Circular treemap", "Part-to-whole & hierarchical", _HIER, [], aliases=("circle packing", "bubble treemap", "packed circles"))
def circular_treemap(fig, d, style):
    """Nested circle packing: circle area = value; parents enclose their children."""
    ax = make_axes(fig, style)
    root = _tree_from(d)
    if not root.children:
        return empty_message(fig, style, "No positive values")
    nodes = L.circle_pack_tree(root)
    tops = [c.name for c in root.children]
    cols = dict(zip(tops, style.series_colors(tops)))
    maxd = max(n.depth for n in nodes)
    for n in sorted(nodes, key=lambda q: q.depth):
        leaf = n.depth == maxd or not n.children
        ax.add_patch(Circle((n.x, n.y), n.r, facecolor=cols[n.top], alpha=style.alpha if leaf else 0.15, edgecolor=cols[n.top] if not leaf else style.axes_background, lw=1.2))
        if leaf and n.r > 0.35 * max(c.r for c in nodes if c.depth == maxd):
            ax.text(n.x, n.y, short(n.name, 10), ha="center", va="center", fontsize=max(style.value_label_size - 1, 5), color="white")
    R = max(np.hypot(n.x, n.y) + n.r for n in nodes)
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal"); ax.axis("off")
    for t in tops:
        ax.add_patch(Rectangle((0, 0), 0, 0, color=cols[t], label=t))
    finish(fig, ax, style, legend=len(tops) > 1, axes_decor=False)
    if style.show_legend and len(tops) > 1:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=style.legend_size, frameon=False, labelcolor=style.text_color)
    return ax


@chart("convex_treemap", "Convex treemap", "Part-to-whole & hierarchical",
       [Role("levels", "any_multi", "Categories (first column is used)"), Role("value", "numeric", "Size (blank = count rows)", required=False)], [],
       aliases=("voronoi treemap", "convex treemaps"))
def convex_treemap(fig, d, style):
    """Voronoi-style treemap: cells are convex polygons (weighted/power diagram) whose areas match the values (to <1 %)."""
    ax = make_axes(fig, style)
    first = _str(d["levels"][0])
    v = numeric(d["value"]) if d.has("value") else np.ones(len(first))
    labels, vals = aggregate(first, v, "sum", "value_desc")
    keep = np.isfinite(vals) & (vals > 0)
    labels, vals = [l for l, k in zip(labels, keep) if k][:40], vals[keep][:40]
    if len(vals) < 2:
        return empty_message(fig, style, "Need at least 2 categories")
    from matplotlib.colors import ListedColormap, to_rgb
    w, h = 360, int(round(360 * style.fig_height / style.fig_width))
    lab, sites, err = L.power_treemap(list(vals), w, h)
    cols = style.series_colors(labels)
    ax.imshow(lab, cmap=ListedColormap([to_rgb(c) for c in cols]), vmin=-0.5, vmax=len(vals) - 0.5, interpolation="nearest", alpha=style.alpha)
    for i in range(len(vals)):
        ax.contour((lab == i).astype(float), levels=[0.5], colors=[style.axes_background], linewidths=1.6)
        ys, xs = np.nonzero(lab == i)
        if xs.size > 200:
            ax.text(xs.mean(), ys.mean(), f"{short(labels[i], 12)}\n{vals[i]:,.4g}", ha="center", va="center", fontsize=max(style.value_label_size - 1, 5), color="white")
    ax.axis("off")
    d.notes.append(f"cell areas match values within {err * 100:.2f}% of the total")
    for i, l in enumerate(labels):
        ax.add_patch(Rectangle((0, 0), 0, 0, color=cols[i], label=l))
    finish(fig, ax, style, legend=len(labels) > 1, axes_decor=False)
    return ax


@chart("sunburst", "Sunburst chart", "Part-to-whole & hierarchical", _HIER, [], aliases=("sunburst diagram", "radial treemap", "ring chart"))
def sunburst(fig, d, style):
    """Concentric rings, one per hierarchy level; angle = share of the parent's value."""
    ax = make_axes(fig, style)
    root = _tree_from(d)
    if not root.children:
        return empty_message(fig, style, "No positive values")
    depth = max(len(c) for c in [d["levels"]])
    tops = [c.name for c in root.children]
    cols = dict(zip(tops, style.series_colors(tops)))
    ring = 1.0 / depth
    def rec(n, a0, a1, lvl):
        tot = sum(c.value for c in n.children)
        a = a0
        for c in n.children:
            span = (a1 - a0) * c.value / tot
            ax.add_patch(Wedge((0, 0), (lvl + 1) * ring, a, a + span, width=ring * 0.97, facecolor=cols[c.top], alpha=max(style.alpha - 0.12 * lvl, 0.35),
                               edgecolor=style.axes_background, lw=0.8))
            if span > 12:
                mid = math.radians(a + span / 2); rr = (lvl + 0.5) * ring
                ax.text(rr * math.cos(mid), rr * math.sin(mid), short(c.name, 9), ha="center", va="center", fontsize=max(style.value_label_size - 1, 5), color="white")
            rec(c, a, a + span, lvl + 1)
            a += span
    rec(root, 0.0, 360.0, 0)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect("equal"); ax.axis("off")
    for t in tops:
        ax.add_patch(Rectangle((0, 0), 0, 0, color=cols[t], label=t))
    finish(fig, ax, style, legend=len(tops) > 1, axes_decor=False)
    if style.show_legend and len(tops) > 1:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=style.legend_size, frameon=False, labelcolor=style.text_color)
    return ax


# ------------------------------------------------------------------ dendrogram
@chart("dendrogram", "Dendrogram", "Part-to-whole & hierarchical",
       [Role("dims", "numeric_multi", "Variables to cluster on"), Role("label", "any", "Row labels", required=False)],
       [Param("by", "Cluster", "choice", "rows", choices=("rows", "variables")),
        Param("method", "Linkage", "choice", "ward", choices=("ward", "average", "complete", "single")),
        Param("standardize", "Standardise variables", "bool", True), Param("max_rows", "Max rows (random sample)", "int", 150, min=5, max=1000)],
       aliases=("cluster dendrogram", "hierarchical clustering tree"))
def dendrogram(fig, d, style, by="rows", method="ward", standardize=True, max_rows=150):
    """Agglomerative hierarchical clustering tree (scipy linkage; Euclidean distance)."""
    from scipy.cluster import hierarchy as H
    ax = make_axes(fig, style)
    X = np.column_stack([numeric(c) for c in d["dims"]])
    names = d.label_list("dims")
    lab = _str(d["label"]) if d.has("label") else np.array([str(i + 1) for i in range(len(X))])
    ok = np.isfinite(X).all(axis=1)
    X, lab = X[ok], lab[ok]
    if len(X) < 3:
        return empty_message(fig, style, "Need at least 3 complete rows")
    if len(X) > max_rows:
        sel = np.sort(np.random.default_rng(1).choice(len(X), max_rows, replace=False))
        X, lab = X[sel], lab[sel]
        d.notes.append(f"random sample of {max_rows} rows (seed 1)")
    if standardize:
        sd = X.std(axis=0, ddof=1); sd[sd == 0] = 1
        X = (X - X.mean(axis=0)) / sd
    if by == "variables":
        M, lbl = X.T, names
        if len(M) < 3:
            return empty_message(fig, style, "Need at least 3 variables to cluster variables")
    else:
        M, lbl = X, list(lab)
    Z = H.linkage(M, method=method, metric="euclidean")
    H.dendrogram(Z, labels=[short(l, 12) for l in lbl], ax=ax, color_threshold=0.7 * Z[-1, 2], above_threshold_color=style.text_color,
                 leaf_rotation=90 if len(lbl) <= 40 else 0, leaf_font_size=style.tick_size, no_labels=len(lbl) > 60)
    for ln in ax.collections:
        ln.set_linewidth(style.line_width)
    finish(fig, ax, style, xlabel="", ylabel=f"{method.capitalize()} linkage distance", legend=False)
    return ax


# ------------------------------------------------------------------ Venn / Euler
def _member(a) -> np.ndarray:
    arr = np.asarray(a, dtype=object)
    out = np.zeros(len(arr), dtype=bool)
    for i, v in enumerate(arr):
        if v is None:
            continue
        if isinstance(v, str):
            out[i] = v.strip().lower() in ("1", "true", "t", "yes", "y", "x")
        else:
            try:
                out[i] = bool(v) and not (isinstance(v, float) and math.isnan(v))
            except Exception:
                out[i] = False
    return out


def region_counts(sets: list[np.ndarray]) -> dict[tuple, int]:
    """Exact count of rows for every membership pattern (e.g. (1,0,1) = in A and C but not B)."""
    k = len(sets)
    M = np.column_stack(sets).astype(int)
    code = (M * (1 << np.arange(k)[::-1])).sum(axis=1)
    cnt = np.bincount(code, minlength=1 << k)
    return {tuple((c >> (k - 1 - j)) & 1 for j in range(k)): int(cnt[c]) for c in range(1, 1 << k)}


_SETS = Role("sets", "any_multi", "Set membership columns (2-3; 1/TRUE/yes = in set)")


@chart("venn", "Venn diagram", "Part-to-whole & hierarchical", [_SETS], [], aliases=("venn chart",))
def venn(fig, d, style):
    """Equal-size overlapping circles with the exact row count in every region (2 or 3 sets)."""
    ax = make_axes(fig, style)
    sets = [_member(c) for c in d["sets"]]
    names = d.label_list("sets")
    k = len(sets)
    if k not in (2, 3):
        return empty_message(fig, style, "Choose 2 or 3 membership columns")
    cnt = region_counts(sets)
    cols = style.series_colors(names)
    if k == 2:
        cs = [(-0.55, 0.0), (0.55, 0.0)]
        lab = {(1, 0): (-1.0, 0), (0, 1): (1.0, 0), (1, 1): (0, 0)}
    else:
        cs = [(-0.5, 0.29), (0.5, 0.29), (0, -0.58)]
        lab = {(1, 0, 0): (-0.85, 0.5), (0, 1, 0): (0.85, 0.5), (0, 0, 1): (0, -1.0), (1, 1, 0): (0, 0.62),
               (1, 0, 1): (-0.52, -0.3), (0, 1, 1): (0.52, -0.3), (1, 1, 1): (0, 0.02)}
    for (cx, cy), c, n in zip(cs, cols, names):
        ax.add_patch(Circle((cx, cy), 1.0, facecolor=c, alpha=0.35, edgecolor=c, lw=2, label=n))
    for key, (x, y) in lab.items():
        ax.text(x, y, f"{cnt.get(key, 0):,}", ha="center", va="center", fontsize=style.label_size, color=style.text_color)
    out = int((~np.column_stack(sets).any(axis=1)).sum())
    d.notes.append(f"{out:,} rows are in none of the sets")
    ax.set_xlim(-2.1, 2.1); ax.set_ylim(-2.0, 2.0); ax.set_aspect("equal"); ax.axis("off")
    finish(fig, ax, style, legend=True, axes_decor=False)
    return ax


def euler_layout(sizes: list[float], pair: dict[tuple, float]):
    """Circle radii (area = size) and centres for 2-3 sets; pair[(i,j)] = overlap area. Exact for 2 sets."""
    r = [math.sqrt(s / math.pi) for s in sizes]
    k = len(sizes)
    if k == 2:
        dist = L.distance_for_overlap(r[0], r[1], pair[(0, 1)])
        return r, [(0.0, 0.0), (dist, 0.0)]
    d01 = L.distance_for_overlap(r[0], r[1], pair[(0, 1)])
    d02 = L.distance_for_overlap(r[0], r[2], pair[(0, 2)])
    d12 = L.distance_for_overlap(r[1], r[2], pair[(1, 2)])
    x2 = (d01 ** 2 + d02 ** 2 - d12 ** 2) / (2 * d01) if d01 else 0.0
    y2 = math.sqrt(max(d02 ** 2 - x2 ** 2, 0.0))
    return r, [(0.0, 0.0), (d01, 0.0), (x2, -y2)]


@chart("euler", "Euler diagram", "Part-to-whole & hierarchical", [_SETS], [], aliases=("euler chart", "area-proportional venn"))
def euler(fig, d, style):
    """Circle areas proportional to set sizes and pairwise overlap areas proportional to the shared counts
    (exact for two sets; a pairwise-exact approximation for three)."""
    ax = make_axes(fig, style)
    sets = [_member(c) for c in d["sets"]]
    names = d.label_list("sets")
    k = len(sets)
    if k not in (2, 3):
        return empty_message(fig, style, "Choose 2 or 3 membership columns")
    sizes = [float(s.sum()) for s in sets]
    if min(sizes) == 0:
        return empty_message(fig, style, "A selected set is empty")
    pair = {(i, j): float((sets[i] & sets[j]).sum()) for i in range(k) for j in range(i + 1, k)}
    r, cs = euler_layout(sizes, pair)
    cols = style.series_colors(names)
    for (cx, cy), rr, c, n, s in zip(cs, r, cols, names, sizes):
        ax.add_patch(Circle((cx, cy), rr, facecolor=c, alpha=0.35, edgecolor=c, lw=2, label=f"{n} ({int(s):,})"))
        ax.text(cx, cy, f"{int(s):,}", ha="center", va="center", fontsize=style.label_size, color=style.text_color)
    xs = [c[0] + rr for c, rr in zip(cs, r)] + [c[0] - rr for c, rr in zip(cs, r)]
    ys = [c[1] + rr for c, rr in zip(cs, r)] + [c[1] - rr for c, rr in zip(cs, r)]
    pad = 0.08 * max(max(xs) - min(xs), max(ys) - min(ys))
    ax.set_xlim(min(xs) - pad, max(xs) + pad); ax.set_ylim(min(ys) - pad, max(ys) + pad); ax.set_aspect("equal"); ax.axis("off")
    d.notes.append("overlaps: " + "; ".join(f"{names[i]}∩{names[j]} = {int(v):,}" for (i, j), v in pair.items()))
    finish(fig, ax, style, legend=True, axes_decor=False)
    return ax


# ------------------------------------------------------------------ gauge / pyramid / funnel
@chart("circular_gauge", "Circular gauge", "Part-to-whole & hierarchical", [Role("y", "numeric", "Value")],
       [Param("agg", "Aggregate", "choice", "mean", choices=("mean", "sum", "median", "max", "min")),
        Param("vmin", "Scale minimum", "float", 0.0), Param("vmax", "Scale maximum", "float", 100.0)],
       aliases=("radial gauge", "gauge chart", "progress ring"))
def circular_gauge(fig, d, style, agg="mean", vmin=0.0, vmax=100.0):
    """One number (an aggregate of the column) shown as a filled ring between a scale minimum and maximum."""
    ax = make_axes(fig, style)
    v = numeric(d["y"]); v = v[np.isfinite(v)]
    if v.size == 0 or vmax == vmin:
        return empty_message(fig, style, "No numeric data / empty scale")
    val = float({"mean": np.mean, "sum": np.sum, "median": np.median, "max": np.max, "min": np.min}[agg](v))
    frac = float(np.clip((val - vmin) / (vmax - vmin), 0, 1))
    ax.add_patch(Wedge((0, 0), 1.0, -45 - 270 * 0 + 0, 225, width=0.22, facecolor=style.grid_color, alpha=0.6, lw=0))
    ax.add_patch(Wedge((0, 0), 1.0, 225 - 270 * frac, 225, width=0.22, facecolor=style.palette_colors()[0], alpha=style.alpha, lw=0))
    ax.text(0, 0.05, fmt(val, style), ha="center", va="center", fontsize=style.title_size * 2, color=style.text_color, fontweight="bold")
    ax.text(0, -0.28, f"{agg} of {d.name('y')}", ha="center", va="center", fontsize=style.label_size, color=style.text_color, alpha=0.8)
    ax.text(-0.72, -0.85, f"{vmin:g}", ha="center", fontsize=style.tick_size, color=style.text_color)
    ax.text(0.72, -0.85, f"{vmax:g}", ha="center", fontsize=style.tick_size, color=style.text_color)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal"); ax.axis("off")
    finish(fig, ax, style, legend=False, axes_decor=False)
    d.notes.append(f"{agg} = {val:.6g} ({frac * 100:.1f}% of the scale)")
    return ax


def _stages(d, agg):
    labels, vals = aggregate(d["x"], d.get("y") if d.has("y") else None, agg if d.has("y") else "count", "appearance")
    vals = np.nan_to_num(np.asarray(vals, float))
    return list(labels), vals


@chart("funnel", "Funnel chart", "Part-to-whole & hierarchical", [Role("x", "categorical", "Stage (in order of appearance)"), VAL],
       [Param("agg", "Aggregate", "choice", "sum", choices=SUM_AGGS), Param("sort", "Sort largest first", "bool", False)],
       aliases=("funnel graph", "conversion funnel"))
def funnel(fig, d, style, agg="sum", sort=False):
    """Centred bars whose width is the stage value; the % shown is relative to the first stage."""
    ax = make_axes(fig, style)
    labels, vals = _stages(d, agg)
    if sort:
        o = np.argsort(-vals, kind="stable"); labels, vals = [labels[i] for i in o], vals[o]
    if not len(vals) or vals.max() <= 0:
        return empty_message(fig, style, "No positive values")
    top = vals[0] if vals[0] > 0 else vals.max()
    cols = style.series_colors(labels)
    for i, (l, v) in enumerate(zip(labels, vals)):
        ax.barh(i, v, left=-v / 2, height=style.bar_width, color=cols[i], alpha=style.alpha)
        ax.text(0, i, f"{fmt(v, style)}  ({v / top * 100:.0f}%)", ha="center", va="center", color="white", fontsize=style.value_label_size + 1)
    ax.set_yticks(range(len(labels)), [short(l) for l in labels]); ax.invert_yaxis()
    ax.set_xticks([])
    finish(fig, ax, style, ylabel=d.name("x"), legend=False, grid_axis="x")
    ax.grid(False)
    return ax


@chart("pyramid", "Pyramid chart", "Part-to-whole & hierarchical", [Role("x", "categorical", "Level (top first)"), VAL],
       [Param("agg", "Aggregate", "choice", "sum", choices=SUM_AGGS)], aliases=("pyramid graph", "triangle chart"))
def pyramid(fig, d, style, agg="sum"):
    """A triangle cut into horizontal bands; each band's area is proportional to its value."""
    ax = make_axes(fig, style)
    labels, vals = _stages(d, agg)
    if not len(vals) or vals.sum() <= 0:
        return empty_message(fig, style, "No positive values")
    cum = np.r_[0, np.cumsum(vals)] / vals.sum()
    H = 1.0
    ycoord = lambda a: H * math.sqrt(a)                         # area measured from the apex grows as height^2
    cols = style.series_colors(labels)
    for i, l in enumerate(labels):
        y0, y1 = ycoord(cum[i]), ycoord(cum[i + 1])            # y measured from the apex (0) down to the base (1)
        w0, w1 = y0, y1
        ax.add_patch(Polygon([(-w0 / 2, y0), (w0 / 2, y0), (w1 / 2, y1), (-w1 / 2, y1)], closed=True, facecolor=cols[i], alpha=style.alpha,
                             edgecolor=style.axes_background, lw=1.2))
        ax.text(0, (y0 + y1) / 2, f"{short(l, 14)}\n{vals[i] / vals.sum() * 100:.1f}%", ha="center", va="center", color="white",
                fontsize=style.value_label_size)
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(1.02, -0.02); ax.set_aspect("equal"); ax.axis("off")
    finish(fig, ax, style, legend=False, axes_decor=False)
    return ax
