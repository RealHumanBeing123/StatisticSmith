"""Comparison charts (part A): bar, column, grouped, lollipop, bullet, dot, dumbbell, pictogram, icon, range, radial bar."""
from __future__ import annotations

import numpy as np

from .base import (AGGS, ChartData, ChartStyle, Param, Role, aggregate, chart, finish, fmt, make_axes, numeric, short)

_ORDER = ("appearance", "value_desc", "value_asc", "alpha")
_CAT_Y = [Role("x", "categorical", "Category"), Role("y", "numeric", "Value (blank = count rows)", required=False)]
_BAR_PARAMS = [Param("agg", "Aggregate", "choice", "sum", choices=AGGS),
               Param("sort", "Order", "choice", "appearance", choices=_ORDER),
               Param("max_bars", "Max bars shown", "int", 30, min=1, max=500)]


def _bars(fig, d: ChartData, style: ChartStyle, agg, sort, max_bars, horizontal: bool):
    ax = make_axes(fig, style)
    labels, vals = aggregate(d["x"], d.get("y") if d.has("y") else None, agg, sort)
    if len(labels) > max_bars:
        d.notes.append(f"showing first {max_bars} of {len(labels)} categories")
        labels, vals = labels[:max_bars], vals[:max_bars]
    pos = np.arange(len(labels))
    colors = [style.color_for(l, 0) if (style.primary_color or l in style.colors) else style.palette_colors()[0] for l in labels]
    if horizontal:
        bars = ax.barh(pos, vals, height=style.bar_width, color=colors, alpha=style.alpha)
        ax.set_yticks(pos, [short(l) for l in labels])
        ax.invert_yaxis()
    else:
        bars = ax.bar(pos, vals, width=style.bar_width, color=colors, alpha=style.alpha)
        ax.set_xticks(pos, [short(l) for l in labels])
    if style.show_values:
        ax.bar_label(bars, labels=[fmt(v, style) for v in vals], fontsize=style.value_label_size, color=style.text_color, padding=2)
    ycap = (agg if d.has("y") else "count").capitalize() + (f" of {d.name('y')}" if d.has("y") and agg != "count" else "")
    cat, val = d.name("x"), ycap
    finish(fig, ax, style, xlabel=val if horizontal else cat, ylabel=cat if horizontal else val,
           legend=False, grid_axis="x" if horizontal else "y")
    return ax


@chart("bar", "Bar chart (horizontal)", "Comparison", _CAT_Y, _BAR_PARAMS, aliases=("bar graph",))
def bar(fig, d, style, agg="sum", sort="appearance", max_bars=30):
    """Horizontal bars, one per category. Best for long labels."""
    return _bars(fig, d, style, agg, sort, max_bars, horizontal=True)


@chart("column", "Column chart (vertical)", "Comparison", _CAT_Y, _BAR_PARAMS, aliases=("column graph", "vertical bar chart"))
def column(fig, d, style, agg="sum", sort="appearance", max_bars=30):
    """Vertical bars, one per category. Best for short labels / natural order."""
    return _bars(fig, d, style, agg, sort, max_bars, horizontal=False)


# =========================================================================== part A, batch 2
# grouped bars, lollipop, bullet, dot plot, dumbbell, range plot, pictogram, icon chart, radial bar
from matplotlib.colors import to_rgb as _to_rgb  # noqa: E402
from matplotlib.patches import Polygon as _Polygon, Rectangle as _Rectangle  # noqa: E402

from .base import empty_message  # noqa: E402

_GAP_ORDER = ("appearance", "gap_desc", "gap_asc", "alpha")


# ----------------------------------------------------------------------------- data helpers
def _as_str(a):
    """Category column -> (str array, valid mask). None / NaN / '' are 'missing'."""
    arr = np.asarray(a)
    if arr.dtype.kind in "fc":
        return arr.astype(str), ~np.isnan(arr)
    s = arr.astype(object).astype(str)
    return s, ~np.isin(s, ["", "nan", "None", "NaN", "NaT", "<NA>"])


def _codes(s, valid):
    """Integer codes in order of first appearance (-1 for missing) + label list."""
    idx = np.flatnonzero(valid)
    codes = np.full(len(s), -1, dtype=np.int64)
    if idx.size == 0:
        return codes, []
    u, first, inv = np.unique(s[idx], return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(u), dtype=np.int64)
    rank[order] = np.arange(len(u))
    codes[idx] = rank[np.ravel(inv)]
    return codes, [str(x) for x in u[order]]


def _agg_table(cat, grp, val, agg="sum"):
    """Vectorised aggregate of ``val`` by category (rows) x group (columns).

    Returns (labels, groups, M[K, max(G,1)]). Empty cells are NaN (0 for counts). Appearance order.
    """
    cs, cv = _as_str(cat)
    cc, labels = _codes(cs, cv)
    if grp is not None:
        gs, gv = _as_str(grp)
        gc, groups = _codes(gs, gv)
    else:
        gc, groups = np.zeros(len(cs), dtype=np.int64), []
    G = max(len(groups), 1)
    K = len(labels)
    ok = (cc >= 0) & (gc >= 0)
    use_val = val is not None and agg != "count"
    v = None
    if use_val:
        v = numeric(val)
        ok &= ~np.isnan(v)
    size = K * G
    if size == 0:
        return labels, groups, np.zeros((K, G))
    cell = cc[ok] * G + gc[ok]
    cnt = np.bincount(cell, minlength=size).astype(float)
    if not use_val:
        return labels, groups, cnt.reshape(K, G)
    vv = v[ok]
    M = np.full(size, np.nan)
    if cell.size:
        if agg in ("sum", "mean"):
            s = np.bincount(cell, weights=vv, minlength=size)
            has = cnt > 0
            M[has] = (s[has] / cnt[has]) if agg == "mean" else s[has]
        elif agg in ("min", "max"):
            o = np.argsort(cell, kind="stable")
            sc, sv = cell[o], vv[o]
            starts = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
            M[sc[starts]] = (np.minimum if agg == "min" else np.maximum).reduceat(sv, starts)
        elif agg == "median":
            o = np.lexsort((vv, cell))
            sc, sv = cell[o], vv[o]
            starts = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
            cells = sc[starts]
            c = cnt[cells].astype(int)
            M[cells] = (sv[starts + (c - 1) // 2] + sv[starts + c // 2]) / 2.0
        else:
            raise ValueError(f"unknown aggregate {agg!r}")
    return labels, groups, M.reshape(K, G)


def _label_means(label, arrays):
    """Mean of each array per label (appearance order). Returns (labels, [vec, ...])."""
    labels, outs = [], []
    for a in arrays:
        labels, _, M = _agg_table(label, None, a, "mean")
        outs.append(M[:, 0])
    if not arrays:
        labels, _, _ = _agg_table(label, None, None, "count")
    return labels, outs


def _order_idx(labels, key, sort):
    """Index permutation for sort in appearance|value_desc|value_asc|gap_desc|gap_asc|alpha."""
    k = len(labels)
    if sort in ("value_desc", "gap_desc"):
        return np.argsort(-np.nan_to_num(np.asarray(key, float), nan=-np.inf), kind="stable")
    if sort in ("value_asc", "gap_asc"):
        return np.argsort(np.nan_to_num(np.asarray(key, float), nan=np.inf), kind="stable")
    if sort == "alpha":
        return np.argsort(np.array([str(x) for x in labels]), kind="stable")
    return np.arange(k)


def _row_total(M):
    M = np.atleast_2d(M)
    return np.where(np.all(np.isnan(M), axis=1), np.nan, np.nansum(M, axis=1))


def _cap(d, n, cap, what="categories"):
    cap = max(int(cap), 1)
    if n > cap:
        d.notes.append(f"showing first {cap} of {n} {what}")
        return cap
    return n


def _ycap(d, agg):
    has = d.has("y")
    return (agg if has else "count").capitalize() + (f" of {d.name('y')}" if has and agg != "count" else "")


def _cat_table(d, agg, sort, cap, with_group=False):
    """Aggregate + order + cap. Returns (labels, groups, M) or None if nothing to draw."""
    if not d.has("x"):
        return None
    val = d["y"] if d.has("y") else None
    grp = d["group"] if (with_group and d.has("group")) else None
    labels, groups, M = _agg_table(d["x"], grp, val, agg)
    if not labels:
        return None
    o = _order_idx(labels, _row_total(M), sort)
    n = _cap(d, len(labels), cap)
    o = o[:n]
    return [labels[i] for i in o], groups, M[o]


def _blend(fg, bg, a):
    f, b = np.array(_to_rgb(fg)), np.array(_to_rgb(bg))
    return tuple(b + (f - b) * a)


def _label_ticks(ax, pos, labels, style, horizontal=True):
    if horizontal:
        ax.set_yticks(pos, [short(l) for l in labels])
        ax.set_ylim(len(labels) - 0.5, -0.5)
    else:
        ax.set_xticks(pos, [short(l) for l in labels])
        ax.set_xlim(-0.5, len(labels) - 0.5)


# ----------------------------------------------------------------------------- grouped bars
_GROUP_ROLES = [Role("x", "categorical", "Category"), Role("group", "categorical", "Series (group)"),
                Role("y", "numeric", "Value (blank = count rows)", required=False)]


def _grouped(fig, d, style, agg, sort, max_bars, horizontal):
    tab = _cat_table(d, agg, sort, max_bars, with_group=True)
    if tab is None:
        return empty_message(fig, style, "No data to chart")
    labels, groups, M = tab
    ax = make_axes(fig, style)
    G = M.shape[1]
    pos = np.arange(len(labels))
    w = style.bar_width / G
    names = groups if groups else [""]
    colors = style.series_colors(groups) if groups else [style.color_for("", 0)]
    for j in range(G):
        off = (j - (G - 1) / 2) * w
        lab = names[j] if groups else None
        if horizontal:
            bars = ax.barh(pos + off, M[:, j], height=w, color=colors[j], alpha=style.alpha, label=lab)
        else:
            bars = ax.bar(pos + off, M[:, j], width=w, color=colors[j], alpha=style.alpha, label=lab)
        if style.show_values:
            ax.bar_label(bars, labels=["" if np.isnan(v) else fmt(v, style) for v in M[:, j]],
                         fontsize=style.value_label_size, color=style.text_color, padding=2)
    _label_ticks(ax, pos, labels, style, horizontal)
    cat, val = d.name("x"), _ycap(d, agg)
    if not style.legend_title and groups:
        style = _with(style, legend_title=d.name("group"))
    finish(fig, ax, style, xlabel=val if horizontal else cat, ylabel=cat if horizontal else val,
           legend=bool(groups), grid_axis="x" if horizontal else "y")
    return ax


def _with(style, **kw):
    import dataclasses
    return dataclasses.replace(style, **kw)


@chart("grouped_column", "Grouped column chart", "Comparison", _GROUP_ROLES, _BAR_PARAMS,
       aliases=("clustered column", "clustered column chart"))
def grouped_column(fig, d, style, agg="sum", sort="appearance", max_bars=30):
    """Clustered vertical bars: one cluster per category, one bar per series."""
    return _grouped(fig, d, style, agg, sort, max_bars, horizontal=False)


@chart("grouped_bar", "Grouped bar chart", "Comparison", _GROUP_ROLES, _BAR_PARAMS,
       aliases=("clustered bar", "clustered bar chart"))
def grouped_bar(fig, d, style, agg="sum", sort="appearance", max_bars=30):
    """Clustered horizontal bars: one cluster per category, one bar per series."""
    return _grouped(fig, d, style, agg, sort, max_bars, horizontal=True)


# ----------------------------------------------------------------------------- lollipop
@chart("lollipop", "Lollipop chart", "Comparison", _CAT_Y,
       [Param("agg", "Aggregate", "choice", "sum", choices=AGGS),
        Param("sort", "Order", "choice", "appearance", choices=_ORDER),
        Param("orientation", "Orientation", "choice", "horizontal", choices=("horizontal", "vertical")),
        Param("max_items", "Max items shown", "int", 30, min=1, max=500)],
       aliases=("lollipop graph",))
def lollipop(fig, d, style, agg="sum", sort="appearance", orientation="horizontal", max_items=30):
    """A thin stem and a dot per category - a lighter alternative to bars."""
    tab = _cat_table(d, agg, sort, max_items)
    if tab is None:
        return empty_message(fig, style, "No data to chart")
    labels, _, M = tab
    vals = M[:, 0]
    ax = make_axes(fig, style)
    pos = np.arange(len(labels))
    color = style.color_for("", 0)
    horiz = orientation != "vertical"
    if horiz:
        ax.hlines(pos, 0, vals, colors=color, linewidth=style.line_width, alpha=style.alpha)
        ax.plot(vals, pos, ls="none", marker="o", ms=style.marker_size * 1.6, color=color, alpha=style.alpha)
    else:
        ax.vlines(pos, 0, vals, colors=color, linewidth=style.line_width, alpha=style.alpha)
        ax.plot(pos, vals, ls="none", marker="o", ms=style.marker_size * 1.6, color=color, alpha=style.alpha)
    _label_ticks(ax, pos, labels, style, horiz)
    if style.show_values:
        for p, v in zip(pos, vals):
            if np.isfinite(v):
                ax.annotate(fmt(v, style), (v, p) if horiz else (p, v), xytext=(7, 0) if horiz else (0, 7),
                            textcoords="offset points", fontsize=style.value_label_size, color=style.text_color,
                            ha="left" if horiz else "center", va="center" if horiz else "bottom")
    cat, val = d.name("x"), _ycap(d, agg)
    finish(fig, ax, style, xlabel=val if horiz else cat, ylabel=cat if horiz else val, legend=False,
           grid_axis="x" if horiz else "y")
    return ax


# ----------------------------------------------------------------------------- bullet
@chart("bullet", "Bullet graph", "Comparison",
       [Role("label", "categorical", "Label"), Role("actual", "numeric", "Actual value"),
        Role("target", "numeric", "Target", required=False),
        Role("ranges", "numeric_multi", "Range upper bounds (1-3)", required=False,
             help="Up to three upper bounds, drawn as graded background bands (darkest = lowest).")],
       [Param("max_items", "Max bullets shown", "int", 20, min=1, max=200)],
       aliases=("bullet chart",))
def bullet(fig, d, style, max_items=20):
    """One bullet per label: actual (narrow bar) vs target (tick) over graded qualitative ranges."""
    if not d.has("label") or not d.has("actual"):
        return empty_message(fig, style, "No data to chart")
    rng_cols = [numeric(c) for c in (d.get("ranges") or [])][:3] if d.has("ranges") else []
    arrays = [d["actual"]] + ([d["target"]] if d.has("target") else []) + rng_cols
    labels, vecs = _label_means(d["label"], arrays)
    if not labels:
        return empty_message(fig, style, "No data to chart")
    n = _cap(d, len(labels), max_items, "bullets")
    labels = labels[:n]
    vecs = [v[:n] for v in vecs]
    actual = vecs[0]
    target = vecs[1] if d.has("target") else None
    R = np.column_stack(vecs[(2 if d.has("target") else 1):]) if rng_cols else np.zeros((n, 0))
    if R.shape[1]:
        R = np.sort(R, axis=1)  # ascending per row, NaN last
    ax = make_axes(fig, style)
    pos = np.arange(n)
    m = R.shape[1]
    if m:
        fracs = np.linspace(0.40, 0.12, m) if m > 1 else [0.25]
        for j in range(m - 1, -1, -1):  # widest first so narrower bands paint on top
            w = np.where(np.isfinite(R[:, j]), R[:, j], 0.0)
            bars = ax.barh(pos, w, height=0.8, color=_blend(style.text_color, style.axes_background, fracs[j]),
                           zorder=1)
            for p in bars:
                p.set_gid(f"band{j}")
    bars = ax.barh(pos, actual, height=0.28, color=style.color_for("actual", 0), alpha=1.0, zorder=3,
                   label=d.name("actual", "Actual") or "Actual")
    for p in bars:
        p.set_gid("actual")
    if target is not None:
        for i, t in enumerate(target):
            if np.isfinite(t):
                ln, = ax.plot([t, t], [i - 0.3, i + 0.3], color=style.text_color, lw=max(style.line_width * 1.5, 3),
                              solid_capstyle="butt", zorder=4, gid="target",
                              label=(d.name("target", "Target") or "Target") if i == int(np.argmax(np.isfinite(target))) else None)
    if style.show_values:
        for i, v in enumerate(actual):
            if np.isfinite(v):
                ax.annotate(fmt(v, style), (v, i), xytext=(4, -10), textcoords="offset points",
                            fontsize=style.value_label_size, color=style.text_color)
    _label_ticks(ax, pos, labels, style, True)
    finish(fig, ax, style, xlabel=d.name("actual"), ylabel=d.name("label"), grid_axis="x")
    return ax


# ----------------------------------------------------------------------------- dot plot
@chart("dot_plot", "Dot plot (Cleveland)", "Comparison",
       [Role("x", "categorical", "Category"), Role("y", "numeric", "Value"),
        Role("group", "categorical", "Group (colour)", required=False)],
       [Param("agg", "Aggregate", "choice", "mean", choices=AGGS),
        Param("sort", "Order", "choice", "value_desc", choices=_ORDER),
        Param("max_items", "Max items shown", "int", 30, min=1, max=500)],
       aliases=("cleveland dot plot", "dot chart"))
def dot_plot(fig, d, style, agg="mean", sort="value_desc", max_items=30):
    """One dot per category (or category x group) on a horizontal guide line."""
    if not d.has("x") or not d.has("y"):
        return empty_message(fig, style, "No data to chart")
    tab = _cat_table(d, agg, sort, max_items, with_group=True)
    if tab is None:
        return empty_message(fig, style, "No data to chart")
    labels, groups, M = tab
    ax = make_axes(fig, style)
    pos = np.arange(len(labels))
    fin = M[np.isfinite(M)]
    if fin.size:
        ax.hlines(pos, min(fin.min(), 0.0), max(fin.max(), 0.0), colors=style.grid_color, linewidth=1.0, zorder=1)
    colors = style.series_colors(groups) if groups else [style.color_for("", 0)]
    for j in range(M.shape[1]):
        ax.plot(M[:, j], pos, ls="none", marker="o", ms=style.marker_size * 1.5, color=colors[j], alpha=style.alpha,
                label=groups[j] if groups else None, zorder=3)
    _label_ticks(ax, pos, labels, style, True)
    if groups and not style.legend_title:
        style = _with(style, legend_title=d.name("group"))
    finish(fig, ax, style, xlabel=_ycap(d, agg), ylabel=d.name("x"), legend=bool(groups), grid_axis="x")
    return ax


# ----------------------------------------------------------------------------- dumbbell / range
def _pair_table(d, a_role, b_role, sort, cap):
    if not (d.has("label") and d.has(a_role) and d.has(b_role)):
        return None
    labels, (a, b) = _label_means(d["label"], [d[a_role], d[b_role]])
    if not labels:
        return None
    gap = np.abs(b - a)
    o = _order_idx(labels, gap, sort)
    o = o[:_cap(d, len(labels), cap)]
    return [labels[i] for i in o], a[o], b[o]


@chart("dumbbell", "Dumbbell chart", "Comparison",
       [Role("label", "categorical", "Label"), Role("start", "numeric", "Start value"),
        Role("end", "numeric", "End value")],
       [Param("sort", "Order", "choice", "appearance", choices=_GAP_ORDER),
        Param("max_items", "Max items shown", "int", 30, min=1, max=500)],
       aliases=("dumbbell plot", "connected dot plot"))
def dumbbell(fig, d, style, sort="appearance", max_items=30):
    """Two dots joined by a connector per label: shows the change between two values."""
    tab = _pair_table(d, "start", "end", sort, max_items)
    if tab is None:
        return empty_message(fig, style, "No data to chart")
    labels, a, b = tab
    ax = make_axes(fig, style)
    pos = np.arange(len(labels))
    sn, en = d.name("start", "Start") or "Start", d.name("end", "End") or "End"
    ca, cb = style.series_colors([sn, en])
    ax.hlines(pos, a, b, colors=style.grid_color if style.grid_color != style.axes_background else style.text_color,
             linewidth=max(style.line_width * 1.5, 3), zorder=1)
    ax.plot(a, pos, ls="none", marker="o", ms=style.marker_size * 1.5, color=ca, label=sn, zorder=3)
    ax.plot(b, pos, ls="none", marker="o", ms=style.marker_size * 1.5, color=cb, label=en, zorder=3)
    _label_ticks(ax, pos, labels, style, True)
    finish(fig, ax, style, xlabel="Value", ylabel=d.name("label"), grid_axis="x")
    return ax


@chart("range_plot", "Range plot", "Comparison",
       [Role("label", "categorical", "Label"), Role("low", "numeric", "Low value"), Role("high", "numeric", "High value")],
       [Param("sort", "Order", "choice", "appearance", choices=_GAP_ORDER),
        Param("max_items", "Max items shown", "int", 30, min=1, max=500),
        Param("show_arrow", "Arrowhead low -> high", "bool", False)],
       aliases=("floating bar", "floating bar chart", "range bar"))
def range_plot(fig, d, style, sort="appearance", max_items=30, show_arrow=False):
    """A floating bar from low to high per category."""
    tab = _pair_table(d, "low", "high", sort, max_items)
    if tab is None:
        return empty_message(fig, style, "No data to chart")
    labels, lo, hi = tab
    ax = make_axes(fig, style)
    pos = np.arange(len(labels))
    ax.barh(pos, hi - lo, left=lo, height=style.bar_width * 0.5, color=style.color_for("", 0), alpha=style.alpha, zorder=2)
    if show_arrow:
        for i in range(len(labels)):
            if np.isfinite(lo[i]) and np.isfinite(hi[i]) and lo[i] != hi[i]:
                ax.annotate("", xy=(hi[i], i), xytext=(lo[i], i), zorder=4,
                            arrowprops=dict(arrowstyle="-|>", color=style.text_color, lw=max(style.line_width * 0.8, 1.2),
                                            shrinkA=0, shrinkB=0, mutation_scale=12))
    if style.show_values:
        for i in range(len(labels)):
            if np.isfinite(lo[i]) and np.isfinite(hi[i]):
                ax.annotate(f"{fmt(lo[i], style)} - {fmt(hi[i], style)}", (max(lo[i], hi[i]), i), xytext=(5, 0),
                            textcoords="offset points", fontsize=style.value_label_size, color=style.text_color, va="center")
    _label_ticks(ax, pos, labels, style, True)
    finish(fig, ax, style, xlabel="Value", ylabel=d.name("label"), legend=False, grid_axis="x")
    return ax


# ----------------------------------------------------------------------------- icons
def _circle_poly(cx, cy, r, n=72):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def _unit_icon(kind):
    """Polygons (list of Nx2) of an icon inside the unit box [-.5,.5]^2 (non-overlapping parts)."""
    if kind == "square":
        return [np.array([[-.45, -.45], [.45, -.45], [.45, .45], [-.45, .45]])]
    if kind == "triangle":
        return [np.array([[-.5, -.43], [.5, -.43], [0, .43]])]
    if kind == "star":
        t = np.pi / 2 + np.arange(10) * np.pi / 5
        r = np.where(np.arange(10) % 2 == 0, .5, .2)
        return [np.column_stack([r * np.cos(t), r * np.sin(t) - .04])]
    if kind == "heart":
        t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        h = y.max() - y.min()
        return [np.column_stack([x / h, (y - (y.max() + y.min()) / 2) / h])]
    if kind == "person":
        head = _circle_poly(0, .31, .17)
        body = np.array([[-.3, -.5], [-.3, 0.0], [-.18, .1], [.18, .1], [.3, 0.0], [.3, -.5]])
        return [head, body]
    return [_circle_poly(0, 0, .5)]  # circle


def _icon_polys(kind, cx, cy, s):
    return [p * s + np.array([cx, cy]) for p in _unit_icon(kind)]


def _poly_area(xy):
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _nice_unit(vmax, max_icons=20):
    if not np.isfinite(vmax) or vmax <= 0:
        return 1.0
    raw = vmax / max_icons
    e = np.floor(np.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        u = m * 10.0 ** e
        if u >= raw * (1 - 1e-12):
            return float(u)
    return float(10.0 ** (e + 1))


_PICTO_ICONS = ("person", "circle", "square", "star", "heart", "triangle")


@chart("pictogram", "Pictogram", "Comparison", _CAT_Y,
       [Param("agg", "Aggregate", "choice", "sum", choices=AGGS),
        Param("unit_value", "Value per icon (0 = auto)", "float", 0.0, min=0),
        Param("icon", "Icon", "choice", "person", choices=_PICTO_ICONS),
        Param("sort", "Order", "choice", "appearance", choices=_ORDER),
        Param("max_items", "Max items shown", "int", 15, min=1, max=100)],
       aliases=("isotype", "unit chart", "icon array"))
def pictogram(fig, d, style, agg="sum", unit_value=0.0, icon="person", sort="appearance", max_items=15):
    """Repeated icons per category; each icon stands for ``unit_value`` (last icon clipped for fractions)."""
    tab = _cat_table(d, agg, sort, max_items)
    if tab is None:
        return empty_message(fig, style, "No data to chart")
    labels, _, M = tab
    vals = M[:, 0]
    pos_vals = vals[np.isfinite(vals) & (vals > 0)]
    unit = float(unit_value) if unit_value and unit_value > 0 else _nice_unit(pos_vals.max() if pos_vals.size else 0.0)
    n_icons = np.where(np.isfinite(vals) & (vals > 0), vals / unit, 0.0)
    if n_icons.max(initial=0) > 100:
        d.notes.append("icon count capped at 100 per row; increase the value per icon")
        n_icons = np.minimum(n_icons, 100.0)
    ax = make_axes(fig, style)
    s = 0.8
    for i, n in enumerate(n_icons):
        full = int(np.floor(n + 1e-9))
        frac = float(n - full)
        if frac < 1e-9:
            frac = 0.0
        color = style.color_for(labels[i], i)
        for k in range(full + (1 if frac > 0 else 0)):
            part = frac if (k == full) else 1.0
            clip = None
            if part < 1.0:
                clip = _Rectangle((k - s / 2, i - 0.5), part * s, 1.0, transform=ax.transData, visible=False)
            for poly in _icon_polys(icon, k, i, s):
                p = _Polygon(poly, closed=True, fc=color, ec="none", alpha=style.alpha, gid=f"icon:{i}:{k}")
                ax.add_patch(p)
                if clip is not None:
                    p.set_clip_path(clip)
        if style.show_values and np.isfinite(vals[i]):
            ax.text(np.ceil(n) + 0.1, i, fmt(vals[i], style), va="center", fontsize=style.value_label_size,
                    color=style.text_color)
    mx = float(np.ceil(n_icons.max(initial=0)))
    ax.plot([], [], ls="none", marker="s", color=style.color_for("", 0), label=f"1 icon = {unit:,.6g}")
    ax.set_xlim(-0.6, max(mx, 1) - 0.4 + (1.6 if style.show_values else 0.2))
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(labels)), [short(l) for l in labels])
    ax.set_xticks([])
    ax.set_aspect("equal", adjustable="box")
    finish(fig, ax, style, ylabel=d.name("x"), xlabel=_ycap(d, agg), legend=True, grid_axis="y")
    ax.grid(False)
    return ax


@chart("icon_chart", "Proportional icon chart", "Comparison", _CAT_Y,
       [Param("agg", "Aggregate", "choice", "sum", choices=AGGS),
        Param("icon", "Icon", "choice", "circle", choices=("circle", "square", "person", "star")),
        Param("sort", "Order", "choice", "appearance", choices=_ORDER),
        Param("max_items", "Max items shown", "int", 16, min=1, max=100)],
       aliases=("proportional area chart", "proportional symbol chart"))
def icon_chart(fig, d, style, agg="sum", icon="circle", sort="appearance", max_items=16):
    """One icon per category whose AREA is proportional to its value."""
    tab = _cat_table(d, agg, sort, max_items)
    if tab is None:
        return empty_message(fig, style, "No data to chart")
    labels, _, M = tab
    vals = M[:, 0]
    good = np.isfinite(vals) & (vals > 0)
    vmax = vals[good].max() if good.any() else 0.0
    ax = make_axes(fig, style)
    n = len(labels)
    cols = min(n, 8)
    smax, dx, dy = 1.0, 1.35, 1.9
    for i in range(n):
        cx, cy = (i % cols) * dx, -(i // cols) * dy
        if good[i]:
            side = smax * np.sqrt(vals[i] / vmax)  # area ~ side^2
            for poly in _icon_polys(icon, cx, cy, side):
                ax.add_patch(_Polygon(poly, closed=True, fc=style.color_for(labels[i], i), ec="none",
                                      alpha=style.alpha, gid=f"icon:{i}"))
        ax.text(cx, cy - 0.62, short(labels[i], 14), ha="center", va="top", fontsize=style.tick_size,
                color=style.text_color)
        if style.show_values and np.isfinite(vals[i]):
            ax.text(cx, cy + 0.6, fmt(vals[i], style), ha="center", va="bottom", fontsize=style.value_label_size,
                    color=style.text_color)
    rows = (n - 1) // cols + 1
    ax.set_xlim(-0.8, (cols - 1) * dx + 0.8)
    ax.set_ylim(-(rows - 1) * dy - 1.1, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.text(0.5, -0.02, f"Icon area is proportional to {_ycap(d, agg).lower()}", transform=ax.transAxes, ha="center",
            va="top", fontsize=style.tick_size, color=style.text_color, alpha=0.75)
    finish(fig, ax, style, legend=False, axes_decor=False)
    return ax


# ----------------------------------------------------------------------------- radial bar
@chart("radial_bar", "Radial bar chart", "Comparison", _CAT_Y,
       [Param("agg", "Aggregate", "choice", "sum", choices=AGGS),
        Param("sort", "Order", "choice", "value_desc", choices=_ORDER),
        Param("max_angle", "Angle of the largest bar (deg)", "float", 270.0, min=10, max=360),
        Param("max_items", "Max rings shown", "int", 12, min=1, max=100)],
       aliases=("circular bar chart", "radial bar graph"))
def radial_bar(fig, d, style, agg="sum", sort="value_desc", max_angle=270.0, max_items=12):
    """Concentric ring segments; angular extent is proportional to value (largest = ``max_angle``)."""
    tab = _cat_table(d, agg, sort, max_items)
    if tab is None:
        return empty_message(fig, style, "No data to chart")
    labels, _, M = tab
    vals = M[:, 0]
    good = np.isfinite(vals) & (vals > 0)
    if not good.any():
        return empty_message(fig, style, "No positive values to chart")
    max_angle = float(min(max(max_angle, 1.0), 360.0))
    ax = make_axes(fig, style, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    n = len(labels)
    hole, step = 1.5, 1.0
    h = step * min(max(style.bar_width, 0.1), 1.0)
    vmax = vals[good].max()
    ext = np.where(good, vals / vmax, 0.0) * np.deg2rad(max_angle)
    for k in range(n):  # first category = outermost ring
        r0 = hole + (n - 1 - k) * step
        t = ax.bar(0, h, width=np.deg2rad(max_angle), bottom=r0, align="edge", color=style.grid_color, alpha=0.35,
                   linewidth=0)
        t[0].set_gid("track")
        b = ax.bar(0, h, width=ext[k], bottom=r0, align="edge", color=style.color_for(labels[k], k),
                   alpha=style.alpha, linewidth=0)
        b[0].set_gid("ring")
        ax.text(0, r0 + h / 2, short(labels[k], 16) + " ", ha="right", va="center", fontsize=style.tick_size,
                color=style.text_color, clip_on=False)
        if style.show_values and np.isfinite(vals[k]):
            ax.text(ext[k], r0 + h / 2, " " + fmt(vals[k], style), ha="left", va="center",
                    fontsize=style.value_label_size, color=style.text_color)
    ax.set_ylim(0, hole + n * step)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    finish(fig, ax, style, polar=True, legend=False)
    return ax
