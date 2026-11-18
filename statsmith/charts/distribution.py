"""Distribution charts: histogram, density, ridgeline, horizon, radial histogram, strip, jitter,
1-D heatmap, beeswarm, box plot, violin plot."""
from __future__ import annotations

import numpy as np

from .base import (ChartData, ChartStyle, Param, Role, chart, empty_message, finish, first_seen, make_axes, numeric, short)

_X = Role("x", "numeric", "Variable")
_G = Role("group", "categorical", "Group", required=False)
MAX_KDE = 20000


# ------------------------------------------------------------------ helpers
def groups(d: ChartData, xrole="x", grole="group", max_groups=40):
    """-> list of (label, finite float array). One group named after the variable when no grouping."""
    x = numeric(d[xrole])
    if not d.has(grole):
        return [(d.name(xrole), x[np.isfinite(x)])]
    g = np.asarray(d[grole], dtype=object).astype(str)
    out = []
    for lab in first_seen(g.tolist())[:max_groups]:
        v = x[g == lab]
        out.append((lab, v[np.isfinite(v)]))
    return out


def bin_edges(values: np.ndarray, bins, lo=None, hi=None):
    """Bin edges for ``bins`` = int | 'auto' | 'sturges' | 'fd' | 'scott' | 'sqrt' (numpy rules)."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([0.0, 1.0])
    rng = (values.min() if lo is None else lo, values.max() if hi is None else hi)
    if rng[0] == rng[1]:
        rng = (rng[0] - 0.5, rng[1] + 0.5)
    b = int(bins) if str(bins).isdigit() else bins
    return np.histogram_bin_edges(values, bins=b, range=rng)


def kde(values: np.ndarray, grid: np.ndarray, bw="scott"):
    """Gaussian KDE evaluated on ``grid`` (random subsample above MAX_KDE for speed, seeded)."""
    from scipy.stats import gaussian_kde
    v = values[np.isfinite(values)]
    if v.size > MAX_KDE:
        v = np.random.default_rng(1).choice(v, MAX_KDE, replace=False)
    if v.size < 2 or np.ptp(v) == 0:
        return np.zeros_like(grid)
    return gaussian_kde(v, bw_method=bw)(grid)


def tukey_stats(v: np.ndarray, whis: float = 1.5):
    """Box statistics exactly as defined by Tukey: quartiles by linear interpolation (R type 7),
    whiskers end at the most extreme data point within whis*IQR of the box."""
    v = np.sort(v[np.isfinite(v)])
    if v.size == 0:
        return None
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo_f, hi_f = q1 - whis * iqr, q3 + whis * iqr
    inside = v[(v >= lo_f) & (v <= hi_f)]
    return dict(med=med, q1=q1, q3=q3, whislo=inside.min(), whishi=inside.max(),
                fliers=v[(v < lo_f) | (v > hi_f)], mean=float(v.mean()), n=int(v.size))


def _colors(style, labels):
    return style.series_colors(labels)


# ------------------------------------------------------------------ histogram
@chart("histogram", "Histogram", "Distribution", [_X, _G],
       [Param("bins", "Bins (number or rule)", "text", "auto"),
        Param("stat", "Y axis", "choice", "count", choices=("count", "percent", "density", "frequency")),
        Param("overlay", "Groups", "choice", "overlay", choices=("overlay", "stacked", "side-by-side")),
        Param("normal_curve", "Overlay normal curve", "bool", False)],
       aliases=("histogram chart",))
def histogram(fig, d, style, bins="auto", stat="count", overlay="overlay", normal_curve=False):
    """Counts (or density) of a numeric variable in equal-width bins; groups share the same bin edges."""
    ax = make_axes(fig, style)
    gs = [(l, v) for l, v in groups(d) if v.size]
    if not gs:
        return empty_message(fig, style, "No numeric data to plot")
    allv = np.concatenate([v for _, v in gs])
    edges = bin_edges(allv, bins)
    width = np.diff(edges)
    cols = _colors(style, [l for l, _ in gs])
    total = allv.size
    bottoms = np.zeros(len(edges) - 1)
    k = len(gs)
    for i, (lab, v) in enumerate(gs):
        cnt = np.histogram(v, bins=edges)[0].astype(float)
        if stat == "percent":
            h = cnt / v.size * 100
        elif stat == "density":
            h = cnt / (v.size * width)
        elif stat == "frequency":
            h = cnt / width
        else:
            h = cnt
        kw = dict(color=cols[i], alpha=style.alpha if overlay != "overlay" or k == 1 else min(style.alpha, 0.6),
                  edgecolor=style.axes_background, linewidth=0.6, label=lab if k > 1 else None, align="edge")
        if overlay == "stacked" and k > 1:
            ax.bar(edges[:-1], h, width=width, bottom=bottoms, **kw)
            bottoms += h
        elif overlay == "side-by-side" and k > 1:
            ax.bar(edges[:-1] + width * i / k, h, width=width / k, **kw)
        else:
            ax.bar(edges[:-1], h, width=width, **kw)
    if normal_curve:
        from scipy.stats import norm
        mu, sd = allv.mean(), allv.std(ddof=1) if allv.size > 1 else 0
        if sd > 0:
            xs = np.linspace(edges[0], edges[-1], 300)
            scale = {"density": 1.0, "count": total * width.mean(), "percent": 100 * width.mean(),
                     "frequency": total}[stat]
            ax.plot(xs, norm.pdf(xs, mu, sd) * scale, color=style.text_color, lw=style.line_width, label="Normal fit")
    ylab = {"count": "Count", "percent": "Percent", "density": "Density", "frequency": "Frequency density"}[stat]
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=ylab, legend=k > 1 or normal_curve)
    return ax


# ------------------------------------------------------------------ density
@chart("density", "Density plot", "Distribution", [_X, _G],
       [Param("fill", "Fill", "bool", True), Param("bw", "Bandwidth rule", "choice", "scott", choices=("scott", "silverman"))],
       aliases=("density chart", "kernel density plot", "kde"))
def density(fig, d, style, fill=True, bw="scott"):
    """Gaussian kernel density estimate of each group."""
    ax = make_axes(fig, style)
    gs = [(l, v) for l, v in groups(d) if v.size > 1]
    if not gs:
        return empty_message(fig, style, "Need at least 2 numeric values")
    allv = np.concatenate([v for _, v in gs])
    pad = 0.05 * (np.ptp(allv) or 1)
    grid = np.linspace(allv.min() - 3 * pad, allv.max() + 3 * pad, 400)
    for (lab, v), c in zip(gs, _colors(style, [l for l, _ in gs])):
        y = kde(v, grid, bw)
        ax.plot(grid, y, color=c, lw=style.line_width, label=lab if len(gs) > 1 else None)
        if fill:
            ax.fill_between(grid, y, color=c, alpha=0.25 * style.alpha + 0.1)
    finish(fig, ax, style, xlabel=d.name("x"), ylabel="Density", legend=len(gs) > 1)
    return ax


# ------------------------------------------------------------------ ridgeline
@chart("ridgeline", "Ridgeline plot", "Distribution", [_X, Role("group", "categorical", "Group")],
       [Param("overlap", "Overlap", "float", 0.6, min=0, max=3), Param("max_groups", "Max groups", "int", 25, min=2, max=100)],
       aliases=("joyplot", "ridgeline chart"))
def ridgeline(fig, d, style, overlap=0.6, max_groups=25):
    """Stacked, partially overlapping density curves - one row per group."""
    ax = make_axes(fig, style)
    gs = [(l, v) for l, v in groups(d, max_groups=max_groups) if v.size > 1]
    if not gs:
        return empty_message(fig, style, "Need a numeric variable and a grouping variable")
    allv = np.concatenate([v for _, v in gs])
    pad = 0.05 * (np.ptp(allv) or 1)
    grid = np.linspace(allv.min() - 3 * pad, allv.max() + 3 * pad, 300)
    dens = [kde(v, grid) for _, v in gs]
    peak = max(x.max() for x in dens) or 1
    cols = _colors(style, [l for l, _ in gs])
    for i, ((lab, _), y) in enumerate(zip(gs, dens)):
        off = (len(gs) - 1 - i)
        yy = off + y / peak * (1 + overlap)
        ax.fill_between(grid, off, yy, color=cols[i], alpha=style.alpha, zorder=i + 2, lw=0)
        ax.plot(grid, yy, color=style.axes_background, lw=1.2, zorder=i + 2.5)
    ax.set_yticks(range(len(gs)), [short(l) for l, _ in reversed(gs)])
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=d.name("group"), legend=False, grid_axis="x")
    return ax


# ------------------------------------------------------------------ horizon
@chart("horizon", "Horizon chart", "Distribution",
       [Role("y", "numeric", "Value"), Role("order", "any", "Order by (blank = row order)", required=False), _G],
       [Param("bands", "Bands", "int", 3, min=1, max=6), Param("max_points", "Max points", "int", 4000, min=50, max=100000)],
       aliases=("horizon graph",))
def horizon(fig, d, style, bands=3, max_points=4000):
    """Value above/below the median folded into ``bands`` layered colour bands; one strip per group."""
    ax = make_axes(fig, style)
    y = numeric(d["y"])
    order = np.arange(len(y)) if not d.has("order") else np.argsort(np.asarray(d["order"]), kind="stable")
    labs = [("", np.arange(len(y)))] if not d.has("group") else None
    if labs is None:
        g = np.asarray(d["group"], dtype=object).astype(str)
        labs = [(l, np.flatnonzero(g == l)) for l in first_seen(g.tolist())[:12]]
    n = len(labs)
    ok = np.isfinite(y)
    med = np.median(y[ok]) if ok.any() else 0
    amp = np.max(np.abs(y[ok] - med)) if ok.any() else 1
    amp = amp or 1
    step = amp / bands
    pos, neg = style.palette_colors()[0], style.palette_colors()[4]
    for i, (lab, idx) in enumerate(labs):
        idx = idx[np.isin(idx, order)]
        idx = idx[np.argsort(order[idx] if False else idx)]
        sub = y[idx]
        if sub.size > max_points:
            sel = np.linspace(0, sub.size - 1, max_points).astype(int)
            sub = sub[sel]
        x = np.arange(sub.size)
        base = n - 1 - i
        dev = np.nan_to_num(sub - med, nan=0.0)
        for b in range(bands):
            a = 0.25 + 0.75 * (b + 1) / bands
            up = np.clip(dev - b * step, 0, step) / step
            dn = np.clip(-dev - b * step, 0, step) / step
            ax.fill_between(x, base, base + up * 0.9, color=pos, alpha=a * 0.8 if b else a * 0.5, lw=0)
            ax.fill_between(x, base, base + dn * 0.9, color=neg, alpha=a * 0.8 if b else a * 0.5, lw=0)
    if d.has("group"):
        ax.set_yticks([n - 1 - i + 0.45 for i in range(n)], [short(l) for l, _ in labs])
    else:
        ax.set_yticks([])
    finish(fig, ax, style, xlabel="Observation", ylabel=d.name("y"), legend=False)
    d.notes.append(f"Folded around the median ({med:.4g}); each band = {step:.4g}; blue = above, red = below")
    return ax


# ------------------------------------------------------------------ radial histogram
@chart("radial_histogram", "Radial histogram", "Distribution",
       [Role("x", "numeric", "Variable (e.g. hour, angle)")],
       [Param("bins", "Bins", "int", 24, min=3, max=360), Param("period", "Full circle = (blank/0 = data range)", "float", 0.0, min=0)],
       aliases=("circular histogram", "rose diagram", "polar histogram"))
def radial_histogram(fig, d, style, bins=24, period=0.0):
    """Histogram wrapped around a circle (hour of day, compass direction ...)."""
    ax = make_axes(fig, style, polar=True)
    v = numeric(d["x"])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return empty_message(fig, style, "No numeric data")
    lo = v.min()
    per = period or (np.ptp(v) or 1.0)
    ang = ((v - lo) / per % 1.0) * 2 * np.pi
    cnt, edges = np.histogram(ang, bins=bins, range=(0, 2 * np.pi))
    ax.bar(edges[:-1], cnt, width=2 * np.pi / bins, align="edge", color=style.palette_colors()[0], alpha=style.alpha,
           edgecolor=style.axes_background)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.tick_params(colors=style.text_color, labelsize=style.tick_size)
    ax.set_facecolor(style.axes_background)
    ax.grid(style.show_grid, color=style.grid_color)
    finish(fig, ax, style, title=d.name("x"), polar=True, legend=False)
    return ax


# ------------------------------------------------------------------ strip / jitter / beeswarm
def _cat_positions(d):
    gs = groups(d)
    return [(l, v) for l, v in gs if v.size]


@chart("strip", "Strip plot", "Distribution", [_X, _G], [Param("max_points", "Max points per group", "int", 5000, min=100, max=200000)],
       aliases=("strip chart", "dot strip plot"))
def strip(fig, d, style, max_points=5000):
    """Every observation as a dot on a single line per group (overlaps are not separated)."""
    ax = make_axes(fig, style)
    gs = _cat_positions(d)
    if not gs:
        return empty_message(fig, style, "No numeric data")
    rng = np.random.default_rng(1)
    cols = _colors(style, [l for l, _ in gs])
    for i, (lab, v) in enumerate(gs):
        if v.size > max_points:
            v = rng.choice(v, max_points, replace=False)
        ax.scatter(v, np.full(v.size, i), s=style.marker_size ** 2 / 2, color=cols[i], alpha=min(style.alpha, 0.5), lw=0)
    ax.set_yticks(range(len(gs)), [short(l) for l, _ in gs])
    ax.set_ylim(-0.6, len(gs) - 0.4)
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=d.name("group") if d.has("group") else "", legend=False, grid_axis="x")
    return ax


@chart("jitter", "Jitter plot", "Distribution", [_X, _G],
       [Param("spread", "Jitter spread", "float", 0.3, min=0.02, max=0.5), Param("max_points", "Max points per group", "int", 5000, min=100, max=200000)],
       aliases=("jitter chart", "jittered strip plot"))
def jitter(fig, d, style, spread=0.3, max_points=5000):
    """Like a strip plot but with random vertical jitter (seeded) so overlapping points are visible."""
    ax = make_axes(fig, style)
    gs = _cat_positions(d)
    if not gs:
        return empty_message(fig, style, "No numeric data")
    rng = np.random.default_rng(1)
    cols = _colors(style, [l for l, _ in gs])
    for i, (lab, v) in enumerate(gs):
        if v.size > max_points:
            v = rng.choice(v, max_points, replace=False)
        ax.scatter(v, i + rng.uniform(-spread, spread, v.size), s=style.marker_size ** 2 / 2, color=cols[i],
                   alpha=min(style.alpha, 0.6), lw=0)
    ax.set_yticks(range(len(gs)), [short(l) for l, _ in gs])
    ax.set_ylim(-0.6, len(gs) - 0.4)
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=d.name("group") if d.has("group") else "", legend=False, grid_axis="x")
    return ax


def beeswarm_offsets(v: np.ndarray, radius: float) -> np.ndarray:
    """Deterministic swarm layout: points are placed in value order at the smallest |offset| that keeps
    every pair at least ``2*radius`` apart (in the same units as v and the offset axis)."""
    order = np.argsort(v, kind="stable")
    off = np.zeros(len(v))
    placed_x: list[float] = []
    placed_o: list[float] = []
    for j in order:
        x = v[j]
        cand = [0.0]
        near = [(px, po) for px, po in zip(reversed(placed_x), reversed(placed_o)) if x - px < 2 * radius]
        for px, po in near:
            dx = x - px
            dy = np.sqrt(max((2 * radius) ** 2 - dx * dx, 0.0))
            cand += [po + dy, po - dy]
        best, best_abs = 0.0, np.inf
        for c in sorted(cand, key=abs):
            if all((x - px) ** 2 + (c - po) ** 2 >= (2 * radius) ** 2 - 1e-12 for px, po in near):
                best, best_abs = c, abs(c)
                break
        off[j] = best
        placed_x.append(x)
        placed_o.append(best)
        if len(placed_x) > 400:                       # only recent neighbours can collide
            placed_x, placed_o = placed_x[-400:], placed_o[-400:]
    return off


@chart("beeswarm", "Beeswarm plot", "Distribution", [_X, _G],
       [Param("max_points", "Max points per group", "int", 1500, min=50, max=20000)], aliases=("bee swarm", "swarm plot", "beeswarm chart"))
def beeswarm(fig, d, style, max_points=1500):
    """Each observation drawn once; points are nudged sideways just enough that none overlap."""
    ax = make_axes(fig, style)
    gs = _cat_positions(d)
    if not gs:
        return empty_message(fig, style, "No numeric data")
    rng = np.random.default_rng(1)
    allv = np.concatenate([v for _, v in gs])
    xr = np.ptp(allv) or 1.0
    cols = _colors(style, [l for l, _ in gs])
    W, H = fig.get_size_inches() * fig.dpi * 0.8          # approximate axes size in pixels
    px_per_unit_x = W / (xr * 1.1)
    px_per_group = H / max(len(gs), 1)
    r_px = style.marker_size / 2 * fig.dpi / 72 * 1.05      # marker radius in pixels
    for i, (lab, v) in enumerate(gs):
        if v.size > max_points:
            v = rng.choice(v, max_points, replace=False)
            d.notes.append(f"{lab}: random sample of {max_points} points")
        off_px = beeswarm_offsets(v * px_per_unit_x, r_px)      # swarm computed in pixel space -> round dots
        off = np.clip(off_px / px_per_group, -0.45, 0.45)
        ax.scatter(v, i + off, s=style.marker_size ** 2, color=cols[i], alpha=style.alpha, lw=0)
    ax.set_yticks(range(len(gs)), [short(l) for l, _ in gs])
    ax.set_ylim(-0.6, len(gs) - 0.4)
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=d.name("group") if d.has("group") else "", legend=False, grid_axis="x")
    return ax


# ------------------------------------------------------------------ 1-D heatmap
@chart("one_d_heatmap", "1-D heatmap", "Distribution", [_X, _G],
       [Param("bins", "Bins", "int", 60, min=5, max=400)], aliases=("one-dimensional heatmap", "1D heatmap", "heat strip"))
def one_d_heatmap(fig, d, style, bins=60):
    """Colour-coded counts along a single axis - a histogram shown as a coloured strip (one strip per group)."""
    ax = make_axes(fig, style)
    gs = [(l, v) for l, v in groups(d) if v.size]
    if not gs:
        return empty_message(fig, style, "No numeric data")
    allv = np.concatenate([v for _, v in gs])
    edges = bin_edges(allv, bins)
    mat = np.array([np.histogram(v, bins=edges)[0] for _, v in gs], dtype=float)
    im = ax.imshow(mat, aspect="auto", cmap=style.cmap, extent=[edges[0], edges[-1], len(gs), 0], interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Count", color=style.text_color)
    cb.ax.tick_params(colors=style.text_color, labelsize=style.tick_size)
    ax.set_yticks(np.arange(len(gs)) + 0.5, [short(l) for l, _ in gs])
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=d.name("group") if d.has("group") else "", legend=False, grid_axis="x")
    ax.grid(False)
    return ax


# ------------------------------------------------------------------ box plot
@chart("box", "Box plot", "Distribution", [_X, _G],
       [Param("whis", "Whisker length (x IQR)", "float", 1.5, min=0.5, max=5),
        Param("show_mean", "Show mean", "bool", True), Param("show_n", "Show n under labels", "bool", False),
        Param("horizontal", "Horizontal", "bool", False), Param("notch", "Notched (95% CI of median)", "bool", False)],
       aliases=("box and whisker plot", "box-and-whisker", "boxplot", "box chart"))
def box(fig, d, style, whis=1.5, show_mean=True, show_n=False, horizontal=False, notch=False):
    """Tukey box plot: box = Q1..Q3, line = median, whiskers = furthest point within whis x IQR, dots = outliers."""
    ax = make_axes(fig, style)
    gs = [(l, tukey_stats(v, whis)) for l, v in groups(d)]
    gs = [(l, s) for l, s in gs if s]
    if not gs:
        return empty_message(fig, style, "No numeric data")
    stats = []
    for l, s in gs:
        e = dict(s)
        e["label"] = short(l) + (f"\n(n={s['n']})" if show_n else "")
        if notch:
            half = 1.57 * (s["q3"] - s["q1"]) / np.sqrt(s["n"])
            e["cilo"], e["cihi"] = s["med"] - half, s["med"] + half
        stats.append(e)
    cols = _colors(style, [l for l, _ in gs])
    bp = ax.bxp(stats, orientation="horizontal" if horizontal else "vertical", patch_artist=True, showmeans=show_mean, shownotches=notch,
                showfliers=True, widths=style.bar_width * 0.7, meanprops=dict(marker="D", markerfacecolor="white",
                markeredgecolor=style.text_color, markersize=style.marker_size - 1),
                medianprops=dict(color=style.text_color, lw=style.line_width),
                whiskerprops=dict(color=style.text_color), capprops=dict(color=style.text_color),
                flierprops=dict(marker="o", markersize=style.marker_size / 1.8, markerfacecolor="none", markeredgecolor=style.text_color, alpha=0.6))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(style.alpha)
        patch.set_edgecolor(style.text_color)
    val = d.name("x")
    cat = d.name("group") if d.has("group") else ""
    finish(fig, ax, style, xlabel=val if horizontal else cat, ylabel=cat if horizontal else val, legend=False,
           grid_axis="x" if horizontal else "y")
    return ax


# ------------------------------------------------------------------ violin
@chart("violin", "Violin plot", "Distribution", [_X, _G],
       [Param("inner", "Inside", "choice", "box", choices=("box", "median", "none")),
        Param("horizontal", "Horizontal", "bool", False), Param("bw", "Bandwidth rule", "choice", "scott", choices=("scott", "silverman"))],
       aliases=("violin chart", "violin graph"))
def violin(fig, d, style, inner="box", horizontal=False, bw="scott"):
    """Mirrored kernel-density shape for each group, with quartiles / median marked inside."""
    ax = make_axes(fig, style)
    gs = [(l, v) for l, v in groups(d) if v.size > 1]
    if not gs:
        return empty_message(fig, style, "Need at least 2 values per group")
    cols = _colors(style, [l for l, _ in gs])
    allv = np.concatenate([v for _, v in gs])
    pad = 0.05 * (np.ptp(allv) or 1)
    for i, (lab, v) in enumerate(gs):
        grid = np.linspace(v.min(), v.max(), 200)
        y = kde(v, grid, bw)
        w = y / (y.max() or 1) * 0.42 * (style.bar_width / 0.8)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        if horizontal:
            ax.fill_between(grid, i - w, i + w, color=cols[i], alpha=style.alpha, lw=0.8, edgecolor=style.text_color)
            if inner == "box":
                ax.plot([q1, q3], [i, i], color=style.text_color, lw=4, solid_capstyle="butt")
                ax.plot([v.min(), v.max()], [i, i], color=style.text_color, lw=1)
                ax.scatter([med], [i], color="white", s=18, zorder=5)
            elif inner == "median":
                ax.plot([med, med], [i - 0.15, i + 0.15], color=style.text_color, lw=2)
        else:
            ax.fill_betweenx(grid, i - w, i + w, color=cols[i], alpha=style.alpha, lw=0.8, edgecolor=style.text_color)
            if inner == "box":
                ax.plot([i, i], [q1, q3], color=style.text_color, lw=4, solid_capstyle="butt")
                ax.plot([i, i], [v.min(), v.max()], color=style.text_color, lw=1)
                ax.scatter([i], [med], color="white", s=18, zorder=5)
            elif inner == "median":
                ax.plot([i - 0.15, i + 0.15], [med, med], color=style.text_color, lw=2)
    ticks = [short(l) for l, _ in gs]
    if horizontal:
        ax.set_yticks(range(len(gs)), ticks)
        ax.set_ylim(-0.6, len(gs) - 0.4)
        ax.set_xlim(allv.min() - pad, allv.max() + pad)
    else:
        ax.set_xticks(range(len(gs)), ticks)
        ax.set_xlim(-0.6, len(gs) - 0.4)
        ax.set_ylim(allv.min() - pad, allv.max() + pad)
    val = d.name("x")
    cat = d.name("group") if d.has("group") else ""
    finish(fig, ax, style, xlabel=val if horizontal else cat, ylabel=cat if horizontal else val, legend=False,
           grid_axis="x" if horizontal else "y")
    return ax
