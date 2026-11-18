"""Correlation charts: heatmap, bubble, scatter, connected scatter, hexbin, contour.

Large inputs: scatter-like charts sample (seeded) or switch to density rendering above ``max_points``.
"""
from __future__ import annotations

import numpy as np

from .base import (AGGS, ChartData, ChartStyle, Param, Role, chart, empty_message, finish, make_axes, numeric,
                   short)

SEED = 20240611
KENDALL_CAP = 20000


# --------------------------------------------------------------------------- shared helpers
def cat_strings(a):
    """Categorical column -> (str array, valid mask). None / NaN count as missing."""
    a = np.asarray(a)
    if a.dtype.kind == "O":
        valid = np.fromiter((not (v is None or (isinstance(v, float) and v != v)) for v in a), bool, len(a))
    elif a.dtype.kind == "f":
        valid = ~np.isnan(a)
    else:
        valid = np.ones(len(a), bool)
    return a.astype(str), valid


def split_groups(g, mask):
    """Rows where ``mask`` is True and the group is present -> [(label, row indices)] in first-seen order."""
    strs, valid = cat_strings(g)
    idx = np.flatnonzero(np.asarray(mask) & valid)
    if idx.size == 0:
        return []
    uniq, first, inv = np.unique(strs[idx], return_index=True, return_inverse=True)
    return [(str(uniq[k]), idx[inv == k]) for k in np.argsort(first)]


def sample_mask(n_total: int, keep: np.ndarray, max_points: int, seed: int = SEED):
    """Boolean mask of at most ``max_points`` rows drawn (seeded) from ``keep``."""
    idx = np.flatnonzero(keep)
    if idx.size <= max_points:
        return keep.copy()
    rng = np.random.default_rng(seed)
    pick = rng.choice(idx, size=int(max_points), replace=False)
    out = np.zeros(n_total, bool)
    out[pick] = True
    return out


def _maxpts(v) -> int:
    try:
        return max(int(v), 10)
    except (TypeError, ValueError):
        return 100000


def lin_fit(x, y):
    """Simple OLS. Returns dict(slope, intercept, n, xbar, sxx, s2) or None."""
    n = len(x)
    if n < 2:
        return None
    xbar, ybar = x.mean(), y.mean()
    dx = x - xbar
    sxx = float(np.dot(dx, dx))
    if sxx <= 0:
        return None
    slope = float(np.dot(dx, y - ybar) / sxx)
    icpt = float(ybar - slope * xbar)
    res = y - (icpt + slope * x)
    s2 = float(np.dot(res, res) / (n - 2)) if n > 2 else 0.0
    return dict(slope=slope, intercept=icpt, n=n, xbar=float(xbar), sxx=sxx, s2=s2)


def _pt_size(style: ChartStyle) -> float:
    return max(style.marker_size ** 2 * 0.6, 4.0)


def _cbar(fig, mappable, ax, style, label=""):
    cb = fig.colorbar(mappable, ax=ax)
    cb.ax.tick_params(colors=style.text_color, labelsize=style.tick_size)
    cb.outline.set_edgecolor(style.grid_color)
    if label:
        cb.set_label(label, color=style.text_color, fontsize=style.label_size)
    return cb


def _lum_color(rgba):
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return "#ffffff" if lum < 0.5 else "#111111"


def _pair_xy(d: ChartData, *roles):
    arrs = [numeric(d[r]) for r in roles]
    n = min(len(a) for a in arrs)
    arrs = [a[:n] for a in arrs]
    m = np.ones(n, bool)
    for a in arrs:
        m &= np.isfinite(a)
    return arrs, m


# --------------------------------------------------------------------------- heatmap
def _group_reduce(codes, v, k, agg):
    """Aggregate ``v`` (finite) by integer ``codes`` in [0,k). Empty cells -> NaN (count -> 0)."""
    if agg == "count":
        return np.bincount(codes, minlength=k).astype(float)
    cnt = np.bincount(codes, minlength=k).astype(float)
    if agg in ("sum", "mean"):
        s = np.bincount(codes, weights=v, minlength=k)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = s / cnt if agg == "mean" else s
        out[cnt == 0] = np.nan
        return out
    out = np.full(k, np.nan)
    order = np.argsort(codes, kind="stable")
    sc, sv = codes[order], v[order]
    bounds = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]]) if sc.size else np.array([], int)
    ends = np.r_[bounds[1:], sc.size]
    f = {"min": np.min, "max": np.max, "median": np.median}[agg]
    for b, e in zip(bounds, ends):
        out[sc[b]] = f(sv[b:e])
    return out


def corr_matrix(cols, method="pearson", notes=None):
    """Pairwise-complete correlation matrix (NaN where undefined)."""
    from scipy import stats
    k = len(cols)
    M = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(i, k):
            a, b = cols[i], cols[j]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 2:
                continue
            x, y = a[m], b[m]
            if i == j:
                M[i, i] = 1.0 if np.ptp(x) > 0 else np.nan
                continue
            if np.ptp(x) == 0 or np.ptp(y) == 0:
                continue
            if method == "kendall":
                if x.size > KENDALL_CAP:
                    pick = np.sort(np.random.default_rng(SEED).choice(x.size, KENDALL_CAP, replace=False))
                    x, y = x[pick], y[pick]
                    if notes is not None and not any("Kendall" in s for s in notes):
                        notes.append(f"Kendall tau computed on a seeded sample of {KENDALL_CAP:,} rows per pair")
                r = float(stats.kendalltau(x, y)[0])
            else:
                if method == "spearman":
                    x, y = stats.rankdata(x), stats.rankdata(y)
                dx, dy = x - x.mean(), y - y.mean()
                den = np.sqrt(np.dot(dx, dx) * np.dot(dy, dy))
                r = float(np.dot(dx, dy) / den) if den > 0 else np.nan
            M[i, j] = M[j, i] = np.clip(r, -1, 1) if np.isfinite(r) else np.nan
    return M


@chart("heatmap", "Heatmap", "Correlation",
       [Role("dims", "numeric_multi", "Variables (correlation mode)", required=False),
        Role("row", "categorical", "Row (pivot mode)", required=False),
        Role("col", "categorical", "Column (pivot mode)", required=False),
        Role("value", "numeric", "Value (pivot mode; blank = count)", required=False)],
       [Param("mode", "Mode", "choice", "correlation", choices=("correlation", "pivot")),
        Param("method", "Correlation method", "choice", "pearson", choices=("pearson", "spearman", "kendall")),
        Param("annotate", "Show values in cells", "bool", True),
        Param("agg", "Aggregate (pivot)", "choice", "sum", choices=AGGS)],
       aliases=("correlation matrix", "correlogram"))
def heatmap(fig, d, style, mode="correlation", method="pearson", annotate=True, agg="sum"):
    """Correlation matrix (diverging, centred at 0, -1..1) or an aggregated row x column pivot (sequential)."""
    if mode == "pivot":
        if not (d.has("row") and d.has("col")):
            return empty_message(fig, style, "Pivot heatmap needs a Row and a Column variable")
        rs, rv = cat_strings(d["row"])
        cs, cv = cat_strings(d["col"])
        n = min(len(rs), len(cs))
        ok = rv[:n] & cv[:n]
        has_val = d.has("value") and agg != "count"
        if has_val:
            val = numeric(d["value"])[:n]
            ok &= np.isfinite(val)
        if not ok.any():
            return empty_message(fig, style, "No data to draw")
        rs, cs = rs[:n][ok], cs[:n][ok]
        ru, rf, ri = np.unique(rs, return_index=True, return_inverse=True)
        cu, cf, ci = np.unique(cs, return_index=True, return_inverse=True)
        ro, co = np.argsort(np.argsort(rf)), np.argsort(np.argsort(cf))   # first-seen order
        rows_l = [str(x) for x in ru[np.argsort(rf)]]
        cols_l = [str(x) for x in cu[np.argsort(cf)]]
        ri, ci = ro[ri], co[ci]
        if len(rows_l) > 100 or len(cols_l) > 100:
            d.notes.append("pivot truncated to the first 100 rows / columns")
            keep = (ri < 100) & (ci < 100)
            ri, ci = ri[keep], ci[keep]
            vv = val[ok][keep] if has_val else None
            rows_l, cols_l = rows_l[:100], cols_l[:100]
        else:
            vv = val[ok] if has_val else None
        nr, nc = len(rows_l), len(cols_l)
        M = _group_reduce(ri * nc + ci, vv if has_val else np.zeros(ri.size), nr * nc,
                          agg if has_val else "count").reshape(nr, nc)
        if not has_val:
            M[M == 0] = np.nan  # empty cell
            M = np.where(np.isnan(M), np.nan, M)
        cap = (agg if has_val else "count").capitalize() + (f" of {d.name('value')}" if has_val else "")
        ax = make_axes(fig, style)
        im = ax.imshow(np.ma.masked_invalid(M), cmap=style.cmap, aspect="auto", interpolation="nearest")
        xl, yl = d.name("col"), d.name("row")
        cbl = cap
    else:
        cols = [numeric(c) for c in (d.get("dims") or [])]
        if len(cols) < 2:
            return empty_message(fig, style, "Select at least two numeric variables")
        M = corr_matrix(cols, method, d.notes)
        rows_l = cols_l = [short(n, 16) for n in d.label_list("dims")][:len(cols)]
        if not np.isfinite(M).any():
            return empty_message(fig, style, "No valid pairs to correlate")
        ax = make_axes(fig, style)
        im = ax.imshow(np.ma.masked_invalid(M), cmap=style.diverging_cmap, vmin=-1, vmax=1,
                       aspect="auto", interpolation="nearest")
        xl = yl = ""
        cbl = f"{method.capitalize()} correlation"
        nr = nc = len(cols)
    ax.set_xticks(range(nc), [short(c, 14) for c in cols_l])
    ax.set_yticks(range(nr), [short(r, 14) for r in rows_l])
    if nc > 6 and not style.rotate_xticks:
        for lab in ax.get_xticklabels():
            lab.set_rotation(45)
            lab.set_ha("right")
    if annotate and nr * nc <= 400:
        cm = im.get_cmap()
        for i in range(nr):
            for j in range(nc):
                v = M[i, j]
                if np.isfinite(v):
                    txt = f"{v:.2f}" if mode != "pivot" else (f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3g}")
                    ax.text(j, i, txt, ha="center", va="center", fontsize=style.value_label_size,
                            color=_lum_color(cm(im.norm(v))))
    _cbar(fig, im, ax, style, cbl)
    finish(fig, ax, style, xlabel=xl, ylabel=yl, legend=False)
    ax.grid(False)
    return ax


# --------------------------------------------------------------------------- bubble
@chart("bubble", "Bubble chart", "Correlation",
       [Role("x", "numeric", "X"), Role("y", "numeric", "Y"), Role("size", "numeric", "Size"),
        Role("group", "categorical", "Group (colour)", required=False)],
       [Param("max_bubble", "Largest bubble area (pt²)", "float", 600.0, min=10, max=20000),
        Param("alpha", "Opacity", "float", 0.6, min=0.05, max=1.0),
        Param("max_points", "Max bubbles drawn", "int", 100000, min=100, max=2000000)],
       aliases=("bubble plot",))
def bubble(fig, d, style, max_bubble=600.0, alpha=0.6, max_points=100000):
    """Bubble AREA is proportional to the size variable (so the radius scales with sqrt(size)).

    Non-positive / missing sizes are dropped."""
    (x, y, s), m = _pair_xy(d, "x", "y", "size")
    pos = m & (s > 0)
    if m.sum() > pos.sum():
        d.notes.append(f"{int(m.sum() - pos.sum())} rows with size <= 0 not drawn")
    if pos.sum() == 0:
        return empty_message(fig, style, "No valid (x, y, size) rows to draw")
    smax = float(s[pos].max())
    keep = sample_mask(len(x), pos, _maxpts(max_points))
    if keep.sum() < pos.sum():
        d.notes.append(f"sampled {int(keep.sum()):,} of {int(pos.sum()):,} bubbles (seeded)")
    area = float(max_bubble) * s / smax
    ax = make_axes(fig, style)
    if d.has("group"):
        groups = split_groups(d["group"], keep)
    else:
        groups = [("", np.flatnonzero(keep))]
    for i, (lab, idx) in enumerate(groups):
        ax.scatter(x[idx], y[idx], s=area[idx], color=style.color_for(lab, i) if lab else style.color_for("", 0),
                   alpha=alpha, edgecolors=style.axes_background, linewidths=0.6, label=lab or None)
    ax.margins(0.08)
    ax.text(0.99, 0.01, f"bubble area ∝ {d.name('size', 'size')} (largest = {smax:,.4g})", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=max(style.tick_size - 1, 6), color=style.text_color, alpha=0.7)
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=d.name("y"), grid_axis="both")
    return ax


# --------------------------------------------------------------------------- scatter
def _trend(ax, x, y, trend, degree, show_ci, color, style, label_prefix="", d=None):
    """Draw a trend line for one series. Returns nothing."""
    from scipy import stats
    if x.size < 3 or np.ptp(x) == 0:
        return
    xs = np.linspace(x.min(), x.max(), 200)
    lw = style.line_width
    if trend == "linear":
        f = lin_fit(x, y)
        if f is None:
            return
        x2 = np.array([x.min(), x.max()])
        ax.plot(x2, f["intercept"] + f["slope"] * x2, color=color, lw=lw, label=f"{label_prefix}linear fit")
        if show_ci and f["n"] > 2:
            t = stats.t.ppf(0.975, f["n"] - 2)
            yh = f["intercept"] + f["slope"] * xs
            se = np.sqrt(f["s2"] * (1.0 / f["n"] + (xs - f["xbar"]) ** 2 / f["sxx"]))
            ax.fill_between(xs, yh - t * se, yh + t * se, color=color, alpha=0.18, lw=0, label=f"{label_prefix}95% CI")
    elif trend == "polynomial":
        deg = int(min(max(degree, 1), 8, x.size - 1))
        p = np.polynomial.Polynomial.fit(x, y, deg)
        ax.plot(xs, p(xs), color=color, lw=lw, label=f"{label_prefix}polynomial (deg {deg})")
    elif trend == "lowess":
        from statsmodels.nonparametric.smoothers_lowess import lowess
        xx, yy = x, y
        if x.size > 20000:
            pick = np.sort(np.random.default_rng(SEED).choice(x.size, 20000, replace=False))
            xx, yy = x[pick], y[pick]
            if d is not None:
                d.notes.append("LOWESS fitted on a seeded sample of 20,000 points")
        fit = lowess(yy, xx, frac=0.6667, it=2, delta=0.005 * float(np.ptp(xx)), return_sorted=True)
        ax.plot(fit[:, 0], fit[:, 1], color=color, lw=lw, label=f"{label_prefix}LOWESS")


@chart("scatter", "Scatter plot", "Correlation",
       [Role("x", "numeric", "X"), Role("y", "numeric", "Y"),
        Role("group", "categorical", "Group (colour)", required=False)],
       [Param("trend", "Trend line", "choice", "none", choices=("none", "linear", "polynomial", "lowess")),
        Param("degree", "Polynomial degree", "int", 2, min=1, max=8),
        Param("show_ci", "95% CI band (linear)", "bool", False),
        Param("max_points", "Max points drawn", "int", 100000, min=100, max=2000000)],
       aliases=("scatterplot", "xy plot", "scatter chart"))
def scatter(fig, d, style, trend="none", degree=2, show_ci=False, max_points=100000):
    """Scatter with optional linear / polynomial / LOWESS trend per group.

    Above ``max_points`` an ungrouped plot becomes a hexbin density; grouped plots are sampled (seeded)."""
    (x, y), m = _pair_xy(d, "x", "y")
    nfin = int(m.sum())
    if nfin == 0:
        return empty_message(fig, style, "No finite (x, y) pairs to draw")
    mp = _maxpts(max_points)
    grouped = d.has("group")
    ax = make_axes(fig, style)
    dense = nfin > mp and not grouped
    if dense:
        d.notes.append(f"{nfin:,} points > {mp:,}: drawn as hexbin density")
        hb = ax.hexbin(x[m], y[m], gridsize=60, mincnt=1, cmap=style.cmap, bins="log", linewidths=0.1)
        _cbar(fig, hb, ax, style, "log10(count)")
        series = [("", np.flatnonzero(m))]
    else:
        keep = sample_mask(len(x), m, mp)
        if keep.sum() < nfin:
            d.notes.append(f"sampled {int(keep.sum()):,} of {nfin:,} points (seeded)")
        series = split_groups(d["group"], keep) if grouped else [("", np.flatnonzero(keep))]
        for i, (lab, idx) in enumerate(series):
            ax.scatter(x[idx], y[idx], s=_pt_size(style), color=style.color_for(lab or "", i), alpha=style.alpha,
                       linewidths=0, label=lab or None)
    if trend != "none":
        # trend uses every finite row of the group (not only the sampled ones)
        full = split_groups(d["group"], m) if grouped else [("", np.flatnonzero(m))]
        for i, (lab, idx) in enumerate(full):
            col = style.color_for(lab or "", i) if not dense else style.palette_colors()[4]
            _trend(ax, x[idx], y[idx], trend, degree, show_ci and trend == "linear", col, style,
                   label_prefix=f"{lab}: " if grouped else "", d=d)
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=d.name("y"), grid_axis="both")
    return ax


# --------------------------------------------------------------------------- connected scatter
def _order_index(o):
    o = np.asarray(o)
    try:
        if o.dtype.kind in "fiubM":
            key = o.astype(float) if o.dtype.kind != "M" else o
            key = np.where(np.isnan(key), np.inf, key) if o.dtype.kind == "f" else key
            return np.argsort(key, kind="stable")
        return np.argsort(o, kind="stable")
    except TypeError:
        return np.argsort(o.astype(str), kind="stable")


@chart("connected_scatter", "Connected scatter plot", "Correlation",
       [Role("x", "numeric", "X"), Role("y", "numeric", "Y"),
        Role("order", "any", "Order by (blank = row order)", required=False)],
       aliases=("connected scatterplot", "path plot"))
def connected_scatter(fig, d, style):
    """Points joined by a line in the given order; first / last points are labelled and an arrow shows direction."""
    (x, y), m = _pair_xy(d, "x", "y")
    n = len(x)
    idx = np.arange(n)
    if d.has("order"):
        o = np.asarray(d["order"])[:n]
        idx = _order_index(o)
    idx = idx[m[idx]]
    if idx.size == 0:
        return empty_message(fig, style, "No finite (x, y) pairs to draw")
    if idx.size > 200000:
        idx = idx[np.unique(np.linspace(0, idx.size - 1, 200000).astype(int))]
        d.notes.append("path thinned to 200,000 evenly spaced points")
    xs, ys = x[idx], y[idx]
    col = style.color_for("", 0)
    ax = make_axes(fig, style)
    ms = style.marker_size if idx.size <= 2000 else 1.5
    ax.plot(xs, ys, "-", marker="o", color=col, lw=style.line_width * 0.8, markersize=ms, alpha=style.alpha,
            markerfacecolor=col, markeredgewidth=0, label="path")
    kw = dict(fontsize=style.label_size, color=style.text_color, fontweight="bold")
    ax.annotate("start (1)", (xs[0], ys[0]), xytext=(-8, 8), textcoords="offset points", ha="right", **kw)
    ax.annotate(f"end ({xs.size})", (xs[-1], ys[-1]), xytext=(8, -10), textcoords="offset points", ha="left", **kw)
    if xs.size >= 2 and (xs[-1] != xs[-2] or ys[-1] != ys[-2]):
        ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-2], ys[-2]),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=style.line_width, mutation_scale=16))
    ax.margins(0.1)
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=d.name("y"), legend=False, grid_axis="both")
    return ax


# --------------------------------------------------------------------------- hexbin
@chart("hexbin", "Hexbin plot", "Correlation",
       [Role("x", "numeric", "X"), Role("y", "numeric", "Y")],
       [Param("gridsize", "Hexagons across", "int", 30, min=5, max=200),
        Param("log_counts", "Log colour scale", "bool", False),
        Param("min_count", "Minimum count to draw", "int", 1, min=1, max=100000)],
       aliases=("hex bin", "hexagonal binning"))
def hexbin(fig, d, style, gridsize=30, log_counts=False, min_count=1):
    """2-D density with hexagonal bins (counts of finite points)."""
    (x, y), m = _pair_xy(d, "x", "y")
    if m.sum() < 2:
        return empty_message(fig, style, "Need at least two finite (x, y) pairs")
    xs, ys = x[m], y[m]
    ax = make_axes(fig, style)
    pad = lambda a: (a.min() - 0.5, a.max() + 0.5) if np.ptp(a) == 0 else (a.min(), a.max())
    x0, x1 = pad(xs)
    y0, y1 = pad(ys)
    hb = ax.hexbin(xs, ys, gridsize=int(gridsize), mincnt=int(max(min_count, 1)), cmap=style.cmap,
                   bins="log" if log_counts else None, extent=(x0, x1, y0, y1), linewidths=0.2,
                   edgecolors="face")
    _cbar(fig, hb, ax, style, "log10(count)" if log_counts else "Count")
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=d.name("y"), legend=False, grid_axis="both")
    ax.grid(False)
    return ax


# --------------------------------------------------------------------------- contour
def _kde_grid(x, y, grid):
    """Gaussian KDE on a regular grid via 2-D histogram + gaussian smoothing (Scott bandwidth)."""
    from scipy.ndimage import gaussian_filter
    n = x.size
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    sx = sx if sx > 0 else 1.0
    sy = sy if sy > 0 else 1.0
    f = n ** (-1.0 / 6.0)
    hx, hy = f * sx, f * sy
    x0, x1 = x.min() - 3 * hx, x.max() + 3 * hx
    y0, y1 = y.min() - 3 * hy, y.max() + 3 * hy
    H, xe, ye = np.histogram2d(x, y, bins=grid, range=[[x0, x1], [y0, y1]])
    bw_x, bw_y = (x1 - x0) / grid, (y1 - y0) / grid
    dens = gaussian_filter(H / (n * bw_x * bw_y), sigma=(hx / bw_x, hy / bw_y), mode="constant")
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    return xc, yc, dens.T


@chart("contour", "Contour plot", "Correlation",
       [Role("x", "numeric", "X"), Role("y", "numeric", "Y"), Role("z", "numeric", "Z (blank = density of x, y)", required=False)],
       [Param("levels", "Levels", "int", 8, min=2, max=50),
        Param("filled", "Filled", "bool", True),
        Param("grid", "Grid resolution", "int", 100, min=20, max=400),
        Param("method", "Interpolation (with Z)", "choice", "linear", choices=("linear", "cubic", "nearest"))],
       aliases=("contour chart", "density contour", "isoline"))
def contour(fig, d, style, levels=8, filled=True, grid=100, method="linear"):
    """Contours of z (scipy griddata onto a regular grid) or, without z, a 2-D Gaussian KDE of the points."""
    from scipy.interpolate import griddata
    grid = int(min(max(grid, 20), 400))
    levels = int(min(max(levels, 2), 50))
    has_z = d.has("z")
    roles = ("x", "y", "z") if has_z else ("x", "y")
    arrs, m = _pair_xy(d, *roles)
    if m.sum() < (4 if has_z else 3):
        return empty_message(fig, style, "Not enough finite points for a contour plot")
    x, y = arrs[0][m], arrs[1][m]
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return empty_message(fig, style, "x and y must both vary")
    if has_z:
        z = arrs[2][m]
        if x.size > 20000:
            pick = np.sort(np.random.default_rng(SEED).choice(x.size, 20000, replace=False))
            x, y, z = x[pick], y[pick], z[pick]
            d.notes.append("interpolation used a seeded sample of 20,000 points")
        gx = np.linspace(x.min(), x.max(), grid)
        gy = np.linspace(y.min(), y.max(), grid)
        GX, GY = np.meshgrid(gx, gy)
        try:
            Z = griddata((x, y), z, (GX, GY), method=method)
        except Exception:  # degenerate geometry (e.g. collinear points)
            Z = griddata((x, y), z, (GX, GY), method="nearest")
        cbl = d.name("z")
        xc, yc = gx, gy
        lv = levels
    else:
        xc, yc, Z = _kde_grid(x, y, grid)
        cbl = "Density"
        top = float(np.nanmax(Z))
        lv = np.linspace(top * 0.05, top, levels + 1) if top > 0 else levels
    Z = np.ma.masked_invalid(Z)
    if Z.count() == 0 or np.ptp(Z.compressed()) == 0:
        return empty_message(fig, style, "Z is constant; nothing to contour")
    ax = make_axes(fig, style)
    if filled:
        cs = ax.contourf(xc, yc, Z, levels=lv, cmap=style.cmap)
        ax.contour(xc, yc, Z, levels=cs.levels, colors=style.axes_background, linewidths=0.4, alpha=0.6)
    else:
        cs = ax.contour(xc, yc, Z, levels=lv, cmap=style.cmap, linewidths=style.line_width * 0.7)
    _cbar(fig, cs, ax, style, cbl)
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=d.name("y"), legend=False, grid_axis="both")
    return ax
