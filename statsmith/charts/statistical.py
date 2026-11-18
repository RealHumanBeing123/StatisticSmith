"""Statistical charts: pareto, Q-Q plot, ECDF, scatter matrix, error bars, regression, residual diagnostics,
control chart (individuals), Bland-Altman.

Every number drawn here is computed with numpy / scipy / statsmodels so tests can verify it independently.
"""
from __future__ import annotations

import numpy as np

from .base import (AGGS, ChartData, ChartStyle, Param, Role, _style_text, aggregate, chart, empty_message, fmt,
                   finish, make_axes, numeric, short)
from .correlation import SEED, _maxpts, _order_index, _pt_size, cat_strings, sample_mask, split_groups

DISTS = ("normal", "exponential", "uniform", "t")
MR_D2 = 1.128  # d2 constant for moving ranges of size 2


def _ppf(dist: str, df: float):
    from scipy import stats
    if dist == "exponential":
        return stats.expon.ppf
    if dist == "uniform":
        return stats.uniform.ppf
    if dist == "t":
        return lambda p: stats.t.ppf(p, max(float(df), 1.0))
    return stats.norm.ppf


def _blom(n: int) -> np.ndarray:
    return (np.arange(1, n + 1) - 0.375) / (n + 0.25)


def _clear_grid_fig(fig, style: ChartStyle):
    """Clear ``fig`` and apply figure-level style exactly like make_axes (without leaving an Axes behind)."""
    make_axes(fig, style).remove()


def _prep_axis(ax, style: ChartStyle):
    ax.set_facecolor(style.axes_background)
    _style_text(ax, style)
    if style.show_grid:
        ax.grid(True, color=style.grid_color, linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)


def _groups(d: ChartData, mask: np.ndarray):
    if d.has("group"):
        g = np.asarray(d["group"])
        n = min(len(g), len(mask))
        m = mask.copy()
        m[n:] = False
        return split_groups(g[:n], m[:n]) if n else []
    return [("", np.flatnonzero(mask))]


# --------------------------------------------------------------------------- pareto
@chart("pareto", "Pareto chart", "Statistical",
       [Role("x", "categorical", "Category"), Role("y", "numeric", "Value (blank = count)", required=False)],
       [Param("agg", "Aggregate", "choice", "sum", choices=AGGS),
        Param("show_80", "Show 80% line", "bool", True),
        Param("max_bars", "Max bars (rest -> 'Other')", "int", 40, min=3, max=200)],
       aliases=("pareto diagram", "80/20 chart"))
def pareto(fig, d, style, agg="sum", show_80=True, max_bars=40):
    """Bars sorted descending with the cumulative percentage line on a secondary axis."""
    cat, valid = cat_strings(d["x"])
    val = numeric(d["y"]) if d.has("y") and agg != "count" else None
    n = len(cat) if val is None else min(len(cat), len(val))
    ok = valid[:n]
    if val is not None:
        ok = ok & np.isfinite(val[:n])
    if not ok.any():
        return empty_message(fig, style, "No data to draw")
    labels, vals = aggregate(cat[:n][ok], None if val is None else val[:n][ok], agg if val is not None else "count",
                             order="value_desc")
    vals = np.nan_to_num(np.asarray(vals, float), nan=0.0)
    if (vals < 0).any():
        d.notes.append("negative totals clipped to 0 in the Pareto chart")
        vals = np.clip(vals, 0, None)
    total = float(vals.sum())
    if total <= 0:
        return empty_message(fig, style, "All totals are zero")
    mb = int(max(max_bars, 3))
    if len(vals) > mb:
        d.notes.append(f"{len(vals) - mb + 1} smallest categories folded into 'Other'")
        labels = labels[: mb - 1] + ["Other"]
        vals = np.r_[vals[: mb - 1], vals[mb - 1:].sum()]
    cum = np.cumsum(vals) / total * 100.0
    ax = make_axes(fig, style)
    pos = np.arange(len(vals))
    bars = ax.bar(pos, vals, width=style.bar_width, color=style.color_for("", 0), alpha=style.alpha, label="Value")
    ax.set_xticks(pos, [short(l, 14) for l in labels])
    if style.show_values:
        for p, v in zip(pos, vals):
            ax.text(p, v, fmt(v, style), ha="center", va="bottom", fontsize=style.value_label_size, color=style.text_color)
    ax2 = ax.twinx()
    ax2.plot(pos, cum, color=style.palette_colors()[1], marker="o", lw=style.line_width, markersize=style.marker_size * 0.7,
             label="Cumulative %")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative %", fontsize=style.label_size, color=style.text_color)
    ax2.grid(False)
    if show_80:
        ax2.axhline(80, color=style.palette_colors()[4], ls="--", lw=1.2, label="80%")
    _style_text(ax2, style)
    ax2.spines["right"].set_visible(style.show_spines)
    if style.rotate_xticks == 0 and len(vals) > 8:
        for lab in ax.get_xticklabels():
            lab.set_rotation(45)
            lab.set_ha("right")
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=d.name("y", "Count") if val is not None else "Count",
           legend=False)
    return ax


# --------------------------------------------------------------------------- Q-Q
@chart("qq_plot", "Q-Q plot", "Statistical",
       [Role("y", "numeric", "Variable"), Role("group", "categorical", "Group", required=False)],
       [Param("dist", "Theoretical distribution", "choice", "normal", choices=DISTS),
        Param("df", "Degrees of freedom (t)", "float", 5.0, min=1, max=1000),
        Param("max_points", "Max points drawn", "int", 100000, min=100, max=2000000)],
       aliases=("qq plot", "quantile-quantile", "probability plot"))
def qq_plot(fig, d, style, dist="normal", df=5.0, max_points=100000):
    """Sorted data vs theoretical quantiles at Blom positions (i-0.375)/(n+0.25); reference line through Q1 and Q3."""
    y = numeric(d["y"])
    m = np.isfinite(y)
    if m.sum() < 2:
        return empty_message(fig, style, "Need at least two finite values")
    ppf = _ppf(dist, df)
    mp = _maxpts(max_points)
    ax = make_axes(fig, style)
    groups = _groups(d, m)
    drew = False
    for i, (lab, idx) in enumerate(groups):
        ys = np.sort(y[idx])
        n = ys.size
        if n < 2:
            continue
        th = ppf(_blom(n))
        col = style.color_for(lab or "", i)
        if n > mp:
            pick = np.unique(np.linspace(0, n - 1, mp).astype(int))
            d.notes.append(f"Q-Q plot thinned to {pick.size:,} evenly spaced order statistics"
                           + (f" ({lab})" if lab else ""))
            ax.scatter(th[pick], ys[pick], s=_pt_size(style), color=col, alpha=style.alpha, linewidths=0, label=lab or None)
        else:
            ax.scatter(th, ys, s=_pt_size(style), color=col, alpha=style.alpha, linewidths=0, label=lab or None)
        x1, x2 = float(ppf(0.25)), float(ppf(0.75))
        q1, q3 = np.quantile(ys, [0.25, 0.75])
        slope = (q3 - q1) / (x2 - x1)
        xr = np.array([th[0], th[-1]])
        ax.plot(xr, q1 + slope * (xr - x1), color=col if len(groups) > 1 else style.palette_colors()[4], lw=style.line_width,
                label=None if len(groups) > 1 else "Q1-Q3 line")
        drew = True
    if not drew:
        return empty_message(fig, style, "Need at least two values in a group")
    dn = dist if dist != "t" else f"t (df={df:g})"
    finish(fig, ax, style, xlabel=f"Theoretical quantiles ({dn})", ylabel=f"Sample quantiles ({d.name('y')})",
           grid_axis="both")
    return ax


# --------------------------------------------------------------------------- ECDF
@chart("ecdf", "ECDF", "Statistical",
       [Role("x", "numeric", "Variable"), Role("group", "categorical", "Group", required=False)],
       [Param("band", "DKW 95% confidence band", "bool", False),
        Param("max_points", "Max steps drawn", "int", 100000, min=100, max=2000000)],
       aliases=("empirical cdf", "cumulative distribution"))
def ecdf(fig, d, style, band=False, max_points=100000):
    """Empirical CDF step function F(x_(i)) = i/n; optional Dvoretzky-Kiefer-Wolfowitz 95% band."""
    x = numeric(d["x"])
    m = np.isfinite(x)
    if not m.any():
        return empty_message(fig, style, "No finite values to draw")
    mp = _maxpts(max_points)
    ax = make_axes(fig, style)
    groups = _groups(d, m)
    for i, (lab, idx) in enumerate(groups):
        xs = np.sort(x[idx])
        n = xs.size
        F = np.arange(1, n + 1) / n
        col = style.color_for(lab or "", i)
        px, pF = xs, F
        if n > mp:
            pick = np.unique(np.r_[np.linspace(0, n - 1, mp).astype(int), n - 1])
            px, pF = xs[pick], F[pick]
            d.notes.append(f"ECDF thinned to {pick.size:,} evenly spaced steps" + (f" ({lab})" if lab else ""))
        ax.step(px, pF, where="post", color=col, lw=style.line_width, label=lab or None)
        if band:
            eps = float(np.sqrt(np.log(2.0 / 0.05) / (2.0 * n)))
            ax.fill_between(px, np.clip(pF - eps, 0, 1), np.clip(pF + eps, 0, 1), step="post", color=col, alpha=0.15,
                            lw=0, label=f"{lab + ': ' if lab else ''}95% DKW band")
    ax.set_ylim(0, 1.02)
    finish(fig, ax, style, xlabel=d.name("x"), ylabel="Cumulative proportion", grid_axis="both")
    return ax


# --------------------------------------------------------------------------- scatter matrix
@chart("scatter_matrix", "Scatter plot matrix", "Statistical",
       [Role("dims", "numeric_multi", "Variables (2-8)")],
       [Param("diagonal", "Diagonal", "choice", "hist", choices=("hist", "kde")),
        Param("max_points", "Max points per panel", "int", 20000, min=100, max=2000000)],
       aliases=("splom", "pairs plot", "pair plot"))
def scatter_matrix(fig, d, style, diagonal="hist", max_points=20000):
    """k x k grid: histogram / KDE of each variable on the diagonal, pairwise scatter elsewhere."""
    from scipy import stats
    cols = [numeric(c) for c in (d.get("dims") or [])]
    if len(cols) < 2:
        return empty_message(fig, style, "Select at least two numeric variables")
    names = d.label_list("dims")
    if len(cols) > 8:
        d.notes.append("scatter matrix limited to the first 8 variables")
        cols, names = cols[:8], names[:8]
    names = (names + [f"v{i + 1}" for i in range(len(cols))])[: len(cols)]
    k = len(cols)
    mp = _maxpts(max_points)
    _clear_grid_fig(fig, style)
    first = None
    noted = False
    for i in range(k):
        for j in range(k):
            ax = fig.add_subplot(k, k, i * k + j + 1)
            first = first or ax
            _prep_axis(ax, style)
            if i == j:
                v = cols[i][np.isfinite(cols[i])]
                col = style.color_for("", 0)
                if v.size:
                    if diagonal == "kde" and v.size > 1 and np.ptp(v) > 0:
                        vv = v if v.size <= 5000 else np.random.default_rng(SEED).choice(v, 5000, replace=False)
                        g = np.linspace(v.min(), v.max(), 200)
                        dens = stats.gaussian_kde(vv)(g)
                        ax.fill_between(g, dens, color=col, alpha=style.alpha * 0.6, lw=0)
                        ax.plot(g, dens, color=col, lw=style.line_width * 0.7)
                    else:
                        nb = int(min(max(np.ceil(np.log2(v.size)) + 1, 5), 40))
                        ax.hist(v, bins=nb, color=col, alpha=style.alpha, edgecolor=style.axes_background, linewidth=0.4)
                ax.tick_params(labelleft=False)
            else:
                a, b = cols[j], cols[i]
                nn = min(a.size, b.size)
                ok = np.isfinite(a[:nn]) & np.isfinite(b[:nn])
                keep = sample_mask(nn, ok, mp, seed=SEED + i * 10 + j)
                if keep.sum() < ok.sum() and not noted:
                    d.notes.append(f"scatter panels sampled to {mp:,} points (seeded)")
                    noted = True
                ax.scatter(a[:nn][keep], b[:nn][keep], s=max(_pt_size(style) * 0.5, 3), color=style.color_for("", 0),
                           alpha=style.alpha * 0.7, linewidths=0)
                ax.tick_params(labelleft=(j == 0))
            ax.tick_params(labelbottom=(i == k - 1))
            ax.tick_params(labelsize=max(style.tick_size - 2, 5))
            if i == k - 1:
                ax.set_xlabel(short(names[j], 14), fontsize=max(style.label_size - 1, 6), color=style.text_color)
            if j == 0:
                ax.set_ylabel(short(names[i], 14), fontsize=max(style.label_size - 1, 6), color=style.text_color)
    if style.title:
        fig.suptitle(style.title, fontsize=style.title_size, color=style.text_color)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return first


# --------------------------------------------------------------------------- error bars
@chart("error_bar", "Error bar chart", "Statistical",
       [Role("x", "categorical", "Category"), Role("y", "numeric", "Value"),
        Role("group", "categorical", "Group", required=False)],
       [Param("kind", "Error bars", "choice", "t_ci", choices=("t_ci", "sd", "se")),
        Param("ci_level", "Confidence level (t_ci)", "float", 0.95, min=0.5, max=0.999)],
       aliases=("error bars", "mean plot", "dot and whisker"))
def error_bar(fig, d, style, kind="t_ci", ci_level=0.95):
    """Group means with t confidence intervals (n-1 df), +-SD or +-SE."""
    from scipy import stats
    xs_all, vx = cat_strings(d["x"])
    y = numeric(d["y"])
    n = min(len(xs_all), len(y))
    ok = vx[:n] & np.isfinite(y[:n])
    if not ok.any():
        return empty_message(fig, style, "No data to draw")
    xs, yy = xs_all[:n], y[:n]
    cats = list(dict.fromkeys(xs[ok].tolist()))
    groups = _groups(d, ok) if d.has("group") else [("", np.flatnonzero(ok))]
    ng = len(groups)
    width = 0.6 / ng
    level = float(min(max(ci_level, 0.5), 0.999))
    ax = make_axes(fig, style)
    thin = False
    for gi, (lab, idx) in enumerate(groups):
        px, pm, ph = [], [], []
        for ci, c in enumerate(cats):
            v = yy[idx][xs[idx] == c]
            if v.size == 0:
                continue
            mean = float(v.mean())
            if v.size < 2:
                half, thin = 0.0, True
            else:
                sd = float(v.std(ddof=1))
                se = sd / np.sqrt(v.size)
                half = sd if kind == "sd" else se if kind == "se" else float(stats.t.ppf((1 + level) / 2, v.size - 1)) * se
            px.append(ci + (gi - (ng - 1) / 2) * width)
            pm.append(mean)
            ph.append(half)
        col = style.color_for(lab or "", gi)
        ax.errorbar(px, pm, yerr=ph, fmt="o", color=col, ecolor=col, elinewidth=style.line_width, capsize=4,
                    markersize=style.marker_size, label=lab or None)
    if thin:
        d.notes.append("cells with a single observation have no error bar")
    ax.set_xticks(range(len(cats)), [short(c, 14) for c in cats])
    ax.set_xlim(-0.6, len(cats) - 0.4)
    what = {"t_ci": f"mean with {level * 100:g}% t interval", "sd": "mean +- SD", "se": "mean +- SE"}[kind]
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=f"{d.name('y')} ({what})", grid_axis="y")
    return ax


# --------------------------------------------------------------------------- regression
def _design(x, degree, xm, sx):
    u = (np.asarray(x, float) - xm) / sx
    return np.vander(u, degree + 1, increasing=True)


def _fit_poly(x, y, degree):
    """OLS polynomial fit on a centred / scaled x. Returns (result, xm, sx, degree)."""
    import statsmodels.api as sm
    xm, sx = float(x.mean()), float(x.std())
    sx = sx if sx > 0 else 1.0
    deg = int(min(max(degree, 1), 8, np.unique(x).size - 1))
    return sm.OLS(y, _design(x, deg, xm, sx)).fit(), xm, sx, deg


@chart("regression_plot", "Regression plot", "Statistical",
       [Role("x", "numeric", "X"), Role("y", "numeric", "Y")],
       [Param("degree", "Polynomial degree", "int", 1, min=1, max=8),
        Param("conf", "Confidence level", "float", 0.95, min=0.5, max=0.999),
        Param("show_prediction", "Show prediction band", "bool", False),
        Param("max_points", "Max points drawn", "int", 100000, min=100, max=2000000)],
       aliases=("regression", "fit plot", "trend plot"))
def regression_plot(fig, d, style, degree=1, conf=0.95, show_prediction=False, max_points=100000):
    """OLS fit (statsmodels) with a confidence band for the mean and optional prediction band for new observations."""
    x, y = numeric(d["x"]), numeric(d["y"])
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.ptp(x[m]) == 0:
        return empty_message(fig, style, "Need at least 3 points with varying x")
    conf = float(min(max(conf, 0.5), 0.999))
    xf, yf = x[m], y[m]
    res, xm, sx, deg = _fit_poly(xf, yf, degree)
    if xf.size <= deg + 1:
        return empty_message(fig, style, "Not enough points for this polynomial degree")
    grid = np.linspace(xf.min(), xf.max(), 200)
    sf = res.get_prediction(_design(grid, deg, xm, sx)).summary_frame(alpha=1 - conf)
    ax = make_axes(fig, style)
    keep = sample_mask(n, m, _maxpts(max_points))
    if keep.sum() < m.sum():
        d.notes.append(f"{int(keep.sum()):,} of {int(m.sum()):,} points drawn (seeded sample); fit uses all rows")
    ax.scatter(x[keep], y[keep], s=_pt_size(style), color=style.color_for("", 0), alpha=style.alpha * 0.7, linewidths=0,
               label="data")
    col = style.palette_colors()[4]
    if show_prediction:
        ax.fill_between(grid, sf["obs_ci_lower"], sf["obs_ci_upper"], color=col, alpha=0.10, lw=0,
                        label=f"{conf * 100:g}% prediction band")
    ax.fill_between(grid, sf["mean_ci_lower"], sf["mean_ci_upper"], color=col, alpha=0.25, lw=0,
                    label=f"{conf * 100:g}% confidence band")
    ax.plot(grid, sf["mean"], color=col, lw=style.line_width, label="fit")
    if deg == 1:
        b1 = float(res.params[1]) / sx
        b0 = float(res.params[0]) - float(res.params[1]) * xm / sx
        eq = f"y = {b0:.4g} + {b1:.4g}·x"
    else:
        eq = f"polynomial degree {deg}"
    ax.text(0.02, 0.98, f"{eq}\nR² = {res.rsquared:.4f}, n = {int(res.nobs):,}", transform=ax.transAxes, va="top",
            fontsize=style.tick_size, color=style.text_color,
            bbox=dict(facecolor=style.axes_background, edgecolor=style.grid_color, alpha=0.85))
    finish(fig, ax, style, xlabel=d.name("x"), ylabel=d.name("y"), grid_axis="both")
    return ax


@chart("residual_plots", "Residual diagnostics", "Statistical",
       [Role("x", "numeric", "X"), Role("y", "numeric", "Y")],
       [Param("degree", "Polynomial degree", "int", 1, min=1, max=8),
        Param("max_points", "Max points drawn", "int", 100000, min=100, max=2000000)],
       aliases=("residual plot", "regression diagnostics", "diagnostic plots"))
def residual_plots(fig, d, style, degree=1, max_points=100000):
    """2x2 panel: residuals vs fitted, normal Q-Q of studentised residuals, scale-location, residuals vs leverage."""
    from scipy import stats
    x, y = numeric(d["x"]), numeric(d["y"])
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4 or np.ptp(x[m]) == 0:
        return empty_message(fig, style, "Need at least 4 points with varying x")
    xf, yf = x[m], y[m]
    res, xm, sx, deg = _fit_poly(xf, yf, degree)
    p = deg + 1
    if xf.size <= p + 1:
        return empty_message(fig, style, "Not enough points for this polynomial degree")
    X = _design(xf, deg, xm, sx)
    e = np.asarray(res.resid, float)
    fitted = np.asarray(res.fittedvalues, float)
    s = float(np.sqrt(np.dot(e, e) / (xf.size - p)))
    h = np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X)
    r = e / (s * np.sqrt(np.clip(1 - h, 1e-12, None))) if s > 0 else np.zeros_like(e)
    mp = _maxpts(max_points)
    keep = sample_mask(xf.size, np.ones(xf.size, bool), mp)
    if not keep.all():
        d.notes.append(f"diagnostic panels show a seeded sample of {int(keep.sum()):,} of {xf.size:,} points")
    col = style.color_for("", 0)
    ref = style.palette_colors()[4]
    _clear_grid_fig(fig, style)
    axs = [fig.add_subplot(2, 2, k + 1) for k in range(4)]
    for a in axs:
        _prep_axis(a, style)
    sz = _pt_size(style)
    lab = dict(fontsize=style.label_size, color=style.text_color)
    a0, a1, a2, a3 = axs
    a0.scatter(fitted[keep], e[keep], s=sz, color=col, alpha=style.alpha * 0.7, linewidths=0)
    a0.axhline(0, color=ref, lw=1.2, ls="--")
    a0.set(title="Residuals vs fitted")
    a0.set_xlabel("Fitted values", **lab)
    a0.set_ylabel("Residuals", **lab)
    rs = np.sort(r[keep])
    th = stats.norm.ppf(_blom(rs.size))
    a1.scatter(th, rs, s=sz, color=col, alpha=style.alpha * 0.7, linewidths=0)
    lim = np.array([min(th.min(), rs.min()), max(th.max(), rs.max())])
    a1.plot(lim, lim, color=ref, lw=1.2, ls="--")
    a1.set(title="Normal Q-Q")
    a1.set_xlabel("Theoretical quantiles", **lab)
    a1.set_ylabel("Studentised residuals", **lab)
    a2.scatter(fitted[keep], np.sqrt(np.abs(r[keep])), s=sz, color=col, alpha=style.alpha * 0.7, linewidths=0)
    a2.set(title="Scale-location")
    a2.set_xlabel("Fitted values", **lab)
    a2.set_ylabel("sqrt(|studentised residual|)", **lab)
    a3.scatter(h[keep], r[keep], s=sz, color=col, alpha=style.alpha * 0.7, linewidths=0)
    a3.axhline(0, color=ref, lw=1.2, ls="--")
    a3.axvline(2.0 * p / xf.size, color=ref, lw=1.0, ls=":")
    a3.set(title="Residuals vs leverage")
    a3.set_xlabel("Leverage", **lab)
    a3.set_ylabel("Studentised residuals", **lab)
    for a in axs:
        a.title.set_color(style.text_color)
        a.title.set_fontsize(style.label_size)
    if style.title:
        fig.suptitle(style.title, fontsize=style.title_size, color=style.text_color)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return a0


# --------------------------------------------------------------------------- control chart
@chart("control_chart", "Control chart (individuals)", "Statistical",
       [Role("y", "numeric", "Measurement"), Role("order", "any", "Order by (blank = row order)", required=False)],
       [Param("sigma", "Limit width (sigma)", "float", 3.0, min=0.5, max=6.0)],
       aliases=("individuals chart", "I chart", "shewhart chart", "process control chart"))
def control_chart(fig, d, style, sigma=3.0):
    """Individuals chart: centre = mean, limits = mean +- sigma * MRbar / 1.128; points beyond the limits are flagged."""
    y = numeric(d["y"])
    idx = np.arange(len(y))
    if d.has("order"):
        idx = _order_index(np.asarray(d["order"])[: len(y)])
    idx = idx[np.isfinite(y[idx])]
    if idx.size < 2:
        return empty_message(fig, style, "Need at least two finite values")
    v = y[idx]
    center = float(v.mean())
    mrbar = float(np.abs(np.diff(v)).mean())
    sd = mrbar / MR_D2
    sigma = float(sigma)
    ucl, lcl = center + sigma * sd, center - sigma * sd
    t = np.arange(1, v.size + 1)
    out = (v > ucl) | (v < lcl)
    ax = make_axes(fig, style)
    col = style.color_for("", 0)
    ax.plot(t, v, "-", marker="o" if v.size <= 2000 else None, color=col, lw=style.line_width * 0.6,
            markersize=style.marker_size * 0.7, label="value")
    ax.axhline(center, color=style.palette_colors()[2], lw=style.line_width * 0.8, label=f"CL = {center:.4g}")
    ax.axhline(ucl, color=style.palette_colors()[4], lw=1.4, ls="--", label=f"UCL = {ucl:.4g}")
    ax.axhline(lcl, color=style.palette_colors()[4], lw=1.4, ls="--", label=f"LCL = {lcl:.4g}")
    if out.any():
        ax.scatter(t[out], v[out], s=_pt_size(style) * 2.2, color=style.palette_colors()[4], zorder=5, linewidths=0,
                   label=f"out of limits ({int(out.sum())})")
        d.notes.append(f"{int(out.sum())} point(s) outside the {sigma:g}-sigma limits")
    finish(fig, ax, style, xlabel="Observation", ylabel=d.name("y"), grid_axis="y")
    return ax


# --------------------------------------------------------------------------- Bland-Altman
@chart("bland_altman", "Bland-Altman plot", "Statistical",
       [Role("a", "numeric", "Method A"), Role("b", "numeric", "Method B")],
       [Param("k", "Limits (SD multiples)", "float", 1.96, min=1, max=4),
        Param("max_points", "Max points drawn", "int", 100000, min=100, max=2000000)],
       aliases=("bland altman", "mean-difference plot", "tukey mean-difference"))
def bland_altman(fig, d, style, k=1.96, max_points=100000):
    """Difference (A-B) against the mean of both methods; bias line and bias +- k*SD limits of agreement."""
    a, b = numeric(d["a"]), numeric(d["b"])
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return empty_message(fig, style, "Need at least 3 paired measurements")
    mean, diff = (a + b) / 2.0, a - b
    bias = float(diff[m].mean())
    sd = float(diff[m].std(ddof=1))
    k = float(k)
    keep = sample_mask(n, m, _maxpts(max_points))
    if keep.sum() < m.sum():
        d.notes.append(f"{int(keep.sum()):,} of {int(m.sum()):,} points drawn (seeded sample); statistics use all pairs")
    ax = make_axes(fig, style)
    ax.scatter(mean[keep], diff[keep], s=_pt_size(style), color=style.color_for("", 0), alpha=style.alpha * 0.8,
               linewidths=0, label="pairs")
    red = style.palette_colors()[4]
    ax.axhline(bias, color=style.palette_colors()[2], lw=style.line_width, label=f"bias = {bias:.4g}")
    ax.axhline(bias + k * sd, color=red, lw=1.4, ls="--", label=f"+{k:g} SD = {bias + k * sd:.4g}")
    ax.axhline(bias - k * sd, color=red, lw=1.4, ls="--", label=f"-{k:g} SD = {bias - k * sd:.4g}")
    finish(fig, ax, style, xlabel=f"Mean of {d.name('a', 'A')} and {d.name('b', 'B')}",
           ylabel=f"Difference ({d.name('a', 'A')} - {d.name('b', 'B')})", grid_axis="both")
    return ax
