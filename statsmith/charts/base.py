"""Shared contract for every chart in Statistic Smith.

A chart is a plain function registered with the ``@chart`` decorator::

    @chart("bar", "Bar chart", "Comparison",
           roles=[Role("x", "categorical", "Category"), Role("y", "numeric", "Value", required=False)],
           params=[Param("agg", "Aggregate", "choice", "sum", choices=AGGS)])
    def bar(fig, d, style, agg="sum"):
        ax = make_axes(fig, style)
        ...
        finish(fig, ax, style, xlabel=d.name("x"), ylabel=d.name("y"))

* ``fig``   - a ``matplotlib.figure.Figure`` (cleared by ``make_axes``)
* ``d``     - :class:`ChartData` (numpy arrays keyed by role name)
* ``style`` - :class:`ChartStyle` (everything the graph editor can change)

The graph editor re-calls the same function with a modified ``ChartStyle``, so
**every** visual choice must flow through ``style`` (colors, sizes, labels...).
Charts never call ``plt.show()`` and never use the pyplot state machine.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

CATEGORIES = [
    "Comparison",
    "Correlation",
    "Part-to-whole & hierarchical",
    "Data over time",
    "Distribution",
    "Geospatial & other",
    "Statistical",
]

AGGS = ("sum", "mean", "median", "count", "min", "max")

PALETTES: dict[str, list[str]] = {
    # Okabe-Ito derived, colour-blind safe, readable on light and dark backgrounds
    "Smith": ["#2E86C1", "#E67E22", "#27AE60", "#8E44AD", "#C0392B", "#16A085", "#F1C40F", "#34495E", "#E84393", "#7F8C8D"],
    "Okabe-Ito": ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9", "#F0E442", "#000000"],
    "Tableau10": ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"],
    "Pastel": ["#A6CEE3", "#B2DF8A", "#FB9A99", "#FDBF6F", "#CAB2D6", "#FFFF99", "#8DD3C7", "#FCCDE5"],
    "Set2": ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"],
    "Grayscale": ["#222222", "#555555", "#888888", "#AAAAAA", "#CCCCCC", "#EEEEEE"],
}
SEQUENTIAL_CMAPS = ["viridis", "plasma", "magma", "cividis", "Blues", "Greens", "Oranges", "Reds", "YlGnBu", "YlOrRd"]
DIVERGING_CMAPS = ["RdBu_r", "coolwarm", "PuOr", "BrBG", "PiYG", "Spectral_r"]


# --------------------------------------------------------------------------- specs
@dataclass
class Role:
    """A variable slot the user fills from the dataset.

    kind: numeric | categorical | any | datetime | text |
          numeric_multi | any_multi   (the *_multi kinds take 1..n columns)
    """
    name: str
    kind: str
    label: str = ""
    required: bool = True
    help: str = ""

    @property
    def multi(self) -> bool:
        return self.kind.endswith("_multi")


@dataclass
class Param:
    name: str
    label: str
    kind: str  # int | float | choice | bool | text | file
    default: Any = None
    choices: Sequence[Any] = ()
    min: float | None = None
    max: float | None = None
    help: str = ""


@dataclass
class ChartSpec:
    id: str
    name: str
    category: str
    roles: list[Role]
    params: list[Param]
    fn: Callable
    description: str = ""
    aliases: tuple[str, ...] = ()
    group: str = ""           # optional sub-group label inside the category menu
    needs_geometry: bool = False


REGISTRY: dict[str, ChartSpec] = {}


def chart(id: str, name: str, category: str, roles: Iterable[Role], params: Iterable[Param] = (),
          description: str = "", aliases: Iterable[str] = (), group: str = ""):
    """Register a chart function."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")

    def deco(fn):
        if id in REGISTRY:
            raise ValueError(f"duplicate chart id {id!r}")
        REGISTRY[id] = ChartSpec(id, name, category, list(roles), list(params), fn,
                                 description or (fn.__doc__ or "").strip(), tuple(aliases), group)
        return fn
    return deco


def specs_by_category() -> dict[str, list[ChartSpec]]:
    out: dict[str, list[ChartSpec]] = {c: [] for c in CATEGORIES}
    for s in REGISTRY.values():
        out[s.category].append(s)
    return out


# --------------------------------------------------------------------------- data + style
class ChartData:
    """Columns handed to a chart, keyed by role name.

    ``cols[role]`` is a 1-D numpy array (single roles) or a list of 1-D arrays (``*_multi`` roles).
    ``names[role]`` is the matching column name (or list of names).
    ``notes`` collects human-readable remarks (e.g. "sampled 200,000 of 41,000,000 rows").
    """

    def __init__(self, cols: dict[str, Any] | None = None, names: dict[str, Any] | None = None):
        self.cols: dict[str, Any] = dict(cols or {})
        self.names: dict[str, Any] = dict(names or {})
        self.notes: list[str] = []

    def __getitem__(self, role: str):
        return self.cols[role]

    def get(self, role: str, default=None):
        return self.cols.get(role, default)

    def has(self, role: str) -> bool:
        v = self.cols.get(role)
        return v is not None and (len(v) > 0 if hasattr(v, "__len__") else True)

    def name(self, role: str, default: str = "") -> str:
        n = self.names.get(role, default)
        if isinstance(n, (list, tuple)):
            return ", ".join(map(str, n))
        return str(n) if n is not None else default

    def label_list(self, role: str) -> list[str]:
        n = self.names.get(role, [])
        return list(n) if isinstance(n, (list, tuple)) else [str(n)]

    def __len__(self) -> int:
        for v in self.cols.values():
            if isinstance(v, list):
                return len(v[0]) if v else 0
            return len(v)
        return 0


@dataclass
class ChartStyle:
    title: str = ""
    subtitle: str = ""
    xlabel: str = ""          # "" -> use the chart's default label
    ylabel: str = ""
    legend_title: str = ""
    title_size: float = 14.0
    label_size: float = 11.0
    tick_size: float = 9.0
    legend_size: float = 9.0
    value_label_size: float = 8.0
    font_family: str = "DejaVu Sans"
    fig_width: float = 8.0
    fig_height: float = 5.5
    dpi: int = 100
    palette: str = "Smith"
    colors: dict = field(default_factory=dict)   # label -> colour overrides
    primary_color: str = ""                      # single-series colour override
    cmap: str = "viridis"
    diverging_cmap: str = "RdBu_r"
    background: str = "#ffffff"
    axes_background: str = "#ffffff"
    text_color: str = "#222222"
    grid_color: str = "#d9d9d9"
    show_grid: bool = True
    grid_axis: str = "y"        # x | y | both
    show_legend: bool = True
    legend_loc: str = "best"
    show_spines: bool = True
    xlim: tuple | None = None
    ylim: tuple | None = None
    xscale: str = "linear"
    yscale: str = "linear"
    line_width: float = 2.0
    marker_size: float = 6.0
    bar_width: float = 0.8
    alpha: float = 0.9
    show_values: bool = False
    value_format: str = "{:,.2f}"
    rotate_xticks: int = 0
    invert_x: bool = False
    invert_y: bool = False

    # ---- helpers
    def palette_colors(self, n: int | None = None) -> list[str]:
        base = PALETTES.get(self.palette) or PALETTES["Smith"]
        if n is None:
            return list(base)
        return [base[i % len(base)] for i in range(n)]

    def color_for(self, label: Any, index: int = 0) -> str:
        key = str(label)
        if key in self.colors:
            return self.colors[key]
        if self.primary_color and index == 0 and key not in self.colors:
            return self.primary_color
        return self.palette_colors()[index % len(self.palette_colors())]

    def series_colors(self, labels: Sequence[Any]) -> list[str]:
        return [self.color_for(l, i) for i, l in enumerate(labels)]

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["xlim"] = list(self.xlim) if self.xlim else None
        d["ylim"] = list(self.ylim) if self.ylim else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ChartStyle":
        names = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        for k in ("xlim", "ylim"):
            if kw.get(k):
                kw[k] = tuple(kw[k])
        return cls(**kw)

    @classmethod
    def for_theme(cls, theme: str) -> "ChartStyle":
        if theme == "dark":
            return cls(background="#1e1f22", axes_background="#25272b", text_color="#e6e6e6", grid_color="#3b3e44")
        return cls()


# --------------------------------------------------------------------------- drawing helpers
def make_axes(fig, style: ChartStyle, polar: bool = False, projection: str | None = None, rect=None):
    """Clear ``fig``, apply figure-level style, return a fresh Axes."""
    fig.clear()
    fig.set_size_inches(style.fig_width, style.fig_height, forward=False)
    fig.set_dpi(style.dpi)
    fig.patch.set_facecolor(style.background)
    kw: dict[str, Any] = {}
    if polar:
        kw["projection"] = "polar"
    elif projection:
        kw["projection"] = projection
    ax = fig.add_subplot(111, **kw) if rect is None else fig.add_axes(rect, **kw)
    ax.set_facecolor(style.axes_background)
    return ax


def _style_text(ax, style: ChartStyle):
    ax.tick_params(colors=style.text_color, labelsize=style.tick_size)
    for sp in ax.spines.values():
        sp.set_color(style.grid_color)
        sp.set_visible(style.show_spines)
    ax.xaxis.label.set_color(style.text_color)
    ax.yaxis.label.set_color(style.text_color)


def finish(fig, ax, style: ChartStyle, *, xlabel: str | None = None, ylabel: str | None = None,
           title: str | None = None, legend: bool | None = None, polar: bool = False,
           axes_decor: bool = True, grid_axis: str | None = None):
    """Apply labels/grid/limits/legend from ``style``; user overrides beat chart defaults."""
    import matplotlib as mpl
    mpl.rcParams["font.family"] = style.font_family if style.font_family else "DejaVu Sans"
    final_title = style.title or (title or "")
    if final_title:
        ax.set_title(final_title, fontsize=style.title_size, color=style.text_color, loc="left" if style.subtitle else "center",
                     pad=18 if style.subtitle else 6)
    if style.subtitle:
        ax.text(0.0, 1.02, style.subtitle, transform=ax.transAxes, fontsize=max(style.label_size - 1, 6),
                color=style.text_color, alpha=0.75, va="bottom")
    if axes_decor and not polar:
        xl = style.xlabel or xlabel or ""
        yl = style.ylabel or ylabel or ""
        if xl:
            ax.set_xlabel(xl, fontsize=style.label_size, color=style.text_color)
        if yl:
            ax.set_ylabel(yl, fontsize=style.label_size, color=style.text_color)
        if style.xscale != "linear":
            ax.set_xscale(style.xscale)
        if style.yscale != "linear":
            ax.set_yscale(style.yscale)
        if style.xlim:
            ax.set_xlim(style.xlim)
        if style.ylim:
            ax.set_ylim(style.ylim)
        if style.invert_x:
            ax.invert_xaxis()
        if style.invert_y:
            ax.invert_yaxis()
        gx = grid_axis or style.grid_axis
        ax.grid(False)
        if style.show_grid:
            ax.grid(True, axis=gx, color=style.grid_color, linewidth=0.8, alpha=0.9)
            ax.set_axisbelow(True)
        if style.rotate_xticks:
            for lab in ax.get_xticklabels():
                lab.set_rotation(style.rotate_xticks)
                lab.set_ha("right" if style.rotate_xticks > 0 else "left")
    _style_text(ax, style)
    handles, labels = ax.get_legend_handles_labels()
    show = style.show_legend if legend is None else (legend and style.show_legend)
    if show and handles:
        leg = ax.legend(loc=style.legend_loc, fontsize=style.legend_size, title=style.legend_title or None,
                        facecolor=style.axes_background, edgecolor=style.grid_color, labelcolor=style.text_color)
        if leg and leg.get_title():
            leg.get_title().set_color(style.text_color)
            leg.get_title().set_fontsize(style.legend_size)
    elif ax.get_legend():
        ax.get_legend().remove()
    try:
        fig.tight_layout()
    except Exception:
        pass
    return ax


def first_seen(values: Sequence[Any]) -> list:
    """Unique values in order of first appearance (stable, unlike np.unique)."""
    seen: dict = {}
    for v in values:
        if v not in seen:
            seen[v] = None
    return list(seen)


def aggregate(cat: np.ndarray, val: np.ndarray | None, agg: str = "sum", order: str = "appearance"):
    """Aggregate ``val`` by category. Returns (labels, values).

    ``val is None`` or ``agg == 'count'`` -> counts. NaNs in ``val`` are ignored.
    order: appearance | value_desc | value_asc | alpha
    """
    cat = np.asarray(cat, dtype=object)
    labels = first_seen(cat.tolist())
    idx = {l: i for i, l in enumerate(labels)}
    codes = np.fromiter((idx[c] for c in cat), dtype=np.int64, count=len(cat))
    k = len(labels)
    if val is None or agg == "count":
        out = np.bincount(codes, minlength=k).astype(float)
    else:
        v = np.asarray(val, dtype=float)
        ok = ~np.isnan(v)
        out = np.full(k, np.nan)
        for i in range(k):
            sel = v[(codes == i) & ok]
            if sel.size == 0:
                continue
            out[i] = {"sum": np.sum, "mean": np.mean, "median": np.median, "min": np.min, "max": np.max}[agg](sel)
    labels_arr = np.array(labels, dtype=object)
    if order == "value_desc":
        o = np.argsort(-np.nan_to_num(out, nan=-np.inf), kind="stable")
    elif order == "value_asc":
        o = np.argsort(np.nan_to_num(out, nan=np.inf), kind="stable")
    elif order == "alpha":
        o = np.argsort(np.array([str(x) for x in labels_arr]), kind="stable")
    else:
        o = np.arange(k)
    return [str(x) for x in labels_arr[o]], out[o]


def short(v: Any, width: int = 18) -> str:
    s = str(v)
    return s if len(s) <= width else s[: width - 1] + "…"


def fmt(v: float, style: ChartStyle) -> str:
    try:
        return style.value_format.format(v)
    except Exception:
        return str(v)


def numeric(a: Any) -> np.ndarray:
    """Coerce to float array (non-numeric -> NaN)."""
    arr = np.asarray(a)
    if arr.dtype.kind in "fiu":
        return arr.astype(float)
    if arr.dtype.kind == "b":
        return arr.astype(float)
    out = np.full(arr.shape, np.nan)
    for i, v in enumerate(arr.tolist()):
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def empty_message(fig, style: ChartStyle, msg: str):
    ax = make_axes(fig, style)
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", color=style.text_color, fontsize=style.label_size, transform=ax.transAxes)
    return ax
