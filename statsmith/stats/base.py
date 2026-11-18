"""Registry + result container for statistical analyses.

An analysis is a function ``fn(d: ChartData, **params) -> Result`` registered with ``@analysis``. It uses the same
``Role`` / ``Param`` / ``ChartData`` objects as charts so the UI can build one dialog for both.

Every analysis also declares ``code``: templates producing equivalent Stata / R / Python so results can be
cross-checked in other packages (this is what the .do-file generator writes).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from ..charts.base import ChartData, Param, Role  # re-exported for convenience

__all__ = ["ChartData", "Param", "Role", "Result", "AnalysisSpec", "REGISTRY", "analysis", "by_category",
           "clean_pair", "clean", "fmt_p", "fnum", "table_text", "CATEGORY_ORDER"]

CATEGORY_ORDER = [
    "Summary & descriptive",
    "Means & location (parametric)",
    "Nonparametric & rank tests",
    "Categorical & contingency",
    "Normality & distribution fit",
    "Variance & ANOVA",
    "Correlation & association",
    "Regression & modelling",
    "Multivariate",
    "Time series & quality",
]


DEFAULT_DECIMALS = [4]          # the Settings dialog changes this (text results use it)


@dataclass
class Result:
    title: str
    sections: list = field(default_factory=list)       # (heading, str) text blocks / tables, in order
    values: dict = field(default_factory=dict)         # machine-readable numbers (used by tests and the terminal)
    notes: list = field(default_factory=list)
    decimals: int = field(default_factory=lambda: DEFAULT_DECIMALS[0])

    # -- builders
    def text(self, heading: str, body: str) -> "Result":
        self.sections.append((heading, body))
        return self

    def table(self, heading: str, headers: list[str], rows: list[list], decimals: int | None = None) -> "Result":
        self.sections.append((heading, table_text(headers, rows, decimals or self.decimals)))
        return self

    def kv(self, heading: str, pairs: Iterable[tuple[str, Any]], decimals: int | None = None) -> "Result":
        dec = decimals or self.decimals
        pairs = list(pairs)
        w = max((len(str(k)) for k, _ in pairs), default=0)
        body = "\n".join(f"  {str(k):<{w}}  {fnum(v, dec)}" for k, v in pairs)
        self.sections.append((heading, body))
        return self

    def note(self, s: str) -> "Result":
        self.notes.append(s)
        return self

    def to_text(self) -> str:
        bar = "=" * max(60, len(self.title) + 4)
        out = [bar, f"  {self.title}", bar]
        for h, b in self.sections:
            out.append("")
            if h:
                out.append(h)
                out.append("-" * len(h))
            out.append(b.rstrip("\n"))
        if self.notes:
            out += ["", "Notes", "-----"] + [f"  * {n}" for n in self.notes]
        return "\n".join(out) + "\n"

    def __str__(self) -> str:
        return self.to_text()


@dataclass
class AnalysisSpec:
    id: str
    name: str
    category: str
    roles: list[Role]
    params: list[Param]
    fn: Callable
    code: dict                       # {"stata": fmt, "r": fmt, "python": fmt}  (callables or format strings)
    description: str = ""
    reference: str = ""              # where the maths comes from (shown in the help / README)


REGISTRY: dict[str, AnalysisSpec] = {}


def analysis(id: str, name: str, category: str, roles: Iterable[Role], params: Iterable[Param] = (),
             code: dict | None = None, reference: str = "", description: str = ""):
    if category not in CATEGORY_ORDER:
        raise ValueError(f"unknown category {category!r}")

    def deco(fn):
        if id in REGISTRY:
            raise ValueError(f"duplicate analysis id {id!r}")
        REGISTRY[id] = AnalysisSpec(id, name, category, list(roles), list(params), fn, code or {},
                                    description or (fn.__doc__ or "").strip().split("\n")[0], reference)
        return fn
    return deco


def by_category() -> dict[str, list[AnalysisSpec]]:
    out: dict[str, list[AnalysisSpec]] = {c: [] for c in CATEGORY_ORDER}
    for s in REGISTRY.values():
        out[s.category].append(s)
    return out


# --------------------------------------------------------------------------- data helpers
def clean(a) -> np.ndarray:
    """Finite floats only."""
    x = np.asarray(a, dtype=float).ravel()
    return x[np.isfinite(x)]


def clean_pair(a, b) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise-complete (listwise) finite pairs."""
    x, y = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "NaN"
    if p < 1e-4:
        return f"{p:.3e}"
    return f"{p:.4f}"


def fnum(v: Any, dec: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Inf" if v > 0 else "-Inf"
        if v != 0 and (abs(v) < 10 ** -dec or abs(v) >= 1e9):
            return f"{v:.{dec}e}"
        if float(v).is_integer() and abs(v) < 1e12:
            return f"{v:,.0f}"
        return f"{v:,.{dec}f}"
    return str(v)


def table_text(headers: list[str], rows: list[list], dec: int = 4) -> str:
    cells = [[fnum(c, dec) for c in r] for r in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in cells)) if cells else len(str(h)) for i, h in enumerate(headers)]
    line = "  ".join(f"{str(h):>{w}}" if i else f"{str(h):<{w}}" for i, (h, w) in enumerate(zip(headers, widths)))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(f"{c:>{w}}" if i else f"{c:<{w}}" for i, (c, w) in enumerate(zip(r, widths))) for r in cells]
    return "\n".join([line, sep] + body)
