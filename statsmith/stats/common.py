"""Helpers shared by the analysis modules."""
from __future__ import annotations

import numpy as np
from scipy import stats

from ..charts.base import AGGS, ChartData, Param, Role  # noqa: F401
from .base import Result, clean, clean_pair, fmt_p  # noqa: F401

ALT = Param("alternative", "Alternative hypothesis", "choice", "two-sided", choices=("two-sided", "less", "greater"))
CONF = Param("conf", "Confidence level", "float", 0.95, min=0.5, max=0.9999)
ALPHA = Param("alpha", "Significance level (α)", "float", 0.05, min=0.0001, max=0.5)

Y = Role("y", "numeric", "Response / variable")
X = Role("x", "numeric", "Variable")
GROUP = Role("group", "categorical", "Group")
XS = Role("xs", "numeric_multi", "Variables")


def groups_of(y, g, min_n: int = 1) -> dict[str, np.ndarray]:
    """{label: clean numeric array} in order of first appearance."""
    y = np.asarray(y, dtype=float)
    g = np.asarray(g, dtype=object)
    out: dict[str, np.ndarray] = {}
    order: list[str] = []
    seen = set()
    for v in g.tolist():
        k = "(missing)" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
        if k not in seen:
            seen.add(k)
            order.append(k)
    keys = np.array(["(missing)" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v) for v in g.tolist()], dtype=object)
    for k in order:
        if k == "(missing)":
            continue
        a = y[keys == k]
        a = a[np.isfinite(a)]
        if a.size >= min_n:
            out[k] = a
    return out


def need(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)


def two_groups(d: ChartData, what: str = "this test"):
    gs = groups_of(d["y"], d["group"])
    need(len(gs) == 2, f"{what} needs a grouping variable with exactly 2 groups (found {len(gs)}: "
                       f"{', '.join(list(gs)[:6])}{'…' if len(gs) > 6 else ''})")
    (a, x), (b, y) = list(gs.items())
    return a, x, b, y


def sig_note(p: float, alpha: float = 0.05) -> str:
    return (f"p = {fmt_p(p)} < α = {alpha}: reject H0" if p < alpha else f"p = {fmt_p(p)} ≥ α = {alpha}: fail to reject H0")


def alt_label(alt: str, name: str, null: str) -> str:
    sym = {"two-sided": "≠", "less": "<", "greater": ">"}[alt]
    return f"H0: {name} = {null}   vs   H1: {name} {sym} {null}"


def p_from_t(t: float, df: float, alt: str) -> float:
    if alt == "two-sided":
        return float(2 * stats.t.sf(abs(t), df))
    return float(stats.t.cdf(t, df) if alt == "less" else stats.t.sf(t, df))


def p_from_z(z: float, alt: str) -> float:
    if alt == "two-sided":
        return float(2 * stats.norm.sf(abs(z)))
    return float(stats.norm.cdf(z) if alt == "less" else stats.norm.sf(z))


def ci_t(mean: float, se: float, df: float, conf: float, alt: str = "two-sided"):
    if alt == "two-sided":
        q = stats.t.ppf(1 - (1 - conf) / 2, df)
        return mean - q * se, mean + q * se
    q = stats.t.ppf(conf, df)
    return (-np.inf, mean + q * se) if alt == "less" else (mean - q * se, np.inf)


def rank_data(a):
    return stats.rankdata(a, method="average")


def tie_term(a) -> float:
    """sum(t^3 - t) over tie groups."""
    _, c = np.unique(a, return_counts=True)
    return float(np.sum(c ** 3 - c))
