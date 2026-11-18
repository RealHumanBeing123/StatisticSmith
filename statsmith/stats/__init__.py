"""Statistical analyses. Importing this package registers every analysis; use :func:`run` to execute one."""
from __future__ import annotations

from .base import CATEGORY_ORDER, REGISTRY, AnalysisSpec, Result, by_category  # noqa: F401
# plain static imports (PyInstaller can see them; a missing module is a loud error, never silently skipped)
from . import (descriptive, location, nonparametric, categorical, normality, anova, regression,  # noqa: F401,E402
               multivariate, timeseries)


def run(analysis_id: str, cols: dict, names: dict | None = None, **params) -> Result:
    """Run ``analysis_id`` on ``cols`` ({role: array | [arrays]}). Defaults come from the spec's params."""
    from ..charts.base import ChartData
    spec = REGISTRY[analysis_id]
    kw = {p.name: p.default for p in spec.params}
    kw.update(params)
    if names is None:
        names = {k: ([f"{k}{i + 1}" for i in range(len(v))] if isinstance(v, list) else k) for k, v in cols.items()}
    return spec.fn(ChartData(cols, names), **kw)
