"""Chart library: registry + renderer.

Importing this package registers every chart module. Use :func:`render` to draw a chart
onto a matplotlib ``Figure``.
"""
from __future__ import annotations

from .base import (CATEGORIES, REGISTRY, ChartData, ChartSpec, ChartStyle, Param, Role,  # noqa: F401
                   specs_by_category)

from . import (comparison_a, comparison_b, correlation, statistical, distribution,  # noqa: F401,E402
               partwhole, temporal, geospatial)


def render(chart_id: str, data: ChartData, style: ChartStyle | None = None, fig=None, **opts):
    """Draw ``chart_id`` onto ``fig`` (created if None). Returns the Figure."""
    from matplotlib.figure import Figure
    spec = REGISTRY[chart_id]
    style = style or ChartStyle()
    fig = fig or Figure(figsize=(style.fig_width, style.fig_height), dpi=style.dpi)
    defaults = {p.name: p.default for p in spec.params}
    defaults.update(opts)
    spec.fn(fig, data, style, **defaults)
    return fig
