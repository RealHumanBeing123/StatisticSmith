import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from matplotlib.figure import Figure

from statsmith.charts import ChartData, ChartStyle, render


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def style():
    return ChartStyle()


def draw(chart_id, cols, names=None, style=None, **opts):
    """Render a chart headlessly and return (fig, ChartData). ``cols`` maps role -> array / list of arrays."""
    d = ChartData(cols, names or {k: k for k in cols})
    fig = render(chart_id, d, style or ChartStyle(), fig=Figure(), **opts)
    return fig, d


@pytest.fixture(name="draw")
def draw_fixture():
    return draw
