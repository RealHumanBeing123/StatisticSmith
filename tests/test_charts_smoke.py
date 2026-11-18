"""Every registered chart must render with typical data, with NaNs, and with tiny/empty data without raising."""
import numpy as np
import pytest
from matplotlib.figure import Figure

from statsmith.charts import REGISTRY, ChartData, ChartStyle, render


def synth(spec, n=200, seed=0, nan=False):
    rng = np.random.default_rng(seed)
    cols, names = {}, {}
    for r in spec.roles:
        k = r.kind
        def one(kind):
            if kind.startswith("numeric"):
                a = rng.normal(50, 10, n) if r.name not in ("size", "weight", "value", "w") else rng.uniform(1, 10, n)
                if r.name in ("lat",): a = rng.uniform(25, 48, n)
                if r.name in ("lon",): a = rng.uniform(-120, -70, n)
                if nan: a[::17] = np.nan
                return a
            if kind == "datetime":
                return np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-01") + n).astype("datetime64[ns]")
            return rng.choice(["a", "b", "c", "d"], n).astype(object)
        if r.multi:
            cols[r.name] = [one(k) for _ in range(3)]
            names[r.name] = [f"{r.name}{i}" for i in range(3)]
        else:
            cols[r.name] = one(k); names[r.name] = r.name
    return ChartData(cols, names)


@pytest.mark.parametrize("cid", sorted(REGISTRY))
def test_chart_renders(cid):
    spec = REGISTRY[cid]
    for nan in (False, True):
        fig = render(cid, synth(spec, nan=nan), ChartStyle(), fig=Figure())
        assert fig.axes, cid
    fig = render(cid, synth(spec, n=3), ChartStyle(), fig=Figure())          # tiny
    for dark in (True,):
        render(cid, synth(spec), ChartStyle.for_theme("dark"), fig=Figure())
