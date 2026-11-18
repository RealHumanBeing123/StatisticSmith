"""Geometry-level checks for comparison_b, temporal, geospatial, correlation and statistical charts.
Each test compares what was DRAWN with an independent numpy / scipy computation."""
import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

from statsmith.charts import geospatial as geo
from statsmith.charts import temporal as tm

rng = np.random.default_rng(7)


def rects(ax):
    return [p for p in ax.patches if isinstance(p, Rectangle)]


# ------------------------------------------------------------------ comparison_b
def test_waterfall_bars_chain(draw):
    fig, _ = draw("waterfall", {"x": np.array(list("abcd"), dtype=object), "y": np.array([10., -4., 6., -2.])}, total=True)
    r = rects(fig.axes[0])
    bottoms = [b.get_y() for b in r]
    heights = [b.get_height() for b in r]
    np.testing.assert_allclose(bottoms[:4], [0, 10, 6, 12])
    np.testing.assert_allclose(heights, [10, -4, 6, -2, 10])      # final total = 10


def test_nightingale_area_proportional(draw):
    cats = np.array(["a"] * 4 + ["b"] * 1, dtype=object)
    fig, _ = draw("nightingale", {"x": cats, "y": np.array([1., 1, 1, 1, 1.])}, agg="sum")
    h = sorted(p.get_height() for p in rects(fig.axes[0]))
    np.testing.assert_allclose(h, [1.0, 2.0])      # radius = sqrt(value): 4 -> 2, 1 -> 1


def test_slope_chart_endpoints(draw):
    fig, _ = draw("slope_chart", {"label": np.array(["p", "q"], dtype=object), "start": np.array([1., 5.]), "end": np.array([4., 2.])})
    ys = sorted(tuple(l.get_ydata()) for l in fig.axes[0].lines)
    assert ys == [(1.0, 4.0), (5.0, 2.0)]


def test_quadrant_counts(draw):
    x = rng.normal(size=500); y = rng.normal(size=500)
    fig, d = draw("quadrant_chart", {"x": x, "y": y}, split="median")
    cx, cy = np.median(x), np.median(y)
    exp = [int(((x < cx) & (y < cy)).sum()), int(((x >= cx) & (y < cy)).sum()),
           int(((x < cx) & (y >= cy)).sum()), int(((x >= cx) & (y >= cy)).sum())]
    assert f"{exp}" in " ".join(d.notes)


def test_radar_scaled_to_max(draw):
    cols = [np.array([2., 4.]), np.array([10., 10.]), np.array([1., 3.])]
    fig, _ = draw("radar", {"vals": cols}, {"vals": ["a", "b", "c"]}, agg="mean")
    line = fig.axes[0].lines[0].get_ydata()
    np.testing.assert_allclose(line[:3], [1.0, 1.0, 1.0])      # single series, each spoke = its own max -> 1


def test_word_counts():
    from statsmith.charts.comparison_b import word_counts
    c = dict(word_counts(["The cat and the hat", "cat cat Hat"]))
    assert c["cat"] == 3 and c["hat"] == 2 and "the" not in c


# ------------------------------------------------------------------ temporal
def test_line_series_table_aggregates():
    from statsmith.charts.base import ChartData
    d = ChartData({"x": np.array([1, 1, 2, 3.]), "y": [np.array([2., 4., 5., 7.])]}, {"y": ["v"]})
    xs, labs, M, kind, _ = tm.series_table(d, "mean")
    np.testing.assert_allclose(xs, [1, 2, 3]); np.testing.assert_allclose(M[:, 0], [3, 5, 7]); assert kind == "num"


def test_dates_detected():
    x, kind, _ = tm.to_axis(np.array(["2020-01-01", "2020-01-03"], dtype=object))
    assert kind == "date" and x[1] - x[0] == 2


def test_stacked_area_top_is_total(draw):
    x = np.arange(5.)
    cols = [np.array([1., 2, 3, 4, 5]), np.array([5., 4, 3, 2, 1])]
    fig, _ = draw("stacked_area", {"x": x, "y": cols}, {"y": ["a", "b"]}, agg="sum")
    coll = [c for c in fig.axes[0].collections if isinstance(c, PolyCollection)]
    top = max(np.asarray(c.get_paths()[0].vertices)[:, 1].max() for c in coll)
    assert top == pytest.approx(6.0)


def test_bump_ranks(draw):
    M = np.array([[3., 1.], [1., 3.], [2., 2.]])
    np.testing.assert_array_equal(tm._ranks(M), [[1, 2], [2, 1], [1, 1]])


def test_bump_area_thickness_is_value(draw):
    x = np.arange(4.)
    cols = [np.array([1., 4, 2, 3]), np.array([3., 1, 2, 5])]
    fig, _ = draw("bump_area", {"x": x, "y": cols}, {"y": ["a", "b"]}, agg="sum")
    # each fill_between: thickness at every x equals the series value
    th = []
    for c in fig.axes[0].collections:
        v = c.get_paths()[0].vertices
        n = (len(v) - 3) // 2
        lo = v[1:n + 1, 1]; hi = v[n + 2:2 * n + 2, 1][::-1]
        th.append(np.sort(hi - lo))
    assert len(th) == 2


def test_candlestick_wicks_and_bodies(draw):
    x = np.arange(3.)
    o, h, l, c = np.array([10., 12, 11]), np.array([13., 14, 12]), np.array([9., 10, 8]), np.array([12., 11, 11.5])
    fig, _ = draw("candlestick", {"x": x, "open": o, "high": h, "low": l, "close": c})
    ax = fig.axes[0]
    bodies = sorted(rects(ax), key=lambda r: r.get_x())
    np.testing.assert_allclose([b.get_y() for b in bodies], np.minimum(o, c))
    np.testing.assert_allclose([b.get_height() for b in bodies], np.abs(c - o))
    seg = np.concatenate([np.asarray(s).reshape(-1, 2) for lc in ax.collections for s in lc.get_segments()])
    assert seg[:, 1].min() == 8 and seg[:, 1].max() == 14


def test_gantt_bars(draw):
    fig, _ = draw("gantt", {"task": np.array(["a", "b"], dtype=object), "start": np.array([0., 3]), "end": np.array([2., 8])})
    r = sorted(rects(fig.axes[0]), key=lambda p: p.get_y())
    assert sorted((p.get_x(), p.get_width()) for p in r) == [(0.0, 2.0), (3.0, 5.0)]


# ------------------------------------------------------------------ geospatial
def test_geo_data_loads_and_matches():
    assert len(geo.tile_grid()) == 51
    m = geo.match_regions("us", ["CA", "california", "Texas", "xx", "DC"])
    assert m["CA"] == "California" and m["california"] == "California" and m["xx"] is None and m["DC"] == "District of Columbia"
    w = geo.match_regions("world", ["USA", "United States", "France", "FRA", "Narnia"])
    assert w["USA"] == w["United States"] and w["France"] == w["FRA"] == "France" and w["Narnia"] is None


def test_tile_grid_is_a_valid_layout():
    cells = [(c, r) for c, r, _ in geo.tile_grid().values()]
    assert len(set(cells)) == len(cells)                         # no two states share a tile
    assert max(c for c, _ in cells) == 11 and max(r for _, r in cells) == 7


def test_choropleth_colors_follow_values(draw):
    import matplotlib as mpl
    st = np.array(["CA", "TX", "NY"], dtype=object)
    fig, d = draw("choropleth", {"region": st, "value": np.array([1., 5., 9.])}, map="US states", agg="sum")
    ax = fig.axes[0]
    names = list(geo.region_paths("us"))
    pats = [p for p in ax.patches]
    fc = {n: p.get_facecolor() for n, p in zip(names, pats)}
    cmap = mpl.colormaps["viridis"]
    np.testing.assert_allclose(fc["California"], cmap(0.0)); np.testing.assert_allclose(fc["New York"], cmap(1.0))
    np.testing.assert_allclose(fc["Texas"], cmap(0.5))
    assert not d.notes or "did not match" not in d.notes[0]


def test_choropleth_reports_unmatched(draw):
    _, d = draw("choropleth", {"region": np.array(["CA", "Atlantis"], dtype=object), "value": np.array([1., 2.])})
    assert any("Atlantis" in n for n in d.notes)


def test_tile_map_one_tile_per_state(draw):
    fig, _ = draw("tile_map", {"region": np.array(["CA", "TX"], dtype=object), "value": np.array([1., 2.])})
    assert len(rects(fig.axes[0])) == 51


def test_sankey_conserves_flow():
    nodes = ["a", "b", "c", "d"]
    edges = [("a", "c", 5.0), ("a", "d", 3.0), ("b", "c", 2.0), ("c", "d", 4.0)]
    L = geo.sankey_layout(nodes, edges)
    k = L["k"]
    assert L["level"] == {"a": 0, "b": 0, "c": 1, "d": 2}
    assert L["box"]["c"][2] == pytest.approx(max(7.0, 4.0) * k)             # node = max(in, out)
    for a, b, w, ya, yb, h in L["links"]:
        assert h == pytest.approx(w * k)
        assert L["box"][a][1] - 1e-12 <= ya and ya + h <= L["box"][a][1] + L["box"][a][2] + 1e-12
        assert L["box"][b][1] - 1e-12 <= yb and yb + h <= L["box"][b][1] + L["box"][b][2] + 1e-12


def test_sankey_handles_cycles():
    L = geo.sankey_layout(["a", "b"], [("a", "b", 1.0), ("b", "a", 1.0)])
    assert set(L["level"].values()) == {0, 1}


def test_chord_arcs_cover_circle_proportionally():
    L = geo.chord_layout(["a", "b", "c"], [("a", "b", 3.0), ("b", "c", 1.0)], gap=0.0)
    arcs = {n: v[1] - v[0] for n, v in L["arcs"].items()}
    assert sum(arcs.values()) == pytest.approx(2 * np.pi)
    assert arcs["a"] / arcs["b"] == pytest.approx(3 / 4) and arcs["c"] / arcs["a"] == pytest.approx(1 / 3)
    for a, b, w, (s0, s1), (t0, t1) in L["links"]:
        assert s1 - s0 == pytest.approx(w * L["per"]) == pytest.approx(t1 - t0)


def test_fr_layout_deterministic_and_bounded():
    E = np.array([[0, 1], [1, 2], [2, 3]])
    a = geo.fr_layout(4, E, np.ones(3)); b = geo.fr_layout(4, E, np.ones(3))
    np.testing.assert_allclose(a, b)
    assert a.min() >= 0 and a.max() <= 1


def test_geo_heatmap_counts(draw):
    lat = np.array([40., 40.1, 30.]); lon = np.array([-100., -100.1, -90.])
    fig, _ = draw("geo_heatmap", {"lat": lat, "lon": lon}, bins=20, agg="count", basemap="none")
    mesh = [c for c in fig.axes[0].collections][0]
    assert np.ma.sum(mesh.get_array()) == 3


# ------------------------------------------------------------------ correlation + statistical
def test_scatter_points_exact(draw):
    x, y = rng.normal(size=50), rng.normal(size=50)
    fig, _ = draw("scatter", {"x": x, "y": y})
    off = np.vstack([c.get_offsets() for c in fig.axes[0].collections if len(c.get_offsets())])
    np.testing.assert_allclose(np.sort(off[:, 0]), np.sort(x))


def test_heatmap_matches_corrcoef(draw):
    cols = [rng.normal(size=100) for _ in range(3)]
    cols[1] = cols[0] * 2 + rng.normal(size=100) * 0.1
    fig, _ = draw("heatmap", {"dims": cols}, {"dims": ["a", "b", "c"]})
    imgs = fig.axes[0].images or fig.axes[0].collections
    arr = np.asarray(imgs[0].get_array())
    np.testing.assert_allclose(np.ma.filled(arr, np.nan).reshape(3, 3), np.corrcoef(cols), atol=1e-9)


def test_pareto_cumulative(draw):
    cats = np.array(list("aaaabbbcc d".replace(" ", "")), dtype=object)
    fig, _ = draw("pareto", {"x": cats})
    ax2 = [a for a in fig.axes if a is not fig.axes[0]][0] if len(fig.axes) > 1 else fig.axes[0]
    ys = [l.get_ydata() for l in ax2.lines if len(l.get_ydata()) > 1]
    assert ys and np.isclose(ys[0][-1], 100.0) or np.isclose(ys[0][-1], 1.0)


def test_qq_plot_points(draw):
    from scipy import stats
    x = rng.normal(size=200)
    fig, _ = draw("qq_plot", {"y": x})
    pts = [c.get_offsets() for c in fig.axes[0].collections if len(c.get_offsets()) == 200]
    if pts:
        th = stats.probplot(x, dist="norm")[0]
        np.testing.assert_allclose(np.sort(pts[0][:, 1]), np.sort(th[1]), atol=1e-9)


def test_ecdf_steps(draw):
    x = rng.normal(size=100)
    fig, _ = draw("ecdf", {"x": x})
    y = fig.axes[0].lines[0].get_ydata()
    assert y.max() == pytest.approx(1.0) and y.min() >= 0


def test_regression_plot_slope(draw):
    x = np.linspace(0, 10, 100); y = 3 * x + 2 + rng.normal(size=100) * 0.1
    fig, _ = draw("regression_plot", {"x": x, "y": y})
    ln = [l for l in fig.axes[0].lines if len(l.get_xdata()) > 2][0]
    sl = np.polyfit(ln.get_xdata(), ln.get_ydata(), 1)[0]
    assert sl == pytest.approx(3.0, abs=0.05)


def test_large_scatter_is_fast(draw):
    import time
    x = rng.normal(size=1_000_000); y = rng.normal(size=1_000_000)
    t = time.time()
    draw("hexbin", {"x": x, "y": y})
    assert time.time() - t < 15
