"""Geometry-level verification of comparison charts (part A).

Tests read the *drawn artists* (bar heights, wedge angles, ...) and compare them with numbers computed
independently with numpy, so a chart that looks plausible but plots wrong values fails.
"""
import numpy as np
import pytest

from tests.conftest import draw


def test_column_heights_match_group_sums():
    x = np.array(list("aabbbcc"), dtype=object)
    y = np.array([1, 2, 3, 4, 5, 6, 7], float)
    fig, _ = draw("column", {"x": x, "y": y}, agg="sum")
    ax = fig.axes[0]
    assert [round(p.get_height(), 9) for p in ax.patches] == [3.0, 12.0, 13.0]


def test_bar_mean_sorted_descending():
    x = np.array(list("aabbbcc"), dtype=object)
    y = np.array([1, 2, 3, 4, 5, 6, 7], float)
    fig, _ = draw("bar", {"x": x, "y": y}, agg="mean", sort="value_desc")
    ax = fig.axes[0]
    widths = [p.get_width() for p in ax.patches]
    assert widths == pytest.approx([6.5, 4.0, 1.5])


def test_bar_counts_when_no_value_column():
    x = np.array(list("aabbbc"), dtype=object)
    fig, _ = draw("column", {"x": x}, agg="sum")
    assert [p.get_height() for p in fig.axes[0].patches] == [2, 3, 1]


def test_style_overrides_reach_the_chart():
    from statsmith.charts import ChartStyle
    x = np.array(list("abc"), dtype=object)
    y = np.array([1.0, 2.0, 3.0])
    st = ChartStyle(title="My title", xlabel="X!", ylabel="Y!", primary_color="#ff0000", title_size=20)
    fig, _ = draw("column", {"x": x, "y": y}, style=st)
    ax = fig.axes[0]
    assert ax.get_title() == "My title" and ax.get_xlabel() == "X!" and ax.get_ylabel() == "Y!"
    assert ax.title.get_fontsize() == 20
    assert ax.patches[0].get_facecolor()[:3] == pytest.approx((1.0, 0.0, 0.0))


# =========================================================================== part A, batch 2
from matplotlib.patches import Polygon as _Poly  # noqa: E402

CATS = np.array(list("aabbbccaabc"), dtype=object)
GRPS = np.array(["g1", "g2", "g1", "g1", "g2", "g2", "g1", "g1", "g2", "g2", "g1"], dtype=object)
VALS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], float)
CAT_LIST, GRP_LIST = ["a", "b", "c"], ["g1", "g2"]
_FUNCS = {"sum": np.sum, "mean": np.mean, "median": np.median, "min": np.min, "max": np.max}


def _expected_cells(agg):
    M = np.full((3, 2), np.nan)
    for i, c in enumerate(CAT_LIST):
        for j, g in enumerate(GRP_LIST):
            sel = (CATS == c) & (GRPS == g)
            if sel.any():
                M[i, j] = sel.sum() if agg == "count" else _FUNCS[agg](VALS[sel])
    return M


@pytest.mark.parametrize("agg", ["sum", "mean", "median", "min", "max", "count"])
def test_grouped_column_heights_match_cells(agg):
    fig, _ = draw("grouped_column", {"x": CATS, "group": GRPS, "y": VALS}, agg=agg)
    ax = fig.axes[0]
    M = _expected_cells(agg)
    h = np.array([p.get_height() for p in ax.patches]).reshape(2, 3).T  # series-major drawing order
    np.testing.assert_allclose(h, M, rtol=1e-12)
    assert len(ax.patches) == 6
    assert [t.get_text() for t in ax.get_xticklabels()] == CAT_LIST
    assert [t.get_text() for t in ax.get_legend().get_texts()] == GRP_LIST


def test_grouped_bar_widths_and_offsets():
    fig, _ = draw("grouped_bar", {"x": CATS, "group": GRPS, "y": VALS}, agg="sum")
    ax = fig.axes[0]
    M = _expected_cells("sum")
    w = np.array([p.get_width() for p in ax.patches]).reshape(2, 3).T
    np.testing.assert_allclose(w, M)
    # the two bars of one cluster sit side by side, centred on the category slot
    cy = np.array([p.get_y() + p.get_height() / 2 for p in ax.patches]).reshape(2, 3).T
    np.testing.assert_allclose(cy.mean(axis=1), [0, 1, 2], atol=1e-12)
    assert np.all(cy[:, 0] != cy[:, 1])


def test_grouped_count_without_value_column():
    fig, _ = draw("grouped_column", {"x": CATS, "group": GRPS})
    h = np.array([p.get_height() for p in fig.axes[0].patches]).reshape(2, 3).T
    np.testing.assert_allclose(h, _expected_cells("count"))


@pytest.mark.parametrize("orient", ["horizontal", "vertical"])
def test_lollipop_stem_lengths_equal_values(orient):
    fig, _ = draw("lollipop", {"x": CATS, "y": VALS}, agg="sum", orientation=orient)
    ax = fig.axes[0]
    exp = np.array([VALS[CATS == c].sum() for c in CAT_LIST])
    segs = np.array(ax.collections[0].get_segments())  # (n, 2, 2)
    k = 0 if orient == "horizontal" else 1  # value axis
    np.testing.assert_allclose(segs[:, 0, k], 0.0)
    np.testing.assert_allclose(segs[:, 1, k] - segs[:, 0, k], exp)
    dots = ax.lines[0]
    np.testing.assert_allclose(dots.get_xdata() if k == 0 else dots.get_ydata(), exp)


def test_lollipop_sorted_desc():
    fig, _ = draw("lollipop", {"x": CATS, "y": VALS}, agg="sum", sort="value_desc")
    exp = sorted((VALS[CATS == c].sum() for c in CAT_LIST), reverse=True)
    np.testing.assert_allclose(fig.axes[0].lines[0].get_xdata(), exp)


def _bullet_data():
    return {"label": np.array(["p", "q", "r"], dtype=object), "actual": np.array([70.0, 40.0, 95.0]),
            "target": np.array([80.0, 50.0, 90.0]),
            "ranges": [np.array([60.0, 30.0, 50.0]), np.array([90.0, 70.0, 80.0]), np.array([100.0, 100.0, 120.0])]}


def test_bullet_actual_target_and_ranges():
    cols = _bullet_data()
    fig, _ = draw("bullet", cols)
    ax = fig.axes[0]
    actual = [p.get_width() for p in ax.patches if p.get_gid() == "actual"]
    np.testing.assert_allclose(actual, cols["actual"])
    tx = [ln.get_xdata()[0] for ln in ax.lines if ln.get_gid() == "target"]
    np.testing.assert_allclose(tx, cols["target"])
    for ln in ax.lines:
        if ln.get_gid() == "target":
            assert ln.get_xdata()[0] == ln.get_xdata()[1]
    R = np.sort(np.column_stack(cols["ranges"]), axis=1)
    for j in range(3):
        w = [p.get_width() for p in ax.patches if p.get_gid() == f"band{j}"]
        np.testing.assert_allclose(w, R[:, j])
    assert max(p.get_width() for p in ax.patches) == pytest.approx(120.0)


def test_bullet_without_target_or_ranges():
    fig, _ = draw("bullet", {"label": np.array(["p", "q"], dtype=object), "actual": np.array([3.0, 4.0])})
    ax = fig.axes[0]
    assert [p.get_width() for p in ax.patches] == [3.0, 4.0]
    assert not [ln for ln in ax.lines if ln.get_gid() == "target"]


def test_dumbbell_endpoints_equal_start_end():
    lab = np.array(["u", "v", "w", "u"], dtype=object)
    s = np.array([1.0, 5.0, 3.0, 3.0])
    e = np.array([4.0, 2.0, 9.0, 5.0])
    fig, _ = draw("dumbbell", {"label": lab, "start": s, "end": e})
    ax = fig.axes[0]
    exp_s, exp_e = np.array([2.0, 5.0, 3.0]), np.array([4.5, 2.0, 9.0])
    segs = np.array(ax.collections[0].get_segments())
    np.testing.assert_allclose(segs[:, 0, 0], exp_s)
    np.testing.assert_allclose(segs[:, 1, 0], exp_e)
    by = {ln.get_label(): ln for ln in ax.lines}
    np.testing.assert_allclose(by["start"].get_xdata(), exp_s)
    np.testing.assert_allclose(by["end"].get_xdata(), exp_e)
    np.testing.assert_allclose(by["end"].get_ydata(), [0, 1, 2])


def test_dumbbell_gap_sort():
    lab = np.array(["u", "v", "w"], dtype=object)
    s, e = np.array([1.0, 5.0, 3.0]), np.array([2.0, 0.0, 9.0])
    fig, _ = draw("dumbbell", {"label": lab, "start": s, "end": e}, sort="gap_desc")
    by = {ln.get_label(): ln for ln in fig.axes[0].lines}
    np.testing.assert_allclose(by["start"].get_xdata(), [3.0, 5.0, 1.0])
    np.testing.assert_allclose(by["end"].get_xdata(), [9.0, 0.0, 2.0])


def test_range_plot_bar_extents():
    lab = np.array(["u", "v", "w"], dtype=object)
    lo, hi = np.array([1.0, -2.0, 5.0]), np.array([4.0, 3.0, 5.5])
    fig, _ = draw("range_plot", {"label": lab, "low": lo, "high": hi})
    ax = fig.axes[0]
    np.testing.assert_allclose([p.get_x() for p in ax.patches], lo)
    np.testing.assert_allclose([p.get_x() + p.get_width() for p in ax.patches], hi)
    np.testing.assert_allclose([p.get_width() for p in ax.patches], hi - lo)
    assert len(ax.patches) == 3


def test_range_plot_arrows_optional():
    lab = np.array(["u", "v"], dtype=object)
    cols = {"label": lab, "low": np.array([1.0, 2.0]), "high": np.array([3.0, 6.0])}
    n0 = len(draw("range_plot", cols)[0].axes[0].texts)
    n1 = len(draw("range_plot", cols, show_arrow=True)[0].axes[0].texts)
    assert n1 == n0 + 2


def test_dot_plot_positions_sorted_desc():
    fig, _ = draw("dot_plot", {"x": CATS, "y": VALS}, agg="mean")
    exp = sorted((VALS[CATS == c].mean() for c in CAT_LIST), reverse=True)
    dots = [ln for ln in fig.axes[0].lines if ln.get_marker() == "o"][0]
    np.testing.assert_allclose(dots.get_xdata(), exp)
    np.testing.assert_allclose(dots.get_ydata(), [0, 1, 2])


def test_dot_plot_groups():
    fig, _ = draw("dot_plot", {"x": CATS, "y": VALS, "group": GRPS}, agg="sum", sort="appearance")
    M = _expected_cells("sum")
    dots = {ln.get_label(): ln for ln in fig.axes[0].lines if ln.get_marker() == "o"}
    assert set(dots) == set(GRP_LIST)
    for j, g in enumerate(GRP_LIST):
        np.testing.assert_allclose(dots[g].get_xdata(), M[:, j])


def _icons(ax):
    return [p for p in ax.patches if isinstance(p, _Poly) and (p.get_gid() or "").startswith("icon")]


def _clip_width(p, ax):
    """Width (data units) of the patch's clip region, or None if unclipped."""
    cp = p.get_clip_path()
    if cp is not None:
        ext = cp.get_fully_transformed_path().get_extents()
    elif p.get_clip_box() is not None and p.get_clip_box().bounds != ax.bbox.bounds:
        ext = p.get_clip_box()  # matplotlib turns Rectangle clips into a clip box (default box == axes bbox)
    else:
        return None
    (x0, _), (x1, _) = ax.transData.inverted().transform([(ext.x0, ext.y0), (ext.x1, ext.y1)])
    return x1 - x0


@pytest.mark.parametrize("icon", ["person", "circle", "star"])
def test_pictogram_icon_count_and_fractional_last(icon):
    x = np.array(list("abc"), dtype=object)
    y = np.array([6.5, 2.0, 4.25])
    fig, _ = draw("pictogram", {"x": x, "y": y}, unit_value=1.0, icon=icon, agg="sum")
    ax = fig.axes[0]
    per_row = {}
    for p in _icons(ax):
        _, i, k = p.get_gid().split(":")
        per_row.setdefault(int(i), set()).add(int(k))
    counts = [len(per_row[i]) for i in range(3)]
    assert counts == [7, 2, 5]  # ceil(value/unit)
    # full icons are unclipped; the last icon of rows with a fractional remainder is clipped to that fraction
    full_w = None
    for p in _icons(ax):
        _, i, k = p.get_gid().split(":")
        i, k = int(i), int(k)
        n = y[i]
        cw = _clip_width(p, ax)
        if k < int(np.floor(n)):
            assert cw is None
        else:
            assert cw is not None
            full = 0.8
            assert cw == pytest.approx((n - np.floor(n)) * full, rel=1e-6)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["1 icon = 1"]


def test_pictogram_auto_unit_and_no_overflow():
    x = np.array(list("ab"), dtype=object)
    y = np.array([1000.0, 250.0])
    fig, _ = draw("pictogram", {"x": x, "y": y})
    ax = fig.axes[0]
    ks = {int(p.get_gid().split(":")[2]) for p in _icons(ax)}
    assert max(ks) + 1 <= 20
    unit = float(ax.get_legend().get_texts()[0].get_text().split("=")[1].replace(",", ""))
    rows = {}
    for p in _icons(ax):
        _, i, k = p.get_gid().split(":")
        rows.setdefault(int(i), set()).add(int(k))
    assert len(rows[0]) == int(np.ceil(1000 / unit)) and len(rows[1]) == int(np.ceil(250 / unit))


@pytest.mark.parametrize("icon", ["circle", "square", "person", "star"])
def test_icon_chart_area_ratio_equals_value_ratio(icon):
    x = np.array(list("abcd"), dtype=object)
    y = np.array([100.0, 25.0, 50.0, 4.0])
    fig, _ = draw("icon_chart", {"x": x, "y": y}, icon=icon)
    ax = fig.axes[0]
    area = {}
    for p in _icons(ax):
        xy = p.get_xy()
        i = int(p.get_gid().split(":")[1])
        area[i] = area.get(i, 0.0) + 0.5 * abs(np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], -1)))
    assert sorted(area) == [0, 1, 2, 3]
    got = np.array([area[i] for i in range(4)])
    np.testing.assert_allclose(got / got[0], y / y[0], rtol=0.01)
    assert np.argmax(got) == 0


def test_icon_chart_skips_nonpositive():
    x = np.array(list("abc"), dtype=object)
    fig, _ = draw("icon_chart", {"x": x, "y": np.array([4.0, -1.0, np.nan])})
    ids = {int(p.get_gid().split(":")[1]) for p in _icons(fig.axes[0])}
    assert ids == {0}


def test_radial_bar_angles_proportional():
    x = np.array(list("abcd"), dtype=object)
    y = np.array([10.0, 40.0, 20.0, 5.0])
    fig, _ = draw("radial_bar", {"x": x, "y": y}, max_angle=270.0)
    ax = fig.axes[0]
    rings = [p for p in ax.patches if p.get_gid() == "ring"]
    tracks = [p for p in ax.patches if p.get_gid() == "track"]
    exp = np.sort(y)[::-1]  # value_desc default, first = outermost
    w = np.array([p.get_width() for p in rings])
    np.testing.assert_allclose(w, exp / exp.max() * np.deg2rad(270.0))
    assert w.max() == pytest.approx(np.deg2rad(270.0))
    np.testing.assert_allclose([p.get_width() for p in tracks], np.deg2rad(270.0))
    bottoms = [p.get_y() for p in rings]
    assert bottoms == sorted(bottoms, reverse=True)  # first (largest) ring is outermost


def test_radial_bar_custom_max_angle_and_sum():
    fig, _ = draw("radial_bar", {"x": CATS, "y": VALS}, max_angle=180.0, agg="sum", sort="appearance")
    rings = [p for p in fig.axes[0].patches if p.get_gid() == "ring"]
    sums = np.array([VALS[CATS == c].sum() for c in CAT_LIST])
    np.testing.assert_allclose([p.get_width() for p in rings], sums / sums.max() * np.pi)


# ----------------------------------------------------------------------------- smoke: degenerate input
def _smoke_cols(chart_id, kind):
    def cat(vals):
        return np.array(vals, dtype=object)
    if kind == "empty":
        e, ec = np.array([], float), np.array([], dtype=object)
        n = {"x": ec, "y": e, "group": ec, "label": ec, "actual": e, "target": e, "ranges": [e],
             "start": e, "end": e, "low": e, "high": e}
    elif kind == "single":
        n = {"x": cat(["a"]), "y": np.array([5.0]), "group": cat(["g"]), "label": cat(["a"]),
             "actual": np.array([5.0]), "target": np.array([6.0]), "ranges": [np.array([4.0]), np.array([9.0])],
             "start": np.array([1.0]), "end": np.array([2.0]), "low": np.array([1.0]), "high": np.array([2.0])}
    else:  # nan
        nan = np.full(4, np.nan)
        c = cat(["a", None, "b", "a"])
        n = {"x": c, "y": np.array([1.0, np.nan, np.nan, 3.0]), "group": cat(["g", "g", None, "h"]), "label": c,
             "actual": np.array([1.0, np.nan, np.nan, 3.0]), "target": np.array([np.nan, 2.0, 2.0, np.nan]),
             "ranges": [nan, np.array([5.0, np.nan, 6.0, 7.0])], "start": np.array([1.0, np.nan, 2.0, 3.0]),
             "end": np.array([np.nan, 1.0, 2.0, 4.0]), "low": np.array([1.0, np.nan, 2.0, 3.0]),
             "high": np.array([np.nan, 1.0, 5.0, 4.0])}
    from statsmith.charts import REGISTRY
    return {r.name: n[r.name] for r in REGISTRY[chart_id].roles}


BATCH2 = ["grouped_bar", "grouped_column", "lollipop", "bullet", "dot_plot", "dumbbell", "range_plot",
          "pictogram", "icon_chart", "radial_bar"]


def test_all_batch2_ids_registered():
    from statsmith.charts import REGISTRY
    for cid in BATCH2 + ["bar", "column"]:
        assert cid in REGISTRY and REGISTRY[cid].category == "Comparison"


@pytest.mark.parametrize("kind", ["empty", "single", "nan"])
@pytest.mark.parametrize("chart_id", BATCH2 + ["bar", "column"])
def test_smoke_degenerate_inputs(chart_id, kind):
    import warnings
    from statsmith.charts import ChartStyle
    for st in (ChartStyle(), ChartStyle(show_values=True)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, _ = draw(chart_id, _smoke_cols(chart_id, kind), style=st)
            fig.canvas.draw() if fig.canvas is not None and hasattr(fig.canvas, "draw") else None
        assert fig.axes
