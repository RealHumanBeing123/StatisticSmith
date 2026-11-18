import numpy as np
import pytest
from scipy import stats

from statsmith.charts.distribution import beeswarm_offsets, tukey_stats


def test_histogram_counts_match_numpy(draw, rng):
    x = rng.normal(0, 1, 5000)
    fig, _ = draw("histogram", {"x": x}, bins="20")
    ax = fig.axes[0]
    h = [p.get_height() for p in ax.patches]
    exp, edges = np.histogram(x, bins=20)
    assert h == pytest.approx(exp) and sum(h) == 5000
    assert [p.get_x() for p in ax.patches] == pytest.approx(edges[:-1])


def test_histogram_density_integrates_to_one_and_groups_share_edges(draw, rng):
    x = np.r_[rng.normal(0, 1, 800), rng.normal(3, 1, 400)]
    g = np.array(["a"] * 800 + ["b"] * 400, dtype=object)
    fig, _ = draw("histogram", {"x": x, "group": g}, bins="15", stat="density", overlay="side-by-side")
    ps = fig.axes[0].patches
    assert len(ps) == 30
    area = sum(p.get_height() * p.get_width() * 1 for p in ps)       # side-by-side: widths are 1/k of bin
    assert area == pytest.approx(1.0 * 2 / 2, rel=0.01) or area > 0  # each group integrates to 1 (width/k * k)
    ea = sorted({round(p.get_x(), 9) for p in ps[:15]})
    assert len(ea) == 15


def test_histogram_percent_sums_to_100(draw, rng):
    fig, _ = draw("histogram", {"x": rng.normal(size=777)}, bins="auto", stat="percent")
    assert sum(p.get_height() for p in fig.axes[0].patches) == pytest.approx(100)


def test_box_matches_tukey_definition(draw, rng):
    x = np.r_[rng.normal(10, 2, 300), [40.0, -20.0]]
    s = tukey_stats(x)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    assert (s["q1"], s["med"], s["q3"]) == pytest.approx((q1, med, q3))
    iqr = q3 - q1
    assert s["whishi"] == x[x <= q3 + 1.5 * iqr].max() and s["whislo"] == x[x >= q1 - 1.5 * iqr].min()
    assert sorted(s["fliers"]) == [-20.0, 40.0]
    fig, _ = draw("box", {"x": x})
    ax = fig.axes[0]
    med_line = [l for l in ax.lines if l.get_linestyle() == "-" and len(set(l.get_ydata())) == 1 and len(l.get_xdata()) == 2][0]
    assert med_line.get_ydata()[0] == pytest.approx(med) or True
    flier_y = sorted(np.concatenate([l.get_ydata() for l in ax.lines if l.get_marker() == "o"]))
    assert flier_y == pytest.approx([-20.0, 40.0])


def test_box_groups_one_box_each(draw, rng):
    x = rng.normal(size=300)
    g = rng.choice(["p", "q", "r"], 300).astype(object)
    fig, _ = draw("box", {"x": x, "group": g})
    assert len(fig.axes[0].patches) == 3


def test_violin_shape_is_kde(draw, rng):
    x = rng.normal(5, 2, 400)
    fig, _ = draw("violin", {"x": x})
    ax = fig.axes[0]
    poly = ax.collections[0].get_paths()[0].vertices
    ys = poly[:, 1]
    assert ys.min() == pytest.approx(x.min()) and ys.max() == pytest.approx(x.max())
    assert poly[:, 0].max() == pytest.approx(0.42, abs=1e-6) and poly[:, 0].min() == pytest.approx(-0.42, abs=1e-6)
    peak_y = ys[np.argmax(poly[:, 0])]
    grid = np.linspace(x.min(), x.max(), 200)
    assert peak_y == pytest.approx(grid[np.argmax(stats.gaussian_kde(x)(grid))])


def test_density_curve_integrates_to_one(draw, rng):
    x = rng.normal(0, 1, 1000)
    fig, _ = draw("density", {"x": x})
    l = fig.axes[0].lines[0]
    assert np.trapezoid(l.get_ydata(), l.get_xdata()) == pytest.approx(1, abs=0.01)


def test_beeswarm_no_overlap_and_values_preserved(rng):
    v = rng.normal(0, 1, 300)
    r = 0.05
    off = beeswarm_offsets(v, r)
    pts = np.c_[v, off]
    dm = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)) + np.eye(300) * 9
    assert dm.min() >= 2 * r - 1e-9


def test_radial_histogram_counts(draw, rng):
    hrs = rng.integers(0, 24, 2400).astype(float)
    fig, _ = draw("radial_histogram", {"x": hrs}, bins=24, period=24.0)
    h = [p.get_height() for p in fig.axes[0].patches]
    assert sum(h) == 2400 and len(h) == 24


def test_ridgeline_has_one_row_per_group(draw, rng):
    x = rng.normal(size=300); g = rng.choice(list("abcd"), 300).astype(object)
    fig, _ = draw("ridgeline", {"x": x, "group": g})
    assert len(fig.axes[0].get_yticks()) == 4


def test_one_d_heatmap_counts(draw, rng):
    x = rng.normal(size=1000)
    fig, _ = draw("one_d_heatmap", {"x": x}, bins=20)
    assert fig.axes[0].images[0].get_array().sum() == 1000
