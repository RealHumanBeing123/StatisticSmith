import numpy as np
import pytest

from statsmith.charts.layout import (build_tree, circle_pack_tree, distance_for_overlap, lens_area, pack_circles,
                                     power_treemap, squarify, treemap_layout)


def test_squarify_areas_proportional_inside_bounds_no_overlap(rng):
    v = rng.uniform(1, 100, 25)
    rects = squarify(list(v), 0, 0, 160, 90)
    areas = np.array([w * h for _, _, w, h in rects])
    assert areas == pytest.approx(v / v.sum() * 160 * 90, rel=1e-9)
    for x, y, w, h in rects:
        assert x >= -1e-9 and y >= -1e-9 and x + w <= 160 + 1e-9 and y + h <= 90 + 1e-9
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, b = rects[i], rects[j]
            ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]); oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
            assert not (ox > 1e-9 and oy > 1e-9)


def test_squarify_known_example_bruls():
    # Bruls et al. 2000 example: 6,6,4,3,2,2,1 in 6x4 -> first column holds 6 and 6
    r = squarify([6, 6, 4, 3, 2, 2, 1], 0, 0, 6, 4)
    assert r[0][2] == pytest.approx(3.0) and r[1][2] == pytest.approx(3.0)
    assert r[0][3] == pytest.approx(2.0) and r[1][3] == pytest.approx(2.0)


def test_tree_and_nested_layout():
    paths = [("a", "x"), ("a", "y"), ("b", "x"), ("b", "x")]
    t = build_tree(paths, np.array([1.0, 2.0, 3.0, 4.0]))
    assert t.value == 10 and {c.name: c.value for c in t.children} == {"a": 3, "b": 7}
    nodes = treemap_layout(t, 100, 100)
    top = [n for n in nodes if n.depth == 1]
    assert sum(n.w * n.h for n in top) == pytest.approx(10000)
    for n in nodes:
        if n.depth == 2:
            par = next(p for p in top if p.name == n.top)
            assert n.x >= par.x - 1e-9 and n.x + n.w <= par.x + par.w + 1e-9


def test_circle_packing_no_overlap_and_nesting(rng):
    r = list(rng.uniform(1, 5, 20))
    pos = np.array(pack_circles(r))
    for i in range(20):
        for j in range(i + 1, 20):
            assert np.hypot(*(pos[i] - pos[j])) >= r[i] + r[j] - 1e-6
    t = build_tree([("a", str(i)) for i in range(5)] + [("b", str(i)) for i in range(4)], None)
    nodes = circle_pack_tree(t)
    for n in nodes:
        if n.depth == 2:
            p = next(q for q in nodes if q.depth == 1 and q.name == n.top)
            assert np.hypot(n.x - p.x, n.y - p.y) + n.r <= p.r + 1e-6


def test_power_treemap_areas_match():
    lab, sites, err = power_treemap([50, 30, 15, 5], w=160, h=120)
    assert err < 0.01
    share = np.bincount(lab.ravel()) / lab.size
    assert share == pytest.approx([0.5, 0.3, 0.15, 0.05], abs=0.01)


def test_lens_area_and_distance_roundtrip():
    assert lens_area(1, 1, 0) == pytest.approx(np.pi)
    assert lens_area(1, 1, 2) == 0
    assert lens_area(1, 1, 1) == pytest.approx(2 * np.pi / 3 - np.sqrt(3) / 2)      # known closed form
    d = distance_for_overlap(2, 1, 1.3)
    assert lens_area(2, 1, d) == pytest.approx(1.3, rel=1e-6)
