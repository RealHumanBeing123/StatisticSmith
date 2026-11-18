"""Pure geometry used by hierarchical charts (no matplotlib): tree building, squarified treemap,
circle packing, power-diagram (convex) treemap, circle-overlap solver for Euler diagrams."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Node:
    name: str
    value: float = 0.0
    children: list["Node"] = field(default_factory=list)
    depth: int = 0
    # filled by layouts
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    r: float = 0.0
    top: str = ""            # name of the top-level ancestor (for colouring)

    @property
    def leaf(self):
        return not self.children


def build_tree(paths: list[tuple], values: np.ndarray | None = None) -> Node:
    """``paths`` = one tuple of labels per row (outer -> inner). Value = sum of ``values`` (or row count)."""
    root = Node("root")
    index: dict[tuple, Node] = {(): root}
    for i, path in enumerate(paths):
        v = 1.0 if values is None else float(values[i])
        if not np.isfinite(v) or v <= 0:
            continue
        for depth in range(1, len(path) + 1):
            key = tuple(path[:depth])
            node = index.get(key)
            if node is None:
                node = Node(str(path[depth - 1]), depth=depth, top=str(path[0]))
                index[key] = node
                index[key[:-1]].children.append(node)
            if depth == len(path):
                node.value += v
    def total(n: Node) -> float:
        if n.children:
            kids = sum(total(c) for c in n.children)
            n.value = kids + (n.value if False else 0.0)
        return n.value
    total(root)
    def order(n: Node):
        n.children.sort(key=lambda c: -c.value)
        for c in n.children:
            order(c)
    order(root)
    return root


# --------------------------------------------------------------------------- squarified treemap (Bruls et al. 2000)
def _worst(row: list[float], side: float) -> float:
    s = sum(row)
    return max(max(side * side * r / (s * s), s * s / (side * side * r)) for r in row)


def squarify(values: list[float], x: float, y: float, w: float, h: float):
    """Return [(x, y, w, h)] for each value (same order); areas are proportional to values."""
    vals = np.asarray(values, dtype=float)
    out: list[tuple] = [None] * len(vals)
    tot = vals.sum()
    if tot <= 0 or w <= 0 or h <= 0:
        return [(x, y, 0.0, 0.0)] * len(vals)
    areas = list(vals / tot * w * h)
    idx = list(range(len(vals)))
    cx, cy, cw, ch = x, y, w, h
    row: list[int] = []
    def lay(row_idx):
        nonlocal cx, cy, cw, ch
        s = sum(areas[i] for i in row_idx)
        if cw >= ch:                              # lay the row along the shorter side (height)
            rw = s / ch if ch else 0
            yy = cy
            for i in row_idx:
                hh = areas[i] / rw if rw else 0
                out[i] = (cx, yy, rw, hh)
                yy += hh
            cx += rw; cw -= rw
        else:
            rh = s / cw if cw else 0
            xx = cx
            for i in row_idx:
                ww = areas[i] / rh if rh else 0
                out[i] = (xx, cy, ww, rh)
                xx += ww
            cy += rh; ch -= rh
    for i in idx:
        side = min(cw, ch)
        cand = row + [i]
        if row and _worst([areas[j] for j in cand], side) > _worst([areas[j] for j in row], side):
            lay(row)
            row = [i]
        else:
            row = cand
    if row:
        lay(row)
    return out


def treemap_layout(root: Node, w=100.0, h=100.0, pad=0.0, header=0.0) -> list[Node]:
    """Assign x,y,w,h to every node (recursively squarified inside parents). Returns all non-root nodes."""
    root.x, root.y, root.w, root.h = 0.0, 0.0, w, h
    nodes: list[Node] = []
    def rec(n: Node):
        if not n.children:
            return
        x, y, ww, hh = n.x + pad, n.y + pad + (header if n.depth > 0 else 0), n.w - 2 * pad, n.h - 2 * pad - (header if n.depth > 0 else 0)
        ww, hh = max(ww, 0), max(hh, 0)
        rects = squarify([c.value for c in n.children], x, y, ww, hh)
        for c, (rx, ry, rw, rh) in zip(n.children, rects):
            c.x, c.y, c.w, c.h = rx, ry, rw, rh
            nodes.append(c)
            rec(c)
    rec(root)
    return nodes


# --------------------------------------------------------------------------- circle packing
def pack_circles(radii: list[float]):
    """Greedy pack: place circles (largest first) tangent to two already-placed circles at the position
    closest to the centroid that overlaps nothing. Returns [(x, y)] in the input order. O(n^2)."""
    n = len(radii)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: -radii[i])
    pos = np.zeros((n, 2))
    placed: list[int] = []
    for k, i in enumerate(order):
        r = radii[i]
        if k == 0:
            pos[i] = (0, 0); placed.append(i); continue
        if k == 1:
            j = placed[0]; pos[i] = (radii[j] + r, 0); placed.append(i); continue
        cen = pos[placed].mean(axis=0)
        best, bd = None, np.inf
        P = pos[placed]; R = np.array([radii[j] for j in placed])
        for a in range(len(placed)):
            for b in range(a + 1, len(placed)):
                ja, jb = placed[a], placed[b]
                da, db = radii[ja] + r, radii[jb] + r
                v = pos[jb] - pos[ja]
                dist = np.hypot(*v)
                if dist == 0 or dist > da + db or dist < abs(da - db):
                    continue
                ca = (da * da - db * db + dist * dist) / (2 * dist)
                hsq = da * da - ca * ca
                if hsq < 0:
                    continue
                hh = math.sqrt(hsq)
                base = pos[ja] + v * ca / dist
                perp = np.array([-v[1], v[0]]) / dist
                for sgn in (1, -1):
                    c = base + sgn * hh * perp
                    if np.all(np.hypot(*(P - c).T) >= R + r - 1e-9):
                        dd = np.hypot(*(c - cen))
                        if dd < bd:
                            best, bd = c, dd
        if best is None:                                  # numerical corner case: put it outside
            best = np.array([P[:, 0].max() + R.max() + r, 0.0])
        pos[i] = best
        placed.append(i)
    return [tuple(p) for p in pos]


def enclose(circles: list[tuple]) -> tuple[float, float, float]:
    """Smallest circle (centre, radius) containing all circles [(x,y,r)] - minimax solved with Nelder-Mead."""
    from scipy.optimize import minimize
    c = np.array(circles, dtype=float)
    if len(c) == 1:
        return float(c[0, 0]), float(c[0, 1]), float(c[0, 2])
    f = lambda p: (np.hypot(c[:, 0] - p[0], c[:, 1] - p[1]) + c[:, 2]).max()
    x0 = np.array([((c[:, 0] - c[:, 2]).min() + (c[:, 0] + c[:, 2]).max()) / 2,
                   ((c[:, 1] - c[:, 2]).min() + (c[:, 1] + c[:, 2]).max()) / 2])
    res = minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-12, maxiter=2000))
    return float(res.x[0]), float(res.x[1]), float(f(res.x))


def circle_pack_tree(root: Node) -> list[Node]:
    """Nested circle packing. Leaf radius = sqrt(value); parents enclose their children. Fills x, y, r (absolute)."""
    out: list[Node] = []
    def rec(n: Node):
        if not n.children:
            n.r = math.sqrt(max(n.value, 0)); return
        for c in n.children:
            rec(c)
        pos = pack_circles([c.r for c in n.children])
        for c, (px, py) in zip(n.children, pos):
            c.x, c.y = px, py
        cx, cy, R = enclose([(c.x, c.y, c.r) for c in n.children])
        for c in n.children:
            c.x -= cx; c.y -= cy
        n.r = R * 1.04 + 1e-9
    def place(n: Node, ox, oy):
        for c in n.children:
            c.x += ox; c.y += oy
            out.append(c)
            place(c, c.x, c.y)
    rec(root)
    place(root, 0.0, 0.0)
    return out


# --------------------------------------------------------------------------- convex (power-diagram) treemap
def power_treemap(values: list[float], w: int = 480, h: int = 360, iters: int = 60, seed: int = 1):
    """Rasterised weighted-Voronoi (power diagram) treemap. Cells are convex. Returns (label_image[h, w], sites[n, 2], area_error).
    Weights are adapted until each cell's pixel share matches value / sum(values)."""
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    target = vals / vals.sum()
    rng = np.random.default_rng(seed)
    sites = np.c_[rng.uniform(0.1, 0.9, n) * w, rng.uniform(0.1, 0.9, n) * h]
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5]
    wt = np.zeros(n)
    scale = (w * h) / n
    lab = None
    for it in range(iters):
        d = ((pts[:, None, :] - sites[None]) ** 2).sum(-1) - wt[None]
        lab = d.argmin(1)
        share = np.bincount(lab, minlength=n) / len(pts)
        for i in range(n):                                   # Lloyd move toward the centroid
            m = lab == i
            if m.any():
                sites[i] = 0.5 * sites[i] + 0.5 * pts[m].mean(0)
        wt += (target - share) * scale * 1.5
    d = ((pts[:, None, :] - sites[None]) ** 2).sum(-1) - wt[None]
    lab = d.argmin(1)
    for _ in range(80):                                      # final weight-only polish (sites fixed)
        share = np.bincount(lab, minlength=n) / len(pts)
        if np.abs(share - target).max() < 0.002:
            break
        wt += (target - share) * scale
        lab = (((pts[:, None, :] - sites[None]) ** 2).sum(-1) - wt[None]).argmin(1)
    share = np.bincount(lab, minlength=n) / len(pts)
    return lab.reshape(h, w), sites, float(np.abs(share - target).max())


# --------------------------------------------------------------------------- Euler / Venn helpers
def lens_area(r1: float, r2: float, d: float) -> float:
    """Area of intersection of two circles with radii r1, r2 and centre distance d."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    b = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    c = 0.5 * math.sqrt(max((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2), 0))
    return a + b - c


def distance_for_overlap(r1: float, r2: float, overlap: float) -> float:
    """Centre distance giving the requested lens area (bisection)."""
    lo, hi = abs(r1 - r2), r1 + r2
    if overlap <= 0:
        return hi
    if overlap >= math.pi * min(r1, r2) ** 2:
        return lo
    for _ in range(80):
        mid = (lo + hi) / 2
        if lens_area(r1, r2, mid) > overlap:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
