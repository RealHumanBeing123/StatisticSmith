"""Geospatial & other charts: choropleth, geo heat map, tile map (US grid), Sankey, chord, arc, network, flowchart.

Boundary data (offline, bundled in ``statsmith/data/geo``): US states (US Census, via PublicaMundi/MappingAPI)
and world countries (Natural Earth, via johan/world.geo.json). US tile-grid layout: hafen/grid-designer
``us_state_grid1`` (11 columns x 7 rows, includes DC).
"""
from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from .base import (AGGS, Param, Role, aggregate, chart, empty_message, finish, first_seen, fmt, make_axes, numeric, short)

GEO_DIR = Path(__file__).resolve().parents[1] / "data" / "geo"


# =========================================================================== geometry data
@lru_cache(maxsize=1)
def tile_grid():
    """-> {code: (col, row, name)} with row 1 at the top."""
    with open(GEO_DIR / "us_tile_grid.csv", newline="", encoding="utf-8") as f:
        return {r["code"]: (int(r["col"]), int(r["row"]), r["name"]) for r in csv.DictReader(f)}


def _norm(s) -> str:
    s = re.sub(r"[^a-z0-9 ]", "", str(s).lower().replace("&", "and")).strip()
    return re.sub(r"^the ", "", re.sub(r"\s+", " ", s))


@lru_cache(maxsize=1)
def _us_keys() -> dict[str, str]:
    """normalised name / USPS code -> canonical state name."""
    out = {}
    for code, (_, _, name) in tile_grid().items():
        out[_norm(code)] = name
        out[_norm(name)] = name
    out["pr"] = out["puerto rico"] = "Puerto Rico"
    out["washington dc"] = out["dc"] = "District of Columbia"
    return out


_WORLD_ALIASES = {
    "us": "United States of America", "usa": "United States of America", "united states": "United States of America",
    "uk": "United Kingdom", "great britain": "United Kingdom", "britain": "United Kingdom",
    "south korea": "South Korea", "republic of korea": "South Korea", "korea": "South Korea",
    "north korea": "North Korea", "russian federation": "Russia", "czechia": "Czech Republic",
    "ivory coast": "Ivory Coast", "cote divoire": "Ivory Coast", "burma": "Myanmar", "viet nam": "Vietnam",
    "tanzania": "United Republic of Tanzania", "drc": "Democratic Republic of the Congo",
    "dr congo": "Democratic Republic of the Congo", "congo kinshasa": "Democratic Republic of the Congo",
    "congo": "Republic of the Congo", "macedonia": "Macedonia", "north macedonia": "Macedonia",
    "eswatini": "Swaziland", "uae": "United Arab Emirates", "serbia": "Republic of Serbia",
    "bahamas": "The Bahamas", "timor-leste": "East Timor", "east timor": "East Timor", "guinea-bissau": "Guinea Bissau",
}


@lru_cache(maxsize=2)
def regions(kind: str):
    """Load boundaries: list of (name, key_set, [polygons]) where polygon = list of rings ([ [lon,lat],... ])."""
    fn = "us_states.geojson" if kind == "us" else "world.geojson"
    gj = json.load(open(GEO_DIR / fn, encoding="utf-8"))
    out = []
    for ft in gj["features"]:
        name = ft["properties"]["name"]
        g = ft["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        keys = {_norm(name)}
        if kind == "world" and ft.get("id"):
            keys.add(_norm(ft["id"]))
        out.append((name, keys, polys))
    return out


def match_regions(kind: str, labels):
    """label -> canonical region name (or None)."""
    regs = regions(kind)
    by_key = {k: n for n, ks, _ in regs for k in ks}
    res = {}
    for lab in labels:
        k = _norm(lab)
        if kind == "us":
            res[lab] = _us_keys().get(k)
        else:
            res[lab] = by_key.get(k) or by_key.get(_norm(_WORLD_ALIASES.get(k, "")))
    return res


def _inset(name, ring):
    """Alaska/Hawaii insets for the US map (not to scale)."""
    a = np.asarray(ring, dtype=float)
    if name == "Alaska":
        a[a[:, 0] > 0, 0] -= 360
        a[:, 0] = (a[:, 0] + 150) * 0.35 - 113
        a[:, 1] = (a[:, 1] - 64) * 0.5 + 28
    elif name == "Hawaii":
        a[:, 0] += 52
        a[:, 1] += 7
    return a


def region_paths(kind: str, inset=True):
    from matplotlib.path import Path as P
    out = {}
    for name, _, polys in regions(kind):
        verts, codes = [], []
        for poly in polys:
            for ring in poly:
                a = _inset(name, ring) if (kind == "us" and inset) else np.asarray(ring, dtype=float)
                if len(a) < 3:
                    continue
                verts.extend(a.tolist())
                codes.extend([P.MOVETO] + [P.LINETO] * (len(a) - 2) + [P.CLOSEPOLY])
        if verts:
            out[name] = P(np.array(verts), codes)
    return out


def _draw_regions(ax, kind, values: dict, style, nodata="#cfcfcf", cmap=None, norm=None):
    import matplotlib as mpl
    from matplotlib.patches import PathPatch
    paths = region_paths(kind)
    for name, p in paths.items():
        v = values.get(name)
        fc = nodata if v is None or not np.isfinite(v) else cmap(norm(v))
        ax.add_patch(PathPatch(p, facecolor=fc, edgecolor=style.background, lw=0.5))
    ax.autoscale_view()
    ax.set_aspect(1 / np.cos(np.radians(38 if kind == "us" else 20)))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return paths


def _regional_values(d, kind, agg):
    labs = np.asarray(d["region"], dtype=object).astype(str)
    names, vals = aggregate(labs, d.get("value") if d.has("value") else None, agg)
    m = match_regions(kind, names)
    out, miss = {}, []
    for n, v in zip(names, vals):
        r = m.get(n)
        if r is None:
            miss.append(n)
        else:
            out[r] = out.get(r, 0.0) + float(v) if (r in out and agg in ("sum", "count")) else float(v)
    return out, miss


# =========================================================================== choropleth
@chart("choropleth", "Choropleth map", "Geospatial & other",
       [Role("region", "categorical", "Region (US state name/code or country name/ISO3)"),
        Role("value", "numeric", "Value (blank = count rows)", required=False)],
       [Param("map", "Map", "choice", "US states", choices=("US states", "World countries")),
        Param("agg", "Aggregate", "choice", "sum", choices=AGGS)], aliases=("filled map", "region map"))
def choropleth(fig, d, style, map="US states", agg="sum"):
    """Regions shaded by value. Regions with no data are grey; unmatched labels are listed in the notes."""
    import matplotlib as mpl
    kind = "us" if map.startswith("US") else "world"
    vals, miss = _regional_values(d, kind, agg)
    ax = make_axes(fig, style)
    cmap = mpl.colormaps[style.cmap]
    arr = np.array(list(vals.values())) if vals else np.array([0.0, 1.0])
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    norm = mpl.colors.Normalize(lo, hi if hi > lo else lo + 1)
    _draw_regions(ax, kind, vals, style, nodata=style.grid_color, cmap=cmap, norm=norm)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(colors=style.text_color, labelsize=style.tick_size)
    cb.set_label(style.legend_title or (d.name("value") if d.has("value") else "count"), color=style.text_color)
    if miss:
        d.notes.append(f"{len(miss)} label(s) did not match a region: {', '.join(map_short(miss))}")
    if kind == "us":
        d.notes.append("Alaska and Hawaii are shown as insets (not to scale)")
    finish(fig, ax, style, legend=False, axes_decor=False)
    return ax


def map_short(items, n=6):
    items = [str(i) for i in items]
    return items[:n] + (["…"] if len(items) > n else [])


# =========================================================================== geo heat map
@chart("geo_heatmap", "Geographic heat map", "Geospatial & other",
       [Role("lat", "numeric", "Latitude"), Role("lon", "numeric", "Longitude"),
        Role("value", "numeric", "Value (blank = count points)", required=False)],
       [Param("bins", "Grid cells across", "int", 80, min=10, max=400),
        Param("agg", "Cell value", "choice", "sum", choices=("sum", "mean", "count")),
        Param("basemap", "Outline", "choice", "auto", choices=("auto", "US states", "World", "none"))],
       aliases=("density map", "point density map"))
def geo_heatmap(fig, d, style, bins=80, agg="sum", basemap="auto"):
    """Points binned on a latitude/longitude grid and coloured by count / sum / mean of value."""
    import matplotlib as mpl
    from matplotlib.patches import PathPatch
    lat, lon = numeric(d["lat"]), numeric(d["lon"])
    ok = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    if not ok.any():
        return empty_message(fig, style, "No valid coordinates")
    lat, lon = lat[ok], lon[ok]
    w = numeric(d["value"])[ok] if d.has("value") else None
    if w is not None:
        good = np.isfinite(w)
        lat, lon, w = lat[good], lon[good], w[good]
    ax = make_axes(fig, style)
    pad_x, pad_y = max(np.ptp(lon) * 0.05, 0.5), max(np.ptp(lat) * 0.05, 0.5)
    xr, yr = (lon.min() - pad_x, lon.max() + pad_x), (lat.min() - pad_y, lat.max() + pad_y)
    ny = max(int(bins * (yr[1] - yr[0]) / (xr[1] - xr[0])), 5) if xr[1] > xr[0] else bins
    rng = [[yr[0], yr[1]], [xr[0], xr[1]]]
    cnt, ye, xe = np.histogram2d(lat, lon, bins=[ny, bins], range=rng)
    if w is None or agg == "count":
        Z = cnt
    else:
        s, _, _ = np.histogram2d(lat, lon, bins=[ny, bins], range=rng, weights=w)
        Z = s if agg == "sum" else np.divide(s, cnt, out=np.zeros_like(s), where=cnt > 0)
    Z = np.ma.masked_where(cnt == 0, Z)
    bm = basemap
    if bm == "auto":
        bm = "US states" if (-130 < lon.min() and lon.max() < -60 and 15 < lat.min() and lat.max() < 72) else "World"
    if bm != "none":
        paths = region_paths("us" if bm.startswith("US") else "world", inset=False)
        for p in paths.values():
            ax.add_patch(PathPatch(p, facecolor="none", edgecolor=style.grid_color, lw=0.6))
    m = ax.pcolormesh(xe, ye, Z, cmap=style.cmap, alpha=style.alpha, shading="flat")
    ax.set_xlim(*xr)
    ax.set_ylim(*yr)
    ax.set_aspect(1 / max(np.cos(np.radians((yr[0] + yr[1]) / 2)), 0.2))
    cb = fig.colorbar(m, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(colors=style.text_color, labelsize=style.tick_size)
    cb.set_label("count" if (w is None or agg == "count") else f"{agg} of {d.name('value')}", color=style.text_color)
    finish(fig, ax, style, xlabel=d.name("lon"), ylabel=d.name("lat"), legend=False, grid_axis="both")
    return ax


# =========================================================================== tile map
@chart("tile_map", "Tile grid map (US states)", "Geospatial & other",
       [Role("region", "categorical", "State (name or 2-letter code)"), Role("value", "numeric", "Value (blank = count rows)", required=False)],
       [Param("agg", "Aggregate", "choice", "sum", choices=AGGS), Param("labels", "Show state codes", "bool", True)],
       aliases=("tile grid map", "cartogram", "hex tile map"))
def tile_map(fig, d, style, agg="sum", labels=True):
    """Each state is one equal-sized square arranged roughly where it lies on the map (11 columns x 7 rows + DC)."""
    import matplotlib as mpl
    from matplotlib.patches import Rectangle
    vals, miss = _regional_values(d, "us", agg)
    ax = make_axes(fig, style)
    cmap = mpl.colormaps[style.cmap]
    arr = np.array(list(vals.values())) if vals else np.array([0.0, 1.0])
    lo, hi = float(arr.min()), float(arr.max())
    norm = mpl.colors.Normalize(lo, hi if hi > lo else lo + 1)
    for code, (c, r, name) in tile_grid().items():
        v = vals.get(name)
        fc = style.grid_color if v is None else cmap(norm(v))
        ax.add_patch(Rectangle((c - 0.48, r - 0.48), 0.96, 0.96, facecolor=fc, edgecolor=style.background))
        if labels:
            rgb = mpl.colors.to_rgb(fc)
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            ax.text(c, r, code, ha="center", va="center", fontsize=style.value_label_size + 1, weight="bold",
                    color="#111111" if lum > 0.55 else "#ffffff")
            if style.show_values and v is not None:
                ax.text(c, r + 0.3, fmt(v, style), ha="center", va="center", fontsize=style.value_label_size - 2,
                        color="#111111" if lum > 0.55 else "#ffffff")
    ax.set_xlim(0.4, 11.6)
    ax.set_ylim(7.6, 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.03, pad=0.02)
    cb.ax.tick_params(colors=style.text_color, labelsize=style.tick_size)
    if miss:
        d.notes.append(f"{len(miss)} label(s) did not match a state: {', '.join(map_short(miss))}")
    finish(fig, ax, style, legend=False, axes_decor=False)
    return ax


# =========================================================================== graph helpers (flows / networks)
_EDGE_ROLES = [Role("source", "categorical", "From"), Role("target", "categorical", "To"),
               Role("value", "numeric", "Weight (blank = count rows)", required=False)]


def edge_table(d, max_edges=150, drop_self=True):
    """Aggregate rows into unique directed edges. -> (nodes, [(u, v, w)], notes)."""
    s = np.asarray(d["source"], dtype=object).astype(str)
    t = np.asarray(d["target"], dtype=object).astype(str)
    w = numeric(d["value"]) if d.has("value") else np.ones(len(s))
    ok = np.isfinite(w) & (w > 0) & ~np.isin(s, ["", "nan", "None"]) & ~np.isin(t, ["", "nan", "None"])
    acc: dict = {}
    for a, b, x in zip(s[ok], t[ok], w[ok]):
        if drop_self and a == b:
            continue
        acc[(a, b)] = acc.get((a, b), 0.0) + float(x)
    edges = sorted(((a, b, x) for (a, b), x in acc.items()), key=lambda e: -e[2])
    notes = []
    if len(edges) > max_edges:
        notes.append(f"showing the {max_edges} largest of {len(edges)} links")
        edges = edges[:max_edges]
    nodes = first_seen([n for e in edges for n in e[:2]])
    return nodes, edges, notes


def layer_levels(nodes, edges):
    """Longest-path layering; edges that would create a cycle are ignored for layout. -> {node: level}"""
    adj = {n: [] for n in nodes}
    for a, b, _ in edges:
        adj[a].append(b)
    state, back = {}, set()
    def dfs(u):
        state[u] = 1
        for v in adj[u]:
            if state.get(v) == 1:
                back.add((u, v))
            elif v not in state:
                dfs(v)
        state[u] = 2
    import sys
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 5000))
    for n in nodes:
        if n not in state:
            dfs(n)
    dag = [(a, b) for a, b, _ in edges if (a, b) not in back]
    indeg = {n: 0 for n in nodes}
    for a, b in dag:
        indeg[b] += 1
    level = {n: 0 for n in nodes}
    queue = [n for n in nodes if indeg[n] == 0]
    out = {n: [] for n in nodes}
    for a, b in dag:
        out[a].append(b)
    while queue:
        u = queue.pop(0)
        for v in out[u]:
            level[v] = max(level[v], level[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return level


def _order_layers(layers, edges):
    """Reduce crossings with two barycentre sweeps."""
    pos = {n: i for L in layers for i, n in enumerate(L)}
    nbr = {}
    for a, b, _ in edges:
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    for _ in range(2):
        for L in layers:
            bary = {n: np.mean([pos[m] for m in nbr.get(n, [n])]) for n in L}
            L.sort(key=lambda n: bary[n])
            for i, n in enumerate(L):
                pos[n] = i
    return layers


# =========================================================================== sankey
def sankey_layout(nodes, edges, pad_frac=0.04):
    """Pure layout (no drawing) so it can be tested: returns dict with node boxes and link slots.

    Node height = max(inflow, outflow) * k. Links get an exact slice (height = value * k) at both ends.
    """
    level = layer_levels(nodes, edges)
    nl = max(level.values()) + 1
    layers = [[n for n in nodes if level[n] == i] for i in range(nl)]
    layers = _order_layers(layers, edges)
    inflow = {n: 0.0 for n in nodes}
    outflow = {n: 0.0 for n in nodes}
    for a, b, w in edges:
        outflow[a] += w
        inflow[b] += w
    size = {n: max(inflow[n], outflow[n]) for n in nodes}
    k = min((1 - pad_frac * max(len(L) - 1, 0)) / sum(size[n] for n in L) for L in layers if L)
    box = {}
    for i, L in enumerate(layers):
        total = sum(size[n] for n in L) * k + pad_frac * max(len(L) - 1, 0)
        y = (1 - total) / 2
        for n in L:
            box[n] = [i, y, size[n] * k]
            y += size[n] * k + pad_frac
    pos = {n: box[n][1] for n in nodes}
    out_cur = dict(pos)
    in_cur = dict(pos)
    links = []
    ycenter = lambda n: box[n][1] + box[n][2] / 2
    for a, b, w in sorted(edges, key=lambda e: (ycenter(e[1]), ycenter(e[0]))):
        pass
    outs = {n: sorted([e for e in edges if e[0] == n], key=lambda e: ycenter(e[1])) for n in nodes}
    ins = {n: sorted([e for e in edges if e[1] == n], key=lambda e: ycenter(e[0])) for n in nodes}
    slot_out, slot_in = {}, {}
    for n in nodes:
        y = box[n][1]
        for e in outs[n]:
            slot_out[e[:2]] = y
            y += e[2] * k
        y = box[n][1]
        for e in ins[n]:
            slot_in[e[:2]] = y
            y += e[2] * k
    for a, b, w in edges:
        links.append((a, b, w, slot_out[(a, b)], slot_in[(a, b)], w * k))
    return {"level": level, "layers": layers, "box": box, "links": links, "k": k, "nlayers": nl}


@chart("sankey", "Sankey diagram", "Geospatial & other", _EDGE_ROLES,
       [Param("max_links", "Max links", "int", 60, min=5, max=500)], aliases=("flow diagram", "alluvial"))
def sankey(fig, d, style, max_links=60):
    """Nodes in columns; link width is proportional to flow. Node height = max(in, out)."""
    nodes, edges, notes = edge_table(d, max_links)
    d.notes.extend(notes)
    if not edges:
        return empty_message(fig, style, "No links (need From, To and positive weights)")
    L = sankey_layout(nodes, edges)
    ax = make_axes(fig, style)
    colors = {n: style.color_for(n, i) for i, n in enumerate(nodes)}
    nw = 0.025
    g = np.linspace(0, 1, 40)
    sm = 3 * g ** 2 - 2 * g ** 3
    nl = max(L["nlayers"] - 1, 1)
    for a, b, w, ya, yb, h in L["links"]:
        x0, x1 = L["box"][a][0] / nl + nw, L["box"][b][0] / nl
        if x1 <= x0:
            x1 = x0 + 1e-3
        xs = x0 + (x1 - x0) * g
        lo = ya + (yb - ya) * sm
        ax.fill_between(xs, lo, lo + h, color=colors[a], alpha=0.4 * style.alpha / 0.9, lw=0)
    for n in nodes:
        i, y, h = L["box"][n]
        ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle((i / nl, y), nw, max(h, 1e-4), color=colors[n]))
        left = i == 0
        ax.text(i / nl + (-0.008 if left else nw + 0.008), y + h / 2, short(n, 20), ha="right" if left else "left", va="center",
                fontsize=style.value_label_size + 1, color=style.text_color)
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(1.02, -0.02)
    ax.axis("off")
    finish(fig, ax, style, legend=False, axes_decor=False)
    return ax


# =========================================================================== chord diagram
def chord_layout(nodes, edges, gap=0.04):
    """Angles (radians) for node arcs and link ends. Arc length of a node = total weight touching it."""
    tot = {n: 0.0 for n in nodes}
    for a, b, w in edges:
        tot[a] += w
        tot[b] += w
    full = sum(tot.values())
    usable = 2 * np.pi - gap * len(nodes)
    per = usable / full
    start, ang = {}, 0.0
    for n in nodes:
        start[n] = ang
        ang += tot[n] * per + gap
    cur = dict(start)
    ends = []
    for a, b, w in edges:
        a0 = cur[a]; cur[a] += w * per
        b0 = cur[b]; cur[b] += w * per
        ends.append((a, b, w, (a0, cur[a]), (b0, cur[b])))
    return {"arcs": {n: (start[n], start[n] + tot[n] * per) for n in nodes}, "links": ends, "per": per, "tot": tot}


@chart("chord_diagram", "Chord diagram", "Geospatial & other", _EDGE_ROLES,
       [Param("max_links", "Max links", "int", 80, min=5, max=500)], aliases=("circular flow diagram",))
def chord_diagram(fig, d, style, max_links=80):
    """Nodes share a circle (arc length = total flow); ribbons connect nodes, width = flow."""
    from matplotlib.patches import PathPatch, Wedge
    from matplotlib.path import Path as P
    nodes, edges, notes = edge_table(d, max_links)
    d.notes.extend(notes)
    if not edges:
        return empty_message(fig, style, "No links (need From, To and positive weights)")
    L = chord_layout(nodes, edges)
    ax = make_axes(fig, style)
    colors = {n: style.color_for(n, i) for i, n in enumerate(nodes)}
    pt = lambda t, r=1.0: (r * np.cos(t), r * np.sin(t))
    for n, (a0, a1) in L["arcs"].items():
        ax.add_patch(Wedge((0, 0), 1.08, np.degrees(a0), np.degrees(a1), width=0.07, color=colors[n]))
        mid = (a0 + a1) / 2
        x, y = pt(mid, 1.16)
        ax.text(x, y, short(n, 16), ha="center", va="center", fontsize=style.value_label_size + 1, color=style.text_color,
                rotation=np.degrees(mid) - (180 if np.cos(mid) < 0 else 0), rotation_mode="anchor")
    for a, b, w, (s0, s1), (t0, t1) in L["links"]:
        arc = lambda u0, u1: [pt(u) for u in np.linspace(u0, u1, 6)]
        sa, tb = arc(s0, s1), arc(t0, t1)
        verts = [sa[0]] + sa[1:] + [(0, 0), tb[0]] + tb[1:] + [(0, 0), sa[0]]
        codes = [P.MOVETO] + [P.LINETO] * 5 + [P.CURVE3, P.CURVE3] + [P.LINETO] * 5 + [P.CURVE3, P.CURVE3]
        ax.add_patch(PathPatch(P(verts, codes), facecolor=colors[a], edgecolor=style.background, lw=0.3, alpha=0.55))
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    finish(fig, ax, style, legend=False, axes_decor=False)
    return ax


# =========================================================================== arc diagram
@chart("arc_diagram", "Arc diagram", "Geospatial & other", _EDGE_ROLES,
       [Param("max_links", "Max links", "int", 100, min=5, max=500), Param("sort", "Node order", "choice", "weight", choices=("weight", "appearance", "alpha"))],
       aliases=("arc chart",))
def arc_diagram(fig, d, style, max_links=100, sort="weight"):
    """Nodes on a line; a semicircular arc joins each linked pair, thickness = weight."""
    nodes, edges, notes = edge_table(d, max_links)
    d.notes.extend(notes)
    if not edges:
        return empty_message(fig, style, "No links (need From, To and positive weights)")
    deg = {n: 0.0 for n in nodes}
    for a, b, w in edges:
        deg[a] += w
        deg[b] += w
    order = sorted(nodes, key=lambda n: -deg[n]) if sort == "weight" else (sorted(nodes) if sort == "alpha" else nodes)
    pos = {n: i for i, n in enumerate(order)}
    ax = make_axes(fig, style)
    mx = max(w for *_, w in edges)
    th = np.linspace(0, np.pi, 60)
    for a, b, w in edges:
        x0, x1 = sorted((pos[a], pos[b]))
        r = (x1 - x0) / 2
        ax.plot((x0 + x1) / 2 + r * np.cos(th), r * np.sin(th), color=style.color_for(a, nodes.index(a)),
                lw=0.6 + 7 * w / mx, alpha=0.55, solid_capstyle="round")
    ax.scatter(range(len(order)), np.zeros(len(order)), s=[20 + 160 * deg[n] / max(deg.values()) for n in order],
               color=style.text_color, zorder=3)
    for n in order:
        ax.text(pos[n], -0.04 * len(order) ** 0.7, short(n, 14), rotation=45 if len(order) > 8 else 0, ha="right" if len(order) > 8 else "center",
                va="top", fontsize=style.value_label_size, color=style.text_color)
    ax.set_xlim(-1, len(order))
    ax.set_ylim(-0.12 * len(order) ** 0.8, len(order) / 2 + 0.5)
    ax.axis("off")
    finish(fig, ax, style, legend=False, axes_decor=False)
    return ax


# =========================================================================== network
def fr_layout(n, edges_idx, weights, iters=120, seed=1):
    """Fruchterman-Reingold force layout (vectorised). edges_idx: (m, 2) int array. Returns (n, 2) in [0, 1]."""
    rng = np.random.default_rng(seed)
    P = rng.uniform(-0.5, 0.5, (n, 2))
    if n == 1:
        return np.full((1, 2), 0.5)
    k = np.sqrt(1.0 / n)
    t = 0.1
    ei = np.asarray(edges_idx, dtype=int).reshape(-1, 2)
    w = np.asarray(weights, dtype=float)
    w = w / w.max() if w.size else w
    for _ in range(iters):
        delta = P[:, None, :] - P[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        np.fill_diagonal(dist, 1.0)
        dist = np.maximum(dist, 1e-4)
        F = (k * k / dist ** 2)[:, :, None] * delta
        disp = F.sum(axis=1)
        if ei.size:
            dv = P[ei[:, 0]] - P[ei[:, 1]]
            dl = np.maximum(np.linalg.norm(dv, axis=1), 1e-4)
            f = (dl / k)[:, None] * dv * (0.3 + 0.7 * w)[:, None]
            np.add.at(disp, ei[:, 0], -f)
            np.add.at(disp, ei[:, 1], f)
        ln = np.maximum(np.linalg.norm(disp, axis=1), 1e-9)
        P += disp / ln[:, None] * np.minimum(ln, t)[:, None]
        t *= 0.96
    P -= P.min(0)
    span = P.max(0)
    span[span == 0] = 1
    return P / span


@chart("network_diagram", "Network diagram", "Geospatial & other", _EDGE_ROLES,
       [Param("max_links", "Max links", "int", 150, min=5, max=1000), Param("labels", "Show node labels", "bool", True),
        Param("directed", "Draw arrowheads", "bool", False)], aliases=("node-link diagram", "graph", "force-directed graph"))
def network_diagram(fig, d, style, max_links=150, labels=True, directed=False):
    """Force-directed (Fruchterman-Reingold, seeded) layout; node size = weighted degree, line width = weight."""
    nodes, edges, notes = edge_table(d, max_links)
    d.notes.extend(notes)
    if not edges:
        return empty_message(fig, style, "No links (need From, To and positive weights)")
    idx = {n: i for i, n in enumerate(nodes)}
    E = np.array([(idx[a], idx[b]) for a, b, _ in edges])
    W = np.array([w for *_, w in edges])
    P = fr_layout(len(nodes), E, W)
    deg = np.zeros(len(nodes))
    np.add.at(deg, E[:, 0], W)
    np.add.at(deg, E[:, 1], W)
    ax = make_axes(fig, style)
    for (a, b), w in zip(E, W):
        if directed:
            ax.annotate("", xy=P[b], xytext=P[a], arrowprops=dict(arrowstyle="-|>", color=style.text_color, alpha=0.4,
                                                                  lw=0.5 + 3 * w / W.max(), shrinkA=6, shrinkB=6))
        else:
            ax.plot(*zip(P[a], P[b]), color=style.text_color, alpha=0.3, lw=0.5 + 3 * w / W.max(), zorder=1)
    ax.scatter(P[:, 0], P[:, 1], s=40 + 500 * deg / deg.max(), c=[style.color_for(n, i) for i, n in enumerate(nodes)],
               edgecolor=style.background, zorder=2, alpha=style.alpha)
    if labels and len(nodes) <= 80:
        for n, (x, y) in zip(nodes, P):
            ax.text(x, y, short(n, 14), fontsize=style.value_label_size, ha="center", va="bottom", color=style.text_color, zorder=3)
    ax.margins(0.08)
    ax.axis("off")
    finish(fig, ax, style, legend=False, axes_decor=False)
    return ax


# =========================================================================== flowchart
@chart("flowchart", "Flowchart", "Geospatial & other",
       [Role("source", "categorical", "From step"), Role("target", "categorical", "To step")],
       [Param("max_steps", "Max steps", "int", 40, min=3, max=150)], aliases=("process diagram", "flow chart"))
def flowchart(fig, d, style, max_steps=40):
    """Top-to-bottom process diagram: one box per step, arrows follow the From -> To pairs."""
    from matplotlib.patches import FancyBboxPatch
    d2 = type(d)({"source": d["source"], "target": d["target"]}, d.names)
    nodes, edges, notes = edge_table(d2, max_edges=max_steps * 2)
    d.notes.extend(notes)
    if not edges:
        return empty_message(fig, style, "No steps (need From and To)")
    level = layer_levels(nodes, edges)
    nl = max(level.values()) + 1
    layers = _order_layers([[n for n in nodes if level[n] == i] for i in range(nl)], edges)
    width = max(len(L) for L in layers)
    pos = {}
    for i, L in enumerate(layers):
        for j, n in enumerate(L):
            pos[n] = ((j + 0.5 - len(L) / 2) * 2.6, -i * 1.6)
    ax = make_axes(fig, style)
    hw, hh = 1.05, 0.38
    for a, b, _ in edges:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        if y0 == y1:                                    # same layer (cycle edge): side-to-side
            ax.annotate("", xy=(x1 - np.sign(x1 - x0) * hw, y1), xytext=(x0 + np.sign(x1 - x0) * hw, y0),
                        arrowprops=dict(arrowstyle="-|>", color=style.text_color, lw=1.2))
        else:
            up = y1 > y0
            ax.annotate("", xy=(x1, y1 + (-hh if up else hh)), xytext=(x0, y0 + (hh if up else -hh)),
                        arrowprops=dict(arrowstyle="-|>", color=style.text_color, lw=1.2,
                                        connectionstyle="arc3,rad=0.0" if abs(x1 - x0) < 0.1 else "arc3,rad=0.12"))
    for i, n in enumerate(nodes):
        x, y = pos[n]
        ax.add_patch(FancyBboxPatch((x - hw, y - hh), 2 * hw, 2 * hh, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor=style.color_for(n, level[n]), edgecolor=style.text_color, lw=1.0, alpha=0.9))
        ax.text(x, y, short(n, 22), ha="center", va="center", fontsize=style.value_label_size + 1, color="#111111", wrap=True)
    ax.set_xlim(-width * 1.4 - 1, width * 1.4 + 1)
    ax.set_ylim(-(nl - 1) * 1.6 - 1, 1)
    ax.axis("off")
    finish(fig, ax, style, legend=False, axes_decor=False)
    return ax
