"""Reference-point (nearest-vertex) operator on both networks (Table 7, columns 1 to 4).

Part of the reproducibility package for
"Crime concentration after spatial publication: aligning random baselines with
observation processes".

Data are located through the SPATIAL_DATA environment variable and default to
<repo>/data/raw; outputs are written to <repo>/outputs.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("SPATIAL_DATA", REPO / "data" / "raw"))
OUTDIR = Path(os.environ.get("SPATIAL_OUTPUT", REPO / "outputs"))
OUTDIR.mkdir(parents=True, exist_ok=True)


from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = OUTDIR
rng = np.random.default_rng(20261120)
STREET = {'residential', 'primary', 'secondary', 'tertiary', 'unclassified',
          'living_street', 'trunk'}
N = 73254


def gini(y):
    h = np.bincount(y)
    v = np.arange(len(h))
    end = np.cumsum(h)
    start = end - h + 1
    return (np.sum(v * h * (start + end)) / (len(y) * y.sum()) - (len(y) + 1) / len(y)) \
        if y.sum() else np.nan


def build(src, lat_hint=None):
    feats = json.load(open(src, encoding='utf8'))['features']
    kept = {}
    for f in feats:
        hw = f['properties'].get('highway')
        hw = hw if isinstance(hw, list) else [hw]
        if not any(h in STREET for h in hw):
            continue
        if f['geometry']['type'] != 'LineString':
            continue
        cs = tuple(tuple(x[:2]) for x in f['geometry']['coordinates'])
        kept.setdefault(min(cs, cs[::-1]), f)
    coords = [np.array(kept[cs]['geometry']['coordinates'], dtype=float) for cs in kept]
    lat0 = lat_hint or float(np.mean([c[:, 1].mean() for c in coords]))
    scale = np.array([111320 * np.cos(np.deg2rad(lat0)), 110540.])
    A, Z, owner, lengths, verts = [], [], [], [], []
    for i, cs in enumerate(kept):
        pts = coords[i] * scale
        verts.extend(pts)                      # original OSM nodes
        total = 0.0
        for a, z in zip(pts[:-1], pts[1:]):
            length = np.linalg.norm(z - a)
            if length <= 0:
                continue
            n = int(np.ceil(length / 25))
            edges = a + (z - a) * np.linspace(0, 1, n + 1)[:, None]
            A.extend(edges[:-1]); Z.extend(edges[1:]); owner.extend([i] * n)
            total += length
        lengths.append(total)
    V = np.unique(np.round(np.array(verts), 6), axis=0)
    return (np.array(A), np.array(Z), np.array(owner), np.array(lengths), V)


def run(name, src, lat_hint=None):
    print("\n########## %s ##########" % name, flush=True)
    A, Z, owner, L, V = build(src, lat_hint)
    mid = (A + Z) / 2
    vec = Z - A
    sq = np.sum(vec * vec, axis=1)
    w = np.sqrt(sq)
    tree = cKDTree(mid)
    vtree = cKDTree(V)
    print("units %d | pieces %d | vertices %d | %.0f km" %
          (len(L), len(mid), len(V), L.sum() / 1000), flush=True)

    def match(points):
        out = []
        for off in range(0, len(points), 20000):
            P = points[off:off + 20000]
            dist, ix = tree.query(P, k=16)
            delta = P[:, None, :] - A[ix]
            t = np.clip(np.sum(delta * vec[ix], axis=2) / sq[ix], 0, 1)
            dd = np.linalg.norm(delta - t[:, :, None] * vec[ix], axis=2)
            hit = ix[np.arange(len(P)), dd.argmin(axis=1)]
            unc = np.where(dist[:, -1] - 12.5 <= dd.min(axis=1))[0]
            for j in unc:
                cand = np.asarray(tree.query_ball_point(P[j], float(dd[j].min() + 12.500001)))
                de = P[j] - A[cand]
                tt = np.clip(np.sum(de * vec[cand], axis=1) / sq[cand], 0, 1)
                hit[j] = cand[np.argmin(np.linalg.norm(de - tt[:, None] * vec[cand], axis=1))]
            out.extend(owner[hit])
        return np.asarray(out)

    _, vix = vtree.query(mid, k=1)
    snapped = V[vix]
    mr = match(mid)
    mv = match(snapped)
    moved = float(np.mean(mr != mv))
    print("vertex snapping moves %.1f%% of quadrature points to another unit" %
          (100 * moved), flush=True)

    rows = []
    for label, mp in [('match', mr), ('vertex', mv)]:
        q = np.bincount(mp, weights=w, minlength=len(L)); q = q / q.sum()
        null = np.array([gini(rng.multinomial(N, q)) for _ in range(999)])
        for sigma in (0.0, 0.15):
            rate = np.exp(rng.normal(-sigma * sigma / 2, sigma, len(L))) if sigma \
                else np.ones(len(L))
            qq = np.bincount(mp, weights=w * rate[owner], minlength=len(L)); qq = qq / qq.sum()
            st = np.array([gini(rng.multinomial(N, qq)) for _ in range(500)])
            ps = (1 + (null[:, None] >= st).sum(axis=0)) / 1000.0
            r = (ps <= 0.05).mean()
            z = norm.ppf(.975); n = 500
            c = (r + z * z / (2 * n)) / (1 + z * z / n)
            h = z * np.sqrt(r * (1 - r) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            rows.append(dict(network=name, mode=label, sigma=sigma, rejection=r,
                             low=c - h, high=c + h, mean_gini=st.mean(),
                             moved_frac=moved))
            print("  %-6s sigma %.2f rejection %.3f (mean Gini %.3f)" %
                  (label, sigma, r, st.mean()), flush=True)
    return rows


out = []
out += run('Chicago', str(RAW / "chicago_roads.geojson"), 41.88)
out += run('London', str(RAW / "london_roads.geojson"))
df = pd.DataFrame(out)
df.to_csv(OUT / 'vertex_operator.csv', index=False, encoding='utf-8-sig')
print("\nwrote vertex_operator.csv")
print(df.round(3).to_string(index=False))
