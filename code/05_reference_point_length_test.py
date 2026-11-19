"""Direct length test under the reference-point operator (Table 7, columns 5 and 6).

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

ROOT = Path(__file__).resolve().parents[1]
OUT = OUTDIR
rng = np.random.default_rng(20261121)
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
        verts.extend(pts)
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


rows = []
for name, src, hint in [
        ('Chicago', str(RAW / "chicago_roads.geojson"), 41.88),
        ('London', str(RAW / "london_roads.geojson"), None)]:
    print("###", name, flush=True)
    A, Z, owner, L, V = build(src, hint)
    mid = (A + Z) / 2
    vec = Z - A
    sq = np.sum(vec * vec, axis=1)
    w = np.sqrt(sq)
    tree = cKDTree(mid)
    vtree = cKDTree(V)
    _, vix = vtree.query(mid, k=1)
    snapped = V[vix]

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

    base_map = match(mid)
    vert_map = match(snapped)
    q0 = np.bincount(base_map, weights=w, minlength=len(L)); q0 = q0 / q0.sum()
    base_null = np.array([gini(rng.multinomial(N, q0)) for _ in range(999)])
    qv = np.bincount(vert_map, weights=w, minlength=len(L)); qv = qv / qv.sum()
    vert_null = np.array([gini(rng.multinomial(N, qv)) for _ in range(999)])

    rate = np.ones(len(L))
    qq = np.bincount(vert_map, weights=w * rate[owner], minlength=len(L)); qq = qq / qq.sum()
    st = np.array([gini(rng.multinomial(N, qq)) for _ in range(500)])
    p_len = (1 + (base_null[:, None] >= st).sum(axis=0)) / 1000.0
    p_mat = (1 + (vert_null[:, None] >= st).sum(axis=0)) / 1000.0
    r_len = float((p_len <= 0.05).mean())
    r_mat = float((p_mat <= 0.05).mean())
    print("  direct length rejection %.3f | matched rejection %.3f | mean Gini %.3f "
          "(null %.3f)" % (r_len, r_mat, st.mean(), base_null.mean()), flush=True)
    rows.append(dict(network=name, direct_length_rejection=r_len,
                     matched_rejection=r_mat, mean_gini=float(st.mean()),
                     length_null_mean=float(base_null.mean()),
                     length_null_sd=float(base_null.std())))

pd.DataFrame(rows).to_csv(OUT / 'vertex_length_test.csv', index=False, encoding='utf-8-sig')
print("\nwrote vertex_length_test.csv")
