"""London replication of the equal-risk calibration (Tables 5 and 6).

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
rng = np.random.default_rng(20261119)
STREET = {'residential', 'primary', 'secondary', 'tertiary', 'unclassified',
          'living_street', 'trunk'}
SRC = str(RAW / "london_roads.geojson")


def gini(y):
    h = np.bincount(y)
    v = np.arange(len(h))
    end = np.cumsum(h)
    start = end - h + 1
    return (np.sum(v * h * (start + end)) / (len(y) * y.sum()) - (len(y) + 1) / len(y)) \
        if y.sum() else np.nan


print("loading", SRC, flush=True)
feats = json.load(open(SRC, encoding='utf8'))['features']
kept, excluded = {}, 0
for f in feats:
    hw = f['properties'].get('highway')
    hw = hw if isinstance(hw, list) else [hw]
    if not any(h in STREET for h in hw):
        excluded += 1
        continue
    if f['geometry']['type'] != 'LineString':
        continue
    cs = tuple(tuple(x[:2]) for x in f['geometry']['coordinates'])
    kept.setdefault(min(cs, cs[::-1]), f)
print("raw %d | excluded %d | canonical %d" % (len(feats), excluded, len(kept)), flush=True)

# project around London's centroid
lats = [c[1] for cs in kept for c in kept[cs]['geometry']['coordinates']]
lat0 = float(np.mean(lats))
scale = np.array([111320 * np.cos(np.deg2rad(lat0)), 110540.])

A, Z, owner, lengths = [], [], [], []
for i, cs in enumerate(kept):
    pts = np.array(cs) * scale
    total = 0.0
    for a, z in zip(pts[:-1], pts[1:]):
        length = np.linalg.norm(z - a)
        if length <= 0:
            continue
        n = int(int(np.ceil(length / 25)))
        edges = a + (z - a) * np.linspace(0, 1, n + 1)[:, None]
        A.extend(edges[:-1]); Z.extend(edges[1:]); owner.extend([i] * n)
        total += length
    lengths.append(total)
A = np.array(A); Z = np.array(Z); owner = np.array(owner); L = np.array(lengths)
mid = (A + Z) / 2
vec = Z - A
sq = np.sum(vec * vec, axis=1)
w = np.sqrt(sq)
print("units %d | pieces %d | network %.0f km" % (len(L), len(mid), L.sum() / 1000), flush=True)
tree = cKDTree(mid)


def match(points):
    out = []
    for off in range(0, len(points), 20000):
        P = points[off:off + 20000]
        dist, ix = tree.query(P, k=16)
        delta = P[:, None, :] - A[ix]
        t = np.clip(np.sum(delta * vec[ix], axis=2) / sq[ix], 0, 1)
        dd = np.linalg.norm(delta - t[:, :, None] * vec[ix], axis=2)
        best = dd.argmin(axis=1)
        hit = ix[np.arange(len(P)), best]
        unc = np.where(dist[:, -1] - 12.5 <= dd.min(axis=1))[0]
        for j in unc:
            cand = np.asarray(tree.query_ball_point(P[j], float(dd[j].min() + 12.500001)))
            de = P[j] - A[cand]
            tt = np.clip(np.sum(de * vec[cand], axis=1) / sq[cand], 0, 1)
            hit[j] = cand[np.argmin(np.linalg.norm(de - tt[:, None] * vec[cand], axis=1))]
        out.extend(owner[hit])
        print("   matched %d/%d" % (min(off + 20000, len(points)), len(points)), flush=True)
    return np.asarray(out)


mappings = {'identity': owner}
m = match(mid)
mappings['match'] = m
ident_disagree = float(np.average(m != owner, weights=w))
print("identity disagreement: %.6f" % ident_disagree, flush=True)
for grid in (60, 160, 250):
    mappings[str(grid)] = match(np.round(mid / grid) * grid)
    print("grid %d mapped" % grid, flush=True)

N = 73254
base = None
nulls = {}
for k, mp in mappings.items():
    q = np.bincount(mp, weights=w, minlength=len(L)); q = q / q.sum()
    nulls[k] = np.array([gini(rng.multinomial(N, q)) for _ in range(999)])
    print("null bank %s mean Gini %.4f" % (k, nulls[k].mean()), flush=True)
base = nulls['identity']


def rej(mp, null, rate, R):
    q = np.bincount(mp, weights=w * rate[owner], minlength=len(L)); q = q / q.sum()
    st = np.array([gini(rng.multinomial(N, q)) for _ in range(R)])
    ps = (1 + (null[:, None] >= st).sum(axis=0)) / 1000.0
    r = (ps <= 0.05).mean()
    z = norm.ppf(.975); n = R
    c = (r + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(r * (1 - r) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return r, c - h, c + h, st.mean()


rows = []
for sigma in (0.0, 0.10, 0.15, 0.20):
    rate = np.exp(rng.normal(-sigma * sigma / 2, sigma, len(L))) if sigma else np.ones(len(L))
    for k, mp in mappings.items():
        for method, null in (('length', base), ('operator', nulls[k])):
            r, lo, hi, mg = rej(mp, null, rate, 500)
            rows.append(dict(network='London', sigma=sigma, mode=k, method=method,
                             rejection=r, low=lo, high=hi, mean_gini=mg))
            print("  sigma %.2f %-9s %-8s %.3f" % (sigma, k, method, r), flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / 'london_replication.csv', index=False, encoding='utf-8-sig')
json.dump(dict(network='London', raw_records=len(feats), excluded=excluded,
               units=len(L), pieces=len(mid), km=float(L.sum() / 1000),
               identity_disagreement=ident_disagree, seed=20261119),
          open(OUT / 'london_support.json', 'w'), indent=2)

print("\n=== London equal risk ===")
print(df[df.sigma == 0].pivot(index='mode', columns='method',
                              values='rejection').round(3).to_string())
print("\n=== London power ===")
print(df[df.method == 'operator'].pivot(index='sigma', columns='mode',
                                        values='rejection').round(3).to_string())
