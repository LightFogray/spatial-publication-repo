"""Detection boundary and grid-origin sensitivity (Tables 3 and 4).

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
rng = np.random.default_rng(20261118)
STREET = {'residential', 'primary', 'secondary', 'tertiary', 'unclassified',
          'living_street', 'trunk'}


def gini(y):
    h = np.bincount(y)
    v = np.arange(len(h))
    end = np.cumsum(h)
    start = end - h + 1
    return (np.sum(v * h * (start + end)) / (len(y) * y.sum()) - (len(y) + 1) / len(y)) \
        if y.sum() else np.nan


def build_support():
    raw = json.load(open(str(RAW / "chicago_roads.geojson"),
                         encoding='utf8'))['features']
    kept = {}
    for f in raw:
        hw = f['properties'].get('highway')
        hw = hw if isinstance(hw, list) else [hw]
        if not any(h in STREET for h in hw):
            continue
        if f['geometry']['type'] != 'LineString':
            continue
        cs = tuple(tuple(x[:2]) for x in f['geometry']['coordinates'])
        kept.setdefault(min(cs, cs[::-1]), f)
    scale = np.array([111320 * np.cos(np.deg2rad(41.88)), 110540.])
    A, Z, owner, lengths = [], [], [], []
    for i, cs in enumerate(kept):
        pts = np.array(cs) * scale
        total = 0.0
        for a, z in zip(pts[:-1], pts[1:]):
            length = np.linalg.norm(z - a)
            if length <= 0:
                continue
            n = int(np.ceil(length / 25))
            edges = a + (z - a) * np.linspace(0, 1, n + 1)[:, None]
            A.extend(edges[:-1]); Z.extend(edges[1:])
            owner.extend([i] * n); total += length
        lengths.append(total)
    return (np.array(A), np.array(Z), np.array(owner), np.array(lengths))


A, Z, owner, L = build_support()
mid = (A + Z) / 2
vec = Z - A
sq = np.sum(vec * vec, axis=1)
w = np.sqrt(sq)
tree = cKDTree(mid)


def match(points):
    out = []
    for off in range(0, len(points), 5000):
        P = points[off:off + 5000]
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
    return np.asarray(out)


N = 73254
p0 = L / L.sum()


def null_bank(mapping, B=999):
    q = np.bincount(mapping, weights=w, minlength=len(L)); q = q / q.sum()
    return np.array([gini(rng.multinomial(N, q)) for _ in range(B)])


def rej_rate(mapping, null, rate, R=500):
    q = np.bincount(mapping, weights=w * rate[owner], minlength=len(L)); q = q / q.sum()
    stats = np.array([gini(rng.multinomial(N, q)) for _ in range(R)])
    ps = (1 + (null[:, None] >= stats).sum(axis=0)) / 1000.0
    return (ps <= 0.05).mean(), stats.mean()


mappings = {'identity': owner, 'match': match(mid)}
for grid in (60, 160, 250):
    mappings[str(grid)] = match(np.round(mid / grid) * grid)
nulls = {k: null_bank(v) for k, v in mappings.items()}
base_null = nulls['identity']          # direct length baseline, fixed across operators

rows = []

# ---------- A. power curve ----------
SIGMAS = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
for sigma in SIGMAS:
    rate = np.exp(rng.normal(-sigma * sigma / 2, sigma, len(L))) if sigma else np.ones(len(L))
    for mode, mapping in mappings.items():
        for method, null in (('length', base_null), ('operator', nulls[mode])):
            r, mg = rej_rate(mapping, null, rate, R=500)
            z = norm.ppf(.975); n = 500
            c = (r + z * z / (2 * n)) / (1 + z * z / n)
            half = z * np.sqrt(r * (1 - r) / n + z * z / (4 * n * n)) / (1 + z * z / n)
            rows.append(dict(experiment='power', sigma=sigma, mode=mode, method=method,
                             rejection=r, low=c - half, high=c + half, mean_gini=mg))
    print('power sigma %.2f done' % sigma, flush=True)

# ---------- B. grid origin ----------
OFF = 12
for grid in (60, 160, 250):
    for k in range(OFF):
        ox, oy = rng.uniform(0, grid, 2)
        shifted = np.round((mid - [ox, oy]) / grid) * grid + [ox, oy]
        mp = match(shifted)
        null = null_bank(mp, B=999)
        rate = np.ones(len(L))
        r, mg = rej_rate(mp, null, rate, R=200)
        rl, _ = rej_rate(mp, base_null, rate, R=200)
        rows.append(dict(experiment='origin', sigma=0.0, mode=str(grid), method='operator',
                         rejection=r, low=np.nan, high=np.nan, mean_gini=mg,
                         ox=ox, oy=oy, offset_index=k))
        rows.append(dict(experiment='origin', sigma=0.0, mode=str(grid), method='length',
                         rejection=rl, low=np.nan, high=np.nan, mean_gini=mg,
                         ox=ox, oy=oy, offset_index=k))
    print('origin grid %d done' % grid, flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / 'extensions_power_origin.csv', index=False, encoding='utf-8-sig')

print("\n=== A. detection boundary (matched test, operator column) ===")
pw = df[(df.experiment == 'power') & (df.method == 'operator')]
print(pw.pivot(index='sigma', columns='mode', values='rejection').round(3).to_string())
print("\n=== A. direct length test under the same alternatives ===")
pl = df[(df.experiment == 'power') & (df.method == 'length')]
print(pl.pivot(index='sigma', columns='mode', values='rejection').round(3).to_string())

print("\n=== B. grid origin: equal-risk rejection across %d offsets ===" % OFF)
og = df[df.experiment == 'origin']
print(og.groupby(['mode', 'method']).rejection.agg(['mean', 'min', 'max']).round(3).to_string())
