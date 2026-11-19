"""Empirical application to the London release (Table 9).

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
import glob
import json
import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = OUTDIR
rng = np.random.default_rng(20261122)
STREET = {'residential', 'primary', 'secondary', 'tertiary', 'unclassified',
          'living_street', 'trunk'}
BASE = str(RAW / "london_crime")


def gini(y):
    h = np.bincount(y)
    v = np.arange(len(h))
    end = np.cumsum(h)
    start = end - h + 1
    return (np.sum(v * h * (start + end)) / (len(y) * y.sum()) - (len(y) + 1) / len(y)) \
        if y.sum() else np.nan


print("building London support", flush=True)
feats = json.load(open(str(RAW / "london_roads.geojson"),
                       encoding='utf8'))['features']
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
lat0 = float(np.mean([c[:, 1].mean() for c in coords]))
scale = np.array([111320 * np.cos(np.deg2rad(lat0)), 110540.])
A, Z, owner, lengths = [], [], [], []
for i, cs in enumerate(kept):
    pts = coords[i] * scale
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
A = np.array(A); Z = np.array(Z); owner = np.array(owner); L = np.array(lengths)
mid = (A + Z) / 2
vec = Z - A
sq = np.sum(vec * vec, axis=1)
w = np.sqrt(sq)
tree = cKDTree(mid)
print("units %d | pieces %d | %.0f km" % (len(L), len(mid), L.sum() / 1000), flush=True)


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


print("loading London 2024 violence", flush=True)
frames = []
for f in sorted(glob.glob(os.path.join(BASE, "*", "*.csv"))):
    m = os.path.basename(f)[:7]
    if "2024-01" <= m <= "2024-12":
        frames.append(pd.read_csv(f, usecols=["Latitude", "Longitude", "Crime type"],
                                  low_memory=False))
lc = pd.concat(frames).dropna(subset=["Latitude", "Longitude"])
lc = lc[lc["Crime type"] == "Violence and sexual offences"]
ev = lc[["Longitude", "Latitude"]].to_numpy() * scale
print("events %d" % len(ev), flush=True)

y = np.bincount(match(ev), minlength=len(L))
N = int(y.sum())
g_obs = gini(y)
print("observed Gini %.5f on %d incident(s)" % (g_obs, N), flush=True)

# observed snap points = distinct published coordinates
snap = np.unique(np.round(ev, 2), axis=0)
print("distinct published snap points (in use): %d" % len(snap), flush=True)
stree = cKDTree(snap)
_, s_index = stree.query(mid, k=1)
snapped_support = snap[s_index]
snap_map = match(snapped_support)
q_snap = np.bincount(snap_map, weights=w, minlength=len(L)); q_snap = q_snap / q_snap.sum()
q_len = L / L.sum()

rows = []
for label, q in (("direct length null", q_len), ("snap-point null", q_snap)):
    null = np.array([gini(rng.multinomial(N, q)) for _ in range(999)])
    p = (1 + int((null >= g_obs).sum())) / 1000.0
    rows.append(dict(null=label, null_mean=float(null.mean()), null_sd=float(null.std()),
                     ratio=float(null.mean() / g_obs), p=float(p),
                     lower_tail=float((null < g_obs).mean())))
    print("  %-20s mean %.5f  ratio %.4f  p %.3f" %
          (label, null.mean(), null.mean() / g_obs, p), flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "london_empirical.csv", index=False, encoding="utf-8-sig")
json.dump(dict(network="London", events=N, units=len(L), gini=float(g_obs),
               snap_points_in_use=int(len(snap)), pieces=len(mid), seed=20261122),
          open(OUT / "london_empirical_support.json", "w"), indent=2)
print("\nwrote london_empirical.csv")
print(df.round(4).to_string(index=False))
