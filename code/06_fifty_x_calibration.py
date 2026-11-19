"""50-X calibration on both networks (Table 8).

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
rng = np.random.default_rng(20261125)
STREET = {'residential', 'primary', 'secondary', 'tertiary', 'unclassified',
          'living_street', 'trunk'}


def share50(y):
    """percentage of units, taken from the most concentrated, holding 50% of the total"""
    y = np.asarray(y, float)
    tot = y.sum()
    if tot <= 0:
        return np.nan
    s = np.sort(y)[::-1]
    c = np.cumsum(s)
    k = int(np.searchsorted(c, 0.5 * tot - 1e-9) + 1)
    return 100.0 * k / len(y)


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
    print("\n########## %s ##########" % name, flush=True)
    A, Z, owner, L, V = build(src, hint)
    mid = (A + Z) / 2
    vec = Z - A
    sq = np.sum(vec * vec, axis=1)
    w = np.sqrt(sq)
    tree = cKDTree(mid)
    N = 73254 if name == 'Chicago' else 254109

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

    maps = {'identity': owner, 'match': match(mid)}
    for g in (60, 160, 250):
        maps[str(g)] = match(np.round(mid / g) * g)
    _, vix = cKDTree(V).query(mid, k=1)
    maps['vertex'] = match(V[vix])
    print("units %d | pieces %d | mean 50-X under identity null pending" % (len(L), len(mid)),
          flush=True)

    for mode, mp in maps.items():
        q = np.bincount(mp, weights=w, minlength=len(L)); q = q / q.sum()
        null = np.array([share50(rng.multinomial(N, q)) for _ in range(999)])
        # equal risk
        st = np.array([share50(rng.multinomial(N, q)) for _ in range(500)])
        p_eq = (1 + (null[:, None] <= st).sum(axis=0)) / 1000.0
        # power at sigma 0.15
        rate = np.exp(rng.normal(-0.15 ** 2 / 2, 0.15, len(L)))
        qh = np.bincount(mp, weights=w * rate[owner], minlength=len(L)); qh = qh / qh.sum()
        sth = np.array([share50(rng.multinomial(N, qh)) for _ in range(500)])
        p_h = (1 + (null[:, None] <= sth).sum(axis=0)) / 1000.0
        rows.append(dict(network=name, mode=mode, null_mean=float(null.mean()),
                         eq_size=float((p_eq <= 0.05).mean()),
                         power_015=float((p_h <= 0.05).mean()),
                         het_mean=float(sth.mean())))
        print("  %-8s null 50-X %.3f%% | size %.3f | power(0.15) %.3f | het mean %.3f%%" %
              (mode, null.mean(), (p_eq <= 0.05).mean(), (p_h <= 0.05).mean(), sth.mean()),
              flush=True)

    # the direct length baseline never sees the operator
    q0 = L / L.sum()
    null0 = np.array([share50(rng.multinomial(N, q0)) for _ in range(999)])
    for mode, mp in maps.items():
        q = np.bincount(mp, weights=w, minlength=len(L)); q = q / q.sum()
        st = np.array([share50(rng.multinomial(N, q)) for _ in range(500)])
        p = (1 + (null0[:, None] <= st).sum(axis=0)) / 1000.0
        rows.append(dict(network=name, mode=mode, null_mean=float(null0.mean()),
                         eq_size=float((p <= 0.05).mean()), power_015=np.nan,
                         het_mean=float(st.mean()), baseline='length'))
        print("  %-8s [length null %.3f%%] direct-test size %.3f" %
              (mode, null0.mean(), (p <= 0.05).mean()), flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "fifty_x_calibration.csv", index=False, encoding="utf-8-sig")
print("\nwrote fifty_x_calibration.csv")
