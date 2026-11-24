# -*- coding: utf-8 -*-
"""Is the rejection from the observed union robust to the points it is still missing?

The all-crime union holds the points that carried at least one published event in 2024. The master
list also holds points that carried none, and those are missing. Whether the union's p = 0.001 can
be trusted turns on which way those missing points move the null: if adding reference points makes
the null less concentrated, the observed statistic becomes more exceptional and the rejection
strengthens, so the p computed on the union is conservative.

This script pads the union with random road centroid points, as Section 5.8 does, and tracks the
null mean and the p-value as the reference set grows. The padding points are not the missing master
list points; they stand in for them by entering the reference set without any association with
crime.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("SPATIAL_DATA", REPO / "data" / "raw"))
OUT = Path(os.environ.get("SPATIAL_OUTPUT", REPO / "outputs"))
STREET = {"residential", "primary", "secondary", "tertiary", "unclassified",
          "living_street", "trunk"}


def gini(y):
    h = np.bincount(y)
    v = np.arange(len(h))
    end = np.cumsum(h)
    start = end - h + 1
    return float(np.sum(v * h * (start + end)) / (len(y) * y.sum()) - (len(y) + 1) / len(y))


def main():
    t0 = time.time()
    feats = json.load(open(str(RAW / "london_roads.geojson"), encoding="utf8"))["features"]
    kept = {}
    for f in feats:
        hw = f["properties"].get("highway")
        hw = hw if isinstance(hw, list) else [hw]
        if not any(h in STREET for h in hw):
            continue
        if f["geometry"]["type"] != "LineString":
            continue
        cs = tuple(tuple(x[:2]) for x in f["geometry"]["coordinates"])
        kept.setdefault(min(cs, cs[::-1]), f)
    coords = [np.array(kept[cs]["geometry"]["coordinates"], dtype=float) for cs in kept]
    lat0 = float(np.mean([c[:, 1].mean() for c in coords]))
    scale = np.array([111320 * np.cos(np.deg2rad(lat0)), 110540.])
    A, Z, owner, lengths = [], [], [], []
    for i, cs in enumerate(kept):
        pts = coords[i] * scale
        total = 0.0
        for a, z in zip(pts[:-1], pts[1:]):
            L = np.linalg.norm(z - a)
            if L <= 0:
                continue
            n = int(np.ceil(L / 25))
            edges = a + (z - a) * np.linspace(0, 1, n + 1)[:, None]
            A.extend(edges[:-1]); Z.extend(edges[1:]); owner.extend([i] * n)
            total += L
        lengths.append(total)
    A = np.array(A); Z = np.array(Z); owner = np.array(owner); L = np.array(lengths)
    mid = (A + Z) / 2
    vec = Z - A
    sq = np.sum(vec * vec, axis=1)
    w = np.sqrt(sq)
    tree = cKDTree(mid)
    print("  support %d units, %d pieces  [%.0f s]" % (len(L), len(mid), time.time() - t0), flush=True)

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

    # observed data: 2024 violence, unchanged
    ev = np.array([[float(r["lat"]), float(r["lon"])] for r in
                   csv.DictReader(open(RAW / "london_allcrime" / "violence_events.csv",
                                       encoding="utf-8"))])
    counts = np.array([int(r["n"]) for r in
                       csv.DictReader(open(RAW / "london_allcrime" / "violence_events.csv",
                                           encoding="utf-8"))])
    ev = np.column_stack([ev[:, 1], ev[:, 0]]) * scale
    N = int(counts.sum())
    y = np.bincount(match(np.repeat(ev, counts, axis=0)), minlength=len(L))
    obs = gini(y)
    print("  %d incidents, observed Gini %.5f  [%.0f s]" % (N, obs, time.time() - t0), flush=True)

    raw = np.array([[float(r["lat"]), float(r["lon"])] for r in
                    csv.DictReader(open(RAW / "london_allcrime" / "all_2024_points.csv",
                                        encoding="utf-8"))])
    union = np.column_stack([raw[:, 1], raw[:, 0]]) * scale

    rng = np.random.default_rng(20261118)
    # candidate padding: quadrature locations far from any observed published point
    utree = cKDTree(union)
    d, _ = utree.query(mid, k=1)
    far = mid[d > 100.0]
    print("  padding candidates (no published point within 100 m): %d" % len(far), flush=True)

    rows = []
    for extra in (0, 5000, 10000, 20000, 40000):
        if extra > len(far):
            extra = len(far)
        pad = far[rng.choice(len(far), extra, replace=False)] if extra else np.empty((0, 2))
        ref = np.vstack([union, pad])
        stree = cKDTree(ref)
        _, idx = stree.query(mid, k=1)
        q = np.bincount(match(ref[idx]), weights=w, minlength=len(L)); q /= q.sum()
        null = np.array([gini(rng.multinomial(N, q)) for _ in range(999)])
        p = float((1 + (null >= obs).sum()) / 1000.0)
        rows.append(dict(extra=extra, points=int(len(ref)), null_mean=float(null.mean()),
                         ratio=float(null.mean() / obs), p=p))
        print("    +%5d padding -> %d points | null mean %.4f | ratio %.4f | p %.3f"
              % (extra, len(ref), null.mean(), null.mean() / obs, p), flush=True)

    json.dump(dict(N=N, observed_gini=obs, rows=rows),
              open(OUT / "union_padding.json", "w"), indent=1)
    print("  wrote union_padding.json")


if __name__ == "__main__":
    main()
