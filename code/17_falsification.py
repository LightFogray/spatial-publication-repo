# -*- coding: utf-8 -*-
"""Falsification tests for the claim that a larger reference set lowers the null Gini.

Adding reference points moves the nearest-point map, so the induced allocation can become more or
less concentrated; containment of one reference set in another does not order their Ginis. The
earlier padding experiment only tried one family, road centroids more than 100 m from any published
point, which excludes exactly the case that would break the monotonicity claim: a missing reference
point adjacent to one already in use.

Three tasks:
  A  diagnose the reference points that never become the nearest point to any quadrature location;
  B  test several padding families, including ones designed to concentrate the allocation;
  C  test the non-violent and ASB points on their own, to see whether the union's effect comes from
     categories other than the one being tested.
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

    ev = np.array([[float(r["lat"]), float(r["lon"])] for r in
                   csv.DictReader(open(RAW / "london_allcrime" / "violence_events.csv", encoding="utf-8"))])
    counts = np.array([int(r["n"]) for r in
                       csv.DictReader(open(RAW / "london_allcrime" / "violence_events.csv", encoding="utf-8"))])
    ev = np.column_stack([ev[:, 1], ev[:, 0]]) * scale
    N = int(counts.sum())
    y = np.bincount(match(np.repeat(ev, counts, axis=0)), minlength=len(L))
    obs = gini(y)
    print("  %d incidents, observed Gini %.5f  [%.0f s]" % (N, obs, time.time() - t0), flush=True)

    raw = np.array([[float(r["lat"]), float(r["lon"])] for r in
                    csv.DictReader(open(RAW / "london_allcrime" / "all_2024_points.csv", encoding="utf-8"))])
    union = np.column_stack([raw[:, 1], raw[:, 0]]) * scale

    # ---------- A: why do some reference points never win? ----------
    stree = cKDTree(union)
    _, idx = stree.query(mid, k=1)
    used = np.unique(idx)
    unused = np.setdiff1d(np.arange(len(union)), used)
    tree_pts = cKDTree(mid)
    d_unused, _ = tree_pts.query(union[unused], k=1)
    print("\n  A. reference points never nearest to any quadrature location: %d of %d"
          % (len(unused), len(union)), flush=True)
    print("     distance to the network for those points: median %.0f m, 90th %.0f m, max %.0f m"
          % (np.median(d_unused), np.quantile(d_unused, .9), d_unused.max()), flush=True)
    print("     of them, %d lie beyond 100 m of the network"
          % int((d_unused > 100).sum()), flush=True)

    # ---------- C: the non-violent and ASB points on their own ----------
    viol = np.array([[float(r["lat"]), float(r["lon"])] for r in
                     csv.DictReader(open(RAW / "london_allcrime" / "violence_events.csv", encoding="utf-8"))])
    viol = np.column_stack([viol[:, 1], viol[:, 0]]) * scale
    vtree = cKDTree(viol)
    dmin, _ = vtree.query(union, k=1)
    nonviol = union[dmin > 1e-6]
    print("\n  C. union %d, violent %d, points new to the union %d"
          % (len(union), len(viol), len(nonviol)), flush=True)

    rng = np.random.default_rng(20261118)

    def report(label, ref):
        st = cKDTree(ref)
        _, ix = st.query(mid, k=1)
        q = np.bincount(match(ref[ix]), weights=w, minlength=len(L)); q /= q.sum()
        null = np.array([gini(rng.multinomial(N, q)) for _ in range(999)])
        p = float((1 + (null >= obs).sum()) / 1000.0)
        print("    %-44s %6d pts | null mean %.4f | p %.3f"
              % (label, len(ref), null.mean(), p), flush=True)
        return dict(label=label, points=int(len(ref)), null_mean=float(null.mean()), p=p)

    print("\n  B1. reference sets", flush=True)
    rows = [report("violent subset", viol), report("all-crime union", union),
            report("non-violent and ASB only", nonviol)]

    # ---------- B: padding families, including adversarial ones ----------
    utree = cKDTree(union)
    d_far, _ = utree.query(mid, k=1)
    far = mid[d_far > 100.0]                       # the earlier family
    near = mid[d_far <= 100.0]                     # the case the earlier family excluded
    # concentrate: quadrature locations in the units that already carry most of the allocation
    unit_of_mid = match(union[idx])          # one unit per quadrature location
    q_union = np.bincount(unit_of_mid, weights=w, minlength=len(L))
    heavy = np.argsort(q_union)[-max(1000, len(q_union) // 50):]
    heavy_mask = np.isin(owner, heavy)
    heavy_pts = mid[heavy_mask]
    print("\n  B2. padding families (%d far, %d near, %d in heavy units)"
          % (len(far), len(near), len(heavy_pts)), flush=True)
    rows += [report("union + 5,000 far (>100 m)", np.vstack([union, far[rng.choice(len(far), 5000, replace=False)]])),
             report("union + 5,000 near (<=100 m)", np.vstack([union, near[rng.choice(len(near), 5000, replace=False)]])),
             report("union + 5,000 in heavy units", np.vstack([union, heavy_pts[rng.choice(len(heavy_pts), 5000, replace=False)]])),
             report("union + 10,000 near (<=100 m)", np.vstack([union, near[rng.choice(len(near), 10000, replace=False)]])),
             report("union + 10,000 in heavy units", np.vstack([union, heavy_pts[rng.choice(len(heavy_pts), 10000, replace=False)]]))]

    json.dump(dict(N=N, observed_gini=obs,
                   unused_reference_points=len(unused),
                   unused_median_distance=float(np.median(d_unused)),
                   rows=rows), open(OUT / "falsification.json", "w"), indent=1)
    print("  wrote falsification.json")


if __name__ == "__main__":
    main()
