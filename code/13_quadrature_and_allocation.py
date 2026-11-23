# -*- coding: utf-8 -*-
"""Quadrature step and the allocation sweep behind Table 6.

Part one checks that the quadrature step is immaterial: the support is rebuilt with pieces of
12.5 m, 25 m and 50 m, the 160 m grid operator is applied to each, and the allocation it induces is
compared with the identity allocation of the same support.

Part two is the sweep in Table 6. Operators from three families are applied to the same network at
25 m: grid publication at five spacings, Gaussian jitter at five per-axis standard deviations, and
the nearest-vertex operator of Section 5.5. For each, the mean positional displacement, the total
variation distance between the operated allocation and the identity allocation, the concentration
shift in null standard deviations, and the size of the direct test are reported. The matched null is
also evaluated for one stochastic operator, to show that the remedy transfers.

Outputs go to quadrature.json and allocation_table.json.
"""
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
OUT.mkdir(parents=True, exist_ok=True)

STREET = {"residential", "primary", "secondary", "tertiary", "unclassified",
          "living_street", "trunk"}


def gini(y):
    h = np.bincount(y)
    v = np.arange(len(h))
    end = np.cumsum(h)
    start = end - h + 1
    return float(np.sum(v * h * (start + end)) / (len(y) * y.sum()) - (len(y) + 1) / len(y))


def build(step):
    feats = json.load(open(RAW / "chicago_roads.geojson", encoding="utf8"))["features"]
    kept, excluded, nodes = {}, 0, set()
    for f in feats:
        hw = f["properties"].get("highway")
        hw = hw if isinstance(hw, list) else [hw]
        if not any(h in STREET for h in hw):
            excluded += 1
            continue
        if f["geometry"]["type"] != "LineString":
            continue
        cs = tuple(tuple(x[:2]) for x in f["geometry"]["coordinates"])
        kept.setdefault(min(cs, cs[::-1]), f)
        nodes.update(cs)
    lats = [c[1] for cs in kept for c in kept[cs]["geometry"]["coordinates"]]
    scale = np.array([111320 * np.cos(np.deg2rad(float(np.mean(lats)))), 110540.])
    A, Z, owner, lengths = [], [], [], []
    for i, cs in enumerate(kept):
        pts = np.array(cs) * scale
        total = 0.0
        for a, z in zip(pts[:-1], pts[1:]):
            length = np.linalg.norm(z - a)
            if length <= 0:
                continue
            n = int(int(np.ceil(length / step)))
            edges = a + (z - a) * np.linspace(0, 1, n + 1)[:, None]
            A.extend(edges[:-1]); Z.extend(edges[1:]); owner.extend([i] * n)
            total += length
        lengths.append(total)
    A = np.array(A); Z = np.array(Z); owner = np.array(owner); L = np.array(lengths)
    mid = (A + Z) / 2
    vec = Z - A
    return dict(A=A, Z=Z, owner=owner, L=L, mid=mid, vec=vec,
                sq=np.sum(vec * vec, axis=1), w=np.linalg.norm(vec, axis=1),
                tree=cKDTree(mid), verts=np.array(sorted(nodes)) * scale,
                raw=len(feats), excluded=excluded)


def match(sup, points):
    A, Z, vec, sq, tree, owner = sup["A"], sup["Z"], sup["vec"], sup["sq"], sup["tree"], sup["owner"]
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


def alloc(sup, mp):
    q = np.bincount(mp, weights=sup["w"], minlength=len(sup["L"]))
    return q / q.sum()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    a = ap.parse_args()
    reps, nulls, N = a.reps, 999, 73254
    rng = np.random.default_rng(20261118)

    # ---- part one: does the quadrature step matter? ----
    print("  quadrature", flush=True)
    quad = {}
    for step in (12.5, 25.0, 50.0):
        t0 = time.time()
        sup = build(step)
        q_id = alloc(sup, sup["owner"])
        q_160 = alloc(sup, match(sup, np.round(sup["mid"] / 160) * 160))
        bank = np.array([gini(rng.multinomial(N, q_id)) for _ in range(nulls)])
        obs = np.array([gini(rng.multinomial(N, q_160)) for _ in range(reps)])
        size = float(((1 + (bank[:, None] >= obs).sum(axis=0)) / (nulls + 1.0) <= 0.05).mean())
        quad[str(step)] = dict(pieces=int(len(sup["mid"])), units=int(len(sup["L"])),
                               km=float(sup["L"].sum() / 1000),
                               tv=float(0.5 * np.abs(q_160 - q_id).sum()),
                               null_mean=float(bank.mean()), size=size)
        print("    step %5.1f m | pieces %7d | TV %.5f | null mean %.4f | size %.3f  [%.0f s]"
              % (step, len(sup["mid"]), quad[str(step)]["tv"], bank.mean(), size, time.time() - t0),
              flush=True)
        if step == 25.0:
            sup = sup
    json.dump(quad, open(OUT / "quadrature.json", "w"), indent=1)

    # ---- part two: the allocation sweep ----
    print("  allocation sweep", flush=True)
    q_id = alloc(sup, sup["owner"])
    bank = np.array([gini(rng.multinomial(N, q_id)) for _ in range(nulls)])
    sd = float(bank.std(ddof=1))
    mid = sup["mid"]
    rows = []

    def evaluate(name, points):
        q = alloc(sup, match(sup, points))
        obs = np.array([gini(rng.multinomial(N, q)) for _ in range(reps)])
        ps = (1 + (bank[:, None] >= obs).sum(axis=0)) / (nulls + 1.0)
        rows.append(dict(name=name, d_pos=float(np.mean(np.linalg.norm(points - mid, axis=1))),
                         tv=float(0.5 * np.abs(q - q_id).sum()),
                         z=float((obs.mean() - bank.mean()) / sd),
                         size=float((ps <= 0.05).mean())))
        r = rows[-1]
        print("    %-22s %8.2f m | TV %.5f | Z %+8.1f | size %.3f"
              % (r["name"], r["d_pos"], r["tv"], r["z"], r["size"]), flush=True)
        return q

    evaluate("identity", mid.copy())
    vdist, vidx = cKDTree(sup["verts"]).query(mid, k=1)
    evaluate("nearest vertex", sup["verts"][vidx])
    for grid in (8, 16, 32, 64, 160):
        evaluate("grid %d m" % grid, np.round(mid / grid) * grid)
    jit = {}
    for sig in (2, 5, 10, 20, 50):
        jit[sig] = evaluate("jitter %d m" % sig, mid + rng.normal(0, sig, mid.shape))

    q50 = jit[50]
    mb = np.array([gini(rng.multinomial(N, q50)) for _ in range(nulls)])
    mo = np.array([gini(rng.multinomial(N, q50)) for _ in range(reps)])
    matched = float(((1 + (mb[:, None] >= mo).sum(axis=0)) / (nulls + 1.0) <= 0.05).mean())
    print("    matched null, jitter 50 m: size %.3f" % matched, flush=True)

    json.dump(dict(reps=reps, nulls=nulls, N=N, sd_null=sd, rows=rows,
                   matched_null_jitter50=matched),
              open(OUT / "allocation_table.json", "w"), indent=1)
    print("  wrote quadrature.json and allocation_table.json")


if __name__ == "__main__":
    main()
