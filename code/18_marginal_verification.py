# -*- coding: utf-8 -*-
"""Check the marginal calibration against a genuinely per-event simulation.

The earlier script drew both the observed data and the null from the same estimated marginal, which
tests self-consistency rather than the mechanism. Two things are done here instead.

First, the equivalence that justifies using a multinomial at all is checked directly: a run of true
per-event simulations, in which every event draws its own displacement, is compared with multinomial
draws at the field-averaged allocation. If the two produce the same distribution of the statistic,
the multinomial representation is sound.

Second, the null is estimated from an independent batch of fields, so the observed data and the null
do not share the same plug-in estimate.
"""
import csv
import io
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
    feats = json.load(open(str(RAW / "chicago_roads.geojson"), encoding="utf8"))["features"]
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
    lats = [c[1] for cs in kept for c in kept[cs]["geometry"]["coordinates"]]
    scale = np.array([111320 * np.cos(np.deg2rad(float(np.mean(lats)))), 110540.])
    A, Z, owner, lengths = [], [], [], []
    for i, cs in enumerate(kept):
        pts = np.array(cs) * scale
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
    w = np.linalg.norm(vec, axis=1)
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

    N, sig = 73254, 5.0
    rng = np.random.default_rng(20261118)
    p = w / w.sum()

    def marginal(fields):
        acc = np.zeros(len(L))
        for _ in range(fields):
            mp = match(mid + rng.normal(0, sig, mid.shape))
            q = np.bincount(mp, weights=w, minlength=len(L)); q /= q.sum()
            acc += q
        return acc / fields

    print("  estimating two independent marginals, 50 fields each", flush=True)
    qA, qB = marginal(50), marginal(50)
    print("    total variation between the two estimates: %.5f  [%.0f s]"
          % (0.5 * np.abs(qA - qB).sum(), time.time() - t0), flush=True)

    # ---- equivalence: true per-event simulation against multinomial at qA ----
    reps = 40
    sim, mul = [], []
    for _ in range(reps):
        idx = rng.choice(len(mid), N, p=p)
        pts = mid[idx] + rng.normal(0, sig, (N, 2))     # every event draws its own displacement
        sim.append(gini(np.bincount(match(pts), minlength=len(L))))
        mul.append(gini(rng.multinomial(N, qA)))
    sim, mul = np.array(sim), np.array(mul)
    print("    true per-event simulation: mean %.5f, sd %.5f" % (sim.mean(), sim.std(ddof=1)), flush=True)
    print("    multinomial at the marginal: mean %.5f, sd %.5f" % (mul.mean(), mul.std(ddof=1)), flush=True)
    print("    difference of means %.5f  [%.0f s]" % (mul.mean() - sim.mean(), time.time() - t0), flush=True)

    # ---- size with an independently estimated null ----
    bank = np.array([gini(rng.multinomial(N, qB)) for _ in range(999)])
    obs = np.array([gini(rng.multinomial(N, qA)) for _ in range(500)])
    size = float(((1 + (bank[:, None] >= obs).sum(axis=0)) / 1000.0 <= 0.05).mean())
    print("    matched size, observed from qA and null from qB: %.3f" % size, flush=True)

    json.dump(dict(sigma=sig, N=N, fields=50, tv_between_marginals=float(0.5 * np.abs(qA - qB).sum()),
                   sim_mean=float(sim.mean()), sim_sd=float(sim.std(ddof=1)),
                   mult_mean=float(mul.mean()), mult_sd=float(mul.std(ddof=1)),
                   size_independent_null=size),
              open(OUT / "marginal_verification.json", "w"), indent=1)
    print("  wrote marginal_verification.json")


if __name__ == "__main__":
    main()
