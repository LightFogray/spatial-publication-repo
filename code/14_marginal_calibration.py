# -*- coding: utf-8 -*-
"""Marginal calibration for a per-event stochastic operator.

Two mechanisms have to be kept apart, and the manuscript so far only covers the first.

  Fixed field.  One displacement field is drawn for the whole support. Conditionally on that field
  the allocation is a fixed vector q_U, and drawing observed and null data from q_U tests conditional
  calibration.

  Per-event displacement.  Each event draws its own displacement, so the unit an event lands in is
  an i.i.d. categorical draw with probabilities qbar = E_U[q_U]. The count vector is then
  Multinomial(N, qbar), and the matched null must use qbar, not any single field's allocation.

Estimating qbar by averaging the allocations induced by K fields is valid only for the second
mechanism, because the average of the allocations is the mean of the categorical probabilities. It
would be wrong for the first, where the marginal over fields is a mixture of multinomials.

This script estimates qbar for Gaussian jitter on the Chicago support, checks that the estimate
settles as K grows, and reports the size of the matched test under three null constructions: the
correct marginal, a single field used as if it were the marginal, and the identity allocation.
"""
import argparse
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


def build():
    feats = json.load(open(RAW / "chicago_roads.geojson", encoding="utf8"))["features"]
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
    return dict(A=A, Z=Z, owner=owner, L=L, mid=mid, vec=vec,
                sq=np.sum(vec * vec, axis=1), w=np.linalg.norm(vec, axis=1),
                tree=cKDTree(mid))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=int, default=100)
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--nulls", type=int, default=999)
    ap.add_argument("--N", type=int, default=73254)
    a = ap.parse_args()
    sig, N, rng = a.sigma, a.N, np.random.default_rng(20261118)

    t0 = time.time()
    sup = build()
    mid, w, L = sup["mid"], sup["w"], sup["L"]
    q0 = np.bincount(sup["owner"], weights=w, minlength=len(L)); q0 /= q0.sum()
    print("  support %d units, %d pieces  [%.0f s]" % (len(L), len(mid), time.time() - t0), flush=True)

    # qbar = E_U[q_U], estimated by averaging the allocations of K independent fields
    acc = np.zeros(len(L))
    snapshots, first_q = {}, None
    for k in range(a.fields):
        mp = match(sup, mid + rng.normal(0, sig, mid.shape))
        q = np.bincount(mp, weights=w, minlength=len(L)); q /= q.sum()
        if first_q is None:
            first_q = q.copy()
            first_mp = mp
        acc += q
        if (k + 1) in (10, 25, 50, 100):
            qb = acc / (k + 1)
            snapshots[k + 1] = qb.copy()
            print("    K=%3d  running mean  [%.0f s]" % (k + 1, time.time() - t0), flush=True)
    qbar = acc / a.fields
    for k, qb in snapshots.items():
        pass
    tv_to_final = {k: float(0.5 * np.abs(qb - qbar).sum()) for k, qb in snapshots.items()}
    print("  TV between the K-field estimate and the 100-field estimate:")
    for k, v in tv_to_final.items():
        print("    K=%3d  TV %.5f" % (k, v))

    def size(obs_q, null_q, label):
        bank = np.array([gini(rng.multinomial(N, null_q)) for _ in range(a.nulls)])
        obs = np.array([gini(rng.multinomial(N, obs_q)) for _ in range(a.reps)])
        s = float(((1 + (bank[:, None] >= obs).sum(axis=0)) / (a.nulls + 1.0) <= 0.05).mean())
        print("    %-42s size %.3f" % (label, s), flush=True)
        return s

    print("  sizes, observed data drawn from the marginal qbar:", flush=True)
    res = {}
    res["direct_identity_null"] = size(qbar, q0, "direct null (identity allocation)")
    res["matched_marginal"] = size(qbar, qbar, "matched null at the marginal qbar (correct)")
    res["matched_single_field"] = size(qbar, first_q, "matched null at one field (the trap)")
    res["tv_qbar_q0"] = float(0.5 * np.abs(qbar - q0).sum())
    res["tv_qbar_field1"] = float(0.5 * np.abs(qbar - first_q).sum())
    res["tv_convergence"] = tv_to_final
    res["fields"] = a.fields
    res["sigma"] = sig
    json.dump(res, open(OUT / "marginal_calibration.json", "w"), indent=1)
    print("  wrote marginal_calibration.json")


if __name__ == "__main__":
    main()
