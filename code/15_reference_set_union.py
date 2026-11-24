# -*- coding: utf-8 -*-
"""Does a larger observed reference set change the London answer?

The violent-crime release uses 47,533 of the master list's points. Adding every other crime type and
anti-social behaviour for the same year raises the observed subset to 60,897 points, still a subset
of the withheld list but a substantially larger one. This script holds the target events fixed at
the 254,109 violent incidents and asks what the operator-matched test returns when the reference set
is the violent subset, the all-crime union, and the direct length baseline.

The turnover value of 52,533 is not used as a decision rule. It was obtained by padding the
violent subset with random road centroids, and the union is not that experiment; the null is
recomputed from the actual coordinates of each reference set.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from remote_zip import fetch_member  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("SPATIAL_DATA", REPO / "data" / "raw"))
OUT = Path(os.environ.get("SPATIAL_OUTPUT", REPO / "outputs"))
OUT.mkdir(parents=True, exist_ok=True)
STREET = {"residential", "primary", "secondary", "tertiary", "unclassified",
          "living_street", "trunk"}
VIOLENCE = "violence and sexual offences"


def gini(y):
    h = np.bincount(y)
    v = np.arange(len(h))
    end = np.cumsum(h)
    start = end - h + 1
    return float(np.sum(v * h * (start + end)) / (len(y) * y.sum()) - (len(y) + 1) / len(y))


def violence_events():
    """distinct published coordinates of the 2024 violent incidents, with their counts"""
    cache = RAW / "london_allcrime" / "violence_events.csv"
    if cache.exists():
        pts, cnt = [], []
        for r in csv.DictReader(open(cache, encoding="utf-8")):
            pts.append((float(r["lat"]), float(r["lon"])))
            cnt.append(int(r["n"]))
        return np.array(pts), np.array(cnt)
    from collections import Counter
    tally = Counter()
    for m in range(1, 13):
        month = "2024-%02d" % m
        name, blob = fetch_member("https://data.police.uk/data/archive/2024-12.zip",
                                  "%s/%s-metropolitan-street.csv" % (month, month))
        for row in csv.DictReader(io.StringIO(blob.decode("utf-8", "replace"))):
            if (row.get("Crime type") or "").strip().lower() != VIOLENCE:
                continue
            try:
                lat, lon = float(row["Latitude"]), float(row["Longitude"])
            except (TypeError, ValueError):
                continue
            tally[(round(lat, 7), round(lon, 7))] += 1
        print("    %s done, %d points" % (month, len(tally)), flush=True)
    with open(cache, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "n"])
        for (lat, lon), n in sorted(tally.items()):
            w.writerow([lat, lon, n])
    return (np.array([k for k in tally], dtype=float),
            np.array([tally[k] for k in tally], dtype=int))


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
    print("  support %d units, %d pieces, %.0f km  [%.0f s]"
          % (len(L), len(mid), L.sum() / 1000, time.time() - t0), flush=True)

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

    pts, cnt = violence_events()
    N = int(cnt.sum())
    # the caches store (lat, lon); the projection expects (lon, lat) * (metres per degree)
    pts = np.column_stack([pts[:, 1], pts[:, 0]]) * scale
    ev = np.repeat(pts, cnt, axis=0)
    y = np.bincount(match(ev), minlength=len(L))
    obs = gini(y)
    print("  %d incidents at %d distinct points, observed Gini %.5f  [%.0f s]"
          % (N, len(pts), obs, time.time() - t0), flush=True)

    allraw = np.array([[float(r["lat"]), float(r["lon"])] for r in
                       csv.DictReader(open(RAW / "london_allcrime" / "all_2024_points.csv",
                                           encoding="utf-8"))])
    allpts = np.column_stack([allraw[:, 1], allraw[:, 0]]) * scale
    sets = {
        "violent subset": pts,
        "all crime and ASB union": allpts,
    }
    rng = np.random.default_rng(20261118)
    q_len = L / L.sum()
    rows = {}

    def evaluate(label, q):
        null = np.array([gini(rng.multinomial(N, q)) for _ in range(999)])
        p = float((1 + (null >= obs).sum()) / 1000.0)
        rows[label] = dict(points=int(len(q.nonzero()[0])), null_mean=float(null.mean()),
                           null_sd=float(null.std(ddof=1)), ratio=float(null.mean() / obs), p=p)
        print("    %-26s null mean %.4f | ratio %.4f | p %.3f"
              % (label, null.mean(), null.mean() / obs, p), flush=True)

    print("  tests, %d incidents, observed Gini %.5f" % (N, obs), flush=True)
    evaluate("direct length null", q_len)
    for label, setpts in sets.items():
        # snap every quadrature location to its nearest published point, as the release does
        stree = cKDTree(setpts)
        _, idx = stree.query(mid, k=1)
        q = np.bincount(match(setpts[idx]), weights=w, minlength=len(L)); q /= q.sum()
        print("    %s: %d reference points, %d in use on the network"
              % (label, len(setpts), len(np.unique(match(setpts)))), flush=True)
        evaluate("snap-point null, " + label, q)

    json.dump(dict(N=N, distinct_points=int(len(pts)), observed_gini=obs,
                   reference_sets={k: int(len(v)) for k, v in sets.items()}, rows=rows),
              open(OUT / "reference_set_union.json", "w"), indent=1)
    print("  wrote reference_set_union.json")


if __name__ == "__main__":
    main()
