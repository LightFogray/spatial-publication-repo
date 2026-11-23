"""Re-estimate the identity-condition size with an independent null bank per replication.

Part of the reproducibility package for "Crime concentration after spatial publication".
The calibration shares one 999-draw bank across the 500 replications in a condition, which makes
the p-values dependent; this repeats the identity condition with a fresh bank per replication to
show that the reported size is Monte Carlo variation rather than a distorted level. Reports to
Supplementary Text S4.
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("SPATIAL_OUTPUT", REPO / "outputs"))


def gini(y):
    h = np.bincount(y)
    v = np.arange(len(h))
    end = np.cumsum(h)
    start = end - h + 1
    return (np.sum(v * h * (start + end)) / (len(y) * y.sum()) - (len(y) + 1) / len(y))


def wilson(rate, n, z=1.959963984540054):
    centre = (rate + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--nulls", type=int, default=999)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(OUT / "chicago_canonical_counts.csv", encoding="utf-8")))
    L = np.array([float(r["length_m"]) for r in rows])
    cnt = np.array([int(r["count"]) for r in rows])
    N = int(cnt.sum())
    q = L / L.sum()
    print("  support: %d units, %d events" % (len(L), N))

    rng = np.random.default_rng(20261118)

    # --- shared bank, reproducing the published figure -------------------------
    bank = np.array([gini(rng.multinomial(N, q)) for _ in range(a.nulls)])
    stats = np.array([gini(rng.multinomial(N, q)) for _ in range(a.reps)])
    ps = (1 + (bank[:, None] >= stats).sum(axis=0)) / (a.nulls + 1)
    shared = float((ps <= 0.05).mean())
    lo, hi = wilson(shared, a.reps)
    print("  shared bank      : size %.3f  (95%% CI %.3f to %.3f)" % (shared, lo, hi))

    # --- one fresh bank per replication ---------------------------------------
    ps2 = np.empty(a.reps)
    for k in range(a.reps):
        b = np.array([gini(rng.multinomial(N, q)) for _ in range(a.nulls)])
        ps2[k] = (1 + (b >= stats[k]).sum()) / (a.nulls + 1)
        if (k + 1) % 100 == 0:
            print("    %d / %d" % (k + 1, a.reps), flush=True)
    independent = float((ps2 <= 0.05).mean())
    lo2, hi2 = wilson(independent, a.reps)
    print("  independent banks: size %.3f  (95%% CI %.3f to %.3f)" % (independent, lo2, hi2))

    import json
    json.dump(dict(units=int(len(L)), events=N, reps=a.reps, nulls=a.nulls,
                   shared=shared, shared_ci=[lo, hi],
                   independent=independent, independent_ci=[lo2, hi2]),
              open(OUT / "identity_bank_check.json", "w"), indent=1)
    print("  wrote identity_bank_check.json")


if __name__ == "__main__":
    main()
