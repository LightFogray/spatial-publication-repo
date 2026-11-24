# -*- coding: utf-8 -*-
"""Recover a larger observed reference subset from the 2024 London release.

The master list of anonymous map points is withheld, but every point that carried at least one
published event appears in the release. Using only violent crime gives the 47,533 points the paper
works with. Adding every other crime type and anti-social behaviour, month by month, can only add
points, and the union is a reference set closer to the master list while still consisting only of
points the publisher actually used.

Two facts about the source make this cheap. A monthly archive is cumulative, so the December
archive holds every month of 2024 and the member list is read once. And the server supports byte
ranges, so only the Metropolitan street file is fetched, about 20 MB per month rather than 1.75 GB.

Each month is cached separately, so an interrupted run resumes where it stopped. The violent-crime
rows are counted on their own, which lets the known figures (254,109 events, 47,533 distinct
points) act as a check that this is the extract the analysis used.
"""
import argparse
import csv
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_zip import fetch_member  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("SPATIAL_OUTPUT", REPO / "outputs"))
OUT.mkdir(parents=True, exist_ok=True)
CACHE = REPO / "data" / "raw" / "london_allcrime"
CACHE.mkdir(parents=True, exist_ok=True)
ARCHIVE = "https://data.police.uk/data/archive/%s.zip"
ARCHIVE_MONTH = "2024-12"
VIOLENCE = "violence and sexual offences"


def month_path(month):
    return CACHE / ("month_%s.csv" % month)


def load_month(month):
    p = month_path(month)
    if not p.exists():
        return None
    pts, n_ev, n_viol = {}, 0, 0
    _totals = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["lat"]:
                nn_ev = int(r["events"]); nn_vi = int(r["viol_events"])
                _totals[month] = (nn_ev, nn_vi)
                continue
            key = (float(r["lat"]), float(r["lon"]))
            pts[key] = max(pts.get(key, 0), int(r["in_violence"]))
            n_ev += int(r["events"])
            n_viol += int(r["viol_events"])
    if month in _totals:
        n_ev, n_viol = _totals[month]
    return pts, n_ev, n_viol


def fetch_month(month):
    name, blob = fetch_member(ARCHIVE % ARCHIVE_MONTH,
                              "%s/%s-metropolitan-street.csv" % (month, month))
    pts, n_ev, n_viol = {}, 0, 0
    for row in csv.DictReader(io.StringIO(blob.decode("utf-8", "replace"))):
        try:
            lat, lon = float(row["Latitude"]), float(row["Longitude"])
        except (TypeError, ValueError):
            continue
        cat = (row.get("Crime type") or "").strip().lower()
        key = (round(lat, 7), round(lon, 7))
        flag = 1 if cat == VIOLENCE else 0
        pts[key] = max(pts.get(key, 0), flag)
        n_ev += 1
        if flag:
            n_viol += 1
    with open(month_path(month), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "in_violence", "events", "viol_events"])
        for key, flag in sorted(pts.items()):
            w.writerow([key[0], key[1], flag, 0, 0])
        w.writerow(["", "", "", n_ev, n_viol])
    return pts, n_ev, n_viol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=["2024-%02d" % m for m in range(1, 13)])
    a = ap.parse_args()

    all_pts, viol_pts = {}, set()
    n_all = n_viol = 0
    for month in a.months:
        t0 = time.time()
        got = load_month(month)
        fresh = got is None
        if fresh:
            try:
                got = fetch_month(month)
            except Exception as e:
                print("  %s FAILED: %s" % (month, e), flush=True)
                continue
        pts, n_ev, n_viol_m = got
        n_all += n_ev
        n_viol += n_viol_m
        for key, flag in pts.items():
            all_pts[key] = max(all_pts.get(key, 0), flag)
            if flag:
                viol_pts.add(key)
        print("  %s %s %6d events | violence %6d | union %d  [%.0f s]"
              % (month, "new " if fresh else "cach", n_ev, n_viol_m, len(all_pts),
                 time.time() - t0), flush=True)

    print("\n  violence and sexual offences: %6d events, %d distinct points"
          % (n_viol, len(viol_pts)))
    print("  all crime and ASB:            %6d events, %d distinct points" % (n_all, len(all_pts)))

    with open(CACHE / "all_2024_points.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lat", "lon", "in_violence"])
        for key, flag in sorted(all_pts.items()):
            w.writerow([key[0], key[1], flag])
    json.dump(dict(events_all=n_all, points_all=len(all_pts), events_violence=n_viol,
                   points_violence=len(viol_pts), months=a.months),
              open(CACHE / "summary.json", "w"), indent=1)
    print("  wrote %s" % (CACHE / "all_2024_points.csv"))


if __name__ == "__main__":
    main()
