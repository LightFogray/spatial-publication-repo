"""Download the Chicago incident extract used by the analysis.

Source: City of Chicago open data, dataset ijzp-q8t2 (Crimes - 2001 to Present), OGL-style
open licence. The query below is the exact one used for the paper: calendar year 2024, records
with coordinates, restricted to the eight local categories that make up the ICCS 01-03 basket.

BATTERY carries the largest share and is easy to drop by accident; an earlier version of this
project used a keyword filter that omitted it.

Usage:  python data/download_chicago_crime.py [--out PATH]
"""
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

KEEP = ["BATTERY", "ASSAULT", "HOMICIDE", "CRIMINAL SEXUAL ASSAULT", "SEX OFFENSE",
        "STALKING", "KIDNAPPING", "INTIMIDATION"]
EXPECTED = 73488          # records before the road-extent clip


def get(q, tries=5, timeout=240):
    url = BASE + "?" + urllib.parse.urlencode(q)
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                        timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:
            print("  retry %d: %r" % (i + 1, e), file=sys.stderr, flush=True)
            time.sleep(3 + 3 * i)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(__file__).resolve().parent / "raw" / "chicago_2024_violent.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    inlist = ",".join("'%s'" % c for c in KEEP)
    where = ("date >= '2024-01-01T00:00:00' AND date < '2025-01-01T00:00:00' "
             "AND latitude IS NOT NULL AND primary_type in(%s)" % inlist)

    counts = get({"$select": "primary_type,count(*) as n", "$group": "primary_type",
                  "$where": where, "$order": "n DESC"}) or []
    total = sum(int(r["n"]) for r in counts)
    print("records by category:")
    for r in counts:
        print("  %-26s %s" % (r["primary_type"], r["n"]))
    print("  total: %d" % total)
    if total != EXPECTED:
        print("  NOTE: expected %d; the published extract grows as late reports arrive."
              % EXPECTED)

    rows, off, page = [], 0, 50000
    while off < total:
        j = get({"$select": "latitude,longitude,primary_type", "$limit": page,
                 "$offset": off, "$where": where})
        if not j:
            break
        rows.extend(j)
        off += len(j)
        print("  fetched %d / %d" % (off, total), flush=True)
        if len(j) < page:
            break

    import csv
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["latitude", "longitude", "primary_type"])
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (out, len(rows)))


if __name__ == "__main__":
    main()
