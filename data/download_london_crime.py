"""Download the London incident extract used by the analysis.

Source: data.police.uk monthly archive, published by the UK Home Office under the Open
Government Licence. Each monthly archive is about 1.7 GB and contains one CSV per police force;
only the Metropolitan Police file is needed for the London network, and only the
"Violence and sexual offences" rows are kept.

The full year is therefore a ~20 GB download. Use --months to fetch a subset if you are only
checking the pipeline.

Output layout, which is what the analysis scripts glob for:
    data/raw/london_crime/2024-01/2024-01-metropolitan-street.csv

Usage:  python data/download_london_crime.py [--months 2024-01 2024-02 ...] [--force met]
"""
import argparse
import csv
import io
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ARCHIVE = "https://data.police.uk/data/archive/%s.zip"
UA = {"User-Agent": "Mozilla/5.0"}
VIOLENCE = "Violence and sexual offences"


def fetch_month(month, force, outroot, timeout=1800):
    url = ARCHIVE % month
    print("  %s  (%s)" % (month, url), flush=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r, tempfile.NamedTemporaryFile(
            suffix=".zip", delete=False) as tmp:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            tmp.write(chunk)
        tmp.flush()
        zpath = Path(tmp.name)
    kept = 0
    try:
        with zipfile.ZipFile(zpath) as z:
            members = [n for n in z.namelist()
                       if n.endswith("-%s-street.csv" % force) and not n.startswith("__")]
            if not members:
                print("    no %s street file in archive" % force, file=sys.stderr)
                return 0
            dst = outroot / month
            dst.mkdir(parents=True, exist_ok=True)
            for m in members:
                with z.open(m) as fh:
                    rd = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8",
                                                         errors="replace"))
                    rows = [row for row in rd if row.get("Crime type") == VIOLENCE]
                out = dst / Path(m).name
                with open(out, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=rd.fieldnames)
                    w.writeheader()
                    w.writerows(rows)
                kept += len(rows)
                print("    %s -> %d violence rows" % (out.name, len(rows)))
    finally:
        zpath.unlink(missing_ok=True)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=["2024-%02d" % m for m in range(1, 13)])
    ap.add_argument("--force", default="metropolitan")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    outroot = Path(a.out) if a.out else Path(__file__).resolve().parent / "raw" / "london_crime"
    outroot.mkdir(parents=True, exist_ok=True)

    total = 0
    for month in a.months:
        total += fetch_month(month, a.force, outroot)
    print("total violence rows: %d" % total)
    print("expected for the full 2024 Metropolitan year: about 252,700")


if __name__ == "__main__":
    main()
