"""Ordnance Survey road-name base for the London region (Table 10, road counts).

Part of the reproducibility package for
"Crime concentration after spatial publication: aligning random baselines with
observation processes".

Data are located through the SPATIAL_DATA environment variable and default to
<repo>/data/raw; outputs are written to <repo>/outputs.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("SPATIAL_DATA", REPO / "data" / "raw"))
OUTDIR = Path(os.environ.get("SPATIAL_OUTPUT", REPO / "outputs"))
OUTDIR.mkdir(parents=True, exist_ok=True)


import zipfile
import csv
import io
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
OUT = OUTDIR
ZIP = RAW / "opname_csv_gb.zip"

ROAD_LT = {"Named Road", "Numbered Road", "Road Number", "Motorway", "A Road", "B Road"}
z = zipfile.ZipFile(ZIP)
files = [i.filename for i in z.infolist() if i.filename.lower().endswith(".csv")]
print("files:", len(files), flush=True)

london_names = Counter()
london_rows = 0
gb_rows = 0
regions = Counter()
pts = defaultdict(list)

for k, f in enumerate(files):
    with z.open(f) as fh:
        for row in csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace")):
            if len(row) < 28:
                continue
            lt = row[7]
            if lt not in ROAD_LT:
                continue
            gb_rows += 1
            reg = row[27]
            regions[reg] += 1
            if reg == "London":
                london_rows += 1
                nm = row[2].strip()
                london_names[nm] += 1
                try:
                    pts[nm].append((float(row[8]), float(row[9])))
                except ValueError:
                    pass
    if k % 100 == 0:
        print("  scanned %d/%d | GB road rows %d | London %d" % (k, len(files), gb_rows,
                                                                 london_rows), flush=True)

print("\nGB road-type rows:", gb_rows)
print("London road-type rows:", london_rows)
print("London distinct road names:", len(london_names))
print("\nregions by road rows (top 8):")
for r, n in regions.most_common(8):
    print("   %-28s %d" % (r[:28], n))
print("\nmost duplicated London names:")
for nm, n in london_names.most_common(6):
    print("   %-28s %d localities" % (nm[:28], n))

json.dump(dict(gb_road_rows=gb_rows, london_road_rows=london_rows,
               london_distinct_names=len(london_names),
               top_regions=regions.most_common(8)),
          open(OUT / "os_open_names_london.json", "w"), indent=2)
print("\nwrote os_open_names_london.json")
