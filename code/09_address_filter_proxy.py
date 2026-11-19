"""Postcode-density proxy for the address filter (Table 10, pruning shares).

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


from pathlib import Path
import csv
import io
import json
import sys
import zipfile
import numpy as np
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
OUT = OUTDIR
ZIP = RAW / "opname_csv_gb.zip"
ROAD_LT = {"Named Road", "Numbered Road", "Road Number", "Motorway", "A Road", "B Road"}

roads, posts = [], []
z = zipfile.ZipFile(ZIP)
files = [i.filename for i in z.infolist() if i.filename.lower().endswith(".csv")]
for k, f in enumerate(files):
    with z.open(f) as fh:
        for row in csv.reader(io.TextIOWrapper(fh, encoding="utf-8", errors="replace")):
            if len(row) < 28:
                continue
            lt = row[7]
            if lt not in ROAD_LT and lt != "Postcode":
                continue
            if row[27] != "London":
                continue
            try:
                x = float(row[8]); y = float(row[9])
            except ValueError:
                continue
            (roads if lt in ROAD_LT else posts).append((x, y))
    if k % 150 == 0:
        print("  %d/%d | roads %d | postcodes %d" % (k, len(files), len(roads), len(posts)),
              flush=True)

roads = np.array(roads); posts = np.array(posts)
print("\n=== check 1: both layers present and co-located ===")
print("  road records: %d | postcode records: %d" % (len(roads), len(posts)))
print("  road    E %.0f-%.0f  N %.0f-%.0f" % (roads[:, 0].min(), roads[:, 0].max(),
                                              roads[:, 1].min(), roads[:, 1].max()))
print("  postcode E %.0f-%.0f  N %.0f-%.0f" % (posts[:, 0].min(), posts[:, 0].max(),
                                               posts[:, 1].min(), posts[:, 1].max()))
assert len(roads) > 50000 and len(posts) > 10000, "layer counts look wrong"
assert abs(np.median(roads[:, 0]) - np.median(posts[:, 0])) < 40000, "layers not co-located"
print("  PASS", flush=True)

tree = cKDTree(posts)
print("\n=== check 2: postcodes per road point, by radius ===")
rows = []
for r in (100, 200, 300, 500):
    cnt = np.array([len(c) for c in tree.query_ball_point(roads, r=r, workers=-1)])
    nopc = int((cnt == 0).sum())
    rows.append(dict(radius_m=r, no_postcode=nopc, prune_pct=round(100 * nopc / len(roads), 2),
                     median=int(np.median(cnt))))
    print("  r=%3d m | road points with no postcode: %6d (%.2f%%) | median %d" %
          (r, nopc, 100 * nopc / len(roads), np.median(cnt)))

print("\n=== check 3: the filter should bite harder away from the centre ===")
cen = np.array([530000.0, 180000.0])
d = np.hypot(roads[:, 0] - cen[0], roads[:, 1] - cen[1])
cnt200 = np.array([len(c) for c in tree.query_ball_point(roads, r=200, workers=-1)])
for lo, hi, lab in [(0, 5000, "inner 5 km"), (5000, 12000, "5-12 km"), (12000, 1e9, "outer")]:
    m = (d >= lo) & (d < hi)
    if m.sum() == 0:
        continue
    print("  %-12s n=%6d | no postcode at r=200: %.2f%%" %
          (lab, m.sum(), 100 * (cnt200[m] == 0).mean()))

survive = int((cnt200 > 0).sum())
print("\n=== result ===")
print("  road base                 : %d" % len(roads))
print("  surviving a 1-postcode rule at r=200 m: %d (%.1f%%)" %
      (survive, 100 * survive / len(roads)))
print("  the turnover in Section 5.8 is 52,533, so the road component alone is %s it" %
      ("above" if survive > 52533 else "below"))

json.dump(dict(road_records=int(len(roads)), postcode_records=int(len(posts)),
               rows=rows, survivors_r200=survive, turnover=52533),
          open(OUT / "postcode_proxy_pruning.json", "w"), indent=2)
print("\nwrote postcode_proxy_pruning.json")
