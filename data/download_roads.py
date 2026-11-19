"""Download the OpenStreetMap road extracts used by the analysis.

BBBike publishes per-city OSM extracts as .osm.pbf and .osm.gz (OpenStreetMap contributors,
ODbL). It does not publish GeoJSON, so the two files the analysis reads were converted from the
archived PBFs. The exact converted files are frozen on Zenodo with a DOI so the published
numbers can be reproduced bit for bit; the script here fetches the current extract and converts
it, which gives an equivalent network rather than an identical one because OSM is edited daily.

Conversion, once the PBF is downloaded (either tool works):

    osmium export -f geojson -o chicago_roads.geojson Chicago.osm.pbf
    ogr2ogr -f GeoJSON chicago_roads.geojson Chicago.osm.pbf lines

Only records whose `highway` property is one of residential, primary, secondary, tertiary,
unclassified, living_street or trunk are used, and the analysis filters again at load time, so
a superset extract is fine.

Usage:  python data/download_roads.py --city chicago [--convert]
"""
import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

BBBIKE = "https://download.bbbike.org/osm/bbbike/%s/%s.osm.pbf"
CITIES = {"chicago": ("Chicago", "chicago_roads.geojson"),
          "london": ("London", "london_roads.geojson")}
UA = {"User-Agent": "Mozilla/5.0"}


def download(url, dst, timeout=3600):
    print("  %s" % url, flush=True)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as fh:
        total = 0
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
            if total % (64 << 20) < (1 << 22):
                print("    %.0f MB" % (total / 1e6), flush=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=sorted(CITIES), required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--convert", action="store_true",
                    help="run osmium export after downloading")
    a = ap.parse_args()
    name, geojson = CITIES[a.city]
    outdir = Path(a.out) if a.out else Path(__file__).resolve().parent / "raw"
    outdir.mkdir(parents=True, exist_ok=True)

    pbf = outdir / ("%s.osm.pbf" % name)
    if not pbf.exists():
        download(BBBIKE % (name, name), pbf)
    print("  pbf: %s (%.0f MB)" % (pbf, pbf.stat().st_size / 1e6))

    if a.convert:
        if not shutil.which("osmium"):
            sys.exit("osmium not found; convert manually (see the module docstring)")
        dst = outdir / geojson
        subprocess.run(["osmium", "export", "-f", "geojson", "-o", str(dst), str(pbf)],
                       check=True)
        print("  wrote %s (%.0f MB)" % (dst, dst.stat().st_size / 1e6))
    else:
        print("  convert with: osmium export -f geojson -o %s %s" % (geojson, pbf.name))


if __name__ == "__main__":
    main()
