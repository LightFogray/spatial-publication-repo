"""Download OS Open Names, used to bound the withheld anonymisation master list.

Source: Ordnance Survey Open Names, available under the Open Government Licence from the OS
Data Hub. The archive is about 100 MB and contains one CSV per postcode district with no header
row; the columns used here are 6 (TYPE), 7 (LOCAL_TYPE), 8 (easting), 9 (northing) and
27 (region), in British National Grid metres.

The download endpoint returns JSON metadata unless `redirect=true` is passed, which is the one
non-obvious detail of this source.

Usage:  python data/download_os_open_names.py [--out PATH]
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = ("https://api.os.uk/downloads/v1/products/OpenNames/downloads"
       "?area=GB&format=CSV&redirect=true")
UA = {"User-Agent": "Mozilla/5.0"}
MD5 = None          # set after the first download; OS publishes it in the metadata endpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dst = Path(a.out) if a.out else Path(__file__).resolve().parent / "raw" / "opname_csv_gb.zip"
    dst.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(URL, headers=UA)
    h = hashlib.md5()
    with urllib.request.urlopen(req, timeout=3600) as r, open(dst, "wb") as fh:
        head = r.read(4)
        if head[:2] != b"PK":
            sys.exit("response is not a zip; the endpoint needs redirect=true")
        fh.write(head)
        h.update(head)
        total = len(head)
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            fh.write(chunk)
            h.update(chunk)
            total += len(chunk)
            if total % (32 << 20) < (1 << 22):
                print("  %.0f MB" % (total / 1e6), flush=True)

    print("wrote %s (%.0f MB)" % (dst, dst.stat().st_size / 1e6))
    print("md5: %s" % h.hexdigest())
    if MD5 and h.hexdigest() != MD5:
        sys.exit("md5 mismatch against the archived copy")


if __name__ == "__main__":
    main()
