# -*- coding: utf-8 -*-
"""Read one member out of a remote zip without downloading the archive.

The monthly national archives are 1.7 GB, of which the Metropolitan Police file is a small
fraction, and the connection to data.police.uk sustains about 0.15 MB/s, so fetching the whole
archive for one force costs hours. The server supports byte ranges, so the central directory can be
read from the tail of the file and the single member fetched directly.

Usage as a module:  fetch_member(url, "metropolitan") -> bytes
"""
import struct
import time
import urllib.request
import zlib

UA = {"User-Agent": "Mozilla/5.0"}
EOCD = b"PK\x05\x06"
CEN = b"PK\x01\x02"


def _get(url, start, end, tries=5):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={**UA, "Range": "bytes=%d-%d" % (start, end)})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(3 * (k + 1))
    raise last


def _size(url, tries=5):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return int(r.headers["Content-Range"].split("/")[1])
        except Exception as e:
            last = e
            time.sleep(3 * (k + 1))
    raise last


def fetch_member(url, needle):
    total = _size(url)
    tail_len = min(65536, total)
    tail = _get(url, total - tail_len, total - 1)
    i = tail.rfind(EOCD)
    if i < 0:
        raise RuntimeError("no end-of-central-directory record")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    cd = _get(url, cd_off, cd_off + cd_size - 1)

    pos, found = 0, None
    while True:
        j = cd.find(CEN, pos)
        if j < 0:
            break
        (method,) = struct.unpack("<H", cd[j + 10:j + 12])
        csize, usize = struct.unpack("<II", cd[j + 20:j + 28])
        nlen, elen = struct.unpack("<HH", cd[j + 28:j + 32])
        (lho,) = struct.unpack("<I", cd[j + 42:j + 46])
        name = cd[j + 46:j + 46 + nlen].decode("utf-8", "replace")
        if name == needle or (needle.lower() in name.lower() and name.lower().endswith(".csv")
                              and "-street.csv" in name.lower()):
            found = (name, method, csize, usize, lho)
            break
        pos = j + 46 + nlen + elen + 0
    if not found:
        raise RuntimeError("member not found")
    name, method, csize, usize, lho = found

    head = _get(url, lho, lho + 29)
    nlen, elen = struct.unpack("<HH", head[26:30])
    data_off = lho + 30 + nlen + elen
    blob = _get(url, data_off, data_off + csize - 1)
    if method == 0:
        return name, blob
    if method == 8:
        return name, zlib.decompress(blob, -15)
    raise RuntimeError("unsupported compression method %d" % method)
