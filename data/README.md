# Data sources and licences

No raw data ships with this repository. Everything below is public and can be re-fetched with
the script named in the last column.

| Data | Used for | Source | Licence | Script |
| --- | --- | --- | --- | --- |
| Chicago incidents, 2024 | Tables 1–4, 8 | City of Chicago open data, dataset `ijzp-q8t2` | Open data, attribution | `download_chicago_crime.py` |
| Chicago street network | audit, all Chicago results | OpenStreetMap via BBBike extract | ODbL | `download_roads.py --city chicago` |
| London incidents, 2024 | Tables 5–7, 9, 10 | data.police.uk monthly archive | Open Government Licence | `download_london_crime.py` |
| London street network | Tables 5–10 | OpenStreetMap via BBBike extract | ODbL | `download_roads.py --city london` |
| OS Open Names (GB) | Tables 9–10 | Ordnance Survey Open Names, OS Data Hub | Open Government Licence | `download_os_open_names.py` |
| Gridded population (not used) | — | — | — | — |

## The two files that cannot be regenerated exactly

`chicago_roads.geojson` (44.7 MB) and `london_roads.geojson` (141.8 MB) were converted from the
BBBike `.osm.pbf` extracts obtained in March 2026 and September 2026 respectively. Both post-date
the 2024 incidents by one to two years, which the manuscript reports in Section 3.1 and lists as a
limitation. Because OpenStreetMap is
edited continuously, re-downloading today gives a similar network with different geometry
details, and the audit counts in Table 1 would move by a few records.

Both converted files are archived together with the exact London and Chicago incident extracts:

> **[Zenodo DOI to be inserted on acceptance]**

Exact files to archive, with the checksums that identify them:

| File | Features | Bytes | MD5 |
| --- | --- | --- | --- |
| `chicago_roads.geojson` | 77,357 | 44,695,487 | `9705cdb96656afb41534b5570e573971` |
| `london_roads.geojson` | 369,272 | 141,782,582 | `97d35133e4dbf06345c999c373ffe191` |

`download_roads.py` fetches the current PBF and can convert it with `osmium export`, which is
enough to run the pipeline end to end. Use the archived files to reproduce the published
numbers.

## Download sizes and times

| Script | Download | Time |
| --- | --- | --- |
| `download_chicago_crime.py` | about 15 MB | a few minutes |
| `download_london_crime.py` | about 20 GB for the full year, 1.7 GB per month | an hour or more |
| `download_roads.py` | 100–300 MB per city | minutes |
| `download_os_open_names.py` | 103 MB | minutes |

`download_london_crime.py --months 2024-01 2024-02` fetches a subset if you only want to check
that the pipeline runs.

## Field notes that cost time to discover

- The Socrata endpoint used for Chicago returns rows in an unstable order when an `$order`
  clause is present, and a long page can silently reset the connection. The downloader pages by
  `$offset` without `$order`.
- `data.police.uk` publishes one CSV per force per month inside a single national archive; there
  is no per-force download endpoint.
- The OS download API returns JSON metadata unless `redirect=true` is appended; without it the
  response is a 220-byte JSON document rather than the archive.
- OS Open Names CSVs have no header row. Columns 6, 7, 8, 9 and 27 are TYPE, LOCAL_TYPE, easting,
  northing and region.
