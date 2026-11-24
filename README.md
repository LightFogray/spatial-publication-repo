# The published location is not the place

Reproducibility package for the manuscript

> **The published location is not the place: spatial anonymisation and the calibration of street-network concentration statistics**

The paper asks whether a random baseline for a street-network concentration statistic stays
calibrated when published coordinates are transformed and reassigned before counting, and whether
the distance those points move predicts the damage. This package contains the analysis
code, the derived tables behind every number in the paper, and scripts that re-fetch the public
data.

---

## What is here

```
code/      seventeen analysis scripts, one per block of results
data/      download scripts and the data manifest
outputs/   derived tables and diagnostics (the values printed in the paper)
figures/   the four figures, redrawn by code/11_make_figures.py
docs/      which script produces which table
```

Nothing in `outputs/` was edited by hand. Each file is written by the script that owns it, so a
re-run overwrites it with the same content up to the random seed.

---

## Requirements

Python 3.10 or newer, with:

```
numpy
pandas
scipy
matplotlib
```

`numpy`, `scipy` and `pandas` do most of the work; `matplotlib` is only needed by the figure
script. No GIS stack is required: the point-in-polyline matching is implemented directly in
`code/`.

The versions used for the paper are recorded in `environment.txt`.

---

## Data

None of the raw data are in this repository: the two road networks are 45 MB and 142 MB, the
London archives total about 20 GB, and OS Open Names is 100 MB. `data/README.md` lists every
source, its licence, and the script that fetches it.

The two converted road files are the one input that cannot be regenerated identically, because
OpenStreetMap is edited continuously. They are archived at https://doi.org/10.5281/zenodo.24370009 and
`data/download_roads.py` fetches and converts the current extract, which yields an equivalent
network rather than a bit-identical one. The archived copies were obtained in March 2026 (Chicago) and September 2026 (London) and
therefore post-date the 2024 incidents, a mismatch the manuscript reports rather than
corrects.

Point the analysis at your data directory with an environment variable:

```bash
export SPATIAL_DATA=/path/to/data/raw      # defaults to ./data/raw
export SPATIAL_OUTPUT=/path/to/outputs     # defaults to ./outputs
```

---

## Running the analysis

Run the scripts in numeric order from the repository root. They take from a minute to about
twenty minutes each, dominated by the polyline matching over roughly 1.4 million quadrature
points.

```bash
python code/01_chicago_audit_and_calibration.py
python code/02_detection_boundary_and_origin.py
python code/03_london_replication.py
python code/04_reference_point_operator.py
python code/05_reference_point_length_test.py
python code/06_fifty_x_calibration.py
python code/07_london_application.py
python code/08_os_road_base.py
python code/09_address_filter_proxy.py
python code/10_reference_set_sweep.py
python code/11_make_figures.py
```

Each script prints its headline numbers as it runs. `docs/reproduction.md` maps every table in
the manuscript to the script and the output file that produce it.

---

## Checking the results without re-running anything

`outputs/` already contains the values printed in the paper. To confirm that a freshly cloned
copy is consistent, compare the printed numbers against these files:

| Manuscript | Output file | Key values |
| --- | --- | --- |
| Table 1 | `chicago_corrected.json` | Gini 0.79611, null 0.51897, p 0.001 |
| Table 2 | `calibration_summary.csv` | sizes 0.062 / 0.048 / 0.054 / 0.034, 1.000 |
| Table 3 | `extensions_power_origin.csv` | origin sweep, 12 offsets per grid |
| Table 4 | `extensions_power_origin.csv` | detection boundary |
| Tables 5 and 6 | `london_replication.csv` | London sizes and power |
| Table 7 | `vertex_operator.csv`, `vertex_length_test.csv` | 14.1% moved, sizes 0.062 / 0.036 |
| Table 8 | `fifty_x_calibration.csv` | 19.20% → 5.22% on Chicago |
| Table 9 | `london_empirical.csv` | p 0.001 and 1.000 |
| Table 10 | `postcode_proxy_pruning.json` | 1.79% at 100 m, 0.50% at 200 m |
| Table 11 | `reference_set_sweep.csv` | turnover at 52,533 |

---

## Seeds

Every script fixes its own seed, so a re-run reproduces the same draws on any machine with the
same numpy version. The seeds are the date style used throughout, for example 20261117 in
script 01 and 20261126 in script 07.

---

## Licence

Code: MIT (see `LICENSE`). Derived outputs: CC BY 4.0. The upstream data keep their own
licences, listed in `data/README.md`; the OpenStreetMap extracts are ODbL and the police data
are Open Government Licence, both of which permit redistribution with attribution.
