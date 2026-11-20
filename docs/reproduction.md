# Which script produces which table

Run from the repository root. Each script writes into `outputs/` (override with
`SPATIAL_OUTPUT`) and reads from `data/raw` (override with `SPATIAL_DATA`).

| Script | Manuscript | Output |
| --- | --- | --- |
| `01_chicago_audit_and_calibration.py` | §3.1 audit, Tables 1 and 2 | `network_audit.json`, `chicago_corrected.json`, `calibration_summary.csv`, `calibration_replicates.csv`, `chicago_canonical_counts.csv` |
| `02_detection_boundary_and_origin.py` | Tables 3 and 4 | `extensions_power_origin.csv` |
| `03_london_replication.py` | Tables 5 and 6 | `london_replication.csv`, `london_support.json` |
| `04_reference_point_operator.py` | Table 7, columns 1 to 4 | `vertex_operator.csv` |
| `05_reference_point_length_test.py` | Table 7, columns 5 and 6 | `vertex_length_test.csv` |
| `06_fifty_x_calibration.py` | Table 8 | `fifty_x_calibration.csv` |
| `07_london_application.py` | Table 9 | `london_empirical.csv`, `london_empirical_support.json` |
| `08_os_road_base.py` | Table 10, road counts | `os_open_names_london.json` |
| `09_address_filter_proxy.py` | Table 10, pruning shares | `postcode_proxy_pruning.json` |
| `10_reference_set_sweep.py` | Figure 4 | `reference_set_sweep.csv` |

## Ordering

Scripts 01 and 02 need the Chicago road file and the Chicago incident file. Scripts 03, 07 and 10
need the London road file and the London incident files. Script 08 and 09 need the OS Open Names
archive. Scripts 04 to 06 need both road files.
Nothing depends on a previous script's output, so they can be run in any order once the data are
in place.

## Figures

`11_make_figures.py` draws all four figures from the derived tables, so no figure is hand-edited:

| Figure | File | Read from |
| --- | --- | --- |
| 1 | `fig1_mechanism.png` | `calibration_summary.csv` |
| 2 | `fig2_calibration.png` | `calibration_summary.csv`, `extensions_power_origin.csv` |
| 3 | `fig3_fifty_x.png` | `fifty_x_calibration.csv` |
| 4 | `fig4_reference_set.png` | `reference_set_sweep.csv` |

Figure 2 was produced by script 02 during the original analysis and is included as drawn.
