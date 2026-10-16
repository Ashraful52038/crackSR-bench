# Results

Metric tables written by the scripts land here (CSV/JSON are committed; images are not).

| File | Written by |
|---|---|
| `pipeline_x4_real.csv` | `scripts/pipeline_eval.py` (main table: PSNR/SSIM + IoU/F1 + hallucination) |
| `efficiency_sr.csv`, `efficiency_seg.csv` | `scripts/benchmark_efficiency.py` |
| `vis/` | qualitative figures from `pipeline_eval.py` (git-ignored) |

Make paper tables with `python scripts/make_table.py results/<file>.csv --latex`.
