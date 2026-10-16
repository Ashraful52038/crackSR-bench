"""Turn result CSVs into Markdown or LaTeX tables for the paper.

    python scripts/make_table.py results/pipeline_x4_real.csv --cols method psnr ssim iou f1 halluc_rate
    python scripts/make_table.py results/efficiency_sr.csv --latex
"""
import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--cols", nargs="*", help="Columns to keep (default: all)")
    parser.add_argument("--sort", help="Column to sort by (descending)")
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True)
    if args.cols:
        df = df[args.cols]
    if args.sort:
        df = df.sort_values(args.sort, ascending=False)
    if args.latex:
        print(df.to_latex(index=False, float_format=f"%.{args.digits}f"))
    else:
        print(df.to_markdown(index=False, floatfmt=f".{args.digits}f"))


if __name__ == "__main__":
    main()
