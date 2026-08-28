# Copyright (C) 2026 Mohammad Amir Khusru Akhtar
from __future__ import annotations
import argparse
import json
from .analysis import run


def main():
    parser = argparse.ArgumentParser(description="MorphoPreserve-DVI reproducible analysis")
    parser.add_argument("--data", default="data/ranchi_nasal_morphometry.csv")
    parser.add_argument("--out", default="results")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--splits", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    summary = run(args.data, args.out, repeats=args.repeats, splits=args.splits, bootstrap=args.bootstrap)
    print(json.dumps({
        "analysis_n": summary["analysis_n"],
        "acl_auc": summary["acl_auc"],
        "acl_auc_95ci": summary["acl_auc_95ci"],
        "sensitivity_analysis": summary["sensitivity_analysis"],
    }, indent=2))


if __name__ == "__main__":
    main()
