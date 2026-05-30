"""Download the 20 Newsgroups corpus via scikit-learn and save it as CSV.

20 Newsgroups is a public, well-known text-clustering benchmark distributed
inside scikit-learn (no Kaggle, no scraping). The download URL is
sklearn's hosted mirror and the dataset has been audited by the scientific
Python community for ~25 years — no malware, PII, or phishing content.

Usage
-----
    python data/raw/twenty_newsgroups/download.py            # train subset (~11k)
    python data/raw/twenty_newsgroups/download.py --subset all   # full 20k
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

from sklearn.datasets import fetch_20newsgroups

OUT = pathlib.Path(__file__).resolve().parent / "twenty_newsgroups.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="train", choices=("train", "test", "all"),
                    help="Which split to download (default: train).")
    ap.add_argument("--out", default=str(OUT), help="Output CSV path.")
    args = ap.parse_args()

    print(f"[download] Fetching 20 Newsgroups subset={args.subset!r} via sklearn ...")
    bundle = fetch_20newsgroups(
        subset=args.subset,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    docs = bundle.data
    targets = bundle.target
    target_names = bundle.target_names
    n = len(docs)
    print(f"[download] {n:,} posts across {len(target_names)} categories. "
          f"Writing → {args.out}")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "category", "text"])
        kept = 0
        for i, (doc, t) in enumerate(zip(docs, targets)):
            doc = (doc or "").strip()
            # Drop empty / near-empty posts so downstream clustering isn't
            # fed noise rows (the 'remove=' cleanup can leave some empties).
            if len(doc) < 20:
                continue
            writer.writerow([i, target_names[t], doc])
            kept += 1

    print(f"[download] Done. Kept {kept:,} non-empty posts (dropped {n - kept:,}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
