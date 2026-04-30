#!/usr/bin/env python3
"""
Step 1: Download the Cao et al. DMS escape datasets from the Bloom Lab GitHub repository.

Sources:
  - 2022_Cao_Omicron     : 247 antibodies (Cao et al. Nature 2022, Omicron escape paper)
  - 2022_Cao_convergent  : 3,333 antibodies (Cao et al. 2022, convergent RBD evolution paper)
  - 2022_Cao_BA2-4-5     : 1,538 antibodies (Cao et al. 2022, BA.2/BA.4/BA.5 escape paper)

Together these three datasets cover the ~3,195 antibodies Holt et al. started with.
"""

import os
import urllib.request
import pandas as pd

BLOOM_BASE = "https://raw.githubusercontent.com/jbloomlab/SARS2_RBD_Ab_escape_maps/main/data"

DATASETS = {
    "omicron": {
        "antibodies": f"{BLOOM_BASE}/2022_Cao_Omicron/antibodies.csv",
        "escape":     f"{BLOOM_BASE}/2022_Cao_Omicron/data.csv",
    },
    "convergent": {
        "antibodies": f"{BLOOM_BASE}/2022_Cao_convergent/antibody_info.csv",
        "escape":     f"{BLOOM_BASE}/2022_Cao_convergent/use_res_clean.csv",
    },
    "ba245": {
        "antibodies": f"{BLOOM_BASE}/2022_Cao_BA2-4-5/Abinfo.tsv",
        "escape":     f"{BLOOM_BASE}/2022_Cao_BA2-4-5/data.csv",
        "clusters":   f"{BLOOM_BASE}/2022_Cao_BA2-4-5/antibody_clusters.csv",
    },
}

RAW_DIR = "data/raw"


def download_file(url: str, dest_path: str) -> None:
    """Download a file from url to dest_path, skipping if it already exists."""
    if os.path.exists(dest_path):
        print(f"  [skip] {dest_path} already exists")
        return
    print(f"  Downloading {url.split('/')[-1]} ...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    print(f"  Saved -> {dest_path}")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    for dataset_name, files in DATASETS.items():
        print(f"\n=== Dataset: {dataset_name} ===")
        for file_type, url in files.items():
            ext = url.split(".")[-1]
            dest = os.path.join(RAW_DIR, f"{dataset_name}_{file_type}.{ext}")
            download_file(url, dest)

    # Quick validation: print row counts for each antibody file
    print("\n=== Row counts (antibody metadata files) ===")
    counts = {
        "omicron":    pd.read_csv(os.path.join(RAW_DIR, "omicron_antibodies.csv")),
        "convergent": pd.read_csv(os.path.join(RAW_DIR, "convergent_antibodies.csv")),
        "ba245":      pd.read_csv(os.path.join(RAW_DIR, "ba245_antibodies.tsv"), sep="\t"),
    }
    for name, df in counts.items():
        print(f"  {name}: {len(df):,} antibodies, columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
