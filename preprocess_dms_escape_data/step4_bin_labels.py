#!/usr/bin/env python3
"""
Step 4: Derive the 12 epitope bin labels for every antibody.

The 12 bins are: A, B, C, D1, D2, E1, E2.1, E2.2, E3, F1, F2, F3
These correspond to spatially distinct regions on the SARS-CoV-2 RBD surface.

Strategy:
  - The convergent dataset already has pre-assigned bin labels in its escape CSV
    (the 'group' column). We use those directly — no clustering needed.
  - The omicron dataset only has 6 coarse bins (A–F). We assign each omicron
    antibody to one of the 12 bins using nearest-centroid assignment:
      1. Build per-residue escape vectors for all convergent antibodies
      2. Compute the 12 per-bin centroids (mean escape vector per bin)
      3. Assign each omicron antibody to the bin whose centroid is most
         similar to its own escape vector (cosine similarity)

Output: data/processed/labeled_antibodies.csv
  Adds column 'bin_label' (one of the 12 bins) to annotated_antibodies.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

RAW_DIR  = "data/raw"
PROC_DIR = "data/processed"
INPUT    = os.path.join(PROC_DIR, "annotated_antibodies.csv")
OUTPUT   = os.path.join(PROC_DIR, "labeled_antibodies.csv")

BINS_12 = ["A", "B", "C", "D1", "D2", "E1", "E2.1", "E2.2", "E3", "F1", "F2", "F3"]


# ---------------------------------------------------------------------------
# Build per-residue escape vectors
# ---------------------------------------------------------------------------

def build_escape_matrix(escape_df: pd.DataFrame,
                        ab_col: str,
                        antibodies: list[str]) -> pd.DataFrame:
    """
    Convert a long-format escape table into a (antibodies × sites) matrix.

    For each antibody and each RBD site, we sum mut_escape across all mutations
    at that site. This gives a single scalar per (antibody, site) that captures
    the total escape weight at that position — the approach used by the Bloom Lab
    and cited in Holt et al.

    Returns a DataFrame with antibody IDs as index and RBD site positions as columns.
    Missing values (sites with no escape data for an antibody) are filled with 0.
    """
    # Filter to antibodies we care about
    esc = escape_df[escape_df[ab_col].isin(antibodies)].copy()

    # Sum mutation-level escape -> site-level escape
    site_escape = (
        esc.groupby([ab_col, "site"])["mut_escape"]
        .sum()
        .reset_index()
    )

    # Pivot to wide format: rows = antibodies, columns = sites
    matrix = site_escape.pivot(index=ab_col, columns="site", values="mut_escape")
    matrix = matrix.reindex(index=antibodies).fillna(0.0)
    matrix.index.name = "antibody_id"
    return matrix


# ---------------------------------------------------------------------------
# Nearest-centroid assignment
# ---------------------------------------------------------------------------

def compute_bin_centroids(escape_matrix: pd.DataFrame,
                          bin_labels: pd.Series) -> pd.DataFrame:
    """
    Compute the mean escape vector for each bin.
    escape_matrix: (n_antibodies × n_sites), index = antibody_id
    bin_labels: Series indexed by antibody_id, values = bin names
    Returns DataFrame of shape (n_bins × n_sites).
    """
    df = escape_matrix.copy()
    df["bin"] = bin_labels
    centroids = df.groupby("bin").mean()
    return centroids


def assign_nearest_centroid(escape_matrix: pd.DataFrame,
                            centroids: pd.DataFrame) -> pd.Series:
    """
    Assign each antibody to the bin whose centroid has the highest
    cosine similarity to the antibody's escape vector.
    Returns a Series indexed by antibody_id with the assigned bin.
    """
    # Align columns (sites) between matrix and centroids
    shared_sites = escape_matrix.columns.intersection(centroids.columns)
    X = escape_matrix[shared_sites].values          # (n_query × n_sites)
    C = centroids[shared_sites].values              # (n_bins × n_sites)

    # cosine_similarity returns (n_query × n_bins)
    sims = cosine_similarity(X, C)
    best_bin_idx = np.argmax(sims, axis=1)
    assigned_bins = centroids.index[best_bin_idx]

    return pd.Series(assigned_bins, index=escape_matrix.index, name="bin_label")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading annotated antibodies ...")
    df = pd.read_csv(INPUT)
    print(f"  {len(df):,} antibodies")

    # ------------------------------------------------------------------
    # 1. Load escape data
    # ------------------------------------------------------------------
    print("\nLoading escape data ...")
    conv_esc = pd.read_csv(os.path.join(RAW_DIR, "convergent_escape.csv"))
    om_esc   = pd.read_csv(os.path.join(RAW_DIR, "omicron_escape.csv"))
    print(f"  Convergent escape: {conv_esc.shape[0]:,} rows, "
          f"{conv_esc['antibody'].nunique():,} antibodies")
    print(f"  Omicron escape:    {om_esc.shape[0]:,} rows, "
          f"{om_esc['condition'].nunique():,} antibodies")

    # ------------------------------------------------------------------
    # 2. Assign bins for convergent antibodies (pre-labeled)
    #    The 'group' column in the convergent escape CSV is the 12-bin label.
    # ------------------------------------------------------------------
    print("\nAssigning bins for convergent dataset (pre-labeled) ...")
    conv_labels = (
        conv_esc[["antibody", "group"]]
        .drop_duplicates("antibody")
        .rename(columns={"antibody": "antibody_id", "group": "bin_label"})
    )
    conv_labels = conv_labels[conv_labels["bin_label"].isin(BINS_12)]
    print(f"  {len(conv_labels):,} convergent antibodies with 12-bin labels")
    print(f"  Bin distribution:")
    print(conv_labels["bin_label"].value_counts().sort_index().to_string())

    # ------------------------------------------------------------------
    # 3. Assign bins for omicron antibodies (nearest centroid)
    #    Omicron dataset only has 6 coarse bins — we re-assign to 12.
    # ------------------------------------------------------------------
    print("\nBuilding escape matrix for nearest-centroid assignment ...")

    omicron_ids = df[df["dataset"] == "omicron"]["antibody_id"].tolist()
    conv_ids_with_labels = conv_labels["antibody_id"].tolist()

    if len(omicron_ids) > 0:
        # Build escape matrix for convergent antibodies (to derive centroids)
        conv_matrix = build_escape_matrix(
            escape_df=conv_esc,
            ab_col="antibody",
            antibodies=conv_ids_with_labels,
        )
        conv_bin_series = conv_labels.set_index("antibody_id")["bin_label"]

        # Compute per-bin centroids from convergent antibodies
        centroids = compute_bin_centroids(conv_matrix, conv_bin_series)
        print(f"  Centroids shape: {centroids.shape}  (12 bins × {centroids.shape[1]} sites)")

        # Build escape matrix for omicron antibodies
        om_matrix = build_escape_matrix(
            escape_df=om_esc,
            ab_col="condition",
            antibodies=omicron_ids,
        )
        print(f"  Omicron escape matrix: {om_matrix.shape}")

        # Assign omicron antibodies to nearest centroid
        omicron_bin_labels = assign_nearest_centroid(om_matrix, centroids)
        omicron_labels = omicron_bin_labels.reset_index()
        omicron_labels.columns = ["antibody_id", "bin_label"]
        print(f"  Omicron bin assignment complete:")
        print(omicron_labels["bin_label"].value_counts().sort_index().to_string())
    else:
        omicron_labels = pd.DataFrame(columns=["antibody_id", "bin_label"])

    # ------------------------------------------------------------------
    # 4. Merge all bin labels into the main dataframe
    # ------------------------------------------------------------------
    all_labels = pd.concat([conv_labels, omicron_labels], ignore_index=True)
    df = df.merge(all_labels, on="antibody_id", how="left")

    n_labeled = df["bin_label"].notna().sum()
    n_unlabeled = df["bin_label"].isna().sum()
    print(f"\nLabeling summary:")
    print(f"  Labeled:   {n_labeled:,} / {len(df):,}")
    print(f"  Unlabeled: {n_unlabeled:,}  (no escape data — will be excluded from training)")

    # Drop antibodies with no bin label
    df = df[df["bin_label"].notna()].copy()
    print(f"  Final labeled dataset: {len(df):,} antibodies")

    # ------------------------------------------------------------------
    # 5. Save escape matrix for labeled antibodies (needed for Step 7 validation)
    # ------------------------------------------------------------------
    print("\nSaving labeled antibodies ...")
    df.to_csv(OUTPUT, index=False)
    print(f"Saved -> {OUTPUT}")

    # Final bin distribution
    print("\n=== Final bin distribution ===")
    print(df["bin_label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
