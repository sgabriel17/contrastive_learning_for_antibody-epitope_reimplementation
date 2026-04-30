#!/usr/bin/env python3
"""
Step 5: Build clone groups and perform a clone-aware 80/10/10 train/val/test split.

Clone group definition (from Holt et al.):
  Two antibodies belong to the same clone if they share:
    (1) The same heavy chain V gene (IGHV)
    (2) The same light chain V gene (IGKV or IGLV)
    (3) CDRH3 amino acid sequence identity >= 65%

Why clone-aware splitting matters:
  Antibodies in the same clone are highly similar in sequence. If one clone
  member is in train and another in test, the model can "memorize" the
  sequence pattern and appear to generalize when it actually hasn't.
  Splitting by clone group ensures no related sequences leak across sets.

Algorithm:
  1. For each pair of antibodies sharing both V genes, compute CDRH3 identity
  2. Build a graph: nodes = antibodies, edges = pairs with identity >= 65%
  3. Find connected components of this graph → each component is a clone group
  4. Randomly assign clone groups (not individual antibodies) to splits
     targeting 80% train / 10% val / 10% test by antibody count

Output: data/processed/split_antibodies.csv
  Adds columns 'clone_group_id' and 'split' to labeled_antibodies.csv
"""

import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict

PROC_DIR   = "data/processed"
INPUT      = os.path.join(PROC_DIR, "labeled_antibodies.csv")
OUTPUT     = os.path.join(PROC_DIR, "split_antibodies.csv")

RANDOM_SEED  = 42
TRAIN_FRAC   = 0.80
VAL_FRAC     = 0.10
# TEST_FRAC  = 0.10  (remainder)
CDR3_ID_THRESHOLD = 0.65   # 65% identity threshold from Holt et al.


# ---------------------------------------------------------------------------
# CDRH3 sequence identity
# ---------------------------------------------------------------------------

def cdr3_identity(seq_a: str, seq_b: str) -> float:
    """
    Compute amino acid sequence identity between two CDRH3 sequences.

    Uses the simple aligned-identity definition:
        identity = (matching positions) / (max length of the two sequences)

    For sequences of different length, we align them at the N-terminus
    (left-pad the shorter one with gaps) — this is standard for CDR3
    length-normalized identity as used in clonotype analysis.

    Returns a float in [0, 1].
    """
    if not isinstance(seq_a, str) or not isinstance(seq_b, str):
        return 0.0
    if seq_a == seq_b:
        return 1.0

    len_a, len_b = len(seq_a), len(seq_b)
    max_len = max(len_a, len_b)
    if max_len == 0:
        return 0.0

    # Pad shorter sequence to match length (left-align, gap = '_')
    if len_a < len_b:
        seq_a = seq_a.ljust(max_len, '_')
    else:
        seq_b = seq_b.ljust(max_len, '_')

    matches = sum(a == b for a, b in zip(seq_a, seq_b))
    return matches / max_len


# ---------------------------------------------------------------------------
# Union-Find (for efficient connected components)
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# ---------------------------------------------------------------------------
# Clone group construction
# ---------------------------------------------------------------------------

def build_clone_groups(df: pd.DataFrame) -> pd.Series:
    """
    Assign a clone_group_id (integer) to each antibody.
    Antibodies that don't share V genes with any other antibody get their
    own singleton clone group.

    Returns a Series indexed by df.index, values = clone_group_id (int).
    """
    # Only consider antibodies with all required annotations
    required = ["heavy_v_gene", "light_v_gene", "cdrh3_aa"]
    valid_mask = df[required].notna().all(axis=1)
    valid = df[valid_mask].copy().reset_index(drop=False)
    valid.index.name = "local_idx"

    n = len(valid)
    uf = UnionFind(n)

    # Group by (heavy_v_gene, light_v_gene) to limit comparisons
    # Only pairs sharing both V genes are candidates for clonality
    vgene_groups = valid.groupby(["heavy_v_gene", "light_v_gene"]).indices

    total_pairs_checked = 0
    total_clonal_pairs  = 0

    for (hv, lv), indices in vgene_groups.items():
        idx_list = list(indices)
        if len(idx_list) < 2:
            continue

        # Pairwise CDRH3 identity within this V gene group
        for i in range(len(idx_list)):
            for j in range(i + 1, len(idx_list)):
                a, b = idx_list[i], idx_list[j]
                cdr3_a = valid.loc[a, "cdrh3_aa"]
                cdr3_b = valid.loc[b, "cdrh3_aa"]
                identity = cdr3_identity(cdr3_a, cdr3_b)
                total_pairs_checked += 1
                if identity >= CDR3_ID_THRESHOLD:
                    uf.union(a, b)
                    total_clonal_pairs += 1

    print(f"  CDRH3 pairs checked: {total_pairs_checked:,}")
    print(f"  Clonal pairs found:  {total_clonal_pairs:,} "
          f"(identity >= {CDR3_ID_THRESHOLD:.0%})")

    # Assign clone group IDs (renumber roots 0, 1, 2, ...)
    root_to_group = {}
    group_counter = 0
    local_group_ids = []
    for i in range(n):
        root = uf.find(i)
        if root not in root_to_group:
            root_to_group[root] = group_counter
            group_counter += 1
        local_group_ids.append(root_to_group[root])

    valid["clone_group_id"] = local_group_ids

    # Map back to original df index
    result = pd.Series(index=df.index, dtype=object, name="clone_group_id")
    result[valid["index"]] = valid["clone_group_id"].values

    # Antibodies missing V gene / CDR3 get their own unique singleton groups
    n_valid_groups = group_counter
    missing_mask = ~valid_mask
    for offset, orig_idx in enumerate(df.index[missing_mask]):
        result[orig_idx] = n_valid_groups + offset

    return result.astype(int)


# ---------------------------------------------------------------------------
# Clone-aware split
# ---------------------------------------------------------------------------

def clone_aware_split(df: pd.DataFrame,
                      train_frac: float = TRAIN_FRAC,
                      val_frac:   float = VAL_FRAC,
                      seed:       int   = RANDOM_SEED) -> pd.Series:
    """
    Assign each antibody to 'train', 'val', or 'test'.

    Algorithm:
    1. Get all unique clone groups and their sizes
    2. Shuffle clone groups (fixed seed for reproducibility)
    3. Greedily assign clone groups to train until target fraction is reached,
       then to val, then remainder to test
    4. Return a Series of split assignments indexed by df.index

    This guarantees no clone group is split across train/val/test.
    """
    rng = random.Random(seed)

    # Get clone group sizes
    group_sizes = df.groupby("clone_group_id").size().to_dict()
    groups = list(group_sizes.keys())
    rng.shuffle(groups)

    total = len(df)
    train_target = int(total * train_frac)
    val_target   = int(total * val_frac)

    split_map = {}
    train_count = val_count = 0

    for group_id in groups:
        size = group_sizes[group_id]
        if train_count < train_target:
            split_map[group_id] = "train"
            train_count += size
        elif val_count < val_target:
            split_map[group_id] = "val"
            val_count += size
        else:
            split_map[group_id] = "test"

    return df["clone_group_id"].map(split_map).rename("split")


# ---------------------------------------------------------------------------
# Validation: verify no clone bleeds across splits
# ---------------------------------------------------------------------------

def validate_split(df: pd.DataFrame) -> None:
    """
    Check that no clone group appears in more than one split.
    Raises AssertionError if any leakage is found.
    """
    group_splits = df.groupby("clone_group_id")["split"].nunique()
    leaked = group_splits[group_splits > 1]
    if len(leaked) > 0:
        raise AssertionError(
            f"LEAKAGE DETECTED: {len(leaked)} clone groups span multiple splits!"
            f"{leaked}"
        )
    print("  [OK] No clone group spans multiple splits — split is clean.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading labeled antibodies ...")
    df = pd.read_csv(INPUT)
    print(f"  {len(df):,} antibodies, {df['bin_label'].nunique()} bins")

    # Drop antibodies missing CDR3 annotation — can't assign to a clone
    before = len(df)
    df = df[df["cdrh3_aa"].notna()].copy()
    print(f"  Dropped {before - len(df)} with missing CDRH3 — {len(df):,} remain")

    # ------------------------------------------------------------------
    # Step 5a: Build clone groups
    # ------------------------------------------------------------------
    print("\nBuilding clone groups ...")
    print(f"  (comparing pairs sharing same heavy + light V gene, CDRH3 >= {CDR3_ID_THRESHOLD:.0%})")
    df["clone_group_id"] = build_clone_groups(df)

    n_groups = df["clone_group_id"].nunique()
    n_singletons = (df.groupby("clone_group_id").size() == 1).sum()
    print(f"  Total clone groups: {n_groups:,}")
    print(f"  Singletons:         {n_singletons:,} ({100*n_singletons/n_groups:.1f}%)")
    print(f"  Multi-member clones: {n_groups - n_singletons:,}")

    # ------------------------------------------------------------------
    # Step 5b: Clone-aware split
    # ------------------------------------------------------------------
    print(f"\nSplitting by clone group (seed={RANDOM_SEED}) ...")
    df["split"] = clone_aware_split(df)

    counts = df["split"].value_counts()
    total  = len(df)
    print(f"  train: {counts['train']:,}  ({100*counts['train']/total:.1f}%)")
    print(f"  val:   {counts['val']:,}  ({100*counts['val']/total:.1f}%)")
    print(f"  test:  {counts['test']:,}  ({100*counts['test']/total:.1f}%)")

    # ------------------------------------------------------------------
    # Step 5c: Validate — no clone bleeds across splits
    # ------------------------------------------------------------------
    print("\nValidating split integrity ...")
    validate_split(df)

    # ------------------------------------------------------------------
    # Step 5d: Check bin distribution across splits
    # A severe bin imbalance in val/test could cause misleading eval.
    # ------------------------------------------------------------------
    print("\n=== Bin distribution per split ===")
    bin_split = df.pivot_table(
        index="bin_label", columns="split",
        values="antibody_id", aggfunc="count", fill_value=0
    )
    # Reorder columns
    for col in ["train", "val", "test"]:
        if col not in bin_split.columns:
            bin_split[col] = 0
    print(bin_split[["train", "val", "test"]].to_string())

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved -> {OUTPUT}")
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
