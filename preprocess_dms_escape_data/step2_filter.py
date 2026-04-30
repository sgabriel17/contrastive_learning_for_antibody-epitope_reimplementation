#!/usr/bin/env python3
"""
Step 2: Filter each dataset to index-strain (D614G/WT) binders, merge the three
datasets, and deduplicate by exact heavy+light chain sequence.

Holt et al. started with 3,195 antibodies from 2 Cao papers and retained 3,093
after filtering to index-strain binders. The Bloom Lab repo splits this across
three separate dataset folders (Omicron, convergent, BA2-4-5).

Output: data/processed/filtered_antibodies.csv
  Columns:
    antibody_id   - unique name/identifier
    heavy_seq     - heavy chain amino acid sequence (signal peptide stripped)
    light_seq     - light chain amino acid sequence (signal peptide stripped)
    d614g_ic50    - neutralization IC50 against index strain (numeric, ng/mL)
    dataset       - which Cao dataset the antibody came from
"""

import os
import re
import pandas as pd
import numpy as np

RAW_DIR  = "data/raw"
PROC_DIR = "data/processed"


# ---------------------------------------------------------------------------
# Signal peptide stripping
# ---------------------------------------------------------------------------
# Sequences in the Bloom Lab data sometimes include N-terminal signal peptides
# (typically 19-30 aa for IgG). We use a simple heuristic: truncate to the
# canonical variable-region start (VH starts QVQL/EVQL/DVQL/AVQL;
# VL starts DIQM/DIVM/EIVL/QSVL/SYEL etc.).
# IMGT-numbered sequences should already begin at position 1, but we apply
# this defensively to handle the raw sequences from Cao.

VH_START_PATTERNS = re.compile(r"((?:Q|E|D|A|V)VQL|(?:Q|E|D)IQL|QMQL)")
VL_START_PATTERNS = re.compile(r"((?:D|E|A|S|Q|I)(?:IQ|IV|VQ|SY|VV|TV|EL)(?:M|L|V)T)")


def strip_signal_peptide(seq: str, chain: str) -> str:
    """
    Remove N-terminal signal peptide from a raw antibody sequence.
    Falls back to the full sequence if no canonical start is found.
    """
    if not isinstance(seq, str) or len(seq) < 20:
        return seq
    pattern = VH_START_PATTERNS if chain == "heavy" else VL_START_PATTERNS
    match = pattern.search(seq)
    if match:
        return seq[match.start():]
    # Fallback: return as-is (may still have signal peptide)
    return seq


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------

def load_omicron(raw_dir: str) -> pd.DataFrame:
    """
    Load the 2022_Cao_Omicron dataset.
    All 247 antibodies have a numeric D614G_IC50, so all are index-strain binders.
    Columns of interest: name, D614G_IC50, Hchain, Lchain, epitope group.
    """
    path = os.path.join(raw_dir, "omicron_antibodies.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={
        "name":         "antibody_id",
        "Hchain":       "heavy_seq_raw",
        "Lchain":       "light_seq_raw",
        "D614G_IC50":   "d614g_ic50_raw",
        "epitope group":"bloom_epitope_group",
    })
    # IC50 is already numeric for this dataset; convert defensively
    df["d614g_ic50"] = pd.to_numeric(df["d614g_ic50_raw"], errors="coerce")
    df["dataset"] = "omicron"
    return df[["antibody_id", "heavy_seq_raw", "light_seq_raw",
               "d614g_ic50", "bloom_epitope_group", "dataset"]]


def load_convergent(raw_dir: str) -> pd.DataFrame:
    """
    Load the 2022_Cao_convergent dataset.
    Already contains Heavy/Light chain V gene pre-assignments.
    Index-strain binders: rows where D614G IC50 is a real number (not '>10').
    """
    path = os.path.join(raw_dir, "convergent_antibodies.csv")
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Antibody  Name":     "antibody_id",
        "D614G":              "d614g_ic50_raw",
        "Heavy chain V gene": "heavy_v_gene",
        "Heavy chain J gene": "heavy_j_gene",
        "Light chain V gene": "light_v_gene",
        "Light chain J gene": "light_j_gene",
        "Heavy chain AA":     "heavy_seq_raw",
        "Light chain AA":     "light_seq_raw",
    })
    df["d614g_ic50"] = pd.to_numeric(df["d614g_ic50_raw"], errors="coerce")
    df["dataset"] = "convergent"
    df["bloom_epitope_group"] = None
    return df[["antibody_id", "heavy_seq_raw", "light_seq_raw",
               "d614g_ic50", "heavy_v_gene", "heavy_j_gene",
               "light_v_gene", "light_j_gene",
               "bloom_epitope_group", "dataset"]]


def load_ba245(raw_dir: str) -> pd.DataFrame:
    """
    Load the 2022_Cao_BA2-4-5 dataset.
    Note: this file does not contain sequences directly — sequences must be
    looked up from the original paper's supplementary data or obtained via
    Genbank. We include this file here for neutralization metadata and
    epitope group clustering (antibody_clusters.csv).

    If you have the sequences, add them to a file at:
        data/raw/ba245_sequences.csv  (columns: antibody_id, heavy_seq, light_seq)
    """
    ab_path  = os.path.join(raw_dir, "ba245_antibodies.tsv")
    cl_path  = os.path.join(raw_dir, "ba245_clusters.csv")
    seq_path = os.path.join(raw_dir, "ba245_sequences.csv")

    df = pd.read_csv(ab_path, sep="\t")
    df = df.rename(columns={
        "Unnamed: 0":  "antibody_id",
        "D614G_IC50":  "d614g_ic50_raw",
    })
    df["d614g_ic50"] = pd.to_numeric(df["d614g_ic50_raw"], errors="coerce")

    # Merge cluster/epitope group labels
    if os.path.exists(cl_path):
        clusters = pd.read_csv(cl_path)[["antibody", "group"]].rename(
            columns={"antibody": "antibody_id", "group": "bloom_epitope_group"}
        )
        df = df.merge(clusters, on="antibody_id", how="left")
    else:
        df["bloom_epitope_group"] = None

    # Merge sequences if the supplementary file exists
    if os.path.exists(seq_path):
        seqs = pd.read_csv(seq_path)
        df = df.merge(seqs, on="antibody_id", how="left")
        df = df.rename(columns={"heavy_seq": "heavy_seq_raw",
                                 "light_seq": "light_seq_raw"})
    else:
        print("  [warning] ba245_sequences.csv not found — BA2-4-5 sequences skipped.")
        df["heavy_seq_raw"] = None
        df["light_seq_raw"] = None

    df["dataset"] = "ba245"
    return df[["antibody_id", "heavy_seq_raw", "light_seq_raw",
               "d614g_ic50", "bloom_epitope_group", "dataset"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PROC_DIR, exist_ok=True)

    print("Loading datasets ...")
    omicron    = load_omicron(RAW_DIR)
    convergent = load_convergent(RAW_DIR)
    ba245      = load_ba245(RAW_DIR)

    all_ab = pd.concat([omicron, convergent, ba245], ignore_index=True)
    print(f"  Total rows before filtering: {len(all_ab):,}")

    # ------------------------------------------------------------------
    # Step 2a: Filter to index-strain binders
    # Keep only antibodies with a real, finite numeric D614G IC50 value.
    # ">10", "Inf", "Inf*", "--" all become NaN after pd.to_numeric(..., errors='coerce').
    # ------------------------------------------------------------------
    binders = all_ab[all_ab["d614g_ic50"].notna()].copy()
    print(f"  Index-strain binders:        {len(binders):,}")

    # ------------------------------------------------------------------
    # Step 2b: Drop rows with missing sequences
    # ------------------------------------------------------------------
    has_seq = binders[binders["heavy_seq_raw"].notna() &
                      binders["light_seq_raw"].notna()].copy()
    print(f"  With both chain sequences:   {len(has_seq):,}")

    # ------------------------------------------------------------------
    # Step 2c: Strip signal peptides
    # ------------------------------------------------------------------
    has_seq["heavy_seq"] = has_seq["heavy_seq_raw"].apply(
        lambda s: strip_signal_peptide(s, "heavy")
    )
    has_seq["light_seq"] = has_seq["light_seq_raw"].apply(
        lambda s: strip_signal_peptide(s, "light")
    )

    # ------------------------------------------------------------------
    # Step 2d: Deduplicate by exact heavy+light sequence pair
    # When duplicates exist (same antibody appears in >1 Cao dataset),
    # keep the row with the most metadata (prefer convergent which has V genes).
    # ------------------------------------------------------------------
    has_seq["seq_key"] = has_seq["heavy_seq"] + "||" + has_seq["light_seq"]

    # Sort so 'convergent' rows (which have V gene data) are kept preferentially
    priority = {"convergent": 0, "omicron": 1, "ba245": 2}
    has_seq["_prio"] = has_seq["dataset"].map(priority)
    has_seq = has_seq.sort_values("_prio")
    deduped = has_seq.drop_duplicates(subset="seq_key", keep="first").copy()
    deduped = deduped.drop(columns=["_prio", "seq_key", "heavy_seq_raw", "light_seq_raw"])

    print(f"  After deduplication:         {len(deduped):,}")
    print(f"  (Holt et al. report 3,093 after filtering — expect similar count)")

    # ------------------------------------------------------------------
    # Step 2e: Save
    # ------------------------------------------------------------------
    out_path = os.path.join(PROC_DIR, "filtered_antibodies.csv")
    deduped.reset_index(drop=True).to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(f"Columns: {list(deduped.columns)}")

    # Dataset breakdown
    print("\n=== Breakdown by source dataset ===")
    print(deduped["dataset"].value_counts().to_string())


if __name__ == "__main__":
    main()
