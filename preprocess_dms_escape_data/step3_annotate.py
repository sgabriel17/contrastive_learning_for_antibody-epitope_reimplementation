#!/usr/bin/env python3
"""
Step 3: Annotate each antibody with:
  - Heavy and light chain V gene (IMGT nomenclature)
  - Heavy and light chain J gene
  - CDRH3 amino acid sequence
  - CDRL3 amino acid sequence

Strategy:
  - The 'convergent' dataset already ships with V/J gene assignments.
  - The 'omicron' and 'ba245' antibodies need to be annotated using ANARCI.

ANARCI (Antibody Numbering And Receptor ClassIfication) installation:
    pip install abnumber   # lightweight wrapper; uses HMMER internally
    # OR
    conda install -c bioconda anarci

If ANARCI is unavailable, this script falls back to a regex-based CDR3
extraction heuristic using canonical CDR3 flanking residues (less reliable
but useful for quick exploration).

Output: data/processed/annotated_antibodies.csv
  New columns added to filtered_antibodies.csv:
    heavy_v_gene, heavy_j_gene, light_v_gene, light_j_gene  (IMGT names)
    cdrh3_aa                                                  (CDRH3 aa sequence)
    cdrl3_aa                                                  (CDRL3 aa sequence)
    annotation_source  ('precomputed', 'anarci', or 'heuristic')
"""

import os
import re
import pandas as pd
import numpy as np

PROC_DIR = "data/processed"
INPUT    = os.path.join(PROC_DIR, "filtered_antibodies.csv")
OUTPUT   = os.path.join(PROC_DIR, "annotated_antibodies.csv")


# ---------------------------------------------------------------------------
# ANARCI-based annotation (primary method)
# ---------------------------------------------------------------------------

def annotate_with_anarci(sequences: list[str], chain_type: str) -> list[dict]:
    """
    Run ANARCI on a list of amino acid sequences.
    chain_type: 'H' for heavy, 'K'/'L' for kappa/lambda light.

    Returns a list of dicts, one per input sequence, with keys:
        v_gene, j_gene, cdr3_aa, success (bool)
    """
    try:
        from anarci import anarci
    except ImportError:
        raise ImportError(
            "ANARCI not installed. Run: pip install abnumber  "
            "or  conda install -c bioconda anarci"
        )

    results = []
    for seq in sequences:
        if not isinstance(seq, str) or len(seq) < 20:
            results.append({"v_gene": None, "j_gene": None,
                             "cdr3_aa": None, "success": False})
            continue
        try:
            numbered, alignment_details, hit_tables = anarci(
                [("seq", seq)],
                scheme="imgt",
                assign_germline=True,
                allowed_species=["human"],
            )
            if numbered[0] is None:
                results.append({"v_gene": None, "j_gene": None,
                                 "cdr3_aa": None, "success": False})
                continue

            details = alignment_details[0][0]
            v_gene  = details.get("germline_alignment", {}).get("v_gene", [None, None])[1]
            j_gene  = details.get("germline_alignment", {}).get("j_gene", [None, None])[1]

            # Extract CDR3 using IMGT positions 105-117 (inclusive)
            num_seq  = numbered[0][0]  # list of ((pos, insertion), aa)
            cdr3_pos = set(range(105, 118))
            cdr3_aas = [aa for (pos, ins), aa in num_seq
                        if pos in cdr3_pos and aa != "-"]
            cdr3 = "".join(cdr3_aas) if cdr3_aas else None

            results.append({
                "v_gene":   v_gene,
                "j_gene":   j_gene,
                "cdr3_aa":  cdr3,
                "success":  True,
            })
        except Exception:
            results.append({"v_gene": None, "j_gene": None,
                             "cdr3_aa": None, "success": False})
    return results


# ---------------------------------------------------------------------------
# Heuristic CDR3 extraction (fallback when ANARCI is unavailable)
# ---------------------------------------------------------------------------
# VH CDR3: between conserved Cys (C) at end of FR3 and Trp-Gly-X-Gly (WGXG)
#           or Trp-Gly-Gln-Gly (WGQG) motif in FR4
# VL CDR3: between conserved Cys at end of FR3 and Phe-Gly-X-Gly (FGXG) in FR4

CDRH3_PATTERN = re.compile(r"C([A-Z]{3,30}?)W[GA][QRX]G")
CDRL3_PATTERN = re.compile(r"C([A-Z]{3,25}?)FG[A-Z]G")


def extract_cdr3_heuristic(seq: str, chain: str) -> str | None:
    """
    Extract CDR3 sequence using conserved flanking residue motifs.
    Less reliable than ANARCI but requires no dependencies.
    """
    if not isinstance(seq, str):
        return None
    pattern = CDRH3_PATTERN if chain == "heavy" else CDRL3_PATTERN
    matches = list(pattern.finditer(seq))
    if not matches:
        return None
    # Take the last match (closest to the C-terminus, most likely FR3/CDR3 boundary)
    return matches[-1].group(1)


# ---------------------------------------------------------------------------
# Main annotation loop
# ---------------------------------------------------------------------------

def annotate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate all antibodies in df with V/J genes and CDR3 sequences.

    Logic:
    1. Rows from the 'convergent' dataset already have V/J gene columns —
       copy those and only run CDR3 extraction.
    2. All other rows go through ANARCI (or heuristic fallback).
    """
    # Initialise output columns as object dtype (required for pandas 2.x string assignment)
    for col in ["heavy_v_gene", "heavy_j_gene", "light_v_gene", "light_j_gene",
                "cdrh3_aa", "cdrl3_aa", "annotation_source"]:
        if col not in df.columns:
            df[col] = pd.array([None] * len(df), dtype=object)
        else:
            # Ensure existing columns are object dtype so strings can be assigned
            df[col] = df[col].astype(object)

    # --- 1. Propagate pre-computed V/J genes from the convergent dataset ---
    precomp_mask = df["dataset"] == "convergent"
    # Columns may already exist from step 2; ensure they're populated.
    for col in ["heavy_v_gene", "heavy_j_gene", "light_v_gene", "light_j_gene"]:
        src_col = col  # same name in convergent loader output
        if src_col in df.columns:
            # Already filled from load_convergent; mark source
            pass
    df.loc[precomp_mask, "annotation_source"] = "precomputed"

    n_precomp = precomp_mask.sum()
    print(f"  Pre-computed annotations (convergent dataset): {n_precomp:,}")

    # --- 2. Try ANARCI for remaining rows ---
    needs_annotation = df["annotation_source"].isna()
    n_needs = needs_annotation.sum()
    print(f"  Rows needing annotation (ANARCI or heuristic): {n_needs:,}")

    anarci_available = False
    try:
        import anarci  # noqa: F401
        anarci_available = True
        print("  ANARCI detected — using ANARCI for annotation.")
    except ImportError:
        print("  ANARCI not installed — falling back to heuristic CDR3 extraction.")
        print("  Install with:  pip install abnumber  (or conda install -c bioconda anarci)")

    to_annotate = df[needs_annotation].copy()

    if anarci_available and len(to_annotate) > 0:
        print(f"  Running ANARCI on {len(to_annotate):,} heavy chains ...")
        h_results = annotate_with_anarci(to_annotate["heavy_seq"].tolist(), "H")
        print(f"  Running ANARCI on {len(to_annotate):,} light chains ...")
        l_results = annotate_with_anarci(to_annotate["light_seq"].tolist(), "L")

        to_annotate["heavy_v_gene"] = [r["v_gene"]  for r in h_results]
        to_annotate["heavy_j_gene"] = [r["j_gene"]  for r in h_results]
        to_annotate["cdrh3_aa"]     = [r["cdr3_aa"] for r in h_results]
        to_annotate["light_v_gene"] = [r["v_gene"]  for r in l_results]
        to_annotate["light_j_gene"] = [r["j_gene"]  for r in l_results]
        to_annotate["cdrl3_aa"]     = [r["cdr3_aa"] for r in l_results]
        to_annotate["annotation_source"] = to_annotate.apply(
            lambda r: "anarci" if h_results[r.name]["success"] else "anarci_failed",
            axis=1,
        )
    else:
        # Heuristic fallback
        to_annotate["cdrh3_aa"] = to_annotate["heavy_seq"].apply(
            lambda s: extract_cdr3_heuristic(s, "heavy")
        )
        to_annotate["cdrl3_aa"] = to_annotate["light_seq"].apply(
            lambda s: extract_cdr3_heuristic(s, "light")
        )
        to_annotate["annotation_source"] = "heuristic"

    # Write back annotated rows (pandas 2.x: use loc to avoid dtype coercion issues)
    for col in ["heavy_v_gene", "heavy_j_gene", "light_v_gene", "light_j_gene",
                "cdrh3_aa", "cdrl3_aa", "annotation_source"]:
        if col in to_annotate.columns:
            df.loc[needs_annotation, col] = to_annotate[col].values

    # --- 3. Extract CDR3 for convergent rows (have V genes but no CDR3 yet) ---
    # The convergent file does not include CDR3 sequences directly.
    if anarci_available:
        conv_no_cdr3 = precomp_mask & df["cdrh3_aa"].isna()
        if conv_no_cdr3.sum() > 0:
            print(f"  Extracting CDR3 for {conv_no_cdr3.sum():,} convergent rows via ANARCI ...")
            conv_rows = df[conv_no_cdr3].copy()
            h_res = annotate_with_anarci(conv_rows["heavy_seq"].tolist(), "H")
            l_res = annotate_with_anarci(conv_rows["light_seq"].tolist(), "L")
            df.loc[conv_no_cdr3, "cdrh3_aa"] = [r["cdr3_aa"] for r in h_res]
            df.loc[conv_no_cdr3, "cdrl3_aa"] = [r["cdr3_aa"] for r in l_res]
    else:
        conv_no_cdr3 = precomp_mask & df["cdrh3_aa"].isna()
        df.loc[conv_no_cdr3, "cdrh3_aa"] = df.loc[conv_no_cdr3, "heavy_seq"].apply(
            lambda s: extract_cdr3_heuristic(s, "heavy")
        )
        df.loc[conv_no_cdr3, "cdrl3_aa"] = df.loc[conv_no_cdr3, "light_seq"].apply(
            lambda s: extract_cdr3_heuristic(s, "light")
        )

    return df


def main():
    print("Loading filtered antibodies ...")
    df = pd.read_csv(INPUT)
    print(f"  {len(df):,} antibodies loaded")

    print("\nAnnotating ...")
    df = annotate_dataframe(df)

    # ------------------------------------------------------------------
    # Validation summary
    # ------------------------------------------------------------------
    print("\n=== Annotation Summary ===")
    print(df["annotation_source"].value_counts().to_string())

    h_v_filled = df["heavy_v_gene"].notna().sum()
    l_v_filled = df["light_v_gene"].notna().sum()
    cdrh3_filled = df["cdrh3_aa"].notna().sum()
    cdrl3_filled = df["cdrl3_aa"].notna().sum()
    print(f"\n  heavy_v_gene filled:  {h_v_filled:,} / {len(df):,}")
    print(f"  light_v_gene filled:  {l_v_filled:,} / {len(df):,}")
    print(f"  cdrh3_aa filled:      {cdrh3_filled:,} / {len(df):,}")
    print(f"  cdrl3_aa filled:      {cdrl3_filled:,} / {len(df):,}")

    # Warn on rows missing V gene or CDR3 — these cannot be used for
    # clone-aware splitting in Step 5 and should be inspected or dropped.
    incomplete = df[df["heavy_v_gene"].isna() | df["cdrh3_aa"].isna()]
    if len(incomplete) > 0:
        print(f"\n  [warning] {len(incomplete)} rows missing heavy V gene or CDRH3 — "
              f"will be excluded from clone-aware splitting in Step 5.")
        incomplete[["antibody_id", "dataset", "annotation_source"]].to_csv(
            os.path.join(PROC_DIR, "annotation_failures.csv"), index=False
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    col_order = [
        "antibody_id", "dataset",
        "heavy_seq", "light_seq",
        "heavy_v_gene", "heavy_j_gene",
        "light_v_gene", "light_j_gene",
        "cdrh3_aa", "cdrl3_aa",
        "d614g_ic50", "bloom_epitope_group",
        "annotation_source",
    ]
    # Only keep columns that exist
    col_order = [c for c in col_order if c in df.columns]
    df[col_order].to_csv(OUTPUT, index=False)
    print(f"\nSaved -> {OUTPUT}")


if __name__ == "__main__":
    main()
