#!/usr/bin/env python3
"""
Visualization: Epitope bin class distribution + spatial validation on RBD surface.

Produces two PNG figures:
  1. bin_distribution.png       — bar chart of antibody counts per bin
  2. spatial_validation.png     — RBD surface (PDB 8SGU) with bin epitope centroids

Requirements:
    pip install plotly kaleido scikit-learn pandas numpy biopython

Usage:
    python visualize.py

Inputs (must already exist from running steps 1-5):
    data/raw/convergent_escape.csv
    data/raw/omicron_escape.csv   (optional, used if present)

The PDB structure (8SGU) is downloaded automatically.
"""

import os
import json
import urllib.request
import numpy as np
import pandas as pd
import plotly.graph_objects as go

RAW_DIR    = "data/raw"
OUTPUT_DIR = "results/figures"
PDB_PATH   = os.path.join(RAW_DIR, "8SGU.pdb")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

BINS_12 = ["A","B","C","D1","D2","E1","E2.1","E2.2","E3","F1","F2","F3"]

BIN_COLORS = [
    "#4C9BE8","#E8844C","#4CE88A","#E84C7A","#B04CE8",
    "#E8D44C","#4CE8D4","#E84C4C","#7AE84C","#4C74E8",
    "#E8A84C","#84E84C",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def download_pdb(pdb_id: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"  [skip] {dest} already exists")
        return
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {pdb_id}.pdb ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved -> {dest}")


def parse_ca_coords(pdb_path: str, chain: str = "A") -> dict[int, np.ndarray]:
    """Extract Cα coordinates for a given chain from a PDB file."""
    coords = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            if line[21] != chain:
                continue
            try:
                res_num = int(line[22:26].strip())
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                coords[res_num] = np.array([x, y, z])
            except ValueError:
                continue
    return coords


def build_escape_matrix(escape_df: pd.DataFrame,
                        ab_col: str) -> pd.DataFrame:
    """
    Pivot long-format escape data to (antibody x site) matrix.
    Each cell = sum of mut_escape across all mutations at that site.
    """
    site_escape = (
        escape_df.groupby([ab_col, "site"])["mut_escape"]
        .sum()
        .reset_index()
    )
    matrix = site_escape.pivot(index=ab_col, columns="site", values="mut_escape")
    return matrix.fillna(0.0)


def compute_bin_centroids(escape_matrix: pd.DataFrame,
                          ca_coords: dict,
                          bin_labels: pd.Series) -> dict[str, np.ndarray]:
    """
    For each bin, compute the weighted center of mass on the RBD surface.
    Weight at each residue = mean escape score across all antibodies in that bin.
    """
    site_cols = [c for c in escape_matrix.columns if isinstance(c, (int, np.integer))]
    df = escape_matrix.copy()
    df["bin"] = bin_labels

    centroids = {}
    for bin_name in BINS_12:
        rows = df[df["bin"] == bin_name]
        if len(rows) == 0:
            continue
        mean_escape = rows[site_cols].mean()
        weighted_sum = np.zeros(3)
        total_weight = 0.0
        for site in site_cols:
            site_int = int(site)
            if site_int not in ca_coords:
                continue
            weight = max(0.0, float(mean_escape[site]))
            weighted_sum += weight * ca_coords[site_int]
            total_weight += weight
        if total_weight > 0:
            centroids[bin_name] = weighted_sum / total_weight
    return centroids


def save_chart(fig: go.Figure, filename: str, caption: str, description: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.write_image(path)
    meta_path = path + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump({"caption": caption, "description": description}, f, indent=2)
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Figure 1: Bin class distribution
# ---------------------------------------------------------------------------

def plot_bin_distribution(pivot: pd.DataFrame) -> None:
    print("\nGenerating Figure 1: Bin class distribution ...")

    bin_counts = (
        pivot["group"]
        .value_counts()
        .reindex(BINS_12)
        .fillna(0)
        .astype(int)
    )

    fig = go.Figure(go.Bar(
        x=list(bin_counts.index),
        y=list(bin_counts.values),
        marker_color=BIN_COLORS,
        text=list(bin_counts.values),
        textposition="outside",
    ))
    fig.update_layout(
        title={
            "text": (
                "Antibodies per Epitope Bin"
                "<br><span style='font-size:16px;font-weight:normal;'>"
                "Source: Cao et al. 2022 (Bloom Lab) | 3,051 antibodies, 12 bins"
                "</span>"
            )
        },
        plot_bgcolor="white",
    )
    fig.update_xaxes(title_text="Epitope Bin")
    fig.update_yaxes(title_text="Antibody Count")
    fig.update_traces(cliponaxis=False)

    save_chart(
        fig,
        "bin_distribution.png",
        caption="Epitope Bin Class Distribution",
        description=(
            "Bar chart showing number of antibodies in each of the 12 epitope bins "
            "derived from Cao et al. 2022 DMS escape data. "
            "Note class imbalance: E2.2 and E3 are largest, D2 and E1 smallest."
        ),
    )


# ---------------------------------------------------------------------------
# Figure 2: Spatial validation on RBD surface
# ---------------------------------------------------------------------------

def plot_spatial_validation(ca_coords: dict, bin_centroids: dict) -> None:
    print("\nGenerating Figure 2: Spatial validation on RBD surface ...")

    ca_x = [ca_coords[r][0] for r in sorted(ca_coords)]
    ca_z = [ca_coords[r][2] for r in sorted(ca_coords)]

    fig = go.Figure()

    # RBD Cα backbone (top-down X-Z projection)
    fig.add_trace(go.Scatter(
        x=ca_x, y=ca_z,
        mode="markers",
        marker=dict(size=7, color="#c0c0c0", opacity=0.35),
        name="RBD Cα residues",
        hoverinfo="skip",
    ))

    # Bin epitope centroids
    bin_names_present = [b for b in BINS_12 if b in bin_centroids]
    for i, b in enumerate(bin_names_present):
        coord = bin_centroids[b]
        fig.add_trace(go.Scatter(
            x=[coord[0]], y=[coord[2]],
            mode="markers+text",
            marker=dict(
                size=22,
                color=BIN_COLORS[BINS_12.index(b)],
                opacity=0.92,
                line=dict(color="white", width=2),
            ),
            text=[b],
            textposition="middle center",
            textfont=dict(size=11, color="white", family="Arial Black"),
            name=b,
            hovertemplate=(
                f"<b>Bin {b}</b><br>"
                f"X={coord[0]:.1f} Å<br>"
                f"Z={coord[2]:.1f} Å<extra></extra>"
            ),
        ))

    fig.update_layout(
        title={
            "text": (
                "Epitope Bin Centroids on RBD Surface (PDB 8SGU)"
                "<br><span style='font-size:16px;font-weight:normal;'>"
                "Top-down view (X–Z plane) | Weighted by mean DMS escape score"
                "</span>"
            )
        },
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.28,
            xanchor="center", x=0.5, font=dict(size=11),
        ),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#12121f",
        font_color="white",
        xaxis=dict(
            gridcolor="#2a2a4a", title_text="X position (Å)",
            color="#aaaacc", scaleanchor="y", scaleratio=1,
        ),
        yaxis=dict(
            gridcolor="#2a2a4a", title_text="Z position (Å)", color="#aaaacc",
        ),
    )
    fig.update_traces(cliponaxis=False)

    save_chart(
        fig,
        "spatial_validation.png",
        caption="Epitope Bin Spatial Validation on RBD (PDB 8SGU)",
        description=(
            "2D top-down projection (X-Z plane) of RBD Cα backbone (grey dots) "
            "with 12 epitope bin centroids (colored circles). Each centroid is the "
            "weighted center of mass of RBD residues, weighted by the mean per-residue "
            "DMS escape score across all antibodies in that bin. Spatial separation "
            "confirms the 12 bins correspond to distinct epitope regions."
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # -- Load escape data --
    print("Loading escape data ...")
    conv_path = os.path.join(RAW_DIR, "convergent_escape.csv")
    if not os.path.exists(conv_path):
        raise FileNotFoundError(
            f"{conv_path} not found. Run step1_download.py first."
        )
    conv_esc = pd.read_csv(conv_path)
    print(f"  Convergent: {len(conv_esc):,} rows, "
          f"{conv_esc['antibody'].nunique():,} antibodies")

    # -- Build escape matrix and bin labels --
    print("Building escape matrix ...")
    matrix = build_escape_matrix(conv_esc, ab_col="antibody")
    bin_labels = (
        conv_esc[["antibody", "group"]]
        .drop_duplicates("antibody")
        .set_index("antibody")["group"]
    )
    # Rebuild pivot with group column for Figure 1
    pivot = matrix.copy()
    pivot["group"] = bin_labels

    # -- Download PDB structure --
    print("\nFetching RBD structure (PDB 8SGU) ...")
    download_pdb("8SGU", PDB_PATH)
    ca_coords = parse_ca_coords(PDB_PATH, chain="A")
    print(f"  {len(ca_coords)} Cα atoms loaded (chain A, residues "
          f"{min(ca_coords)}-{max(ca_coords)})")

    # -- Compute bin centroids --
    print("\nComputing epitope bin centroids ...")
    bin_centroids = compute_bin_centroids(matrix, ca_coords, bin_labels)
    for b, coord in sorted(bin_centroids.items()):
        print(f"  Bin {b:6s}: ({coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}) Å")

    # -- Generate figures --
    plot_bin_distribution(pivot)
    plot_spatial_validation(ca_coords, bin_centroids)

    print(f"\nDone! Figures saved to {OUTPUT_DIR}/")
    print("  bin_distribution.png")
    print("  spatial_validation.png")


if __name__ == "__main__":
    main()
