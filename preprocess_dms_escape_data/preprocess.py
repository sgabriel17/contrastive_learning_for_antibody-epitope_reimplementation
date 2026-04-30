import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/jbloomlab/SARS2_RBD_Ab_escape_maps/d34cfe76edca0f263c61c4355eae8841523d7856/data/2022_Cao_Omicron"

# --- Step 1: Load what we already have ---
escape_df = pd.read_csv(f"{BASE_URL}/data.csv")
ab_df = pd.read_csv(f"{BASE_URL}/antibodies.csv")
ab_df = ab_df.rename(columns={"name": "condition", "epitope group": "epitope_bin"})

# --- Step 2: Find antibodies missing sequences ---
escape_conditions = set(escape_df["condition"].unique())
ab_conditions = set(ab_df["condition"].unique())
missing = escape_conditions - ab_conditions
print(f"Antibodies with sequences already: {len(ab_conditions)}")
print(f"Antibodies missing sequences:      {len(missing)}")

# --- Step 3: Download CoV-AbDab ---
covabdab = pd.read_csv("https://opig.stats.ox.ac.uk/webapps/covabdab/static/downloads/CoV-AbDab_080224.csv")
print(f"\nCoV-AbDab shape: {covabdab.shape}")
print(covabdab.columns.tolist())

# Step 4: Match by name using correct CoV-AbDab column names
covabdab_cols = covabdab[["Name", "VHorVHH", "VL"]].copy()
covabdab_cols = covabdab_cols.rename(columns={
    "Name":     "condition",
    "VHorVHH":  "Hchain",
    "VL":       "Lchain"
})

matched = covabdab_cols[covabdab_cols["condition"].isin(missing)]
print(f"Matched from CoV-AbDab: {len(matched)} / {len(missing)} missing antibodies")

# Preview matches
print(matched.head(5))


# check All_NAbs_Mutation

all_nabs = pd.read_csv(f"{BASE_URL}/All_NAbs_Mutation.csv")
print(all_nabs.shape)
print(all_nabs.columns.tolist())
print(all_nabs.head(3))