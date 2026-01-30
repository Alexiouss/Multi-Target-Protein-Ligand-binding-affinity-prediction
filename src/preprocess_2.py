from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem


# ----------------------------
# Helpers
# ----------------------------
def canonicalize_smiles(smiles: str) -> str:
    s = "" if smiles is None else str(smiles)
    mol = Chem.MolFromSmiles(s) if s else None
    return Chem.MolToSmiles(mol) if mol else s


def safe_numeric(series: pd.Series) -> pd.Series:
    """
    Robust numeric conversion:
    - "" -> NaN
    - "03.05" stays numeric (3.05) if it's in CSV as string
    - date-like spreadsheet artifacts (e.g. "05/08/2025") -> NaN
    """
    s = series.astype("string").str.strip()

    # common spreadsheet date-like patterns -> NaN
    # (you can add more patterns if you see them)
    date_like = (
        s.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", na=False) |
        s.str.match(r"^\d{4}-\d{1,2}-\d{1,2}$", na=False)
    )
    s = s.mask(date_like, other=pd.NA)

    s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    return pd.to_numeric(s, errors="coerce")


def first_non_null(x: pd.Series):
    x = x.replace("", np.nan)
    x = x.dropna()
    return x.iloc[0] if len(x) else np.nan


# ----------------------------
# Main processing
# ----------------------------
def preprocess_csv(
    in_csv: str,
    out_csv: str,
    enforce_full_matrix: bool = True,
):
    in_csv = str(in_csv)
    out_csv = str(out_csv)

    df = pd.read_csv(in_csv, dtype="string", keep_default_na=False)

    # ---- Required columns (adjust if your headers differ)
    required = ["SMILES", "Protein", "target_chembl_id", "binding_affinity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Some files have "ChEMBL ID" instead of "ChEMBL ID" / "ChEMBL_ID"
    # We keep whatever exists. If none exists, we won't fill it.
    chembl_id_col = None
    for c in ["ChEMBL ID", "ChEMBL_ID", "chembl_id"]:
        if c in df.columns:
            chembl_id_col = c
            break

    # ---- Normalize core fields
    df["SMILES"] = df["SMILES"].astype("string").str.strip()
    df["Protein"] = df["Protein"].astype("string").str.strip()
    df["target_chembl_id"] = df["target_chembl_id"].astype("string").str.strip()

    if "not_measured" not in df.columns:
        df["not_measured"] = "0"
    df["not_measured"] = safe_numeric(df["not_measured"]).fillna(0).astype(int)

    # binding_affinity -> numeric, garbage -> NaN
    df["binding_affinity"] = safe_numeric(df["binding_affinity"])

    # ---- canonical_smiles
    if "canonical_smiles" not in df.columns:
        df["canonical_smiles"] = df["SMILES"].apply(canonicalize_smiles)
    else:
        # fill canonical_smiles where empty
        df["canonical_smiles"] = df["canonical_smiles"].replace("", np.nan)
        df.loc[df["canonical_smiles"].isna(), "canonical_smiles"] = df.loc[
            df["canonical_smiles"].isna(), "SMILES"
        ].apply(canonicalize_smiles)

    # ---- Define "molecule-level" columns to replicate across all targets for same SMILES
    # (these are the ones that are currently blank in your generated rows)
    molecule_cols = []
    for c in ["Molecular Weight", "LogP", "HBA", "HBD", "TPSA", "Name"]:
        if c in df.columns:
            molecule_cols.append(c)

    # numeric molecule cols -> numeric
    for c in ["Molecular Weight", "LogP", "HBA", "HBD", "TPSA"]:
        if c in df.columns:
            df[c] = safe_numeric(df[c])

    # ---- Build TARGET META mapping (target_chembl_id -> Protein, ChEMBL ID, etc.)
    # Use rows that actually have those fields filled.
    meta_cols = ["Protein"]
    if chembl_id_col is not None:
        meta_cols.append(chembl_id_col)

    target_meta = (
        df[df["target_chembl_id"].notna() & (df["target_chembl_id"] != "")]
          .groupby("target_chembl_id")[meta_cols]
          .agg(first_non_null)
          .reset_index()
    )
    target_meta = target_meta.set_index("target_chembl_id").to_dict(orient="index")

    # ---- Fill missing target meta in rows (Protein / ChEMBL ID) using mapping
    def fill_target_meta_row(row):
        tid = row["target_chembl_id"]
        if not tid or pd.isna(tid):
            return row
        meta = target_meta.get(tid, None)
        if meta is None:
            return row
        # fill Protein / ChEMBL ID if empty
        if "Protein" in meta:
            if (row.get("Protein", "") == "") or pd.isna(row.get("Protein", np.nan)):
                row["Protein"] = meta["Protein"]
        if chembl_id_col is not None and chembl_id_col in meta:
            if (row.get(chembl_id_col, "") == "") or pd.isna(row.get(chembl_id_col, np.nan)):
                row[chembl_id_col] = meta[chembl_id_col]
        return row

    df = df.apply(fill_target_meta_row, axis=1)

    # ---- Fill missing molecule-level cols within each canonical_smiles group
    # Template per molecule = first non-null values of those cols
    if molecule_cols:
        templates = (
            df.groupby("canonical_smiles", as_index=True)[molecule_cols]
            .agg(first_non_null)
        )

        for c in molecule_cols:
            df[c] = df[c].replace("", np.nan)
            df[c] = df[c].fillna(df["canonical_smiles"].map(templates[c]))

    # ---- Ensure "not_measured=1 rows have binding_affinity=0"
    # (your rule: missing-target rows => affinity 0, not_measured 1)
    df.loc[df["not_measured"] == 1, "binding_affinity"] = 0.0

    # ---- If enforce_full_matrix: for each molecule, ensure a row exists for every target
    if enforce_full_matrix:
        all_targets = sorted(df["target_chembl_id"].dropna().unique().tolist())
        if len(all_targets) == 0:
            raise RuntimeError("No targets found in target_chembl_id column.")

        # For each canonical_smiles, create missing target rows using a template row for that molecule
        new_rows = []
        for csmi, sub in df.groupby("canonical_smiles", sort=False):
            present = set(sub["target_chembl_id"].dropna().tolist())
            missing_targets = [t for t in all_targets if t not in present]
            if not missing_targets:
                continue

            # template row: take first row in group (already filled molecule cols)
            base = sub.iloc[0].copy()

            for t in missing_targets:
                r = base.copy()
                r["target_chembl_id"] = t

                # Fill target meta
                meta = target_meta.get(t, {})
                if "Protein" in df.columns:
                    r["Protein"] = meta.get("Protein", r.get("Protein", ""))

                if chembl_id_col is not None:
                    r[chembl_id_col] = meta.get(chembl_id_col, r.get(chembl_id_col, ""))

                # Mark missing measurement
                r["binding_affinity"] = 0.0
                r["not_measured"] = 1

                # Optional: clear assay-specific fields if they exist
                for c in ["standard_type", "standard_value", "standard_units"]:
                    if c in df.columns:
                        r[c] = ""  # keep empty, it’s fine

                new_rows.append(r)

        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

            # After adding, re-fill molecule-level columns once more (safety)
            if molecule_cols:
                templates = (
                    df.groupby("canonical_smiles")[molecule_cols]
                      .agg(first_non_null)
                      .reset_index()
                      .set_index("canonical_smiles")
                )
                for c in molecule_cols:
                    df[c] = df[c].replace("", np.nan)
                    df[c] = df[c].fillna(df["canonical_smiles"].map(templates[c]))

            # enforce the rule again
            df.loc[df["not_measured"] == 1, "binding_affinity"] = 0.0

    # ---- Final cleanup: make sure required numeric types are numeric
    df["binding_affinity"] = pd.to_numeric(df["binding_affinity"], errors="coerce").fillna(0.0)
    df["not_measured"] = pd.to_numeric(df["not_measured"], errors="coerce").fillna(0).astype(int)

    # Save
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("✅ Saved:", out_csv)
    print("Rows:", len(df))
    print("Targets:", df["target_chembl_id"].nunique())
    print("Molecules:", df["canonical_smiles"].nunique())
    print("Missing not_measured==1 rows:", int((df["not_measured"] == 1).sum()))


if __name__ == "__main__":
    # ---- EDIT THESE PATHS
    IN  = r"data\chembl\chembl_affinity_dataset.csv"
    OUT = r"data\chembl\chembl_affinity_dataset_filled.csv"

    preprocess_csv(IN, OUT, enforce_full_matrix=True)
