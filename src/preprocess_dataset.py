from pathlib import Path
import pandas as pd
import numpy as np

def complete_csv_with_not_measured(
    input_csv: str,
    output_csv: str,
    smiles_col="SMILES",
    protein_col="Protein",
    affinity_col="binding_affinity",
    target_id_col="target_chembl_id",
):
    df = pd.read_csv(input_csv).copy()
    df[affinity_col] = pd.to_numeric(df[affinity_col], errors="coerce")

    smiles = df[smiles_col].astype(str).dropna().unique()
    proteins = df[protein_col].astype(str).dropna().unique()

    # optional: pick canonical target_chembl_id per protein (for new rows)
    target_lookup = None
    if target_id_col in df.columns:
        target_lookup = (
            df.dropna(subset=[protein_col, target_id_col])
              .groupby(protein_col)[target_id_col]
              .agg(lambda s: s.value_counts().index[0])
              .to_dict()
        )

    full = pd.MultiIndex.from_product([smiles, proteins], names=[smiles_col, protein_col]).to_frame(index=False)
    merged = full.merge(df, on=[smiles_col, protein_col], how="left")

    # not_measured = 1 if affinity missing, else 0
    merged["not_measured"] = merged[affinity_col].isna().astype(int)

    # fill missing affinity with 0
    merged[affinity_col] = merged[affinity_col].fillna(0.0)

    # fill target_id for imputed rows if possible
    if target_lookup is not None:
        merged[target_id_col] = merged[target_id_col].fillna(merged[protein_col].map(target_lookup))

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return {
        "rows_out": int(len(merged)),
        "imputed_rows": int(merged["not_measured"].sum()),
        "measured_rows": int((merged["not_measured"] == 0).sum()),
    }

if __name__ == "__main__":
    complete_csv_with_not_measured('data/chembl/chembl_affinity_dataset.csv','data/chembl/chembl_affinity_dataset.csv')