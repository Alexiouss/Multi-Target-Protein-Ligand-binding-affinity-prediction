"""
Data loader for cached ChEMBL dataset CSV.

Returns:
  train_loader, val_loader, test_loader, label_stats

Batch keys:
  - molecules: List[str]
  - protein_names: List[str]
  - target_chembl_ids: List[str]
  - target_indices: LongTensor [B]
  - labels: FloatTensor [B]                      (single label per row, raw)
  - individual_affinities: FloatTensor [B, T]    (standardized scalar in the single target column)
  - affinity_mask: FloatTensor [B, T]            (1 where measured, 0 where not_measured)
  - molecular_properties: FloatTensor [B, 5]
"""

from pathlib import Path
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from rdkit import Chem


# ----------------------------
# Helpers
# ----------------------------
def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol else smiles

def random_split(df, test_size=0.2, val_size=0.2, seed=42, stratify_col="target_chembl_id"):
    """
    Random row split (no grouping by SMILES).
    Optionally stratify by target_chembl_id to preserve target distribution.
    """
    strat = None
    if stratify_col is not None and stratify_col in df.columns:
        # sklearn stratify requires no NaNs and enough samples per class
        strat = df[stratify_col].astype(str)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=strat
    )

    strat2 = None
    if strat is not None:
        strat2 = train_df[stratify_col].astype(str)

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size / (1.0 - test_size),
        random_state=seed,
        shuffle=True,
        stratify=strat2
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def safe_float(x, default=np.nan) -> float:
    """
    Robust float conversion:
    - '' / None / NaN -> default
    - '  ' -> default
    - normal numeric strings -> float
    """
    if x is None:
        return float(default)
    if isinstance(x, (float, int, np.floating, np.integer)):
        # already numeric
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return float(default)
    try:
        return float(s)
    except Exception:
        return float(default)


def safe_int(x, default=0) -> int:
    if x is None:
        return int(default)
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return int(default)
    try:
        return int(float(s))
    except Exception:
        return int(default)


def safe_numeric_series(series: pd.Series) -> pd.Series:
    """
    Convert a string series to numeric safely.
    - empty -> NaN
    - date-like artifacts -> NaN
    """
    s = series.astype("string").str.strip()

    # kill common date formats caused by spreadsheet parsing
    date_like = (
        s.str.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", na=False) |
        s.str.match(r"^\d{4}-\d{1,2}-\d{1,2}$", na=False)
    )
    s = s.mask(date_like, other=pd.NA)

    s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "null": pd.NA})
    return pd.to_numeric(s, errors="coerce")


def group_split(df, test_size=0.2, val_size=0.2, seed=42):
    """
    Prevent any molecule (canonical SMILES) from appearing in multiple splits.
    """
    if "canonical_smiles" not in df.columns:
        df["canonical_smiles"] = df["SMILES"].apply(canonicalize_smiles)

    groups = df["canonical_smiles"].astype(str).unique()
    train_groups, test_groups = train_test_split(groups, test_size=test_size, random_state=seed)
    train_groups, val_groups = train_test_split(
        train_groups,
        test_size=val_size / (1.0 - test_size),
        random_state=seed,
    )

    train = df[df["canonical_smiles"].isin(train_groups)].reset_index(drop=True)
    val   = df[df["canonical_smiles"].isin(val_groups)].reset_index(drop=True)
    test  = df[df["canonical_smiles"].isin(test_groups)].reset_index(drop=True)
    return train, val, test


def compute_target_stats(train_df: pd.DataFrame, unique_targets: list):
    stats = {}
    for t in unique_targets:
        sub = train_df[train_df["target_chembl_id"].astype(str) == str(t)]
        if "not_measured" in sub.columns:
            sub = sub[sub["not_measured"].astype(int) == 0]

        y = pd.to_numeric(sub["binding_affinity"], errors="coerce").dropna().astype(float).values
        if len(y) < 2:
            mu, sd = 0.0, 1.0
        else:
            mu = float(np.mean(y))
            sd = float(np.std(y) + 1e-12)

        stats[str(t)] = {"mean": mu, "std": sd, "n": int(len(y))}
    return stats


# ----------------------------
# Dataset + collate
# ----------------------------
class ChEMBLAffinityDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_to_idx: dict, target_stats: dict, use_chembl: bool):
        self.df = df.reset_index(drop=True)
        self.target_to_idx = target_to_idx
        self.target_stats = target_stats
        self.use_chembl = use_chembl

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        target_id = str(r["target_chembl_id"])
        target_idx = self.target_to_idx[target_id]

        # binding affinity may still be NaN in weird rows; treat as missing
        y = safe_float(r.get("binding_affinity", np.nan), default=np.nan)

        not_measured = safe_int(r.get("not_measured", 0), default=0)

        # Standardize label if ChEMBL
        if self.use_chembl:
            mu = self.target_stats[target_id]["mean"]
            sd = self.target_stats[target_id]["std"]
            # if not_measured OR y is NaN, set std label to 0 and mask will handle it
            if not_measured == 0 and np.isfinite(y):
                y_std = (y - mu) / sd
            else:
                y_std = 0.0
        else:
            y_std = 0.0 if (not_measured == 1 or not np.isfinite(y)) else float(y)

        mw   = safe_float(r.get("Molecular Weight", np.nan), default=np.nan)
        logp = safe_float(r.get("LogP", np.nan), default=np.nan)
        hba  = safe_float(r.get("HBA", np.nan), default=np.nan)
        hbd  = safe_float(r.get("HBD", np.nan), default=np.nan)
        tpsa = safe_float(r.get("TPSA", np.nan), default=np.nan)

        return {
            "molecules": str(r["SMILES"]),
            "protein_name": str(r.get("Protein", "")),
            "target_chembl_id": target_id,
            "target_idx": int(target_idx),
            "label_std": float(y_std),
            "label_raw": float(y) if np.isfinite(y) else float("nan"),
            "not_measured": int(not_measured),
            "molecular_properties": torch.tensor([mw, logp, hba, hbd, tpsa], dtype=torch.float32),
        }


def chembl_collate_fn(batch, n_targets: int):
    B = len(batch)
    T = int(n_targets)

    molecules = [b["molecules"] for b in batch]
    protein_names = [b["protein_name"] for b in batch]
    target_chembl_ids = [b["target_chembl_id"] for b in batch]

    labels_std = torch.tensor([b["label_std"] for b in batch], dtype=torch.float32)  # [B]
    target_indices = torch.tensor([b["target_idx"] for b in batch], dtype=torch.long)
    not_measured_flags = torch.tensor([float(b["not_measured"]) for b in batch], dtype=torch.float32)  # [B]

    labels = torch.tensor(
        [0.0 if (b["label_raw"] != b["label_raw"]) else float(b["label_raw"]) for b in batch],
        dtype=torch.float32
    )

    molecular_properties = torch.stack([b["molecular_properties"] for b in batch], dim=0)

    individual_affinities = torch.zeros((B, T), dtype=torch.float32)
    affinity_mask = torch.zeros((B, T), dtype=torch.float32)

    individual_affinities[torch.arange(B), target_indices] = labels_std
    affinity_mask[torch.arange(B), target_indices] = 1.0 - not_measured_flags  # 1 if measured else 0

    return {
        "molecules": molecules,
        "protein_names": protein_names,
        "target_chembl_ids": target_chembl_ids,
        "target_indices": target_indices,
        "labels": labels,
        "individual_affinities": individual_affinities,
        "affinity_mask": affinity_mask,
        "molecular_properties": molecular_properties,
    }


# ----------------------------
# Loader factory
# ----------------------------
def create_data_loaders_from_chembl_csv(
    csv_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    test_split: float = 0.1,
    val_split: float = 0.1,
    seed: int = 42,
    use_chembl: bool = False,
):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"ChEMBL CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, dtype="string", keep_default_na=False)

    required = ["SMILES", "Protein", "target_chembl_id", "binding_affinity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # strip core columns
    for c in ["SMILES", "Protein", "target_chembl_id"]:
        df[c] = df[c].astype("string").str.strip()

    # robust numeric parsing for affinity
    df["binding_affinity"] = safe_numeric_series(df["binding_affinity"])

    # not_measured
    if "not_measured" not in df.columns:
        df["not_measured"] = "0"
    df["not_measured"] = safe_numeric_series(df["not_measured"]).fillna(0).astype(int)

    # robust numeric parsing for descriptors (so strings '' become NaN early)
    for c in ["Molecular Weight", "LogP", "HBA", "HBD", "TPSA"]:
        if c in df.columns:
            df[c] = safe_numeric_series(df[c])

    # drop rows missing required identifiers
    df = df.dropna(subset=["SMILES", "Protein", "target_chembl_id"]).copy()

    # build target mapping
    unique_targets = sorted(df["target_chembl_id"].astype(str).unique().tolist())
    target_to_idx = {t: i for i, t in enumerate(unique_targets)}
    n_targets = len(unique_targets)

    #train_df, val_df, test_df = group_split(df, test_size=test_split, val_size=val_split, seed=seed)
    train_df, val_df, test_df = random_split(
        df,
        test_size=test_split,
        val_size=val_split,
        seed=seed,
        stratify_col="target_chembl_id",  # set to None if you want fully unstratified randomness
    )
    target_stats = compute_target_stats(train_df, unique_targets)

    # target weights (inverse sqrt frequency), normalized to mean 1.0
    counts = np.array([target_stats[str(t)]["n"] for t in unique_targets], dtype=np.float32)
    counts = np.clip(counts, 1.0, None)
    w = 1.0 / np.sqrt(counts)
    w = w / (w.mean() + 1e-12)

    label_stats = {
        "target_stats": target_stats,
        "targets": unique_targets,
        "target_to_idx": target_to_idx,
        "n_targets": n_targets,
        "target_weights": w.tolist(),
    }

    train_ds = ChEMBLAffinityDataset(train_df, target_to_idx, target_stats, use_chembl=use_chembl)
    val_ds   = ChEMBLAffinityDataset(val_df,   target_to_idx, target_stats, use_chembl=use_chembl)
    test_ds  = ChEMBLAffinityDataset(test_df,  target_to_idx, target_stats, use_chembl=use_chembl)

    collate = lambda b: chembl_collate_fn(b, n_targets=n_targets)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=collate, pin_memory=True)

    print("📊 ChEMBL Data loaders created:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples")
    print(f"  Targets (unique target_chembl_id): {n_targets}")
    print("Target stats:")
    for tid, s in label_stats["target_stats"].items():
        print(tid, "mean", s["mean"], "std", s["std"], "n", s["n"])

    return train_loader, val_loader, test_loader, label_stats


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "chembl" / "chembl_affinity_dataset.csv"
    train_loader, val_loader, test_loader, stats = create_data_loaders_from_chembl_csv(
        csv_path=str(csv_path),
        batch_size=8,
        num_workers=0,
        use_chembl=True,
    )
    batch = next(iter(train_loader))
    print("Batch individual_affinities:", batch["individual_affinities"].shape)
    print("Batch affinity_mask:", batch["affinity_mask"].shape)
    print("Example target idx:", batch["target_indices"][0].item(), "protein:", batch["protein_names"][0])
