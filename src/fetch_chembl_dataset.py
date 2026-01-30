"""
Fetches bioactivity data from ChEMBL and saves a cached CSV.

Output columns:
ChEMBL ID, Name, SMILES, Molecular Weight, LogP, HBA, HBD, TPSA, Protein, binding_affinity,
target_chembl_id, standard_type, standard_value, standard_units
"""

import os
import random
from pathlib import Path

import pandas as pd

from chembl_webresource_client.new_client import new_client
from rdkit import Chem


# -----------------------------
# Helpers
# -----------------------------
def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol else smiles


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def count_target_activities(
    target_chembl_id: str,
    standard_types=("Ki", "Kd", "IC50", "EC50"),
    max_items=30000,
):
    """
    Approx activity count by reading up to `max_items` (fast enough for selection).
    """
    act_api = new_client.activity
    q = (
        act_api.filter(
            target_chembl_id=target_chembl_id,
            standard_type__in=list(standard_types),
            standard_relation="=",
        )
        .only(["activity_id"])
    )

    n = 0
    it = iter(q)
    while n < max_items:
        try:
            next(it)
            n += 1
        except StopIteration:
            break
        except Exception:
            break
    return n


def pick_high_data_targets(candidate_target_ids, top_k=3):
    tgt_api = new_client.target
    scored = []
    for tid in candidate_target_ids:
        try:
            t = tgt_api.get(tid)
            name = t.get("pref_name", tid)
            n = count_target_activities(tid)
            scored.append((n, tid, name))
            print(f"{tid:>12} | {name[:55]:55} | approx_n={n}")
        except Exception as e:
            print(f"Skipping {tid}: {e}")

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]


def fetch_chembl_dataset_for_targets(
    target_ids,
    n_per_target=3000,
    standard_types=("Ki", "Kd", "IC50", "EC50"),
    require_pchembl=True,
    seed=42,
):
    rng = random.Random(seed)

    act_api = new_client.activity
    mol_api = new_client.molecule
    tgt_api = new_client.target

    rows = []

    for target_chembl_id in target_ids:
        t = tgt_api.get(target_chembl_id)
        target_name = t.get("pref_name", target_chembl_id)

        q = (
            act_api.filter(
                target_chembl_id=target_chembl_id,
                standard_type__in=list(standard_types),
                standard_relation="=",
            )
            .only(
                [
                    "molecule_chembl_id",
                    "pchembl_value",
                    "standard_type",
                    "standard_value",
                    "standard_units",
                ]
            )
        )

        acts = []
        for a in q:
            if require_pchembl and a.get("pchembl_value") is None:
                continue
            if a.get("molecule_chembl_id") is None:
                continue
            acts.append(a)
            if len(acts) >= n_per_target:
                break

        rng.shuffle(acts)

        fetched = 0
        for a in acts:
            mid = a["molecule_chembl_id"]
            try:
                m = mol_api.get(mid)
            except Exception:
                continue

            smi = None
            ms = m.get("molecule_structures") or {}
            smi = ms.get("canonical_smiles")

            if not smi:
                continue

            props = m.get("molecule_properties") or {}

            row = {
                "ChEMBL ID": mid,
                "Name": m.get("pref_name", None),
                "SMILES": smi,
                "canonical_smiles": canonicalize_smiles(smi),
                "Molecular Weight": safe_float(props.get("full_mwt")),
                "LogP": safe_float(props.get("alogp")),
                "HBA": safe_float(props.get("hba")),
                "HBD": safe_float(props.get("hbd")),
                "TPSA": safe_float(props.get("psa")),
                "Protein": target_name,
                "target_chembl_id": target_chembl_id,
                "binding_affinity": safe_float(a.get("pchembl_value")),  # pChEMBL (recommended)
                "standard_type": a.get("standard_type"),
                "standard_value": safe_float(a.get("standard_value")),
                "standard_units": a.get("standard_units"),
            }

            rows.append(row)
            fetched += 1

        print(f"✓ {target_name} ({target_chembl_id}) fetched: {fetched} rows")

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["binding_affinity", "canonical_smiles", "Protein"])
    df = df.drop_duplicates(subset=["canonical_smiles", "target_chembl_id", "binding_affinity"])
    return df


# -----------------------------
# Main script
# -----------------------------
def build_and_save_dataset(
    out_csv_path: str,
    candidate_targets,
    top_k_targets=3,
    n_per_target=3000,
    seed=42,
):
    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n🔎 Selecting high-data targets from candidate list...")
    top = pick_high_data_targets(candidate_targets, top_k=top_k_targets)

    if len(top) == 0:
        raise RuntimeError("No targets were selected. Expand candidate list or check connectivity.")

    target_ids = [tid for _, tid, _ in top]
    print("\nSelected targets:")
    for n, tid, name in top:
        print(f" - {name} ({tid}) ~{n} activities (approx)")

    print("\n📥 Fetching ChEMBL dataset...")
    df = fetch_chembl_dataset_for_targets(
        target_ids=target_ids,
        n_per_target=n_per_target,
        seed=seed,
    )

    print(f"\n💾 Saving CSV to: {out_csv_path}")
    df.to_csv(out_csv_path, index=False)
    print(f"✅ Done. Rows={len(df):,}, Targets={df['target_chembl_id'].nunique()}")

    return out_csv_path


if __name__ == "__main__":
    # Adjust this pool whenever you want.
    # The script selects the top_k by approximate activity count.
    CANDIDATES = [
        # Kinases
        "CHEMBL203",   # EGFR
        "CHEMBL279",   # ABL1
        # GPCRs
        "CHEMBL216",   # DRD2
        "CHEMBL248",   # ADRB2
        # Nuclear receptors
        "CHEMBL206",   # ESR1
        # Enzymes
        "CHEMBL204",   # ACHE
        "CHEMBL1824",  # HMGCR
    ]

    project_root = Path(__file__).resolve().parents[1]
    out_csv = project_root / "data" / "chembl" / "chembl_affinity_dataset.csv"

    build_and_save_dataset(
        out_csv_path=str(out_csv),
        candidate_targets=CANDIDATES,
        top_k_targets=3,
        n_per_target=3000,
        seed=42,
    )
