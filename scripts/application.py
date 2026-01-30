# application.py

from typing import Literal
from pathlib import Path
import json
import csv
import threading

from train import TrainingManager, request_training_stop, reset_training_stop, get_training_status

EncodingType = Literal["angle", "amplitude"]
ModelType = Literal["gcn", "gine"]

ALLOWED_QUBITS_ANGLE = [4, 6, 8, 10, 12, 14]

# Paths to config files
BASE_CONFIG_PATH = Path("config.json")          # original, full config (with n_targets etc.)
UI_CONFIG_PATH = Path("config_ui_run.json")     # config used for UI runs

# Global training thread handle
TRAINING_THREAD: threading.Thread | None = None


def validate_config(n_qubits: int, encoding_type: str, reupload: bool) -> tuple[int, str, bool]:
    """
    Validate and normalize the quantum configuration according to the rules.
    """
    encoding = encoding_type.lower()

    if encoding not in ("angle", "amplitude"):
        raise ValueError(f"Invalid encoding_type '{encoding_type}'. Must be 'angle' or 'amplitude'.")

    if encoding == "amplitude":
        # amplitude: fixed 4 qubits, no reuploading
        return 4, "amplitude", bool(reupload)

    # encoding == "angle"
    # snap n_qubits to nearest allowed even value in [4, 14]
    if n_qubits not in ALLOWED_QUBITS_ANGLE:
        # fallback: clamp & snap
        if n_qubits < 4:
            n_qubits = 4
        elif n_qubits > 14:
            n_qubits = 14
        if n_qubits % 2 != 0:
            n_qubits += 1

    # handle special case reupload -> 14 qubits
    if reupload:
        n_qubits = 4

    return n_qubits, "angle", bool(reupload)


def validate_training_hparams(
    num_layers: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> tuple[int, int, float, int]:
    """
    Validate basic training hyperparameters.
    """
    if num_layers < 1:
        num_layers = 1
    if epochs < 1:
        epochs = 1
    if learning_rate <= 0:
        learning_rate = 1e-3
    if batch_size <= 0:
        batch_size = 32
    return num_layers, epochs, float(f"{learning_rate:.5f}"), batch_size


def validate_model_type(model_type: str) -> str:
    mt = model_type.lower()
    if mt not in ("gcn", "gine"):
        raise ValueError(f"Invalid model_type '{model_type}'. Must be 'gcn' or 'gine'.")
    return mt


def _merge_ui_config(
    n_qubits: int,
    encoding: str,
    reupload: bool,
    num_layers: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    use_chembl: bool = False,
    csv_path: str | None = None,
) -> str:
    """
    Load the base config.json, override only the UI-controlled fields,
    and write a merged config to UI_CONFIG_PATH.

    This keeps keys like 'n_targets' intact so the model doesn't crash.
    """
    if not BASE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Base config file '{BASE_CONFIG_PATH}' not found. "
            "Make sure it exists and contains the full configuration (including n_targets)."
        )

    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Ensure sections exist
    model_cfg = cfg.setdefault("model", {})
    train_cfg = cfg.setdefault("training", {})
    data_cfg = cfg.setdefault("data", {})


    # Override model-related parameters from UI
    model_cfg["n_qubits"] = n_qubits
    model_cfg["encoding_type"] = encoding
    model_cfg["use_data_reuploading"] = reupload
    model_cfg["num_layers"] = num_layers

    # Override training hyperparams
    train_cfg["epochs"] = epochs
    train_cfg["batch_size"] = batch_size
    train_cfg["learning_rate"] = float(f"{learning_rate:.5f}")
    # keep num_workers from base config if present; default 0 otherwise
    train_cfg.setdefault("num_workers", 0)


    # ---- Data selection (synthetic vs CSV) ----
    data_cfg["use_chembl"] = bool(use_chembl)

    # If user enabled CSV dataset, persist the path and derive targets from the CSV
    if use_chembl:
        if not csv_path:
            # default location used by train.py if nothing else is given
            csv_path = str(Path("data") / "chembl" / "chembl_affinity_dataset.csv")

        data_cfg["csv_path"] = csv_path

        # Derive target list from CSV "Protein" column (fallbacks included)
        raw_targets: set[str] = set()
        try:
            with open(csv_path, "r", encoding="utf-8") as fcsv:
                reader = csv.DictReader(fcsv)
                for row in reader:
                    p = row.get("Protein") or row.get("protein") or row.get("target") or row.get("Target")
                    if p:
                        raw_targets.add(p.strip())
        except FileNotFoundError:
            raise FileNotFoundError(
                f"CSV dataset path not found: {csv_path}. "
                "Upload the CSV from the UI or set data.csv_path in config.json."
            )

        if not raw_targets:
            raise ValueError(
                "Could not derive any proteins from the CSV. "
                "Expected a 'Protein' column (or a similar target column)."
            )

        # ---- Normalize to CHEMBL IDs (Option B robust) ----
        chembl_map: dict[str, str] = cfg.get("chembl_target_map", {})  # {CHEMBLxxx: Display Name}
        # reverse map: {Display Name: CHEMBLxxx}
        reverse_map: dict[str, str] = {v: k for k, v in chembl_map.items()}

        def looks_like_chembl(x: str) -> bool:
            return isinstance(x, str) and x.upper().startswith("CHEMBL")

        normalized_ids: set[str] = set()
        unmapped: set[str] = set()

        for t in raw_targets:
            if looks_like_chembl(t):
                normalized_ids.add(t)
            elif t in reverse_map:
                normalized_ids.add(reverse_map[t])
            else:
                unmapped.add(t)

        if unmapped:
            raise ValueError(
                "CSV contains target names that cannot be mapped to CHEMBL IDs. "
                f"Unmapped: {sorted(unmapped)}. "
                "Add them to config.json -> chembl_target_map (CHEMBL_ID -> Display Name), "
                "or put CHEMBL IDs directly in the CSV Protein column."
            )

        chembl_ids_sorted = sorted(normalized_ids)

        # Build targets dict: keys CHEMBL IDs, values pretty names (fallback to ID)
        cfg["targets"] = {cid: chembl_map.get(cid, cid) for cid in chembl_ids_sorted}
        cfg["targets_order"] = chembl_ids_sorted
        model_cfg["n_targets"] = len(chembl_ids_sorted)


    # Write merged config for the TrainingManager to use
    with UI_CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    return str(UI_CONFIG_PATH)


def _training_worker(config_path: str, model_type: str,train_mode: str) -> None:
    """
    Background worker that instantiates TrainingManager and runs the full training.
    """
    try:
        manager = TrainingManager(config_path=config_path, model_type=model_type)
        manager.train_both_models(variant=train_mode)
    except Exception as e:
        # This message is what you saw: "⚠️ Training error in background worker: 'n_targets'"
        print(f"⚠️ Training error in background worker: {e}")


def run_training(
    n_qubits: int,
    encoding_type: str,
    reupload: bool,
    num_layers: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    model_type: str,
    train_mode: str = "both",
    use_chembl: bool = False,
    csv_path: str | None = None,
) -> str:
    """
    Called by the Streamlit UI.

    - Validates and normalizes the configuration
    - Merges UI settings into the base config.json
    - Starts a background training thread
    """
    global TRAINING_THREAD

    # 1) Validate UI inputs
    n_qubits, encoding, reupload = validate_config(n_qubits, encoding_type, reupload)
    num_layers, epochs, learning_rate, batch_size = validate_training_hparams(
        num_layers, epochs, learning_rate, batch_size
    )
    model_type_norm = validate_model_type(model_type)
    
    train_mode_norm = train_mode.lower()
    if train_mode_norm not in ("both", "quantum", "classical"):
        raise ValueError(f"Invalid train_mode '{train_mode}'. Must be 'both', 'quantum', or 'classical'.")


    # 2) Merge with base config.json so we keep keys like n_targets
    merged_config_path = _merge_ui_config(
        n_qubits=n_qubits,
        encoding=encoding,
        reupload=reupload,
        num_layers=num_layers,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_chembl=use_chembl,
        csv_path=csv_path,
    )

    # 3) Reset stop flag for a fresh run
    reset_training_stop()

    # 4) Start background thread if not already running
    if TRAINING_THREAD is None or not TRAINING_THREAD.is_alive():
        TRAINING_THREAD = threading.Thread(
            target=_training_worker,
            args=(merged_config_path, model_type_norm,train_mode_norm),
            daemon=True,
        )
        TRAINING_THREAD.start()

        return (
            "Training started in background.\n"
            f"Using config: {merged_config_path}\n"
        )
    else:
        return "Training is already running; not starting another run."


def run_testing(model_type: str, train_mode: str) -> str:
    from test import run_all_model_tests

    # Use the merged UI config used for training/testing
    config_path = str(UI_CONFIG_PATH)

    out = run_all_model_tests(config_path=config_path, num_workers=0)

    if not out.get("ok"):
        return f"❌ Testing failed: {out.get('message')}"

    msg = (
        f"✅ Testing finished.\n\n"
        f"Tested: {out['tested']}\n"
        f"Failed: {len(out['failed'])}\n"
        f"Summary: {out['summary_path']}\n"
        f"Results: {out['results_dir']}\n"
        f"Plots: {out['plots_dir']}\n"
    )

    if out["failed"]:
        preview = "\n".join([f"- {x['model']}: {x['error']}" for x in out["failed"][:5]])
        msg += "\n⚠️ Failures (first 5):\n\n" + preview

    return msg


def stop_training() -> str:
    """
    Request the running training loop(s) to stop at the next safe point.
    """
    request_training_stop()
    return "Stop requested. Training will stop after the current batch/epoch."


def is_training_running() -> bool:
    """
    Returns True if the background training thread is alive.
    """
    t = TRAINING_THREAD
    return t is not None and t.is_alive()

def get_ui_training_status() -> dict:
    """
    Wrapper for UI to read training status without importing train.py directly.
    """
    return get_training_status()

