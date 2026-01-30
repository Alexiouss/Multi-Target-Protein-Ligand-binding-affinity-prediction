"""
Training Script for Quantum Multi-Target Drug Discovery
Trains quantum and classical variants with comprehensive comparison
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import time
from pathlib import Path
import warnings
import importlib
import threading
from tqdm import tqdm
torch.autograd.set_detect_anomaly(False)
import math
import hashlib
from sklearn.decomposition import PCA


warnings.filterwarnings('ignore')

# -----------------------------
# Paths / imports
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
src_dir = os.path.join(project_dir, 'src')

# Global flag to request training stop
STOP_TRAINING = threading.Event()

TRAINING_STATUS = {
    "model_name": None,
    "phase": "idle",      # "idle", "train", "val", "done"
    "epoch": 0,
    "total_epochs": 0,
    "batch": 0,
    "total_batches": 0,
    "steps_done": 0,
    "start_time": None,
    "iters_per_sec": 0.0,
}


def _update_training_status(
    model_name: str,
    phase: str,
    epoch: int,
    total_epochs: int,
    batch: int,
    total_batches: int,
    is_step: bool = False,
):
    TRAINING_STATUS["model_name"] = model_name
    TRAINING_STATUS["phase"] = phase
    TRAINING_STATUS["epoch"] = epoch
    TRAINING_STATUS["total_epochs"] = total_epochs
    TRAINING_STATUS["batch"] = batch
    TRAINING_STATUS["total_batches"] = total_batches

    if is_step:
        now = time.time()
        if TRAINING_STATUS["start_time"] is None:
            TRAINING_STATUS["start_time"] = now
            TRAINING_STATUS["steps_done"] = 0
            TRAINING_STATUS["iters_per_sec"] = 0.0
        else:
            TRAINING_STATUS["steps_done"] += 1
            elapsed = now - TRAINING_STATUS["start_time"]
            if elapsed > 0 and TRAINING_STATUS["steps_done"] > 0:
                TRAINING_STATUS["iters_per_sec"] = TRAINING_STATUS["steps_done"] / elapsed



def get_training_status():
    """
    Return a shallow copy of current training status for UI.
    """
    return dict(TRAINING_STATUS)


if src_dir not in sys.path:
    sys.path.append(src_dir)

# NEW: import the quantum attention layer so we can find it inside the model
try:
    from quantum_attention_refactored import QuantumAttentionLayer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure quantum_attention_refactored.py is in src/ directory")
    QuantumAttentionLayer = None  # graceful fallback

# Data loader is shared for all models
try:
    from data_loader import create_data_loaders
    from data_loader_chembl import create_data_loaders_from_chembl_csv
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure data_loader.py is in src/ directory")
    sys.exit(1)

# Device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_model_module(model_type: str):
    """
    Dynamically import the correct multi-target model module depending on
    the selected backbone ("gcn" or "gine").

    Assumes:
      - src/multi_target_model_multiple_models_gcn.py
      - src/multi_target_model_multiple_models_gine.py

    Each must define:
      - MultiTargetPredictor
      - MultiTargetLoss
      - calculate_metrics
    """
    mt = model_type.lower()
    if mt == "gcn":
        module_name = "multi_target_model_gcn_refactored"
    elif mt == "gine":
        module_name = "multi_target_model_multiple_models_gine"
    else:
        raise ValueError(f"Unknown model_type '{model_type}', expected 'gcn' or 'gine'.")

    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Could not import module '{module_name}'. "
            f"Make sure the file exists in src/ and is named '{module_name}.py'."
        ) from e

    # Basic API check
    required_attrs = ["MultiTargetPredictor", "MultiTargetLoss", "calculate_metrics"]
    for attr in required_attrs:
        if not hasattr(module, attr):
            raise AttributeError(f"Module '{module_name}' is missing required attribute '{attr}'.")

    return module


class TrainingManager:
    """
    Manages the training of quantum and classical models
    """

    def __init__(self, config_path: str = "config.json", model_type: str = "gcn"):
        # Path to config
        self.config_path = config_path
        self.model_type = model_type.lower()

        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # Training config (epochs/batch_size/lr can be overridden by UI via config_ui_run.json)
        self.training_config = self.config.get('training', {
            'epochs': 20,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'num_workers': 0
        })
        self.model_config = self.config.get("model", {})


        
        self.stop_event = STOP_TRAINING
        # Dynamic import of model module (GCN vs GINE)
        model_module = _load_model_module(self.model_type)
        self.ModelClass = model_module.MultiTargetPredictor
        self.LossClass = model_module.MultiTargetLoss
        self.calculate_metrics_fn = model_module.calculate_metrics

        # Create results directories
        self.results_dir = Path(os.path.join(project_dir, "results"))
        self.models_dir = self.results_dir / "models"
        self.plots_dir = self.results_dir / "plots" / "training_curves"
        self.metrics_dir = self.results_dir / "metrics"

        for dir_path in [self.models_dir, self.plots_dir, self.metrics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Training history
        self.training_history = {
            'quantum': {
                'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': [], 'train_individual_losses': [],'val_individual_losses': []
            },
            'classical': {
                'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': [], 'train_individual_losses': [],'val_individual_losses': []
            }
        }

        # NEW: gradient norms per training step
        self.grad_history = {
            'quantum': {
                'global_norm': [],
                'max_abs': []
            },
            'classical': {
                'global_norm': [],
                'max_abs': []
            }
        }
        #Log attention
        self.attention_summary = {
            "quantum": [], # list of dicts (one per epoch when quantum is trained)
            "classical":[]
        }

        print(f"✓ Training Manager initialized")
        print(f"✓ Using backbone model_type = {self.model_type.upper()}")
        print(f"✓ Config path: {self.config_path}")
        print(f"✓ Results will be saved to: {self.results_dir}")
        print(f"✓ Device: {device}")

    def _ensure_pca_for_current_run(
        self,
        train_loader,
        max_batches: int = 50,
        random_state: int = 42,
    ):
        """
        Automatically (re)computes PCA if and only if the current UI config
        does not match the PCA stored on disk.
        """

        model_cfg = self.config["model"]
        training_cfg = self.config.get("training", {})
        data_cfg = self.config.get("data", {})

        encoding_type = model_cfg["encoding_type"]
        n_qubits = int(model_cfg.get("n_qubits", 0))
        target_dim = int(model_cfg["feature_dim"])
        n_targets = int(model_cfg["n_targets"])

        input_dim = 2 * target_dim  # [query || key]

        # ---- Decide PCA shape and filename ----
        if encoding_type == "amplitude":
            out_dim = 16
            default_path = f"pca_amplitude_{self.model_type}.npz"
            pca_path = model_cfg.get("pca_path", default_path)
            # also write it back to config so the quantum module loads the same file
            model_cfg["pca_path"] = pca_path

        elif encoding_type == "angle":
            out_dim = 2 * n_qubits
            pca_path = f"pca_angle_{n_qubits}_{self.model_type}.npz"
            # store it in config so QuantumMultiTargetAttention can load it
            model_cfg["angle_pca_path"] = pca_path

        else:
            print(f"⚠️ PCA skipped (encoding_type='{encoding_type}')")
            return
        
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        # ---- Build a strict signature ----
        signature = {
            "version": 1,
            "encoding_type": encoding_type,
            "n_qubits": n_qubits,
            "target_dim": target_dim,
            "input_dim": input_dim,
            "out_dim": out_dim,
            "n_targets": n_targets,
            "synthetic_samples": int(data_cfg.get("synthetic_samples", -1)),
            "use_chembl": bool(data_cfg.get("use_chembl", False)),
            "max_batches": max_batches,
            "random_state": random_state,
            "backbone": self.model_type,  # ✅ GCN vs GINE
        }

        sig_hash = hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode()
        ).hexdigest()

        # ---- Check existing PCA ----
        if os.path.exists(pca_path):
            try:
                old = np.load(pca_path, allow_pickle=True)
                meta = json.loads(old["meta"].item())
                if meta.get("sig_hash") == sig_hash:
                    print(f"✓ PCA OK ({encoding_type}) → {pca_path}")
                    return
            except Exception:
                pass

        print(f"🔁 Computing PCA ({encoding_type}) → {pca_path}")
        print(f"   input_dim={input_dim}, out_dim={out_dim}, backbone={self.model_type}")

        # ---- Build TEMP quantum model (correct backbone) ----
        ModelClass = self.ModelClass  # already resolved GCN/GINE earlier
        temp_model = ModelClass(
            config_path=self.config_path,
            model_type="quantum",
        ).to(device)
        temp_model.eval()

        qa = temp_model.attention.quantum_attention
        target_embeddings = qa.target_embeddings

        X_chunks = []

        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= max_batches:
                    break

                smiles = batch["molecules"]
                mol_feats, _ = temp_model.graph_encoder(smiles)
                mol_feats = mol_feats.to(device)

                queries = qa.query_projection(mol_feats)
                queries = qa.query_norm(queries)  # [B, D]

                keys = []
                for t in range(n_targets):
                    tid = torch.tensor([t], device=device)
                    emb = target_embeddings(tid).squeeze(0)
                    key = qa.key_projection(emb.unsqueeze(0)).squeeze(0)
                    key = qa.key_norm(key)
                    keys.append(key)

                keys = torch.stack(keys, dim=0)  # [T, D]

                B, D = queries.shape
                q_exp = queries.unsqueeze(1).expand(-1, n_targets, -1)
                k_exp = keys.unsqueeze(0).expand(B, -1, -1)

                combined = torch.cat([q_exp, k_exp], dim=-1)  # [B,T,2D]
                combined = combined.reshape(-1, combined.shape[-1])

                X_chunks.append(combined.cpu().numpy())

        X = np.concatenate(X_chunks, axis=0)

        if X.shape[1] != input_dim:
            raise RuntimeError("PCA input dimension mismatch")

        pca = PCA(n_components=out_dim, random_state=random_state)
        pca.fit(X)

        meta = dict(signature)
        meta["sig_hash"] = sig_hash
        meta["explained_var_sum"] = float(pca.explained_variance_ratio_.sum())

        np.savez(
            pca_path,
            components=pca.components_,
            mean=pca.mean_,
            meta=np.array(json.dumps(meta), dtype=object),
        )

        print(f"💾 PCA saved → {pca_path} | var={meta['explained_var_sum']:.4f}")


    def _compute_grad_stats(self, model):
        """
        Compute global L2 norm and max absolute value of all gradients
        for the given model. Returns (global_norm, max_abs).
        """
        total_sq = 0.0
        max_abs = 0.0

        for p in model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            total_sq += float(g.pow(2).sum().item())
            max_abs = max(max_abs, float(g.abs().max().item()))

        global_norm = math.sqrt(total_sq) if total_sq > 0 else 0.0
        return global_norm, max_abs
    
    def _set_backbone_trainable(self, model, trainable: bool):
        """
        Freeze/unfreeze the graph encoder (GCN/GINE backbone).
        Minimal, safe, and reversible.
        """
        if not hasattr(model, "graph_encoder"):
            print("⚠️ model has no attribute 'graph_encoder' — cannot freeze backbone.")
            return

        for p in model.graph_encoder.parameters():
            p.requires_grad = trainable

        state = "trainable" if trainable else "FROZEN"
        print(f"🔧 Backbone (graph_encoder) is now {state}.")

    def _should_freeze_backbone(self, model, model_name: str) -> bool:
        if model_name.lower() != "quantum":
            return False
        enc = self._get_encoding_type(model)
        return (enc == "amplitude")




    def create_models(self):
        """
        Creates quantum and classical models
        """
        print("\n🔬 Creating Models...")

        # Quantum model (uses quantum attention inside MultiTargetPredictor)
        self.quantum_model = self.ModelClass(
            config_path=self.config_path,
            model_type="quantum",  
        )

        # Classical model (classical attention)
        self.classical_model = self.ModelClass(
            config_path=self.config_path,
            model_type="classical",
        )

        # Move to device
        self.quantum_model.to(device)
        self.classical_model.to(device)

        # Parameter counts
        q_params = sum(p.numel() for p in self.quantum_model.parameters() if p.requires_grad)
        c_params = sum(p.numel() for p in self.classical_model.parameters() if p.requires_grad)

        diff = ((q_params - c_params) / c_params * 100) if c_params > 0 else float('inf')
        print(f"✓ Quantum model parameters: {q_params:,}")
        print(f"✓ Classical model parameters: {c_params:,}")
        print(f"✓ Parameter difference: {diff:+.1f}%")

        
        return self.quantum_model, self.classical_model
    

    def flatten_measured(self,pred, target, mask):
        # pred/target/mask: [N,T]
        m = mask.bool()
        return pred[m].detach().cpu(), target[m].detach().cpu()
    
    def _get_quantum_signature(self, model):
        """
        Extract quantum-specific configuration for naming.
        Returns None if model is not quantum.
        """
        try:
            qc = model.attention.quantum_attention.quantum_circuit
            return {
                "encoding": qc.encoding_type,
                "n_qubits": qc.n_qubits,
                "reupload": int(qc.use_data_reuploading),
            }
        except Exception:
            return None

    
    
    def _get_encoding_type(self, model):
        # Try top-level (if you set it somewhere)
        enc = getattr(model, "encoding_type", None)
        if enc is not None:
            return enc

        # Try common nested locations
        try:
            return model.attention.quantum_attention.quantum_circuit.encoding_type
        except Exception:
            pass

        return None



    def create_optimizers(self):
        """
        Creates optimizers and learning rate schedulers
        """
        lr = float(self.training_config.get('learning_rate', 1e-3))

        # Optimizers
        self.quantum_optimizer = optim.Adam(
            self.quantum_model.parameters(),
            lr=lr,
            weight_decay=1e-5
        )

        self.classical_optimizer = optim.Adam(
            self.classical_model.parameters(),
            lr=lr,
            weight_decay=1e-5
        )

        # Learning rate schedulers
        self.quantum_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.quantum_optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        self.classical_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.classical_optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Loss function (comes from selected model module)
        self.loss_fn = self.LossClass(
            target_weights=getattr(self, "target_weights", None),
            loss_type="mse"
        )


        print(f"✓ Optimizers created with learning rate: {lr}")

    def train_epoch(self, model, optimizer, train_loader, model_name):
        model.train()

        # epoch accumulators
        epoch_loss_sum = 0.0   # accumulates (loss * batch_measured_count)
        epoch_cnt = 0.0        # total measured count across epoch (for normalization)


        per_target_sse = None  # shape [T]
        per_target_cnt = None  # shape [T]

        all_predictions = []
        all_targets = []
        all_masks = []

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"{model_name} Train", leave=False)):
            if self.stop_event.is_set():
                print(f"🛑 Stop requested, breaking {model_name} training loop.")
                break

            optimizer.zero_grad(set_to_none=True)

            predictions, attention_weights, _ = model(batch["molecules"])  # [B,T] expected

            targets = batch["individual_affinities"].to(device)  # [B,T]
            mask = batch.get("affinity_mask", None)
            if mask is None:
                mask = torch.ones_like(targets, device=device)
            else:
                mask = mask.to(device)

            # ---------- measured-only SSE and counts ----------
            diff2 = (predictions - targets) ** 2
            diff2_m = diff2 * mask

            # total
            sse = diff2_m.sum()
            cnt = mask.sum().detach().clamp_min(1e-12)  # measured count in this batch


            loss, individual_losses = self.loss_fn(
                predictions, targets, attention_weights, mask=mask
            )
            # accumulate loss in a way consistent with measured-count normalization
            epoch_loss_sum += float(loss.detach().item()) * float(cnt.item())
            epoch_cnt += float(cnt.item())


            if not torch.isfinite(loss):
                print(f"[LOSS_NAN] {model_name} loss is non-finite at batch {batch_idx}: {loss.item()}")
                raise RuntimeError(f"{model_name}: non-finite loss encountered.")
            
            if batch_idx == 0:
                with torch.no_grad():
                    print("  preds stats:",
                        "mean=", float(predictions[mask.bool()].mean().item()) if mask.sum() > 0 else None,
                        "std=",  float(predictions[mask.bool()].std().item())  if mask.sum() > 1 else None,
                        "min=",  float(predictions[mask.bool()].min().item())  if mask.sum() > 0 else None,
                        "max=",  float(predictions[mask.bool()].max().item())  if mask.sum() > 0 else None)
                    
            
            def grad_diagnostics(model):
                import math
                total_sq = 0.0
                max_abs = 0.0
                n = 0
                n_nonfinite = 0

                for p in model.parameters():
                    if p.grad is None:
                        continue
                    g = p.grad.detach()
                    if not torch.isfinite(g).all():
                        n_nonfinite += 1
                        continue
                    total_sq += float(g.norm(2).item() ** 2)
                    max_abs = max(max_abs, float(g.abs().max().item()))
                    n += g.numel()

                total_norm = math.sqrt(total_sq) if total_sq > 0 else 0.0
                return total_norm, max_abs, n_nonfinite, n


            loss.backward()

            # optional grad stats logging
            global_norm, max_abs = self._compute_grad_stats(model)
            key = model_name.lower()
            if key in self.grad_history:
                self.grad_history[key]["global_norm"].append(global_norm)
                self.grad_history[key]["max_abs"].append(max_abs)

            # clip
            max_norm = 1.0
            if model_name.lower() == "quantum":
                encoding = self._get_encoding_type(model)
                if encoding == "amplitude":
                    max_norm = 0.1
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            optimizer.step()

            if batch_idx == 0:
                # Check whether *any* parameter actually changed
                # (uses the first trainable parameter as a probe)
                for p in model.parameters():
                    if p.requires_grad:
                        if not hasattr(self, "_probe_param_prev"):
                            self._probe_param_prev = p.detach().clone()
                            print("  probe param init:", float(self._probe_param_prev.view(-1)[0].item()))
                        else:
                            delta = (p.detach() - self._probe_param_prev).abs().mean().item()
                            print("  probe param mean |Δ| after 1 step:", float(delta))
                            self._probe_param_prev = p.detach().clone()
                        break



            # per-target
            sse_t = diff2_m.sum(dim=0).detach()          # [T]
            cnt_t = mask.sum(dim=0).detach()             # [T]

            if per_target_sse is None:
                per_target_sse = sse_t.clone()
                per_target_cnt = cnt_t.clone()
            else:
                per_target_sse += sse_t
                per_target_cnt += cnt_t

            all_predictions.append(predictions.detach())
            all_targets.append(targets.detach())
            all_masks.append(mask.detach())

            if batch_idx == 0:
                print("DEBUG train batch0:")
                print("  targets shape:", targets.shape)
                print("  mask shape:", mask.shape)
                print("  mask sum:", float(mask.sum().item()))
                print("  mask mean:", float(mask.float().mean().item()))
                per_t = mask.sum(dim=0).detach().cpu().numpy().tolist()
                print("  measured per target:", per_t)
                measured_vals = targets[mask.bool()]
                print("  measured targets: n=", measured_vals.numel(),
                    "mean=", float(measured_vals.mean().item()) if measured_vals.numel() else None,
                    "std=", float(measured_vals.std().item()) if measured_vals.numel() > 1 else None)

            _update_training_status(
                model_name=model_name,
                phase="train",
                epoch=TRAINING_STATUS["epoch"],
                total_epochs=TRAINING_STATUS["total_epochs"],
                batch=batch_idx + 1,
                total_batches=len(train_loader),
                is_step=True,
            )

        if len(all_predictions) == 0:
            return float("nan"), {"overall": {"pearson": 0.0, "rmse": float("nan"), "mae": float("nan"), "concordance_index": 0.5}}, {}

        # ---------- epoch outputs ----------
        avg_loss = epoch_loss_sum / max(epoch_cnt, 1.0)


        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        # ✅ total + per-target pearson/CI computed using the same mask
        metrics = self.calculate_metrics_fn(all_predictions, all_targets, model.target_names, mask=all_masks)

        # ✅ per-target measured-only MSE (NOT averaged per-batch; true epoch aggregate)
        per_target_mse = (per_target_sse / per_target_cnt.clamp_min(1.0)).cpu().numpy().tolist()
        per_target_dict = {t: float(v) for t, v in zip(model.target_names, per_target_mse)}

        return avg_loss, metrics, per_target_dict




    def validate_epoch(self, model, val_loader, model_name):
        model.eval()

        epoch_loss_sum = 0.0
        epoch_cnt = 0.0


        per_target_sse = None
        per_target_cnt = None

        all_predictions = []
        all_targets = []
        all_masks = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"{model_name} Val", leave=False)):
                if self.stop_event.is_set():
                    print(f"🛑 Stop requested, breaking {model_name} validation loop.")
                    break

                predictions, attention_weights, _ = model(batch["molecules"])
                targets = batch["individual_affinities"].to(device)




                mask = batch.get("affinity_mask", None)
                if mask is None:
                    mask = torch.ones_like(targets, device=device)
                else:
                    mask = mask.to(device)

                if batch_idx == 0:
                    print("DEBUG val batch0 preds stats:",
                        "mean=", float(predictions[mask.bool()].mean().item()) if mask.sum() > 0 else None,
                        "std=",  float(predictions[mask.bool()].std().item())  if mask.sum() > 1 else None)

                diff2 = (predictions - targets) ** 2
                diff2_m = diff2 * mask

                sse = diff2_m.sum()
                cnt = mask.sum().detach().clamp_min(1e-12)

                loss, _ = self.loss_fn(predictions, targets, attention_weights=None, mask=mask)
                epoch_loss_sum += float(loss.item()) * float(cnt.item())
                epoch_cnt += float(cnt.item())

                sse_t = diff2_m.sum(dim=0).detach()
                cnt_t = mask.sum(dim=0).detach()

                if per_target_sse is None:
                    per_target_sse = sse_t.clone()
                    per_target_cnt = cnt_t.clone()
                else:
                    per_target_sse += sse_t
                    per_target_cnt += cnt_t

                all_predictions.append(predictions.detach())
                all_targets.append(targets.detach())
                all_masks.append(mask.detach())

                _update_training_status(
                    model_name=model_name,
                    phase="val",
                    epoch=TRAINING_STATUS["epoch"],
                    total_epochs=TRAINING_STATUS["total_epochs"],
                    batch=batch_idx + 1,
                    total_batches=len(val_loader),
                    is_step=True,
                )

        if len(all_predictions) == 0:
            return float("nan"), {"overall": {"pearson": 0.0, "rmse": float("nan"), "mae": float("nan"), "concordance_index": 0.5}}, {}

        avg_loss = epoch_loss_sum / max(epoch_cnt, 1.0)

        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        # ✅ total + per-target pearson/CI with mask
        metrics = self.calculate_metrics_fn(all_predictions, all_targets, model.target_names, mask=all_masks)

        per_target_mse = (per_target_sse / per_target_cnt.clamp_min(1.0)).cpu().numpy().tolist()
        per_target_dict = {t: float(v) for t, v in zip(model.target_names, per_target_mse)}

        return avg_loss, metrics, per_target_dict




    def train_model(self, model, optimizer, scheduler, train_loader, val_loader, model_name, epochs):
        """
        Complete training for one model
        """
        print(f"\n🚀 Training {model_name} Model...")

        best_val_loss = float('inf')
        best_metrics = None
        patience_counter = 0
        max_patience = 10

        # reset timing for this model
        TRAINING_STATUS["start_time"] = None
        TRAINING_STATUS["steps_done"] = 0
        TRAINING_STATUS["iters_per_sec"] = 0.0

        # model_name is "Quantum" or "Classical"
        train_mode = model_name.lower()

        bs = int(self.training_config.get("batch_size", 32))
        lr = float(self.training_config.get("learning_rate", 1e-3))
        epochs = int(self.training_config.get("epochs", 20))
        num_layers = int(self.model_config.get("num_layers", 3))

        # Format LR nicely
        if lr < 1e-3:
            lr_str = f"{lr:.0e}"
        else:
            lr_str = f"{lr}".replace(".", "p")

        timestamp = time.strftime("%Y%m%d-%H%M%S")

        quantum_sig = None
        if model_name.lower() == "quantum":
            quantum_sig = self._get_quantum_signature(model)

        if quantum_sig is not None:
            enc = quantum_sig["encoding"]
            nq = quantum_sig["n_qubits"]
            reup = quantum_sig["reupload"]

            self.run_name = (
                f"{train_mode}_{self.model_type}_"
                f"enc-{enc}_q{nq}_reup{reup}_"
                f"layers{num_layers}_"
                f"bs{bs}_lr{lr_str}_ep{epochs}_{timestamp}"
            )
        else:
            # classical or fallback
            self.run_name = (
                f"{train_mode}_{self.model_type}_"
                f"layers{num_layers}_"
                f"bs{bs}_lr{lr_str}_ep{epochs}_{timestamp}"
            )


        print(f"✓ Run name: {self.run_name}")

        # -----------------------------
        # Amplitude-only warmup freezing
        # -----------------------------
        WARMUP_EPOCHS = 2  # 1–3
        enc = self._get_encoding_type(model)
        print(f"DEBUG encoding_type resolved as: {enc}")

        freeze_backbone = (model_name.lower() == "quantum" and enc == "amplitude")
        if freeze_backbone:
            print(f"🔒 Freezing backbone (graph_encoder) for first {WARMUP_EPOCHS} epochs (amplitude only).")
            self._set_backbone_trainable(model, trainable=False)
        else:
            print("DEBUG: backbone freezing is OFF.")




        for epoch in range(int(epochs)):
            if self.stop_event.is_set():
                print(f"🛑 Stop requested, aborting {model_name} training epoch early.")
                break

            #Set epoch-level status (batch info will be filled in train/val loops)
            _update_training_status(
                model_name=model_name,
                phase="train",
                epoch=epoch + 1,
                total_epochs=int(epochs),
                batch=0,
                total_batches=len(train_loader),
            )

            start_time = time.time()

            print(f"\n--- Epoch {epoch + 1}/{epochs} ({model_name}) ---")
            # Unfreeze exactly when warmup is finished
            if freeze_backbone and epoch == WARMUP_EPOCHS:
                print("🔓 Warm-up finished → unfreezing backbone.")
                self._set_backbone_trainable(model, trainable=True)

            # Training
            train_loss, train_metrics,train_ind_losses = self.train_epoch(model, optimizer, train_loader, model_name)

            if self.stop_event.is_set():
                print(f"🛑 Stop requested after {model_name} training; skipping validation.")
                break
            
            # Validation
            val_loss, val_metrics,val_ind_losses = self.validate_epoch(model, val_loader, model_name)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Save metrics
            key = model_name.lower()
            self.training_history[key]['train_loss'].append(train_loss)
            self.training_history[key]['val_loss'].append(val_loss)
            self.training_history[key]['train_metrics'].append(train_metrics)
            self.training_history[key]['val_metrics'].append(val_metrics)
            self.training_history[key]['train_individual_losses'].append(train_ind_losses)
            self.training_history[key]['val_individual_losses'].append(val_ind_losses)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                patience_counter = 0

                best_name = f"{self.run_name}_best.pt"
                best_path = self.models_dir / best_name

                checkpoint = {
                    "quantum_signature": quantum_sig,
                    "model_state_dict": model.state_dict(),
                    "model_type_variant": train_mode,       # "quantum" or "classical"
                    "backbone": self.model_type,            # "gcn" or "gine"
                    "run_name": self.run_name,
                    "timestamp": timestamp,
                    "config_path": self.config_path,        # should be config_ui_run.json in UI runs
                    "model_config": dict(self.model_config),
                    "training_config": dict(self.training_config),
                    "dataset_meta": dict(getattr(self, "dataset_meta", {})),
                    "best_val_loss": float(best_val_loss),
                    "best_val_metrics": best_metrics,
                }

                torch.save(checkpoint, best_path)
                print(f"    ✓ New best model saved to: {best_path}")

            else:
                patience_counter += 1

            epoch_time = time.time() - start_time

            # Progress report
            print(f"    Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"    Train CI: {train_metrics['overall']['concordance_index']:.4f}, "
                  f"Val CI: {val_metrics['overall']['concordance_index']:.4f}")
            print(f"    Train Pearson: {train_metrics['overall']['pearson']:.4f}, "
                  f"Val Pearson: {val_metrics['overall']['pearson']:.4f}")
            print(f"    Epoch Time: {epoch_time:.2f}s")
            targets = model.target_names
            train_loss_str = ", ".join(
                f"{t}: {train_ind_losses[t]:.4f}" for t in targets
            )
            val_loss_str = ", ".join(
                f"{t}: {val_ind_losses[t]:.4f}" for t in targets
            )

            print(f"    Train per-target losses: {train_loss_str}")
            print(f"    Val   per-target losses: {val_loss_str}")

            # Early stopping
            if patience_counter >= max_patience:
                print(f"    Early stopping triggered after {epoch + 1} epochs")
                break

        print(f"\n✅ {model_name} training completed!")
        if best_metrics is not None:
            print(f"✓ Best validation loss: {best_val_loss:.4f}")
            print(f"✓ Best validation CI: {best_metrics['overall']['concordance_index']:.4f}")

        # Safety: ensure backbone is unfrozen at the end
        if freeze_backbone:
            self._set_backbone_trainable(model, trainable=True)

        return best_metrics

    def train_both_models(self, variant: str = "both"):
        """
        Trains the model (quantum &/or classical) according to the variant.

        variant ∈ {"both", "quantum", "classical"}
        """


        # Create data loaders
        print("📊 Creating data loaders...")
        bs = int(self.training_config.get("batch_size", 32))
        nw = int(self.training_config.get("num_workers", 0))
        print(f"   → Using batch_size={bs}, num_workers={nw}")

        config = self.config

        data_cfg = config["data"]
        use_chembl = data_cfg.get("use_chembl", False)

        if use_chembl:
            print("🌐 Using ChEMBL-based dataset")
            project_root = Path(__file__).resolve().parents[1]

            # Prefer path from config (set by UI), fallback to default repo location
            csv_path_cfg = data_cfg.get("csv_path")
            if csv_path_cfg:
                csv_path = Path(csv_path_cfg)
                if not csv_path.is_absolute():
                    csv_path = project_root / csv_path
            else:
                csv_path = project_root / "data" / "chembl" / "chembl_affinity_dataset.csv"

            train_loader, val_loader, _, label_stats = create_data_loaders_from_chembl_csv(
                csv_path=str(csv_path),
                batch_size=config["training"]["batch_size"],
                num_workers=config["training"]["num_workers"],
                use_chembl=True  # Pass use_chembl flag
            )
            chembl_targets = list(label_stats["targets"])
            self.config["targets"] = {t: t for t in chembl_targets}
            self.config["model"]["n_targets"] = len(chembl_targets)
            self.quantum_model = self.ModelClass(config_path=self.config_path, model_type="quantum", target_names_override=chembl_targets)
            self.classical_model = self.ModelClass(config_path=self.config_path, model_type="classical", target_names_override=chembl_targets)
            self.target_weights = label_stats.get("target_weights", None)
            # ✅ store dataset metadata for checkpoint saving
            self.dataset_meta = {
                "use_chembl": True,
                "csv_path": str(csv_path) if use_chembl else None,
                # This must match the model's output order
                "target_names": chembl_targets,
                "n_targets": len(chembl_targets),
            }
        else:
            print("🧪 Using synthetic dataset")
            train_loader, val_loader, test_loader = create_data_loaders(
                batch_size=config["training"]["batch_size"],
                num_workers=config["training"]["num_workers"]
            )
            self.target_weights = None
            label_stats = None  # No need for stats with synthetic dataset
            full_dataset = None  # synthetic loader can be adapted to also return full_dataset if you want plots for it

            # ✅ store dataset metadata for checkpoint saving
            self.dataset_meta = {
                "use_chembl": bool(use_chembl),
                "csv_path": str(csv_path) if use_chembl else None,
                # This must match the model's output order
                "target_names": list(self.config.get("targets", {}).keys()),
                "n_targets": int(self.config["model"]["n_targets"]),
            }
            
        # NEW: ensure PCA files match the current config_ui_run.json BEFORE model creation
        self._ensure_pca_for_current_run(train_loader)
        # Create models and optimizers
        self.create_models()
        self.create_optimizers()
        self.quantum_model.graph_encoder.gcn.debug_weights()
        self.classical_model.graph_encoder.gcn.debug_weights()


        try:
            sample_batch = next(iter(train_loader))
        except StopIteration:
            raise RuntimeError("Train loader is empty – cannot run sanity check.")

        self.quantum_model.eval()
        with torch.no_grad():
            preds, attn, _ = self.quantum_model(sample_batch["molecules"])
            if not torch.isfinite(preds).all():
                raise RuntimeError(
                    "Sanity check failed: Quantum model predictions contain NaNs/Infs "
                    "on the very first forward pass."
                )

        # Use epochs from training_config
        epochs = int(self.training_config.get('epochs', 20))

        quantum_metrics = None
        classical_metrics = None

        try:

            # --- Quantum ---
            if variant in ("both", "quantum"):
                quantum_metrics = self.train_model(
                    self.quantum_model,
                    self.quantum_optimizer,
                    self.quantum_scheduler,
                    train_loader,
                    val_loader,
                    "Quantum",
                    epochs
                )
            if variant in ("both", "classical"):
                classical_metrics = self.train_model(
                    self.classical_model,
                    self.classical_optimizer,
                    self.classical_scheduler,
                    train_loader,
                    val_loader,
                    "Classical",
                    epochs
                )
        except Exception as e:
            print(f"Training interrupted with error:{e}")

        finally:
            # Save final results (whatever we trained)
            self.save_training_results(quantum_metrics, classical_metrics)

        return quantum_metrics, classical_metrics




    def save_training_results(self, quantum_metrics, classical_metrics):
        """
        Saves the training results
        """
        print("\n💾 Saving training results...")

        # Save training history
        with open(self.metrics_dir / "training_history.json", 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, default=str)

        # Save best metrics
        results = {
            'quantum_best_metrics': quantum_metrics,
            'classical_best_metrics': classical_metrics,
            'training_config': self.training_config,
            'model_config': self.model_config,
            'grad_history': self.grad_history
        }

        with open(self.metrics_dir / "training_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        # Create training plots
        self.plot_training_curves()

        print("✓ Training results saved successfully!")

    def plot_training_curves(self):
        """
        Create plots for the training curves.
        """
        print("📈 Creating training plots...")

        # Make a subfolder per run so plots never overwrite each other
        run_dir = self.plots_dir / self.run_name
        run_dir.mkdir(parents=True, exist_ok=True)


        def _align(a, b):
            n = min(len(a), len(b))
            return a[:n], b[:n], n

        # Check what we actually have
        q_has = len(self.training_history["quantum"]["train_loss"]) > 0
        c_has = len(self.training_history["classical"]["train_loss"]) > 0

        if not q_has and not c_has:
            print("⚠️ No epochs recorded for quantum or classical; skipping plots.")
            return

        # ---------- CASE 1: BOTH MODELS AVAILABLE ----------
        if q_has and c_has:
            print("📊 Plotting comparison: Quantum vs Classical")

            # --- Loss curves ---
            q_tr, c_tr, n = _align(
                self.training_history["quantum"]["train_loss"],
                self.training_history["classical"]["train_loss"],
            )
            q_val, c_val, _ = _align(
                self.training_history["quantum"]["val_loss"],
                self.training_history["classical"]["val_loss"],
            )
            if n == 0:
                print("⚠️ Histories exist but have no overlapping epochs; skipping plots.")
                return
            x = range(1, n + 1)

            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.plot(x, q_tr, "b-", label="Quantum Train", alpha=0.8)
            plt.plot(x, q_val, "b--", label="Quantum Val", alpha=0.8)
            plt.plot(x, c_tr, "r-", label="Classical Train", alpha=0.8)
            plt.plot(x, c_val, "r--", label="Classical Val", alpha=0.8)
            plt.title("Training Loss Comparison")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # --- CI curves ---
            q_train_ci = [m["overall"]["concordance_index"] for m in self.training_history["quantum"]["train_metrics"]]
            q_val_ci = [m["overall"]["concordance_index"] for m in self.training_history["quantum"]["val_metrics"]]
            c_train_ci = [m["overall"]["concordance_index"] for m in self.training_history["classical"]["train_metrics"]]
            c_val_ci = [m["overall"]["concordance_index"] for m in self.training_history["classical"]["val_metrics"]]

            q_train_ci, c_train_ci, n2 = _align(q_train_ci, c_train_ci)
            q_val_ci, c_val_ci, _ = _align(q_val_ci, c_val_ci)
            x2 = range(1, n2 + 1)

            plt.subplot(1, 3, 2)
            plt.plot(x2, q_train_ci, "b-", label="Quantum Train", alpha=0.8)
            plt.plot(x2, q_val_ci, "b--", label="Quantum Val", alpha=0.8)
            plt.plot(x2, c_train_ci, "r-", label="Classical Train", alpha=0.8)
            plt.plot(x2, c_val_ci, "r--", label="Classical Val", alpha=0.8)
            plt.title("Concordance Index Comparison")
            plt.xlabel("Epoch")
            plt.ylabel("Concordance Index")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # --- Pearson curves ---
            q_tr_p = [m["overall"]["pearson"] for m in self.training_history["quantum"]["train_metrics"]]
            q_val_p = [m["overall"]["pearson"] for m in self.training_history["quantum"]["val_metrics"]]
            c_tr_p = [m["overall"]["pearson"] for m in self.training_history["classical"]["train_metrics"]]
            c_val_p = [m["overall"]["pearson"] for m in self.training_history["classical"]["val_metrics"]]

            q_tr_p, c_tr_p, n3 = _align(q_tr_p, c_tr_p)
            q_val_p, c_val_p, _ = _align(q_val_p, c_val_p)
            x3 = range(1, n3 + 1)

            plt.subplot(1, 3, 3)
            plt.plot(x3, q_tr_p, "b-", label="Quantum Train", alpha=0.8)
            plt.plot(x3, q_val_p, "b--", label="Quantum Val", alpha=0.8)
            plt.plot(x3, c_tr_p, "r-", label="Classical Train", alpha=0.8)
            plt.plot(x3, c_val_p, "r--", label="Classical Val", alpha=0.8)
            plt.title("Pearson Correlation Comparison")
            plt.xlabel("Epoch")
            plt.ylabel("Pearson")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(run_dir / "training_comparison.png", dpi=300, bbox_inches="tight")
            plt.close()

        # ---------- CASE 2: ONLY ONE MODEL AVAILABLE ----------
        else:
            if c_has:
                key = "classical"
                color = "r"
                title_prefix = "Classical"
            else:
                key = "quantum"
                color = "b"
                title_prefix = "Quantum"

            print(f"📊 Plotting single model: {title_prefix}")

            tr = self.training_history[key]["train_loss"]
            val = self.training_history[key]["val_loss"]
            n = len(tr)
            if n == 0:
                print(f"⚠️ No epochs recorded for {title_prefix}; skipping plots.")
                return
            x = range(1, n + 1)

            plt.figure(figsize=(15, 5))
            # Loss
            plt.subplot(1, 3, 1)
            plt.plot(x, tr, f"{color}-", label=f"{title_prefix} Train", alpha=0.8)
            plt.plot(x, val, f"{color}--", label=f"{title_prefix} Val", alpha=0.8)
            plt.title(f"{title_prefix} Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # CI
            train_ci = [m["overall"]["concordance_index"] for m in self.training_history[key]["train_metrics"]]
            val_ci = [m["overall"]["concordance_index"] for m in self.training_history[key]["val_metrics"]]
            plt.subplot(1, 3, 2)
            plt.plot(x, train_ci, f"{color}-", label=f"{title_prefix} Train", alpha=0.8)
            plt.plot(x, val_ci, f"{color}--", label=f"{title_prefix} Val", alpha=0.8)
            plt.title(f"{title_prefix} Concordance Index")
            plt.xlabel("Epoch")
            plt.ylabel("Concordance Index")
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Pearson
            train_p = [m["overall"]["pearson"] for m in self.training_history[key]["train_metrics"]]
            val_p = [m["overall"]["pearson"] for m in self.training_history[key]["val_metrics"]]
            plt.subplot(1, 3, 3)
            plt.plot(x, train_p, f"{color}-", label=f"{title_prefix} Train", alpha=0.8)
            plt.plot(x, val_p, f"{color}--", label=f"{title_prefix} Val", alpha=0.8)
            plt.title(f"{title_prefix} Pearson Correlation")
            plt.xlabel("Epoch")
            plt.ylabel("Pearson")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            out_name = f"{key}_training_curves.png"
            plt.savefig(run_dir / out_name, dpi=300, bbox_inches="tight")
            plt.close()

        # ---------- Per-target performance (if any model has metrics) ----------
        # Prefer the exact keys present in stored metrics (most reliable)
        key_for_targets = "classical" if len(self.training_history["classical"]["val_metrics"]) > 0 else "quantum"
        first = self.training_history[key_for_targets]["val_metrics"][0]

        # metrics dict structure: {"overall": {...}, "<target1>": {...}, "<target2>": {...}}
        target_names = [k for k in first.keys() if k != "overall"]
        if target_names:
            if key_for_targets is not None and len(self.training_history[key_for_targets]["val_metrics"]) > 0:
                plt.figure(figsize=(20, 4))
                for i, target in enumerate(target_names):
                    t_ci = []
                    for m in self.training_history[key_for_targets]["val_metrics"]:
                        if target in m and "concordance_index" in m[target]:
                            t_ci.append(m[target]["concordance_index"])
                    x_t = range(1, len(t_ci) + 1)

                    plt.subplot(1, len(target_names), i + 1)
                    plt.plot(x_t, t_ci, "k-", label=key_for_targets.capitalize(), alpha=0.8)
                    plt.title(f"{target} CI")
                    plt.xlabel("Epoch")
                    plt.ylabel("Concordance Index")
                    plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(run_dir / f"{key_for_targets}_per_target_performance.png",
                            dpi=300, bbox_inches="tight")
                plt.close()

        # --- Per-target metrics: Pearson, RMSE, MAE ---
        if target_names:
            def plot_per_target_metric(metric_key: str, fname_suffix: str, y_label: str):
                q_has = len(self.training_history["quantum"]["val_metrics"]) > 0
                c_has = len(self.training_history["classical"]["val_metrics"]) > 0
                if not (q_has or c_has):
                    print(f"⚠️ No val_metrics for {metric_key}; skipping per-target {metric_key} plots.")
                    return

                num_targets = len(target_names)
                fig, axes = plt.subplots(
                    1, num_targets,
                    figsize=(4 * num_targets, 4),
                    sharey=False
                )
                if num_targets == 1:
                    axes = [axes]

                for i, target in enumerate(target_names):
                    ax = axes[i]

                    # Quantum line (if available)
                    if q_has:
                        try:
                            q_vals = []
                            for m in self.training_history["quantum"]["val_metrics"]:
                                if target in m and metric_key in m[target]:
                                    q_vals.append(m[target][metric_key])
                            ax.plot(
                                range(1, len(q_vals) + 1),
                                q_vals,
                                "b-",
                                label="Quantum",
                                alpha=0.8,
                            )
                        except KeyError:
                            pass  # metric not present for quantum

                    # Classical line (if available)
                    if c_has:
                        try:
                            c_vals = [
                                m[target][metric_key]
                                for m in self.training_history["classical"]["val_metrics"]
                            ]
                            ax.plot(
                                range(1, len(c_vals) + 1),
                                c_vals,
                                "r-",
                                label="Classical",
                                alpha=0.8,
                            )
                        except KeyError:
                            pass  # metric not present for classical

                    ax.set_title(f"{target} {metric_key}")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel(y_label)
                    ax.grid(True, alpha=0.3)
                    ax.legend()

                plt.tight_layout()
                out_path = run_dir / f"per_target_{fname_suffix}.png"
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"✓ Saved per-target {metric_key} plot to {out_path}")

            # Call for each metric we care about
            plot_per_target_metric("pearson", "pearson", "Pearson r")
            plot_per_target_metric("rmse", "rmse", "RMSE")
            plot_per_target_metric("mae", "mae", "MAE")


        print("✓ Training plots saved!")


def request_training_stop():
    """
    Can be called from outside to ask all running TrainingManager instances
    to stop at the next safe point.
    """
    STOP_TRAINING.set()

def reset_training_stop():
    """
    Clear the global stop flag so a new training run can start cleanly.
    """
    STOP_TRAINING.clear()