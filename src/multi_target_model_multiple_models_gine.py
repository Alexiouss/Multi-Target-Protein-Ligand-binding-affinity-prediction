"""
Multi-Target Model for Quantum Drug Discovery
Combines Graph Encoder, Quantum Attention and Classical variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import warnings
from pathlib import Path
import sys
import os
import math
from lifelines.utils import concordance_index# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
src_dir = os.path.join(project_dir, 'src')
sys.path.append(src_dir)


import datetime
DEBUG_LOG_PATH = "amplitude_debug.log"

def log_debug(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

# Import modules
try:
    from graph_encoder_gin import MolecularGraphEncoder
    from quantum_attention_refactored import QuantumMultiTargetAttention
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure graph_encoder.py and quantum_attention.py are in src/ directory")

warnings.filterwarnings('ignore')

# Device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def ci_lifelines(y_true, y_pred):
    # lifelines assumes higher prediction = higher risk (worse),
    # but for our use (higher y_pred = higher y_true) it's the same C-index.
    return concordance_index(y_true, y_pred)

class ClassicalAttentionLayer(nn.Module):
    """
    Classical single-head attention using PyTorch's built-in MultiheadAttention.
    Mirrors the interface of QuantumMultiTargetAttention:
    
    Input: molecular_features [B, molecular_dim]
    Output:
        - attention_weights [B, n_targets]
        - attended_features [B, target_dim]
    """

    def __init__(self, molecular_dim=256, target_dim=128, n_targets=5, num_heads=1, dropout=0.1):
        super().__init__()

        self.molecular_dim = molecular_dim
        self.target_dim = target_dim
        self.n_targets = n_targets
        self.num_heads = num_heads

        # Target embeddings (same idea as quantum side)
        self.target_embeddings = nn.Embedding(n_targets, target_dim)
        nn.init.xavier_normal_(self.target_embeddings.weight)

        # Projection for molecular features → query
        self.query_projection = nn.Linear(molecular_dim, target_dim)

        # Multi-head attention module (here num_heads = 1 for fairness)
        self.attention = nn.MultiheadAttention(
            embed_dim=target_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True   # inputs/outputs: [B, seq_len, D]
        )

        self.dropout = nn.Dropout(dropout)

        # Optional output projection (keeps same dim, but adds flexibility)
        self.out_projection = nn.Linear(target_dim, target_dim)

        self.to(device)

    def forward(self, molecular_features):
        """
        Args:
            molecular_features: [B, molecular_dim]

        Returns:
            attention_weights: [B, n_targets]
            attended_features: [B, target_dim]
        """
        batch_size = molecular_features.size(0)
        molecular_features = molecular_features.to(device)

        # ---- 1. Build Q from molecule ----
        # molecular_features: [B, 256]
        # Q: [B, 1, D]
        Q = self.query_projection(molecular_features).unsqueeze(1)

        # ---- 2. Build K, V from target embeddings ----
        # target_embs: [T, D]
        target_indices = torch.arange(self.n_targets, device=device)
        target_embs = self.target_embeddings(target_indices)

        # K, V as sequences per batch: [B, T, D]
        KV = target_embs.unsqueeze(0).expand(batch_size, -1, -1)

        # ---- 3. MultiheadAttention ----
        # Q: [B, 1, D], K: [B, T, D], V: [B, T, D]
        attended, attn_weights = self.attention(
            query=Q,     # [B, 1, D]
            key=KV,      # [B, T, D]
            value=KV     # [B, T, D]
        )
        # attended: [B, 1, D]
        attended = attended.squeeze(1)  # [B, D]

        # attn_weights: [B, 1, T]  (since tgt_len = 1)
        attention_weights = attn_weights.squeeze(1)  # [B, T]
        attention_weights = self.dropout(attention_weights)

        # ---- 4. Optional output projection ----
        attended_features = self.out_projection(attended)  # [B, D]

        return attention_weights, attended_features


class MultiTargetPredictor(nn.Module):
    """
    Base class for multi-target drug discovery models
    """
    
    def __init__(self, config_path="config.json", model_type="quantum", target_names_override=None):
        super(MultiTargetPredictor, self).__init__()

        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.model_type = model_type
        self.feature_dim = self.config['model']['feature_dim']
        self.use_chembl = self.config['data']['use_chembl']

        # --- Targets (safe override) ---
        config_target_names = list(self.config["targets"].keys())

        if target_names_override is None:
            self.target_names = config_target_names
        else:
            self.target_names = list(target_names_override)

            #Strong safety checks
            if len(self.target_names) == 0:
                raise ValueError("target_names_override is empty.")

            # If you want it strict (recommended): must be subset of config targets
            missing = [t for t in self.target_names if t not in config_target_names]
            if missing:
                raise ValueError(
                    f"target_names_override contains unknown targets: {missing}. "
                    f"Known targets from config: {config_target_names}"
                )

        # Effective number of targets must match heads / attention dims / predictors
        self.n_targets = len(self.target_names)

        # --- Graph encoder ---
        self.graph_encoder = MolecularGraphEncoder(
            output_dim=256,
            hidden_dim=64,
            num_layers=3,
            dropout=0.1
        )

        # --- Attention ---
        if model_type == "quantum":
            # IMPORTANT: Quantum attention also needs to know n_targets
            self.attention = QuantumMultiTargetAttention(config_path)

        else:
            self.attention = ClassicalAttentionLayer(
                molecular_dim=256,
                target_dim=self.feature_dim,
                n_targets=self.n_targets
            )

        # --- Target-specific heads ---
        self.target_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1)
            ) for _ in range(self.n_targets)
        ])

        self.output_activation = nn.Sigmoid()
        self.pk_min = 3.0
        self.pk_max = 12.0
        self.to(device)

    
    def forward(self, smiles_list):
        """
        Forward pass for multi-target prediction
        
        Args:
            smiles_list: List from SMILES strings
            
        Returns:
            predictions: [batch_size, n_targets] binding affinity predictions (pKd)
            attention_weights: [batch_size, n_targets] attention weights
            molecular_features: [batch_size, 256] molecular representations
        """
        # 1. Graph encoding (shared backbone)
        molecular_features, valid_indices = self.graph_encoder(smiles_list)

        #DEBUG: check graph encoder output
        if not torch.isfinite(molecular_features).all():
            log_debug(
                f"[GINe_OUT_NAN] min={molecular_features.min().item()} "
                f"max={molecular_features.max().item()} "
                f"any_nan={torch.isnan(molecular_features).any().item()} "
                f"any_inf={torch.isinf(molecular_features).any().item()}"
            )
            raise RuntimeError("Graph encoder produced NaNs in molecular_features")

        
        out = self.attention(molecular_features)

        # Support both classical (2 outputs) and quantum (3 outputs)
        if isinstance(out, (tuple, list)) and len(out) == 2:
            attention_weights, attended_features = out
            raw_scores = None
        elif isinstance(out, (tuple, list)) and len(out) >= 3:
            attention_weights, attended_features, raw_scores = out[:3]
        else:
            raise RuntimeError(f"Unexpected attention output type/len: {type(out)}")

        
        # 3. Shared target-specific prediction heads
        predictions = []
        for target_idx in range(self.n_targets):
            
            # Prediction head (shared between classical & quantum models)
            pred = self.target_predictors[target_idx](attended_features)    # [B, 1]
            predictions.append(pred)
        
        predictions = torch.cat(predictions, dim=1)  # [B, n_targets]
        
        # 4. Shared scaling to realistic pKd range [3, 12]
        # Sigmoid -> [0,1], then linear map to [pk_min, pk_max]
        if not self.use_chembl:
                predictions = self.output_activation(predictions)  # Sigmoid -> [0,1]
                predictions = predictions * (self.pk_max - self.pk_min) + self.pk_min  # Scale to [3,12]
        
        return predictions, attention_weights, molecular_features

    
    def predict_single_target(self, smiles_list, target_name):
        """
        Prediction for a single target
        """
        if target_name not in self.target_names:
            raise ValueError(f"Target {target_name} not found. Available: {self.target_names}")
        
        target_idx = self.target_names.index(target_name)
        predictions, attention_weights, _ = self.forward(smiles_list)
        
        return predictions[:, target_idx], attention_weights[:, target_idx]
    
    def get_attention_visualization(self, smiles_list):
        """
        Returns attention weights for visualization
        """
        with torch.no_grad():
            predictions, attention_weights, molecular_features = self.forward(smiles_list)
        
        return {
            'smiles': smiles_list,
            'targets': self.target_names,
            'attention_weights': attention_weights.cpu().numpy(),
            'predictions': predictions.cpu().numpy()
        }

class MultiTargetLoss(nn.Module):
    def __init__(self, target_weights=None, loss_type='mse'):
        super().__init__()
        self.target_weights = target_weights
        self.loss_type = loss_type

        if loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction='none')
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss(reduction='none')
        elif loss_type == 'huber':
            self.criterion = nn.HuberLoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, predictions, targets, attention_weights=None, mask=None, eps=1e-12):
        """
        predictions: [B,T]
        targets:     [B,T]
        mask:        [B,T] where 1=measured, 0=not measured (optional)
        Returns:
          total_loss (scalar), individual_losses ([T])
        """
        B, T = predictions.shape

        # base per-entry loss [B,T]
        losses = self.criterion(predictions, targets)

        # optional attention weighting
        if attention_weights is not None:
            aw = attention_weights.detach()
            losses = losses * (1.0 + aw)

        # optional target weights
        if self.target_weights is not None:
            tw = torch.tensor(self.target_weights, device=losses.device, dtype=losses.dtype)
            losses = losses * tw.unsqueeze(0)

        # mask (measured-only)
        if mask is None:
            mask = torch.ones_like(losses)

        losses = losses * mask

        # per-target mean over measured entries (avoid divide by 0)
        denom_t = mask.sum(dim=0).clamp_min(eps)          # [T]
        individual_losses = losses.sum(dim=0) / denom_t   # [T]

        # overall mean over measured entries
        denom_all = mask.sum().clamp_min(eps)
        total_loss = losses.sum() / denom_all

        return total_loss, individual_losses


def calculate_metrics(predictions, targets, target_names, mask=None):
    """
    Compute metrics only on measured entries (mask==1).
    predictions/targets: torch tensors [N,T]
    mask: torch tensor [N,T] with 1=measured, 0=not measured (optional)
    """
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np

    preds = predictions.detach().cpu().numpy()
    trues = targets.detach().cpu().numpy()

    if mask is None:
        m = np.ones_like(trues, dtype=bool)
    else:
        m = mask.detach().cpu().numpy().astype(bool)

    metrics = {}

    # ---- overall: flatten only measured entries
    overall_preds = preds[m]
    overall_trues = trues[m]

    if overall_preds.size < 2 or np.std(overall_preds) == 0 or np.std(overall_trues) == 0:
        pear = 0.0
    else:
        pear = float(pearsonr(overall_preds, overall_trues)[0])

    # CI needs >=2 non-equal targets; if not, fall back to 0.5
    try:
        overall_ci = float(ci_lifelines(overall_trues, overall_preds))
    except Exception:
        overall_ci = 0.5

    metrics["overall"] = {
        "pearson": pear,
        "rmse": float(np.sqrt(mean_squared_error(overall_trues, overall_preds))) if overall_preds.size > 0 else float("nan"),
        "mae": float(mean_absolute_error(overall_trues, overall_preds)) if overall_preds.size > 0 else float("nan"),
        "concordance_index": overall_ci,
        "n_measured": int(overall_preds.size),
    }

    # ---- per-target metrics
    for i, target_name in enumerate(target_names):
        mi = m[:, i]
        tp = preds[:, i][mi]
        tt = trues[:, i][mi]

        if tp.size < 2 or np.std(tp) == 0 or np.std(tt) == 0:
            pear_i = 0.0
        else:
            pear_i = float(pearsonr(tp, tt)[0])

        try:
            ci_i = float(ci_lifelines(tt, tp))
        except Exception:
            ci_i = 0.5

        metrics[target_name] = {
            "pearson": pear_i,
            "rmse": float(np.sqrt(mean_squared_error(tt, tp))) if tp.size > 0 else float("nan"),
            "mae": float(mean_absolute_error(tt, tp)) if tp.size > 0 else float("nan"),
            "concordance_index": ci_i,
            "n_measured": int(tp.size),
        }

    return metrics

def test_multi_target_model():
    """
    Test function for the complete multi-target system
    """
    print("🧪 Testing Multi-Target Model System...")
    
    try:
        # Test configuration
        config = {
            "model": {
                "n_targets": 5,
                "n_qubits": 4,
                "feature_dim": 64
            },
            "targets": {
                "CDK2": "Cyclin-dependent kinase 2",
                "DRD2": "Dopamine receptor D2", 
                "HIV1_PR": "HIV-1 protease",
                "hERG": "hERG potassium channel",
                "ESR1": "Estrogen receptor alpha"
            },
            "training": {
                "batch_size": 4,
                "epochs": 20,
                "learning_rate": 0.001
            }
        }
        
        # Save test config
        with open('test_config.json', 'w') as f:
            json.dump(config, f)
        
        print("🔬 Testing Quantum Model...")
        # Create quantum model
        quantum_model = MultiTargetPredictor('test_config.json', model_type="quantum")
        quantum_model.eval()
        
        print("🖥️ Testing Classical Model...")
        # Create classical model
        classical_model = MultiTargetPredictor('test_config.json', model_type="classical")
        classical_model.eval()
        
        # Test data
        test_smiles = [
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen-like
            "CC1=CC=C(C=C1)C(=O)C2=CC=CC=C2",  # Benzophenone-like
            "CN1CCN(CC1)C2=C(C=C3C(=C2)N=CN3C4=CC=CC=C4F)C#N",  # Complex drug-like
            "CCO"  # Ethanol (simple)
        ]
        
        print(f"✓ Test SMILES: {len(test_smiles)}")
        print(f"✓ Device: {device}")
        
        # Test quantum model
        with torch.no_grad():
            q_predictions, q_attention, q_molecular = quantum_model(test_smiles)
            
        print(f"✓ Quantum predictions shape: {q_predictions.shape}")
        print(f"✓ Quantum attention shape: {q_attention.shape}")
        print(f"✓ Quantum prediction range: [{q_predictions.min():.3f}, {q_predictions.max():.3f}]")
        
        # Test classical model
        with torch.no_grad():
            c_predictions, c_attention, c_molecular = classical_model(test_smiles)
            
        print(f"✓ Classical predictions shape: {c_predictions.shape}")
        print(f"✓ Classical attention shape: {c_attention.shape}")
        print(f"✓ Classical prediction range: [{c_predictions.min():.3f}, {c_predictions.max():.3f}]")
        
        # Test loss function
        print("\n📊 Testing Loss Function...")
        loss_fn = MultiTargetLoss(loss_type='mse')
        
        # Synthetic targets
        targets = torch.randn_like(q_predictions) * 2 + 7  # Random targets around 7 pKd
        
        q_total_loss, q_individual_losses = loss_fn(q_predictions, targets, q_attention)
        c_total_loss, c_individual_losses = loss_fn(c_predictions, targets, c_attention)
        
        print(f"✓ Quantum total loss: {q_total_loss:.4f}")
        print(f"✓ Classical total loss: {c_total_loss:.4f}")
        
        # Test single target prediction
        print("\n🎯 Testing Single Target Prediction...")
        q_cdk2_pred, q_cdk2_att = quantum_model.predict_single_target(test_smiles, "CDK2")
        c_cdk2_pred, c_cdk2_att = classical_model.predict_single_target(test_smiles, "CDK2")
        
        print(f"✓ CDK2 quantum prediction shape: {q_cdk2_pred.shape}")
        print(f"✓ CDK2 classical prediction shape: {c_cdk2_pred.shape}")
        
        # Test attention visualization
        print("\n👁️ Testing Attention Visualization...")
        q_viz = quantum_model.get_attention_visualization(test_smiles[:2])
        c_viz = classical_model.get_attention_visualization(test_smiles[:2])
        
        print(f"✓ Quantum attention visualization keys: {list(q_viz.keys())}")
        print(f"✓ Classical attention visualization keys: {list(c_viz.keys())}")
        
        # Sample comparison
        print(f"\n📋 Sample Predictions Comparison:")
        print(f"Molecule: {test_smiles[0][:30]}...")
        print("Target | Quantum | Classical | Attention (Q) | Attention (C)")
        print("-" * 65)
        
        target_names = quantum_model.target_names
        for i, target in enumerate(target_names):
            q_pred = q_predictions[0, i].item()
            c_pred = c_predictions[0, i].item()
            q_att = q_attention[0, i].item()
            c_att = c_attention[0, i].item()
            print(f"{target:6} | {q_pred:7.3f} | {c_pred:9.3f} | {q_att:11.3f} | {c_att:11.3f}")
        
        # Parameter count comparison
        q_params = sum(p.numel() for p in quantum_model.parameters() if p.requires_grad)
        c_params = sum(p.numel() for p in classical_model.parameters() if p.requires_grad)
        
        print(f"\n📈 Model Comparison:")
        print(f"✓ Quantum model parameters: {q_params:,}")
        print(f"✓ Classical model parameters: {c_params:,}")
        print(f"✓ Parameter reduction: {(1 - q_params/c_params)*100:.1f}%")
        
        print("\n✅ Multi-Target Model test successful!")
        print("🎉 All the components work correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in the Multi-Target Model test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_multi_target_model()