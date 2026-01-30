"""
Quantum Attention Mechanism for Multi-Target Drug Discovery
"""

import os
import datetime
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import json
import warnings
import math
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# Device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import math
import torch
import numpy as np
import pennylane as qml

DEBUG_LOG_PATH = "amplitude_debug.log"

def log_debug(msg: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {msg}\n")


class QuantumAttentionCircuit:
    def __init__(
        self,
        n_qubits=14,
        n_targets=5,
        feature_dim=128,
        n_layers=3,
        device_name="lightning.qubit",
        encoding_type="angle",
        use_data_reuploading=False,
    ):
        self.n_qubits = n_qubits
        self.n_targets = n_targets
        self.feature_dim = feature_dim
        self.n_layers = n_layers
        self.encoding_type = encoding_type
        self.use_data_reuploading = use_data_reuploading

        # how many features we TRY to push per upload (RY + RZ)
        self.features_per_upload = self.n_qubits * 2

        #PCA
        self.pca_output_dim = 16
        self.pca_components = None
        self.pca_mean = None

        # --- NEW: PCA for angle encoding ---
        self.angle_pca_components = None
        self.angle_pca_mean = None
        self.angle_pca_output_dim = None

        self.dev = qml.device(device_name, wires=n_qubits)
        print(f"✓ Quantum device: {n_qubits} qubits | encoding={encoding_type} | reupload={use_data_reuploading}")
        self.circuit = self._create_enhanced_circuit()
    
    def set_angle_pca(self, components: np.ndarray, mean: np.ndarray):
        """
        Set PCA parameters for angle encoding.

        components: shape [2*n_qubits, feature_dim]
        mean:       shape [feature_dim]
        """
        self.angle_pca_components = torch.tensor(components, dtype=torch.float32, device=device)
        self.angle_pca_mean = torch.tensor(mean, dtype=torch.float32, device=device)
        self.angle_pca_output_dim = self.angle_pca_components.shape[0]


    def set_pca(self, components: np.ndarray, mean: np.ndarray):
        """
        Set PCA parameters for amplitude encoding.

        components: shape [pca_output_dim, feature_dim]
        mean:       shape [feature_dim]
        """
        # Store as torch tensors on the correct device
        self.pca_components = torch.tensor(
            components, dtype=torch.float32, device=device
        )
        self.pca_mean = torch.tensor(
            mean, dtype=torch.float32, device=device
        )



    def _create_enhanced_circuit(self):
        diff_method = "adjoint" if self.encoding_type == "angle" else "parameter-shift"
        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def enhanced_circuit(query_features, key_features, params):
            # Combine query and key into a single feature vector
            combined = torch.cat([query_features, key_features], dim=-1)

            if self.encoding_type == "amplitude":
                if self.use_data_reuploading:
                    # STRICT data re-uploading with amplitude encoding:
                    # same combined vector encoded before every variational layer
                    self._amplitude_reupload_encode_and_layers(combined, params)
                else:
                    # Single amplitude encoding + stacked variational layers
                    self._amplitude_encode(combined)
                    self._enhanced_variational_layers(params)

            else:  # angle
                if self.use_data_reuploading:
                    # angle encoding with (fixed) streaming-style reuploading
                    self._angle_reupload_true(combined, params)
                else:
                    # original angle encoding: encode once, then variational layers
                    self._angle_dense_encode(combined)
                    self._enhanced_variational_layers(params)

            # measurements: Z on all qubits, Y on first 1–2 qubits
            measurements = []
            for i in range(self.n_qubits):
                measurements.append(qml.expval(qml.PauliZ(i)))
            for i in range(min(2, self.n_qubits)):
                measurements.append(qml.expval(qml.PauliY(i)))
            return measurements

        return enhanced_circuit

    def _angle_dense_encode(self, combined):
        """
        Dense angle encoding without reupload:
        - Each qubit gets up to 2 distinct features (RY and RZ).
        - Total used features per upload = 2 * n_qubits.
        """

        # NEW: apply PCA for angle encoding if available
        if (
            self.encoding_type == "angle"
            and self.angle_pca_components is not None
            and self.angle_pca_mean is not None
        ):
            x = combined.to(self.angle_pca_components.device).float()
            x_centered = x - self.angle_pca_mean
            combined = torch.matmul(self.angle_pca_components, x_centered)  # [2*n_qubits]

        # unchanged: still uses 2 features per qubit
        self._angle_encode_slice_two_per_qubit(combined, start_idx=0)


    # -------------------- amplitude (as before) --------------------
    def _amplitude_encode(self, combined):
        """
        Amplitude encoding with PCA reduction.

        If PCA is set:
        combined (len = feature_dim) --> PCA --> len = 16
        Then we pad/truncate this 16-dim vector to 2**n_qubits as before.
        """
        # --- NEW: apply PCA only if parameters are available ---
        if self.encoding_type == "amplitude" and self.pca_components is not None and self.pca_mean is not None:
            # combined is 1D vector: [feature_dim]
            x = combined

            # Ensure same device / dtype
            x = x.to(self.pca_components.device).float()

            # Center
            x_centered = x - self.pca_mean

            # Project: [feature_dim] -> [pca_output_dim=16]
            # components: [16, feature_dim]
            x_reduced = torch.matmul(self.pca_components, x_centered)  # [16]
            combined = x_reduced

        # --- rest is your previous amplitude logic, but now 'combined' is 16-dim ---
        target_len = 2 ** self.n_qubits

        if combined.shape[0] > target_len:
            vec = combined[:target_len]
        elif combined.shape[0] < target_len:
            pad_len = target_len - combined.shape[0]
            vec = torch.cat([combined, torch.zeros(pad_len, dtype=combined.dtype, device=combined.device)])
        else:
            vec = combined

        # --- safe normalization ---
        eps = 1e-8
        norm = torch.linalg.norm(vec)
        norm2 = torch.linalg.norm(vec).item()
        if not np.isfinite(norm2) or norm2 < 1e-8:
            log_debug(f"[AMP_VEC_BAD] norm={norm2} min={vec.min().item()} max={vec.max().item()}")
        if norm < eps or not torch.isfinite(norm):
            # fallback to |00...0> (i.e., amplitude vector [1,0,0,...])
            vec = torch.zeros_like(vec)
            vec[0] = 1.0
        else:
            vec = vec / norm
        qml.AmplitudeEmbedding(features=vec, wires=range(self.n_qubits), normalize=False)


    def _amplitude_reupload_encode_and_layers(self, combined, params):
        """
        Strict data re-uploading for amplitude encoding:
        |psi(x, theta)> = Π_{layer} [ U_theta_layer U_amp(x) ] |0>

        We:
        - encode the SAME 'combined' vector before every variational layer
        - use a separate parameter block per layer (3 angles per qubit)
        """
        n_params_per_layer = self.n_qubits * 3

        for layer in range(self.n_layers):
            # Re-upload SAME classical input via amplitude encoding
            self._amplitude_encode(combined)

            # Apply one variational layer with layer-specific parameters
            base = layer * n_params_per_layer
            for qubit in range(self.n_qubits):
                p0 = base + qubit * 3
                qml.RX(params[p0],     wires=qubit)
                qml.RY(params[p0 + 1], wires=qubit)
                qml.RZ(params[p0 + 2], wires=qubit)

            # Entanglement pattern can depend on layer index
            self._entangle(layer)

    # -------------------- angle + data re-uploading (fixed) --------------------
    def _angle_reupload_true(self, combined, params):
        """
        True data reuploading for angle encoding:
        |ψ(x, θ)> = Π_layer [ U_θ_layer U_enc(x) ] |0>

        - Same classical 'combined' vector is encoded at every layer
        - Different parameter block per layer (3 angles per qubit per layer)
        """
        n_params_per_layer = self.n_qubits * 3

        for layer in range(self.n_layers):
            # Re-encode the SAME combined vector (dense encoding)
            self._angle_dense_encode(combined)

            # Layer-specific variational block
            base = layer * n_params_per_layer
            for qubit in range(self.n_qubits):
                p0 = base + qubit * 3
                qml.RX(params[p0],     wires=qubit)
                qml.RY(params[p0 + 1], wires=qubit)
                qml.RZ(params[p0 + 2], wires=qubit)

            # Entangling pattern (same as everywhere else)
            self._entangle(layer)

    def _angle_encode_slice_two_per_qubit(self, combined, start_idx):
        """
        For each qubit:
          RY ← feature[k]
          RZ ← feature[k+1]
        but ONLY if those indices exist.
        """
        total_len = combined.shape[0]
        for q in range(self.n_qubits):
            idx_ry = start_idx + 2 * q
            idx_rz = start_idx + 2 * q + 1

            if idx_ry < total_len:
                angle_ry = torch.tanh(combined[idx_ry]) * np.pi
                qml.RY(angle_ry, wires=q)

            if idx_rz < total_len:
                angle_rz = torch.tanh(combined[idx_rz]) * np.pi
                qml.RZ(angle_rz, wires=q)

    # -------------------- variational layers --------------------
    def _enhanced_variational_layers(self, params):
        n_params_per_layer = self.n_qubits * 3
        for layer in range(self.n_layers):
            base = layer * n_params_per_layer
            for qubit in range(self.n_qubits):
                p0 = base + qubit * 3
                qml.RX(params[p0], wires=qubit)
                qml.RY(params[p0 + 1], wires=qubit)
                qml.RZ(params[p0 + 2], wires=qubit)
            self._entangle(layer)

    def _variational_layer(self, params, layer_idx):
        n_params_per_layer = self.n_qubits * 3
        base = layer_idx * n_params_per_layer
        for qubit in range(self.n_qubits):
            p0 = base + qubit * 3
            qml.RX(params[p0], wires=qubit)
            qml.RY(params[p0 + 1], wires=qubit)
            qml.RZ(params[p0 + 2], wires=qubit)
        self._entangle(layer_idx)

    def _entangle(self, layer):
        if layer % 2 == 0:
            for q in range(self.n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        else:
            for q in range(self.n_qubits):
                qml.CNOT(wires=[q, (q + 1) % self.n_qubits])
        if layer > 0:
            for q in range(0, self.n_qubits - 1, 2):
                qml.CZ(wires=[q, q + 1])



class QuantumAttentionLayer(nn.Module):
    """
    Quantum attention layer
    """

    def __init__(self, 
                molecular_dim=256, 
                target_dim=128, 
                n_qubits=6, 
                n_targets=5, 
                n_layers=3,        
                device_name="lightning.qubit",
                encoding_type="angle",
                use_data_reuploading=False):
        super(QuantumAttentionLayer, self).__init__()

        self.molecular_dim = molecular_dim
        self.target_dim = target_dim
        self.n_qubits = n_qubits
        self.n_targets = n_targets
        self.n_layers = n_layers

        # Target embeddings
        self.target_embeddings = nn.Embedding(n_targets, target_dim)
        # Xavier initialization
        nn.init.xavier_normal_(self.target_embeddings.weight)

        # Multi-layer projections for feature transformation
        #Lighter projection
        self.query_projection = nn.Sequential(
            nn.Linear(molecular_dim, target_dim),
            nn.LayerNorm(target_dim),
            nn.ReLU(),
        )
        #Lighter projections
        self.key_projection = nn.Sequential(
            nn.Linear(target_dim, target_dim),
            nn.ReLU(),
        )

        self.value_projection = nn.Sequential(
            nn.Linear(target_dim, target_dim),
            nn.ReLU(),
        )

        # Enhanced quantum circuit
        self.quantum_circuit = QuantumAttentionCircuit(
            n_qubits,
            n_targets,
            target_dim * 2,
            n_layers,
            device_name=device_name,
            encoding_type=encoding_type,
            use_data_reuploading=use_data_reuploading
        )
        
        n_params_per_circuit = n_qubits * 3 * n_layers

        self.quantum_params = nn.ParameterList([
            nn.Parameter(torch.normal(0, 0.1, (n_params_per_circuit,), dtype=torch.float32))
            for _ in range(n_targets)
        ])

        # Enhanced attention head with more measurements
        n_measurements = n_qubits + min(2, n_qubits)  # Z + Y measurements
        #Simpler attention_head
        self.attention_head = nn.Sequential(
            nn.Linear(n_measurements, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Temperature parameter for attention sharpening
        self.attention_temperature = nn.Parameter(torch.tensor(1.0))

        # Normalization layers
        self.query_norm = nn.LayerNorm(target_dim)
        self.key_norm = nn.LayerNorm(target_dim)
        self.value_norm = nn.LayerNorm(target_dim)

        self.to(device)

    def forward(self, molecular_features, target_indices=None):

        """Forward pass"""

        batch_size = molecular_features.size(0)
        molecular_features = molecular_features.to(device)

        # Enhanced query projection
        queries = self.query_projection(molecular_features)
        queries = self.query_norm(queries)

        # Compute all target keys and values
        all_target_keys = []
        all_target_values = []

        for target_idx in range(self.n_targets):
            target_id = torch.tensor([target_idx], device=device)
            target_emb = self.target_embeddings(target_id).squeeze(0)

            # Batch for normalization
            target_batch = target_emb.unsqueeze(0)
            key = self.key_projection(target_batch).squeeze(0)
            value = self.value_projection(target_batch).squeeze(0)

            key = self.key_norm(key)
            value = self.value_norm(value)

            all_target_keys.append(key)
            all_target_values.append(value)

        # Enhanced quantum attention computation
        raw_attention_scores = []

        for target_idx in range(self.n_targets):
            batch_scores = []

            for sample_idx in range(batch_size):
                query = queries[sample_idx]
                key = all_target_keys[target_idx]
                # Before combined = torch.cat(...)
                if not torch.isfinite(query).all():
                    log_debug(f"[QUERY_NAN] target={target_idx} sample={sample_idx} "
                            f"min={query.min().item()} max={query.max().item()} "
                            f"contains_nan={torch.isnan(query).any().item()}")
                if not torch.isfinite(key).all():
                    log_debug(f"[KEY_NAN] target={target_idx} sample={sample_idx} "
                            f"min={key.min().item()} max={key.max().item()} "
                            f"contains_nan={torch.isnan(key).any().item()}")

                combined = torch.cat([query, key], dim=-1)

                # ---------------- DEBUG BLOCK START ----------------
                # Check NaN / Inf
                if not torch.isfinite(combined).all():
                    log_debug(
                        f"NON-FINITE DETECTED | target={target_idx} | sample={sample_idx} | "
                        f"min={combined.min().item()} | max={combined.max().item()} | "
                        f"NaN={torch.isnan(combined).any().item()} | Inf={torch.isinf(combined).any().item()} | "
                        f"vector={combined.detach().cpu().numpy().tolist()}"
                    )
                    # optionally raise to halt execution
                    raise RuntimeError("Non-finite combined vector before amplitude encoding")

                # Check weird norms
                norm = torch.linalg.norm(combined).item()
                if not np.isfinite(norm) or norm == 0.0 or norm > 1e6:
                    log_debug(
                        f"BAD NORM DETECTED | target={target_idx} | sample={sample_idx} | "
                        f"norm={norm} | min={combined.min().item()} | max={combined.max().item()} | "
                        f"vector={combined.detach().cpu().numpy().tolist()}"
                    )
                    # optionally raise
                    raise RuntimeError("Bad norm in combined vector before amplitude encoding")
                # ---------------- DEBUG BLOCK END ----------------


                query_in = query
                if self.quantum_circuit.encoding_type == "amplitude":
                    query_in = query.detach()   # <-- THE KEY TEST

                quantum_measurements = self.quantum_circuit.circuit(
                    query_in, key, self.quantum_params[target_idx]
                )

                # Process measurements
                measurements_tensor = torch.stack(quantum_measurements).float()

                # Enhanced attention computation
                attention_score = self.attention_head(measurements_tensor.unsqueeze(0))
                batch_scores.append(attention_score.squeeze())

            batch_scores = torch.stack(batch_scores, dim=0)
            raw_attention_scores.append(batch_scores)

        # Stack and temperature-scaled softmax
        raw_scores = torch.stack(raw_attention_scores, dim=1)
        attention_weights = F.softmax(raw_scores / self.attention_temperature, dim=1)

        # Enhanced attended features computation
        attended_features = torch.zeros(batch_size, self.target_dim, device=device)

        for target_idx in range(self.n_targets):
            weight = attention_weights[:, target_idx].unsqueeze(1)
            target_value = all_target_values[target_idx].unsqueeze(0).expand(batch_size, -1)
            attended_features += weight * target_value

        return attention_weights, attended_features,raw_scores

class QuantumMultiTargetAttention(nn.Module):
    def __init__(
        self,
        config_path: str = "config.json",
        n_qubits: int = 4,
        encoding_type: str = "angle",
        use_data_reuploading: bool = False,
        device_name: str = "lightning.qubit",
        n_targets_override: Optional[int] = None,
        target_names_override: Optional[List[str]] = None,
    ):

        super(QuantumMultiTargetAttention, self).__init__()

        with open(config_path, 'r') as f:
            config = json.load(f)

        model_config = config['model']
        # --- targets from config ---
        config_target_names = list(config['targets'].keys())
        config_n_targets = int(model_config['n_targets'])
        # --- apply overrides safely ---
        if target_names_override is not None:
            target_names = list(target_names_override)
        else:
            target_names = config_target_names

        if n_targets_override is not None:
            n_targets = int(n_targets_override)
        else:
            n_targets = len(target_names)

        # ✅ strict consistency checks
        if n_targets != len(target_names):
            raise ValueError(
                f"n_targets_override={n_targets} but len(target_names_override)={len(target_names)}"
            )

        missing = [t for t in target_names if t not in config_target_names]
        if missing:
            raise ValueError(f"Unknown targets in override: {missing}. Known: {config_target_names}")

        self.target_names = target_names
        self.n_targets = n_targets
        self.n_qubits = int(model_config.get("n_qubits",n_qubits))
        self.feature_dim = model_config['feature_dim']
        self.n_layers = model_config.get('n_layers', 3)
        encoding_type = model_config.get("encoding_type", encoding_type)
        use_data_reuploading = bool(model_config.get("use_data_reuploading", use_data_reuploading))
        self.encoding_type = encoding_type
        effective_n_layers = self.n_layers

        if encoding_type=="amplitude":
            effective_n_layers = min(self.n_layers,1)

        if encoding_type=="amplitude" and use_data_reuploading:
            effective_n_layers = 2

        # Use user-provided quantum settings (device fixed by default)
        self.quantum_attention = QuantumAttentionLayer(
            molecular_dim=256,
            target_dim=self.feature_dim,
            n_qubits=self.n_qubits,
            n_targets=self.n_targets,
            n_layers=effective_n_layers,
            device_name=device_name,          # "lightning.qubit"
            encoding_type=encoding_type,
            use_data_reuploading=use_data_reuploading
        )
        if encoding_type == "amplitude":
            p = model_config.get("pca_path", "pca_amplitude.npz")
            if os.path.exists(p):
                d = np.load(p, allow_pickle=True)
                self.quantum_attention.quantum_circuit.set_pca(
                    d["components"], d["mean"]
                )
                print(f"✓ Loaded AMPLITUDE PCA: {p}")

        elif encoding_type == "angle":
            # NEW: allow config override (so we can store backbone-specific file)
            p = model_config.get("angle_pca_path", f"pca_angle_{self.n_qubits}.npz")
            if os.path.exists(p):
                d = np.load(p, allow_pickle=True)
                self.quantum_attention.quantum_circuit.set_angle_pca(d["components"], d["mean"])
                print(f"✓ Loaded ANGLE PCA: {p}")
            else:
                # Backward-compat fallback to old naming if angle_pca_path points to missing file
                p2 = f"pca_angle_{self.n_qubits}.npz"
                if p2 != p and os.path.exists(p2):
                    d = np.load(p2, allow_pickle=True)
                    self.quantum_attention.quantum_circuit.set_angle_pca(d["components"], d["mean"])
                    print(f"✓ Loaded ANGLE PCA (fallback): {p2}")

        self.to(device)

    def forward(self, molecular_features):
        """
        Forward pass
        """
        # Quantum attention
        attention_weights, attended_features,raw_scores=self.quantum_attention(molecular_features)
        return attention_weights,attended_features,raw_scores
    
    def get_interpretability_analysis(self, molecular_features):
        """
        Detailed analysis for interpretability
        """
        with torch.no_grad():  # Disable gradients for analysis
            attention_weights, attented_features = self.forward(molecular_features)

        analysis = {
            'attention_distribution': {
                'weights': attention_weights.detach().cpu().numpy(),
                'entropy': -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=1).detach().cpu().numpy(),
                'max_attention': attention_weights.max(dim=1)[0].detach().cpu().numpy(),
                'top_target': attention_weights.argmax(dim=1).detach().cpu().numpy()
            },
            'quantum_parameters': {
                f'target_{i}': self.quantum_attention.quantum_params[i].detach().cpu().numpy()
                for i in range(self.n_targets)
            }
        }

        return analysis





