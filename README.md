# Multi-Target Protein–Ligand Binding Affinity Prediction

This repository contains **classical and hybrid quantum–classical models** for **multi-target protein–ligand binding affinity prediction**, with a focus on **attention mechanisms** and their feasibility in the **NISQ era**.

The work is developed as part of a **diploma thesis** and is intended as a **research prototype**, not a production system.

---

## 🔬 Research Questions

This project investigates the following questions:

- **Can quantum attention mechanisms be meaningfully applied to multi-target binding affinity prediction under NISQ constraints?**
- **How much does attention matter** in a ligand-centric, multi-target setting?
- How do **quantum attention variants** compare to **classical attention** when all other components are kept identical?

The emphasis is on **architectural comparison, feasibility, and interpretability**, rather than leaderboard performance.

---

## 🧠 High-Level Architecture

### Ligand Representation
- SMILES strings are converted into molecular graphs.
- Graph encoders:
  - GCN
  - GINE
- Output: fixed-size molecular embeddings shared across all targets.

### Protein Representation
- Proteins are **not explicitly encoded** in the current version.
- Instead, **learned target embeddings** act as placeholders.
- **Explicit protein encoding** (e.g. residue-level representations) is planned as future work.

### Attention Mechanisms

Two attention variants are implemented:

#### Classical Attention
- Standard QKV attention
- Implemented using PyTorch’s attention primitives
- Serves as the baseline

#### Quantum Attention
- Parameterized Quantum Circuits (PQCs)
- Angle encoding and amplitude encoding
- Optional data re-uploading
- Per-target quantum circuits
- Fully differentiable via PennyLane

Both variants share:
- The same graph encoder
- The same prediction heads
- The same training and evaluation pipeline

---

## 🧩 Project Structure

```text
src/
 ├─ data_loader.py
 ├─ data_loader_chembl.py
 ├─ graph_encoder.py
 ├─ graph_encoder_gin.py
 ├─ quantum_attention_refactored.py
 ├─ multi_target_model_gcn_refactored.py
 ├─ multi_target_model_multiple_models_gine.py
 ├─ preprocess_dataset.py
 ├─ fit_pca_unified.py
 ├─ draw_circuits.py
 ├─ visualize_attention_space.py

scripts/
 ├─ train.py
 ├─ test.py
 ├─ application.py
 ├─ app_ui.py
 └─ analysis/
    ├─ quantum_training_diagnostics.py
    ├─ plot_quantum_correlations.py
    ├─ plot_bloch_vectors.py
    ├─ plot_bitstrings_probs.py
    ├─ interpret_target_ablation.py
    ├─ interpret_attention_memory.py
    ├─ analyze_pca_embeddings.py
    ├─ analyze_tsne_embeddings.py
    └─ run_all_analysis.py

data/
 └─ chembl/
    └─ chembl_affinity_dataset.csv

results/
