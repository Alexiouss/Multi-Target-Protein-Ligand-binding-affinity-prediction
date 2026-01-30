import streamlit as st
from application import (
    run_training,
    stop_training,
    is_training_running,
    run_testing,
    get_ui_training_status,
)

import time
from pathlib import Path




st.set_page_config(
    page_title="Quantum Multi-Target Attention",
    page_icon="🔮",
    layout="centered",
)

st.title("🔮 Quantum Multi-Target Attention UI")

st.markdown(
    "Use this panel to configure and launch the "
    "**QuantumMultiTargetAttention** experiment."
)

# Session state for remembering angle-only qubits
if "angle_n_qubits" not in st.session_state:
    st.session_state["angle_n_qubits"] = 8

# Two-column layout
col1, col2 = st.columns(2)


train_variant_label = st.radio(
    "Which model do you want to train?",
    options=["Quantum only", "Classical only"],
    index=0,
)

if train_variant_label == "Quantum only":
    train_mode = "quantum"
else:
    train_mode = "classical"

# LEFT: Quantum config
with col1:
    st.subheader("Quantum Configuration")

    if train_mode == "classical":
        # No quantum controls needed – classical only
        st.info(
            "Quantum configuration is not needed when training **only the classical model**. "
            "Default quantum settings will be used internally if required."
        )

        # Provide harmless defaults so run_training always gets valid values
        encoding_type = "angle"
        effective_encoding = "angle"
        reupload = False
        effective_reupload = False
        effective_n_qubits = 8

    else:
        # --- existing quantum UI, unchanged ---
        encoding_type = st.selectbox(
            "Encoding type",
            options=["angle", "amplitude"],
            index=0,
            help=(
                "Angle encoding: optional data re-uploading and configurable number of qubits.\n"
                "Amplitude encoding: fixed 4 qubits,using PCA (256→16) with optional data re-uploading."
            ),
            key="encoding_type",
        )

        # Reupload is allowed for both 'angle' and 'amplitude' (only irrelevant in classical mode)
        reupload = st.checkbox(
            "Enable data re-uploading",
            value=st.session_state.get("reupload", False),
            disabled=False,
            help=(
                "When enabled, the same encoded input is re-uploaded before each variational "
                "layer in the quantum circuit."
            ),
            key="reupload",
        )


        if encoding_type == "amplitude":
            slider_disabled = True
            slider_value = 4
        elif encoding_type == "angle" and reupload:
            slider_disabled = True
            slider_value = 4
        else:
            slider_disabled = False
            slider_value = st.session_state["angle_n_qubits"]

        n_qubits = st.slider(
            "Number of qubits",
            min_value=4,
            max_value=14,
            value=slider_value,
            step=2,
            disabled=slider_disabled,
        )

        if encoding_type == "angle" and not reupload:
            st.session_state["angle_n_qubits"] = n_qubits

        effective_encoding = encoding_type
        effective_reupload = reupload

        if effective_encoding == "amplitude":
            # Amplitude encoding: 8 qubits fixed, but re-uploading is allowed
            effective_n_qubits = 4
            st.info(
                "With **amplitude** encoding, the number of qubits is fixed to **4**. Using PCA to reduce the dimentionality from 256→16 "
                "You can optionally enable data re-uploading via the checkbox."
            )
        elif effective_encoding == "angle" and effective_reupload:
            effective_n_qubits = 4
            st.info(
                "With **angle** encoding and **data re-uploading enabled**, "
                "the number of qubits is fixed to **4**."
            )
        else:
            effective_n_qubits = n_qubits
            st.info(
                "With **angle** encoding and **no data re-uploading**, you can choose "
                "the number of qubits from the slider (4–14, even values)."
            )



# RIGHT: Model & training
with col2:
    st.subheader("Model & Training")

    model_type = st.selectbox(
        "Model type",
        options=["GCN", "GINE"],
        index=0,
        help="Choose which graph encoder / model variant to use.",
    )

    num_layers = st.slider(
        "Number of layers",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
    )

    epochs = st.slider(
        "Epochs",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
    )

    learning_rate = st.number_input(
        "Learning rate",
        min_value=0.0000,
        max_value=1.0,
        value=0.0010,
        step=0.00001,
        format="%.5f",
    )

    batch_size_options = [4, 8, 16, 32, 64, 128]
    batch_size = st.selectbox(
        "Batch size",
        options=batch_size_options,
        index=2,
    )

    st.markdown("#### Data")
    use_chembl = st.checkbox(
        "Use CSV dataset (ChEMBL-style) instead of synthetic data",
        value=st.session_state.get("use_chembl", False),
        help="If enabled, training will use the uploaded CSV and targets will be taken from its Protein column.",
        key="use_chembl",
    )

    csv_saved_path = None
    if use_chembl:
        uploaded = st.file_uploader(
            "Upload dataset CSV",
            type=["csv"],
            help="CSV must contain at least columns: SMILES (or canonical_smiles) and Protein and binding_affinity.",
        )
        if uploaded is not None:
            # Save under project_root/data/chembl so train.py can find it consistently
            project_root = Path(__file__).resolve().parent
            data_dir = project_root / "data" / "chembl"
            data_dir.mkdir(parents=True, exist_ok=True)
            csv_path = data_dir / "chembl_affinity_dataset.csv"
            csv_path.write_bytes(uploaded.getbuffer())
            csv_saved_path = str(csv_path)
            st.success(f"CSV saved to: {csv_path}")
        else:
            st.warning("Upload a CSV file to use the CSV dataset.")

st.markdown("---")
st.markdown("### Configuration summary")

summary_lines = [
    f"train_mode    = '{train_mode}'",
    f"model_type    = '{model_type}'",
]

if train_mode != "classical":
    summary_lines.extend([
        f"n_qubits      = {effective_n_qubits}",
        f"encoding_type = '{effective_encoding}'",
        f"reupload       = {effective_reupload}",
    ])

summary_lines.extend([
    f"use_chembl     = {use_chembl}",
    f"csv_path      = '{csv_saved_path}'" if use_chembl else "csv_path      = None",
    f"num_layers     = {num_layers}",
    f"epochs         = {epochs}",
    f"learning_rate  = {float(learning_rate):.5f}",
    f"batch_size     = {batch_size}",
])

st.code("\n".join(summary_lines), language="python")


run_col, stop_col = st.columns(2)

st.markdown("---")
st.subheader("Testing")

# Button is always clickable; we check state when it’s pressed
test_button = st.button(
    "🧪 Run Testing",
    use_container_width=True,
)

if test_button:
    if is_training_running():
        st.warning("Cannot run testing while training is still running.")
    else:
        with st.spinner("Running evaluation on all saved models..."):
            msg = run_testing(model_type=model_type, train_mode=train_mode)
        st.success(msg)




with run_col:
    run_button = st.button("🚀 Start training", use_container_width=True)

with stop_col:
    stop_button = st.button("🛑 Stop Training", use_container_width=True)

if run_button:
    try:
        with st.spinner("Starting training in background..."):
            msg = run_training(
                n_qubits=effective_n_qubits,
                encoding_type=effective_encoding,
                reupload=effective_reupload,
                num_layers=num_layers,
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                model_type=model_type,
                train_mode = train_mode,
                use_chembl=use_chembl,
                csv_path=csv_saved_path
            )
    except Exception as e:
        st.error(f"Training failed to start: {e}")
    else:
        st.success(msg)

if stop_button:
    msg = stop_training()
    st.warning(msg)

# --- Progress bar & status ---
status_placeholder = st.empty()
progress_bar = st.progress(0)

if is_training_running():
    # Track last seen step to update only when iteration changes
    last_epoch = None
    last_batch = None
    last_phase = None

    while is_training_running():
        status = get_ui_training_status()
        total_epochs = status.get("total_epochs", 0) or 0
        epoch = status.get("epoch", 0) or 0
        batch = status.get("batch", 0) or 0
        total_batches = status.get("total_batches", 0) or 0
        phase = status.get("phase", "idle")
        model_name = status.get("model_name", "N/A")
        iters_per_sec = status.get("iters_per_sec", 0.0) or 0.0

        # progress fraction *within current epoch*
        frac = 0.0
        if total_batches > 0:
            frac = batch / total_batches 
            frac = max(0.0, min(1.0, frac))


        # only update UI when we detect a new iteration (epoch/batch/phase changed)
        if (epoch, batch, phase) != (last_epoch, last_batch, last_phase):
            progress_bar.progress(int(frac * 100))

            if iters_per_sec > 0:
                sec_per_it = 1.0 / iters_per_sec
                speed_str = f"{iters_per_sec:.2f} it/s ({sec_per_it:.3f} s/it)"
            else:
                speed_str = "estimating speed..."

            status_text = (
                f"**Model:** {model_name} | "
                f"**Phase:** {phase} | "
                f"Epoch {epoch}/{total_epochs} | "
                f"Batch {batch}/{total_batches} | "
                f"{speed_str}"
            )
            status_placeholder.markdown(status_text)

            last_epoch, last_batch, last_phase = epoch, batch, phase

        # small sleep to avoid hammering the CPU, but still very responsive
        time.sleep(0.05)

    # After loop ends, read final status once
    final_status = get_ui_training_status()
    progress_bar.progress(100)
    status_placeholder.markdown(
        f"✅ Training finished for **{final_status.get('model_name', 'N/A')}**."
    )
else:
    status_placeholder.info("✅ No active training.")
    progress_bar.progress(0)
