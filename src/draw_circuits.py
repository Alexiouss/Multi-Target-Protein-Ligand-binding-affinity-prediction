"""
Generate 4 Qiskit circuit-diagram PNGs.

Outputs:
  01_dense_angle.png
  02_dense_angle_reupload.png
  03_amplitude.png
  04_amplitude_reupload.png

Notes:
- Dense angle encoding: per qubit q uses RY(x[2q]) then RZ(x[2q+1]).
- Reuploading: re-encode the SAME x (or amplitude embedding) at every layer.
- Variational block per layer: RX/RY/RZ per qubit with layer-specific params.
- Entanglement:
    if layer even: chain CNOT (q -> q+1)
    else: ring CNOT (q -> (q+1) mod n)
    if layer > 0: CZ on pairs (0,1), (2,3), ...
- Measurements:
    Z on all qubits: direct measure in computational basis
    Y on first min(2,n_qubits): basis rotate with Sdg + H, then measure
  (Expectation values are estimated from repeated shots.)
"""

from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import ParameterVector, Gate
from qiskit.visualization import circuit_drawer


# -------------------------
# Helpers
# -------------------------

def entangle_(qc: QuantumCircuit, layer: int, n_qubits: int):
    # layer % 2 == 0: chain CNOT q -> q+1 for q=0..n-2
    if layer % 2 == 0:
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
    else:
        # ring CNOT q -> (q+1) mod n for q=0..n-1
        for q in range(n_qubits):
            qc.cx(q, (q + 1) % n_qubits)

    # if layer > 0: CZ on pairs (0,1), (2,3), ...
    if layer > 0:
        for q in range(0, n_qubits - 1, 2):
            qc.cz(q, q + 1)


def angle_dense_encode(qc: QuantumCircuit, features: ParameterVector, n_qubits: int):
    """
    Dense angle encoding (diagram form):
      For each qubit q:
        RY(features[2q])
        RZ(features[2q+1])

    In PennyLane you do angle = tanh(feature) * pi.
    For a circuit diagram, we keep it symbolic as x_k.
    """
    if len(features) < 2 * n_qubits:
        raise ValueError(f"Need at least {2*n_qubits} features, got {len(features)}")

    for q in range(n_qubits):
        qc.ry(features[2 * q], q)
        qc.rz(features[2 * q + 1], q)


def variational_block(qc: QuantumCircuit, thetas: ParameterVector, layer: int, n_qubits: int):
    """
    Per-layer params: RX, RY, RZ per qubit, packed as:
      base = layer * (n_qubits*3)
      p0 = base + qubit*3
    """
    base = layer * (n_qubits * 3)
    for q in range(n_qubits):
        p0 = base + q * 3
        qc.rx(thetas[p0], q)
        qc.ry(thetas[p0 + 1], q)
        qc.rz(thetas[p0 + 2], q)


def amplitude_embedding_placeholder(qc: QuantumCircuit, n_qubits: int):
    amp = Gate(name="AmpEnc", num_qubits=n_qubits, params=[])
    qc.append(amp, list(range(n_qubits)))


def add_real_measurements(qc: QuantumCircuit, n_qubits: int):
    """
    Implements measurement set using Qiskit's native measurement icon.

    PennyLane returns:
      - expval(Z_i) for all qubits i
      - expval(Y_i) for i in {0,1} (or fewer if n_qubits<2)

    Qiskit diagram equivalent:
      - Z basis: measure directly
      - Y basis: apply Sdg then H, then measure
    """
    n_y = min(2, n_qubits)
    creg = ClassicalRegister(n_qubits + n_y, "c")
    qc.add_register(creg)

    c_idx = 0

    # Z on all qubits
    for q in range(n_qubits):
        qc.measure(q, creg[c_idx])
        c_idx += 1

    # Y on first 1–2 qubits: Sdg + H then measure
    for q in range(n_y):
        qc.sdg(q)
        qc.h(q)
        qc.measure(q, creg[c_idx])
        c_idx += 1


# -------------------------
# Build the 4 circuits
# -------------------------

def build_dense_angle(n_qubits: int, n_layers: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="DenseAngle")

    # after PCA (conceptually) you use 2 features per qubit => 2*n_qubits
    x = ParameterVector("x", 2 * n_qubits)

    # theta params: n_layers * n_qubits * 3
    theta = ParameterVector("θ", n_layers * n_qubits * 3)

    # encode once
    angle_dense_encode(qc, x, n_qubits)
    qc.barrier()

    # variational layers + entangle
    for layer in range(n_layers):
        variational_block(qc, theta, layer, n_qubits)
        entangle_(qc, layer, n_qubits)
        qc.barrier()

    add_real_measurements(qc, n_qubits)
    return qc


def build_dense_angle_reupload(n_qubits: int, n_layers: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="DenseAngleReupload")

    x = ParameterVector("x", 2 * n_qubits)
    theta = ParameterVector("θ", n_layers * n_qubits * 3)

    for layer in range(n_layers):
        # re-encode SAME x every layer
        angle_dense_encode(qc, x, n_qubits)
        qc.barrier()

        variational_block(qc, theta, layer, n_qubits)
        entangle_(qc, layer, n_qubits)
        qc.barrier()

    add_real_measurements(qc, n_qubits)
    return qc


def build_amplitude(n_qubits: int, n_layers: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Amplitude")

    theta = ParameterVector("θ", n_layers * n_qubits * 3)

    # encode once (placeholder)
    amplitude_embedding_placeholder(qc, n_qubits)
    qc.barrier()

    for layer in range(n_layers):
        variational_block(qc, theta, layer, n_qubits)
        entangle_(qc, layer, n_qubits)
        qc.barrier()

    add_real_measurements(qc, n_qubits)
    return qc


def build_amplitude_reupload(n_qubits: int, n_layers: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="AmplitudeReupload")

    theta = ParameterVector("θ", n_layers * n_qubits * 3)

    for layer in range(n_layers):
        # re-encode SAME vector every layer (placeholder)
        amplitude_embedding_placeholder(qc, n_qubits)
        qc.barrier()

        variational_block(qc, theta, layer, n_qubits)
        entangle_(qc, layer, n_qubits)
        qc.barrier()

    add_real_measurements(qc, n_qubits)
    return qc


# -------------------------
# Render / save images
# -------------------------

def save_circuit_png(qc: QuantumCircuit, filename: str, fold: int = 28):
    circuit_drawer(qc, output="mpl", fold=fold, filename=filename)


if __name__ == "__main__":
    # Change these to match your thesis figure preference
    n_qubits = 4
    n_layers = 3

    c1 = build_dense_angle(n_qubits, n_layers)
    c2 = build_dense_angle_reupload(n_qubits, n_layers)
    c3 = build_amplitude(n_qubits, n_layers)
    c4 = build_amplitude_reupload(n_qubits, n_layers)

    save_circuit_png(c1, "01_dense_angle.png")
    save_circuit_png(c2, "02_dense_angle_reupload.png")
    save_circuit_png(c3, "03_amplitude.png")
    save_circuit_png(c4, "04_amplitude_reupload.png")

    print("Saved PNGs:")
    print("  01_dense_angle.png")
    print("  02_dense_angle_reupload.png")
    print("  03_amplitude.png")
    print("  04_amplitude_reupload.png")
