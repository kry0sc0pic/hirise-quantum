import math

import pennylane as qml
import torch
import torch.nn as nn

N_QUBITS = 8
N_LAYERS = 2


def build_entangled_pair_layer(
    n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS
) -> qml.qnn.TorchLayer:
    """
    Builds the core quantum similarity circuit as a PennyLane TorchLayer.

    Encoding: on wire i, apply RX(π·eₐ[i]) then RY(π·e_b[i]).
    The encoder's L2-normalization ensures e[i] ∈ [-1, 1], so rotation
    angles π·e[i] lie in [-π, π] — the natural domain of rotation gates.

    Axis-choice vs SLIQ (arXiv:2309.15259):
    SLIQ encodes eₐ with RZ rotations on all wires, then e_b with RX rotations
    on all wires (two separate encoding blocks). Because RZ|0⟩ = e^{-iθ/2}|0⟩
    differs from |0⟩ only by a global phase, RZ acts as a no-op on the
    initial |0⟩ state — the first SLIQ encoding block has no effect on
    the ⟨Z_i⟩ measurement values before the entangling layer. Our choice of
    RX then RY on the same wire avoids this degeneracy: RX|0⟩ rotates the
    state away from |0⟩, so both embeddings genuinely affect the pre-ansatz
    state on each wire. Ablation A2 isolates this axis-choice effect.

    Measurements: local PauliZ on each wire (not a single global observable).
    Cerezo et al. (2021, Nature Commun. 12:1791) prove that local observables
    have at-most polynomially vanishing gradients in shallow circuits, whereas
    global observables yield exponentially vanishing gradients. We adopt local
    measurements as a design recommendation from that result.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # TorchLayer passes the full batch (B, 2*n_qubits); slice the feature
        # dimension with [..., :] so batching works via PennyLane parameter broadcasting.
        e_a = inputs[..., :n_qubits]
        e_b = inputs[..., n_qubits:]

        for i in range(n_qubits):
            qml.RX(math.pi * e_a[..., i], wires=i)   # first embedding, X-axis
            qml.RY(math.pi * e_b[..., i], wires=i)   # second embedding, Y-axis

        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class QuantumSimilarityHead(nn.Module):
    """
    Wraps the entangled-pair PQC and maps its local-Z measurements → scalar similarity.

    Forward signature: (e_a, e_b) → similarity ∈ (0, 1)
    where e_a, e_b are L2-normalized encoder outputs, shape (B, n_qubits).
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qlayer = build_entangled_pair_layer(n_qubits, n_layers)
        self.post = nn.Sequential(
            nn.Linear(n_qubits, 1),
            nn.Sigmoid(),
        )

    def forward(self, e_a: torch.Tensor, e_b: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([e_a, e_b], dim=-1)   # (B, 2*n_qubits)
        q_out = self.qlayer(inputs)               # (B, n_qubits)
        return self.post(q_out).squeeze(-1)        # (B,)
