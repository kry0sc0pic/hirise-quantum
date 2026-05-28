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

    Encoding: interleaved RX(π·eₐ[i]) + RY(π·e_b[i]) on wire i.
    This is the key design choice — both embeddings are encoded on the SAME
    qubits via orthogonal rotation axes, creating entanglement between them.
    Contrast with sequential encoding used in SLIQ (arXiv:2309.15259), where
    the two embeddings are encoded independently.

    Measurements: local PauliZ on each wire (not a single global observable).
    Local cost functions provably mitigate barren plateaus (Cerezo et al., 2021,
    Nature Communications 12:6961).
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
