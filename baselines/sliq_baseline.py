"""
SLIQ-style baseline (ablation A2 — axis-choice comparison).

Ablation A2 tests whether the axis choice in the encoding layer matters.
This baseline uses RZ(π·eₐ[i]) then RX(π·e_b[i]) on each wire — matching
the two-block encoding convention from SLIQ (arXiv:2309.15259).

The key technical asymmetry with our model (src/quantum_head.py):
    SLIQ:  RZ(α)|0⟩ = e^{-iα/2}|0⟩  — differs from |0⟩ only by a global
           phase, so ⟨Z_i⟩ is unaffected by α before any entangling gate.
           The first encoding block (RZ on eₐ) is effectively a no-op
           for the Z-measurement observables used throughout.
    Ours:  RX(α)|0⟩ = cos(α/2)|0⟩ - i sin(α/2)|1⟩ — genuinely rotates
           the state, so eₐ does contribute to the pre-ansatz state.

This is NOT a "joint vs sequential" distinction — both schemes apply two
rotations per wire and produce a joint function of (eₐ, e_b). The difference
is the choice of rotation axes and the resulting gate degeneracy.

Keeping the ansatz (StronglyEntanglingLayers, depth L) and measurements
(local PauliZ) identical isolates the axis-choice effect for ablation A2.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import pennylane as qml
import torch
import torch.nn as nn

from src.encoder import CNNEncoder
from src.quantum_head import N_LAYERS, N_QUBITS


def build_sliq_layer(n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS) -> qml.qnn.TorchLayer:
    """
    Sequential encoding: encode eₐ fully, then e_b fully (no interleaving).
    Ablation A2 — swap this layer into QuantumSimilarityHead to test whether
    the interleaved encoding in the main model is doing useful work.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        e_a = inputs[..., :n_qubits]
        e_b = inputs[..., n_qubits:]

        # Sequential encoding — eₐ first, e_b second (same wires)
        for i in range(n_qubits):
            qml.RZ(math.pi * e_a[..., i], wires=i)
        for i in range(n_qubits):
            qml.RX(math.pi * e_b[..., i], wires=i)

        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class SLIQSimilarityHead(nn.Module):
    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS):
        super().__init__()
        self.n_qubits = n_qubits
        self.qlayer = build_sliq_layer(n_qubits, n_layers)
        self.post = nn.Sequential(nn.Linear(n_qubits, 1), nn.Sigmoid())

    def forward(self, e_a: torch.Tensor, e_b: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([e_a, e_b], dim=-1)
        return self.post(self.qlayer(inputs)).squeeze(-1)


class SLIQTerrainSimilarity(nn.Module):
    """SLIQ-style model: same encoder, sequential-encoding quantum head."""

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS):
        super().__init__()
        self.encoder = CNNEncoder(n_qubits=n_qubits)
        self.quantum_head = SLIQSimilarityHead(n_qubits=n_qubits, n_layers=n_layers)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: Optional[torch.Tensor] = None,
        negative: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if positive is None:
            return self.encoder(anchor)
        e_a = self.encoder(anchor)
        e_p = self.encoder(positive)
        e_n = self.encoder(negative)
        return self.quantum_head(e_a, e_p), self.quantum_head(e_a, e_n)

    def similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.quantum_head(self.encoder(x1), self.encoder(x2))
