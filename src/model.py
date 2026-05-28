from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .encoder import CNNEncoder
from .quantum_head import N_LAYERS, N_QUBITS, QuantumSimilarityHead


class QuantumTerrainSimilarity(nn.Module):
    """
    Full quantum metric learning model for HiRISE terrain similarity.

    Encoder (shared weights) → quantum similarity head (entangled-pair PQC).
    Trained end-to-end with triplet loss.
    """

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS):
        super().__init__()
        self.encoder = CNNEncoder(n_qubits=n_qubits)
        self.quantum_head = QuantumSimilarityHead(n_qubits=n_qubits, n_layers=n_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embedding for a batch of images."""
        return self.encoder(x)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: Optional[torch.Tensor] = None,
        negative: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Training mode: pass anchor, positive, negative → (sim_ap, sim_an).
        Inference mode: pass only anchor → embedding tensor.
        """
        if positive is None:
            return self.encoder(anchor)

        e_a = self.encoder(anchor)
        e_p = self.encoder(positive)
        e_n = self.encoder(negative)

        sim_ap = self.quantum_head(e_a, e_p)
        sim_an = self.quantum_head(e_a, e_n)
        return sim_ap, sim_an

    def similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Pairwise similarity score ∈ (0, 1) for two image batches."""
        return self.quantum_head(self.encoder(x1), self.encoder(x2))
