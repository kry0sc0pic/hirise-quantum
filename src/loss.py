import torch
import torch.nn.functional as F


def triplet_loss(
    sim_ap: torch.Tensor,
    sim_an: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Triplet loss using similarity scores from the quantum head.

    d(a, b) = 1 - sim(a, b)   → 0 is identical, 1 is maximally different.
    Loss = mean(max(0, d(a,p) - d(a,n) + margin))

    Using similarity-based distance (not embedding-space L2) keeps the loss
    directly tied to what the quantum head is learning, not just the encoder.
    """
    d_ap = 1.0 - sim_ap
    d_an = 1.0 - sim_an
    return F.relu(d_ap - d_an + margin).mean()
