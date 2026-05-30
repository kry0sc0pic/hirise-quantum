"""
Classical similarity head baseline.

Drop-in replacement for the quantum head: same encoder, same triplet loss,
same training loop — only the similarity scoring function differs.

Concatenates both embeddings → Linear(2*N, 1) + Sigmoid.
Used for ablation A1 (quantum vs. classical head).
"""

from __future__ import annotations

from typing import Optional, Tuple

import argparse
import csv
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import HiRISETripletDataset
from src.encoder import CNNEncoder
from src.loss import triplet_loss
from src.quantum_head import N_QUBITS


class ClassicalSimilarityHead(nn.Module):
    """Concat-then-classify similarity head (no quantum circuit)."""

    def __init__(self, n_qubits: int = N_QUBITS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_qubits * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, e_a: torch.Tensor, e_b: torch.Tensor) -> torch.Tensor:
        x = torch.cat([e_a, e_b], dim=-1)
        return self.net(x).squeeze(-1)


class ClassicalTerrainSimilarity(nn.Module):
    """Full classical baseline model."""

    def __init__(self, n_qubits: int = N_QUBITS):
        super().__init__()
        self.encoder = CNNEncoder(n_qubits=n_qubits)
        self.head = ClassicalSimilarityHead(n_qubits=n_qubits)

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
        return self.head(e_a, e_p), self.head(e_a, e_n)

    def similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x1), self.encoder(x2))


def train_classical(
    data_root: str = "data/hirise",
    epochs: int = 50,
    batch_size: int = 32,
    n_qubits: int = N_QUBITS,
    margin: float = 0.2,
    lr: float = 1e-3,
    val_fraction: float = 0.15,
    checkpoint_dir: str = "checkpoints",
    metrics_csv: Optional[str] = None,
    device_str: str = "cpu",
) -> None:
    device = torch.device(device_str)
    os.makedirs(checkpoint_dir, exist_ok=True)

    full_ds = HiRISETripletDataset(data_root)
    n_val = max(1, int(len(full_ds) * val_fraction))
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ClassicalTerrainSimilarity(n_qubits=n_qubits).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    metrics_csv = metrics_csv or os.path.join(checkpoint_dir, "train_metrics_classical.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=["epoch", "loss", "val_acc"]).writeheader()

    best_val = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for anchor, pos, neg in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            optimizer.zero_grad()
            sim_ap, sim_an = model(anchor, pos, neg)
            loss = triplet_loss(sim_ap, sim_an, margin=margin)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        n_correct = n_total = 0
        with torch.no_grad():
            for anchor, pos, neg in val_loader:
                anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
                sim_ap, sim_an = model(anchor, pos, neg)
                n_correct += (sim_ap > sim_an).sum().item()
                n_total += anchor.size(0)
        val_acc = n_correct / max(n_total, 1)
        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch:3d} | loss {avg_loss:.4f} | val_acc {val_acc:.4f}")

        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=["epoch", "loss", "val_acc"]).writerow(
                {"epoch": epoch, "loss": avg_loss, "val_acc": val_acc}
            )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "n_qubits": n_qubits,
                    "model_type": "classical",
                },
                os.path.join(checkpoint_dir, "best_classical.pt"),
            )

    print(f"[classical] Training complete. Best val_acc = {best_val:.4f}")
    print(f"Training metrics CSV saved to {metrics_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/hirise")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_qubits", type=int, default=N_QUBITS)
    p.add_argument("--metrics_csv", default=None)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    train_classical(
        args.data_root, args.epochs, args.batch_size, args.n_qubits,
        metrics_csv=args.metrics_csv, device_str=args.device,
    )
