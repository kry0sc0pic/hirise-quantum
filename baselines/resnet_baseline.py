"""
ResNet50 strong classical baseline for HiRISE terrain retrieval.

Pre-trained ResNet50 backbone (ImageNet) with a lightweight projection head
and the same classical MLP similarity scorer as baselines/classical_head.py.
This is the "strong classical baseline" that grounds quantum comparisons:
a fine-tuned deep feature extractor represents the upper bound of what
classical retrieval achieves on the same data.

Architecture:
    ResNet50 (pretrained, last FC removed) → (B, 2048)
    → Linear(2048, N) → L2-normalize → (B, N)   [encoder]
    → concat(e_a, e_b) → Linear(2N, 64) + ReLU + Linear(64, 1) + Sigmoid

Usage (recommended):
    python -m baselines.resnet_baseline --data_root data/hirise --epochs 50 \\
        --split_train data/splits/train.json --split_val data/splits/val.json

For evaluation after training:
    python -m eval.eval --checkpoint checkpoints/best_resnet.pt \\
        --data_root data/hirise --split_test data/splits/test.json
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tv_models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import HiRISETripletDataset
from src.loss import triplet_loss
from src.quantum_head import N_QUBITS


class ResNetEncoder(nn.Module):
    """
    Pre-trained ResNet50 backbone with a linear projection head.

    Input:  (B, 1, 64, 64) grayscale images
    Output: (B, n_qubits) L2-normalized embeddings on the unit sphere

    The grayscale single channel is replicated to three channels before
    being passed to ResNet50 (the model expects RGB input).
    """

    def __init__(self, n_qubits: int = N_QUBITS, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet50(weights=weights)
        # Drop the final classification FC; keep everything up to global avg-pool
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 2048, 1, 1)
        self.proj = nn.Linear(2048, n_qubits)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, 3, 1, 1)           # (B, 1, H, W) → (B, 3, H, W)
        x = self.features(x).flatten(1)     # (B, 2048)
        x = self.proj(x)                    # (B, n_qubits)
        return F.normalize(x, p=2, dim=-1)  # unit sphere in R^n_qubits


class ResNetTerrainSimilarity(nn.Module):
    """ResNet50 encoder + classical MLP similarity head."""

    def __init__(self, n_qubits: int = N_QUBITS, pretrained: bool = True):
        super().__init__()
        self.encoder = ResNetEncoder(n_qubits=n_qubits, pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Linear(n_qubits * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

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
        sim_ap = self.head(torch.cat([e_a, e_p], dim=-1)).squeeze(-1)
        sim_an = self.head(torch.cat([e_a, e_n], dim=-1)).squeeze(-1)
        return sim_ap, sim_an

    def similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.encoder(x1), self.encoder(x2)], dim=-1)
        return self.head(x).squeeze(-1)


def train_resnet(
    data_root: str = "data/hirise",
    epochs: int = 50,
    batch_size: int = 32,
    n_qubits: int = N_QUBITS,
    margin: float = 0.2,
    lr_backbone: float = 1e-4,
    lr_head: float = 1e-3,
    val_fraction: float = 0.15,
    checkpoint_dir: str = "checkpoints",
    metrics_csv: Optional[str] = None,
    device_str: str = "cpu",
    seed: int = 42,
    split_train: Optional[str] = None,
    split_val: Optional[str] = None,
    pretrained: bool = True,
) -> None:
    """
    Fine-tune ResNet50 on HiRISE triplets with triplet loss.

    Uses differential learning rates: a lower LR (lr_backbone=1e-4) for the
    pre-trained ResNet weights and a higher LR (lr_head=1e-3) for the
    randomly-initialized projection head and similarity head.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(device_str)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if split_train and split_val:
        train_ds = HiRISETripletDataset(data_root, split_file=split_train)
        val_ds   = HiRISETripletDataset(data_root, split_file=split_val)
    else:
        full_ds = HiRISETripletDataset(data_root)
        n_val = max(1, int(len(full_ds) * val_fraction))
        train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = ResNetTerrainSimilarity(n_qubits=n_qubits, pretrained=pretrained).to(device)
    optimizer = optim.Adam([
        {"params": model.encoder.features.parameters(), "lr": lr_backbone},
        {"params": model.encoder.proj.parameters(),     "lr": lr_head},
        {"params": model.head.parameters(),             "lr": lr_head},
    ])

    metrics_csv = metrics_csv or os.path.join(checkpoint_dir, "train_metrics_resnet.csv")
    csv_fields = ["epoch", "loss", "val_acc"]
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

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
            csv.DictWriter(f, fieldnames=csv_fields).writerow(
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
                    "model_type": "resnet",
                    "seed": seed,
                    "pretrained": pretrained,
                },
                os.path.join(checkpoint_dir, "best_resnet.pt"),
            )

    print(f"[resnet] Training complete. Best val_acc = {best_val:.4f}")
    print(f"Training metrics CSV saved to {metrics_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fine-tune ResNet50 baseline on HiRISE.")
    p.add_argument("--data_root", default="data/hirise")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_qubits", type=int, default=N_QUBITS)
    p.add_argument("--lr_backbone", type=float, default=1e-4,
                   help="LR for pre-trained ResNet50 backbone (default lower than head LR).")
    p.add_argument("--lr_head", type=float, default=1e-3,
                   help="LR for projection head and similarity head.")
    p.add_argument("--metrics_csv", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_train", default=None,
                   help="Path to train.json from scripts/make_splits.py (recommended).")
    p.add_argument("--split_val", default=None,
                   help="Path to val.json from scripts/make_splits.py (recommended).")
    p.add_argument("--no_pretrained", action="store_true",
                   help="Initialize ResNet50 from scratch (ablation only).")
    args = p.parse_args()
    train_resnet(
        args.data_root, args.epochs, args.batch_size, args.n_qubits,
        lr_backbone=args.lr_backbone, lr_head=args.lr_head,
        metrics_csv=args.metrics_csv, device_str=args.device,
        seed=args.seed,
        split_train=args.split_train,
        split_val=args.split_val,
        pretrained=not args.no_pretrained,
    )
