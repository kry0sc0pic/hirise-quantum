"""
Training script for quantum and SLIQ terrain similarity models.

Usage (recommended — with disk splits):
    python -m src.train --data_root data/hirise --epochs 50 \\
        --split_train data/splits/train.json --split_val data/splits/val.json

Usage (legacy — random split, not reproducible):
    python -m src.train --data_root data/hirise --epochs 50 --n_qubits 8

Key design choices:
- Separate Adam learning rates: 1e-3 for CNN, 1e-2 for PQC weights
  (quantum gradients are typically smaller in scale than classical gradients)
- Barren plateau monitor: if mean |grad| of PQC weights < 1e-5 for 5
  consecutive epochs, automatically switch to COBYLA for the quantum layer
- Checkpoint selection: proper encoder-embedding MAP@10 on val set every
  val_every epochs (default 5); triplet ranking proxy on all other epochs
- Seed pinning: set --seed for reproducible runs across the ablation grid
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
from PIL import Image as _PILImage
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import HiRISETripletDataset, _angle_encoding_transform, load_split
from .loss import triplet_loss
from .model import QuantumTerrainSimilarity

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD = True
except ImportError:
    SummaryWriter = None
    _TENSORBOARD = False

# ── Barren-plateau constants ──────────────────────────────────────────────────
_BP_THRESHOLD = 1e-5
_BP_PATIENCE = 5


# ── COBYLA fallback ───────────────────────────────────────────────────────────

def _cobyla_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    margin: float,
    max_iter: int = 40,
) -> float:
    """Single gradient-free epoch for the quantum layer using COBYLA."""
    from scipy.optimize import minimize

    q_params = list(model.quantum_head.qlayer.parameters())
    shapes = [p.shape for p in q_params]
    batches = [b for _, b in zip(range(8), loader)]

    def objective(x: np.ndarray) -> float:
        offset = 0
        with torch.no_grad():
            for p, shape in zip(q_params, shapes):
                n = p.numel()
                p.copy_(torch.tensor(x[offset:offset + n], dtype=p.dtype).reshape(shape).to(device))
                offset += n
        total = 0.0
        for batch in batches:
            a, p_img, n_img = [t.to(device) for t in batch]
            sim_ap, sim_an = model(a, p_img, n_img)
            total += triplet_loss(sim_ap, sim_an, margin=margin).item()
        return total / len(batches)

    x0 = np.concatenate([p.detach().cpu().numpy().flatten() for p in q_params])
    result = minimize(objective, x0, method="COBYLA", options={"maxiter": max_iter, "rhobeg": 0.05})

    offset = 0
    with torch.no_grad():
        for p, shape in zip(q_params, shapes):
            n = p.numel()
            p.copy_(torch.tensor(result.x[offset:offset + n], dtype=p.dtype).reshape(shape).to(device))
            offset += n
    return float(result.fun)


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _val_triplet_acc(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Fast proxy: fraction of val triplets where sim_ap > sim_an."""
    model.eval()
    n_correct = n_total = 0
    for anchor, pos, neg in loader:
        anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
        sim_ap, sim_an = model(anchor, pos, neg)
        n_correct += (sim_ap > sim_an).sum().item()
        n_total += anchor.size(0)
    model.train()
    return n_correct / max(n_total, 1)


@torch.no_grad()
def _compute_encoder_map10(
    model: torch.nn.Module,
    class_index: Dict[str, List[str]],
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """
    Proper MAP@10 via encoder embeddings + cosine similarity on val/test split.

    Uses the encoder only (no quantum head) — this is fast and is the metric
    reported at evaluation time, so it is the correct signal for checkpoint selection.
    """
    transform = _angle_encoding_transform()
    model.eval()

    classes = sorted(class_index.keys())
    samples: List[tuple] = [
        (path, cls_idx)
        for cls_idx, cls in enumerate(classes)
        for path in class_index[cls]
    ]

    all_emb, all_labels = [], []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        imgs = torch.stack([
            transform(_PILImage.open(p).convert("L")) for p, _ in batch
        ]).to(device)
        emb = model.encoder(imgs).cpu().numpy()
        all_emb.append(emb)
        all_labels.extend(lbl for _, lbl in batch)

    emb_arr = np.concatenate(all_emb, axis=0)
    lbl_arr = np.array(all_labels)

    norms = np.linalg.norm(emb_arr, axis=1, keepdims=True) + 1e-9
    normed = emb_arr / norms
    sim = normed @ normed.T

    k = 10
    N = len(lbl_arr)
    ap_scores = []
    for i in range(N):
        row = sim[i].copy()
        row[i] = -np.inf
        top_k = np.argsort(row)[::-1][:k]
        rel = (lbl_arr[top_k] == lbl_arr[i]).astype(float)
        if rel.sum() == 0:
            ap_scores.append(0.0)
        else:
            precisions = np.cumsum(rel) / (np.arange(k) + 1)
            ap_scores.append(float((precisions * rel).sum() / rel.sum()))

    model.train()
    return float(np.mean(ap_scores))


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    data_root: str,
    epochs: int = 50,
    batch_size: int = 16,
    n_qubits: int = 8,
    n_layers: int = 2,
    margin: float = 0.2,
    lr_cnn: float = 1e-3,
    lr_pqc: float = 1e-2,
    val_fraction: float = 0.15,
    checkpoint_dir: str = "checkpoints",
    metrics_csv: Optional[str] = None,
    tensorboard_log_dir: Optional[str] = None,
    wandb_project: Optional[str] = None,
    device_str: str = "cpu",
    model_type: str = "quantum",
    seed: int = 42,
    split_train: Optional[str] = None,
    split_val: Optional[str] = None,
    val_every: int = 5,
) -> None:
    """
    Train a quantum or SLIQ terrain similarity model.

    model_type: 'quantum' (interleaved-encoding PQC) or 'sliq' (sequential axis-choice baseline).
    For the classical head use baselines.classical_head; for ResNet use baselines.resnet_baseline.

    seed: random seed for torch, numpy, and Python random — pin for reproducible ablations.
    split_train / split_val: paths to JSON split files from scripts/make_splits.py.
    val_every: compute proper encoder-embedding MAP@10 on val set every N epochs;
               also used for checkpoint selection when a val split is provided.
    """
    assert n_qubits <= 12, "Hard cap: n_qubits > 12 requires 2^N statevector — memory unsafe."
    assert model_type in ("quantum", "sliq"), (
        f"model_type must be 'quantum' or 'sliq', got {model_type!r}. "
        "For classical head: python -m baselines.classical_head  "
        "For ResNet: python -m baselines.resnet_baseline"
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(device_str)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    val_class_index: Optional[Dict] = None

    if split_train and split_val:
        train_ds = HiRISETripletDataset(data_root, split_file=split_train)
        val_ds   = HiRISETripletDataset(data_root, split_file=split_val)
        val_class_index = load_split(split_val, data_root)
    else:
        full_ds = HiRISETripletDataset(data_root)
        n_val = max(1, int(len(full_ds) * val_fraction))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    if model_type == "quantum":
        model = QuantumTerrainSimilarity(n_qubits=n_qubits, n_layers=n_layers).to(device)
    else:
        from baselines.sliq_baseline import SLIQTerrainSimilarity
        model = SLIQTerrainSimilarity(n_qubits=n_qubits, n_layers=n_layers).to(device)

    cnn_params  = list(model.encoder.parameters())
    pqc_params  = list(model.quantum_head.qlayer.parameters())
    post_params = list(model.quantum_head.post.parameters())

    optimizer = optim.Adam([
        {"params": cnn_params,  "lr": lr_cnn},
        {"params": pqc_params,  "lr": lr_pqc},
        {"params": post_params, "lr": lr_cnn},
    ])

    # ── Metrics CSV ───────────────────────────────────────────────────────────
    metrics_csv = metrics_csv or os.path.join(checkpoint_dir, f"train_metrics_{model_type}.csv")
    os.makedirs(os.path.dirname(metrics_csv) or ".", exist_ok=True)
    csv_fields = ["epoch", "loss", "val_triplet_acc", "val_map10", "q_grad_norm", "cobyla"]
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    # ── TensorBoard / W&B ────────────────────────────────────────────────────
    tb_writer = None
    if tensorboard_log_dir:
        if _TENSORBOARD and SummaryWriter is not None:
            tb_writer = SummaryWriter(log_dir=tensorboard_log_dir)
        else:
            print("TensorBoard is not installed; skipping TensorBoard logging.")

    if _WANDB and wandb_project:
        wandb.init(
            project=wandb_project,
            config=dict(
                n_qubits=n_qubits, n_layers=n_layers, margin=margin,
                lr_cnn=lr_cnn, lr_pqc=lr_pqc, epochs=epochs,
                batch_size=batch_size, seed=seed, model_type=model_type,
            ),
        )

    # ── Training state ────────────────────────────────────────────────────────
    best_val = 0.0
    plateau_window: deque = deque(maxlen=_BP_PATIENCE)
    use_cobyla = False
    q_grad_history: List[float] = []

    n_train_samples = len(train_ds)
    n_val_samples   = len(val_ds)
    print(f"Seed {seed} | Training on {n_train_samples} triplets, validating on {n_val_samples}.")
    print(f"Quantum circuit: {n_qubits} qubits, depth {n_layers}  "
          f"({n_layers * n_qubits * 3} PQC params)")
    if val_class_index is None:
        print("NOTE: no split_val provided — checkpoint selection uses triplet-accuracy proxy.")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_q_grad = 0.0
        n_batches = 0

        for anchor, pos, neg in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

            if use_cobyla:
                optimizer.zero_grad()
                e_a = model.encoder(anchor)
                e_p = model.encoder(pos)
                e_n = model.encoder(neg)
                with torch.no_grad():
                    sim_ap = model.quantum_head(e_a, e_p)
                    sim_an = model.quantum_head(e_a, e_n)
                loss = triplet_loss(sim_ap, sim_an, margin=margin)
                loss.backward()
                for p in pqc_params:
                    if p.grad is not None:
                        p.grad.zero_()
                optimizer.step()
                epoch_loss += loss.item()
            else:
                optimizer.zero_grad()
                sim_ap, sim_an = model(anchor, pos, neg)
                loss = triplet_loss(sim_ap, sim_an, margin=margin)
                loss.backward()

                pqc_grads = [
                    p.grad.abs().mean().item()
                    for p in pqc_params
                    if p.grad is not None
                ]
                if pqc_grads:
                    epoch_q_grad += float(np.mean(pqc_grads))

                optimizer.step()
                epoch_loss += loss.item()

            n_batches += 1

        avg_loss   = epoch_loss   / max(n_batches, 1)
        avg_q_grad = epoch_q_grad / max(n_batches, 1)
        q_grad_history.append(avg_q_grad)

        # ── Barren plateau detection ───────────────────────────────────────
        plateau_window.append(avg_q_grad)
        if (
            len(plateau_window) == _BP_PATIENCE
            and all(g < _BP_THRESHOLD for g in plateau_window)
        ):
            if not use_cobyla:
                print(f"\n[Epoch {epoch}] Barren plateau detected "
                      f"(mean |grad_PQC| = {avg_q_grad:.2e} < {_BP_THRESHOLD:.0e} "
                      f"for {_BP_PATIENCE} epochs). Switching to COBYLA.")
            use_cobyla = True

        if use_cobyla:
            cobyla_loss = _cobyla_epoch(model, train_loader, device, margin)
            avg_loss = cobyla_loss

        # ── Validation ────────────────────────────────────────────────────
        val_triplet_acc = _val_triplet_acc(model, val_loader, device)

        # Proper MAP@10 every val_every epochs (or always if val_class_index available)
        val_map10 = 0.0
        if val_class_index is not None and (epoch % val_every == 0 or epoch == epochs):
            val_map10 = _compute_encoder_map10(model, val_class_index, device)

        print(
            f"Epoch {epoch:3d} | loss {avg_loss:.4f} | "
            f"triplet_acc {val_triplet_acc:.4f} | "
            f"val_map10 {val_map10:.4f} | "
            f"q_grad {avg_q_grad:.2e}"
            + (" [COBYLA]" if use_cobyla else "")
        )

        if _WANDB and wandb_project:
            wandb.log({
                "loss": avg_loss, "val_triplet_acc": val_triplet_acc,
                "val_map10": val_map10, "q_grad_norm": avg_q_grad,
                "cobyla": int(use_cobyla),
            })

        row = {
            "epoch": epoch, "loss": avg_loss,
            "val_triplet_acc": val_triplet_acc, "val_map10": val_map10,
            "q_grad_norm": avg_q_grad, "cobyla": int(use_cobyla),
        }
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", avg_loss, epoch)
            tb_writer.add_scalar("val/triplet_acc", val_triplet_acc, epoch)
            tb_writer.add_scalar("val/map10", val_map10, epoch)
            tb_writer.add_scalar("quantum/q_grad_norm", avg_q_grad, epoch)
            tb_writer.add_scalar("quantum/cobyla", int(use_cobyla), epoch)

        # ── Checkpoint — use proper MAP@10 when available ─────────────────
        # When val_class_index is set, checkpoint_metric = val_map10 (updated
        # every val_every epochs); between those epochs the proxy holds the
        # previous best so checkpointing only fires on MAP@10 improvements.
        checkpoint_metric = val_map10 if (val_class_index is not None and val_map10 > 0) \
            else val_triplet_acc
        if checkpoint_metric > best_val:
            best_val = checkpoint_metric
            ckpt_path = os.path.join(checkpoint_dir, f"best_{model_type}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_map10": val_map10,
                "val_triplet_acc": val_triplet_acc,
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "q_grad_history": q_grad_history,
                "model_type": model_type,
                "seed": seed,
            }, ckpt_path)

    print(f"\n[{model_type}] Training complete. Best checkpoint metric = {best_val:.4f}")
    print(f"Training metrics CSV saved to {metrics_csv}")
    if _WANDB and wandb_project:
        wandb.finish()
    if tb_writer is not None:
        tb_writer.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train QuantumTerrainSimilarity on HiRISE.")
    p.add_argument("--data_root", default="data/hirise")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_qubits", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--lr_cnn", type=float, default=1e-3)
    p.add_argument("--lr_pqc", type=float, default=1e-2)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--metrics_csv", default=None)
    p.add_argument("--tensorboard_log_dir", default=None)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--model_type", default="quantum", choices=["quantum", "sliq"])
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible ablation runs.")
    p.add_argument("--split_train", default=None,
                   help="Path to train.json from scripts/make_splits.py (recommended).")
    p.add_argument("--split_val", default=None,
                   help="Path to val.json from scripts/make_splits.py (recommended).")
    p.add_argument("--val_every", type=int, default=5,
                   help="Compute proper encoder MAP@10 on val set every N epochs.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        margin=args.margin,
        lr_cnn=args.lr_cnn,
        lr_pqc=args.lr_pqc,
        checkpoint_dir=args.checkpoint_dir,
        metrics_csv=args.metrics_csv,
        tensorboard_log_dir=args.tensorboard_log_dir,
        wandb_project=args.wandb_project,
        device_str=args.device,
        model_type=args.model_type,
        seed=args.seed,
        split_train=args.split_train,
        split_val=args.split_val,
        val_every=args.val_every,
    )
