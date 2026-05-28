"""
Training script for QuantumTerrainSimilarity.

Usage:
    python -m src.train --data_root data/hirise --epochs 50 --n_qubits 8

Key design choices:
- Separate Adam learning rates: 1e-3 for CNN, 1e-2 for PQC weights
  (quantum gradients are typically smaller in scale than classical gradients)
- Barren plateau monitor: if mean |grad| of PQC weights < 1e-5 for 5 consecutive
  epochs, automatically switch to COBYLA (gradient-free) for the quantum layer
- Checkpoint saved whenever validation MAP@10 improves
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .dataset import HiRISETripletDataset
from .loss import triplet_loss
from .model import QuantumTerrainSimilarity

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False

# ── Barren-plateau constants ──────────────────────────────────────────────────
_BP_THRESHOLD = 1e-5   # mean |grad| below this → plateau
_BP_PATIENCE = 5       # consecutive epochs below threshold before switching


# ── COBYLA fallback ───────────────────────────────────────────────────────────

def _cobyla_epoch(
    model: QuantumTerrainSimilarity,
    loader: DataLoader,
    device: torch.device,
    margin: float,
    max_iter: int = 40,
) -> float:
    """
    Single gradient-free epoch for the quantum layer using COBYLA.
    Classical encoder parameters are NOT updated here.
    """
    from scipy.optimize import minimize

    q_params = list(model.quantum_head.qlayer.parameters())
    shapes = [p.shape for p in q_params]
    # Use at most 8 batches to keep COBYLA calls tractable
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

    # Apply best parameters
    offset = 0
    with torch.no_grad():
        for p, shape in zip(q_params, shapes):
            n = p.numel()
            p.copy_(torch.tensor(result.x[offset:offset + n], dtype=p.dtype).reshape(shape).to(device))
            offset += n
    return float(result.fun)


# ── Validation MAP@10 ─────────────────────────────────────────────────────────

@torch.no_grad()
def _val_map10(model: QuantumTerrainSimilarity, loader: DataLoader, device: torch.device) -> float:
    """Approximate MAP@10 on the validation triplet loader (uses sim_ap > sim_an as proxy)."""
    model.eval()
    n_correct = n_total = 0
    for anchor, pos, neg in loader:
        anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
        sim_ap, sim_an = model(anchor, pos, neg)
        n_correct += (sim_ap > sim_an).sum().item()
        n_total += anchor.size(0)
    model.train()
    return n_correct / max(n_total, 1)


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
    wandb_project: Optional[str] = None,
    device_str: str = "cpu",
) -> None:
    assert n_qubits <= 12, "Hard cap: n_qubits > 12 requires 2^N statevector — memory unsafe."

    device = torch.device(device_str)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    full_ds = HiRISETripletDataset(data_root)
    n_val = max(1, int(len(full_ds) * val_fraction))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = QuantumTerrainSimilarity(n_qubits=n_qubits, n_layers=n_layers).to(device)

    # Separate optimizer groups — quantum weights need a higher LR
    cnn_params = list(model.encoder.parameters())
    pqc_params = list(model.quantum_head.qlayer.parameters())
    post_params = list(model.quantum_head.post.parameters())

    optimizer = optim.Adam([
        {"params": cnn_params,  "lr": lr_cnn},
        {"params": pqc_params,  "lr": lr_pqc},
        {"params": post_params, "lr": lr_cnn},
    ])

    # ── W&B ───────────────────────────────────────────────────────────────────
    if _WANDB and wandb_project:
        wandb.init(
            project=wandb_project,
            config=dict(
                n_qubits=n_qubits, n_layers=n_layers, margin=margin,
                lr_cnn=lr_cnn, lr_pqc=lr_pqc, epochs=epochs, batch_size=batch_size,
            ),
        )

    # ── Training state ────────────────────────────────────────────────────────
    best_val = 0.0
    plateau_window: deque[float] = deque(maxlen=_BP_PATIENCE)
    use_cobyla = False
    q_grad_history: list[float] = []

    print(f"Training on {n_train} triplets, validating on {n_val}.")
    print(f"Quantum circuit: {n_qubits} qubits, depth {n_layers}  "
          f"({n_layers * n_qubits * 3} PQC params)")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_q_grad = 0.0
        n_batches = 0

        for anchor, pos, neg in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

            if use_cobyla:
                # COBYLA handles the quantum step; encoder still uses Adam
                optimizer.zero_grad()
                e_a = model.encoder(anchor)
                e_p = model.encoder(pos)
                e_n = model.encoder(neg)
                # Encoder-only pass to get classical gradients
                with torch.no_grad():
                    sim_ap = model.quantum_head(e_a, e_p)
                    sim_an = model.quantum_head(e_a, e_n)
                loss = triplet_loss(sim_ap, sim_an, margin=margin)
                loss.backward()
                # Zero quantum gradients — COBYLA handles them separately
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

                # Measure quantum gradient norm before stepping
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

        avg_loss = epoch_loss / max(n_batches, 1)
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
                      f"for {_BP_PATIENCE} epochs). Switching to COBYLA for quantum layer.")
            use_cobyla = True

        if use_cobyla:
            cobyla_loss = _cobyla_epoch(model, train_loader, device, margin)
            avg_loss = cobyla_loss  # overwrite with COBYLA loss estimate

        # ── Validation ────────────────────────────────────────────────────
        val_acc = _val_map10(model, val_loader, device)

        print(
            f"Epoch {epoch:3d} | loss {avg_loss:.4f} | "
            f"val_acc {val_acc:.4f} | "
            f"q_grad {avg_q_grad:.2e}"
            + (" [COBYLA]" if use_cobyla else "")
        )

        if _WANDB and wandb_project:
            wandb.log({
                "loss": avg_loss, "val_acc": val_acc,
                "q_grad_norm": avg_q_grad, "cobyla": int(use_cobyla),
            })

        # ── Checkpoint ────────────────────────────────────────────────────
        if val_acc > best_val:
            best_val = val_acc
            ckpt_path = os.path.join(checkpoint_dir, "best_quantum.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "q_grad_history": q_grad_history,
            }, ckpt_path)

    print(f"\nTraining complete. Best val_acc = {best_val:.4f}")
    if _WANDB and wandb_project:
        wandb.finish()


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
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--device", default="cpu")
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
        wandb_project=args.wandb_project,
        device_str=args.device,
    )
