"""
Evaluation pipeline for QuantumTerrainSimilarity.

Metrics:
  - MAP@10 (primary retrieval metric)
  - Recall@1, Recall@5
  - Precision@1, Precision@5, Precision@10
  - Confusion matrix on hard pairs (bright_dunes/dunes, spiders/swiss_cheese)

Plots:
  - UMAP of learned embeddings, colored by terrain class
  - Gradient norm history over training epochs
  - Hard-pair confusion matrix heatmap

Usage:
    python -m eval.eval --checkpoint checkpoints/best_quantum.pt --data_root data/hirise
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.dataset import HiRISEPairDataset, build_class_index
from src.encoder import CNNEncoder
from src.model import QuantumTerrainSimilarity
from src.quantum_head import N_LAYERS, N_QUBITS


# ── Embedding extraction ──────────────────────────────────────────────────────

class _ImageDataset(Dataset):
    """Flat list of all images in a class-folder directory for embedding extraction."""

    def __init__(self, root: str, transform=None):
        from src.dataset import _angle_encoding_transform
        self.transform = transform or _angle_encoding_transform()
        self.class_index = build_class_index(root)
        self.classes = sorted(self.class_index.keys())
        self.samples: List[Tuple[str, int]] = [
            (path, cls_idx)
            for cls_idx, cls in enumerate(self.classes)
            for path in self.class_index[cls]
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        from PIL import Image
        path, label = self.samples[idx]
        img = self.transform(Image.open(path).convert("L"))
        return img, label


@torch.no_grad()
def compute_embeddings(
    encoder: torch.nn.Module,
    root: str,
    batch_size: int = 64,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
        embeddings: (N, n_qubits) float32
        labels:     (N,)           int
        class_names: list of class name strings
    """
    ds = _ImageDataset(root)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    encoder.eval().to(device)

    all_emb, all_lbl = [], []
    for imgs, labels in tqdm(loader, desc="Extracting embeddings"):
        emb = encoder(imgs.to(device)).cpu().numpy()
        all_emb.append(emb)
        all_lbl.append(labels.numpy())

    return np.concatenate(all_emb), np.concatenate(all_lbl), ds.classes


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def _cosine_similarity_matrix(emb: np.ndarray) -> np.ndarray:
    """(N, N) pairwise cosine similarity."""
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    normed = emb / norms
    return normed @ normed.T


def mean_average_precision(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> float:
    """MAP@K using cosine similarity on embeddings."""
    sim = _cosine_similarity_matrix(embeddings)
    N = len(labels)
    ap_scores = []
    for i in range(N):
        row = sim[i].copy()
        row[i] = -np.inf  # exclude self
        top_k = np.argsort(row)[::-1][:k]
        rel = (labels[top_k] == labels[i]).astype(float)
        if rel.sum() == 0:
            ap_scores.append(0.0)
            continue
        precisions = np.cumsum(rel) / (np.arange(k) + 1)
        ap_scores.append((precisions * rel).sum() / rel.sum())
    return float(np.mean(ap_scores))


def recall_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> float:
    """Recall@K: fraction of queries where at least one same-class result is in top-K."""
    sim = _cosine_similarity_matrix(embeddings)
    N = len(labels)
    hits = 0
    for i in range(N):
        row = sim[i].copy()
        row[i] = -np.inf
        top_k = np.argsort(row)[::-1][:k]
        if (labels[top_k] == labels[i]).any():
            hits += 1
    return hits / N


def precision_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> float:
    """Precision@K: average same-class fraction among each query's top-K results."""
    sim = _cosine_similarity_matrix(embeddings)
    N = len(labels)
    scores = []
    for i in range(N):
        row = sim[i].copy()
        row[i] = -np.inf
        top_k = np.argsort(row)[::-1][:k]
        scores.append(float((labels[top_k] == labels[i]).mean()))
    return float(np.mean(scores))


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    title: str = "UMAP of Terrain Embeddings",
    save_path: Optional[str] = None,
) -> None:
    try:
        from umap import UMAP
    except ImportError:
        print("umap-learn not installed — skipping UMAP plot.")
        return

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embeddings)

    palette = sns.color_palette("tab10", n_colors=len(class_names))
    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[palette[cls_idx]], label=cls_name, alpha=0.7, s=12, linewidths=0,
        )
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"UMAP saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_gradient_norms(
    q_grad_history: List[float],
    save_path: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(1, len(q_grad_history) + 1), q_grad_history, marker="o", ms=3)
    ax.axhline(1e-5, color="red", linestyle="--", label="Barren plateau threshold (1e-5)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean |∇ PQC weights|")
    ax.set_title("Quantum Layer Gradient Norm History")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Gradient norm plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_hard_pair_confusion(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    hard_pairs: List[Tuple[str, str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Retrieval confusion matrix restricted to hard-pair classes."""
    if hard_pairs is None:
        hard_pairs = [("bright_dunes", "dunes"), ("spiders", "swiss_cheese")]

    # Map name variants
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    relevant: List[int] = []
    for a, b in hard_pairs:
        for variant in (a, a.replace("_", " ")):
            if variant in name_to_idx:
                relevant.append(name_to_idx[variant])
        for variant in (b, b.replace("_", " ")):
            if variant in name_to_idx:
                relevant.append(name_to_idx[variant])
    relevant = sorted(set(relevant))
    if len(relevant) < 2:
        print("Hard-pair classes not found in dataset — skipping confusion matrix.")
        return

    mask = np.isin(labels, relevant)
    emb_sub = embeddings[mask]
    lbl_sub = labels[mask]

    sim = _cosine_similarity_matrix(emb_sub)
    sub_classes = [class_names[i] for i in relevant]
    sub_lbl_map = {old: new for new, old in enumerate(relevant)}
    relabeled = np.array([sub_lbl_map[l] for l in lbl_sub])

    n_cls = len(sub_classes)
    confusion = np.zeros((n_cls, n_cls), dtype=int)
    for i in range(len(emb_sub)):
        row = sim[i].copy()
        row[i] = -np.inf
        top1 = int(np.argmax(row))
        confusion[relabeled[i], relabeled[top1]] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion, annot=True, fmt="d", cmap="Blues",
        xticklabels=sub_classes, yticklabels=sub_classes, ax=ax,
    )
    ax.set_xlabel("Retrieved class (top-1)")
    ax.set_ylabel("Query class")
    ax.set_title("Hard-Pair Retrieval Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


# ── Full evaluation pipeline ──────────────────────────────────────────────────

def evaluate_model(
    checkpoint_path: str,
    data_root: str,
    output_dir: str = "eval_outputs",
    device_str: str = "cpu",
) -> Dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str)

    ckpt = torch.load(checkpoint_path, map_location=device)
    n_qubits = ckpt.get("n_qubits", N_QUBITS)
    n_layers = ckpt.get("n_layers", N_LAYERS)

    model = QuantumTerrainSimilarity(n_qubits=n_qubits, n_layers=n_layers)
    model.load_state_dict(ckpt["model_state"])

    embeddings, labels, class_names = compute_embeddings(
        model.encoder, data_root, device=device
    )

    map10 = mean_average_precision(embeddings, labels, k=10)
    r1 = recall_at_k(embeddings, labels, k=1)
    r5 = recall_at_k(embeddings, labels, k=5)
    p1 = precision_at_k(embeddings, labels, k=1)
    p5 = precision_at_k(embeddings, labels, k=5)
    p10 = precision_at_k(embeddings, labels, k=10)

    print(f"\n── Retrieval Metrics ──────────────────")
    print(f"MAP@10    : {map10:.4f}")
    print(f"Recall@1  : {r1:.4f}")
    print(f"Recall@5  : {r5:.4f}")
    print(f"Precision@1 : {p1:.4f}")
    print(f"Precision@5 : {p5:.4f}")
    print(f"Precision@10: {p10:.4f}")

    metrics = {
        "map10": map10,
        "recall@1": r1,
        "recall@5": r5,
        "precision@1": p1,
        "precision@5": p5,
        "precision@10": p10,
    }
    metrics_csv = os.path.join(output_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Metrics CSV saved to {metrics_csv}")

    plot_umap(
        embeddings, labels, class_names,
        title="Quantum Model — UMAP Terrain Embeddings",
        save_path=os.path.join(output_dir, "umap_quantum.png"),
    )

    if "q_grad_history" in ckpt and ckpt["q_grad_history"]:
        plot_gradient_norms(
            ckpt["q_grad_history"],
            save_path=os.path.join(output_dir, "grad_norms.png"),
        )

    plot_hard_pair_confusion(
        embeddings, labels, class_names,
        save_path=os.path.join(output_dir, "confusion_hard_pairs.png"),
    )

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="data/hirise")
    p.add_argument("--output_dir", default="eval_outputs")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    evaluate_model(args.checkpoint, args.data_root, args.output_dir, args.device)
