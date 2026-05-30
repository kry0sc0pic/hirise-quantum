"""
Evaluation pipeline for HiRISE terrain retrieval models.

Metrics:
  - MAP@10 (micro, primary retrieval metric)
  - Macro-MAP@10 (per-class average — corrects for 83.6% "other" class bias)
  - Recall@1, Recall@5
  - Precision@1, Precision@5, Precision@10
  - Confusion matrix on hard pairs (bright_dune/dark_dune, spider/swiss_cheese)

Plots:
  - UMAP of learned embeddings, colored by terrain class
  - Gradient norm history over training epochs (quantum models only)
  - Hard-pair retrieval confusion matrix heatmap

Usage (full dataset):
    python -m eval.eval --checkpoint checkpoints/best_quantum.pt --data_root data/hirise

Usage (held-out test split — recommended):
    python -m eval.eval --checkpoint checkpoints/best_quantum.pt \\
        --data_root data/hirise --split_test data/splits/test.json
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

from src.dataset import HiRISEPairDataset, build_class_index, load_split
from src.encoder import CNNEncoder
from src.model import QuantumTerrainSimilarity
from src.quantum_head import N_LAYERS, N_QUBITS


def _make_model(model_type: str, n_qubits: int, n_layers: int) -> torch.nn.Module:
    if model_type == "quantum":
        return QuantumTerrainSimilarity(n_qubits=n_qubits, n_layers=n_layers)
    elif model_type == "classical":
        from baselines.classical_head import ClassicalTerrainSimilarity
        return ClassicalTerrainSimilarity(n_qubits=n_qubits)
    elif model_type == "sliq":
        from baselines.sliq_baseline import SLIQTerrainSimilarity
        return SLIQTerrainSimilarity(n_qubits=n_qubits, n_layers=n_layers)
    elif model_type == "resnet":
        from baselines.resnet_baseline import ResNetTerrainSimilarity
        return ResNetTerrainSimilarity(n_qubits=n_qubits, pretrained=False)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. Choose quantum | classical | sliq | resnet.")


# ── Embedding extraction ──────────────────────────────────────────────────────

class _ImageDataset(Dataset):
    """Flat list of images for embedding extraction.

    Loads from a split JSON (recommended) or scans data_root directly.
    """

    def __init__(self, root: str, transform=None, split_file: Optional[str] = None):
        from src.dataset import _angle_encoding_transform
        self.transform = transform or _angle_encoding_transform()
        self.class_index = load_split(split_file) if split_file else build_class_index(root)
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
    split_file: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
        embeddings: (N, n_qubits) float32
        labels:     (N,)           int
        class_names: list of class name strings (sorted alphabetically)

    Pass split_file (path to test.json from make_splits.py) to evaluate
    only on the held-out test set. Without split_file, all images in data_root
    are used (includes training data — do NOT use for final reported metrics).
    """
    ds = _ImageDataset(root, split_file=split_file)
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


def _ap_at_k(row_sim: np.ndarray, query_label: int, labels: np.ndarray, k: int) -> float:
    """
    Average Precision @ K for a single query.

    AP@K = sum_r [P(r) * rel(r)] / sum_r rel(r)
    where the sum is over the top-K retrieved items and P(r) is precision at rank r.
    This is the uninterpolated AP definition from Manning et al. (IR textbook, §8.4).
    Queries with zero relevant items in top-K receive AP=0.
    """
    top_k = np.argsort(row_sim)[::-1][:k]
    rel = (labels[top_k] == query_label).astype(float)
    if rel.sum() == 0:
        return 0.0
    precisions = np.cumsum(rel) / (np.arange(k) + 1)
    return float((precisions * rel).sum() / rel.sum())


def mean_average_precision(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> float:
    """
    Micro-MAP@K: AP per query, averaged over ALL queries regardless of class.

    This is dominated by the majority class ("other" = 83.6% of HiRISE v3).
    Use macro_map_at_k() for a class-balanced metric.
    """
    sim = _cosine_similarity_matrix(embeddings)
    N = len(labels)
    ap_scores = []
    for i in range(N):
        row = sim[i].copy()
        row[i] = -np.inf
        ap_scores.append(_ap_at_k(row, labels[i], labels, k))
    return float(np.mean(ap_scores))


def macro_map_at_k(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> float:
    """
    Macro-MAP@K: compute AP per class (average over queries of that class),
    then average the per-class APs.

    Corrects for the 83.6% "other" majority class in HiRISE v3; all eight
    terrain classes contribute equally to the final number.
    """
    sim = _cosine_similarity_matrix(embeddings)
    classes = np.unique(labels)
    class_aps = []
    for cls in classes:
        query_idxs = np.where(labels == cls)[0]
        aps = []
        for i in query_idxs:
            row = sim[i].copy()
            row[i] = -np.inf
            aps.append(_ap_at_k(row, cls, labels, k))
        class_aps.append(float(np.mean(aps)))
    return float(np.mean(class_aps))


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
    hard_pairs: Optional[List[Tuple[str, str]]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Retrieval confusion matrix restricted to hard-pair classes."""
    if hard_pairs is None:
        hard_pairs = [
            ("bright_dune", "dark_dune"),
            ("spider", "swiss_cheese"),
        ]

    name_to_idx = {n: i for i, n in enumerate(class_names)}
    relevant: List[int] = []
    for a, b in hard_pairs:
        if a in name_to_idx:
            relevant.append(name_to_idx[a])
        if b in name_to_idx:
            relevant.append(name_to_idx[b])
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
    model_type: Optional[str] = None,
    device_str: str = "cpu",
    split_test: Optional[str] = None,
) -> Dict[str, float]:
    """
    Evaluate a trained model on the retrieval task.

    Args:
        checkpoint_path: path to .pt checkpoint saved by train.py
        data_root:       root directory of HiRISE data (ImageFolder layout)
        output_dir:      where to write metrics CSV, UMAP, and confusion plots
        model_type:      'quantum' | 'classical' | 'sliq' | 'resnet';
                         inferred from checkpoint if None
        device_str:      'cpu' or 'cuda:N'
        split_test:      path to test.json from make_splits.py; strongly
                         recommended to avoid evaluating on training images
    """
    if split_test is None:
        print(
            "WARNING: split_test not provided — evaluating on full data_root, "
            "which includes training images. Use --split_test data/splits/test.json "
            "for valid held-out evaluation."
        )
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_qubits = ckpt.get("n_qubits", N_QUBITS)
    n_layers = ckpt.get("n_layers", N_LAYERS)
    model_type = model_type or ckpt.get("model_type", "quantum")

    model = _make_model(model_type, n_qubits, n_layers)
    model.load_state_dict(ckpt["model_state"])

    embeddings, labels, class_names = compute_embeddings(
        model.encoder, data_root, device=device, split_file=split_test
    )

    map10       = mean_average_precision(embeddings, labels, k=10)
    macro_map10 = macro_map_at_k(embeddings, labels, k=10)
    r1  = recall_at_k(embeddings, labels, k=1)
    r5  = recall_at_k(embeddings, labels, k=5)
    p1  = precision_at_k(embeddings, labels, k=1)
    p5  = precision_at_k(embeddings, labels, k=5)
    p10 = precision_at_k(embeddings, labels, k=10)

    split_label = "(test split)" if split_test else "(FULL DATASET — includes training images)"
    print(f"\n── Retrieval Metrics {split_label} ──────────────────")
    print(f"MAP@10 (micro)  : {map10:.4f}")
    print(f"MAP@10 (macro)  : {macro_map10:.4f}  ← class-balanced; use for paper")
    print(f"Recall@1        : {r1:.4f}")
    print(f"Recall@5        : {r5:.4f}")
    print(f"Precision@1     : {p1:.4f}")
    print(f"Precision@5     : {p5:.4f}")
    print(f"Precision@10    : {p10:.4f}")

    metrics = {
        "map10_micro": map10,
        "map10_macro": macro_map10,
        "recall@1":   r1,
        "recall@5":   r5,
        "precision@1": p1,
        "precision@5": p5,
        "precision@10": p10,
    }
    metrics_csv = os.path.join(output_dir, f"metrics_{model_type}.csv")
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Metrics CSV saved to {metrics_csv}")

    plot_umap(
        embeddings, labels, class_names,
        title=f"{model_type.upper()} Model — UMAP Terrain Embeddings",
        save_path=os.path.join(output_dir, f"umap_{model_type}.png"),
    )

    if "q_grad_history" in ckpt and ckpt["q_grad_history"]:
        plot_gradient_norms(
            ckpt["q_grad_history"],
            save_path=os.path.join(output_dir, f"grad_norms_{model_type}.png"),
        )

    plot_hard_pair_confusion(
        embeddings, labels, class_names,
        save_path=os.path.join(output_dir, f"confusion_hard_pairs_{model_type}.png"),
    )

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate a HiRISE retrieval model on MAP@10, Recall@K, and plots."
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="data/hirise")
    p.add_argument("--output_dir", default="eval_outputs")
    p.add_argument(
        "--model_type", default=None,
        choices=["quantum", "classical", "sliq", "resnet"],
        help="Override model type (default: read from checkpoint)",
    )
    p.add_argument(
        "--split_test", default=None,
        help="Path to test.json from make_splits.py (strongly recommended)",
    )
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    evaluate_model(
        args.checkpoint, args.data_root, args.output_dir,
        args.model_type, args.device, args.split_test,
    )
