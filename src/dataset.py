import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Visually confusable class pairs — used for hard negative mining.
# bright_dune ↔ dark_dune: both are dune formations, differ mainly in albedo.
# spider ↔ swiss_cheese: both are CO₂-sublimation features in the south polar region.
CONFUSABLE_PAIRS: Dict[str, str] = {
    "bright_dune":  "dark_dune",
    "dark_dune":    "bright_dune",
    "spider":       "swiss_cheese",
    "swiss_cheese": "spider",
}

HARD_NEGATIVE_FRACTION = 0.5


def _angle_encoding_transform() -> transforms.Compose:
    # Standard grayscale normalization: maps pixel values [0, 1] → [-1, 1].
    # The CNN encoder's final L2-normalization (src/encoder.py) projects
    # its output onto the unit sphere, so circuit inputs e[i] ∈ [-1, 1].
    # Rotation gates then receive angles π·e[i] ∈ [-π, π].
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def build_class_index(root: str) -> Dict[str, List[str]]:
    """Scans ImageFolder-style directory → {class_name: [image_path, ...]}."""
    idx: Dict[str, List[str]] = {}
    for class_dir in sorted(Path(root).iterdir()):
        if not class_dir.is_dir():
            continue
        images = (
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.png"))
        )
        if images:
            idx[class_dir.name] = [str(p) for p in images]
    return idx


def load_split(split_file: str, data_root: Optional[str] = None) -> Dict[str, List[str]]:
    """Load a pre-computed split JSON and return absolute paths.

    Paths in the JSON are stored as POSIX-relative (forward-slash, relative to
    data_root) by make_splits.py.  Pass data_root to resolve them to absolute
    paths on the current OS — required for cross-platform portability.
    If data_root is None, paths are returned as-is (backward compat only).
    """
    with open(split_file, encoding="utf-8") as f:
        data = json.load(f)
    if data_root is None:
        return data
    root = Path(data_root).resolve()
    return {
        cls: [str(root / p) for p in paths]
        for cls, paths in data.items()
    }


class HiRISETripletDataset(Dataset):
    """
    Yields (anchor, positive, negative) triplets for metric learning.

    If split_file is provided, loads the class index from that JSON file
    (produced by scripts/make_splits.py) for reproducible partitioning.
    Otherwise, scans data_root for an ImageFolder-style layout.

    Hard negative mining: HARD_NEGATIVE_FRACTION of negatives are drawn from
    confusable class pairs (bright_dune↔dark_dune, spider↔swiss_cheese)
    instead of uniformly at random.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[transforms.Compose] = None,
        hard_negative_fraction: float = HARD_NEGATIVE_FRACTION,
        split_file: Optional[str] = None,
    ):
        self.hard_negative_fraction = hard_negative_fraction
        self.class_index = (
            load_split(split_file, root) if split_file else build_class_index(root)
        )
        if not self.class_index:
            raise ValueError(f"No class directories found under {root}")
        self.classes = sorted(self.class_index.keys())
        self.samples: List[Tuple[str, str]] = [
            (path, cls)
            for cls, paths in self.class_index.items()
            for path in paths
        ]
        self.transform = transform or _angle_encoding_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_path, anchor_class = self.samples[idx]
        positive_path = self._sample_positive(anchor_path, anchor_class)
        negative_path = self._sample_negative(anchor_class)

        anchor = self._load(anchor_path)
        positive = self._load(positive_path)
        negative = self._load(negative_path)
        return anchor, positive, negative

    def _sample_positive(self, exclude_path: str, cls: str) -> str:
        candidates = [p for p in self.class_index[cls] if p != exclude_path]
        return random.choice(candidates) if candidates else exclude_path

    def _sample_negative(self, anchor_class: str) -> str:
        confusable = CONFUSABLE_PAIRS.get(anchor_class)
        if (
            confusable is not None
            and confusable in self.class_index
            and random.random() < self.hard_negative_fraction
        ):
            return random.choice(self.class_index[confusable])
        neg_classes = [c for c in self.classes if c != anchor_class]
        neg_class = random.choice(neg_classes)
        return random.choice(self.class_index[neg_class])

    def _load(self, path: str) -> torch.Tensor:
        return self.transform(Image.open(path).convert("L"))


class HiRISEPairDataset(Dataset):
    """
    Flat (img_a, img_b, label) pairs for retrieval evaluation.
    label=1 same class, label=0 different class.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[transforms.Compose] = None,
        max_same_per_class: int = 150,
        max_diff_per_pair: int = 30,
        split_file: Optional[str] = None,
    ):
        class_index = load_split(split_file, root) if split_file else build_class_index(root)
        self.classes = sorted(class_index.keys())
        self.transform = transform or _angle_encoding_transform()
        self.pairs = self._build_pairs(class_index, max_same_per_class, max_diff_per_pair)

    def _build_pairs(
        self,
        class_index: Dict[str, List[str]],
        max_same: int,
        max_diff: int,
    ) -> List[Tuple[str, str, int]]:
        pairs: List[Tuple[str, str, int]] = []
        classes = sorted(class_index.keys())

        for cls in classes:
            imgs = class_index[cls]
            shuffled = imgs[:]
            random.shuffle(shuffled)
            for i in range(0, min(max_same * 2, len(shuffled) - 1), 2):
                pairs.append((shuffled[i], shuffled[i + 1], 1))

            for other_cls in classes:
                if other_cls == cls:
                    continue
                other_imgs = class_index[other_cls]
                n = min(max_diff, len(imgs), len(other_imgs))
                for a, b in zip(
                    random.sample(imgs, n), random.sample(other_imgs, n)
                ):
                    pairs.append((a, b, 0))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        a_path, b_path, label = self.pairs[idx]
        return self._load(a_path), self._load(b_path), label

    def _load(self, path: str) -> torch.Tensor:
        return self.transform(Image.open(path).convert("L"))
