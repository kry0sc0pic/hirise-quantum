"""
Creates stratified 70/15/15 train/val/test splits for HiRISE data.

Each class is split independently so minority classes appear in all three sets.
Output: data/splits/train.json, val.json, test.json
Format:  {"class_name": ["/abs/path/to/img1.jpg", ...], ...}

Usage:
    python -m scripts.make_splits --data_root data/hirise
    python -m scripts.make_splits --data_root data/hirise --seed 42 --output_dir data/splits
"""

from __future__ import annotations

import argparse
import json
import os
import random

from src.dataset import build_class_index

SPLIT_RATIOS = (0.70, 0.15, 0.15)


def make_splits(
    data_root: str,
    output_dir: str = "data/splits",
    seed: int = 42,
    ratios: tuple = SPLIT_RATIOS,
) -> None:
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    class_index = build_class_index(data_root)
    if not class_index:
        raise ValueError(f"No class directories found under {data_root}")

    train_split: dict = {}
    val_split: dict = {}
    test_split: dict = {}

    print(f"Splitting {sum(len(v) for v in class_index.values())} images "
          f"across {len(class_index)} classes (seed={seed}, ratios={ratios})")

    for cls, paths in sorted(class_index.items()):
        shuffled = paths[:]
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = max(1, int(n * ratios[0]))
        n_val = max(1, int(n * ratios[1]))
        # Remainder goes to test; guard against empty test for tiny classes
        n_test = n - n_train - n_val
        if n_test < 1 and n >= 3:
            n_val -= 1
            n_test = 1
        train_split[cls] = shuffled[:n_train]
        val_split[cls] = shuffled[n_train:n_train + n_val]
        test_split[cls] = shuffled[n_train + n_val:]
        print(f"  {cls:20s}: {len(train_split[cls]):5d} train  "
              f"{len(val_split[cls]):4d} val  {len(test_split[cls]):4d} test")

    for name, split in [("train", train_split), ("val", val_split), ("test", test_split)]:
        out_path = os.path.join(output_dir, f"{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)
        total = sum(len(v) for v in split.values())
        print(f"\nWrote {name}.json  ({total} images) → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Create stratified 70/15/15 HiRISE splits.")
    p.add_argument("--data_root", default="data/hirise")
    p.add_argument("--output_dir", default="data/splits")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    make_splits(args.data_root, args.output_dir, args.seed)
