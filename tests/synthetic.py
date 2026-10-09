"""Tiny synthetic stand-ins for the Kaggle and figshare datasets (pipeline checks only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

CLASS_CENTRE = {"glioma": (20, 20), "meningioma": (44, 20), "pituitary": (32, 44)}


def brain(rng, label: str | None, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Grey ellipse 'brain' with a bright blob whose position depends on the class."""
    yy, xx = np.mgrid[:size, :size]
    img = 40 + 120 * (((yy - size / 2) / (size * 0.42)) ** 2 + ((xx - size / 2) / (size * 0.36)) ** 2 < 1)
    img = img + rng.normal(0, 18, (size, size))
    mask = np.zeros((size, size), np.uint8)
    if label is not None:
        cy, cx = CLASS_CENTRE[label]
        cy, cx = cy + rng.integers(-3, 4), cx + rng.integers(-3, 4)
        blob = (yy - cy) ** 2 + (xx - cx) ** 2 < 36
        img[blob] = 245
        mask[blob] = 255
    return np.clip(img, 0, 255).astype(np.uint8), mask


def make_figshare(root: Path, n_patients: int = 18, slices: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    rows, idx = [], 0
    labels = list(CLASS_CENTRE)
    for p in range(n_patients):
        label = labels[p % 3]
        for _ in range(slices):
            idx += 1
            img, mask = brain(rng, label)
            Image.fromarray(img).save(root / "images" / f"{idx}.png")
            Image.fromarray(mask).save(root / "masks" / f"{idx}.png")
            rows.append({"idx": idx, "pid": f"P{p:03d}", "label": label, "height": 64, "width": 64,
                         "mask_fraction": float((mask > 0).mean()),
                         "image_path": f"images/{idx}.png", "mask_path": f"masks/{idx}.png"})
    meta = pd.DataFrame(rows)
    meta.to_csv(root / "metadata.csv", index=False)
    return meta


def make_kaggle(root: Path, fig_root: Path, fig: pd.DataFrame, seed: int = 1) -> None:
    """Kaggle-like folders: tumour images are re-encoded figshare slices (some copied into both
    Training and Testing); no-tumour images come from a different 'source' (other size, RGB)."""
    rng = np.random.default_rng(seed)
    for split in ["Training", "Testing"]:
        for c in ["glioma", "meningioma", "notumor", "pituitary"]:
            (root / split / c).mkdir(parents=True, exist_ok=True)
    for i, r in fig.iterrows():
        img = Image.open(fig_root / r.image_path).convert("L")
        split = "Testing" if i % 4 == 0 else "Training"
        img.save(root / split / r.label / f"{split[:2]}-{r.idx}.jpg", quality=90)
        if i % 8 == 0:                                       # duplicate copy leaking into the other split
            other = "Training" if split == "Testing" else "Testing"
            img.resize((60, 60)).resize((64, 64)).save(root / other / r.label / f"dup-{r.idx}.jpg", quality=80)
    for k in range(24):
        img, _ = brain(rng, None, size=48)
        split = "Testing" if k % 4 == 0 else "Training"
        Image.fromarray(img).convert("RGB").save(root / split / "notumor" / f"no-{k}.jpg", quality=90)
