"""Robustness: does accuracy survive realistic acquisition differences?

    python -m brainguard.robustness      # after training `effnet_patient`

Each fold's model re-scores ITS OWN held-out patients under perturbations that mimic
another scanner or export pipeline. Clean and perturbed predictions are paired image
by image, so the drop is tested with McNemar (see report.py).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from . import config

PERTURBATIONS = {
    "clean": None,
    "brightness +15%": ("brightness", 0.15),
    "brightness -15%": ("brightness", -0.15),
    "contrast x0.7": ("contrast", 0.7),
    "contrast x1.3": ("contrast", 1.3),
    "gamma 0.7": ("gamma", 0.7),
    "gamma 1.5": ("gamma", 1.5),
    "gaussian noise sd 8": ("noise", 8.0),
    "gaussian noise sd 16": ("noise", 16.0),
    "blur sigma 1.5": ("blur", 1.5),
    "rotation 10 deg": ("rotate", 10.0),
    "JPEG quality 30": ("jpeg", 30),
    "half resolution": ("downsample", 0.5),
}


def make_perturbation(kind: str, value, seed: int = 0):
    import tensorflow as tf

    def blur(img, sigma):
        r = int(math.ceil(3 * sigma))
        x = tf.range(-r, r + 1, dtype=tf.float32)
        k = tf.exp(-(x ** 2) / (2 * sigma ** 2))
        k = k / tf.reduce_sum(k)
        img4 = img[None]
        img4 = tf.nn.conv2d(img4, tf.reshape(k, (-1, 1, 1, 1)), 1, "SAME")
        img4 = tf.nn.conv2d(img4, tf.reshape(k, (1, -1, 1, 1)), 1, "SAME")
        return img4[0]

    def rotate(img, deg):
        """Rotate about the image centre (pure TF op, safe inside tf.data)."""
        h = tf.cast(tf.shape(img)[0], tf.float32)
        w = tf.cast(tf.shape(img)[1], tf.float32)
        a = deg * math.pi / 180.0
        cos, sin = math.cos(a), math.sin(a)
        cx, cy = (w - 1) / 2, (h - 1) / 2
        # output pixel (x, y) samples input at (cos*x - sin*y + tx, sin*x + cos*y + ty)
        tx = cx - cos * cx + sin * cy
        ty = cy - sin * cx - cos * cy
        transform = tf.stack([cos, -sin, tx, sin, cos, ty, 0.0, 0.0])[None]
        out = tf.raw_ops.ImageProjectiveTransformV3(
            images=img[None], transforms=transform, output_shape=tf.shape(img)[:2],
            fill_value=0.0, interpolation="BILINEAR", fill_mode="CONSTANT")
        return out[0]

    fns = {
        "brightness": lambda img: img + 255.0 * value,
        "contrast": lambda img: (img - tf.reduce_mean(img)) * value + tf.reduce_mean(img),
        "gamma": lambda img: 255.0 * tf.pow(img / 255.0, value),
        "noise": lambda img: img + tf.random.stateless_normal(tf.shape(img), seed=[seed, 1], stddev=value),
        "blur": lambda img: blur(img, value),
        "rotate": lambda img: rotate(img, value),
        "jpeg": lambda img: tf.cast(tf.image.adjust_jpeg_quality(tf.cast(img, tf.uint8), value), tf.float32),
        "downsample": lambda img: tf.image.resize(
            tf.image.resize(img, tf.cast(tf.cast(tf.shape(img)[:2], tf.float32) * value, tf.int32), antialias=True),
            tf.shape(img)[:2]),
    }
    return fns[kind]


def run(model_dir: Path, predictions: pd.DataFrame, img_size: int, batch_size: int,
        perturbations: dict = PERTURBATIONS) -> pd.DataFrame:
    import keras

    from .tfdata import make_dataset

    classes = [c[2:] for c in predictions.columns if c.startswith("p_")]
    rows = []
    for k in sorted(predictions.fold.unique()):
        model = keras.models.load_model(model_dir / f"fold{k}.keras")
        part = predictions[predictions.fold == k]
        for name, spec in perturbations.items():
            fn = None if spec is None else make_perturbation(*spec, seed=int(k))
            p = model.predict(make_dataset(part.abs_path, None, img_size, batch_size, perturb=fn), verbose=0)
            out = part[["idx", "pid", "y", "fold"]].copy()
            out["perturbation"] = name
            for i, c in enumerate(classes):
                out[f"p_{c}"] = p[:, i]
            rows.append(out)
        keras.backend.clear_session()
    return pd.concat(rows, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="effnet_patient")
    ap.add_argument("--figshare-dir", type=Path, default=config.FIGSHARE_DIR)
    ap.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    args = ap.parse_args(argv)
    target = args.artifacts_dir / "predictions" / f"robustness_{args.experiment}.csv"
    if target.exists():
        print("robustness already done")
        return pd.read_csv(target)
    pred = pd.read_csv(args.artifacts_dir / "predictions" / f"{args.experiment}.csv", dtype={"pid": str})
    pred["abs_path"] = [str(args.figshare_dir / "images" / f"{i}.png") for i in pred.idx]
    res = run(args.artifacts_dir / "models" / args.experiment, pred, args.img_size, config.BATCH_SIZE)
    res.to_csv(target, index=False)
    return res


if __name__ == "__main__":
    main()
