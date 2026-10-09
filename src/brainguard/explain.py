"""Explainability, and a test of whether the explanations can be trusted.

    python -m brainguard.explain       # after training `effnet_patient`

1. SHAP values by expected gradients (Erion et al. 2021; the algorithm behind
   shap.GradientExplainer): for image x, class t and background images b,
       phi(x) = E_{b, a~U(0,1)} [ (x - b) * d f_t(b + a (x - b)) / dx ]
   Background images come from the fold's TRAINING patients only. Completeness
   (sum of phi ~= f_t(x) - E_b f_t(b)) is recorded for every image. When the `shap`
   package works with the installed Keras, its GradientExplainer is run on a few
   images as an independent cross-check.
2. Grad-CAM on EfficientNet's last convolutional block.
3. Localisation against the radiologist tumour masks (figshare):
   - energy in mask: share of positive attribution inside the tumour; chance level = mask area share
   - pointing game: is the attribution peak inside the (slightly dilated) tumour?
4. Sanity check (Adebayo et al. 2018): attributions from a randomly initialised copy of the
   model should NOT resemble the trained model's; if they do, the maps show image structure,
   not what the model learned.

Each image is explained by the fold model that never saw that patient.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.stats import spearmanr

from . import config


# ----------------------------------------------------------------- attributions

def expected_gradients(model, x: np.ndarray, targets: np.ndarray, background: np.ndarray,
                       n_samples: int, batch: int = 32, seed: int = 0) -> np.ndarray:
    """Returns (n, H, W) attributions (summed over the 3 identical channels)."""
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    out = np.zeros(x.shape[:3], dtype=np.float32)
    for i in range(len(x)):
        b = background[rng.integers(0, len(background), n_samples)]
        a = rng.random(n_samples).astype(np.float32)[:, None, None, None]
        diff = x[i][None] - b
        pts = b + a * diff
        acc = np.zeros(x.shape[1:], dtype=np.float32)
        for s in range(0, n_samples, batch):
            xs = tf.convert_to_tensor(pts[s:s + batch], tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(xs)
                score = tf.cast(model(xs, training=False), tf.float32)[:, int(targets[i])]
            g = tape.gradient(score, xs).numpy()
            acc += (diff[s:s + batch] * g).sum(axis=0)
        out[i] = (acc / n_samples).sum(axis=-1)
    return out


def grad_cam(model, x: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Grad-CAM for the EfficientNet model built in models.py; returns (n, H, W) in [0, 1]."""
    import tensorflow as tf
    base = model.get_layer("efficientnetb0")
    dense = model.get_layer("head_dense")
    xs = tf.convert_to_tensor(x, tf.float32)
    with tf.GradientTape() as tape:
        feats = tf.cast(base(xs, training=False), tf.float32)
        tape.watch(feats)
        # Head computed by hand in float32: under mixed precision (GPU) the Keras layers would
        # cast to float16 and the matmul with a float32 kernel would fail.
        pooled = tf.reduce_mean(feats, axis=(1, 2))
        logits = tf.matmul(pooled, tf.cast(dense.kernel, tf.float32)) + tf.cast(dense.bias, tf.float32)
        score = tf.gather(logits, targets.astype(np.int32), axis=1, batch_dims=1)
    g = tape.gradient(score, feats)
    w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * feats, axis=-1, keepdims=True))
    cam = tf.image.resize(cam, x.shape[1:3], method="bilinear")[..., 0].numpy()
    mx = cam.reshape(len(cam), -1).max(axis=1)[:, None, None]
    return np.where(mx > 0, cam / np.maximum(mx, 1e-12), 0.0)


# ----------------------------------------------------------------- localisation

def load_mask(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as m:
        return np.asarray(m.convert("L").resize((size, size), Image.Resampling.NEAREST)) > 127


def localisation(attr: np.ndarray, mask: np.ndarray, dilate_px: int = 3, smooth: float = 2.0) -> dict:
    pos = np.clip(attr, 0, None)
    total = pos.sum()
    energy = float(pos[mask].sum() / total) if total > 0 else float("nan")
    dil = binary_dilation(mask, iterations=dilate_px)
    peak = np.unravel_index(np.argmax(gaussian_filter(pos, smooth)), pos.shape)
    return {"energy_in_mask": energy, "mask_fraction": float(mask.mean()),
            "pointing_hit": bool(dil[peak]), "pointing_chance": float(dil.mean())}


def map_similarity(a: np.ndarray, b: np.ndarray, size: int = 56) -> float:
    """Spearman rank correlation of |attribution| maps after block-averaging."""
    def pool(m):
        m = np.abs(m)
        f = m.shape[0] // size
        return m[: f * size, : f * size].reshape(size, f, size, f).mean(axis=(1, 3)).ravel() if f > 1 else m.ravel()
    return float(spearmanr(pool(a), pool(b)).statistic)


# ------------------------------------------------------------------------- main

def select_images(pred: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    per_class = max(n // pred.y.nunique(), 1)
    parts = [g.sample(min(per_class, len(g)), random_state=seed) for _, g in pred.groupby("y")]
    return pd.concat(parts).sort_values(["fold", "idx"]).reset_index(drop=True)


def run(pred: pd.DataFrame, fig_dir: Path, model_dir: Path, out_dir: Path, img_size: int,
        n_images: int = config.EXPLAIN_N_IMAGES, n_background: int = config.SHAP_BACKGROUND,
        n_samples: int = config.SHAP_SAMPLES, n_sanity: int = 24, seed: int = config.SEED) -> pd.DataFrame:
    import keras

    from .tfdata import load_array

    out_dir.mkdir(parents=True, exist_ok=True)
    classes = [c[2:] for c in pred.columns if c.startswith("p_")]
    meta = pd.read_csv(fig_dir / "metadata.csv", dtype={"pid": str}).set_index("idx")
    chosen = select_images(pred, n_images, seed)
    rows, sanity, examples = [], [], {}
    rng = np.random.default_rng(seed)
    shap_check = {"status": "not run"}
    for k, part in chosen.groupby("fold"):
        model = keras.models.load_model(model_dir / f"fold{k}.keras")
        train_idx = pred.loc[pred.fold != k, "idx"].to_numpy()
        bg_idx = rng.choice(train_idx, min(n_background, len(train_idx)), replace=False)
        background = load_array([str(fig_dir / meta.loc[i, "image_path"]) for i in bg_idx], img_size)
        x = load_array([str(fig_dir / meta.loc[i, "image_path"]) for i in part.idx], img_size)
        targets = part.y.to_numpy()
        eg = expected_gradients(model, x, targets, background, n_samples, seed=seed + int(k))
        cam = grad_cam(model, x, targets)
        f_x = model.predict(x, verbose=0)[np.arange(len(x)), targets]
        f_bg = model.predict(background, verbose=0)[:, targets].mean(axis=0)
        probs = part[[f"p_{c}" for c in classes]].to_numpy()
        for j, (_, r) in enumerate(part.iterrows()):
            mask = load_mask(fig_dir / meta.loc[r.idx, "mask_path"], img_size)
            le, lc = localisation(eg[j], mask), localisation(cam[j], mask)
            rows.append({"idx": r.idx, "pid": r.pid, "fold": k, "label": classes[r.y],
                         "predicted": classes[int(probs[j].argmax())], "correct": bool(probs[j].argmax() == r.y),
                         "mask_fraction": le["mask_fraction"],
                         "shap_energy_in_mask": le["energy_in_mask"], "shap_pointing_hit": le["pointing_hit"],
                         "cam_energy_in_mask": lc["energy_in_mask"], "cam_pointing_hit": lc["pointing_hit"],
                         "pointing_chance": le["pointing_chance"],
                         "completeness_sum_shap": float(eg[j].sum()),
                         "completeness_f_minus_baseline": float(f_x[j] - f_bg[j])})
        firsts = list(dict.fromkeys(int(t) for t in targets))       # one map per tumour type per fold, for the figure
        for j in [int(np.flatnonzero(targets == t)[0]) for t in firsts]:
            examples[int(part.idx.iloc[j])] = {"image": x[j, ..., 0], "shap": eg[j], "cam": cam[j],
                                               "mask": load_mask(fig_dir / meta.loc[part.idx.iloc[j], "mask_path"],
                                                                 img_size),
                                               "title": f"{classes[targets[j]]} · P={f_x[j]:.2f}"}
        # sanity check on a subset: randomly re-initialised copy of the same architecture
        m = min(n_sanity // max(chosen.fold.nunique(), 1) + 1, len(part))
        rand_model = keras.models.clone_model(model)
        eg_r = expected_gradients(rand_model, x[:m], targets[:m], background, n_samples, seed=seed + 99)
        cam_r = grad_cam(rand_model, x[:m], targets[:m])
        for j in range(m):
            mask = load_mask(fig_dir / meta.loc[part.idx.iloc[j], "mask_path"], img_size)
            sanity.append({"idx": int(part.idx.iloc[j]),
                           "shap_similarity_trained_vs_random": map_similarity(eg[j], eg_r[j]),
                           "cam_similarity_trained_vs_random": map_similarity(cam[j], cam_r[j]),
                           "random_shap_energy_in_mask": localisation(eg_r[j], mask)["energy_in_mask"],
                           "trained_shap_energy_in_mask": localisation(eg[j], mask)["energy_in_mask"]})
        if shap_check["status"] == "not run":
            shap_check = shap_crosscheck(model, x[:3], targets[:3], background, n_samples, eg[:3])
        keras.backend.clear_session()
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "explain_metrics.csv", index=False)
    pd.DataFrame(sanity).to_csv(out_dir / "explain_sanity.csv", index=False)
    (out_dir / "shap_crosscheck.json").write_text(json.dumps(shap_check, indent=2))
    np.savez_compressed(out_dir / "explain_examples.npz",
                        **{f"{k}_{f}": v[f] for k, v in examples.items() for f in ("image", "shap", "cam", "mask")},
                        titles=json.dumps({k: v["title"] for k, v in examples.items()}))
    return res


def shap_crosscheck(model, x, targets, background, n_samples, ours) -> dict:
    """Run the shap package's GradientExplainer (same algorithm) and correlate with ours."""
    try:
        import shap
        e = shap.GradientExplainer(model, background)
        sv = e.shap_values(x, nsamples=n_samples)
        sv = np.asarray(sv)
        if sv.ndim == 5 and sv.shape[0] == len(x):          # (n, H, W, C, classes)
            maps = [sv[i, ..., targets[i]].sum(-1) for i in range(len(x))]
        else:                                                # list per class -> (classes, n, H, W, C)
            maps = [sv[targets[i], i].sum(-1) for i in range(len(x))]
        corr = [float(np.corrcoef(maps[i].ravel(), ours[i].ravel())[0, 1]) for i in range(len(x))]
        return {"status": "ok", "shap_version": shap.__version__, "pearson_r_per_image": corr}
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:500]}


def pick_examples(titles: dict[str, str], max_rows: int = 6) -> list[str]:
    """Balanced figure: per tumour type, cases spread from most to least confident (successes and failures)."""
    by_class: dict[str, list[tuple[float, str]]] = {}
    for k, t in titles.items():
        cls, _, prob = t.partition(" · P=")
        by_class.setdefault(cls, []).append((float(prob or "nan"), k))
    per_class = max(1, max_rows // max(len(by_class), 1))
    keys = []
    for cls in sorted(by_class):
        ranked = sorted(by_class[cls], reverse=True)
        n = min(per_class, len(ranked))
        spots = np.unique(np.round(np.linspace(0, len(ranked) - 1, n)).astype(int))   # spread from most to least confident
        keys += [ranked[i][1] for i in spots]
    return keys[:max_rows]


def plot_examples(npz_path: Path, out_png: Path, max_rows: int = 6) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(npz_path, allow_pickle=False)
    titles = json.loads(str(d["titles"]))
    keys = pick_examples(titles, max_rows)
    if not keys:
        return
    fig, axes = plt.subplots(len(keys), 3, figsize=(9, 3 * len(keys)), dpi=120, squeeze=False)
    for r, k in enumerate(keys):
        img, sh, cam, mask = (d[f"{k}_{f}"] for f in ("image", "shap", "cam", "mask"))
        lim = np.percentile(np.abs(sh), 99.5) or 1
        for c in range(3):
            axes[r, c].imshow(img, cmap="gray")
            axes[r, c].contour(mask, levels=[0.5], colors="#1baf7a", linewidths=1)
            axes[r, c].axis("off")
        axes[r, 1].imshow(gaussian_filter(sh, 1.5), cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.6)
        axes[r, 2].imshow(cam, cmap="inferno", alpha=0.45)
        axes[r, 0].set_title(f"MRI + tumour mask (green)\ntrue: {titles[k]}", fontsize=8, loc="left")
        axes[r, 1].set_title("SHAP (expected gradients)\nred = supports the true class", fontsize=8, loc="left")
        axes[r, 2].set_title("Grad-CAM", fontsize=8, loc="left")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="effnet_patient")
    ap.add_argument("--figshare-dir", type=Path, default=config.FIGSHARE_DIR)
    ap.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    ap.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--n-images", type=int, default=config.EXPLAIN_N_IMAGES)
    ap.add_argument("--n-samples", type=int, default=config.SHAP_SAMPLES)
    ap.add_argument("--n-background", type=int, default=config.SHAP_BACKGROUND)
    args = ap.parse_args(argv)
    out_dir = args.artifacts_dir / "explain"
    if not (out_dir / "explain_metrics.csv").exists():
        pred = pd.read_csv(args.artifacts_dir / "predictions" / f"{args.experiment}.csv", dtype={"pid": str})
        run(pred, args.figshare_dir, args.artifacts_dir / "models" / args.experiment, out_dir, args.img_size,
            n_images=args.n_images, n_background=args.n_background, n_samples=args.n_samples)
    plot_examples(out_dir / "explain_examples.npz", args.reports_dir / "figures" / "explain_examples.png")


if __name__ == "__main__":
    main()
