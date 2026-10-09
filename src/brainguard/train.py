"""Cross-validated training for every experiment.

    python -m brainguard.train --experiments effnet_patient effnet_image cnn_patient effnet_patient_cw kaggle_effnet

Experiments (all on the same folds, so results are paired image by image):

  effnet_patient     MAIN. EfficientNetB0, figshare, patient-grouped 5-fold CV.
  effnet_image       Same model, image-level folds (slices of one patient in train AND test).
                     Exists only to measure how much patient leakage inflates accuracy.
  cnn_patient        Baseline CNN (original BrainGuard architecture), patient folds.
  effnet_patient_cw  MAIN + balanced class weights (does re-weighting help the minority class?).
  kaggle_effnet      EfficientNetB0 trained on Kaggle Training, scored on Kaggle Testing as
                     published, to show how leakage and shortcuts flatter a standard benchmark.

Within each training fold, 15% of patients (images for the image-level experiment) are
held out for early stopping, so the evaluation fold is never used for any training decision.
Every fold writes its predictions immediately; rerunning skips finished folds.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .splits import image_folds, patient_folds


@dataclass(frozen=True)
class Experiment:
    name: str
    dataset: str          # "figshare" | "kaggle"
    model: str            # "effnet" | "cnn"
    split: str            # "patient" | "image" | "official"
    class_weight: bool = False
    save_models: bool = False


EXPERIMENTS = {e.name: e for e in [
    Experiment("effnet_patient", "figshare", "effnet", "patient", save_models=True),
    Experiment("effnet_image", "figshare", "effnet", "image"),
    Experiment("cnn_patient", "figshare", "cnn", "patient"),
    Experiment("effnet_patient_cw", "figshare", "effnet", "patient", class_weight=True),
    Experiment("kaggle_effnet", "kaggle", "effnet", "official"),
]}


@dataclass
class Settings:
    img_size: int = config.IMG_SIZE
    batch_size: int = config.BATCH_SIZE
    max_epochs: int = config.MAX_EPOCHS
    head_epochs: int = config.FINE_TUNE_AT_EPOCH
    patience: int = config.EARLY_STOP_PATIENCE
    lr: float = config.LEARNING_RATE
    fine_tune_lr: float = config.FINE_TUNE_LR
    n_folds: int = config.CV_FOLDS
    weights: str | None = "imagenet"
    seed: int = config.SEED


# ------------------------------------------------------------------ data tables

def figshare_table(fig_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(fig_dir / "metadata.csv", dtype={"pid": str})
    df["abs_path"] = [str(fig_dir / p) for p in df.image_path]
    df["y"] = df.label.map({c: i for i, c in enumerate(config.TUMOUR_CLASSES)})
    return df


def kaggle_table(kaggle_dir: Path, artifacts_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(artifacts_dir / "kaggle_index.csv", dtype={"pid": str})
    df["abs_path"] = [str(kaggle_dir / p) for p in df.path]
    df["y"] = df.label.map({c: i for i, c in enumerate(config.KAGGLE_CLASSES)})
    train_groups = set(df.loc[df.split == "train", "dup_group"])
    df["copy_in_train"] = df.dup_group.isin(train_groups) & (df.split == "test")
    if "pid" in df:
        df["pid"] = df["pid"].fillna("")
        train_pids = set(df.loc[(df.split == "train") & (df.pid != ""), "pid"])
        df["patient_in_train"] = (df.pid != "") & df.pid.isin(train_pids) & (df.split == "test")
    return df


def classes_for(exp: Experiment) -> list[str]:
    return config.TUMOUR_CLASSES if exp.dataset == "figshare" else config.KAGGLE_CLASSES


def assign_folds(df: pd.DataFrame, exp: Experiment, n_folds: int, seed: int) -> np.ndarray:
    if exp.split == "patient":
        return patient_folds(df, n_folds, seed)
    if exp.split == "image":
        return image_folds(df, n_folds, seed)
    return np.where(df.split == "test", 0, -1)            # official: one "fold" = Kaggle Testing


def inner_split(train: pd.DataFrame, grouped: bool, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """~15% of the training part, held out for early stopping only."""
    f = patient_folds(train, 7, seed) if grouped else image_folds(train, 7, seed)
    return train[f != 0], train[f == 0]


# ---------------------------------------------------------------------- training

def fit_model(exp: Experiment, n_classes: int, tr: pd.DataFrame, es: pd.DataFrame, s: Settings):
    import keras
    from sklearn.utils.class_weight import compute_class_weight

    from .models import build, unfreeze_top
    from .tfdata import make_dataset

    keras.utils.set_random_seed(s.seed)
    model = build(exp.model, n_classes, s.img_size, s.weights, s.seed)
    train_ds = make_dataset(tr.abs_path, tr.y, s.img_size, s.batch_size, shuffle=True, seed=s.seed)
    es_ds = make_dataset(es.abs_path, es.y, s.img_size, s.batch_size)
    cw = None
    if exp.class_weight:
        w = compute_class_weight("balanced", classes=np.arange(n_classes), y=tr.y.to_numpy())
        cw = {i: float(v) for i, v in enumerate(w)}
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=s.patience, restore_best_weights=True)
    history = {}

    def compile_(lr):
        model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

    if exp.model == "effnet":
        compile_(s.lr)
        h1 = model.fit(train_ds, validation_data=es_ds, epochs=s.head_epochs, class_weight=cw, verbose=2)
        unfreeze_top(model)
        compile_(s.fine_tune_lr)
        h2 = model.fit(train_ds, validation_data=es_ds, epochs=max(s.max_epochs - s.head_epochs, 1),
                       class_weight=cw, callbacks=[stop], verbose=2)
        history = {k: h1.history[k] + h2.history[k] for k in h2.history}
    else:
        # A CNN trained from scratch with BatchNorm has unreliable validation scores in its first
        # epochs (moving statistics not yet settled). Early stopping must not act before they
        # settle, otherwise it keeps a near-random epoch-1 model (seen in the first Colab run:
        # two folds stopped at epoch 6 with 19-22% accuracy).
        cnn_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=max(s.patience + 3, 3),
                                                 restore_best_weights=True,
                                                 start_from_epoch=min(10, max(s.max_epochs - 1, 0)))
        compile_(s.lr * 0.5)
        rlr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4)
        h = model.fit(train_ds, validation_data=es_ds, epochs=max(s.max_epochs, 30) if s.max_epochs > 5 else s.max_epochs,
                      class_weight=cw, callbacks=[cnn_stop, rlr], verbose=2)
        history = h.history
    return model, {k: [float(v) for v in vals] for k, vals in history.items()}


def run_experiment(exp: Experiment, df: pd.DataFrame, out_dir: Path, s: Settings) -> pd.DataFrame:
    from .tfdata import make_dataset

    classes = classes_for(exp)
    pred_dir = out_dir / "predictions" / exp.name
    model_dir = out_dir / "models" / exp.name
    pred_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["fold"] = assign_folds(df, exp, s.n_folds, s.seed)
    folds = sorted(f for f in df.fold.unique() if f >= 0)
    for k in folds:
        target = pred_dir / f"fold{k}.csv"
        if target.exists():
            print(f"[{exp.name}] fold {k}: already done, skipping")
            continue
        t0 = time.time()
        evaluation = df[df.fold == k]
        pool = df[(df.fold != k) & ((df.fold >= 0) if exp.split != "official" else (df.split == "train"))]
        tr, es = inner_split(pool, grouped=exp.split == "patient", seed=s.seed + k)
        if exp.split == "patient":
            assert not set(evaluation.pid) & set(pool.pid), "patient leakage"
        model, hist = fit_model(exp, len(classes), tr, es, s)
        p = model.predict(make_dataset(evaluation.abs_path, None, s.img_size, s.batch_size), verbose=0)
        keep = [c for c in ["idx", "pid", "label", "y", "path", "split", "size_group", "copy_in_train",
                            "patient_in_train"] if c in evaluation]
        res = evaluation[keep].copy()
        res["fold"] = k
        for i, c in enumerate(classes):
            res[f"p_{c}"] = p[:, i]
        res.to_csv(target, index=False)
        (pred_dir / f"fold{k}_history.json").write_text(json.dumps(
            {"seconds": round(time.time() - t0, 1), "epochs": len(hist.get("loss", [])), "history": hist,
             "n_train": len(tr), "n_early_stop": len(es), "n_eval": len(evaluation)}))
        if exp.save_models:
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save(model_dir / f"fold{k}.keras")
        print(f"[{exp.name}] fold {k}: acc {(p.argmax(1) == res.y.to_numpy()).mean():.3f} "
              f"in {time.time() - t0:.0f}s")
        import keras
        keras.backend.clear_session()
    allp = pd.concat([pd.read_csv(pred_dir / f"fold{k}.csv", dtype={"pid": str}) for k in folds], ignore_index=True)
    allp.to_csv(out_dir / "predictions" / f"{exp.name}.csv", index=False)
    return allp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    ap.add_argument("--figshare-dir", type=Path, default=config.FIGSHARE_DIR)
    ap.add_argument("--kaggle-dir", type=Path, default=config.KAGGLE_DIR)
    ap.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    ap.add_argument("--quick", action="store_true", help="2 folds, 2+1 epochs: pipeline check only")
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
    ap.add_argument("--redo", nargs="*", default=[], help="experiments to retrain from scratch (deletes their saved folds)")
    args = ap.parse_args(argv)
    s = Settings(img_size=args.img_size, weights=None if args.no_pretrained else "imagenet")
    if args.quick:
        s.n_folds, s.max_epochs, s.head_epochs, s.patience = 2, 3, 1, 1
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices("GPU"):
            import keras
            keras.mixed_precision.set_global_policy("mixed_float16")
            print("GPU found: mixed precision on")
        else:
            print("WARNING: no GPU found; training will be very slow")
    except ImportError as e:
        raise SystemExit("TensorFlow is required for training: pip install -e '.[dl]'") from e
    import shutil
    for name in args.redo:
        for d in (args.artifacts_dir / "predictions" / name, args.artifacts_dir / "models" / name):
            shutil.rmtree(d, ignore_errors=True)
        (args.artifacts_dir / "predictions" / f"{name}.csv").unlink(missing_ok=True)
        print(f"[{name}] previous results deleted; retraining")
    out = {}
    for name in args.experiments:
        exp = EXPERIMENTS[name]
        df = (figshare_table(args.figshare_dir) if exp.dataset == "figshare"
              else kaggle_table(args.kaggle_dir, args.artifacts_dir))
        out[name] = run_experiment(exp, df, args.artifacts_dir, s)
    return out


if __name__ == "__main__":
    main()
