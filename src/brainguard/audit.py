"""Dataset audit: exact duplicates, near-duplicates, label conflicts, source confounding.

    python -m brainguard.audit                       # writes reports/kaggle_audit.{md,json}
                                                     # and artifacts/kaggle_index.csv

Runs on CPU, no deep-learning libraries needed. Every later stage reads
artifacts/kaggle_index.csv, so the grouping used to split the data is exactly
the grouping reported here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import config


# ----------------------------------------------------------------------- hashing

def phash63(img: Image.Image) -> int:
    """63-bit DCT perceptual hash: 32x32 greyscale -> 2-D DCT -> 8x8 low
    frequencies without the DC term -> threshold at the median."""
    px = np.asarray(img.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    coeffs = dct(dct(px, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8].flatten()[1:]
    bits = coeffs > np.median(coeffs)
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def thumb64(img: Image.Image) -> np.ndarray:
    """64x64 greyscale, zero-mean, unit-norm vector: dot product of two = Pearson correlation."""
    a = np.asarray(img.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32).ravel()
    a = a - a.mean()
    n = np.linalg.norm(a)
    return a / n if n > 0 else a


def scan_dataset(root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per image file with label, split, geometry, file hash and perceptual hash."""
    rows, thumbs = [], []
    for split_dir, split in config.SPLIT_DIRS.items():
        for label in config.KAGGLE_CLASSES:
            folder = root / split_dir / label
            if not folder.exists():
                raise FileNotFoundError(f"Expected folder {folder}")
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                raw = p.read_bytes()
                with Image.open(p) as im:
                    im.load()
                    rows.append({
                        "path": str(p.relative_to(root)).replace("\\", "/"),
                        "split": split, "label": label, "filename": p.name,
                        "width": im.width, "height": im.height, "mode": im.mode,
                        "file_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                        "phash": phash63(im),
                    })
                    thumbs.append(thumb64(im))
    df = pd.DataFrame(rows)
    df["is_tumour"] = (df["label"] != config.NO_TUMOUR).astype(int)
    df["size_group"] = np.where((df.width == 512) & (df.height == 512), "512x512", "other size")
    return df, np.stack(thumbs)


# ------------------------------------------------------------- near duplicates

def hamming_pairs(hashes: np.ndarray, max_dist: int, chunk: int = 512):
    """All pairs (i < j) whose 63-bit hashes differ in <= max_dist bits."""
    h = hashes.astype(np.uint64)
    rows, cols, dists = [], [], []
    for s in range(0, len(h), chunk):
        block = h[s:s + chunk, None] ^ h[None, :]
        dist = np.bitwise_count(block)
        ii, jj = np.nonzero(dist <= max_dist)
        ii = ii + s
        keep = ii < jj
        rows.append(ii[keep])
        cols.append(jj[keep])
        dists.append(dist[ii[keep] - s, jj[keep]])
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(dists).astype(int)


def cluster_ids(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Connected components: images linked by any chain of near-duplicate pairs share an id."""
    g = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def candidate_pairs(df: pd.DataFrame, thumbs: np.ndarray) -> pd.DataFrame:
    """Hash-similar pairs with their pixel correlation."""
    i, j, d = hamming_pairs(df["phash"].to_numpy(), config.CANDIDATE_HAMMING)
    corr = np.einsum("ij,ij->i", thumbs[i], thumbs[j])
    same_file = df["sha256"].to_numpy()[i] == df["sha256"].to_numpy()[j]
    return pd.DataFrame({"i": i, "j": j, "hamming": d, "corr": np.where(same_file, 1.0, corr)})


def add_groups(df: pd.DataFrame, pairs: pd.DataFrame, min_corr: float = config.DUP_CORRELATION) -> pd.DataFrame:
    """Add exact-duplicate and verified near-duplicate group ids (the latter are the CV groups)."""
    out = df.copy()
    out["exact_group"] = out.groupby("sha256").ngroup()
    e = pairs[pairs["corr"] >= min_corr]
    out["dup_group"] = cluster_ids(len(out), e["i"].to_numpy(), e["j"].to_numpy())
    # exact copies always share a group (identical bytes => identical thumbnails)
    assert out.groupby("exact_group")["dup_group"].nunique().max() == 1
    return out


# ---------------------------------------------------------------------- report

def leakage_summary(df: pd.DataFrame, group_col: str) -> dict:
    g = df.groupby(group_col)
    sizes = g.size()
    multi = sizes[sizes > 1].index
    splits = g["split"].nunique()
    labels = g["label"].nunique()
    cross = splits[splits > 1].index
    test_in_cross = df[(df.split == "test") & df[group_col].isin(cross)]
    return {
        "groups_with_more_than_one_image": int(len(multi)),
        "images_in_those_groups": int(df[group_col].isin(multi).sum()),
        "groups_spanning_train_and_test": int(len(cross)),
        "test_images_with_a_copy_in_train": int(len(test_in_cross)),
        "test_images_with_a_copy_in_train_share": float(len(test_in_cross) / max((df.split == "test").sum(), 1)),
        "groups_with_conflicting_labels": int((labels > 1).sum()),
        "images_in_label_conflict_groups": int(df[group_col].isin(labels[labels > 1].index).sum()),
    }


def audit(df: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    res = {"n_images": int(len(df)),
           "by_split_and_class": {f"{s}/{c}": int(n) for (s, c), n in df.groupby(["split", "label"]).size().items()}}
    res["exact"] = leakage_summary(df, "exact_group")
    res["near"] = leakage_summary(df, "dup_group")
    res["dup_correlation"] = config.DUP_CORRELATION
    lab = df["label"].to_numpy()
    res["candidate_pairs"] = int(len(pairs))
    res["pair_label_agreement_by_correlation"] = [
        {"corr_from": lo, "corr_to": hi, "pairs": int(m.sum()),
         "different_label_pairs": int((lab[pairs.i[m]] != lab[pairs.j[m]]).sum())}
        for lo, hi in [(-1, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.98), (0.98, 1.01)]
        for m in [((pairs["corr"] > lo) & (pairs["corr"] <= hi)).to_numpy()]]

    sens = []
    for c in config.CORRELATION_SENSITIVITY:
        e = pairs[pairs["corr"] >= c]
        tmp = df.assign(g=cluster_ids(len(df), e["i"].to_numpy(), e["j"].to_numpy()))
        sens.append({"min_correlation": c, "n_groups": int(tmp.g.nunique()), **leakage_summary(tmp, "g")})
    res["threshold_sensitivity"] = sens

    # Source confounding: is image geometry / colour mode predictive of the class?
    ct = pd.crosstab(df["label"], df["size_group"])
    res["size_by_class"] = {k: {kk: int(vv) for kk, vv in v.items()} for k, v in ct.to_dict("index").items()}
    ct_mode = pd.crosstab(df["label"], df["mode"])
    res["mode_by_class"] = {k: {kk: int(vv) for kk, vv in v.items()} for k, v in ct_mode.to_dict("index").items()}
    from scipy.stats import chi2_contingency
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    res["size_vs_class_cramers_v"] = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
    res["size_vs_class_chi2_p"] = float(p)
    return res


def to_markdown(r: dict) -> str:
    L = ["# Kaggle dataset audit (generated by `python -m brainguard.audit`; do not edit by hand)", "",
         f"Images: {r['n_images']:,}", "",
         "| Split / class | Images |", "|---|---|"]
    L += [f"| {k} | {v:,} |" for k, v in r["by_split_and_class"].items()]
    for key, title in [("exact", "Exact duplicates (identical file bytes)"),
                       ("near", f"Near-duplicates (similar perceptual hash AND 64x64 pixel correlation "
                                f">= {r['dup_correlation']}, linked transitively) - these are the CV groups")]:
        s = r[key]
        L += ["", f"## {title}", "", "| | |", "|---|---|",
              f"| Groups with more than one image | {s['groups_with_more_than_one_image']:,} |",
              f"| Images in those groups | {s['images_in_those_groups']:,} |",
              f"| Groups spanning Training and Testing | {s['groups_spanning_train_and_test']:,} |",
              f"| **Test images with a copy in Training** | **{s['test_images_with_a_copy_in_train']:,} "
              f"({s['test_images_with_a_copy_in_train_share']:.1%})** |",
              f"| Groups whose copies carry different labels | {s['groups_with_conflicting_labels']:,} "
              f"({s['images_in_label_conflict_groups']:,} images) |"]
    L += ["", "## Why pixel correlation is required (hash-similar candidate pairs)", "",
          f"{r['candidate_pairs']:,} image pairs have perceptual hashes within 12 bits. Pairs that also have high pixel "
          "correlation essentially never disagree on the label; low-correlation pairs are different brains.", "",
          "| Pixel correlation | Pairs | Pairs with different labels |", "|---|---|---|"]
    L += [f"| {max(x['corr_from'], -1):.2f} – {min(x['corr_to'], 1):.2f} | {x['pairs']:,} | {x['different_label_pairs']:,} |"
          for x in r["pair_label_agreement_by_correlation"]]
    L += ["", "## Sensitivity to the correlation cut-off", "",
          "| Min correlation | Groups | Test images with a copy in Training | Label-conflict groups |",
          "|---|---|---|---|"]
    L += [f"| {x['min_correlation']} | {x['n_groups']:,} | {x['test_images_with_a_copy_in_train']:,} "
          f"({x['test_images_with_a_copy_in_train_share']:.1%}) | {x['groups_with_conflicting_labels']} |"
          for x in r["threshold_sensitivity"]]
    L += ["", "## Source confounding: image size by class", "",
          "| Class | 512×512 | Other size |", "|---|---|---|"]
    L += [f"| {c} | {v.get('512x512', 0):,} | {v.get('other size', 0):,} |" for c, v in r["size_by_class"].items()]
    L += ["", f"Association between image size and class: Cramér's V = {r['size_vs_class_cramers_v']:.2f} "
          f"(χ² p = {r['size_vs_class_chi2_p']:.1e}).", "",
          "| Class | " + " | ".join(sorted({m for v in r['mode_by_class'].values() for m in v})) + " |",
          "|---|" + "---|" * len({m for v in r['mode_by_class'].values() for m in v})]
    modes = sorted({m for v in r['mode_by_class'].values() for m in v})
    L += [f"| {c} | " + " | ".join(f"{v.get(m, 0):,}" for m in modes) + " |" for c, v in r["mode_by_class"].items()]
    return "\n".join(L) + "\n"


def figures(df: pd.DataFrame, pairs: pd.DataFrame, root: Path, fig_dir: Path, n_pairs: int = 5) -> None:
    """Example Training/Testing duplicate pairs, and image size by class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    split = df["split"].to_numpy()
    cross = pairs[(split[pairs.i] != split[pairs.j]) & (pairs["corr"] >= config.DUP_CORRELATION)
                  & (pairs["corr"] < 0.999)]
    if len(cross):
        lab = df["label"].to_numpy()
        picks = pd.concat([g.sample(1, random_state=config.SEED) for _, g in cross.groupby(lab[cross.i])])
        picks = picks.head(n_pairs)
        fig, axes = plt.subplots(2, len(picks), figsize=(2.3 * len(picks), 6), dpi=120, squeeze=False)
        for c, (_, r) in enumerate(picks.iterrows()):
            for k, ix in enumerate(sorted([int(r.i), int(r.j)], key=lambda t: split[t] != "train")):
                with Image.open(root / df.path.iloc[ix]) as im:
                    axes[k, c].imshow(im.convert("L"), cmap="gray")
                axes[k, c].set_title(f"{'Training' if split[ix] == 'train' else 'Testing'} · {df.label.iloc[ix]}\n"
                                     f"{df.filename.iloc[ix]}" + (f"\ncorrelation {r['corr']:.3f}" if k == 0 else ""),
                                     fontsize=7, loc="left")
                axes[k, c].axis("off")
        fig.suptitle("Same scan in Kaggle Training (top) and Testing (bottom)", fontsize=10, x=0.01, ha="left")
        fig.tight_layout(h_pad=2.5)
        fig.savefig(fig_dir / "kaggle_duplicates.png")
        plt.close(fig)
    ct = pd.crosstab(df["label"], df["size_group"]).reindex(columns=["512x512", "other size"], fill_value=0)
    share = ct.div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6, 2.6), dpi=150)
    left = np.zeros(len(share))
    for col, color in zip(share.columns, ["#2a78d6", "#eb6834"]):
        ax.barh(share.index, share[col], left=left, color=color, height=0.6, label=col, edgecolor="white", linewidth=2)
        left += share[col].to_numpy()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title("Image size by class: \"no tumour\" comes from a different source", fontsize=10, loc="left", pad=22)
    fig.tight_layout()
    fig.savefig(fig_dir / "kaggle_size_by_class.png")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=config.KAGGLE_DIR)
    ap.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    ap.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    args = ap.parse_args(argv)

    df, thumbs = scan_dataset(args.data)
    pairs = candidate_pairs(df, thumbs)
    df = add_groups(df, pairs)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.artifacts_dir / "kaggle_index.csv", index=False)
    pairs.to_csv(args.artifacts_dir / "duplicate_candidate_pairs.csv", index=False)
    r = audit(df, pairs)
    (args.reports_dir / "kaggle_audit.json").write_text(json.dumps(r, indent=2))
    (args.reports_dir / "kaggle_audit.md").write_text(to_markdown(r))
    figures(df, pairs, args.data, args.reports_dir / "figures")
    print(to_markdown(r))
    return df, r


if __name__ == "__main__":
    main()
