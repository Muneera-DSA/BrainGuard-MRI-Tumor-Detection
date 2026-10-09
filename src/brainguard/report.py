"""Turn saved predictions into the results report: metrics with patient-level CIs and
every hypothesis test. CPU only; runs from the CSVs in artifacts/.

    python -m brainguard.report      # writes reports/results.md, reports/results.json, figures

All confidence intervals resample PATIENTS (2,000 bootstrap draws), because slices of one
patient are correlated. Paired comparisons use the same images in both arms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

from . import config
from .stats import (expected_calibration_error, fisher_rates, group_bootstrap, holm, mcnemar_exact,
                    multiclass_brier)

N_BOOT = 2000
BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"


# ------------------------------------------------------------------- metrics

def proba(df: pd.DataFrame, classes: list[str]) -> np.ndarray:
    return df[[f"p_{c}" for c in classes]].to_numpy(float)


def balanced_acc(y, yhat, n_classes):
    rec = [np.mean(yhat[y == c] == c) for c in range(n_classes) if (y == c).any()]
    return float(np.mean(rec))


def macro_auc(y, p):
    present = np.unique(y)
    if len(present) < 2:
        return float("nan")
    return float(np.mean([roc_auc_score(y == c, p[:, c]) for c in present]))


def summarise(df: pd.DataFrame, classes: list[str], n_boot: int = N_BOOT) -> dict:
    y, p = df.y.to_numpy(), proba(df, classes)
    yhat = p.argmax(1)
    groups = df.pid.to_numpy() if "pid" in df and df.pid.notna().all() else np.arange(len(df))
    nc = len(classes)
    boot = lambda f: group_bootstrap(f, groups, n_boot=n_boot)  # noqa: E731
    r = {
        "n_images": int(len(df)), "n_patients": int(len(np.unique(groups))),
        "accuracy": boot(lambda i: np.mean(yhat[i] == y[i])),
        "balanced_accuracy": boot(lambda i: balanced_acc(y[i], yhat[i], nc)),
        "macro_f1": boot(lambda i: f1_score(y[i], yhat[i], average="macro", labels=range(nc), zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y, yhat)),
        "macro_auc": boot(lambda i: macro_auc(y[i], p[i])),
        "ece": expected_calibration_error(y, p), "brier": multiclass_brier(y, p),
        "confusion": pd.crosstab(pd.Categorical([classes[i] for i in y], classes),
                                 pd.Categorical([classes[i] for i in yhat], classes), dropna=False).to_numpy().tolist(),
        "per_class": {},
    }
    for c, name in enumerate(classes):
        r["per_class"][name] = {
            "n": int((y == c).sum()),
            "sensitivity": boot(lambda i, c=c: np.mean(yhat[i][y[i] == c] == c) if (y[i] == c).any() else np.nan),
            "specificity": boot(lambda i, c=c: np.mean(yhat[i][y[i] != c] != c) if (y[i] != c).any() else np.nan),
            "precision": boot(lambda i, c=c: np.mean(y[i][yhat[i] == c] == c) if (yhat[i] == c).any() else np.nan),
            "auc": boot(lambda i, c=c: roc_auc_score(y[i] == c, p[i][:, c]) if 0 < (y[i] == c).sum() < len(i) else np.nan),
        }
    if "fold" in df:
        acc = df.assign(ok=(yhat == y)).groupby("fold").ok.mean()
        r["fold_accuracy"] = {"mean": float(acc.mean()), "sd": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
                              "per_fold": [float(a) for a in acc]}
    return r


def paired_comparison(a: pd.DataFrame, b: pd.DataFrame, classes: list[str], label_a: str, label_b: str,
                      n_boot: int = N_BOOT) -> dict:
    """a and b scored the same images (joined on idx). Positive diff = a better."""
    m = a.merge(b, on="idx", suffixes=("_a", "_b"))
    y = m.y_a.to_numpy()
    pa = m[[f"p_{c}_a" for c in classes]].to_numpy().argmax(1)
    pb = m[[f"p_{c}_b" for c in classes]].to_numpy().argmax(1)
    nc = len(classes)
    diff = group_bootstrap(lambda i: balanced_acc(y[i], pa[i], nc) - balanced_acc(y[i], pb[i], nc),
                           m.pid_a.to_numpy(), n_boot=n_boot)
    mc = mcnemar_exact(pa == y, pb == y)
    per_class = {c: float(np.mean(pa[y == k] == k) - np.mean(pb[y == k] == k)) for k, c in enumerate(classes)}
    return {"a": label_a, "b": label_b, "n_images": int(len(m)),
            "balanced_accuracy_a": balanced_acc(y, pa, nc), "balanced_accuracy_b": balanced_acc(y, pb, nc),
            "diff_balanced_accuracy": diff, "mcnemar": mc, "per_class_sensitivity_diff": per_class}


# ------------------------------------------------------------------ sections

def kaggle_section(k: pd.DataFrame) -> dict:
    classes = config.KAGGLE_CLASSES
    y, yhat = k.y.to_numpy(), proba(k, classes).argmax(1)
    ok = yhat == y
    out = {"n_test": int(len(k)), "accuracy_as_published": float(ok.mean()),
           "balanced_accuracy": balanced_acc(y, yhat, len(classes))}
    for col, name in [("copy_in_train", "test image has a near-copy in Training"),
                      ("patient_in_train", "test image's patient is in Training")]:
        if col in k:
            f = k[col].fillna(False).astype(bool).to_numpy()
            if f.any() and (~f).any():
                out[col] = {"label": name, "n_flagged": int(f.sum()), "acc_flagged": float(ok[f].mean()),
                            "acc_not_flagged": float(ok[~f].mean()),
                            "fisher": fisher_rates(int((~ok[f]).sum()), int(f.sum()), int((~ok[~f]).sum()), int((~f).sum()))}
    if "size_group" in k:
        out["accuracy_by_size_group"] = {g: float(ok[(k.size_group == g).to_numpy()].mean())
                                         for g in k.size_group.unique()}
    return out


def explain_section(e: pd.DataFrame, sanity: pd.DataFrame | None) -> dict:
    out = {"n_images": int(len(e)), "n_patients": int(e.pid.nunique())}
    for m in ("shap", "cam"):
        excess = (e[f"{m}_energy_in_mask"] - e.mask_fraction).dropna()
        w = sps.wilcoxon(excess, alternative="greater") if len(excess) > 0 and (excess != 0).any() else None
        hits = e[f"{m}_pointing_hit"].astype(bool)
        exp_hits, var = e.pointing_chance.sum(), (e.pointing_chance * (1 - e.pointing_chance)).sum()
        z = (hits.sum() - exp_hits) / np.sqrt(var) if var > 0 else float("nan")
        corr, inc = e[e.correct][f"{m}_energy_in_mask"].dropna(), e[~e.correct][f"{m}_energy_in_mask"].dropna()
        out[m] = {
            "median_energy_in_mask": float(e[f"{m}_energy_in_mask"].median()),
            "median_mask_fraction": float(e.mask_fraction.median()),
            "median_ratio_to_chance": float((e[f"{m}_energy_in_mask"] / e.mask_fraction).median()),
            "wilcoxon_p_energy_above_chance": float(w.pvalue) if w else float("nan"),
            "pointing_hit_rate": float(hits.mean()), "pointing_chance_rate": float(e.pointing_chance.mean()),
            "pointing_p_above_chance": float(sps.norm.sf(z)) if np.isfinite(z) else float("nan"),
            "energy_correct_vs_incorrect": {
                "median_correct": float(corr.median()) if len(corr) else float("nan"),
                "median_incorrect": float(inc.median()) if len(inc) else float("nan"),
                "mannwhitney_p": float(sps.mannwhitneyu(corr, inc).pvalue) if len(corr) and len(inc) else float("nan")},
        }
    c = e[["completeness_sum_shap", "completeness_f_minus_baseline"]].dropna()
    out["shap_completeness_r"] = float(np.corrcoef(c.iloc[:, 0], c.iloc[:, 1])[0, 1]) if len(c) > 2 else float("nan")
    if sanity is not None and len(sanity):
        out["sanity"] = {
            "median_shap_similarity_trained_vs_random": float(sanity.shap_similarity_trained_vs_random.median()),
            "median_cam_similarity_trained_vs_random": float(sanity.cam_similarity_trained_vs_random.median()),
            "median_energy_trained": float(sanity.trained_shap_energy_in_mask.median()),
            "median_energy_random": float(sanity.random_shap_energy_in_mask.median()),
            "wilcoxon_p_trained_gt_random": float(sps.wilcoxon(
                sanity.trained_shap_energy_in_mask - sanity.random_shap_energy_in_mask, alternative="greater").pvalue)
            if (sanity.trained_shap_energy_in_mask != sanity.random_shap_energy_in_mask).any() else float("nan"),
        }
    return out


def robustness_section(r: pd.DataFrame, classes: list[str]) -> list[dict]:
    clean = r[r.perturbation == "clean"]
    out = []
    for name, g in r.groupby("perturbation", sort=False):
        m = clean.merge(g, on="idx", suffixes=("_c", "_p"))
        y = m.y_c.to_numpy()
        pc = m[[f"p_{c}_c" for c in classes]].to_numpy().argmax(1)
        pp = m[[f"p_{c}_p" for c in classes]].to_numpy().argmax(1)
        out.append({"perturbation": name, "balanced_accuracy": balanced_acc(y, pp, len(classes)),
                    "drop_vs_clean": balanced_acc(y, pc, len(classes)) - balanced_acc(y, pp, len(classes)),
                    "mcnemar_p": mcnemar_exact(pc == y, pp == y)["p"] if name != "clean" else float("nan")})
    adj = holm({x["perturbation"]: x["mcnemar_p"] for x in out if x["perturbation"] != "clean"})
    for x in out:
        x["mcnemar_p_holm"] = adj.get(x["perturbation"], float("nan"))
    return out


# ------------------------------------------------------------------- figures

def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=MUTED, labelsize=8.5)


def figures(res: dict, main_df: pd.DataFrame | None, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    classes = config.TUMOUR_CLASSES
    if "main" in res:
        cm = np.array(res["main"]["confusion"])
        fig, ax = plt.subplots(figsize=(4.6, 4), dpi=150)
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] / cm[i].sum() > 0.5 else INK)
        ax.set_xticks(range(3), classes)
        ax.set_yticks(range(3), classes)
        ax.set_xlabel("Predicted", color=MUTED)
        ax.set_ylabel("True", color=MUTED)
        _style(ax, "Confusion matrix, unseen patients (5-fold CV)")
        fig.tight_layout()
        fig.savefig(fig_dir / "confusion_matrix.png")
        plt.close(fig)
    if main_df is not None:
        p = proba(main_df, classes)
        conf, correct = p.max(1), (p.argmax(1) == main_df.y.to_numpy())
        bins = np.quantile(conf, np.linspace(0, 1, 11))
        idx = np.clip(np.digitize(conf, bins[1:-1]), 0, 9)
        xs = [conf[idx == b].mean() for b in range(10) if (idx == b).any()]
        ys = [correct[idx == b].mean() for b in range(10) if (idx == b).any()]
        fig, ax = plt.subplots(figsize=(4.4, 4), dpi=150)
        ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="perfect calibration")
        ax.plot(xs, ys, "o-", color=BLUE, lw=2, ms=5, label="model")
        ax.set_xlabel("Predicted confidence", color=MUTED)
        ax.set_ylabel("Observed accuracy", color=MUTED)
        ax.set_xlim(0.3, 1.01)
        ax.set_ylim(0.3, 1.01)
        ax.legend(frameon=False, fontsize=8)
        _style(ax, "Calibration (unseen patients)")
        fig.tight_layout()
        fig.savefig(fig_dir / "calibration.png")
        plt.close(fig)
    bars = []
    if "main" in res:
        bars.append(("Figshare, unseen patients\n(this project)", res["main"]["accuracy"]["value"], BLUE))
    if "leakage" in res:
        bars.append(("Figshare, image-level split\n(patients leak)", res["leakage"]["accuracy_image_split"], ORANGE))
    if "kaggle" in res:
        bars.append(("Kaggle test set\nas published", res["kaggle"]["accuracy_as_published"], GREY))
    if bars:
        fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=150)
        ax.barh([b[0] for b in bars][::-1], [b[1] for b in bars][::-1], color=[b[2] for b in bars][::-1], height=0.55)
        for i, b in enumerate(bars[::-1]):
            ax.text(b[1] + 0.005, i, f"{b[1]:.1%}", va="center", fontsize=9, color=INK)
        ax.set_xlim(0.5, 1.0)
        ax.set_xlabel("Accuracy", color=MUTED)
        _style(ax, "Same model family, different evaluation protocols")
        fig.tight_layout()
        fig.savefig(fig_dir / "protocol_comparison.png")
        plt.close(fig)
    if "robustness" in res:
        rob = [x for x in res["robustness"] if x["perturbation"] != "clean"]
        fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(rob) + 1.2), dpi=150)
        names = [x["perturbation"] for x in rob][::-1]
        drops = [x["drop_vs_clean"] for x in rob][::-1]
        sig = [x["mcnemar_p_holm"] < 0.05 for x in rob][::-1]
        ax.barh(names, drops, color=[ORANGE if s else GREY for s in sig], height=0.6)
        ax.axvline(0, color="#c9c8c3", lw=1)
        ax.set_xlabel("Drop in balanced accuracy vs clean (orange = significant, Holm-corrected)", color=MUTED, fontsize=8)
        _style(ax, "Robustness to acquisition changes")
        fig.tight_layout()
        fig.savefig(fig_dir / "robustness.png")
        plt.close(fig)


# -------------------------------------------------------------------- markdown

def _ci(d, pct=True):
    f = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.3f}")
    return f"{f(d['value'])} ({f(d['ci'][0])}–{f(d['ci'][1])})"


def to_markdown(res: dict) -> str:
    L = ["# Results (generated by `python -m brainguard.report`; do not edit by hand)", "",
         "Confidence intervals: 95%, patient-level bootstrap (2,000 draws). "
         "p-values in the hypothesis table are Holm-adjusted across that table.", ""]
    if "main" in res:
        m = res["main"]
        L += ["## Main model: EfficientNetB0, figshare, patient-grouped 5-fold CV", "",
              f"{m['n_images']:,} images from {m['n_patients']} patients; every prediction is for a patient the model never saw.", "",
              "| Metric | Value (95% CI) |", "|---|---|",
              f"| Accuracy | {_ci(m['accuracy'])} |", f"| Balanced accuracy | {_ci(m['balanced_accuracy'])} |",
              f"| Macro F1 | {_ci(m['macro_f1'], False)} |", f"| Macro one-vs-rest AUC | {_ci(m['macro_auc'], False)} |",
              f"| Cohen's κ | {m['cohen_kappa']:.3f} |",
              f"| Calibration: ECE / Brier | {m['ece']:.3f} / {m['brier']:.3f} |"]
        if "fold_accuracy" in m:
            fa = m["fold_accuracy"]
            L.append(f"| Accuracy across folds | {fa['mean']:.1%} ± {fa['sd']:.1%} (SD) |")
        L += ["", "| Class | Images | Sensitivity | Specificity | Precision | AUC |", "|---|---|---|---|---|---|"]
        for c, v in m["per_class"].items():
            L.append(f"| {c} | {v['n']:,} | {_ci(v['sensitivity'])} | {_ci(v['specificity'])} | "
                     f"{_ci(v['precision'])} | {_ci(v['auc'], False)} |")
    if "hypotheses" in res:
        L += ["", "## Hypothesis tests", "", "| # | Hypothesis | Evidence | p (Holm) | Verdict |", "|---|---|---|---|---|"]
        for h in res["hypotheses"]:
            L.append(f"| {h['id']} | {h['hypothesis']} | {h['evidence']} | {h['p_holm']:.2g} | {h['verdict']} |")
    if "kaggle" in res:
        k = res["kaggle"]
        L += ["", "## Kaggle benchmark as published (4 classes incl. no tumour)", "",
              f"EfficientNetB0 trained on Kaggle Training, scored on Kaggle Testing ({k['n_test']:,} images): "
              f"accuracy **{k['accuracy_as_published']:.1%}**, balanced accuracy {k['balanced_accuracy']:.1%}.", "",
              "| Test subset | Images | Accuracy in subset | Accuracy elsewhere | Fisher p |", "|---|---|---|---|---|"]
        for col in ("copy_in_train", "patient_in_train"):
            if col in k:
                v = k[col]
                L.append(f"| {v['label']} | {v['n_flagged']:,} | {v['acc_flagged']:.1%} | {v['acc_not_flagged']:.1%} | "
                         f"{v['fisher']['p']:.2g} |")
    if "explain" in res:
        e = res["explain"]
        L += ["", "## Do the explanations point at the tumour?", "",
              f"{e['n_images']} held-out images ({e['n_patients']} patients), each explained by the fold model that never saw "
              "the patient, scored against the radiologist's tumour mask.", "",
              "| | SHAP (expected gradients) | Grad-CAM |", "|---|---|---|"]
        s, c = e["shap"], e["cam"]
        L += [f"| Median share of positive attribution inside the tumour | {s['median_energy_in_mask']:.1%} | {c['median_energy_in_mask']:.1%} |",
              f"| Median tumour area (chance level) | {s['median_mask_fraction']:.1%} | {c['median_mask_fraction']:.1%} |",
              f"| Median ratio to chance | {s['median_ratio_to_chance']:.1f}× | {c['median_ratio_to_chance']:.1f}× |",
              f"| Wilcoxon p (above chance) | {s['wilcoxon_p_energy_above_chance']:.2g} | {c['wilcoxon_p_energy_above_chance']:.2g} |",
              f"| Pointing game: peak inside tumour | {s['pointing_hit_rate']:.1%} (chance {s['pointing_chance_rate']:.1%}) | "
              f"{c['pointing_hit_rate']:.1%} (chance {c['pointing_chance_rate']:.1%}) |",
              f"| Energy in tumour, correct vs wrong predictions | {s['energy_correct_vs_incorrect']['median_correct']:.1%} vs "
              f"{s['energy_correct_vs_incorrect']['median_incorrect']:.1%} | {c['energy_correct_vs_incorrect']['median_correct']:.1%} vs "
              f"{c['energy_correct_vs_incorrect']['median_incorrect']:.1%} |"]
        if "sanity" in e:
            sn = e["sanity"]
            L += ["", f"Sanity check (randomly initialised model): rank similarity of maps trained vs random — SHAP "
                  f"{sn['median_shap_similarity_trained_vs_random']:.2f}, Grad-CAM {sn['median_cam_similarity_trained_vs_random']:.2f} "
                  f"(low = the maps depend on what the model learned). Energy in tumour: trained {sn['median_energy_trained']:.1%} "
                  f"vs random {sn['median_energy_random']:.1%}."]
            for name, key in (("SHAP", "median_shap_similarity_trained_vs_random"), ("Grad-CAM", "median_cam_similarity_trained_vs_random")):
                v = sn[key]
                if np.isfinite(v):
                    L.append(f"- {name}: " + ("**passes** (maps change when the learned weights are removed)." if v < SANITY_MAX_SIMILARITY else
                             f"**does not pass** (similarity ≥ {SANITY_MAX_SIMILARITY}): the maps mostly reflect image structure, "
                             "not what the model learned, so they should not be read as the model's reasoning."))
        L.append(f"SHAP completeness (sum of attributions vs model output change): r = {e['shap_completeness_r']:.2f}.")
    if "robustness" in res:
        L += ["", "## Robustness (held-out patients, paired with clean images)", "",
              "| Perturbation | Balanced accuracy | Drop vs clean (positive = worse) | McNemar p (Holm) |", "|---|---|---|---|"]
        for x in res["robustness"]:
            p = "" if x["perturbation"] == "clean" else f"{x['mcnemar_p_holm']:.2g}"
            L.append(f"| {x['perturbation']} | {x['balanced_accuracy']:.1%} | {x['drop_vs_clean']:+.1%} | {p} |")
    return "\n".join(L) + "\n"


# Adebayo et al. (2018) give no fixed cut-off; 0.5 rank correlation is used here as a
# conservative line between 'clearly model-dependent' and 'mostly input-driven'.
SANITY_MAX_SIMILARITY = 0.5

# ------------------------------------------------------------------------ main

def build(art: Path, reports: Path, n_boot: int = N_BOOT) -> dict:
    pred = art / "predictions"
    read = lambda n: pd.read_csv(pred / f"{n}.csv", dtype={"pid": str}) if (pred / f"{n}.csv").exists() else None  # noqa: E731
    classes = config.TUMOUR_CLASSES
    res, hyps = {}, []
    main = read("effnet_patient")
    if main is not None:
        res["main"] = summarise(main, classes, n_boot)
        # H: some classes are harder (meningioma expected)
        y, yhat = main.y.to_numpy(), proba(main, classes).argmax(1)
        k_men = classes.index("meningioma")
        d = group_bootstrap(lambda i: np.mean(yhat[i][y[i] == k_men] == k_men)
                            - np.mean(yhat[i][y[i] != k_men] == y[i][y[i] != k_men]), main.pid.to_numpy(), n_boot=n_boot)
        hyps.append({"id": "H4", "hypothesis": "Meningioma is detected less reliably than the other classes",
                     "evidence": f"sensitivity difference {d['value']:+.1%} (CI {d['ci'][0]:+.1%} to {d['ci'][1]:+.1%})",
                     # one-sided bootstrap p: share of draws in which meningioma is NOT worse
                     "p": max(1 - d["share_le_0"], 1 / n_boot),
                     "direction_ok": d["value"] < 0})
    for name, hid, text, expect_a_better in [
        ("effnet_image", "H1", "Image-level CV overstates accuracy versus patient-level CV (leakage)", True),
        ("cnn_patient", "H2", "Transfer learning (EfficientNetB0) beats the baseline CNN", False),
        ("effnet_patient_cw", "H3", "Class weighting improves balanced accuracy", True),
    ]:
        other = read(name)
        if main is None or other is None:
            continue
        cmp = paired_comparison(other, main, classes, name, "effnet_patient", n_boot)
        res[f"compare_{name}"] = cmp
        diff = cmp["diff_balanced_accuracy"]
        if name == "effnet_image":
            res["leakage"] = {"accuracy_image_split": float(np.mean(proba(other, classes).argmax(1) == other.y)),
                              "accuracy_patient_split": res["main"]["accuracy"]["value"]}
        d = diff["value"] if expect_a_better else -diff["value"]
        hyps.append({"id": hid, "hypothesis": text,
                     "evidence": f"balanced accuracy {cmp['balanced_accuracy_a']:.1%} ({name}) vs "
                                 f"{cmp['balanced_accuracy_b']:.1%} (main); McNemar {cmp['mcnemar']['a_right_b_wrong']}"
                                 f"/{cmp['mcnemar']['b_right_a_wrong']} discordant",
                     "p": cmp["mcnemar"]["p"], "direction_ok": d > 0})
    kag = read("kaggle_effnet")
    if kag is not None:
        res["kaggle"] = kaggle_section(kag)
        for col, hid in (("copy_in_train", "H5a"), ("patient_in_train", "H5b")):
            if col in res["kaggle"]:
                v = res["kaggle"][col]
                hyps.append({"id": hid, "hypothesis": f"Kaggle accuracy is higher when the {v['label'].replace('test image', 'image')}",
                             "evidence": f"{v['acc_flagged']:.1%} vs {v['acc_not_flagged']:.1%}",
                             "p": v["fisher"]["p"], "direction_ok": v["acc_flagged"] > v["acc_not_flagged"]})
    ex = art / "explain" / "explain_metrics.csv"
    if ex.exists():
        sanity = art / "explain" / "explain_sanity.csv"
        res["explain"] = explain_section(pd.read_csv(ex, dtype={"pid": str}),
                                         pd.read_csv(sanity) if sanity.exists() else None)
        s = res["explain"]["shap"]
        hyps.append({"id": "H6", "hypothesis": "SHAP attribution concentrates on the tumour more than chance",
                     "evidence": f"median {s['median_energy_in_mask']:.1%} inside vs {s['median_mask_fraction']:.1%} area",
                     "p": s["wilcoxon_p_energy_above_chance"], "direction_ok": s["median_ratio_to_chance"] > 1})
        shap_check = art / "explain" / "shap_crosscheck.json"
        if shap_check.exists():
            res["explain"]["shap_library_crosscheck"] = json.loads(shap_check.read_text())
    rob = pred / "robustness_effnet_patient.csv"
    if rob.exists():
        res["robustness"] = robustness_section(pd.read_csv(rob, dtype={"pid": str}), classes)
    if hyps:
        adj = holm({h["id"]: h["p"] for h in hyps})
        for h in hyps:
            h["p_holm"] = adj[h["id"]]
            h["verdict"] = ("supported" if h["p_holm"] < 0.05 and h["direction_ok"]
                            else "rejected (opposite direction)" if h["p_holm"] < 0.05 else "not supported")
        res["hypotheses"] = hyps
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "results.json").write_text(json.dumps(res, indent=2, default=float))
    (reports / "results.md").write_text(to_markdown(res))
    figures(res, main, reports / "figures")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    ap.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    res = build(args.artifacts_dir, args.reports_dir, args.n_boot)
    print(to_markdown(res))
    return res


if __name__ == "__main__":
    main()
