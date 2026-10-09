import json

import numpy as np
import pandas as pd

from brainguard import config, report


def _preds(n_pat=60, per=8, acc=0.85, seed=0, classes=config.TUMOUR_CLASSES):
    rng = np.random.default_rng(seed)
    rows = []
    idx = 0
    for p in range(n_pat):
        y = p % len(classes)
        for _ in range(per):
            idx += 1
            right = rng.random() < acc
            pred = y if right else (y + 1) % len(classes)
            pr = np.full(len(classes), 0.05)
            pr[pred] = 0.9
            rows.append({"idx": idx, "pid": f"P{p}", "y": y, "label": classes[y], "fold": p % 5,
                         **{f"p_{c}": pr[i] for i, c in enumerate(classes)}})
    return pd.DataFrame(rows)


def test_report_end_to_end(tmp_path):
    art = tmp_path / "art"
    (art / "predictions").mkdir(parents=True)
    (art / "explain").mkdir()
    main = _preds(acc=0.85, seed=0)
    main.to_csv(art / "predictions" / "effnet_patient.csv", index=False)
    _preds(acc=0.97, seed=1).to_csv(art / "predictions" / "effnet_image.csv", index=False)
    _preds(acc=0.70, seed=2).to_csv(art / "predictions" / "cnn_patient.csv", index=False)
    _preds(acc=0.85, seed=3).to_csv(art / "predictions" / "effnet_patient_cw.csv", index=False)
    k = _preds(n_pat=40, acc=0.9, seed=4, classes=config.KAGGLE_CLASSES)
    k["copy_in_train"] = np.arange(len(k)) % 2 == 0
    k.loc[k.copy_in_train, [f"p_{c}" for c in config.KAGGLE_CLASSES]] = np.eye(4)[k.loc[k.copy_in_train, "y"]]
    k["size_group"] = "512x512"
    k.to_csv(art / "predictions" / "kaggle_effnet.csv", index=False)
    rob = pd.concat([main.assign(perturbation="clean"),
                     _preds(acc=0.6, seed=5).assign(perturbation="gaussian noise sd 16")])
    rob.to_csv(art / "predictions" / "robustness_effnet_patient.csv", index=False)
    rng = np.random.default_rng(6)
    n = 60
    ex = pd.DataFrame({"idx": range(n), "pid": [f"P{i % 20}" for i in range(n)], "fold": 0,
                       "label": "glioma", "predicted": "glioma", "correct": rng.random(n) < 0.8,
                       "mask_fraction": rng.uniform(0.01, 0.05, n),
                       "shap_energy_in_mask": rng.uniform(0.2, 0.6, n), "shap_pointing_hit": rng.random(n) < 0.7,
                       "cam_energy_in_mask": rng.uniform(0.1, 0.5, n), "cam_pointing_hit": rng.random(n) < 0.6,
                       "pointing_chance": rng.uniform(0.02, 0.08, n),
                       "completeness_sum_shap": np.linspace(0, 1, n), "completeness_f_minus_baseline": np.linspace(0, 1, n)})
    ex.to_csv(art / "explain" / "explain_metrics.csv", index=False)
    pd.DataFrame({"shap_similarity_trained_vs_random": rng.uniform(0, 0.2, 10),
                  "cam_similarity_trained_vs_random": rng.uniform(0, 0.2, 10),
                  "trained_shap_energy_in_mask": rng.uniform(0.3, 0.5, 10),
                  "random_shap_energy_in_mask": rng.uniform(0.0, 0.1, 10)}).to_csv(art / "explain" / "explain_sanity.csv", index=False)

    res = report.build(art, tmp_path / "rep", n_boot=200)
    h = {x["id"]: x for x in res["hypotheses"]}
    assert h["H1"]["verdict"] == "supported"                      # image split looks better (leakage)
    assert h["H2"]["verdict"] == "rejected (opposite direction)" or h["H2"]["direction_ok"]
    assert h["H6"]["verdict"] == "supported"
    assert res["kaggle"]["copy_in_train"]["acc_flagged"] == 1.0
    m = res["main"]
    assert m["accuracy"]["ci"][0] < m["accuracy"]["value"] < m["accuracy"]["ci"][1]
    assert abs(m["accuracy"]["value"] - 0.85) < 0.06
    assert res["robustness"][1]["drop_vs_clean"] > 0.1
    for f in ["results.md", "results.json", "figures/confusion_matrix.png", "figures/calibration.png",
              "figures/protocol_comparison.png", "figures/robustness.png"]:
        assert (tmp_path / "rep" / f).exists(), f
    json.loads((tmp_path / "rep" / "results.json").read_text())


def test_report_handles_partial_runs(tmp_path):
    art = tmp_path / "art"
    (art / "predictions").mkdir(parents=True)
    _preds().to_csv(art / "predictions" / "effnet_patient.csv", index=False)
    res = report.build(art, tmp_path / "rep", n_boot=50)
    assert "main" in res and "kaggle" not in res
