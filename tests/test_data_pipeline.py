import numpy as np
import pandas as pd

from brainguard import audit, crosswalk, shortcut, splits
from synthetic import make_figshare, make_kaggle


def _build(tmp_path):
    fig_root, kag_root = tmp_path / "figshare", tmp_path / "kaggle"
    fig = make_figshare(fig_root)
    make_kaggle(kag_root, fig_root, fig)
    return fig_root, kag_root, fig


def test_audit_finds_planted_duplicates_and_size_confounding(tmp_path):
    _, kag_root, fig = _build(tmp_path)
    df, r = audit.main(["--data", str(kag_root), "--reports-dir", str(tmp_path / "rep"),
                        "--artifacts-dir", str(tmp_path / "art")])
    planted = sum(1 for i in range(len(fig)) if i % 8 == 0)
    assert r["near"]["groups_spanning_train_and_test"] >= planted
    assert r["near"]["groups_with_conflicting_labels"] == 0
    assert r["size_by_class"]["notumor"].get("512x512", 0) == 0
    assert (tmp_path / "rep" / "kaggle_audit.md").exists()
    # every duplicate group is internally consistent
    assert df.groupby("exact_group")["dup_group"].nunique().max() == 1


def test_crosswalk_recovers_patient_ids_and_overlap(tmp_path):
    fig_root, kag_root, fig = _build(tmp_path)
    audit.main(["--data", str(kag_root), "--reports-dir", str(tmp_path / "rep"), "--artifacts-dir", str(tmp_path / "art")])
    cw, r = crosswalk.main(["--kaggle-dir", str(kag_root), "--figshare-dir", str(fig_root),
                            "--artifacts-dir", str(tmp_path / "art"), "--reports-dir", str(tmp_path / "rep")])
    assert r["matched_share_of_tumour_images"] == 1.0
    assert r["matched_no_tumour_images"] == 0
    assert r["label_agreement_share"] == 1.0
    assert r["patients_in_both"] > 0 and r["test_images_from_training_patients"] > 0
    tumour = cw[cw.label != "notumor"]
    truth = fig.set_index("idx").pid
    from_name = tumour.filename.str.extract(r"(\d+)\.jpg$")[0].astype(int)
    assert (tumour.pid.to_numpy() == truth.loc[from_name].to_numpy()).all()


def test_shortcut_detects_source_confounding(tmp_path):
    _, kag_root, _ = _build(tmp_path)
    audit.main(["--data", str(kag_root), "--reports-dir", str(tmp_path / "rep"), "--artifacts-dir", str(tmp_path / "art")])
    df = pd.read_csv(tmp_path / "art" / "kaggle_index.csv")
    df.loc[df.label != "notumor", ["width", "height"]] = 512      # mimic: tumour images 512x512, others not
    r = shortcut.run(df)
    assert r["binary"]["auc"]["auc"] > 0.95
    assert r["rule_not_512_means_no_tumour"]["accuracy"] == 1.0


def test_patient_folds_never_split_a_patient(tmp_path):
    fig = make_figshare(tmp_path / "f", n_patients=30, slices=5)
    f = splits.patient_folds(fig, n_splits=5)
    assert pd.Series(f).groupby(fig.pid).nunique().max() == 1
    assert set(np.unique(f)) == set(range(5))
    g = splits.image_folds(fig, n_splits=5)
    assert pd.Series(g).groupby(fig.pid).nunique().max() > 1      # image folds DO split patients
