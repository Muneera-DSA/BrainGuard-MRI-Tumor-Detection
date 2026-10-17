"""End-to-end check of the deep-learning stages on tiny synthetic images.

Runs in CI with TensorFlow installed (skipped otherwise). It verifies that training,
robustness, explanations and the report fit together; it says nothing about real accuracy.
"""

import json

import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")

from brainguard import audit, crosswalk, explain, report, robustness, train  # noqa: E402
from synthetic import make_figshare, make_kaggle  # noqa: E402


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    fig_root, kag_root, art, rep = tmp / "figshare", tmp / "kaggle", tmp / "art", tmp / "rep"
    fig = make_figshare(fig_root, n_patients=21, slices=4)
    make_kaggle(kag_root, fig_root, fig)
    audit.main(["--data", str(kag_root), "--reports-dir", str(rep), "--artifacts-dir", str(art)])
    crosswalk.main(["--kaggle-dir", str(kag_root), "--figshare-dir", str(fig_root),
                    "--artifacts-dir", str(art), "--reports-dir", str(rep)])
    out = train.main(["--experiments", "effnet_patient", "effnet_image", "cnn_patient", "kaggle_effnet", "resnet50_final",
                      "--figshare-dir", str(fig_root), "--kaggle-dir", str(kag_root), "--artifacts-dir", str(art),
                      "--quick", "--img-size", "64", "--no-pretrained"])
    return {"fig_root": fig_root, "art": art, "rep": rep, "out": out, "fig": fig}


def test_cross_validation_outputs(run):
    p = run["out"]["effnet_patient"]
    assert len(p) == len(run["fig"]) and p.idx.is_unique
    probs = p[[c for c in p.columns if c.startswith("p_")]].to_numpy()
    assert np.allclose(probs.sum(1), 1, atol=1e-3)
    assert p.groupby("pid").fold.nunique().max() == 1                      # no patient in two folds
    assert (run["art"] / "models" / "effnet_patient" / "fold0.keras").exists()
    hist = json.loads((run["art"] / "predictions" / "effnet_patient" / "fold0_history.json").read_text())
    assert hist["n_eval"] > 0 and hist["epochs"] >= 2
    k = run["out"]["kaggle_effnet"]
    assert set(k.split) == {"test"} and "copy_in_train" in k


def test_resume_skips_finished_folds(run, capsys):
    train.main(["--experiments", "effnet_patient", "--figshare-dir", str(run["fig_root"]),
                "--artifacts-dir", str(run["art"]), "--quick", "--img-size", "64", "--no-pretrained"])
    assert "already done" in capsys.readouterr().out


def test_perturbations_change_inputs_but_keep_shape():
    img = tf.random.uniform((64, 64, 1), 0, 255)
    for name, spec in robustness.PERTURBATIONS.items():
        if spec is None:
            continue
        out = robustness.make_perturbation(*spec)(img)
        assert out.shape == img.shape, name
        assert float(tf.reduce_mean(tf.abs(out - img))) > 0, name


def test_rotation_is_a_rotation():
    img = np.zeros((33, 33, 1), np.float32)
    img[16, 28] = 255                                                      # point right of centre
    out = robustness.make_perturbation("rotate", 90.0)(tf.constant(img)).numpy()[..., 0]
    y, x = np.unravel_index(out.argmax(), out.shape)
    assert abs(x - 16) <= 1 and abs(y - 16) >= 10                           # moved to above/below centre


def test_robustness_and_explain_and_report(run):
    art, fig_root = run["art"], run["fig_root"]
    pred = pd.read_csv(art / "predictions" / "effnet_patient.csv", dtype={"pid": str})
    pred["abs_path"] = [str(fig_root / "images" / f"{i}.png") for i in pred.idx]
    subset = {k: robustness.PERTURBATIONS[k] for k in ["clean", "rotation 10 deg", "JPEG quality 30"]}
    rob = robustness.run(art / "models" / "effnet_patient", pred, 64, 16, subset)
    rob.to_csv(art / "predictions" / "robustness_effnet_patient.csv", index=False)
    assert set(rob.perturbation) == set(subset) and len(rob) == 3 * len(pred)

    res = explain.run(pred, fig_root, art / "models" / "effnet_patient", art / "explain" / "effnet_patient", 64,
                      n_images=9, n_background=6, n_samples=12, n_sanity=4)
    assert len(res) == 9 and res.shap_energy_in_mask.between(0, 1).all()
    # expected-gradients completeness holds in expectation: same sign and order as f(x) - E f(b)
    r = np.corrcoef(res.completeness_sum_shap, res.completeness_f_minus_baseline)[0, 1]
    assert np.isnan(r) or r > 0
    assert (art / "explain" / "effnet_patient" / "explain_sanity.csv").exists()
    explain.plot_examples(art / "explain" / "effnet_patient" / "explain_examples.npz", run["rep"] / "figures" / "explain_examples.png")
    assert (run["rep"] / "figures" / "explain_examples.png").exists()


    # the final model (ResNet50 with its own preprocessing layer) must save, reload and be explainable
    from brainguard.tfdata import load_array
    final = pd.read_csv(art / "predictions" / "resnet50_final.csv", dtype={"pid": str})
    final["abs_path"] = [str(fig_root / "images" / f"{i}.png") for i in final.idx]
    model = explain.load_fold_model(art / "models" / "resnet50_final" / "fold0.keras")
    assert type(model.get_layer("preprocess")).__name__ == "Preprocess"
    part = final[final.fold == 0]
    p = model.predict(load_array(list(part.abs_path), 64), verbose=0)
    np.testing.assert_allclose(p, part[[c for c in part.columns if c.startswith("p_")]].to_numpy(), atol=1e-4)   # reload is exact
    rob_f = robustness.run(art / "models" / "resnet50_final", final, 64, 16, subset)
    rob_f.to_csv(art / "predictions" / "robustness_resnet50_final.csv", index=False)
    res_f = explain.run(final, fig_root, art / "models" / "resnet50_final", art / "explain" / "resnet50_final", 64,
                        n_images=9, n_background=6, n_samples=12, n_sanity=4)
    assert len(res_f) == 9 and res_f.cam_energy_in_mask.dropna().between(0, 1).all()

    out = report.build(art, run["rep"], n_boot=50)
    assert "main" in out and "explain" in out and "robustness" in out and "kaggle" in out
    assert out["main_model"]["experiment"] == "resnet50_final" and "explain_reference" in out and "robustness_reference" in out
    assert any(h["id"] == "H1" for h in out["hypotheses"])


def test_explanations_work_under_mixed_precision():
    """Colab trains with mixed_float16; Grad-CAM and expected gradients must cope, for every backbone."""
    import keras

    from brainguard.models import build
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        x = np.random.default_rng(0).uniform(0, 255, (2, 64, 64, 3)).astype("float32")
        for name in ("effnet", "resnet50", "densenet121"):
            model = build(name, 3, 64, weights=None)
            cam = explain.grad_cam(model, x, np.array([0, 2]))
            eg = explain.expected_gradients(model, x, np.array([0, 2]), x, n_samples=4)
            assert cam.shape == (2, 64, 64) and np.isfinite(cam).all(), name
            assert eg.shape == (2, 64, 64) and np.isfinite(eg).all(), name
            keras.backend.clear_session()
    finally:
        keras.mixed_precision.set_global_policy("float32")


def test_architecture_screen_backbones_build_and_fine_tune():
    """Every backbone in the screen builds, predicts valid probabilities, and unfreezes only its top layers
    (normalisation layers stay frozen), under mixed precision as on Colab."""
    import keras

    from brainguard.models import BACKBONES, backbone, build, unfreeze_top
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        x = np.random.default_rng(0).uniform(0, 255, (2, 64, 64, 3)).astype("float32")
        for name in BACKBONES:
            model = build(name, 3, 64, weights=None)
            p = model.predict(x, verbose=0)
            assert p.shape == (2, 3) and np.allclose(p.sum(1), 1, atol=1e-3), name
            base = backbone(model)
            assert not any(layer.trainable for layer in base.layers), name
            unfreeze_top(model)
            trainable = [layer for layer in base.layers if layer.trainable]
            assert 0 < len(trainable) < len(base.layers), name
            assert not any(isinstance(layer, (keras.layers.BatchNormalization, keras.layers.LayerNormalization)) for layer in trainable), name
            keras.backend.clear_session()
    finally:
        keras.mixed_precision.set_global_policy("float32")


def test_preprocess_layer_matches_keras_preprocessing():
    """The saveable Preprocess layer computes exactly what the backbone's preprocess_input does."""
    import keras

    from brainguard.models import BACKBONES, Preprocess
    x = np.random.default_rng(1).uniform(0, 255, (2, 8, 8, 3)).astype("float32")
    for name, (_, prep) in BACKBONES.items():
        if prep is None:
            continue
        np.testing.assert_allclose(keras.ops.convert_to_numpy(Preprocess(name)(x)), keras.ops.convert_to_numpy(prep(x.copy())), rtol=1e-6)
