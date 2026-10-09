"""Run the whole study end to end (designed for one Google Colab GPU session).

    python -m brainguard.pipeline --kaggle-zip Brain_tumor_Image_dataset.zip --workdir /content/drive/MyDrive/brainguard

Stages (each skips work already saved in --workdir, so a disconnected session resumes):
  1. data      unzip the Kaggle dataset; download + convert figshare
  2. audit     Kaggle duplicates, shortcut test, Kaggle<->figshare patient crosswalk   (CPU)
  3. train     all experiments in brainguard.train                                    (GPU)
  4. robust    robustness of the main model                                           (GPU)
  5. explain   SHAP + Grad-CAM + localisation tests                                   (GPU)
  6. report    reports/results.md and figures                                          (CPU)
  7. bundle    results_bundle.zip: reports + predictions (no images, no models)
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from . import audit, config, crosswalk, figshare, report, shortcut


def _find_kaggle_root(extract_dir: Path) -> Path:
    for p in [extract_dir, *extract_dir.rglob("*")]:
        if p.is_dir() and (p / "Training").is_dir() and (p / "Testing").is_dir():
            return p
    raise FileNotFoundError("Could not find Training/ and Testing/ inside the Kaggle zip")


def prepare_data(kaggle_zip: Path | None, data_dir: Path) -> tuple[Path, Path]:
    kaggle_dir = data_dir / "kaggle"
    if not (kaggle_dir / "Training").exists():
        if kaggle_zip is None:
            raise SystemExit("Pass --kaggle-zip (the Kaggle 'Brain Tumor MRI Dataset' zip)")
        tmp = data_dir / "_kaggle_extract"
        with zipfile.ZipFile(kaggle_zip) as z:
            z.extractall(tmp)
        root = _find_kaggle_root(tmp)
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("Training", "Testing"):
            shutil.move(str(root / sub), str(kaggle_dir / sub))
        shutil.rmtree(tmp, ignore_errors=True)
    fig_dir = data_dir / "figshare"
    if not (fig_dir / "metadata.csv").exists():
        figshare.download(data_dir / "figshare_raw")
        figshare.convert(data_dir / "figshare_raw", fig_dir)
    return kaggle_dir, fig_dir


def bundle(workdir: Path, reports: Path, art: Path) -> Path:
    target = workdir / "results_bundle.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for base, sub in [(reports, ""), (art / "predictions", "predictions"), (art / "explain", "explain")]:
            if base.exists():
                for f in base.rglob("*"):
                    if f.is_file() and f.suffix in {".md", ".json", ".csv", ".png", ".npz"}:
                        z.write(f, Path("results") / (sub or "reports") / f.relative_to(base))
    return target


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kaggle-zip", type=Path)
    ap.add_argument("--workdir", type=Path, default=config.PROJECT_ROOT,
                    help="artifacts/ and reports/ go here (use Google Drive on Colab so runs can resume)")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="images go here (default: <workdir>/data; use fast local disk on Colab)")
    ap.add_argument("--stages", nargs="+", default=["data", "audit", "train", "robust", "explain", "report", "bundle"])
    ap.add_argument("--quick", action="store_true", help="tiny run to check the pipeline (minutes)")
    ap.add_argument("--redo", nargs="*", default=[], help="experiments to retrain from scratch, e.g. --redo cnn_patient")
    args = ap.parse_args(argv)

    w = args.workdir
    data_dir, art, reports = (args.data_dir or w / "data"), w / "artifacts", w / "reports"
    for d in (data_dir, art, reports):
        d.mkdir(parents=True, exist_ok=True)
    kaggle_dir, fig_dir = data_dir / "kaggle", data_dir / "figshare"
    common = ["--artifacts-dir", str(art)]

    if "data" in args.stages:
        kaggle_dir, fig_dir = prepare_data(args.kaggle_zip, data_dir)
        (reports / "figshare_summary.md").write_text(
            figshare.summary(__import__("pandas").read_csv(fig_dir / "metadata.csv", dtype={"pid": str})))
    if "audit" in args.stages:
        audit.main(["--data", str(kaggle_dir), "--reports-dir", str(reports), *common])
        shortcut.main(["--reports-dir", str(reports), *common])
        crosswalk.main(["--kaggle-dir", str(kaggle_dir), "--figshare-dir", str(fig_dir), "--reports-dir", str(reports), *common])
    quick = ["--quick"] if args.quick else []
    if "train" in args.stages:
        from . import train
        redo = ["--redo", *args.redo] if args.redo else []
        train.main(["--figshare-dir", str(fig_dir), "--kaggle-dir", str(kaggle_dir), *common, *quick, *redo])
    if "robust" in args.stages:
        from . import robustness
        robustness.main(["--figshare-dir", str(fig_dir), *common])
    if "explain" in args.stages:
        from . import explain
        extra = ["--n-images", "30", "--n-samples", "50"] if args.quick else []
        explain.main(["--figshare-dir", str(fig_dir), "--reports-dir", str(reports), *common, *extra])
    if "report" in args.stages:
        report.main(["--reports-dir", str(reports), *common])
    if "bundle" in args.stages:
        print(f"Results bundle: {bundle(w, reports, art)}")


if __name__ == "__main__":
    main()
