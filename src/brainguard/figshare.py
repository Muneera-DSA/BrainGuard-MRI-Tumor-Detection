"""Download and convert the figshare brain tumour dataset (Cheng et al.; CC BY 4.0).

    python -m brainguard.figshare            # download (~880 MB) + convert

3,064 contrast-enhanced T1 slices from 233 patients, each with a tumour mask and
a patient ID (PID). Each .mat file holds a struct `cjdata` with fields
image (int16), label (1 meningioma, 2 glioma, 3 pituitary), PID, tumorBorder,
tumorMask. Most files are MATLAB v7.3 (HDF5, read with h5py); older-format files
are read with scipy.io.

Output in data/figshare/:
    images/<idx>.png   8-bit greyscale (intensities clipped to the 0.5-99.5th percentile)
    masks/<idx>.png    tumour mask, 0 / 255
    metadata.csv       idx, pid, label, height, width, mask_fraction
"""

from __future__ import annotations

import argparse
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config


def download(raw_dir: Path = config.FIGSHARE_RAW) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, size) in config.FIGSHARE_FILES.items():
        target = raw_dir / name
        if target.exists() and target.stat().st_size == size:
            print(f"✓ {name} already downloaded")
            continue
        print(f"Downloading {name} ({size / 1e6:.0f} MB) ...")
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        if tmp.stat().st_size != size:
            raise IOError(f"{name}: expected {size} bytes, got {tmp.stat().st_size}")
        tmp.rename(target)


def _read_mat(raw: bytes) -> dict:
    """Return image, mask, label, pid from one .mat file's bytes."""
    if raw[:4] == b"\x89HDF" or raw[512:516] == b"\x89HDF":     # v7.3 = HDF5 (512-byte MATLAB header)
        try:
            import h5py
            with h5py.File(io.BytesIO(raw), "r") as f:
                cj = f["cjdata"]
                image, mask = np.asarray(cj["image"]), np.asarray(cj["tumorMask"])
                label, pid_codes = np.asarray(cj["label"]), np.asarray(cj["PID"])
        except ImportError:                                      # fall back to the system libhdf5
            import tempfile

            from ._hdf5_ctypes import read_datasets
            with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as t:
                t.write(raw)
            try:
                d = read_datasets(t.name, {"/cjdata/image": np.int16, "/cjdata/tumorMask": np.uint8,
                                           "/cjdata/label": np.float64, "/cjdata/PID": np.uint16})
            finally:
                Path(t.name).unlink()
            image, mask, label, pid_codes = (d["/cjdata/image"], d["/cjdata/tumorMask"],
                                             d["/cjdata/label"], d["/cjdata/PID"])
        return {"image": image.T, "mask": mask.T,                # MATLAB is column-major
                "label": int(label.ravel()[0]),
                "pid": "".join(chr(int(c)) for c in pid_codes.ravel() if int(c) > 0)}
    from scipy.io import loadmat
    cj = loadmat(io.BytesIO(raw), squeeze_me=True, struct_as_record=False)["cjdata"]
    return {"image": np.asarray(cj.image), "mask": np.asarray(cj.tumorMask),
            "label": int(cj.label), "pid": str(cj.PID).strip()}


def to_uint8(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float64)
    lo, hi = np.percentile(x, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1
    return (np.clip((x - lo) / (hi - lo), 0, 1) * 255).round().astype(np.uint8)


def convert(raw_dir: Path = config.FIGSHARE_RAW, out_dir: Path = config.FIGSHARE_DIR) -> pd.DataFrame:
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    zips = sorted(raw_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No .zip files in {raw_dir}; run download() first")
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for member in z.namelist():
                m = re.search(r"(\d+)\.mat$", member)
                if not m:
                    continue
                idx = int(m.group(1))
                rec = _read_mat(z.read(member))
                img = to_uint8(rec["image"])
                mask = (rec["mask"] > 0).astype(np.uint8) * 255
                Image.fromarray(img).save(out_dir / "images" / f"{idx}.png")
                Image.fromarray(mask).save(out_dir / "masks" / f"{idx}.png")
                rows.append({"idx": idx, "pid": rec["pid"], "label": config.FIGSHARE_LABELS[rec["label"]],
                             "height": img.shape[0], "width": img.shape[1],
                             "mask_fraction": float((mask > 0).mean())})
    meta = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    meta["image_path"] = "images/" + meta["idx"].astype(str) + ".png"
    meta["mask_path"] = "masks/" + meta["idx"].astype(str) + ".png"
    meta.to_csv(out_dir / "metadata.csv", index=False)
    return meta


def summary(meta: pd.DataFrame) -> str:
    per_patient = meta.groupby("pid").size()
    labels_per_patient = meta.groupby("pid")["label"].nunique()
    lines = ["# Figshare dataset summary (generated by `python -m brainguard.figshare`)", "",
             f"Images: {len(meta):,} · patients: {meta.pid.nunique():,} · "
             f"images per patient: median {per_patient.median():.0f}, max {per_patient.max()}", "",
             f"Patients with more than one tumour label: {(labels_per_patient > 1).sum()}", "",
             "| Class | Images | Patients |", "|---|---|---|"]
    for lab, g in meta.groupby("label"):
        lines.append(f"| {lab} | {len(g):,} | {g.pid.nunique():,} |")
    sizes = meta.groupby(["height", "width"]).size()
    lines += ["", "Image sizes: " + ", ".join(f"{h}×{w}: {n:,}" for (h, w), n in sizes.items())]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=config.FIGSHARE_RAW)
    ap.add_argument("--out-dir", type=Path, default=config.FIGSHARE_DIR)
    ap.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args(argv)
    if not args.skip_download:
        download(args.raw_dir)
    meta = convert(args.raw_dir, args.out_dir)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "figshare_summary.md").write_text(summary(meta))
    print(summary(meta))
    return meta


if __name__ == "__main__":
    main()
