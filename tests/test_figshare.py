import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from brainguard import figshare


def _mat_v5(image, mask, label, pid) -> bytes:
    from scipy.io import savemat
    buf = io.BytesIO()
    savemat(buf, {"cjdata": {"image": image, "tumorMask": mask, "label": label, "PID": pid}})
    return buf.getvalue()


def _mat_v73(image, mask, label, pid) -> bytes:
    """Mimic MATLAB v7.3: HDF5, column-major arrays, PID stored as char codes."""
    h5py = pytest.importorskip("h5py")
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        g = f.create_group("cjdata")
        g["image"] = image.T
        g["tumorMask"] = mask.T
        g["label"] = np.array([[float(label)]])
        g["PID"] = np.array([[ord(c)] for c in pid], dtype=np.uint16)
    return buf.getvalue()


def _fake_zip(tmp_path, writer):
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"
    raw.mkdir()
    truth = {}
    with zipfile.ZipFile(raw / "brainTumorDataPublic_1-3.zip", "w") as z:
        for idx, (label, pid) in enumerate([(1, "100360"), (2, "MR040"), (3, "100360")], start=1):
            img = rng.integers(0, 900, (64, 48)).astype(np.int16)
            img[10:20, 5:15] = 3000                            # bright outlier gets clipped
            mask = np.zeros((64, 48), np.uint8)
            mask[30:40, 20:30] = 1
            z.writestr(f"{idx}.mat", writer(img, mask, label, pid))
            truth[idx] = (img, mask, label, pid)
    return raw, truth


@pytest.mark.parametrize("writer", [_mat_v5, _mat_v73])
def test_convert_reads_both_mat_formats(tmp_path, writer):
    raw, truth = _fake_zip(tmp_path, writer)
    meta = figshare.convert(raw, tmp_path / "out")
    assert list(meta["idx"]) == [1, 2, 3]
    assert list(meta["label"]) == ["meningioma", "glioma", "pituitary"]
    assert list(meta["pid"]) == ["100360", "MR040", "100360"]
    img = np.asarray(Image.open(tmp_path / "out" / meta.image_path[0]))
    mask = np.asarray(Image.open(tmp_path / "out" / meta.mask_path[0]))
    assert img.shape == (64, 48) and img.dtype == np.uint8          # orientation preserved
    assert set(np.unique(mask)) == {0, 255} and (mask[30:40, 20:30] == 255).all()
    assert meta.mask_fraction[0] == pytest.approx(100 / (64 * 48))
    assert pd.read_csv(tmp_path / "out" / "metadata.csv").shape[0] == 3
    assert "Patients with more than one tumour label: 1" in figshare.summary(meta)


def test_to_uint8_is_monotonic_and_full_range():
    x = np.arange(1000).reshape(25, 40)
    y = figshare.to_uint8(x)
    assert y.min() == 0 and y.max() == 255
    assert (np.diff(y.ravel().astype(int)) >= 0).all()
