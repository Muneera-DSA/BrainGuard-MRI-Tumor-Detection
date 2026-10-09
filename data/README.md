# Data (not committed)

| Folder | Source | How to get it |
|---|---|---|
| `kaggle/Training`, `kaggle/Testing` | Kaggle *Brain Tumor MRI Dataset* (M. Nickparvar), 7,023 JPEGs | Download the zip from Kaggle and extract it here, or pass `--kaggle-zip` to `brainguard.pipeline` |
| `figshare_raw/*.zip` | figshare brain tumour dataset (J. Cheng, CC BY 4.0) | `python -m brainguard.figshare` downloads the four zips (~880 MB) |
| `figshare/` | converted figshare: `images/`, `masks/`, `metadata.csv` | created by `python -m brainguard.figshare` |
