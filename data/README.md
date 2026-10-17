# Data (not committed)

| Folder | Source | How to get it |
|---|---|---|
| `kaggle/Training`, `kaggle/Testing` | Kaggle *Brain Tumor MRI Dataset* (M. Nickparvar), 7,023 JPEGs | Download the zip from Kaggle and extract it here, or pass `--kaggle-zip` to `brainguard.pipeline` |
| `figshare_raw/*.zip` | figshare brain tumour dataset (J. Cheng, CC BY 4.0) | `python -m brainguard.figshare` downloads the four zips (~880 MB) |
| `figshare/` | converted figshare: `images/`, `masks/`, `metadata.csv` | created by `python -m brainguard.figshare` |

## Small derived files that are committed (`results/data/`)

Used by `notebooks/walkthrough.ipynb`, so the data checks and EDA can be rerun without downloading anything:

| File | Contents |
|---|---|
| `figshare_metadata.csv` | One row per figshare slice: patient ID, tumour type, image size, tumour area. No pixels |
| `kaggle_index.csv.gz` | One row per Kaggle image: split, class, size, colour mode, file size, duplicate groups, and the matched figshare patient. No pixels |
| `examples/*.png` | One normalised slice and its tumour mask per tumour type, from the figshare dataset |

The figshare data are © J. Cheng et al., licensed CC BY 4.0: <https://doi.org/10.6084/m9.figshare.1512427>. Cite: Cheng J. et al., *PLoS ONE* 10(10), 2015.
