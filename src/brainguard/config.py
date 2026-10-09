"""Central configuration. Every choice a reviewer may question lives here."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
KAGGLE_DIR = DATA_DIR / "kaggle"            # Kaggle "Brain Tumor MRI Dataset": Training/ and Testing/
FIGSHARE_RAW = DATA_DIR / "figshare_raw"    # the four brainTumorDataPublic_*.zip files
FIGSHARE_DIR = DATA_DIR / "figshare"        # converted: images/, masks/, metadata.csv
REPORTS_DIR = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"  # models and predictions (not committed)

SEED = 42

# --- Kaggle dataset -----------------------------------------------------------------
KAGGLE_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NO_TUMOUR = "notumor"
SPLIT_DIRS = {"Training": "train", "Testing": "test"}

# --- Figshare dataset (Cheng et al., CC BY 4.0) -----------------------------------
FIGSHARE_ARTICLE = 1512427
FIGSHARE_FILES = {   # name -> (download URL, size in bytes)
    "brainTumorDataPublic_1-766.zip": ("https://ndownloader.figshare.com/files/3381290", 214401279),
    "brainTumorDataPublic_767-1532.zip": ("https://ndownloader.figshare.com/files/3381296", 217848429),
    "brainTumorDataPublic_1533-2298.zip": ("https://ndownloader.figshare.com/files/3381293", 215563856),
    "brainTumorDataPublic_2299-3064.zip": ("https://ndownloader.figshare.com/files/3381302", 231679762),
}
FIGSHARE_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}
TUMOUR_CLASSES = ["glioma", "meningioma", "pituitary"]   # class order used by every model

# --- Duplicate / leakage audit ---------------------------------------------------
# Two images are treated as the same underlying scan when BOTH
#   (a) their 63-bit DCT perceptual hashes differ in <= CANDIDATE_HAMMING bits, and
#   (b) the Pearson correlation of their 64x64 greyscale pixels is >= DUP_CORRELATION.
# Hash similarity alone over-merges (different brains in the same view collide, and
# transitive chains then link different tumour types); requiring pixel correlation
# removes that: at >= 0.95 only 1 of ~7,600 candidate pairs carries different labels.
# Sensitivity to the correlation cut-off is reported.
CANDIDATE_HAMMING = 12
DUP_CORRELATION = 0.95
CORRELATION_SENSITIVITY = [0.90, 0.95, 0.98, 0.995]
# Kaggle image <-> figshare image match (to recover patient IDs for Kaggle images)
CROSSWALK_CORRELATION = 0.95

# --- Modelling --------------------------------------------------------------------
IMG_SIZE = 224
BATCH_SIZE = 32
CV_FOLDS = 5
MAX_EPOCHS = 25
EARLY_STOP_PATIENCE = 5
LEARNING_RATE = 1e-3          # head training
FINE_TUNE_LR = 1e-4           # after unfreezing the top of the backbone
FINE_TUNE_AT_EPOCH = 5        # epochs of head-only training before unfreezing

# --- Explainability ---------------------------------------------------------------
EXPLAIN_N_IMAGES = 150        # held-out images explained (stratified by class)
SHAP_BACKGROUND = 50          # background images for expected gradients
SHAP_SAMPLES = 200            # gradient samples per explained image
