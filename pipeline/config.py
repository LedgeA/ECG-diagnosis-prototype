"""Shared configuration for the four-class ECG image corpus build.

Every path, label definition and render parameter lives here so the stages
stay declarative and the whole build is reproducible from one file.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"

STEMI_DIR = DATASETS / "STEMI-dataset"
STEMI_RECORDS = STEMI_DIR / "ECG_row_data" / "row_data"
STEMI_TRAIN_CSV = STEMI_DIR / "CSV" / "train.csv"

PTBXL_DIR = DATASETS / "ptb-xl-dataset"
PTBXL_DB = PTBXL_DIR / "ptbxl_database.csv"

CHAPMAN_DIR = DATASETS / "chapman-dataset"
CHAPMAN_RECORDS = CHAPMAN_DIR / "WFDBRecords"

KIT = ROOT / "ecg-image-kit" / "codes" / "ecg-image-generator"
# Stage 4 shells out to the kit, so it needs the same interpreter run_all.sh
# uses. Keep the ECGKIT_PY override working in both places.
VENV_PY = Path(os.environ.get(
    "ECGKIT_PY", Path.home() / ".cache" / "ecgkit-venv" / "bin" / "python"))

BUILD = ROOT / "build"
MANIFEST_RAW = BUILD / "manifest_raw.csv"
MANIFEST_FINAL = BUILD / "manifest_final.csv"
STAGED = BUILD / "staged"        # harmonised WFDB, one subdir per shard
RENDERED = BUILD / "rendered"    # clean PNG straight from the kit
IMAGES = BUILD / "images"        # final augmented JPEG corpus
INDEX_CSV = BUILD / "index.csv"

# ---------------------------------------------------------------- classes
CLASSES = ["STEMI", "LVH", "AF", "NORMAL"]

# PTB-XL SCP codes. Presence in scp_codes counts regardless of likelihood:
# unrated statements carry likelihood 0, so a `>= 50` filter would collapse
# AFIB from 1514 to 48.
PTBXL_CLASS_CODES = {
    "LVH": {"LVH"},
    "AF": {"AFIB"},
    "NORMAL": {"NORM"},
}
PTBXL_EXCLUDE_CODES = {"PACE", "AFLT"}   # paced, atrial flutter
PTBXL_AGE_SENTINEL = 300                 # PTB-XL masks age > 89 as 300
AGE_MASKED_AS = 90                       # ...rendered as this instead

# Chapman / Ningbo SNOMED-CT codes.
SNOMED = {
    "LVH_A": "164873001",   # left ventricular hypertrophy (Shaoxing half)
    "LVH_B": "55827005",    # left ventricular hypertrophy (Ningbo half)
    "AFIB": "164889003",
    "AFL": "164890007",     # atrial flutter - excluded, never merged into AF
    "SR": "426783006",
}
CHAPMAN_CLASS_CODES = {
    "LVH": {SNOMED["LVH_A"], SNOMED["LVH_B"]},
    "AF": {SNOMED["AFIB"]},
}
# NORMAL in Chapman is strict: sinus rhythm as the *only* diagnosis.
CHAPMAN_NORMAL_EXACT = {SNOMED["SR"]}
CHAPMAN_EXCLUDE_CODES = {SNOMED["AFL"]}
CHAPMAN_BAD_RECORDS = {"JS01052"}        # malformed header: fs field is "500000/mV"

# STEMI-dataset diagnosis columns; a source-control NORMAL must be negative on
# every one of them.
STEMI_DX_COLUMNS = ["AMI", "OMI", "NSTEMI", "STEMI", "UA"]

# ...and on these as well. VF_VT is an outright malignant rhythm and must never
# be labelled NORMAL; CTO and any culprit-vessel entry imply coronary disease
# that frequently leaves ECG changes. Only 1000 controls are needed out of
# ~8800, so being strict here is free.
STEMI_CONTROL_EXCLUDE = [
    "VF_VT", "CTO", "PCI", "Prior_PCI",
    "LM", "PLAD", "MLAD", "DLAD", "DB",
    "PLCX", "MLCX", "DLCX", "OM", "PRCA", "MRCA", "DRCA",
]

# ---------------------------------------------------------------- signal
FS = 500                 # Hz, native to all three sources
N_SAMPLES = 5000         # 10 s
LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]
ADC_GAIN = 1000.0        # ADU per mV
BANDPASS = (0.05, 40.0)  # Hz - see README for why the low-pass is 40 and not 150

# ---------------------------------------------------------------- corpus
# Usable pools measured by stage 1 after every exclusion:
#   STEMI 1423 | LVH 5487 | AF 2441 | NORMAL 23139
# STEMI is the limiting class, so the larger classes are subsampled and render
# multiplicity equalises the image counts at roughly 6000 per class.
RECORDS_PER_CLASS = {"STEMI": 1423, "LVH": 3000, "AF": 2000, "NORMAL": 3000}
RENDERS_PER_RECORD = {"STEMI": 4, "LVH": 2, "AF": 3, "NORMAL": 2}

# NORMAL is drawn in equal parts from all three sources. The Chongqing third is
# what stops source identity from being a perfect proxy for the STEMI class.
NORMAL_SOURCE_QUOTA = {"stemi": 1000, "ptbxl": 1000, "chapman": 1000}

# Staged records are written in fixed-size chunks so stage 4 can hand the kit
# many small batches and keep every core busy. Each kit invocation pays a ~7 s
# TensorFlow import, so chunks much smaller than this waste real time.
CHUNK_SIZE = 400
WORKERS = 8

SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}
PTBXL_TEST_FOLDS = {10}
PTBXL_VAL_FOLDS = {9}
SEED = 20260818

# Every staged record is renamed to this scheme. The kit prints the record name
# on the sheet as "ID: <name>", and the native names are source-identifying
# (00001_hr / JS00001 / 00101), which would hand the model a free source label.
RECORD_NAME_FMT = "ECG{:06d}"

# ---------------------------------------------------------------- render
# 150 DPI -> 5.91 px/mm -> 147.6 px/s at 25 mm/s -> image Nyquist 73.8 Hz, a
# 1.8x margin over the 40 Hz signal band, and the kit's own recommended floor
# for digitisation. Measured 739 KB/image vs 1.2 MB at 200 DPI.
DPI = 150
# The kit fixes 25 mm/s and 10 mm/mV and prints them on the sheet; these are
# recorded here for the DPI/Nyquist arithmetic above, not to configure it.
PAPER_SPEED = 25         # mm/s
PAPER_GAIN = 10          # mm/mV
MARGIN_PX = 18           # thin margin, applied by stage 5 (kit only does int inches)
# The 1 mm grid, not the added noise, is what resists compression: a clean sheet
# is 2.5 MB at q92/4:4:4 even with zero noise. q85 with 4:2:0 chroma costs
# nothing diagnostically (colour carries no signal here) and halves the corpus.
JPEG_QUALITY = 85
JPEG_SUBSAMPLING = 2     # 4:2:0

# Flags passed to gen_ecg_images_from_data_batch.py. The kit's own distortion
# layer (--augment / --wrinkles) is deliberately NOT used; see stage5.
KIT_RENDER_FLAGS = [
    "--num_columns", "4",        # 3x4 layout
    "--full_mode", "II",         # continuous lead II rhythm strip
    "-r", str(DPI),
    "--pad_inches", "0",         # int-only in the kit; margin added in stage 5
    "--print_header",            # ID / age / sex from header comments
    "--calibration_pulse", "1",
    # Grey grid, matching a real BTL CardioPoint printout: major (0.4,0.4,0.4),
    # minor (0.75,0.75,0.75), black trace. In the kit this palette is only
    # reachable through its "bw" style, which --random_bw 1 selects for every
    # image. It is a grid-colour switch here, not a greyscale conversion.
    "--random_bw", "1",
    "--store_config", "1",
    "--lead_bbox",
    "--lead_name_bbox",
]
