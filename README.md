# ECG Diagnosis Prototype — synthetic paper-ECG corpus

Generates a labelled dataset of **synthetic 12-lead paper ECG images** for
training a direct 2D vision model. Four classes:

| Class | Meaning |
|---|---|
| `STEMI` | ST-elevation myocardial infarction |
| `LVH` | Left ventricular hypertrophy |
| `AF` | Atrial fibrillation (flutter deliberately excluded) |
| `NORMAL` | No target abnormality |

Waveforms come from three public WFDB datasets; images are rendered with
[ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit) and then put
through a custom paper/scan augmentation pass.

---

## Quick start

```bash
bash install.sh                 # one-time: build the Python 3.11 venv
cd pipeline && ./run_all.sh     # ~4 h on 8 cores, ~17.5 GB output
```

You do **not** need to activate the venv. `run_all.sh` invokes the correct
interpreter by absolute path — which matters, because the kit will not run on a
modern default Python.

Output lands in `build/`:

```
build/images/<split>/<class>/ECG######_r<k>.jpg
build/index.csv    image_path, record, uid, cls, split, source, render_k, age, sex, ...
```

Train from `index.csv`; it is the single source of truth for labels and splits.

---

## Output specification

| Property | Value |
|---|---|
| Layout | 3×4 with continuous lead II rhythm strip |
| Paper speed | 25 mm/s |
| Gain | 10 mm/mV |
| Signal band | 0.05–40 Hz |
| Resolution | 150 DPI → 1686 × 1311 px |
| Format | JPEG q85, 4:2:0 |
| Grid | grey — major `(0.4,0.4,0.4)`, minor `(0.75,0.75,0.75)`, black trace |
| Metadata printed | anonymised ID, age, sex, inset from the corner |
| Calibration pulse | on every sheet |
| Margin | thin (18 px) |
| Corpus | 23,692 images from 9,423 records |

Each sheet then gets **paper-level environmental noise** — the damage a physical
printout accumulates, not noise on the waveform:

| Effect | Range |
|---|---|
| Paper texture | photographs of real wrinkled paper, as a shading field (every sheet) |
| Creasing | 1–3 hard fold lines with a bright ridge and shadow side, plus broad bowing (80%) |
| Tilt | ±3°, rotated on an expanded canvas filled with the sheet's own paper colour |
| Perspective | slight keystone, as if photographed rather than scanned (70%) |
| Lighting | directional ramp, corner falloff, and an off-sheet edge shadow |
| Exposure / contrast / white balance | mild, neutral-centred |
| Sensor noise | σ 1.5–4.5 |

The wrinkle texture comes from the 19 photographs the kit ships in
`CreasesWrinkles/wrinkles-dataset/` — the same source its own `--wrinkles` uses.
They are randomly cropped, flipped and rotated, then applied as a *multiplicative
shading field* rather than the kit's full-strength overlay blend. That is what
crumpled paper physically does to reflected light, and it keeps the darkest fold
far above the trace. Texture, hard creases and lighting are layered in that
order, because real paper shows all three.

All of it is bounded so the trace survives. Across a 48-variant sweep, paper-to-
trace contrast stayed between **230 and 252** of a clean sheet's 250 greyscale
levels, and ink coverage between 1.14% and 3.68% against a clean 1.31%. Stage 6
re-checks this on the built corpus.

Per class: 5,692 STEMI and 6,000 each of LVH, AF, NORMAL. Split 80/10/10,
grouped by patient.

---

## Data sources

| Source | Records | Supplies | Licence |
|---|---:|---|---|
| [STEMI dataset](https://www.nature.com/articles/s41597-026-07278-0) (Chongqing) | 19,955 | STEMI + NORMAL controls | CC BY-NC-**ND** 4.0 |
| [PTB-XL](https://physionet.org/content/ptb-xl/) | 21,799 | LVH, AF, NORMAL | CC BY 4.0 |
| [Chapman–Shaoxing + Ningbo](https://physionet.org/content/ecg-arrhythmia/) | 45,019 | LVH, AF, NORMAL | CC BY 4.0 |

Place them under `datasets/` as `STEMI-dataset/`, `ptb-xl-dataset/`,
`chapman-dataset/`. All three are gitignored.

> **Licensing.** The STEMI dataset is CC BY-NC-**ND**. The rendered corpus is a
> derivative work, so it may be used internally but **not redistributed**.

---

## How it works

Six stages, each reading the previous one's output. All are idempotent, so an
interrupted build resumes with `./run_all.sh <n>`.

| | Stage | Does |
|---|---|---|
| 0 | `patch_kit.py` | Insets the kit's printed header away from the paper corner |
| 1 | `stage1_manifest.py` | Resolve labels from all three catalogues; screen out unusable records |
| 2 | `stage2_select.py` | Subsample to target, patient-grouped 80/10/10 split per class |
| 3 | `stage3_transcode.py` | Harmonise to one WFDB dialect, band-pass, anonymise |
| 4 | `stage4_render.py` | Render clean sheets with the kit, in parallel |
| 5 | `stage5_augment.py` | Rotation, illumination, noise, margin, JPEG encode |
| 6 | `stage6_verify.py` | Integrity, balance, pixel stats, source-leakage probe |

Every tunable lives in `pipeline/config.py`.

**Checking a render's annotations.** `visualize_annotations.py`, at the repo
root, draws each lead's bounding box, name-label box, and (with
`--show_pixels`) its traced signal pixels, reading the JSON sidecar
`--store_config 1` writes next to every clean render:

```bash
python visualize_annotations.py -i build/rendered/STEMI/r0/c000/ECG000001-0.png
python visualize_annotations.py -i build/rendered/STEMI/r0/c000/ECG000001-0.png --show_pixels
```

This only works against `build/rendered/` — stage 5's rotation, keystone,
shading and margin move every pixel without touching the JSON, so a box that's
correct there is wrong on the final `build/images/` JPEGs. The script checks
the image's dimensions against what the JSON recorded and refuses to draw on a
mismatch, rather than silently drawing a plausible but wrong box. Output
defaults next to the input image; if that would land inside `build/rendered/`
it redirects to `build/annotated/` instead, since stage 4's resume logic counts
PNGs per chunk directory to decide what's already done, and a stray file there
would throw that off.

**Only the four target classes are ever rendered.** Stage 1 discards everything
else — roughly 21,000 of ~31,000 candidates are never transcoded or stored.

---

## Design decisions a collaborator should know

**The band-pass is 0.05–40 Hz, not the diagnostic 0.05–150 Hz.** At 150 DPI and
25 mm/s the paper carries 147.6 px/s, so image Nyquist is 73.8 Hz — 150 Hz
cannot be represented and would only alias against the 1 mm grid. Nothing in
these four classes lives above 40 Hz. Measured cost is 5.8% of QRS amplitude,
applied uniformly to every class.

**Amplitude is never normalised.** It is the obvious cross-source harmonisation
move and it would erase the voltage criteria that define LVH. Only filtering and
baseline are harmonised.

**Records are renamed `ECG000001…`.** The kit prints `ID: <record name>` on every
sheet, and the native names (`00001_hr` / `JS00001` / `00101`) announce which
dataset a sheet came from.

**The kit's `--augment` and `--wrinkles` flags are not used, but its wrinkle
*textures* are.** Both flags were tested and rejected; `stage5_augment.py`
documents the defects. The two that matter:

- `-t/--temperature` is dead code. `gen_ecg_image_from_data.py` hardcodes
  `temp = randrange(2000,4000)` or `randrange(10000,20000)`, so every augmented
  sheet gets a forced orange or blue cast. `--deterministic_temp` does not help:
  it is declared in argparse and never read anywhere in the kit.
- `--wrinkles` composites its texture with a full-strength overlay blend
  (`low = 2·img·tex`, `high = 1−2(1−img)(1−tex)`), which buries the trace even
  at `-nv 1 -nh 1`.

Stage 5 therefore loads the same photographs directly and applies them as a
bounded multiplicative shading field, which keeps the realism and drops the
damage.

**The kit is patched, and the patch is scripted.** `ecg-image-kit/` is
gitignored, so a fresh clone is stock upstream. `patch_kit.py` re-applies the
one change we need — insetting the printed header — idempotently, and stage 4
refuses to run without it. Do not hand-edit the kit; add to `patch_kit.py`.

**The grey grid is the kit's `bw` style**, selected with `--random_bw 1`. That
flag name is misleading: it picks a grid *palette*, it does not convert the
image to greyscale. It is the only route to the grey/black palette that matches
a real printout, since `--standard_grid_color` only offers brown/pink/blue/green/red.

**A patient never appears under two labels or in two splits.**

---

## Known limitations

**STEMI is single-source.** Every STEMI image comes from the Chongqing cohort, so
source identity partially predicts the label and a model could learn the
acquisition fingerprint instead of ST elevation. Mitigations: identical signal
conditioning, class-independent render styling, anonymised IDs, and one third of
NORMAL drawn from that same cohort (tagged `source_control`, so it can be
ablated). Stage 6's probe measures what survives — **treat a large margin over
chance as a reason to distrust the STEMI numbers**, and report per-source metrics
regardless. This cannot be fully eliminated with these three datasets, and should
be disclosed in any write-up.

**The local PTB-XL copy is incomplete and partly corrupt.** 19,490 of 21,799
signal files, missing in contiguous blocks (subdir `14000` empty, `13000` holds
38), plus 19 zero-byte headers and 40 truncated `.dat` files. The STEMI dataset
has 2 truncated files. Stage 1 screens all of these; pools remain large enough.
Re-download `records500` to recover them.

**Chapman has no patient identifier**, so each record is treated as its own
patient. If the Ningbo half repeats patients, some leakage survives the split.

**`55827005` supplies most Chapman LVH.** Confirmed as Left ventricular
hypertrophy in SNOMED CT, but whether Ningbo used it interchangeably with
`164873001` is unverified. Worth eyeballing ~20 tracings before publishing.

**The STEMI dataset's official test split has no labels** — they are withheld for
a blind evaluation platform. Only its `train.csv` is usable here.

---

## Repository layout

```
install.sh              one-time environment setup
pipeline/               the generator (see pipeline/README.md for detail)
datasets/               input data, gitignored
build/                  generated output, gitignored
ecg-image-kit/          upstream renderer, gitignored
```

## Troubleshooting

**`imgaug` fails on `np.sctypes`** — you are on NumPy 2.x. Re-run `install.sh`;
the environment is pinned to NumPy 1.26 and Python 3.11.

**`KeyError: 'Age'` during render** — a staged record lacks header metadata.
Stage 1 screens for this; if it reappears, re-run from stage 1.

**Build died partway** — resume with `./run_all.sh 4`. Stages 4 and 5 skip
completed work.

**Different venv location** — `ECGKIT_PY=/path/to/bin/python ./run_all.sh`.
