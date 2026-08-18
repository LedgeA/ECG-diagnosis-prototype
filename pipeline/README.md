# Four-class ECG image corpus

Renders STEMI / LVH / AF / NORMAL paper-ECG images from three WFDB datasets via
[ecg-image-kit](../ecg-image-kit). Output is ~23,700 JPEGs at 150 DPI in a
3×4 layout with a continuous lead II rhythm strip, 25 mm/s, 10 mm/mV.

## Run it

```bash
./run_all.sh          # full build, ~4 h on 8 cores, ~17 GB
./run_all.sh 4        # resume from stage 4
WORKERS=16 ./run_all.sh
```

Stages 4 and 5 skip completed work, so an interrupted build restarts cleanly.

## Environment

```bash
bash ../install.sh
```

Builds the venv at `~/.cache/ecgkit-venv` and verifies all 15 imports. Safe to
re-run. `VENV=/some/path bash install.sh` relocates it; then point the pipeline
at it with `ECGKIT_PY=/some/path/bin/python`.

Three things the script exists to get right, all of which break a naive install:

- **Python 3.11.** `imgaug` calls `np.sctypes`, removed in NumPy 2.0, so the
  environment is pinned to NumPy 1.26 — which newer interpreters no longer ship.
- **NumPy pinned last.** TensorFlow pulls NumPy 2.x during its own install, so
  the 1.26 pin has to be re-applied afterwards.
- **Both opencv distributions held below 5.0.** `imgaug` depends on
  `opencv-python` while the pipeline asks for `opencv-python-headless`; the 5.x
  wheels are built against the NumPy 2 ABI, and whichever lands last owns `cv2`.

Pip warns that `ml-dtypes` wants NumPy ≥ 2. Ignore it; every import resolves.

## Stages

| | Script | Does |
|---|---|---|
| 0 | `patch_kit.py` | Inset the kit's printed header; idempotent, run by `run_all.sh` |
| 1 | `stage1_manifest.py` | Resolve labels from all three catalogues into one manifest |
| 2 | `stage2_select.py` | Subsample to target, patient-grouped train/val/test split |
| 3 | `stage3_transcode.py` | Harmonise to one WFDB dialect, band-pass, anonymise |
| 4 | `stage4_render.py` | Render clean sheets with the kit, in parallel |
| 5 | `stage5_augment.py` | Paper/scan augmentation, thin margin, JPEG encode |
| 6 | `stage6_verify.py` | Integrity, balance, pixel stats, source-leakage probe |

All tunables live in `config.py`.

## Output

```
build/images/<split>/<class>/ECG######_r<k>.jpg
build/index.csv     image_path, record, uid, cls, split, source, render_k, ...
```

Class counts: 5,692 STEMI / 6,000 each LVH, AF, NORMAL from 9,423 records.

Optionally, spatial annotations valid for those JPEGs:

```bash
python ../annotate_augmented.py          # -> build/annotations/<split>/<class>/*.json
```

The kit only annotates the clean render, and stage 5 then moves every pixel.
`annotate_augmented.py` replays stage 5's geometry — deterministic, keyed on
`(record, render_k)` — and carries the lead and text boxes onto the augmented
sheet, so they can be recovered for a corpus that is already built without
re-rendering anything. Boxes come out as quadrilaterals (the sheet is tilted
and keystoned); each also carries a derived `*_aabb` upright rectangle. Not
needed to train a whole-image classifier, which takes its label from
`index.csv`.

## Decisions worth knowing

**Only the four target classes are ever rendered.** Stage 1 assigns every record
to STEMI, LVH, AF or NORMAL and discards the rest; 21,386 of 30,809 candidates
are never transcoded, rendered or stored. The one deliberate inclusion beyond
strict diagnostic need is the Chongqing NORMAL controls, which exist to break
the source/class confound described below.

**AF is atrial fibrillation only.** Flutter is excluded from every class rather
than folded in. That drops 8,036 Chapman records and is most of why the LVH pool
fell from 6,027 to 3,804.

**A patient never appears under two labels.** Chongqing patients with any ACS
event are barred from the control pool even on a negative record, and stage 2
drops any remaining patient spanning two classes (66 of them). Splits are then
stratified per class, so each lands on 80/10/10.

**Records without age or sex are dropped** (109 of them). The kit's
`--print_header` raises `KeyError` without both, and printing "Unknown" instead
would leak source — missing metadata is not uniform across the three datasets.
PTB-XL's `age > 89` mask is rendered as 90 rather than discarded.

**The band-pass is 0.05–40 Hz, not 0.05–150 Hz.** At 150 DPI and 25 mm/s the
paper carries 147.6 px/s, so image Nyquist is 73.8 Hz — 150 Hz is not
representable and would only alias against the 1 mm grid. Nothing in these four
classes lives above 40 Hz. Measured cost: 5.8% of QRS amplitude, applied
uniformly to every class, so it compresses the LVH signal slightly but does not
bias between classes.

**Records are renamed `ECG000001…`.** The kit prints `ID: <record name>` on every
sheet and the native names (`00001_hr` / `JS00001` / `00101`) announce the
source in plain text.

**Grid is grey, via `--random_bw 1`.** Despite the name that flag selects a grid
palette rather than converting to greyscale; it is the only way to reach the
kit's grey/black scheme, which matches a real printout. `--standard_grid_color`
offers only brown/pink/blue/green/red.

**Environmental noise is paper-level, not signal-level** — tilt, perspective,
creasing, lighting, mild sensor noise. Amplitudes are capped so the trace stays
readable: worst observed paper-to-trace contrast across sampled variants was 227
of 250 greyscale levels.

**The kit's `--augment` and `--wrinkles` are not used.** Both were tested and
rejected; `stage5_augment.py` documents the six defects, the worst being that
`-t/--temperature` is dead code and every augmented sheet gets a forced orange
or blue cast.

## Known limitations

**STEMI is single-source.** Every STEMI image comes from the Chongqing dataset,
so source identity partly predicts the label. Mitigations: identical signal
conditioning across sources, class-independent render styling, anonymised IDs,
and one third of NORMAL drawn from the same Chongqing cohort. Stage 6's probe
measures what survives — treat a large margin over chance as a reason to
distrust the STEMI numbers, and report per-source metrics regardless.

**The local PTB-XL copy is incomplete.** 19,490 of 21,799 signal files, missing
in contiguous blocks (subdir `14000` is empty, `13000` holds 38). 1,372 usable
labelled records are unreachable. Pools remain large enough; re-download
`records500` if you want them back.

**Chapman has no patient identifier**, so each record is treated as its own
patient. If the Ningbo half repeats patients, some leakage survives the split.

**`55827005` supplies most Chapman LVH.** It is confirmed as Left ventricular
hypertrophy in SNOMED CT, but whether Ningbo used it interchangeably with
`164873001` is unverified. Worth eyeballing ~20 tracings before publishing.

**Licensing.** The STEMI dataset is CC BY-NC-ND 4.0. The rendered corpus is a
derivative work and should not be redistributed.
