# ECG Diagnosis Prototype

Generates a synthetic paper-ECG **image** corpus (STEMI / LVH / AF / NORMAL) from
three WFDB waveform datasets, for training a direct 2D-vision diagnosis model.
Waveforms → [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit)
→ custom paper/scan augmentation → labelled JPEG corpus.

**Start with [README.md](README.md)** — full spec, data sources, design
rationale, known limitations. **[pipeline/README.md](pipeline/README.md)** has
implementer-level stage detail. This file is a handoff snapshot: current state,
what's already been fought through, and what's still open. Treat the other two
as the source of truth if anything here goes stale.

## Current state (as of this handoff)

Stages 1–3 are done. Stage 4 (render) barely started, stopped mid-flight.

```
manifest_final.csv   9,423 records selected            ✓ done
build/staged/        9,423/9,423 WFDB records           ✓ done
build/rendered/       108 PNGs (STEMI class only)        ~0.5% into stage 4
build/images/         0 JPEGs                            not started
build/index.csv       does not exist                     not started
```

Resume with:
```bash
cd pipeline && ./run_all.sh 4
```
This will take the ~4 hours it always takes — stage 4 was not close to done,
whatever partial progress you inherit came from ad-hoc testing runs, not a
real build attempt.

**Before you resume: check for orphaned processes.** This session repeatedly
had `stage4_render.py -j 8` invocations get backgrounded, time out, or get
interrupted mid-render, leaving their `ProcessPoolExecutor` workers alive as
orphans (reparented, 0% CPU, doing nothing, but still occupying process slots).
I found and killed 40 of them at 12:35–13:07 today. Check before launching more:
```bash
pgrep -af "stage4_render|gen_ecg_images|stage5_augment"
pkill -9 -f "stage4_render.py|gen_ecg_images_from_data_batch.py"   # if any found
```
Stage 4/5 are idempotent (skip completed work by counting files), so killing
and resuming is always safe — nothing is lost.

## Environment

```bash
bash install.sh    # idempotent, ~10-15 min, builds ~/.cache/ecgkit-venv
```
Python 3.11 + NumPy 1.26 pinned — the kit's `imgaug` dependency breaks on
NumPy 2.x (`np.sctypes` was removed). Both `opencv-python` and
`opencv-python-headless` are separately pinned `<5` for the same ABI reason;
`imgaug` pulls the former as a transitive dependency, and unpinned it'll drag
NumPy 2.x back in through the side door. Verified from a from-scratch venv, not
just the working one — see README's Environment section for the story.

`pipeline/patch_kit.py` applies one required source patch to the vendored
(gitignored) `ecg-image-kit/` — insets the printed header away from the paper
corner. `run_all.sh` runs it automatically as a preflight step; it's idempotent
and checks a marker string before patching, so re-running is safe.

## The single most important thing to know

**STEMI is single-source.** Every STEMI image comes from the Chongqing dataset;
nothing else supplies it. A vision model can trivially learn "which hospital
scanned this" as a proxy for the STEMI label without ever finding ST elevation.
Mitigations are in place (identical signal conditioning across sources,
class-independent render styling, anonymised record IDs, a third of NORMAL
drawn from that same Chongqing cohort to break the 1:1 source↔class mapping)
but this cannot be fully eliminated with these three datasets. `stage6_verify.py`
runs a patient-grouped source-leakage probe — if it comes back well above
chance on the full corpus, don't trust the STEMI numbers, and report per-source
metrics regardless of what it says. Full writeup in README's Known Limitations.

## Things that looked reasonable and were wrong

Each of these cost real debugging time. Don't redo it.

- **The kit's `--augment`/`--wrinkles` flags are broken, not just ugly.**
  `-t/--temperature` is dead code — `gen_ecg_image_from_data.py` hardcodes a
  forced orange-or-blue cast regardless of what you pass, and
  `--deterministic_temp` is declared in argparse and never read anywhere.
  `--wrinkles` composites its (excellent) wrinkle photographs with a
  full-strength overlay blend that buries the trace even at its lightest
  setting. `stage5_augment.py` reimplements both effects directly — same
  source photographs, bounded multiplicative blend instead of the broken one.
- **`.mat` (Chapman) and `.dat` (PTB-XL/STEMI) are not interchangeable in the
  kit.** `.mat` loads as raw ADC counts, `.dat` loads as millivolts via `wfdb`;
  the kit reads the ADC gain into a variable and never applies it. Chapman
  renders 1000× over-scale and crashes. Stage 3 transcodes everything through
  `wfdb` into one dialect before it ever reaches the kit.
- **The local PTB-XL copy is incomplete and partly corrupt** — not just
  missing files (19,490/21,799, in contiguous gaps), but 19 zero-byte headers
  and 40 truncated signal files that pass an `exists()` check and then crash
  deep inside `wfdb.rdheader`/`rdrecord`. The STEMI dataset has 2 as well.
  Stage 1 validates header parseability *and* signal file size against the
  geometry the header claims before accepting a record. Re-adding this check
  is why stage 1 looks more paranoid than you'd expect — it's paranoid because
  the naive version crashed a multi-hour build partway through, once.
- **A patient can appear under two different labels.** 10 Chongqing patients
  had both a STEMI-positive record and an ACS-negative one that would
  otherwise have qualified as a NORMAL control. Stage 2 drops any patient
  appearing under more than one class before splitting, and stage 6 asserts
  zero cross-class and zero cross-split patients on every run.
- **150 DPI, not 200 or 300, and 0.05–40 Hz, not the diagnostic 0.05–150 Hz.**
  Both are consequences of the same arithmetic: at 150 DPI and 25 mm/s the
  paper only carries 147.6 px/s, so image Nyquist is 73.8 Hz. 150 Hz literally
  cannot be represented at any DPI this corpus renders at and would only alias
  against the 1 mm grid. Full derivation in README.
- **The kit's annotation JSON applies to the clean render only — but the boxes
  are now transferable.** Stage 5's rotation/keystone/shading/margin move every
  pixel without updating the sidecar JSON `--store_config` writes, so the kit's
  own JSON in `build/rendered/` is valid *only* there;
  `visualize_annotations.py` checks dimensions and refuses to draw on a
  mismatch rather than drawing a wrong box.
  `annotate_augmented.py` closes the gap: stage 5's RNG is keyed only by
  `(record, render_k)` and `SEED`, and the geometry draws are consumed before
  any photometric draw, so the exact tilt and keystone of an already-built
  image can be replayed from its name and the boxes carried onto the JPEG.
  Output goes to `build/annotations/<split>/<cls>/<record>_r<k>.json`, sized
  for the augmented image. Verified end-to-end: mapped `plotted_pixels` land
  on ink (mean grey ~46, ~90% below the ink threshold) where ignoring the
  geometry lands on blank paper (~195, indistinguishable from random).
- **Never write into `build/rendered/<class>/r<k>/<chunk>/`.**
  `stage4_render.py`'s resume logic counts `*.png` files in each chunk
  directory against the `.dat` count in the matching staged chunk to decide
  whether it's done. Any tool that writes output there (an annotator, a QA
  script, anything) can silently corrupt that count and make a resumed build
  skip real work. `visualize_annotations.py` detects this and redirects to
  `build/annotated/`.

## Explicit decisions already made (don't re-litigate without a reason)

- AF = atrial fibrillation only. Flutter is excluded from every class, not
  folded into anything.
- Amplitude is never normalised across sources — would erase the voltage
  criteria LVH is defined by. Only filtering and baseline are harmonised.
- Single-label, mutually exclusive. Records matching more than one target
  class are dropped rather than going to a multi-label head.
- Records are renamed `ECG000001…` before rendering. The kit prints
  `ID: <record name>` on every sheet, and the native filenames
  (`00001_hr` / `JS00001` / `00101`) announce the source dataset in plain text.

## File map

```
install.sh                      environment setup (root)
visualize_annotations.py        annotation QA tool (root, faithful port of
                                 code-from-other-project's script + two
                                 correctness fixes - see its own docstring)
annotate_augmented.py           replays stage5's geometry to carry the kit's
                                 boxes onto the augmented JPEGs (root)
tests/test_geometry.py          pins the geometry replay against real PIL
                                 output and freezes the RNG draw order
pipeline/
  config.py                     every tunable; read this first for "why is X"
  geometry.py                   stage5's tilt/keystone/margin, as a mapping
  patch_kit.py                  the one required ecg-image-kit source patch
  stage1_manifest.py            label resolution + corruption screening
  stage2_select.py              subsample + patient-grouped 80/10/10 split
  stage3_transcode.py           WFDB harmonisation, band-pass, anonymisation
  stage4_render.py              parallel kit invocation (clean renders only)
  stage5_augment.py             paper/scan realism (replaces kit's broken --augment/--wrinkles)
  stage6_verify.py              integrity + balance + leakage probe
  run_all.sh                    driver; ./run_all.sh <n> resumes from stage n
datasets/                       gitignored, must be populated manually (see README)
ecg-image-kit/                  gitignored, vendored upstream, patched by patch_kit.py
build/                          gitignored, all pipeline output
code-from-other-project/        reference scripts the user supplied from a prior
                                 project (PTB-XL only); useful as prior art, not
                                 as source of truth - several of its ecg-image-kit
                                 flags are the broken ones above
```

## If something looks off

Check `git status` and `git log` first — this repo has two commits
(`init commit`, `feat: incomplete pipeline (no environmental noise)`) and
several files are modified-but-uncommitted. Nothing in `pipeline/` or the root
scripts has been committed since real testing began, so `git diff` won't help
you find what changed recently — treat the working tree as current truth, not
the commit history.
