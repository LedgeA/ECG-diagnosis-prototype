"""Stage 4 - render clean ECG sheets with ecg-image-kit, in parallel.

The kit emits exactly one image per record and seeds its RNG once per batch
invocation, so N variants of a record means N invocations at N different seeds.
Work is therefore a grid of (class chunk) x (render index), which also gives
enough independent jobs to keep every core busy.

Only the layout and calibration flags are passed. The kit's --augment and
--wrinkles are deliberately unused; stage 5 does that work. See stage5 for the
list of defects that decision rests on.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import config as C
import patch_kit


def _jobs() -> list[tuple[str, Path, int]]:
    out = []
    for cls in C.CLASSES:
        cls_dir = C.STAGED / cls
        if not cls_dir.is_dir():
            continue
        for chunk in sorted(p for p in cls_dir.iterdir() if p.is_dir()):
            for k in range(C.RENDERS_PER_RECORD[cls]):
                out.append((cls, chunk, k))
    return out


def _run(job: tuple[str, Path, int]) -> tuple[str, int, str]:
    cls, chunk, k = job
    out_dir = C.RENDERED / cls / f"r{k}" / chunk.name
    tag = f"{cls}/{chunk.name}/r{k}"

    # Idempotent: a chunk already rendered is skipped, so an interrupted build
    # can simply be re-run.
    if out_dir.is_dir() and any(out_dir.glob("*.png")):
        n_in = len(list(chunk.glob("*.dat")))
        if len(list(out_dir.glob("*.png"))) >= n_in:
            return tag, 0, "skipped"

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(C.VENV_PY), "gen_ecg_images_from_data_batch.py",
        "-i", str(chunk), "-o", str(out_dir),
        "-se", str(C.SEED + k),
        *C.KIT_RENDER_FLAGS,
    ]
    proc = subprocess.run(cmd, cwd=str(C.KIT), capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
        return tag, proc.returncode, tail[0]
    return tag, 0, f"{len(list(out_dir.glob('*.png')))} images"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--workers", type=int, default=C.WORKERS)
    ap.add_argument("--limit", type=int, default=0, help="run only N jobs (smoke test)")
    args = ap.parse_args()

    # The kit is vendored and gitignored, so a fresh clone is stock upstream
    # and would render the header flush against the paper corner.
    if not patch_kit.is_patched():
        print("ecg-image-kit is not patched - run: python patch_kit.py",
              file=sys.stderr)
        return 1

    jobs = _jobs()
    if not jobs:
        print("no staged chunks found - run stage3 first", file=sys.stderr)
        return 1
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"{len(jobs)} render jobs on {args.workers} workers")
    started = time.time()
    failures = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), start=1):
            tag, code, msg = fut.result()
            if code:
                failures.append((tag, msg))
                print(f"  [{i}/{len(jobs)}] FAIL {tag}: {msg}")
            else:
                elapsed = time.time() - started
                rate = i / max(elapsed, 1e-6)
                eta = (len(jobs) - i) / max(rate, 1e-6)
                print(f"  [{i}/{len(jobs)}] {tag}: {msg}  (eta {eta/60:.0f} min)")

    total = sum(1 for _ in C.RENDERED.rglob("*.png"))
    print(f"\nrendered {total} images in {(time.time()-started)/60:.1f} min")
    if failures:
        print(f"{len(failures)} failed jobs:")
        for tag, msg in failures[:20]:
            print(f"    {tag}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
