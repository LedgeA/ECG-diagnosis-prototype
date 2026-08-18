#!/usr/bin/env python3
"""
annotate_augmented.py

Carries ecg-image-kit's annotation boxes from the clean renders in
build/rendered/ onto the augmented JPEGs in build/images/, producing sidecar
JSONs that are actually valid for the images the model trains on.

Why this is possible without re-rendering anything
--------------------------------------------------
Stage 5 moves pixels three times - tilt, optional keystone, margin paste - and
never updated the kit's JSON, which is why `visualize_annotations.py` refuses
to draw on an augmented image. But the augmentation is deterministic: its RNG
is keyed only by (record, render_k) and the build seed, and the geometry draws
are consumed before any photometric draw. So the exact tilt and keystone a
given image was built with can be replayed from its name alone, long after the
fact. `pipeline/geometry.py` does that replay and inverts the mapping; stage 5
calls the same draw function, so the two cannot disagree.

The transform is NOT a homography - PIL's QUAD keystone is bilinear - so boxes
come out as general quadrilaterals rather than axis-aligned rectangles. Each
box is therefore written as its four mapped corners, in the same [y, x] corner
format the kit uses, plus a derived axis-aligned `bounding_box_aabb` for
consumers that need a plain rectangle.

Usage
-----
    python annotate_augmented.py                    # whole corpus
    python annotate_augmented.py --limit 20         # spot check
    python annotate_augmented.py --inplace          # write beside the JPEGs
    python annotate_augmented.py --pixels           # also map plotted_pixels

By default the sidecars land in build/annotations/<split>/<cls>/<stem>.json and
`plotted_pixels` is dropped, which keeps the whole annotation set to a few tens
of MB instead of the ~42 GB the kit's per-sample pixel traces occupy.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))

import config as C          # noqa: E402
import geometry             # noqa: E402
from stage5_augment import _rng   # noqa: E402  - the same stream stage 5 used

BOX_KEYS = ("lead_bounding_box", "text_bounding_box")


def _map_box(box: dict, geom: geometry.Geometry) -> tuple[dict, list[float]]:
    """Map one 4-corner box. Kit stores corners as [y, x]; geometry wants (x, y)."""
    corners = [box[str(i)] for i in range(4)]
    pts = np.array([[c[1], c[0]] for c in corners], dtype=np.float64)
    out = geometry.forward_points(geom, pts)

    mapped = {str(i): [float(out[i][1]), float(out[i][0])] for i in range(4)}
    xs, ys = out[:, 0], out[:, 1]
    aabb = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    return mapped, aabb


def convert(json_path: Path, record: str, render_k: int,
            keep_pixels: bool) -> dict:
    """Build the augmented-image annotation for one render."""
    meta = json.loads(json_path.read_text())

    src_w, src_h = int(meta["width"]), int(meta["height"])
    # Replay only the head of the stream; the photometric draws that follow
    # are irrelevant here but must not be consumed differently.
    geom = geometry.geometry_draws(
        _rng((record, render_k)), (src_w, src_h), C.MARGIN_PX)

    dst_w, dst_h = geom.dst_size
    out = dict(meta)
    out["width"], out["height"] = dst_w, dst_h

    leads = []
    for lead in meta.get("leads", []):
        new = dict(lead)
        for key in BOX_KEYS:
            if key in lead:
                mapped, aabb = _map_box(lead[key], geom)
                new[key] = mapped
                new[f"{key}_aabb"] = aabb

        if "plotted_pixels" in lead:
            if keep_pixels:
                px = np.array([[p[1], p[0]] for p in lead["plotted_pixels"]],
                              dtype=np.float64)
                if len(px):
                    m = geometry.forward_points(geom, px)
                    new["plotted_pixels"] = [[float(p[1]), float(p[0])] for p in m]
            else:
                new.pop("plotted_pixels", None)
        leads.append(new)
    out["leads"] = leads

    # Traceability: what was replayed, so a mismatch can be diagnosed later.
    out["augmentation"] = {
        "source_size": [src_w, src_h],
        "rotation_degrees": geom.degrees,
        "keystone_quad": list(geom.quad) if geom.quad else None,
        "margin_px": geom.margin,
        "note": ("boxes are quadrilaterals - the keystone is bilinear, so a "
                 "rotated box is not an axis-aligned rectangle; *_aabb is the "
                 "enclosing upright box"),
    }
    return out


def _work(job: dict) -> tuple[str, str | None]:
    try:
        out = convert(Path(job["json"]), job["record"], job["k"], job["pixels"])
        dst = Path(job["dst"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(out))
    except Exception as exc:                      # noqa: BLE001 - report, don't abort
        return job["record"], f"{type(exc).__name__}: {exc}"
    return job["record"], None


def _plan(inplace: bool, keep_pixels: bool) -> list[dict]:
    """Pair every corpus image with the JSON of the render it came from."""
    if not C.INDEX_CSV.exists():
        print(f"{C.INDEX_CSV} not found - run stage 5 first", file=sys.stderr)
        return []

    # One pass over the render tree; the chunk directory is not in the index.
    renders: dict[tuple[str, int], Path] = {}
    for js in C.RENDERED.rglob("*.json"):
        record = js.stem.rsplit("-", 1)[0]
        k = int(js.parent.parent.name[1:])        # .../<cls>/r<k>/<chunk>/
        renders[(record, k)] = js

    jobs, missing = [], 0
    for row in csv.DictReader(C.INDEX_CSV.open()):
        key = (row["record"], int(row["render_k"]))
        js = renders.get(key)
        if js is None:
            missing += 1
            continue
        img = Path(row["image_path"])
        dst = (img.with_suffix(".json") if inplace else
               C.BUILD / "annotations" / row["split"] / row["cls"] / f"{img.stem}.json")
        jobs.append({"json": str(js), "record": row["record"],
                     "k": int(row["render_k"]), "dst": str(dst),
                     "pixels": keep_pixels})
    if missing:
        print(f"warning: {missing} corpus images had no matching render JSON")
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--workers", type=int, default=C.WORKERS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--inplace", action="store_true",
                    help="write beside the JPEG so visualize_annotations.py autodetects it")
    ap.add_argument("--pixels", action="store_true",
                    help="also map plotted_pixels (huge: ~42 GB across the corpus)")
    args = ap.parse_args()

    jobs = _plan(args.inplace, args.pixels)
    if not jobs:
        return 1
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"{len(jobs)} annotations on {args.workers} workers")
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            record, err = fut.result()
            if err:
                failures.append((record, err))
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}")

    written = len(jobs) - len(failures)
    root = "beside the JPEGs" if args.inplace else str(C.BUILD / "annotations")
    print(f"\nwrote {written} annotation sidecars -> {root}")
    if failures:
        print(f"{len(failures)} failures, first few:")
        for rec, err in failures[:10]:
            print(f"    {rec}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
