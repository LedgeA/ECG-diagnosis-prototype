"""Stage 5 - paper/scan augmentation, replacing the kit's --augment and --wrinkles.

The kit's distortion layer was tested and rejected. Confirmed defects:

  * -t/--temperature is dead code. gen_ecg_image_from_data.py hardcodes
        blue_temp = random.choice((True, False))
        temp = random.choice(range(2000,4000)) if blue_temp
               else random.choice(range(10000,20000))
    so every image gets a heavy orange or blue cast with no neutral option.
  * -noise 0 crashes (random.choice on an empty range); noise is additive in
    0-255 units, so even 5 balloons a PNG from 0.4 MB to 4.4 MB.
  * iaa.Affine fills rotation corners with black, and with no margin the
    rotation clips the printed ID/Age/Sex header off the top of the sheet.
  * --augment silently requires --store_config, and --lead_bbox silently
    disables cropping.
  * --wrinkles at its lightest setting (-nv 1 -nh 1) still lays a heavy
    grey texture over the whole sheet and buries the trace.

Everything here is therefore done directly, so each effect is bounded, neutral
centred, and drawn from one distribution shared by all four classes. Style must
never correlate with class, or the model learns the renderer instead of the ECG.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

import config as C


@lru_cache(maxsize=1)
def _texture_files() -> tuple[Path, ...]:
    """The kit's wrinkled-paper photographs, reused here as shading fields."""
    root = C.KIT / "CreasesWrinkles" / "wrinkles-dataset"
    if not root.is_dir():
        return ()
    return tuple(sorted(
        p for p in root.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ))


def _rng(seed_parts: tuple) -> np.random.Generator:
    """Deterministic per-image RNG keyed only by output identity, never by class.

    Uses blake2b rather than hash(): CPython randomises string hashing per
    process unless PYTHONHASHSEED is pinned, which would make the corpus
    irreproducible across runs.
    """
    key = "|".join(str(p) for p in seed_parts).encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big") ^ C.SEED)


def _paper_colour(img: np.ndarray) -> np.ndarray:
    """Median of the sheet's corners - the fill used wherever geometry exposes canvas."""
    h, w = img.shape[:2]
    k = max(8, min(h, w) // 40)
    patches = np.concatenate([
        img[:k, :k].reshape(-1, 3), img[:k, -k:].reshape(-1, 3),
        img[-k:, :k].reshape(-1, 3), img[-k:, -k:].reshape(-1, 3),
    ])
    return np.median(patches, axis=0)


def _rotate_on_paper(img: Image.Image, deg: float, fill: tuple[int, int, int]) -> Image.Image:
    """Rotate about the centre with a paper-coloured background.

    expand=True then centre-crop back, so the printed header at the top of the
    sheet survives instead of being clipped the way the kit's Affine clips it.
    """
    w, h = img.size
    big = img.rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor=fill)
    bw, bh = big.size
    left, top = (bw - w) // 2, (bh - h) // 2
    return big.crop((left, top, left + w, top + h))


def _illumination(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uneven ambient light: a directional ramp plus corner falloff.

    Multiplicative and bounded well above zero, so the darkest corner of the
    sheet still reads as paper and the trace stays legible.
    """
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi)
    ramp = (np.cos(ang) * (xx / w - 0.5) + np.sin(ang) * (yy / h - 0.5))
    field = 1.0 + rng.uniform(0.06, 0.16) * ramp * 2.0

    r = np.sqrt((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2)
    field *= 1.0 - rng.uniform(0.04, 0.14) * (r / r.max()) ** 2

    # A soft off-sheet shadow along one edge, as when paper is photographed
    # on a desk rather than scanned flat.
    if rng.random() < 0.5:
        edge = rng.integers(0, 4)
        d = {0: xx / w, 1: 1 - xx / w, 2: yy / h, 3: 1 - yy / h}[int(edge)]
        field *= 1.0 - rng.uniform(0.08, 0.20) * np.exp(-d / rng.uniform(0.04, 0.12))
    return arr * field[..., None]


def _paper_texture(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shade the sheet with a photograph of real wrinkled paper.

    The kit ships 19 such photographs and composites one with a full-strength
    overlay blend, which is why its --wrinkles output buries the trace. The
    texture itself is excellent though - it is the blend that is wrong.

    Here the photograph is turned into a multiplicative shading field centred
    on 1.0, which is what crumpled paper physically does to reflected light,
    and its dynamic range is compressed so the darkest fold still sits far
    above the trace.
    """
    files = _texture_files()
    if not files:
        return arr

    h, w = arr.shape[:2]
    tex = Image.open(files[int(rng.integers(len(files)))]).convert("L")

    # Random crop, flip and rotate so 19 photographs yield far more than 19 looks.
    tw, th = tex.size
    scale = rng.uniform(0.55, 1.0)
    cw, ch = max(16, int(tw * scale)), max(16, int(th * scale))
    x0 = int(rng.integers(0, max(1, tw - cw + 1)))
    y0 = int(rng.integers(0, max(1, th - ch + 1)))
    tex = tex.crop((x0, y0, x0 + cw, y0 + ch))
    if rng.random() < 0.5:
        tex = tex.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        tex = tex.transpose(Image.FLIP_TOP_BOTTOM)
    if rng.random() < 0.5:
        tex = tex.transpose(Image.ROTATE_90)

    field = np.asarray(tex.resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0
    mean = float(field.mean())
    if mean <= 1e-3:
        return arr

    strength = rng.uniform(0.45, 0.85)
    field = 1.0 + (field / mean - 1.0) * strength
    return arr * np.clip(field, 0.78, 1.14)[..., None]


def _creases(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fold lines: a soft bright ridge with a darker shadow on one side.

    Real creases on a printout are a lighting effect, not ink, so this is a
    multiplicative field with a narrow profile. Amplitude is capped so a fold
    never crosses a trace hard enough to break it.
    """
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.ones((h, w), dtype=np.float32)

    for _ in range(int(rng.integers(1, 4))):
        if rng.random() < 0.5:          # horizontal-ish fold
            pos = rng.uniform(0.12, 0.88) * h
            tilt = rng.uniform(-0.06, 0.06)
            dist = yy - (pos + tilt * (xx - w / 2))
        else:                           # vertical-ish fold
            pos = rng.uniform(0.12, 0.88) * w
            tilt = rng.uniform(-0.06, 0.06)
            dist = xx - (pos + tilt * (yy - h / 2))

        width = rng.uniform(0.006, 0.020) * max(h, w)
        amp = rng.uniform(0.10, 0.22)
        # Odd (derivative-of-Gaussian) profile: highlight one side, shadow the
        # other, which is what a physical fold does to reflected light.
        g = np.exp(-(dist ** 2) / (2 * width ** 2))
        field *= 1.0 + amp * g * (dist / width) / 1.6

    # Broad, low-frequency bowing: sheets photographed on a desk are never
    # perfectly flat, and this is the cue that reads as "paper" rather than
    # "scan" at a glance.
    for _ in range(int(rng.integers(1, 3))):
        k = rng.uniform(0.6, 1.8)
        phase = rng.uniform(0, 2 * np.pi)
        axis = yy / h if rng.random() < 0.5 else xx / w
        field *= 1.0 + rng.uniform(0.02, 0.06) * np.sin(2 * np.pi * k * axis + phase)

    return arr * np.clip(field, 0.78, 1.22)[..., None]


def _colour_cast(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mild, neutral-centred white-balance drift - never the kit's bimodal cast."""
    gains = 1.0 + rng.normal(0.0, 0.018, size=3)
    return arr * gains[None, None, :]


def augment(src: Path, dst: Path, seed_parts: tuple) -> None:
    rng = _rng(seed_parts)
    img = Image.open(src).convert("RGB")

    arr0 = np.asarray(img, dtype=np.float32)
    fill = tuple(int(v) for v in _paper_colour(arr0))

    # --- geometry: tilt, then perspective -------------------------------
    img = _rotate_on_paper(img, float(rng.uniform(-3.0, 3.0)), fill)

    if rng.random() < 0.7:                       # keystone, as if photographed
        w, h = img.size
        m = rng.uniform(0.004, 0.018)
        dx, dy = m * w, m * h
        quad = (
            rng.uniform(0, dx), rng.uniform(0, dy),
            rng.uniform(0, dx), h - rng.uniform(0, dy),
            w - rng.uniform(0, dx), h - rng.uniform(0, dy),
            w - rng.uniform(0, dx), rng.uniform(0, dy),
        )
        img = img.transform((w, h), Image.QUAD, quad,
                            resample=Image.BICUBIC, fillcolor=fill)

    # --- paper: texture, then sharp folds, then lighting -----------------
    # Real crumpled paper shows both: a diffuse wrinkle field over the whole
    # sheet, and a few hard fold lines where it was actually creased.
    arr = np.asarray(img, dtype=np.float32)
    if rng.random() < 0.85:
        arr = _paper_texture(arr, rng)
    if rng.random() < 0.65:
        arr = _creases(arr, rng)
    arr = _illumination(arr, rng)
    arr = _colour_cast(arr, rng)
    arr *= rng.uniform(0.92, 1.06)                              # exposure
    mean = arr.mean()
    arr = (arr - mean) * rng.uniform(0.94, 1.08) + mean         # contrast

    # Sensor noise. Kept low deliberately: at 150 DPI the ST segment is only a
    # few pixels tall, and heavy noise buries the very feature STEMI depends on.
    arr += rng.normal(0.0, rng.uniform(1.5, 4.5), size=arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    out = Image.fromarray(arr)

    # --- thin margin, added last ---------------------------------------
    # The kit's --pad_inches is int-valued (0 or a full inch), so the thin
    # border the corpus spec asks for is applied here instead.
    m = C.MARGIN_PX
    canvas = Image.new("RGB", (out.width + 2 * m, out.height + 2 * m), fill)
    canvas.paste(out, (m, m))

    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst, "JPEG", quality=C.JPEG_QUALITY,
                subsampling=C.JPEG_SUBSAMPLING, optimize=True)


INDEX_FIELDS = ["image_path", "record", "uid", "cls", "split", "source",
                "render_k", "patient_id", "age", "sex", "note"]


def _plan() -> list[dict]:
    """Pair every rendered PNG with its manifest row."""
    meta = {r["record"]: r for r in csv.DictReader((C.BUILD / "staged_map.csv").open())}
    jobs = []
    for png in C.RENDERED.rglob("*.png"):
        record = png.stem.rsplit("-", 1)[0]          # kit writes <record>-0.png
        row = meta.get(record)
        if row is None:
            continue
        # .../rendered/<cls>/r<k>/<chunk>/<record>-0.png
        k = int(png.parent.parent.name[1:])
        dst = C.IMAGES / row["split"] / row["cls"] / f"{record}_r{k}.jpg"
        jobs.append({"src": str(png), "dst": str(dst), "record": record, "k": k,
                     "uid": row["uid"], "cls": row["cls"], "split": row["split"],
                     "source": row["source"], "patient_id": row["patient_id"],
                     "age": row["age"], "sex": row["sex"], "note": row["note"]})
    return jobs


def _work(job: dict) -> tuple[dict, str | None]:
    dst = Path(job["dst"])
    if dst.exists() and dst.stat().st_size > 0:
        return job, None                              # idempotent re-runs
    try:
        augment(Path(job["src"]), dst, (job["record"], job["k"]))
    except Exception as exc:                          # noqa: BLE001 - report, don't abort
        return job, f"{type(exc).__name__}: {exc}"
    return job, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--workers", type=int, default=C.WORKERS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("-i", "--input_dir", help="standalone mode: augment a folder")
    ap.add_argument("-o", "--output_dir")
    ap.add_argument("--renders", type=int, default=1)
    args = ap.parse_args()

    # ---- standalone mode, used for spot checks -------------------------
    if args.input_dir:
        if not args.output_dir:
            print("standalone mode needs -o/--output_dir", file=sys.stderr)
            return 1
        src_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
        pngs = sorted(src_dir.rglob("*.png"))[: args.limit or None]
        for png in pngs:
            stem = png.stem.rsplit("-", 1)[0]
            for k in range(args.renders):
                augment(png, out_dir / f"{stem}_a{k}.jpg", (stem, k))
        sizes = [p.stat().st_size for p in out_dir.glob("*.jpg")]
        print(f"wrote {len(sizes)} images -> {out_dir}")
        print(f"mean size {sum(sizes)/max(len(sizes),1)/1024:.0f} KB")
        return 0

    # ---- corpus mode ---------------------------------------------------
    jobs = _plan()
    if not jobs:
        print("no rendered PNGs found - run stage4 first", file=sys.stderr)
        return 1
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"{len(jobs)} images on {args.workers} workers")
    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futures), start=1):
            job, err = fut.result()
            if err:
                failures.append((job["record"], err))
            else:
                rows.append({
                    "image_path": job["dst"], "record": job["record"],
                    "uid": job["uid"], "cls": job["cls"], "split": job["split"],
                    "source": job["source"], "render_k": job["k"],
                    "patient_id": job["patient_id"], "age": job["age"],
                    "sex": job["sex"], "note": job["note"],
                })
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}")

    rows.sort(key=lambda r: (r["cls"], r["record"], r["render_k"]))
    C.INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    with C.INDEX_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)

    sizes = [Path(r["image_path"]).stat().st_size for r in rows]
    print(f"\nwrote {len(rows)} images -> {C.IMAGES}")
    print(f"mean {sum(sizes)/max(len(sizes),1)/1024:.0f} KB, "
          f"total {sum(sizes)/1e9:.1f} GB")
    print(f"index -> {C.INDEX_CSV}")
    if failures:
        print(f"{len(failures)} failures, first few:")
        for rec, err in failures[:10]:
            print(f"    {rec}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
