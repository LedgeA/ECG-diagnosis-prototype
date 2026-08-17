"""Stage 2 - subsample to the corpus target and split by patient.

Runs before transcoding so only the ~9.4k records that reach the corpus are
converted, rather than all 32k candidates.

Two rules carry the weight here:

  * The split is grouped by patient. PTB-XL contributes 18,869 patients across
    21,799 records and the STEMI dataset 17,018 across 17,960, so a naive row
    split would put the same person on both sides of the evaluation.
  * NORMAL is filled to a fixed per-source quota. The Chongqing third is what
    stops "which hospital recorded this" from being a perfect stand-in for the
    STEMI label.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict

import numpy as np

import config as C

FIELDS = ["uid", "source", "abs_path", "fmt", "cls", "patient_id",
          "age", "sex", "note", "split", "n_renders"]


def _group_by_patient(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["patient_id"]].append(r)
    return groups


def _drop_cross_class_patients(rows: list[dict]) -> tuple[list[dict], int]:
    """Remove any patient appearing under two different labels.

    Without this the same person can be both STEMI and NORMAL, which is a
    label contradiction, and it also blocks per-class stratified splitting.
    """
    classes = defaultdict(set)
    for r in rows:
        classes[r["patient_id"]].add(r["cls"])
    bad = {p for p, c in classes.items() if len(c) > 1}
    return [r for r in rows if r["patient_id"] not in bad], len(bad)


def _assign_splits(rows: list[dict], rng: np.random.Generator) -> None:
    """Split whole patients within each class, so every class lands on 80/10/10.

    Assigning globally lets PTB-XL's forced folds eat the test budget and
    starves the classes PTB-XL does not contribute to - STEMI came out at
    8.3% test that way instead of 10%.
    """
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["cls"]].append(r)

    for cls_rows in by_class.values():
        groups = _group_by_patient(cls_rows)
        keys = sorted(groups)
        rng.shuffle(keys)

        # PTB-XL ships its own stratified folds; honour 9 and 10 as val/test.
        forced: dict[str, str] = {}
        for key in keys:
            notes = {r["note"] for r in groups[key] if r["source"] == "ptbxl"}
            folds = {int(n[4:]) for n in notes if n.startswith("fold")}
            if folds & C.PTBXL_TEST_FOLDS:
                forced[key] = "test"
            elif folds & C.PTBXL_VAL_FOLDS:
                forced[key] = "val"

        free = [k for k in keys if k not in forced]
        need_test = max(0, round(len(keys) * C.SPLITS["test"])
                        - sum(v == "test" for v in forced.values()))
        need_val = max(0, round(len(keys) * C.SPLITS["val"])
                       - sum(v == "val" for v in forced.values()))

        plan = {k: "test" for k in free[:need_test]}
        plan.update({k: "val" for k in free[need_test:need_test + need_val]})
        for key in keys:
            split = forced.get(key) or plan.get(key, "train")
            for r in groups[key]:
                r["split"] = split


def _take(rows: list[dict], n: int, rng: np.random.Generator) -> list[dict]:
    if len(rows) <= n:
        return list(rows)
    # Sample whole patients so a patient is never half-in, half-out.
    groups = _group_by_patient(rows)
    keys = sorted(groups)
    rng.shuffle(keys)
    out: list[dict] = []
    for key in keys:
        if len(out) >= n:
            break
        out.extend(groups[key])
    return out[:n]


def main() -> int:
    rows = list(csv.DictReader(C.MANIFEST_RAW.open()))
    if not rows:
        print("manifest_raw.csv is empty - run stage1 first", file=sys.stderr)
        return 1

    rng = np.random.default_rng(C.SEED)
    rows, n_dropped = _drop_cross_class_patients(rows)
    if n_dropped:
        print(f"dropped {n_dropped} patients appearing under two labels")

    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["cls"]].append(r)

    selected: list[dict] = []
    for cls in C.CLASSES:
        want = C.RECORDS_PER_CLASS[cls]
        pool = by_class[cls]

        if cls == "NORMAL":
            # Fixed per-source quota rather than a blind sample, so the
            # Chongqing share is guaranteed and not left to chance.
            picked: list[dict] = []
            for source, quota in C.NORMAL_SOURCE_QUOTA.items():
                subset = [r for r in pool if r["source"] == source]
                got = _take(subset, quota, rng)
                if len(got) < quota:
                    print(f"  warning: NORMAL/{source} wanted {quota}, got {len(got)}")
                picked.extend(got)
            chosen = picked
        else:
            chosen = _take(pool, want, rng)
            if len(chosen) < want:
                print(f"  warning: {cls} wanted {want}, pool has {len(pool)}")

        for r in chosen:
            r["n_renders"] = C.RENDERS_PER_RECORD[cls]
        selected.extend(chosen)

    _assign_splits(selected, rng)

    with C.MANIFEST_FINAL.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in FIELDS} for r in selected)

    # ---- report -------------------------------------------------------
    print(f"\nselected {len(selected)} records -> {C.MANIFEST_FINAL}")
    cls_split = Counter((r["cls"], r["split"]) for r in selected)
    imgs = Counter()
    for r in selected:
        imgs[r["cls"]] += int(r["n_renders"])

    print(f"\n    {'class':9s}{'train':>8s}{'val':>7s}{'test':>7s}{'recs':>8s}{'images':>9s}")
    for cls in C.CLASSES:
        tr, va, te = (cls_split[(cls, s)] for s in ("train", "val", "test"))
        print(f"    {cls:9s}{tr:8d}{va:7d}{te:7d}{tr+va+te:8d}{imgs[cls]:9d}")
    print(f"    {'TOTAL':9s}{'':22s}{len(selected):8d}{sum(imgs.values()):9d}")

    print("\n    NORMAL by source: " + ", ".join(
        f"{s}={sum(1 for r in selected if r['cls']=='NORMAL' and r['source']==s)}"
        for s in ("stemi", "ptbxl", "chapman")))

    # Leakage assertion: no patient may appear in two splits.
    seen: dict[str, str] = {}
    for r in selected:
        prev = seen.setdefault(r["patient_id"], r["split"])
        if prev != r["split"]:
            print(f"    FAIL: patient {r['patient_id']} in {prev} and {r['split']}")
            return 1
    print("    patient-split integrity: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
