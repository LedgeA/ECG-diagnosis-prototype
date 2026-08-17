"""Stage 1 - build one manifest row per candidate record, from all three sources.

Labels are resolved here and nowhere else; every later stage reads the manifest
rather than a source catalogue. Records matching more than one target class are
dropped, as are paced records and atrial flutter.
"""
from __future__ import annotations

import ast
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

import config as C

FIELDS = ["uid", "source", "abs_path", "fmt", "cls",
          "patient_id", "age", "sex", "note"]


def _clean_age(value) -> str:
    try:
        age = float(value)
    except (TypeError, ValueError):
        return ""
    if age != age:          # NaN; Chapman writes literal "NaN" for unknown age
        return ""
    if age >= C.PTBXL_AGE_SENTINEL:
        # PTB-XL masks age > 89 as 300. The patient is genuinely over 89, so
        # render 90 rather than discarding an otherwise usable record.
        return str(C.AGE_MASKED_AS)
    if age <= 0:
        return ""
    return str(int(round(age)))


def _pair_exists(signal: Path) -> bool:
    """A record is usable only if header and signal are both present and whole.

    Existence is not enough. The local copies contain three distinct kinds of
    damage that all pass an exists() check and then fail deep inside wfdb:
    zero-byte .hea files (19 in PTB-XL), headers that parse but describe a
    different geometry, and truncated .dat files written at a block boundary
    (40 in PTB-XL, 2 in the STEMI dataset). Screening here keeps them out of
    the corpus target rather than silently shrinking it at stage 3.
    """
    header = signal.with_suffix(".hea")
    if not signal.exists() or not header.exists():
        return False
    try:
        lines = [ln.strip() for ln in header.read_text().splitlines()
                 if ln.strip() and not ln.startswith("#")]
        if len(lines) < 2:
            return False

        record_line = lines[0].split()
        if (len(record_line) < 4
                or record_line[1] != str(len(C.LEAD_ORDER))
                or record_line[2] != str(C.FS)
                or record_line[3] != str(C.N_SAMPLES)):
            return False

        # Signal line: "<file> <fmt>[+<offset>] ...". Only format 16 is used by
        # all three sources; anything else means our size arithmetic is wrong.
        fmt = lines[1].split()[1]
        base, _, offset = fmt.partition("+")
        if base != "16":
            return False
        expected = len(C.LEAD_ORDER) * C.N_SAMPLES * 2 + int(offset or 0)
        return signal.stat().st_size == expected
    except (OSError, ValueError, IndexError):
        return False


def _sole_class(hits: list[str]) -> str | None:
    """A record is usable only if it lands in exactly one target class."""
    return hits[0] if len(hits) == 1 else None


# ------------------------------------------------------------------ STEMI set
def load_stemi() -> tuple[list[dict], Counter]:
    rows, stats = [], Counter()
    df = pd.read_csv(C.STEMI_TRAIN_CSV, dtype=str).fillna("")

    # A patient who had an ACS event anywhere in the cohort is not a clean
    # control, even on a record that happens to be negative: it would put the
    # same person in both STEMI and NORMAL.
    diseased = {
        r["Patient_id"] for _, r in df.iterrows()
        if any(r.get(c) == "1" for c in C.STEMI_DX_COLUMNS)
        or any(r.get(c, "0") not in ("0", "") for c in C.STEMI_CONTROL_EXCLUDE)
    }

    for _, r in df.iterrows():
        stem = r["ecg_row_record"].replace(".dat", "")
        path = C.STEMI_RECORDS / f"{stem}.dat"
        if not _pair_exists(path):
            stats["missing_file"] += 1
            continue
        if r.get("Paced") == "1":
            stats["excluded_paced"] += 1
            continue

        positives = [c for c in C.STEMI_DX_COLUMNS if r.get(c) == "1"]
        if r.get("STEMI") == "1":
            cls, note = "STEMI", "stemi_positive"
        elif positives:
            stats["excluded_other_acs"] += 1
            continue
        elif r["Patient_id"] in diseased:
            # Either this record carries a malignant rhythm or coronary disease
            # that usually marks the ECG, or another record from the same
            # patient does. Not "normal" either way.
            stats["excluded_impure_control"] += 1
            continue
        else:
            # Clean same-source NORMAL control, so that source identity no
            # longer predicts the STEMI class on its own.
            cls, note = "NORMAL", "source_control"

        # 1 = male, 0 = female in this dataset's CSV.
        sex = "M" if r.get("gender") == "1" else "F"
        rows.append({
            "uid": f"stemi_{stem}", "source": "stemi", "abs_path": str(path),
            "fmt": "dat", "cls": cls, "patient_id": f"stemi_{r['Patient_id']}",
            "age": _clean_age(r.get("age")), "sex": sex, "note": note,
        })
        stats[f"kept_{cls}"] += 1
    return rows, stats


# ------------------------------------------------------------------- PTB-XL
def load_ptbxl() -> tuple[list[dict], Counter]:
    rows, stats = [], Counter()
    df = pd.read_csv(C.PTBXL_DB, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    for ecg_id, r in df.iterrows():
        codes = set(r["scp_codes"].keys())      # presence, not likelihood
        if codes & C.PTBXL_EXCLUDE_CODES:
            stats["excluded_paced_or_flutter"] += 1
            continue

        hits = [cls for cls, want in C.PTBXL_CLASS_CODES.items() if codes & want]
        cls = _sole_class(hits)
        if cls is None:
            stats["excluded_multiclass" if hits else "no_target_label"] += 1
            continue

        path = C.PTBXL_DIR / f"{r['filename_hr']}.dat"
        if not _pair_exists(path):
            # The local PTB-XL copy is incomplete; stage1 reports the shortfall.
            stats["missing_file"] += 1
            continue

        rows.append({
            "uid": f"ptbxl_{ecg_id:05d}", "source": "ptbxl", "abs_path": str(path),
            "fmt": "dat", "cls": cls,
            "patient_id": f"ptbxl_{int(r['patient_id'])}",
            "age": _clean_age(r.get("age")),
            "sex": "M" if int(r["sex"]) == 0 else "F",   # PTB-XL: 0 = male
            "note": f"fold{int(r['strat_fold'])}",
        })
        stats[f"kept_{cls}"] += 1
    return rows, stats


# ------------------------------------------------------------------ Chapman
_DX_RE = re.compile(r"#Dx:\s*(.*)")
_AGE_RE = re.compile(r"#Age:\s*(.*)")
_SEX_RE = re.compile(r"#Sex:\s*(.*)")


def load_chapman() -> tuple[list[dict], Counter]:
    rows, stats = [], Counter()
    for hea in sorted(C.CHAPMAN_RECORDS.rglob("*.hea")):
        stem = hea.stem
        if stem in C.CHAPMAN_BAD_RECORDS:
            stats["excluded_malformed"] += 1
            continue

        text = hea.read_text()
        head = text.split("\n", 1)[0].split()
        if len(head) < 4 or head[2] != str(C.FS) or head[3] != str(C.N_SAMPLES):
            stats["excluded_malformed"] += 1
            continue

        m = _DX_RE.search(text)
        if not m:
            stats["no_dx"] += 1
            continue
        codes = {c.strip() for c in m.group(1).split(",") if c.strip()}
        if codes & C.CHAPMAN_EXCLUDE_CODES:
            stats["excluded_flutter"] += 1
            continue

        hits = [cls for cls, want in C.CHAPMAN_CLASS_CODES.items() if codes & want]
        if codes == C.CHAPMAN_NORMAL_EXACT:      # sinus rhythm and nothing else
            hits.append("NORMAL")
        cls = _sole_class(hits)
        if cls is None:
            stats["excluded_multiclass" if hits else "no_target_label"] += 1
            continue

        mat = hea.with_suffix(".mat")
        if not _pair_exists(mat):
            stats["missing_file"] += 1
            continue

        sex_raw = (_SEX_RE.search(text).group(1).strip().lower()
                   if _SEX_RE.search(text) else "")
        age_raw = (_AGE_RE.search(text).group(1).strip()
                   if _AGE_RE.search(text) else "")
        rows.append({
            "uid": f"chapman_{stem}", "source": "chapman", "abs_path": str(mat),
            "fmt": "mat", "cls": cls,
            # Chapman carries no patient identifier; each record is treated as
            # its own patient. See README - this is an assumption, not a fact.
            "patient_id": f"chapman_{stem}",
            "age": _clean_age(age_raw),
            "sex": "M" if sex_raw.startswith("m") else "F" if sex_raw.startswith("f") else "",
            "note": "",
        })
        stats[f"kept_{cls}"] += 1
    return rows, stats


def main() -> int:
    C.BUILD.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for name, loader in (("STEMI-dataset", load_stemi),
                         ("PTB-XL", load_ptbxl),
                         ("Chapman+Ningbo", load_chapman)):
        rows, stats = loader()
        all_rows.extend(rows)
        print(f"\n{name}: {len(rows)} kept")
        for k in sorted(stats):
            print(f"    {k:28s} {stats[k]:6d}")

    # The kit's --print_header raises KeyError unless both #Age and #Sex are
    # present, and substituting a placeholder would be worse than dropping the
    # record: missing metadata is not uniform across sources, so "Unknown"
    # printed on the sheet would itself become a source cue.
    seen: set[str] = set()
    unique, no_meta = [], 0
    for r in all_rows:
        if r["uid"] in seen:
            continue
        seen.add(r["uid"])
        if not r["age"] or not r["sex"]:
            no_meta += 1
            continue
        unique.append(r)
    if no_meta:
        print(f"\ndropped {no_meta} records lacking age or sex")

    with C.MANIFEST_RAW.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(unique)

    print(f"\nwrote {C.MANIFEST_RAW}  ({len(unique)} rows)")
    print("\npool by class x source")
    grid = Counter((r["cls"], r["source"]) for r in unique)
    sources = ["stemi", "ptbxl", "chapman"]
    print(f"    {'class':10s}" + "".join(f"{s:>10s}" for s in sources) + f"{'total':>10s}")
    for cls in C.CLASSES:
        counts = [grid[(cls, s)] for s in sources]
        print(f"    {cls:10s}" + "".join(f"{c:10d}" for c in counts) + f"{sum(counts):10d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
