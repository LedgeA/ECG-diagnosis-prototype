"""Stage 2 - harmonise every source into one WFDB dialect.

Three jobs, all of them load-bearing:

1. Format.   The kit loads .mat via loadmat()['val'], which returns raw ADC
   counts, while .dat goes through wfdb.rdrecord().p_signal and returns
   millivolts. The gain is read into a variable in extract_leads.py and then
   never applied, so Chapman renders 1000x over-scale and crashes on write.
   Reading everything through wfdb and writing .dat sidesteps it entirely.

2. Signal.   One identical zero-phase band-pass across all three sources. This
   is the main lever against the source/class confound, so it must never be
   applied per-source. Amplitude is deliberately left alone: normalising it
   would erase the voltage criteria that define LVH.

3. Identity. Records are renamed to an anonymous scheme and given #Age/#Sex
   comments. The kit prints "ID: <record name>" on the sheet, and the native
   names announce the source; PTB-XL and the STEMI dataset also carry no header
   comments at all, so metadata printing yields nothing until they are injected.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt

import config as C

_SEX_WORD = {"M": "Male", "F": "Female"}


def _bandpass_sos(fs: int):
    low, high = C.BANDPASS
    nyq = fs / 2.0
    return butter(2, [low / nyq, high / nyq], btype="band", output="sos")


def _canonical_leads(sig_name: list[str]) -> list[int]:
    """Return column indices that put the signal into standard 12-lead order."""
    # Case-folded because PTB-XL writes AVR/AVL/AVF where the other two
    # sources write aVR/aVL/aVF.
    lookup = {name.strip().upper(): i for i, name in enumerate(sig_name)}
    return [lookup[lead.upper()] for lead in C.LEAD_ORDER]


def transcode(row: dict, out_name: str, out_dir, sos) -> str | None:
    """Return None on success, or a short reason string on failure.

    Never raises: a single unreadable record must not abandon a build that is
    already thousands of records in. Stage 1 screens headers, but a truncated
    or corrupt signal file can still surface here.
    """
    try:
        return _transcode(row, out_name, out_dir, sos)
    except Exception as exc:                     # noqa: BLE001 - report and continue
        return f"error_{type(exc).__name__}"


def _transcode(row: dict, out_name: str, out_dir, sos) -> str | None:
    src = row["abs_path"].rsplit(".", 1)[0]
    rec = wfdb.rdrecord(src)                     # mV regardless of .dat/.mat

    sig = np.asarray(rec.p_signal, dtype=np.float64)
    if sig.shape[0] != C.N_SAMPLES or sig.shape[1] != len(C.LEAD_ORDER):
        return "bad_shape"
    if not np.isfinite(sig).all():
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        sig = sig[:, _canonical_leads(rec.sig_name)]
    except KeyError:
        return "bad_leads"

    # Zero-phase so the ST segment is not shifted in time - the whole STEMI
    # class depends on ST morphology surviving this step intact.
    sig = sosfiltfilt(sos, sig, axis=0)
    sig = sig - np.median(sig, axis=0, keepdims=True)

    # Keep well inside int16 at 1000 ADU/mV; real ECGs never reach +-32 mV.
    sig = np.clip(sig, -32.0, 32.0)

    # Both comments are mandatory: the kit's --print_header does a bare
    # dict lookup for 'Age' and 'Sex' and raises KeyError if either is absent.
    # Stage 1 already drops records without them; this is the backstop.
    if not row.get("age") or row.get("sex") not in _SEX_WORD:
        return "missing_metadata"
    comments = [f"Age: {row['age']}", f"Sex: {_SEX_WORD[row['sex']]}"]

    out_dir.mkdir(parents=True, exist_ok=True)
    wfdb.wrsamp(
        record_name=out_name,
        fs=C.FS,
        units=["mV"] * len(C.LEAD_ORDER),
        sig_name=list(C.LEAD_ORDER),
        p_signal=sig,
        fmt=["16"] * len(C.LEAD_ORDER),
        adc_gain=[C.ADC_GAIN] * len(C.LEAD_ORDER),
        baseline=[0] * len(C.LEAD_ORDER),
        comments=comments,
        write_dir=str(out_dir),
    )
    return None


def main() -> int:
    if not C.MANIFEST_FINAL.exists():
        print("manifest_final.csv missing - run stage2 first", file=sys.stderr)
        return 1
    rows = list(csv.DictReader(C.MANIFEST_FINAL.open()))

    sos = _bandpass_sos(C.FS)
    stats = Counter()
    mapping = []

    per_class = Counter()
    for i, row in enumerate(rows, start=1):
        cls = row["cls"]
        # Shard by class, then into fixed-size chunks, so stage 4 has many
        # independent batches to spread across cores.
        chunk = per_class[cls] // C.CHUNK_SIZE
        per_class[cls] += 1
        out_dir = C.STAGED / cls / f"c{chunk:03d}"
        out_name = C.RECORD_NAME_FMT.format(i)

        err = transcode(row, out_name, out_dir, sos)
        if err:
            stats[err] += 1
            continue

        stats[f"ok_{cls}"] += 1
        mapping.append({
            "record": out_name, "uid": row["uid"], "cls": cls,
            "chunk": f"c{chunk:03d}",
            "source": row["source"], "patient_id": row["patient_id"],
            "split": row["split"], "n_renders": row["n_renders"],
            "age": row["age"], "sex": row["sex"], "note": row["note"],
        })

        if i % 1000 == 0:
            print(f"  {i}/{len(rows)}")

    if not mapping:
        print("nothing staged", file=sys.stderr)
        return 1

    out_csv = C.BUILD / "staged_map.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(mapping[0].keys()))
        w.writeheader()
        w.writerows(mapping)

    print(f"\nstaged {len(mapping)} of {len(rows)} records -> {C.STAGED}")
    for k in sorted(stats):
        print(f"    {k:22s} {stats[k]:6d}")
    print(f"wrote {out_csv}")

    failed = len(rows) - len(mapping)
    if failed:
        pct = 100 * failed / len(rows)
        print(f"\n{failed} records ({pct:.1f}%) could not be staged.")
        if pct > 2.0:
            print("That is high enough to skew the class balance - investigate")
            print("before rendering rather than after.")
            return 1
        print("Small enough to ignore; class balance is essentially unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
