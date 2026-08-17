#!/usr/bin/env bash
# Build the four-class ECG image corpus end to end.
#
#   ./run_all.sh              full build
#   ./run_all.sh 4            resume from stage 4
#
# Every stage is idempotent: stages 4 and 5 skip work that already exists, so an
# interrupted build can be restarted with the same command.
set -euo pipefail

cd "$(dirname "$0")"
PY="${ECGKIT_PY:-$HOME/.cache/ecgkit-venv/bin/python}"
FROM="${1:-1}"
WORKERS="${WORKERS:-8}"

if [[ ! -x "$PY" ]]; then
    echo "interpreter not found: $PY" >&2
    echo "create it with the commands in README.md, or set ECGKIT_PY" >&2
    exit 1
fi

run() {
    local n=$1; shift
    if (( n < FROM )); then
        echo "--- stage $n: skipped"
        return
    fi
    echo
    echo "=== stage $n: $* ==="
    local start=$SECONDS
    "$PY" "$@"
    echo "--- stage $n done in $(( (SECONDS - start) / 60 )) min"
}

echo "=== preflight: patching ecg-image-kit ==="
"$PY" patch_kit.py

run 1 stage1_manifest.py
run 2 stage2_select.py
run 3 stage3_transcode.py
run 4 stage4_render.py -j "$WORKERS"
run 5 stage5_augment.py -j "$WORKERS"
run 6 stage6_verify.py

echo
echo "corpus ready: $(cd .. && pwd)/build/images"
echo "index:        $(cd .. && pwd)/build/index.csv"
