#!/usr/bin/env bash
# Build and run all three kernel variants (ref / opt / rvv) on Spike for a
# single input. Emits a comparison summary that flags any mismatch in the
# predicted label, score, or top output so we can catch kernel bugs early.
#
# Usage:
#   tools/run_variants.sh                         # default: MODEL_DIR=artifacts/ds_cnn_M_retrained
#   MODEL_DIR=artifacts/ds_cnn tools/run_variants.sh
#   VLEN=256 tools/run_variants.sh
#   tools/run_variants.sh path/to/custom_input.bin
#
# A non-zero exit status indicates either a build failure or a bit-mismatch
# between variants — i.e. the opt / rvv kernels diverged from reference.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-artifacts/ds_cnn_M_retrained}"
VLEN="${VLEN:-128}"
INPUT_ARG="${1:-}"

cd "$(dirname "$0")/.."

TMP_OUT="$(mktemp -d)"
trap 'rm -rf "$TMP_OUT"' EXIT

RESULT_LINES=()

for KERNEL in ref opt rvv; do
  echo "=== build + run KERNEL=$KERNEL ==="
  make -s KERNEL="$KERNEL" MODEL_DIR="$MODEL_DIR" VLEN="$VLEN"
  OUT="$TMP_OUT/$KERNEL.out"
  if [[ -n "$INPUT_ARG" ]]; then
    make -s KERNEL="$KERNEL" MODEL_DIR="$MODEL_DIR" VLEN="$VLEN" run \
      >"$OUT" 2>&1 <"$INPUT_ARG" || true
  else
    make -s KERNEL="$KERNEL" MODEL_DIR="$MODEL_DIR" VLEN="$VLEN" run >"$OUT" 2>&1 || true
  fi
  echo "--- $KERNEL output ---"
  grep -E '^(RESULT|BENCH|FOOTPRINT|KWS kernel|ARENA)' "$OUT" || true
  RESULT=$(grep '^RESULT' "$OUT" | head -1 || echo "")
  BENCH=$(grep '^BENCH' "$OUT" | head -1 || echo "")
  RESULT_LINES+=("$KERNEL: $RESULT | $BENCH")
done

echo
echo "=== comparison summary ==="
printf '%s\n' "${RESULT_LINES[@]}"

# Extract just the RESULT line for bit-identity check.
REF_RESULT=$(grep '^RESULT' "$TMP_OUT/ref.out" | head -1)
MISMATCH=0
for KERNEL in opt rvv; do
  THIS=$(grep '^RESULT' "$TMP_OUT/$KERNEL.out" | head -1)
  if [[ "$THIS" != "$REF_RESULT" ]]; then
    echo "MISMATCH: $KERNEL diverged from ref"
    echo "  ref: $REF_RESULT"
    echo "  $KERNEL: $THIS"
    MISMATCH=1
  fi
done

if [[ $MISMATCH -eq 0 ]]; then
  echo "OK: ref / opt / rvv produced matching RESULT lines"
fi
exit $MISMATCH
