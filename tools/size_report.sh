#!/usr/bin/env bash
# Reports deployment footprint for the KWS RISC-V build:
#   - .tflite model size
#   - ELF section breakdown (text/rodata/data/bss)
#   - Flash & RAM estimates
#   - Runtime peak tensor-arena usage (by running under spike, optional)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ELF="${ELF:-${ROOT}/build/kws_demo.elf}"
TFLITE="${TFLITE:-${ROOT}/artifacts/ds_cnn/model_int8.tflite}"
SIZE_TOOL="${SIZE_TOOL:-riscv64-unknown-elf-size}"
RUN_SPIKE="${RUN_SPIKE:-0}"

die() { echo "error: $*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

[[ -f "$ELF" ]]    || die "ELF not found: $ELF  (run 'make' first)"
[[ -f "$TFLITE" ]] || die "model not found: $TFLITE"
have "$SIZE_TOOL"  || die "missing $SIZE_TOOL (install riscv toolchain)"

file_size() {
  if stat -f%z "$1" >/dev/null 2>&1; then stat -f%z "$1"     # BSD / macOS
  else                                     stat -c%s "$1"    # GNU / Linux
  fi
}

kb() { awk -v b="$1" 'BEGIN{printf "%.1f KB", b/1024}'; }

model_bytes=$(file_size "$TFLITE")

# Parse per-section sizes from `size -A` output.
# Use awk (bash 3.2 on macOS has no associative arrays).
section_size() {
  "$SIZE_TOOL" -A "$ELF" | awk -v s="$1" '$1==s {print $2; found=1} END{if(!found) print 0}'
}

text=$(section_size .text)
rodata=$(section_size .rodata)
data=$(section_size .data)
sdata=$(section_size .sdata)
bss=$(section_size .bss)
sbss=$(section_size .sbss)

flash=$((text + rodata + data + sdata))
ram=$((data + sdata + bss + sbss))

arena_line=$(grep -nE 'kTensorArenaSize[[:space:]]*=' \
  "$ROOT/tflm_demo/kws_inference.cc" | head -1 || true)

printf '=== KWS TinyML size report ===\n'
printf 'ELF:   %s\n' "$ELF"
printf 'Model: %s\n\n' "$TFLITE"

printf '%s\n' '-- Model --'
printf '  int8 tflite:               %8d B   (%s)\n\n' "$model_bytes" "$(kb "$model_bytes")"

printf '%s\n' '-- ELF sections --'
printf '  .text    (code, Flash)     %8d B   (%s)\n' "$text"   "$(kb "$text")"
printf '  .rodata  (const, Flash)    %8d B   (%s)\n' "$rodata" "$(kb "$rodata")"
printf '  .data    (init RW, F->R)   %8d B   (%s)\n' "$data"   "$(kb "$data")"
printf '  .sdata   (small init RW)   %8d B   (%s)\n' "$sdata"  "$(kb "$sdata")"
printf '  .bss     (zero RW, RAM)    %8d B   (%s)\n' "$bss"    "$(kb "$bss")"
printf '  .sbss    (small zero RW)   %8d B   (%s)\n\n' "$sbss" "$(kb "$sbss")"

printf '%s\n' '-- Deployment footprint --'
printf '  Flash = text+rodata+data+sdata: %8d B  (%s)\n' "$flash" "$(kb "$flash")"
printf '  RAM   = data+sdata+bss+sbss:    %8d B  (%s)\n\n' "$ram" "$(kb "$ram")"

printf '%s\n' '-- Tensor arena (compile-time) --'
[[ -n "$arena_line" ]] && printf '  %s\n' "$arena_line"
printf '  Note: arena is part of .bss above; shrinking it shrinks RAM 1:1.\n\n'

if [[ "$RUN_SPIKE" == "1" ]]; then
  have spike || die "RUN_SPIKE=1 but spike is not on PATH"
  PK="${PK:-/opt/homebrew/Cellar/riscv-pk/main/riscv64-unknown-elf/bin/pk}"
  [[ -x "$PK" ]] || die "PK not found at $PK (override with PK=...)"
  printf '%s\n' '-- Runtime (spike) --'
  spike "$PK" "$ELF" 2>&1 | grep -E 'ARENA|FOOTPRINT' || \
    echo '  (no ARENA/FOOTPRINT lines captured; check binary output)'
else
  printf 'Tip: pass RUN_SPIKE=1 to execute under spike and capture the\n'
  printf '     runtime peak arena usage (the ARENA / FOOTPRINT lines).\n'
fi
