#!/usr/bin/env python3
"""Build and run ref / opt / rvv variants and print a per-op instret
breakdown in the format used in the research write-up.

Columns: Op, ref ticks, opt ticks, rvv ticks, rvv/ref speedup.
Rows:
  - Conv2D first  (the single first-layer 5x3 regular conv, event index 0)
  - Conv2D pointwise 1x1 (×5, averaged per call)
  - DepthwiseConv 5x3    (×5, averaged per call)
  - MEAN (single call, informational)
  - TOTAL invoke (from BENCH line, same as sum of per-op + tiny profiler overhead)

Usage:
    python tools/per_op_table.py
    MODEL_DIR=artifacts/ds_cnn_M_retrained python tools/per_op_table.py
"""
import os
import re
import subprocess
import sys
from collections import defaultdict

MODEL_DIR = os.environ.get("MODEL_DIR", "artifacts/ds_cnn_M_retrained")
VARIANTS = ("ref", "opt", "rvv")

EVENT_RE = re.compile(r"^(\d+),(\w+),(\d+)$")
BENCH_RE = re.compile(r"^BENCH variant=(\S+) invoke_cycles=(\d+) invoke_instret=(\d+)")


def run_variant(kernel: str) -> dict:
    """Build + run one variant, return per-event ticks and the BENCH line."""
    # Build (silent; errors will surface if any).
    subprocess.run(
        ["make", "-s", f"KERNEL={kernel}", f"MODEL_DIR={MODEL_DIR}"],
        check=True,
    )
    proc = subprocess.run(
        ["make", "-s", f"KERNEL={kernel}", f"MODEL_DIR={MODEL_DIR}", "run"],
        check=True,
        capture_output=True,
        text=True,
    )
    events = []
    invoke_instret = None
    for line in proc.stdout.splitlines():
        m = EVENT_RE.match(line)
        if m:
            events.append((int(m.group(1)), m.group(2), int(m.group(3))))
            continue
        m = BENCH_RE.match(line)
        if m:
            invoke_instret = int(m.group(3))
    return {"events": events, "invoke_instret": invoke_instret}


def categorize(events):
    """Split CONV_2D events into first-layer (idx 0) vs pointwise (rest)."""
    conv = [e for e in events if e[1] == "CONV_2D"]
    dw = [e[2] for e in events if e[1] == "DEPTHWISE_CONV_2D"]
    mean = [e[2] for e in events if e[1] == "MEAN"]
    first = conv[0][2] if conv else 0
    pw = [e[2] for e in conv[1:]]
    return {
        "first_conv": first,
        "pointwise_avg": sum(pw) / len(pw) if pw else 0,
        "pointwise_count": len(pw),
        "depthwise_avg": sum(dw) / len(dw) if dw else 0,
        "depthwise_count": len(dw),
        "mean": mean[0] if mean else 0,
    }


def fmt(n):
    if n >= 1e6:
        return f"{n/1e6:>8.2f}M"
    if n >= 1e3:
        return f"{n/1e3:>8.1f}K"
    return f"{n:>9.0f}"


def main():
    data = {k: run_variant(k) for k in VARIANTS}
    cats = {k: categorize(v["events"]) for k, v in data.items()}
    total = {k: v["invoke_instret"] for k, v in data.items()}

    rows = [
        ("Conv2D first (5x3, x1)", "first_conv", 1),
        (f"Conv2D pointwise 1x1 (x{cats['ref']['pointwise_count']}, avg/call)",
         "pointwise_avg", cats["ref"]["pointwise_count"]),
        (f"DepthwiseConv 5x3 (x{cats['ref']['depthwise_count']}, avg/call)",
         "depthwise_avg", cats["ref"]["depthwise_count"]),
        ("MEAN", "mean", 1),
    ]

    header = f"{'Op':<40}{'ref':>11}{'opt':>11}{'rvv':>11}   rvv/ref"
    print(header)
    print("-" * len(header))
    for label, key, _count in rows:
        r, o, v = cats["ref"][key], cats["opt"][key], cats["rvv"][key]
        speedup = f"{r/v:>6.2f}x" if v else "     n/a"
        print(f"{label:<40}{fmt(r)}{fmt(o)}{fmt(v)}   {speedup}")
    print("-" * len(header))
    rT, oT, vT = total["ref"], total["opt"], total["rvv"]
    speedup = f"{rT/vT:>6.2f}x" if vT else "     n/a"
    print(f"{'TOTAL invoke_instret':<40}{fmt(rT)}{fmt(oT)}{fmt(vT)}   {speedup}")
    print(f"\nsoftware contribution  X = ref/opt = {rT/oT:.2f}x")
    print(f"ISA extension         Y = opt/rvv = {oT/vT:.2f}x")
    print(f"combined              X*Y         = {rT/vT:.2f}x")


if __name__ == "__main__":
    main()
