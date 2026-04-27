"""Generate poster figures comparing DNN / DS-CNN / GRU on Speech Commands V2.

Reads the per-tier and baseline numbers below (filled by hand from each
team member's report.json) and produces 4 publication-ready figures:

  1. Pareto plot: Memory (KB, log) vs Test Accuracy
  2. Grouped bar chart: Accuracy across S/M/L
  3. Ops/Memory ratio (log) — illustrates weight reuse profile
  4. Int8 vs Fp32 quantization loss per architecture

Usage:
    python tools/plot_results.py                # writes figures/*.png
    python tools/plot_results.py --pdf          # also writes figures/*.pdf
    python tools/plot_results.py --out poster   # custom output dir
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Per-tier numbers (top table). Acc = int8 test accuracy in %, Mem in KB, Ops in M.
TIER_DATA = {
    "DS-CNN": {
        "S": {"acc": 93.06, "mem": 33.5,  "ops": 5.5},
        "M": {"acc": 93.89, "mem": 54.4,  "ops": 16.2},
        "L": {"acc": 94.83, "mem": 153.0, "ops": 31.5},
    },
    "GRU": {
        "S": {"acc": 89.31, "mem": 7.3,   "ops": 6.04},
        "M": {"acc": 93.97, "mem": 20.6,  "ops": 12.96},
        "L": {"acc": 95.90, "mem": 65.3,  "ops": 34.03},
    },
    "DNN": {
        "S": {"acc": 80.01, "mem": 79.5,  "ops": 0.16},
        "M": {"acc": 80.11, "mem": 190.5, "ops": 0.39},
        "L": {"acc": 82.74, "mem": 358.3, "ops": 0.73},
    },
}

# Baseline numbers (bottom table) — fp32 vs int8 test accuracy on the
# Task 1 baseline architecture for each network.
BASELINE = {
    "DS-CNN": {"fp32_test": 94.82, "int8_test": 94.76},
    "GRU":    {"fp32_test": 94.02, "int8_test": 93.97},
    "DNN":    {"fp32_test": 78.29, "int8_test": 77.40},
}

# Consistent colors across all figures.
COLORS = {
    "DS-CNN": "#2E75B6",  # steel blue
    "GRU":    "#C45BAA",  # magenta-pink
    "DNN":    "#70AD47",  # leaf green
}
ARCH_ORDER = ["DS-CNN", "GRU", "DNN"]
TIER_ORDER = ["S", "M", "L"]
TIER_MARKERS = {"S": "o", "M": "s", "L": "D"}


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


# ---------------------------------------------------------------------------
# Figure 1: Pareto plot
# ---------------------------------------------------------------------------


def fig_pareto(out_dir: Path, also_pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for arch in ARCH_ORDER:
        mems = [TIER_DATA[arch][t]["mem"] for t in TIER_ORDER]
        accs = [TIER_DATA[arch][t]["acc"] for t in TIER_ORDER]
        # Connecting line, lightly transparent
        ax.plot(mems, accs, "-", color=COLORS[arch], linewidth=2,
                alpha=0.5, zorder=1)
        # Per-tier markers
        for tier, mem, acc in zip(TIER_ORDER, mems, accs):
            ax.scatter(mem, acc, s=140, color=COLORS[arch],
                       marker=TIER_MARKERS[tier], edgecolor="white",
                       linewidth=1.5, zorder=3,
                       label=arch if tier == "S" else None)
            ax.annotate(f"{arch}-{tier}", (mem, acc),
                        xytext=(7, 6), textcoords="offset points",
                        fontsize=9, color=COLORS[arch], weight="bold",
                        zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Memory (KB, log scale)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Memory vs Accuracy Pareto Frontier")
    ax.set_ylim(78, 97)

    # Architecture legend
    ax.legend(loc="lower right", title="Architecture")

    # Tier-marker secondary legend
    tier_handles = [
        plt.Line2D([], [], marker=TIER_MARKERS[t], color="gray",
                   linestyle="", markersize=9, label=f"{t} tier")
        for t in TIER_ORDER
    ]
    leg2 = ax.legend(handles=tier_handles, loc="upper left",
                     title="Budget tier", bbox_to_anchor=(0.02, 0.98))
    ax.add_artist(leg2)
    # Re-add the architecture legend (matplotlib drops the first when adding the second)
    arch_handles = [
        plt.Line2D([], [], marker="o", color=COLORS[a], linestyle="-",
                   markersize=9, linewidth=2, label=a)
        for a in ARCH_ORDER
    ]
    ax.legend(handles=arch_handles, loc="lower right", title="Architecture")

    _save(fig, out_dir, "fig1_pareto", also_pdf)


# ---------------------------------------------------------------------------
# Figure 2: Grouped bar chart of accuracy
# ---------------------------------------------------------------------------


def fig_accuracy_bars(out_dir: Path, also_pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    n_archs = len(ARCH_ORDER)
    n_tiers = len(TIER_ORDER)
    width = 0.26
    x = np.arange(n_tiers)

    for i, arch in enumerate(ARCH_ORDER):
        accs = [TIER_DATA[arch][t]["acc"] for t in TIER_ORDER]
        offset = (i - (n_archs - 1) / 2) * width
        bars = ax.bar(x + offset, accs, width, label=arch,
                      color=COLORS[arch], edgecolor="white", linewidth=1.5)
        # Value labels on top
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    acc + 0.4, f"{acc:.1f}%",
                    ha="center", va="bottom", fontsize=9,
                    color=COLORS[arch], weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t} tier" for t in TIER_ORDER])
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy across Budget Tiers")
    ax.set_ylim(75, 100)
    ax.legend(loc="lower right", title="Architecture", ncol=3)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))

    _save(fig, out_dir, "fig2_accuracy_bars", also_pdf)


# ---------------------------------------------------------------------------
# Figure 3: Ops/Memory ratio (log)
# ---------------------------------------------------------------------------


def fig_ops_mem_ratio(out_dir: Path, also_pdf: bool) -> None:
    """Mean Ops/Mem ratio per architecture, averaged across tiers."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    ratios = {}
    for arch in ARCH_ORDER:
        # Ops in MOps, Mem in KB. ratio = (Ops * 1e6) / (Mem * 1024) = ops per byte.
        per_tier = [
            (TIER_DATA[arch][t]["ops"] * 1e6) / (TIER_DATA[arch][t]["mem"] * 1024)
            for t in TIER_ORDER
        ]
        ratios[arch] = (np.mean(per_tier), per_tier)

    archs = ARCH_ORDER
    means = [ratios[a][0] for a in archs]
    colors = [COLORS[a] for a in archs]

    bars = ax.barh(archs, means, color=colors,
                   edgecolor="white", linewidth=1.5, height=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Ops / Memory ratio  (operations per byte, log scale)")
    ax.set_title("Weight-Reuse Profile: Ops/Mem Ratio per Architecture")
    ax.invert_yaxis()  # DS-CNN on top

    # Annotate bars with mean ratio + interpretation
    interpretations = {
        "DS-CNN": "spatial reuse over H × W",
        "GRU":    "temporal reuse over T = 49",
        "DNN":    "no reuse (each weight used once)",
    }
    for bar, arch, mean in zip(bars, archs, means):
        ax.text(mean * 1.15, bar.get_y() + bar.get_height() / 2,
                f"≈ {mean:,.0f} ops/byte\n({interpretations[arch]})",
                va="center", ha="left", fontsize=10,
                color=COLORS[arch])

    ax.set_xlim(0.5, 5e3)
    ax.grid(True, axis="x", alpha=0.25, linestyle="--")
    ax.grid(False, axis="y")

    _save(fig, out_dir, "fig3_ops_mem_ratio", also_pdf)


# ---------------------------------------------------------------------------
# Figure 4: Int8 quantization loss
# ---------------------------------------------------------------------------


def fig_quant_loss(out_dir: Path, also_pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    losses_pp = []
    fp32 = []
    int8 = []
    for arch in ARCH_ORDER:
        b = BASELINE[arch]
        fp32.append(b["fp32_test"])
        int8.append(b["int8_test"])
        losses_pp.append(b["fp32_test"] - b["int8_test"])

    x = np.arange(len(ARCH_ORDER))
    width = 0.36
    bars_fp = ax.bar(x - width/2, fp32, width,
                     color=[COLORS[a] for a in ARCH_ORDER],
                     edgecolor="white", linewidth=1.5,
                     alpha=0.55, label="fp32")
    bars_q = ax.bar(x + width/2, int8, width,
                    color=[COLORS[a] for a in ARCH_ORDER],
                    edgecolor="white", linewidth=1.5,
                    label="int8 (post-training quant)")

    # Annotate: fp32 → int8 with delta on top of int8 bar
    for xi, fp, q, loss in zip(x, fp32, int8, losses_pp):
        ax.text(xi + width/2, q + 0.8,
                f"−{loss:.2f} pp",
                ha="center", va="bottom",
                fontsize=10, weight="bold",
                color="#444")

    ax.set_xticks(x)
    ax.set_xticklabels(ARCH_ORDER)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Quantization Loss: fp32 vs int8 Baseline")
    ax.set_ylim(70, 102)
    ax.legend(loc="lower right")
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))

    _save(fig, out_dir, "fig4_quant_loss", also_pdf)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save(fig, out_dir: Path, name: str, also_pdf: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    fig.savefig(png_path)
    print(f"  wrote {png_path}")
    if also_pdf:
        pdf_path = out_dir / f"{name}.pdf"
        fig.savefig(pdf_path)
        print(f"  wrote {pdf_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures"),
                        help="Output directory (default: figures/)")
    parser.add_argument("--pdf", action="store_true",
                        help="Also write PDF copies alongside PNG.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    print(f"Writing figures to {args.out}/")
    fig_pareto(args.out, args.pdf)
    fig_accuracy_bars(args.out, args.pdf)
    fig_ops_mem_ratio(args.out, args.pdf)
    fig_quant_loss(args.out, args.pdf)
    print("Done.")


if __name__ == "__main__":
    main()
