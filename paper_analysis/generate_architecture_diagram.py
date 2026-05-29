#!/usr/bin/env python3
"""Generate Figure 1: MeshFlowNet architecture diagram."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT_DIR = Path("paper_figures/main")
FIG_NAME = "fig01_architecture"


def add_box(ax, xy, width, height, title, body, facecolor):
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.02,rounding_size=0.015",
        linewidth=1.2, edgecolor="0.15", facecolor=facecolor,
    )
    ax.add_patch(box)
    x, y = xy
    ax.text(x + width / 2, y + height - 0.08, title,
            ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(x + width / 2, y + height / 2 - 0.03, body,
            ha="center", va="center", fontsize=9, linespacing=1.25)


def add_arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, color="0.15", shrinkA=6, shrinkB=6,
    ))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = {
        "local": ((0.03, 0.58), 0.19, 0.30),
        "global": ((0.03, 0.12), 0.19, 0.30),
        "encoder": ((0.28, 0.44), 0.16, 0.28),
        "processor": ((0.50, 0.44), 0.19, 0.28),
        "decoder": ((0.75, 0.44), 0.15, 0.28),
        "output": ((0.92, 0.44), 0.07, 0.28),
    }

    add_box(
        ax, *boxes["local"],
        "CONUS Grid Inputs",
        "0.25 deg, 621 x 1405\nT2max(t,t-1,t-2)\n9 local fields + terrain\nlat/lon, DOY, TOA, mask",
        "#dceefb",
    )
    add_box(
        ax, *boxes["global"],
        "Global ERA5 Context",
        "59 variables x 4 lags\n(t, t-3, t-7, t-14)\n181 x 360, periodic lon\n236 channels",
        "#e8f4df",
    )
    add_box(
        ax, *boxes["encoder"],
        "Grid-to-Mesh Encoder",
        "Bipartite GNN\nland grid -> level-7 mesh\nlocal skip projection\n~0.5M params",
        "#fff2cc",
    )
    add_box(
        ax, *boxes["processor"],
        "Icosahedral Mesh Processor",
        "8608 regional nodes\nmulti-mesh levels 4-7\n8 message-passing rounds\nFiLM conditioning\n~2.9M params",
        "#fde2e2",
    )
    add_box(
        ax, *boxes["decoder"],
        "Mesh-to-Grid Decoder",
        "Bipartite GNN\n8 mesh neighbors/grid point\nresidual grid refiner\n~0.4M params",
        "#eadcf8",
    )
    add_box(
        ax, *boxes["output"],
        "Output",
        "Day-15\nT2max\nz-score",
        "#eeeeee",
    )

    add_arrow(ax, (0.22, 0.73), (0.28, 0.58))
    add_arrow(ax, (0.22, 0.27), (0.50, 0.50))
    add_arrow(ax, (0.44, 0.58), (0.50, 0.58))
    add_arrow(ax, (0.69, 0.58), (0.75, 0.58))
    add_arrow(ax, (0.90, 0.58), (0.92, 0.58))

    ax.text(
        0.50, 0.93,
        "MeshFlowNet: direct day-15 CONUS T2max hindcast",
        ha="center", va="center", fontsize=13, fontweight="bold",
    )
    ax.text(
        0.50, 0.05,
        "Training target is daily PRISM T2max z-score; train-year-only local climatology is used for TAC and anomaly-map verification.",
        ha="center", va="center", fontsize=9,
    )

    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{FIG_NAME}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
