#!/usr/bin/env python3
"""Regenerate the case-5 summary bar chart."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    res = json.loads((RESULTS / "case5_curated_results.json").read_text())
    labels = ["NO", "DPS\nwarmstart", "FlowMap\ncase-5 best"]
    vals = [
        res["no_mse_from_dps_log"],
        res["dps_best_visible"]["mse"],
        res["main_best"]["mse"],
    ]
    colors = ["#d65f5f", "#e6a23c", "#2d6cdf"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    ax.set_ylabel("relative MSE")
    ax.set_title("Hard case i=5 / g=25005")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"{val:.4f}", ha="center")
    ax.set_ylim(0, max(vals) * 1.22)
    fig.tight_layout()
    fig.savefig(FIGURES / "case5_mse_bar.png", dpi=180)


if __name__ == "__main__":
    main()
