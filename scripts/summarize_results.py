#!/usr/bin/env python3
"""Summarize the curated FlowMap inverse results and draw README figures."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def read_json_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        line = line.replace("NaN", "null")
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "count_under_0p08": int((arr < 0.08).sum()),
    }


def parse_dps_hard(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    case_blocks = re.split(r"=+\nCase i=", text)
    out: dict[str, dict] = {}
    for block in case_blocks:
        m = re.match(r"(\d+)\s+\|\s+NO MSE=([0-9.]+)", block)
        if not m:
            continue
        case = m.group(1)
        no_mse = float(m.group(2))
        methods = {}
        for method, val in re.findall(r"^\s+([A-Za-z0-9_]+) best=([0-9.]+)", block, re.M):
            methods[method] = float(val)
        out[case] = {
            "no_mse": no_mse,
            "methods": methods,
            "best_method": min(methods, key=methods.get) if methods else None,
            "best_mse": min(methods.values()) if methods else None,
        }
    return out


def main() -> None:
    FIGURES.mkdir(exist_ok=True)

    v113_log = RESULTS / "prop_ms_lgfmi_sgd_v113_full32_tmid01_lw2_s3.log"
    rows = read_json_rows(v113_log)
    method_vals = [float(r["fmda_auto_mse"]) for r in rows]
    no_vals = [float(r["no_mse"]) for r in rows]
    smooth_vals = [float(r["smooth_mse"]) for r in rows]

    kgml_path = RESULTS / "kgml_unet33m_cva_n32.json"
    kgml = json.loads(kgml_path.read_text())
    kgml_vals = [float(r["kgml_mse"]) for r in kgml["rows"]]

    dps = parse_dps_hard(RESULTS / "dps_hard_restart2.log")

    summary = {
        "main_method": "proposal_ms_lgfmi_grad",
        "best_run": "v113_full32_tmid01_lw2_s3",
        "main_v113": stats(method_vals),
        "neural_operator_baseline_from_eval_ms": stats(no_vals),
        "smoothed_no_baseline_from_eval_ms": stats(smooth_vals),
        "kgml_unet35m_baseline": stats(kgml_vals),
        "dps_hard_partial": dps,
        "notes": [
            "All main-method rows are raw-noise proposals: --no_seeded_frac 0.0 and --bg_mode zero.",
            "No learned inverse corrector is used; the learned model is only the unconditional FlowMap prior / NO baseline.",
            "DPS hard-case log is partial for i=7 and later; treat it as diagnostic, not a full n=32 benchmark.",
        ],
    }
    (RESULTS / "summary_curated.json").write_text(json.dumps(summary, indent=2))

    labels = ["Raw-noise\nFlowMap\n(v113)", "NO\nbaseline", "Smoothed\nNO", "KGML\nUNet 35M"]
    means = [summary["main_v113"]["mean"], summary["neural_operator_baseline_from_eval_ms"]["mean"], summary["smoothed_no_baseline_from_eval_ms"]["mean"], summary["kgml_unet35m_baseline"]["mean"]]
    medians = [summary["main_v113"]["median"], summary["neural_operator_baseline_from_eval_ms"]["median"], summary["smoothed_no_baseline_from_eval_ms"]["median"], summary["kgml_unet35m_baseline"]["median"]]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, means, width, label="mean", color="#2d6cdf")
    ax.bar(x + width / 2, medians, width, label="median", color="#58b368")
    ax.axhline(0.08, color="#d62728", linewidth=1.5, linestyle="--", label="0.08 target")
    ax.set_ylabel("relative MSE")
    ax.set_title("CVA hard split, n=32: raw-noise FlowMap inverse vs baselines")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for i, (m, med) in enumerate(zip(means, medians)):
        ax.text(i - width / 2, m + 0.012, f"{m:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, med + 0.012, f"{med:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "results_summary.png", dpi=180)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.hist(method_vals, bins=np.linspace(0, max(0.30, max(method_vals) + 0.02), 18), color="#2d6cdf", alpha=0.85)
    ax.axvline(np.mean(method_vals), color="black", linewidth=1.4, label=f"mean={np.mean(method_vals):.4f}")
    ax.axvline(0.08, color="#d62728", linewidth=1.5, linestyle="--", label="0.08 target")
    ax.set_xlabel("relative MSE")
    ax.set_ylabel("case count")
    ax.set_title("v113 per-case MSE distribution")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "v113_mse_histogram.png", dpi=180)


if __name__ == "__main__":
    main()
