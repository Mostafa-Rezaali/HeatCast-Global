#!/usr/bin/env python3
"""
Generate training-curve figures and fold summary tables from a metrics CSV.

Expected CSV columns include:
fold, epoch, train_loss, val_r2, val_tac, lr, val_mse, val_ssim,
spatial_anom_r2, variance_ratio, gradient_ratio, mae, crps,
mse_skill_vs_persistence, persistence_tac, correlation.
Missing columns are tolerated and plotted/table-filled as NaN.
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


METRICS_CSV = Path("paper_inputs/fold_epoch_metrics.csv")
METRICS_GLOB = "training_metrics/fold_epoch_metrics_cvfold*.csv"
OUTPUT_ROOT = Path("paper_figures")


def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clean = {}
            for k, v in row.items():
                if k in ("fold", "epoch"):
                    clean[k] = int(float(v))
                else:
                    try:
                        clean[k] = float(v)
                    except Exception:
                        clean[k] = np.nan
            rows.append(clean)
    return rows


def read_all_metrics():
    if METRICS_CSV.exists():
        return read_csv(METRICS_CSV)
    import glob
    rows = []
    for path in sorted(glob.glob(METRICS_GLOB)):
        rows.extend(read_csv(path))
    if not rows:
        raise FileNotFoundError(
            f"No metrics CSV found. Expected {METRICS_CSV} or glob {METRICS_GLOB}."
        )
    return rows


def col(rows, name):
    return np.array([r.get(name, np.nan) for r in rows], dtype=float)


def best_rows_by_fold(rows, metric="val_tac"):
    out = []
    for fold in sorted(set(r["fold"] for r in rows)):
        fr = [r for r in rows if r["fold"] == fold]
        vals = np.array([r.get(metric, np.nan) for r in fr], dtype=float)
        idx = int(np.nanargmax(vals)) if np.isfinite(vals).any() else len(fr) - 1
        out.append(fr[idx])
    return out


def plot_training_curves(rows):
    out = OUTPUT_ROOT / "main"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})

    panels = [
        ("train_loss", "Training loss"),
        ("val_r2", "Validation R2"),
        ("val_tac", "Validation TAC"),
        ("lr", "Learning rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (name, ylabel) in zip(axes.ravel(), panels):
        for fold in sorted(set(r["fold"] for r in rows)):
            fr = [r for r in rows if r["fold"] == fold]
            x = col(fr, "epoch")
            y = col(fr, name)
            ax.plot(x, y, marker="o", linewidth=1.2, markersize=3, label=f"Fold {fold}")
            if name == "val_tac" and np.isfinite(y).any():
                best_epoch = x[int(np.nanargmax(y))]
                ax.axvline(best_epoch, color="0.5", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig05_training_curves.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_variance_gradient(rows):
    out = OUTPUT_ROOT / "supplementary"
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for fold in sorted(set(r["fold"] for r in rows)):
        fr = [r for r in rows if r["fold"] == fold]
        x = col(fr, "epoch")
        axes[0].plot(x, col(fr, "variance_ratio"), marker="o", linewidth=1.1, markersize=3, label=f"Fold {fold}")
        axes[1].plot(x, col(fr, "gradient_ratio"), marker="o", linewidth=1.1, markersize=3, label=f"Fold {fold}")
    axes[0].set_ylabel("Variance ratio")
    axes[1].set_ylabel("Gradient ratio")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figS01_variance_gradient_evolution.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_summary(best):
    out = OUTPUT_ROOT / "main"
    metrics = [
        ("R2", "val_r2", "persistence_r2"),
        ("Spatial R2", "spatial_anom_r2", "persistence_spatial_anom_r2"),
        ("TAC", "val_tac", "persistence_tac"),
        ("MSE skill", "mse_skill_vs_persistence", None),
        ("Corr", "correlation", "persistence_correlation"),
        ("CRPS", "crps", "persistence_mae"),
        ("MAE", "mae", "persistence_mae"),
    ]
    model_means, model_stds, pers_means, pers_stds = [], [], [], []
    for _, mcol, pcol in metrics:
        mv = col(best, mcol)
        model_means.append(np.nanmean(mv))
        model_stds.append(np.nanstd(mv))
        if pcol is None:
            pers_means.append(np.nan)
            pers_stds.append(np.nan)
        else:
            pv = col(best, pcol)
            pers_means.append(np.nanmean(pv))
            pers_stds.append(np.nanstd(pv))

    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - 0.18, model_means, width=0.36, yerr=model_stds, capsize=3, label="MeshFlowNet")
    ax.bar(x + 0.18, pers_means, width=0.36, yerr=pers_stds, capsize=3, label="Persistence")
    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in metrics], rotation=25, ha="right")
    ax.axhline(0, color="0.2", linewidth=0.8)
    ax.set_title("Cross-validated validation metrics at best TAC epoch")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig04_metrics_summary.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_tables(best):
    out = OUTPUT_ROOT / "tables"
    out.mkdir(parents=True, exist_ok=True)

    def mean_std(name):
        vals = col(best, name)
        return np.nanmean(vals), np.nanstd(vals)

    table2 = [
        ("R2", "val_r2", "persistence_r2"),
        ("Spatial Anomaly R2", "spatial_anom_r2", "persistence_spatial_anom_r2"),
        ("TAC", "val_tac", "persistence_tac"),
        ("MSE Skill Score", "mse_skill_vs_persistence", None),
        ("Correlation", "correlation", "persistence_correlation"),
        ("MAE", "mae", "persistence_mae"),
        ("CRPS", "crps", "persistence_mae"),
        ("SSIM", "val_ssim", None),
    ]
    with open(out / "table2_main_results.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lccc}\\hline\n")
        f.write("Metric & MeshFlowNet & Persistence & Improvement \\\\\\hline\n")
        for label, mcol, pcol in table2:
            mm, ms = mean_std(mcol)
            if pcol is None:
                f.write(f"{label} & {mm:.3f} $\\pm$ {ms:.3f} & -- & -- \\\\\n")
            else:
                pm, ps = mean_std(pcol)
                f.write(f"{label} & {mm:.3f} $\\pm$ {ms:.3f} & {pm:.3f} $\\pm$ {ps:.3f} & {mm - pm:+.3f} \\\\\n")
        f.write("\\hline\\end{tabular}\n")

    with open(out / "table3_per_fold.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrr}\\hline\n")
        f.write("Fold & Best epoch & R2 & TAC & Spatial R2 & MAE \\\\\\hline\n")
        for r in best:
            f.write(
                f"{r['fold']} & {r['epoch']} & {r.get('val_r2', np.nan):.3f} & "
                f"{r.get('val_tac', np.nan):.3f} & {r.get('spatial_anom_r2', np.nan):.3f} & "
                f"{r.get('mae', np.nan):.3f} \\\\\n"
            )
        f.write("\\hline\\end{tabular}\n")


def main():
    rows = read_all_metrics()
    best = best_rows_by_fold(rows, metric="val_tac")
    plot_training_curves(rows)
    plot_variance_gradient(rows)
    plot_metrics_summary(best)
    write_tables(best)


if __name__ == "__main__":
    main()
