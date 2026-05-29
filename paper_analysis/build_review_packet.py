#!/usr/bin/env python3
"""
Build a compact review packet from MeshFlowNet cross-validation outputs.

Run from the project root after all folds have exported and stitched:

    python paper_analysis/build_review_packet.py

Outputs:
  paper_figures/review_packet/
    review_summary.txt
    per_fold_best_metrics.csv
    monthly_breakdown.csv
    stitched_tac_maps.png
    training_curves_val_r2_tac.png
    samples/sample_*.png
"""

from pathlib import Path
import csv
import glob
import math

import numpy as np
import matplotlib.pyplot as plt


STITCHED_TAC_NPZ = Path("hindcast_stats/stitched_5fold_tac.npz")
TRAINING_METRICS_GLOB = "training_metrics/fold_epoch_metrics_cvfold*.csv"
SAMPLE_SUMMARY_GLOB = "hindcast_paper_data/hindcast_sample_summary_cvfold*_test.npz"
SELECTED_MAPS_GLOB = "hindcast_paper_data/hindcast_selected_maps_cvfold*_test.npz"
MONTHLY_STATS_GLOB = "hindcast_paper_data/hindcast_monthly_stats_cvfold*_test.npz"
OUTPUT_DIR = Path("paper_figures/review_packet")
EXTENT = [-130.0, -60.0, 25.0, 50.0]
STAT_KEYS = (
    "pred_sum",
    "truth_sum",
    "pred_sq_sum",
    "truth_sq_sum",
    "pred_truth_sum",
    "persist_sum",
    "persist_sq_sum",
    "persist_truth_sum",
    "count",
)


def load_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except Exception:
        return None, None


def add_map(ax, data, title, cmap, vmin, vmax, ccrs=None, cfeature=None):
    data = np.ma.masked_invalid(data)
    if ccrs is not None:
        im = ax.imshow(
            data,
            extent=EXTENT,
            origin="upper",
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.45)
        ax.add_feature(cfeature.STATES, linewidth=0.2)
    else:
        im = ax.imshow(
            data,
            extent=EXTENT,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_xlim(EXTENT[0], EXTENT[1])
        ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_title(title, fontsize=10)
    return im


def corr_from_sums(x_sum, y_sum, x_sq_sum, y_sq_sum, xy_sum, count, mask, min_count=2):
    valid = (mask > 0.5) & (count >= min_count)
    safe_count = np.maximum(count, 1.0)
    cov = xy_sum - (x_sum * y_sum) / safe_count
    x_var = x_sq_sum - (x_sum * x_sum) / safe_count
    y_var = y_sq_sum - (y_sum * y_sum) / safe_count
    denom = np.sqrt(np.maximum(x_var, 0.0) * np.maximum(y_var, 0.0))
    valid &= denom > 1e-12
    corr = np.full(count.shape, np.nan, dtype=np.float32)
    corr[valid] = (cov[valid] / denom[valid]).astype(np.float32)
    mean = float(np.nanmean(corr[valid])) if np.any(valid) else float("nan")
    return corr, mean


def read_training_rows():
    rows = []
    for path in sorted(glob.glob(TRAINING_METRICS_GLOB)):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                clean = {}
                for k, v in row.items():
                    if k in ("run_name", "early_stop_metric"):
                        clean[k] = v
                    elif k in ("fold", "epoch", "early_stop_failures"):
                        clean[k] = int(float(v)) if str(v).strip() else -1
                    else:
                        try:
                            clean[k] = float(v)
                        except Exception:
                            clean[k] = np.nan
                rows.append(clean)
    return rows


def best_rows_by_fold(rows):
    best = []
    for fold in sorted({r["fold"] for r in rows if r.get("fold", -1) >= 0}):
        fr = [r for r in rows if r.get("fold") == fold]
        vals = np.array([r.get("val_tac", np.nan) for r in fr], dtype=float)
        idx = int(np.nanargmax(vals)) if np.isfinite(vals).any() else len(fr) - 1
        best.append(fr[idx])
    return best


def write_best_metrics(best):
    path = OUTPUT_DIR / "per_fold_best_metrics.csv"
    cols = [
        "fold",
        "epoch",
        "val_r2",
        "val_tac",
        "persistence_tac",
        "spatial_anom_r2",
        "mae",
        "crps",
        "mse_skill_vs_persistence",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in best:
            writer.writerow({c: row.get(c, "") for c in cols})
    return path


def plot_training_curves(rows):
    if not rows:
        return None
    path = OUTPUT_DIR / "training_curves_val_r2_tac.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for fold in sorted({r["fold"] for r in rows if r.get("fold", -1) >= 0}):
        fr = sorted([r for r in rows if r.get("fold") == fold], key=lambda r: r["epoch"])
        epochs = np.array([r["epoch"] for r in fr], dtype=float)
        val_r2 = np.array([r.get("val_r2", np.nan) for r in fr], dtype=float)
        val_tac = np.array([r.get("val_tac", np.nan) for r in fr], dtype=float)
        axes[0].plot(epochs, val_r2, marker="o", linewidth=1.2, markersize=3, label=f"fold {fold}")
        axes[1].plot(epochs, val_tac, marker="o", linewidth=1.2, markersize=3, label=f"fold {fold}")
        if np.isfinite(val_tac).any():
            best_epoch = epochs[int(np.nanargmax(val_tac))]
            axes[1].axvline(best_epoch, color="0.6", linestyle="--", linewidth=0.7, alpha=0.5)
    axes[0].set_title("Validation R2")
    axes[1].set_title("Validation TAC")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("R2")
    axes[1].set_ylabel("TAC")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def load_summaries():
    rows = []
    for path in sorted(glob.glob(SAMPLE_SUMMARY_GLOB)):
        with np.load(path, allow_pickle=False) as d:
            n = len(d["dataset_idx"]) if "dataset_idx" in d else 0
            for i in range(n):
                row = {}
                for k in d.files:
                    val = d[k][i]
                    row[k] = val.item() if hasattr(val, "item") else val
                rows.append(row)
    return rows


def load_selected_maps():
    arrays = []
    for path in sorted(glob.glob(SELECTED_MAPS_GLOB)):
        with np.load(path, allow_pickle=False) as d:
            if "pred" in d:
                arrays.append({k: d[k] for k in d.files})
    if not arrays:
        return {}
    out = {"mask": arrays[0]["mask"]}
    keys = [k for k in arrays[0].keys() if k != "mask"]
    for key in keys:
        out[key] = np.concatenate([a[key] for a in arrays if key in a], axis=0)
    return out


def sample_metrics(pred, truth, mask):
    valid = (mask > 0.5) & np.isfinite(pred) & np.isfinite(truth)
    p = pred[valid].astype(float)
    t = truth[valid].astype(float)
    err = p - t
    mae = float(np.mean(np.abs(err)))
    corr = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-8 and t.std() > 1e-8 else float("nan")
    r2 = float(1.0 - np.sum(err ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-8))
    return mae, corr, r2


def group_name(month):
    if month in (5, 6):
        return "may_jun"
    if month in (7, 8):
        return "jul_aug"
    if month == 9:
        return "sep"
    return "other"


def choose_sample_indices(maps):
    months = maps["months"].astype(int)
    years = maps["years"].astype(int)
    fold_ids = maps["fold_ids"].astype(int)
    mask = maps["mask"][0] if maps["mask"].ndim == 3 else maps["mask"]
    truth = maps["truth"].astype(float)
    heat_score = np.nanmax(np.where(mask[None] > 0.5, truth, np.nan), axis=(1, 2))
    quotas = {"may_jun": 3, "jul_aug": 4, "sep": 3}
    chosen = []

    strong = [i for i in np.argsort(-heat_score) if heat_score[i] > 1.5 and group_name(months[i]) in quotas]
    for i in strong[:2]:
        g = group_name(months[i])
        if quotas[g] > 0 and i not in chosen:
            chosen.append(int(i))
            quotas[g] -= 1

    for group, remaining in list(quotas.items()):
        if remaining <= 0:
            continue
        candidates = [i for i in range(len(months)) if group_name(months[i]) == group and i not in chosen]
        candidates = sorted(candidates, key=lambda i: (years[i], months[i], fold_ids[i]))
        if not candidates:
            continue
        positions = np.linspace(0, len(candidates) - 1, min(remaining, len(candidates)), dtype=int)
        for pos in positions:
            idx = int(candidates[pos])
            if idx not in chosen:
                chosen.append(idx)
        quotas[group] -= len(positions)

    if len(chosen) < 10:
        for i in np.argsort(-heat_score):
            if int(i) not in chosen:
                chosen.append(int(i))
            if len(chosen) >= 10:
                break
    return chosen[:10], heat_score


def save_sample_pngs(maps):
    if not maps:
        return []
    out = OUTPUT_DIR / "samples"
    out.mkdir(parents=True, exist_ok=True)
    ccrs, cfeature = load_cartopy()
    subplot_kw = {"projection": ccrs.PlateCarree()} if ccrs is not None else {}
    mask = maps["mask"][0] if maps["mask"].ndim == 3 else maps["mask"]
    chosen, heat_score = choose_sample_indices(maps)
    outputs = []

    for rank, idx in enumerate(chosen, start=1):
        pred = maps["pred"][idx].astype(float)
        truth = maps["truth"][idx].astype(float)
        if "climo" in maps:
            climo = maps["climo"][idx].astype(float)
            pred_plot = pred - climo
            truth_plot = truth - climo
            map_cmap = "RdBu_r"
            map_vmin, map_vmax = -2.5, 2.5
            map_label = "daily-climo anomaly (z)"
        else:
            pred_plot = pred
            truth_plot = truth
            map_cmap = "RdYlBu_r"
            map_vmin, map_vmax = -3, 3
            map_label = "z-score"
        err = pred_plot - truth_plot
        pred_m = np.where(mask > 0.5, pred_plot, np.nan)
        truth_m = np.where(mask > 0.5, truth_plot, np.nan)
        err_m = np.where(mask > 0.5, err, np.nan)
        mae, corr, r2 = sample_metrics(pred, truth, mask)
        fold = int(maps["fold_ids"][idx])
        month = int(maps["months"][idx])
        year = int(maps["years"][idx])
        target_date = str(maps["target_dates"][idx])
        init_date = str(maps["init_dates"][idx])
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), subplot_kw=subplot_kw, constrained_layout=True)
        im0 = add_map(axes[0], truth_m, "Truth", map_cmap, map_vmin, map_vmax, ccrs, cfeature)
        add_map(axes[1], pred_m, "MeshFlowNet", map_cmap, map_vmin, map_vmax, ccrs, cfeature)
        im2 = add_map(axes[2], err_m, "Prediction - Truth", "RdBu_r", -2, 2, ccrs, cfeature)
        fig.colorbar(im0, ax=axes[:2].ravel().tolist(), shrink=0.82, pad=0.012, label=map_label)
        fig.colorbar(im2, ax=axes[2], shrink=0.82, pad=0.012, label="error")
        fig.suptitle(
            f"Fold {fold} | Init {init_date} -> Target {target_date} | "
            f"MAE={mae:.2f}, r={corr:.2f}, R2={r2:.2f}, max truth z={heat_score[idx]:.2f}",
            fontsize=11,
        )
        path = out / f"sample_{rank:02d}_{target_date}_fold{fold}_m{month:02d}_{year}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs.append((path, idx, fold, init_date, target_date, month, year, heat_score[idx], mae, corr, r2))
    return outputs


def plot_stitched_tac_maps():
    if not STITCHED_TAC_NPZ.exists():
        return None, None
    with np.load(STITCHED_TAC_NPZ, allow_pickle=False) as d:
        mask = d["mask"].astype(bool)
        if "weekly7_model_corr_map" in d:
            model = np.where(mask, d["weekly7_model_corr_map"].astype(float), np.nan)
            persist = np.where(mask, d["weekly7_persistence_corr_map"].astype(float), np.nan)
            model_tac = float(d["weekly7_model_tac"])
            persist_tac = float(d["weekly7_persistence_tac"])
            metric_label = "Weekly7 TAC"
        else:
            model = np.where(mask, d["model_corr_map"].astype(float), np.nan)
            persist = np.where(mask, d["persistence_corr_map"].astype(float), np.nan)
            model_tac = float(d["model_tac"])
            persist_tac = float(d["persistence_tac"])
            metric_label = "Daily TAC"
        n_samples = int(np.asarray(d["n_samples"]).item())
        years = d["years"].astype(int)
    diff = model - persist
    ccrs, cfeature = load_cartopy()
    subplot_kw = {"projection": ccrs.PlateCarree()} if ccrs is not None else {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), subplot_kw=subplot_kw, constrained_layout=True)
    im0 = add_map(axes[0], model, f"Model {metric_label}\nmean={model_tac:.3f}", "RdYlGn", -0.1, 0.4, ccrs, cfeature)
    add_map(axes[1], persist, f"Persistence {metric_label}\nmean={persist_tac:.3f}", "RdYlGn", -0.1, 0.4, ccrs, cfeature)
    im2 = add_map(axes[2], diff, f"Improvement\nmean={model_tac - persist_tac:+.3f}", "RdBu_r", -0.15, 0.15, ccrs, cfeature)
    fig.colorbar(im0, ax=axes[:2].ravel().tolist(), shrink=0.78, pad=0.015, label="TAC")
    fig.colorbar(im2, ax=axes[2], shrink=0.78, pad=0.015, label="TAC difference")
    path = OUTPUT_DIR / "stitched_tac_maps.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path, {
        "model_tac": model_tac,
        "persistence_tac": persist_tac,
        "n_samples": n_samples,
        "year_start": int(years.min()),
        "year_end": int(years.max()),
        "n_years": int(len(np.unique(years))),
    }


def aggregate_monthly_stats():
    monthly = {}
    mask = None
    for path in sorted(glob.glob(MONTHLY_STATS_GLOB)):
        with np.load(path, allow_pickle=False) as d:
            if "months" not in d:
                continue
            mask = d["mask"] if mask is None else mask
            months = d["months"].astype(int)
            for i, month in enumerate(months):
                target = monthly.setdefault(month, None)
                stack = {key: d[f"{key}_by_month"][i].astype(np.float64) for key in STAT_KEYS}
                if target is None:
                    monthly[month] = stack
                else:
                    for key in STAT_KEYS:
                        target[key] += stack[key]
    return monthly, mask


def monthly_breakdown(summary_rows):
    monthly_stats, mask = aggregate_monthly_stats()
    rows = []
    for month in range(5, 10):
        sub = [r for r in summary_rows if int(r.get("month", -1)) == month]
        sample_count = len(sub)
        r2 = float(np.nanmean([float(r.get("r2", np.nan)) for r in sub])) if sub else float("nan")
        mae = float(np.nanmean([float(r.get("mae", np.nan)) for r in sub])) if sub else float("nan")
        tac = float("nan")
        pers_tac = float("nan")
        if month in monthly_stats and mask is not None:
            s = monthly_stats[month]
            _, tac = corr_from_sums(
                s["pred_sum"], s["truth_sum"], s["pred_sq_sum"], s["truth_sq_sum"],
                s["pred_truth_sum"], s["count"], mask
            )
            _, pers_tac = corr_from_sums(
                s["persist_sum"], s["truth_sum"], s["persist_sq_sum"], s["truth_sq_sum"],
                s["persist_truth_sum"], s["count"], mask
            )
        rows.append({
            "month": month,
            "n_samples": sample_count,
            "tac": tac,
            "persistence_tac": pers_tac,
            "mean_sample_r2": r2,
            "mean_sample_mae": mae,
        })
    path = OUTPUT_DIR / "monthly_breakdown.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path, rows


def write_training_curve_text(rows, f):
    f.write("\nTraining curves, epoch by epoch\n")
    f.write("===============================\n")
    if not rows:
        f.write("No training metric CSVs found.\n")
        return
    for fold in sorted({r["fold"] for r in rows if r.get("fold", -1) >= 0}):
        f.write(f"\nFold {fold}\n")
        f.write("epoch,val_r2,val_tac,spatial_anom_r2,mae,crps,mse_skill_vs_persistence\n")
        fr = sorted([r for r in rows if r.get("fold") == fold], key=lambda r: r["epoch"])
        for r in fr:
            f.write(
                f"{r['epoch']},{r.get('val_r2', np.nan):.4f},{r.get('val_tac', np.nan):.4f},"
                f"{r.get('spatial_anom_r2', np.nan):.4f},{r.get('mae', np.nan):.4f},"
                f"{r.get('crps', np.nan):.4f},{r.get('mse_skill_vs_persistence', np.nan):.4f}\n"
            )


def write_summary(stitched_info, best, monthly_rows, sample_outputs, training_rows, paths):
    path = OUTPUT_DIR / "review_summary.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("MeshFlowNet Review Packet\n")
        f.write("=========================\n\n")
        f.write("Generated files\n")
        f.write("---------------\n")
        for label, p in paths.items():
            if p is not None:
                f.write(f"{label}: {p}\n")
        f.write("\n")

        if stitched_info:
            f.write("Stitched TAC summary\n")
            f.write("--------------------\n")
            f.write(f"Samples: {stitched_info['n_samples']}\n")
            f.write(f"Years: {stitched_info['year_start']}-{stitched_info['year_end']} ({stitched_info['n_years']} years)\n")
            f.write(f"Model TAC: {stitched_info['model_tac']:.4f}\n")
            f.write(f"Persistence TAC: {stitched_info['persistence_tac']:.4f}\n")
            f.write(f"TAC improvement: {stitched_info['model_tac'] - stitched_info['persistence_tac']:+.4f}\n\n")

        f.write("Per-fold best epoch metrics, selected by validation TAC\n")
        f.write("------------------------------------------------------\n")
        if best:
            f.write("fold,epoch,val_r2,val_tac,persistence_tac,spatial_anom_r2,mae,crps,mse_skill_vs_persistence\n")
            for r in best:
                f.write(
                    f"{r['fold']},{r['epoch']},{r.get('val_r2', np.nan):.4f},"
                    f"{r.get('val_tac', np.nan):.4f},{r.get('persistence_tac', np.nan):.4f},"
                    f"{r.get('spatial_anom_r2', np.nan):.4f},{r.get('mae', np.nan):.4f},"
                    f"{r.get('crps', np.nan):.4f},{r.get('mse_skill_vs_persistence', np.nan):.4f}\n"
                )
        else:
            f.write("No training metric CSVs found.\n")
        f.write("\n")

        f.write("Per-month breakdown\n")
        f.write("-------------------\n")
        f.write("month,n_samples,tac,persistence_tac,mean_sample_r2,mean_sample_mae\n")
        for r in monthly_rows:
            f.write(
                f"{r['month']},{r['n_samples']},{r['tac']:.4f},{r['persistence_tac']:.4f},"
                f"{r['mean_sample_r2']:.4f},{r['mean_sample_mae']:.4f}\n"
            )
        f.write("\n")

        f.write("Selected sample PNGs\n")
        f.write("--------------------\n")
        for p, _, fold, init_date, target_date, month, year, heat, mae, corr, r2 in sample_outputs:
            f.write(
                f"{p.name}: fold={fold}, init={init_date}, target={target_date}, "
                f"month={month}, year={year}, max_truth_z={heat:.2f}, "
                f"MAE={mae:.2f}, r={corr:.2f}, R2={r2:.2f}\n"
            )
        write_training_curve_text(training_rows, f)
    return path


def main():
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tac_map_path, stitched_info = plot_stitched_tac_maps()
    training_rows = read_training_rows()
    best = best_rows_by_fold(training_rows)
    best_csv = write_best_metrics(best) if best else None
    curves_path = plot_training_curves(training_rows)
    summaries = load_summaries()
    monthly_csv, monthly_rows = monthly_breakdown(summaries)
    maps = load_selected_maps()
    sample_outputs = save_sample_pngs(maps)
    summary_path = write_summary(
        stitched_info,
        best,
        monthly_rows,
        sample_outputs,
        training_rows,
        {
            "summary_text": OUTPUT_DIR / "review_summary.txt",
            "stitched_tac_map": tac_map_path,
            "per_fold_best_csv": best_csv,
            "monthly_breakdown_csv": monthly_csv,
            "training_curves": curves_path,
            "sample_png_dir": OUTPUT_DIR / "samples",
        },
    )

    print(f"Wrote review packet to {OUTPUT_DIR}")
    print(f"Summary: {summary_path}")
    if tac_map_path:
        print(f"Spatial TAC map: {tac_map_path}")
    print(f"Sample PNGs: {OUTPUT_DIR / 'samples'}")


if __name__ == "__main__":
    main()
