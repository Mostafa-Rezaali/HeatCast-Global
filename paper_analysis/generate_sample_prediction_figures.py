#!/usr/bin/env python3
"""
Generate sample, seasonal, scatter, and error-distribution figures from
hindcast_paper_data files written by cfm_mesh_train.py --mode export_hindcast.
"""

from pathlib import Path
import glob
import math

import numpy as np
import matplotlib.pyplot as plt


PAPER_DATA_DIR = Path("hindcast_paper_data")
OUTPUT_ROOT = Path("paper_figures")
EXTENT = [-130.0, -60.0, 25.0, 50.0]


def load_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except Exception:
        return None, None


def load_summaries():
    rows = []
    for path in sorted(glob.glob(str(PAPER_DATA_DIR / "hindcast_sample_summary_*_test.npz"))):
        with np.load(path) as d:
            n = len(d["dataset_idx"])
            for i in range(n):
                rows.append({k: d[k][i] for k in d.files})
    return rows


def load_selected_maps():
    arrays = []
    for path in sorted(glob.glob(str(PAPER_DATA_DIR / "hindcast_selected_maps_*_test.npz"))):
        with np.load(path) as d:
            if "pred" in d:
                arrays.append({k: d[k] for k in d.files})
    if not arrays:
        raise FileNotFoundError(f"No selected map files found in {PAPER_DATA_DIR}")
    out = {"mask": arrays[0]["mask"]}
    for key in arrays[0]:
        if key == "mask":
            continue
        out[key] = np.concatenate([a[key] for a in arrays if key in a], axis=0)
    return out


def masked_metrics(pred, truth, mask):
    valid = mask > 0.5
    p = pred[valid].astype(float)
    t = truth[valid].astype(float)
    err = p - t
    r2 = 1.0 - np.sum(err ** 2) / (np.sum((t - t.mean()) ** 2) + 1e-8)
    corr = np.corrcoef(p, t)[0, 1] if p.std() > 1e-8 and t.std() > 1e-8 else np.nan
    return float(np.mean(np.abs(err))), float(corr), float(r2)


def add_map(ax, data, title, cmap, vmin, vmax, ccrs=None, cfeature=None):
    data = np.ma.masked_invalid(data)
    if ccrs is not None:
        im = ax.imshow(data, extent=EXTENT, origin="upper", transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.4)
        ax.add_feature(cfeature.STATES, linewidth=0.2)
    else:
        im = ax.imshow(data, extent=EXTENT, origin="upper", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
    ax.set_title(title, fontsize=9)
    return im


def pick_main_examples(maps, n=4):
    reasons = maps.get("reasons", np.array([""] * len(maps["pred"]))).astype(str)
    heat = np.where(reasons == "heat_area_top")[0]
    months = maps.get("months", np.zeros(len(maps["pred"]), dtype=int))
    picks = []
    for month_group in [(5, 6), (7, 8), (9,), None]:
        if month_group is None:
            candidates = heat
        else:
            candidates = np.where(np.isin(months, month_group))[0]
        for c in candidates:
            if c not in picks:
                picks.append(int(c))
                break
    if len(picks) < n:
        for i in range(len(maps["pred"])):
            if i not in picks:
                picks.append(i)
            if len(picks) >= n:
                break
    return picks[:n]


def figure_samples(maps):
    out = OUTPUT_ROOT / "main"
    out.mkdir(parents=True, exist_ok=True)
    ccrs, cfeature = load_cartopy()
    subplot_kw = {"projection": ccrs.PlateCarree()} if ccrs is not None else {}
    idxs = pick_main_examples(maps, 4)
    mask = maps["mask"][0] if maps["mask"].ndim == 3 else maps["mask"]

    fig, axes = plt.subplots(
        len(idxs),
        3,
        figsize=(13.5, 3.0 * len(idxs)),
        subplot_kw=subplot_kw,
        constrained_layout=True,
    )
    for row, idx in enumerate(idxs):
        pred = maps["pred"][idx].astype(float)
        truth = maps["truth"][idx].astype(float)
        if "climo" in maps:
            climo = maps["climo"][idx].astype(float)
            pred_plot = pred - climo
            truth_plot = truth - climo
            map_cmap = "RdBu_r"
            map_vmin, map_vmax = -2.5, 2.5
        else:
            pred_plot = pred
            truth_plot = truth
            map_cmap = "RdYlBu_r"
            map_vmin, map_vmax = -3, 3
        err = pred_plot - truth_plot
        pred = np.where(mask > 0.5, pred, np.nan)
        truth = np.where(mask > 0.5, truth, np.nan)
        pred_plot = np.where(mask > 0.5, pred_plot, np.nan)
        truth_plot = np.where(mask > 0.5, truth_plot, np.nan)
        err = np.where(mask > 0.5, err, np.nan)
        mae, corr, r2 = masked_metrics(pred, truth, mask)
        title = f"{maps['init_dates'][idx]} to {maps['target_dates'][idx]} | MAE={mae:.2f}, r={corr:.2f}, R2={r2:.2f}"
        add_map(axes[row, 0], truth_plot, "Truth", map_cmap, map_vmin, map_vmax, ccrs, cfeature)
        add_map(axes[row, 1], pred_plot, title, map_cmap, map_vmin, map_vmax, ccrs, cfeature)
        im = add_map(axes[row, 2], err, "Prediction - Truth", "RdBu_r", -2, 2, ccrs, cfeature)
    fig.colorbar(im, ax=axes[:, 2].ravel().tolist(), shrink=0.72, pad=0.015, label="Prediction - truth (z)")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig02_sample_predictions.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_scatter(rows):
    out = OUTPUT_ROOT / "main"
    truth = np.array([float(r["truth_mean"]) for r in rows])
    pred = np.array([float(r["pred_mean"]) for r in rows])
    persist = np.array([float(r["persist_mean"]) for r in rows])
    month = np.array([int(r["month"]) for r in rows])
    fig = plt.figure(figsize=(14, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.045])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    for ax, x, title in [
        (axes[0], pred, "MeshFlowNet"),
        (axes[1], persist, "Persistence"),
    ]:
        sc = ax.scatter(truth, x, c=month, s=8, cmap="viridis", alpha=0.5)
        lo = min(np.nanmin(truth), np.nanmin(x))
        hi = max(np.nanmax(truth), np.nanmax(x))
        ax.plot([lo, hi], [lo, hi], color="k", linewidth=0.8)
        corr = np.corrcoef(truth, x)[0, 1]
        r2 = 1.0 - np.sum((truth - x) ** 2) / (np.sum((truth - truth.mean()) ** 2) + 1e-8)
        ax.set_title(f"{title}: r={corr:.2f}, R2={r2:.2f}")
        ax.set_xlabel("Observed CONUS mean z")
        ax.set_ylabel("Predicted CONUS mean z")
    axes[2].scatter(truth, pred - truth, c=month, s=8, cmap="viridis", alpha=0.5)
    axes[2].axhline(0, color="k", linewidth=0.8)
    axes[2].set_xlabel("Observed CONUS mean z")
    axes[2].set_ylabel("Residual")
    fig.colorbar(sc, cax=cax, label="Month")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig09_scatter_pred_vs_obs.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def figure_error_distribution(maps):
    out = OUTPUT_ROOT / "main"
    mask = maps["mask"][0] if maps["mask"].ndim == 3 else maps["mask"]
    pred = maps["pred"].astype(float)
    truth = maps["truth"].astype(float)
    err = pred - truth
    valid_err = err[:, mask > 0.5].ravel()
    valid_err = valid_err[np.isfinite(valid_err)]
    bias = np.nanmean(np.where(mask[None] > 0.5, err, np.nan), axis=0)
    rmse = np.sqrt(np.nanmean(np.where(mask[None] > 0.5, err ** 2, np.nan), axis=0))

    ccrs, cfeature = load_cartopy()
    subplot_kw = {"projection": ccrs.PlateCarree()} if ccrs is not None else {}
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(valid_err, bins=100, density=True, color="0.35", alpha=0.8)
    mu, sig = float(np.mean(valid_err)), float(np.std(valid_err))
    xs = np.linspace(mu - 4 * sig, mu + 4 * sig, 300)
    ax1.plot(xs, np.exp(-0.5 * ((xs - mu) / sig) ** 2) / (sig * math.sqrt(2 * math.pi)), color="crimson")
    ax1.set_title(f"Error distribution: mean={mu:.3f}, std={sig:.3f}")
    ax1.set_xlabel("Prediction - truth")
    ax1.set_ylabel("Density")
    ax2 = fig.add_subplot(2, 2, 2)
    q = np.linspace(0.01, 0.99, 99)
    emp = np.quantile((valid_err - mu) / (sig + 1e-8), q)
    theo = np.array([math.sqrt(2) * _erfinv(2 * x - 1) for x in q])
    ax2.scatter(theo, emp, s=8)
    ax2.plot([theo.min(), theo.max()], [theo.min(), theo.max()], color="k", linewidth=0.8)
    ax2.set_title("Q-Q plot")
    ax2.set_xlabel("Normal quantile")
    ax2.set_ylabel("Empirical quantile")
    ax3 = fig.add_subplot(2, 2, 3, **subplot_kw)
    add_map(ax3, np.where(mask > 0.5, bias, np.nan), "Mean bias", "RdBu_r", -1, 1, ccrs, cfeature)
    ax4 = fig.add_subplot(2, 2, 4, **subplot_kw)
    im = add_map(ax4, np.where(mask > 0.5, rmse, np.nan), "RMSE", "magma", 0, 2, ccrs, cfeature)
    fig.colorbar(im, ax=[ax3, ax4], shrink=0.78, pad=0.02)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig10_error_distribution.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _erfinv(x):
    # Winitzki approximation, sufficient for Q-Q plotting without scipy.
    a = 0.147
    s = 1 if x >= 0 else -1
    ln = math.log(1 - x * x)
    first = 2 / (math.pi * a) + ln / 2
    return s * math.sqrt(math.sqrt(first * first - ln / a) - first)


def figure_seasonal(rows):
    out = OUTPUT_ROOT / "main"
    months = np.array([int(r["month"]) for r in rows])
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["May", "Jun", "Jul", "Aug", "Sep"]
    xs = np.arange(5)
    mae = []
    r2 = []
    for m in range(5, 10):
        sub = [r for r in rows if int(r["month"]) == m]
        mae.append(np.nanmean([float(r["mae"]) for r in sub]))
        r2.append(np.nanmean([float(r["r2"]) for r in sub]))
    ax.plot(xs, r2, marker="o", label="R2")
    ax2 = ax.twinx()
    ax2.plot(xs, mae, marker="s", color="crimson", label="MAE")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("R2")
    ax2.set_ylabel("MAE")
    ax.set_title("Seasonal skill decomposition from per-sample summaries")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig07_seasonal_skill.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    rows = load_summaries()
    maps = load_selected_maps()
    figure_samples(maps)
    figure_scatter(rows)
    figure_error_distribution(maps)
    figure_seasonal(rows)


if __name__ == "__main__":
    main()
