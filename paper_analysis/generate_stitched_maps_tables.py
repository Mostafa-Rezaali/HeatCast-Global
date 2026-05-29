#!/usr/bin/env python3
"""
Generate stitched-stat outputs:
  - Figure 3 spatial TAC maps
  - Figure 6 regional skill map/profile summary
  - Table 4 regional skill
  - significance_tests.txt

Run after stitched_5fold_tac.npz exists.
"""

from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt


STITCHED_TAC_NPZ = Path("hindcast_stats/stitched_5fold_tac.npz")
OUTPUT_ROOT = Path("paper_figures")
EXTENT = [-130.0, -60.0, 25.0, 50.0]
LAT_RANGE = (25.0, 50.0)
LON_RANGE = (-130.0, -60.0)
MIN_COUNT = 2


def load_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except Exception:
        return None, None


def normal_p_two_sided(z):
    z = np.asarray(z, dtype=np.float64)
    erfc_vec = np.vectorize(math.erfc)
    return erfc_vec(np.abs(z) / math.sqrt(2.0))


def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))


def corr_from_sums(x_sum, y_sum, x_sq_sum, y_sq_sum, xy_sum, count, mask):
    valid = (mask > 0.5) & (count >= MIN_COUNT)
    safe_count = np.maximum(count, 1.0)
    cov = xy_sum - (x_sum * y_sum) / safe_count
    x_var = x_sq_sum - (x_sum * x_sum) / safe_count
    y_var = y_sq_sum - (y_sum * y_sum) / safe_count
    denom = np.sqrt(np.maximum(x_var, 0.0) * np.maximum(y_var, 0.0))
    valid &= denom > 1e-12
    corr = np.full(count.shape, np.nan, dtype=np.float32)
    corr[valid] = (cov[valid] / denom[valid]).astype(np.float32)
    return corr


def map_panel(ax, data, title, cmap, vmin, vmax, ccrs=None, cfeature=None):
    data = np.ma.masked_invalid(data)
    if ccrs is not None:
        im = ax.imshow(data, extent=EXTENT, origin="upper", transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.add_feature(cfeature.STATES, linewidth=0.25)
    else:
        im = ax.imshow(data, extent=EXTENT, origin="upper", cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
        ax.set_xlim(EXTENT[0], EXTENT[1])
        ax.set_ylim(EXTENT[2], EXTENT[3])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    ax.set_title(title, fontsize=11, fontweight="bold")
    return im


def region_masks(shape):
    h, w = shape
    lat = np.linspace(LAT_RANGE[1], LAT_RANGE[0], h)[:, None]
    lon = np.linspace(LON_RANGE[0], LON_RANGE[1], w)[None, :]
    return {
        "Pacific NW": (lon < -110) & (lat >= 42),
        "Southwest": (lon < -105) & (lat < 42),
        "Great Plains N": (lon >= -110) & (lon < -95) & (lat >= 40),
        "Great Plains S": (lon >= -110) & (lon < -95) & (lat < 40),
        "Midwest": (lon >= -95) & (lon < -82) & (lat >= 37),
        "Southeast": (lon >= -95) & (lon < -80) & (lat < 37),
        "Northeast": (lon >= -82) & (lat >= 37),
        "Gulf Coast": (lon >= -100) & (lon < -80) & (lat < 32),
    }


def write_table4(rows):
    out = OUTPUT_ROOT / "tables"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "table4_regional_skill.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrr}\\hline\n")
        f.write("Region & Model TAC & Persistence TAC & Improvement \\\\\\hline\n")
        for name, mt, pt, diff in rows:
            f.write(f"{name} & {mt:.3f} & {pt:.3f} & {diff:+.3f} \\\\\n")
        f.write("\\hline\\end{tabular}\n")


def main():
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    (OUTPUT_ROOT / "main").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "stats").mkdir(parents=True, exist_ok=True)

    with np.load(STITCHED_TAC_NPZ) as data:
        mask = data["mask"].astype(bool)
        count = data["count"].astype(np.float64)
        model = data["model_corr_map"].astype(np.float32)
        persist = data["persistence_corr_map"].astype(np.float32)
        model_tac = float(data["model_tac"])
        persist_tac = float(data["persistence_tac"])

    model = np.where(mask, model, np.nan)
    persist = np.where(mask, persist, np.nan)
    diff = model - persist

    z_model = fisher_z(model)
    z_persist = fisher_z(persist)
    se = 1.0 / np.sqrt(np.maximum(count - 3.0, 1.0))
    p_model = normal_p_two_sided(z_model / se)
    p_diff = normal_p_two_sided((z_model - z_persist) / np.sqrt(2.0 * se ** 2))
    sig_improve = (p_diff < 0.05) & (diff > 0) & mask

    ccrs, cfeature = load_cartopy()
    subplot_kw = {"projection": ccrs.PlateCarree()} if ccrs is not None else {}
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.2),
        subplot_kw=subplot_kw,
        constrained_layout=True,
    )
    im0 = map_panel(axes[0], model, f"A. MeshFlowNet TAC\nCONUS mean={model_tac:.3f}",
                    "RdYlGn", -0.1, 0.4, ccrs, cfeature)
    map_panel(axes[1], persist, f"B. Persistence TAC\nCONUS mean={persist_tac:.3f}",
              "RdYlGn", -0.1, 0.4, ccrs, cfeature)
    im2 = map_panel(axes[2], diff, f"C. TAC Improvement\nmean={model_tac - persist_tac:+.3f}",
                    "RdBu_r", -0.15, 0.15, ccrs, cfeature)

    yy, xx = np.where(sig_improve[::18, ::18])
    if yy.size:
        lons = np.linspace(LON_RANGE[0], LON_RANGE[1], mask.shape[1])[::18][xx]
        lats = np.linspace(LAT_RANGE[1], LAT_RANGE[0], mask.shape[0])[::18][yy]
        if ccrs is not None:
            axes[2].scatter(lons, lats, s=1.2, c="k", transform=ccrs.PlateCarree(), alpha=0.45)
        else:
            axes[2].scatter(lons, lats, s=1.2, c="k", alpha=0.45)

    fig.colorbar(im0, ax=axes[:2].ravel().tolist(), shrink=0.78, pad=0.015, label="TAC")
    fig.colorbar(im2, ax=axes[2], shrink=0.78, pad=0.015, label="TAC difference")
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_ROOT / "main" / f"fig03_tac_spatial_map.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for name, region in region_masks(mask.shape).items():
        valid = region & mask & np.isfinite(model)
        rows.append((name, float(np.nanmean(model[valid])),
                     float(np.nanmean(persist[valid])),
                     float(np.nanmean(diff[valid]))))
    write_table4(rows)

    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    pvals = [r[2] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.18, vals, width=0.36, label="MeshFlowNet")
    ax.bar(x + 0.18, pvals, width=0.36, label="Persistence")
    ax.axhline(0, color="0.2", linewidth=0.8)
    ax.set_ylabel("TAC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.legend(frameon=False)
    ax.set_title("Regional Temporal Anomaly Correlation")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_ROOT / "main" / f"fig06_regional_skill.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    valid_land = mask & np.isfinite(model)
    with open(OUTPUT_ROOT / "stats" / "significance_tests.txt", "w", encoding="utf-8") as f:
        f.write("TAC significance tests\n")
        f.write("======================\n")
        f.write(f"Land pixels tested: {int(valid_land.sum())}\n")
        f.write(f"Model TAC p<0.05 fraction: {float(np.mean((p_model < 0.05)[valid_land])):.4f}\n")
        f.write(f"Model TAC p<0.01 fraction: {float(np.mean((p_model < 0.01)[valid_land])):.4f}\n")
        f.write(f"Model beats persistence p<0.05 fraction: {float(np.mean(sig_improve[valid_land])):.4f}\n")
        f.write(f"CONUS mean model TAC: {model_tac:.4f}\n")
        f.write(f"CONUS mean persistence TAC: {persist_tac:.4f}\n")


if __name__ == "__main__":
    main()
