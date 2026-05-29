#!/usr/bin/env python3
"""Create stitched TAC skill maps and regional summaries for MeshFlowNet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from publication_analysis_utils import (
    EXTENT,
    NOAA_REGION_BOXES,
    STAT_KEYS,
    default_fold_stat_paths,
    ensure_dir,
    land_mean,
    model_persistence_corr_maps,
    region_masks,
    stitch_stats,
)

plt = None
Rectangle = None
TwoSlopeNorm = None


def load_plotting():
    global plt, Rectangle, TwoSlopeNorm
    if plt is not None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from matplotlib.colors import TwoSlopeNorm as _TwoSlopeNorm
    from matplotlib.patches import Rectangle as _Rectangle

    plt = _plt
    Rectangle = _Rectangle
    TwoSlopeNorm = _TwoSlopeNorm


def load_cartopy(disabled: bool = False):
    if disabled:
        return None, None
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        return ccrs, cfeature
    except Exception:
        return None, None


def stitch_weekly7_stats(paths):
    aggregate = None
    mask = None
    prefix_used = None
    years = set()
    total_samples = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            prefix = None
            if all(f"weekly7_{key}" in data for key in STAT_KEYS):
                prefix = "weekly7_"
            elif all(f"weekly_truth7_{key}" in data for key in STAT_KEYS):
                prefix = "weekly_truth7_"
            if prefix is None:
                continue
            if prefix_used is None:
                prefix_used = prefix
            elif prefix != prefix_used:
                raise ValueError(
                    f"Mixed weekly statistic types are not supported: {prefix_used!r} and {prefix!r}"
                )
            if aggregate is None:
                aggregate = {key: np.asarray(data[f"{prefix}{key}"], dtype=np.float64).copy() for key in STAT_KEYS}
                mask = np.asarray(data["mask"], dtype=np.uint8)
            else:
                for key in STAT_KEYS:
                    aggregate[key] += np.asarray(data[f"{prefix}{key}"], dtype=np.float64)
            sample_key = "weekly7_n_samples" if prefix == "weekly7_" else "weekly_truth7_n_samples"
            if sample_key in data:
                total_samples += int(np.asarray(data[sample_key]).item())
            if "years" in data:
                years.update(int(y) for y in np.atleast_1d(data["years"]).astype(int).tolist())
    if aggregate is None or mask is None:
        raise FileNotFoundError("No weekly7_* statistics found in input files.")
    return aggregate, mask, sorted(years), total_samples


def _new_map_axis(fig, ccrs):
    if ccrs is None:
        return fig.add_subplot(1, 1, 1), None
    proj = ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.0)
    return fig.add_subplot(1, 1, 1, projection=proj), ccrs.PlateCarree()


def decorate_map(ax, ccrs, cfeature):
    if ccrs is not None:
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        try:
            ax.coastlines(linewidth=0.6)
            ax.add_feature(cfeature.BORDERS, linewidth=0.4)
            ax.add_feature(cfeature.STATES, linewidth=0.25)
        except Exception:
            pass
    else:
        ax.set_xlim(EXTENT[0], EXTENT[1])
        ax.set_ylim(EXTENT[2], EXTENT[3])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")


def save_map(data, title, output_base, cmap, vmin, vmax, ccrs, cfeature,
             cbar_label="TAC", center=None, skill_positive=None):
    fig = plt.figure(figsize=(9.2, 6.0))
    ax, transform = _new_map_axis(fig, ccrs)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax) if center is not None else None
    kwargs = dict(
        extent=EXTENT,
        origin="upper",
        cmap=cmap,
        interpolation="nearest",
    )
    if norm is None:
        kwargs.update(vmin=vmin, vmax=vmax)
    else:
        kwargs.update(norm=norm)
    if transform is not None:
        kwargs["transform"] = transform
    im = ax.imshow(np.ma.masked_invalid(data), **kwargs)
    decorate_map(ax, ccrs, cfeature)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Approximate positive PNA loading region.
    rect_kwargs = dict(fill=False, edgecolor="black", linewidth=1.0, linestyle="--")
    if transform is not None:
        rect_kwargs["transform"] = transform
    ax.add_patch(Rectangle((-130.0, 35.0), 30.0, 15.0, **rect_kwargs))
    ax.text(
        -129.0,
        49.0,
        "PNA+ loading",
        fontsize=8,
        color="black",
        transform=transform if transform is not None else ax.transData,
    )

    if skill_positive is not None:
        hatch_data = np.where(skill_positive, 1.0, np.nan)
        contour_kwargs = dict(
            extent=EXTENT,
            origin="upper",
            levels=[0.5, 1.5],
            colors="none",
            hatches=["///"],
        )
        if transform is not None:
            contour_kwargs["transform"] = transform
        ax.contourf(hatch_data, **contour_kwargs)

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{output_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def regional_summary(model, persistence, skill, mask):
    rows = []
    for name, box_mask in region_masks(mask.shape).items():
        valid = box_mask & (mask > 0.5)
        rows.append({
            "region": name,
            "model_tac": land_mean(np.where(valid, model, np.nan), valid),
            "persistence_tac": land_mean(np.where(valid, persistence, np.nan), valid),
            "skill": land_mean(np.where(valid, skill, np.nan), valid),
        })
    return rows


def save_regional_bar(rows, output_base):
    names = [r["region"] for r in rows]
    model = [r["model_tac"] for r in rows]
    persist = [r["persistence_tac"] for r in rows]
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(x - 0.18, model, width=0.36, label="MeshFlowNet", color="#2b6cb0")
    ax.bar(x + 0.18, persist, width=0.36, label="Persistence", color="#718096")
    ax.axhline(0, color="0.2", linewidth=0.8)
    ax.set_ylabel("TAC")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_title("Regional Skill Summary")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{output_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_caption(rows, output_path):
    sorted_rows = sorted(rows, key=lambda r: r["skill"])
    low = sorted_rows[0]
    high = sorted_rows[-1]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Physical interpretation notes\n")
        f.write("=============================\n")
        f.write(
            "The dashed box marks an approximate positive PNA loading region. "
            "Skill concentrated in the western and northwestern CONUS is consistent "
            "with large-scale Pacific teleconnection control of warm-season flow regimes. "
            "Weak or negative skill in humid eastern regions is consistent with stronger "
            "synoptic and land-atmosphere noise at day-15 lead.\n\n"
        )
        f.write(
            f"Highest regional mean skill: {high['region']} "
            f"({high['skill']:+.3f}).\n"
        )
        f.write(
            f"Lowest regional mean skill: {low['region']} "
            f"({low['skill']:+.3f}).\n"
        )
        f.write("\nRegion boxes used:\n")
        for name, box in NOAA_REGION_BOXES.items():
            f.write(f"- {name}: lat {box[0]}-{box[1]}, lon {box[2]}-{box[3]}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        help="Fold hindcast stat files or globs. Defaults to hindcast_stats/hindcast_tac_stats_cvfold*_test.npz.",
    )
    parser.add_argument("--output_dir", default="paper_figures", help="Output directory.")
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--metric", choices=["weekly7", "daily"], default="weekly7")
    parser.add_argument("--no_cartopy", action="store_true", help="Use plain matplotlib lat/lon axes.")
    args = parser.parse_args()

    paths = args.files or default_fold_stat_paths(".")
    if args.metric == "weekly7":
        stats, mask, years, total_samples = stitch_weekly7_stats(paths)
        is_tube = False
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                if "multi_lead_tube" in data and int(np.asarray(data["multi_lead_tube"]).item()) == 1:
                    is_tube = True
                    break
        metric_label = "Tube Same-Init 7-Day Mean TAC" if is_tube else "True 7-Day Mean TAC"
        output_prefix = "weekly7_tac"
    else:
        stats, mask, years, total_samples = stitch_stats(paths)
        metric_label = "Daily TAC"
        output_prefix = "tac"
    load_plotting()
    model, persistence = model_persistence_corr_maps(stats, mask, min_count=args.min_count)
    model = np.where(mask > 0.5, model, np.nan)
    persistence = np.where(mask > 0.5, persistence, np.nan)
    skill = model - persistence
    ratio = model / (persistence + 1e-8)

    out = ensure_dir(args.output_dir)
    ccrs, cfeature = load_cartopy(args.no_cartopy)

    save_map(
        model,
        f"MeshFlowNet 15-Day {metric_label} (Stitched 5-Fold CV, 1981-2023)",
        out / f"{output_prefix}_model_map",
        "RdYlBu_r",
        -0.1,
        0.3,
        ccrs,
        cfeature,
    )
    save_map(
        persistence,
        f"Persistence Baseline {metric_label}",
        out / f"{output_prefix}_persistence_map",
        "RdYlBu_r",
        -0.1,
        0.3,
        ccrs,
        cfeature,
    )
    save_map(
        skill,
        f"{metric_label} Skill (MeshFlowNet minus Persistence)",
        out / f"{output_prefix}_skill_map",
        "RdBu_r",
        -0.15,
        0.15,
        ccrs,
        cfeature,
        cbar_label="TAC difference",
        center=0.0,
        skill_positive=(skill > 0) & (mask > 0.5),
    )

    rows = regional_summary(model, persistence, skill, mask)
    save_regional_bar(rows, out / "regional_skill_summary")
    write_caption(rows, out / "tac_skill_physical_interpretation.txt")

    np.savez_compressed(
        out / "tac_skill_maps_data.npz",
        tac_model=model.astype(np.float32),
        tac_persistence=persistence.astype(np.float32),
        tac_skill=skill.astype(np.float32),
        tac_ratio=ratio.astype(np.float32),
        mask=mask.astype(np.uint8),
        years=np.array(years, dtype=np.int16),
        n_samples=np.array(total_samples, dtype=np.int32),
        region_names=np.array([r["region"] for r in rows]),
        region_model_tac=np.array([r["model_tac"] for r in rows], dtype=np.float32),
        region_persistence_tac=np.array([r["persistence_tac"] for r in rows], dtype=np.float32),
        region_skill=np.array([r["skill"] for r in rows], dtype=np.float32),
    )

    print(f"{metric_label} skill map summary")
    print("=" * (len(metric_label) + 18))
    print(f"Files: {len(paths)}")
    print(f"Samples: {total_samples}")
    if years:
        print(f"Years: {years[0]}-{years[-1]} ({len(years)})")
    print(f"CONUS model TAC:       {land_mean(model, mask):.4f}")
    print(f"CONUS persistence TAC: {land_mean(persistence, mask):.4f}")
    print(f"CONUS TAC skill:       {land_mean(skill, mask):+.4f}")
    print(f"Saved figures/data to: {out}")


if __name__ == "__main__":
    main()
