#!/usr/bin/env python3
"""Year-block bootstrap significance tests for stitched MeshFlowNet TAC skill."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from publication_analysis_utils import (
    EXTENT,
    STAT_KEYS,
    ensure_dir,
    fdr_bh,
    land_mean,
    model_persistence_corr_maps,
    region_masks,
)

plt = None


def load_plotting():
    global plt
    if plt is not None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    plt = _plt


def progress(iterable, desc):
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc)
    except Exception:
        return iterable


def load_per_year_stats(path, metric="daily"):
    prefix = "weekly7_" if metric == "weekly7" else ""
    with np.load(path, allow_pickle=False) as data:
        years = np.asarray(data["years"], dtype=np.int16)
        mask = np.asarray(data["mask"], dtype=np.uint8)
        stats = {
            key: np.asarray(data[f"{prefix}{key}_by_year"], dtype=np.float64)
            for key in STAT_KEYS
        }
    return years, mask, stats


def sum_years(stats_by_year, indices):
    return {key: np.sum(value[indices], axis=0) for key, value in stats_by_year.items()}


def save_bootstrap_hist(skill_samples, p_value, ci, output_base, metric_label):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(skill_samples, bins=60, color="#2b6cb0", alpha=0.82)
    ax.axvline(0.0, color="black", linewidth=1.0, label="No skill")
    ax.axvspan(ci[0], ci[1], color="#90cdf4", alpha=0.25, label="95% CI")
    ax.axvline(float(np.mean(skill_samples)), color="#c53030", linewidth=1.2, label="Mean")
    ax.set_xlabel(f"CONUS mean {metric_label} skill (model - persistence)")
    ax.set_ylabel("Bootstrap count")
    ax.set_title(f"Year-block bootstrap {metric_label} skill distribution (p={p_value:.4g})")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{output_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_significance_map(sig_map, skill_map, output_base, metric_label):
    display = np.full(sig_map.shape, np.nan, dtype=np.float32)
    display[sig_map] = skill_map[sig_map]
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    im = ax.imshow(
        np.ma.masked_invalid(display),
        extent=EXTENT,
        origin="upper",
        cmap="RdBu_r",
        vmin=-0.15,
        vmax=0.15,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"FDR-significant {metric_label} skill (p < 0.05)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label(f"{metric_label} skill")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{output_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="hindcast_stats/per_year_tac_stats.npz")
    parser.add_argument("--output_dir", default="paper_figures")
    parser.add_argument("--output", default="bootstrap_results.npz")
    parser.add_argument("--metric", choices=["weekly7", "daily"], default="weekly7")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--rng_seed", type=int, default=42)
    parser.add_argument(
        "--pixel_stride",
        type=int,
        default=4,
        help="Compute per-pixel p-values on this grid stride; 1 means all pixels.",
    )
    args = parser.parse_args()

    load_plotting()
    metric_label = "weekly7 TAC" if args.metric == "weekly7" else "daily TAC"
    years, mask, stats_by_year = load_per_year_stats(args.input, metric=args.metric)
    n_years = len(years)
    rng = np.random.default_rng(args.rng_seed)

    aggregate = {key: np.sum(value, axis=0) for key, value in stats_by_year.items()}
    model_map, persist_map = model_persistence_corr_maps(aggregate, mask)
    skill_map = model_map - persist_map

    model_samples = np.empty(args.n_bootstrap, dtype=np.float32)
    persist_samples = np.empty(args.n_bootstrap, dtype=np.float32)
    region_defs = region_masks(mask.shape)
    region_model_samples = {
        name: np.empty(args.n_bootstrap, dtype=np.float32) for name in region_defs
    }
    region_persist_samples = {
        name: np.empty(args.n_bootstrap, dtype=np.float32) for name in region_defs
    }

    stride = max(1, int(args.pixel_stride))
    row_grid, col_grid = np.indices(mask.shape)
    sampled_pixels = (mask > 0.5) & (row_grid % stride == 0) & (col_grid % stride == 0)
    pixel_le_zero = np.zeros(mask.shape, dtype=np.int32)

    choices = rng.integers(0, n_years, size=(args.n_bootstrap, n_years), endpoint=False)
    for b in progress(range(args.n_bootstrap), desc="Bootstrap"):
        boot_stats = sum_years(stats_by_year, choices[b])
        boot_model, boot_persist = model_persistence_corr_maps(boot_stats, mask)
        boot_skill = boot_model - boot_persist
        model_samples[b] = land_mean(boot_model, mask)
        persist_samples[b] = land_mean(boot_persist, mask)
        for name, region_mask in region_defs.items():
            combined_mask = (mask > 0.5) & region_mask
            region_model_samples[name][b] = land_mean(boot_model, combined_mask.astype(np.uint8))
            region_persist_samples[name][b] = land_mean(boot_persist, combined_mask.astype(np.uint8))
        pixel_le_zero[sampled_pixels] += (boot_skill[sampled_pixels] <= 0.0)

    skill_samples = model_samples - persist_samples
    model_ci = np.percentile(model_samples, [2.5, 97.5])
    persist_ci = np.percentile(persist_samples, [2.5, 97.5])
    skill_ci = np.percentile(skill_samples, [2.5, 97.5])
    p_value = float(np.mean(skill_samples <= 0.0))
    cohens_d = float(np.mean(skill_samples) / (np.std(skill_samples) + 1e-8))
    region_names = np.array(list(region_defs.keys()))
    region_skill_mean = []
    region_skill_ci = []
    region_skill_p = []
    for name in region_names:
        samples = region_model_samples[str(name)] - region_persist_samples[str(name)]
        region_skill_mean.append(float(np.mean(samples)))
        region_skill_ci.append(np.percentile(samples, [2.5, 97.5]).astype(np.float32))
        region_skill_p.append(float(np.mean(samples <= 0.0)))
    region_skill_mean = np.asarray(region_skill_mean, dtype=np.float32)
    region_skill_ci = np.asarray(region_skill_ci, dtype=np.float32)
    region_skill_p = np.asarray(region_skill_p, dtype=np.float32)

    pixel_p = np.full(mask.shape, np.nan, dtype=np.float32)
    pixel_p[sampled_pixels] = pixel_le_zero[sampled_pixels] / float(args.n_bootstrap)
    reject_flat, corrected_flat = fdr_bh(pixel_p[sampled_pixels], alpha=0.05)
    pixel_sig = np.zeros(mask.shape, dtype=bool)
    pixel_q = np.full(mask.shape, np.nan, dtype=np.float32)
    pixel_sig[sampled_pixels] = reject_flat & (skill_map[sampled_pixels] > 0)
    pixel_q[sampled_pixels] = corrected_flat.astype(np.float32)

    out_dir = ensure_dir(args.output_dir)
    name_prefix = "weekly7" if args.metric == "weekly7" else "daily"
    save_bootstrap_hist(
        skill_samples, p_value, skill_ci, out_dir / f"{name_prefix}_bootstrap_skill_distribution",
        metric_label,
    )
    save_significance_map(
        pixel_sig, skill_map, out_dir / f"{name_prefix}_bootstrap_significance_map",
        metric_label,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = out_dir / output_path
    np.savez_compressed(
        output_path,
        years=years,
        model_tac_samples=model_samples,
        persist_tac_samples=persist_samples,
        skill_samples=skill_samples,
        model_ci_95=model_ci.astype(np.float32),
        persist_ci_95=persist_ci.astype(np.float32),
        skill_ci_95=skill_ci.astype(np.float32),
        p_value=np.array(p_value, dtype=np.float32),
        cohens_d=np.array(cohens_d, dtype=np.float32),
        metric=np.array(args.metric),
        region_names=region_names,
        region_skill_mean=region_skill_mean,
        region_skill_ci_95=region_skill_ci,
        region_skill_p_value=region_skill_p,
        pixel_p_values=pixel_p,
        pixel_q_values=pixel_q,
        pixel_significance_map=pixel_sig.astype(np.uint8),
        model_tac_map=model_map.astype(np.float32),
        persistence_tac_map=persist_map.astype(np.float32),
        skill_map=skill_map.astype(np.float32),
        mask=mask.astype(np.uint8),
        pixel_stride=np.array(stride, dtype=np.int16),
    )

    print("Bootstrap significance")
    print("======================")
    print(f"Metric: {metric_label}")
    print(f"Years resampled: {years[0]}-{years[-1]} ({len(years)})")
    print(f"Bootstrap samples: {args.n_bootstrap}")
    print(f"Model 95% CI:           [{model_ci[0]:.4f}, {model_ci[1]:.4f}]")
    print(f"Persistence 95% CI:     [{persist_ci[0]:.4f}, {persist_ci[1]:.4f}]")
    print(f"Skill 95% CI:           [{skill_ci[0]:+.4f}, {skill_ci[1]:+.4f}]")
    print(f"Skill p-value:          {p_value:.6f}")
    print(f"Effect size:            {cohens_d:.3f}")
    print(f"FDR significant sampled pixels: {int(pixel_sig.sum())}")
    print("\nRegional skill bootstrap")
    print("------------------------")
    for i, name in enumerate(region_names):
        lo, hi = region_skill_ci[i]
        print(
            f"{str(name):14s} mean={region_skill_mean[i]:+7.4f} "
            f"CI=[{lo:+.4f}, {hi:+.4f}] p={region_skill_p[i]:.4f}"
        )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
