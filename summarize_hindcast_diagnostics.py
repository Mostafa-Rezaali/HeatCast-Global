#!/usr/bin/env python3
"""Summarize existing MeshFlowNet hindcast diagnostics without rerunning inference."""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np

from publication_analysis_utils import (
    STAT_KEYS,
    default_fold_stat_paths,
    land_mean,
    model_persistence_corr_maps,
    mse_from_sums,
)


def fold_id(path):
    match = re.search(r"cvfold(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else -1


def mean_summary_value(path, key):
    summary_path = path.replace("hindcast_stats", "hindcast_paper_data")
    summary_path = summary_path.replace("hindcast_tac_stats_", "hindcast_sample_summary_")
    if not os.path.exists(summary_path):
        return float("nan")
    with np.load(summary_path, allow_pickle=True) as data:
        if key not in data:
            return float("nan")
        values = np.asarray(data[key], dtype=np.float64)
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if values.size else float("nan")


def load_stats(path):
    with np.load(path, allow_pickle=False) as data:
        stats = {key: np.asarray(data[key], dtype=np.float64) for key in STAT_KEYS}
        mask = np.asarray(data["mask"], dtype=np.uint8)
        years = np.atleast_1d(data["years"]).astype(int).tolist() if "years" in data else []
        n_samples = int(np.asarray(data["n_samples"]).item()) if "n_samples" in data else int(np.nanmax(stats["count"]))
    return stats, mask, years, n_samples


def read_weekly7_values(path):
    with np.load(path, allow_pickle=False) as data:
        if "weekly7_model_tac" in data:
            return (
                float(np.asarray(data["weekly7_model_tac"]).item()),
                float(np.asarray(data["weekly7_persistence_tac"]).item()),
                int(np.asarray(data["weekly7_n_samples"]).item()),
                "weekly7",
            )
        if "weekly_truth7_model_tac" in data:
            return (
                float(np.asarray(data["weekly_truth7_model_tac"]).item()),
                float(np.asarray(data["weekly_truth7_persistence_tac"]).item()),
                int(np.asarray(data["weekly_truth7_n_samples"]).item()),
                "truth7",
            )
    return float("nan"), float("nan"), 0, ""


def print_fold_table(paths):
    print("Per-fold held-out test diagnostics")
    print("==================================")
    print(
        f"{'Fold':>4s} {'Years':>17s} {'N':>5s} {'TAC':>7s} {'Pers':>7s} "
        f"{'Skill':>8s} {'W7':>7s} {'W7Pers':>7s} {'W7Skill':>8s} "
        f"{'MSE':>7s} {'PersMSE':>8s} {'MAE':>7s}"
    )
    print("-" * 112)
    rows = []
    for path in sorted(paths, key=fold_id):
        stats, mask, years, n_samples = load_stats(path)
        model_map, persist_map = model_persistence_corr_maps(stats, mask)
        model_tac = land_mean(model_map, mask)
        persist_tac = land_mean(persist_map, mask)
        model_mse = mse_from_sums(
            stats["pred_sq_sum"], stats["truth_sq_sum"], stats["pred_truth_sum"],
            stats["count"], mask,
        )
        persist_mse = mse_from_sums(
            stats["persist_sq_sum"], stats["truth_sq_sum"], stats["persist_truth_sum"],
            stats["count"], mask,
        )
        mae = mean_summary_value(path, "mae")
        weekly_tac, weekly_pers, weekly_n, weekly_kind = read_weekly7_values(path)
        year_text = f"{min(years)}-{max(years)}" if years else "-"
        rows.append((fold_id(path), model_tac - persist_tac, model_tac, persist_tac))
        print(
            f"{fold_id(path):4d} {year_text:>17s} {n_samples:5d} "
            f"{model_tac:7.4f} {persist_tac:7.4f} {model_tac - persist_tac:+8.4f} "
            f"{weekly_tac:7.4f} {weekly_pers:7.4f} {weekly_tac - weekly_pers:+8.4f} "
            f"{model_mse:7.3f} {persist_mse:8.3f} {mae:7.3f}"
        )
    print()
    if rows:
        rows_sorted = sorted(rows, key=lambda x: x[1])
        print(
            f"Worst TAC-skill fold: {rows_sorted[0][0]} "
            f"({rows_sorted[0][1]:+.4f}; model={rows_sorted[0][2]:.4f}, pers={rows_sorted[0][3]:.4f})"
        )
        print(
            f"Best TAC-skill fold:  {rows_sorted[-1][0]} "
            f"({rows_sorted[-1][1]:+.4f}; model={rows_sorted[-1][2]:.4f}, pers={rows_sorted[-1][3]:.4f})"
        )
        print()


def load_monthly_files(pattern):
    files = sorted(glob.glob(pattern))
    aggregate = {}
    mask = None
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            months = np.asarray(data["months"], dtype=np.int16)
            mask = np.asarray(data["mask"], dtype=np.uint8)
            for i, month in enumerate(months):
                if int(month) not in aggregate:
                    aggregate[int(month)] = {
                        key: np.zeros_like(np.asarray(data[f"{key}_by_month"][i], dtype=np.float64))
                        for key in STAT_KEYS
                    }
                for key in STAT_KEYS:
                    aggregate[int(month)][key] += np.asarray(data[f"{key}_by_month"][i], dtype=np.float64)
    return aggregate, mask, files


def print_monthly_table(pattern):
    monthly, mask, files = load_monthly_files(pattern)
    if not files or mask is None:
        print("No monthly stats found.")
        return
    print("Monthly stitched diagnostics")
    print("============================")
    print(f"{'Month':>5s} {'NpixMean':>9s} {'TAC':>7s} {'Pers':>7s} {'Skill':>8s} {'MSE':>7s} {'PersMSE':>8s}")
    print("-" * 68)
    month_names = {5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
    rows = []
    for month in sorted(monthly):
        stats = monthly[month]
        model_map, persist_map = model_persistence_corr_maps(stats, mask)
        model_tac = land_mean(model_map, mask)
        persist_tac = land_mean(persist_map, mask)
        model_mse = mse_from_sums(
            stats["pred_sq_sum"], stats["truth_sq_sum"], stats["pred_truth_sum"],
            stats["count"], mask,
        )
        persist_mse = mse_from_sums(
            stats["persist_sq_sum"], stats["truth_sq_sum"], stats["persist_truth_sum"],
            stats["count"], mask,
        )
        n_mean = land_mean(stats["count"], mask)
        rows.append((month, model_tac - persist_tac, model_tac, persist_tac))
        print(
            f"{month_names.get(month, str(month)):>5s} {n_mean:9.1f} "
            f"{model_tac:7.4f} {persist_tac:7.4f} {model_tac - persist_tac:+8.4f} "
            f"{model_mse:7.3f} {persist_mse:8.3f}"
        )
    print()
    if rows:
        rows_sorted = sorted(rows, key=lambda x: x[1])
        print(
            f"Worst month by TAC skill: {month_names.get(rows_sorted[0][0], rows_sorted[0][0])} "
            f"({rows_sorted[0][1]:+.4f})"
        )
        print(
            f"Best month by TAC skill:  {month_names.get(rows_sorted[-1][0], rows_sorted[-1][0])} "
            f"({rows_sorted[-1][1]:+.4f})"
        )
        print()

    weekly_monthly, weekly_mask, weekly_files = load_weekly7_monthly_files(pattern)
    if weekly_files and weekly_mask is not None and weekly_monthly:
        print("Monthly true-weekly7 diagnostics")
        print("================================")
        print(f"{'Month':>5s} {'NpixMean':>9s} {'TAC':>7s} {'Pers':>7s} {'Skill':>8s}")
        print("-" * 46)
        for month in sorted(weekly_monthly):
            stats = weekly_monthly[month]
            model_map, persist_map = model_persistence_corr_maps(stats, weekly_mask)
            model_tac = land_mean(model_map, weekly_mask)
            persist_tac = land_mean(persist_map, weekly_mask)
            n_mean = land_mean(stats["count"], weekly_mask)
            print(
                f"{month_names.get(month, str(month)):>5s} {n_mean:9.1f} "
                f"{model_tac:7.4f} {persist_tac:7.4f} {model_tac - persist_tac:+8.4f}"
            )
        print()


def load_weekly7_monthly_files(pattern):
    files = sorted(glob.glob(pattern))
    aggregate = {}
    mask = None
    used = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            if "weekly7_months" not in data:
                continue
            months = np.asarray(data["weekly7_months"], dtype=np.int16)
            mask = np.asarray(data["mask"], dtype=np.uint8)
            used.append(path)
            for i, month in enumerate(months):
                if int(month) not in aggregate:
                    aggregate[int(month)] = {
                        key: np.zeros_like(np.asarray(data[f"weekly7_{key}_by_month"][i], dtype=np.float64))
                        for key in STAT_KEYS
                    }
                for key in STAT_KEYS:
                    aggregate[int(month)][key] += np.asarray(
                        data[f"weekly7_{key}_by_month"][i], dtype=np.float64
                    )
    return aggregate, mask, used


def print_region_table(path):
    if not os.path.exists(path):
        print(f"Region data not found: {path}")
        return
    with np.load(path, allow_pickle=False) as data:
        names = np.asarray(data["region_names"]).astype(str)
        model = np.asarray(data["region_model_tac"], dtype=np.float64)
        persist = np.asarray(data["region_persistence_tac"], dtype=np.float64)
        skill = np.asarray(data["region_skill"], dtype=np.float64)
    print("Regional TAC skill")
    print("==================")
    print(f"{'Region':<14s} {'TAC':>7s} {'Pers':>7s} {'Skill':>8s}")
    print("-" * 40)
    order = np.argsort(skill)
    for i in order:
        print(f"{names[i]:<14s} {model[i]:7.4f} {persist[i]:7.4f} {skill[i]:+8.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Fold hindcast stat files or globs.")
    parser.add_argument("--monthly_glob", default="hindcast_paper_data/hindcast_monthly_stats_cvfold*_test.npz")
    parser.add_argument("--region_data", default="paper_figures/tac_skill_maps_data.npz")
    args = parser.parse_args()

    paths = args.files or default_fold_stat_paths(".")
    if not paths:
        raise FileNotFoundError("No fold hindcast stat files found.")
    print_fold_table(paths)
    print_monthly_table(args.monthly_glob)
    print_region_table(args.region_data)


if __name__ == "__main__":
    main()
