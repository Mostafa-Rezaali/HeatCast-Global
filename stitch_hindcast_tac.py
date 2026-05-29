#!/usr/bin/env python3
"""
Stitch compact cross-validated hindcast TAC-stat files.

Each input file is produced by:
  cfm_mesh_train.py --mode export_hindcast

The files contain sufficient statistics, not full forecast maps, so combining
five folds stays small and avoids re-reading giant prediction arrays.
"""

import argparse
import glob
import os

import numpy as np


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
WEEKLY7_PREFIX = "weekly7_"
LEGACY_WEEKLY_TRUTH7_PREFIX = "weekly_truth7_"


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif os.path.exists(pattern):
            paths.append(pattern)
    return sorted(dict.fromkeys(paths))


def corr_from_sums(x_sum, y_sum, x_sq_sum, y_sq_sum, xy_sum, count, mask, min_count):
    valid = (mask > 0.5) & (count >= min_count)
    safe_count = np.maximum(count, 1.0)
    cov = xy_sum - (x_sum * y_sum) / safe_count
    x_var = x_sq_sum - (x_sum * x_sum) / safe_count
    y_var = y_sq_sum - (y_sum * y_sum) / safe_count
    denom = np.sqrt(np.maximum(x_var, 0.0) * np.maximum(y_var, 0.0))
    valid &= denom > 1e-12
    corr = np.full(count.shape, np.nan, dtype=np.float32)
    corr[valid] = (cov[valid] / denom[valid]).astype(np.float32)
    return corr, float(np.nanmean(corr[valid])) if np.any(valid) else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Stitch cross-validated hindcast TAC stats.")
    parser.add_argument("files", nargs="+", help="Input .npz files or glob patterns.")
    parser.add_argument("--output", default=None, help="Optional aggregate .npz output path.")
    parser.add_argument("--min_count", type=int, default=2, help="Minimum samples per grid point.")
    args = parser.parse_args()

    paths = expand_inputs(args.files)
    if not paths:
        raise FileNotFoundError(f"No input files matched: {args.files}")

    aggregate = None
    weekly_aggregate = None
    weekly_prefix_used = None
    mask = None
    all_years = set()
    total_samples = 0
    weekly_samples = 0
    any_multi_lead_tube = False

    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            if aggregate is None:
                aggregate = {key: np.array(data[key], dtype=np.float64) for key in STAT_KEYS}
                mask = np.array(data["mask"], dtype=np.uint8)
            else:
                for key in STAT_KEYS:
                    aggregate[key] += np.array(data[key], dtype=np.float64)
            weekly_prefix = None
            if all(f"{WEEKLY7_PREFIX}{key}" in data for key in STAT_KEYS):
                weekly_prefix = WEEKLY7_PREFIX
            elif all(f"{LEGACY_WEEKLY_TRUTH7_PREFIX}{key}" in data for key in STAT_KEYS):
                weekly_prefix = LEGACY_WEEKLY_TRUTH7_PREFIX
            if weekly_prefix is not None:
                if weekly_prefix_used is None:
                    weekly_prefix_used = weekly_prefix
                if weekly_prefix != weekly_prefix_used:
                    raise ValueError(
                        f"Mixed weekly statistic types are not supported: "
                        f"{weekly_prefix_used!r} and {weekly_prefix!r}"
                    )
                if weekly_aggregate is None:
                    weekly_aggregate = {
                        key: np.array(data[f"{weekly_prefix}{key}"], dtype=np.float64)
                        for key in STAT_KEYS
                    }
                else:
                    for key in STAT_KEYS:
                        weekly_aggregate[key] += np.array(
                            data[f"{weekly_prefix}{key}"], dtype=np.float64
                        )
                sample_key = (
                    "weekly7_n_samples"
                    if weekly_prefix == WEEKLY7_PREFIX else "weekly_truth7_n_samples"
                )
                if sample_key in data:
                    weekly_samples += int(np.asarray(data[sample_key]).item())
            total_samples += int(np.asarray(data["n_samples"]).item())
            all_years.update(int(y) for y in np.atleast_1d(data["years"]).astype(int).tolist())
            if "multi_lead_tube" in data and int(np.asarray(data["multi_lead_tube"]).item()) == 1:
                any_multi_lead_tube = True

    model_corr, model_tac = corr_from_sums(
        aggregate["pred_sum"],
        aggregate["truth_sum"],
        aggregate["pred_sq_sum"],
        aggregate["truth_sq_sum"],
        aggregate["pred_truth_sum"],
        aggregate["count"],
        mask,
        args.min_count,
    )
    persistence_corr, persistence_tac = corr_from_sums(
        aggregate["persist_sum"],
        aggregate["truth_sum"],
        aggregate["persist_sq_sum"],
        aggregate["truth_sq_sum"],
        aggregate["persist_truth_sum"],
        aggregate["count"],
        mask,
        args.min_count,
    )

    years = sorted(all_years)
    print("\nCross-validated hindcast TAC")
    print("============================")
    print(f"Files:              {len(paths)}")
    print(f"Samples:            {total_samples}")
    print(f"Years:              {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"Model TAC:          {model_tac:.4f}")
    print(f"Persistence TAC:    {persistence_tac:.4f}")
    print(f"TAC improvement:    {model_tac - persistence_tac:+.4f}")

    weekly_model_corr = None
    weekly_persistence_corr = None
    weekly_model_tac = float("nan")
    weekly_persistence_tac = float("nan")
    if weekly_aggregate is not None:
        weekly_model_corr, weekly_model_tac = corr_from_sums(
            weekly_aggregate["pred_sum"],
            weekly_aggregate["truth_sum"],
            weekly_aggregate["pred_sq_sum"],
            weekly_aggregate["truth_sq_sum"],
            weekly_aggregate["pred_truth_sum"],
            weekly_aggregate["count"],
            mask,
            args.min_count,
        )
        weekly_persistence_corr, weekly_persistence_tac = corr_from_sums(
            weekly_aggregate["persist_sum"],
            weekly_aggregate["truth_sum"],
            weekly_aggregate["persist_sq_sum"],
            weekly_aggregate["truth_sq_sum"],
            weekly_aggregate["persist_truth_sum"],
            weekly_aggregate["count"],
            mask,
            args.min_count,
        )
        if weekly_prefix_used == WEEKLY7_PREFIX and any_multi_lead_tube:
            print("\nTube same-init 7-day mean TAC")
            print("=============================")
        elif weekly_prefix_used == WEEKLY7_PREFIX:
            print("\nTrue 7-day mean TAC")
            print("===================")
        else:
            print("\nLegacy 7-day truth-mean diagnostic")
            print("==================================")
        print(f"Samples:            {weekly_samples}")
        print(f"Model TAC:          {weekly_model_tac:.4f}")
        print(f"Persistence TAC:    {weekly_persistence_tac:.4f}")
        print(f"TAC improvement:    {weekly_model_tac - weekly_persistence_tac:+.4f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_payload = {
            **aggregate,
            "mask": mask,
            "years": np.array(years, dtype=np.int16),
            "n_samples": np.array(total_samples, dtype=np.int32),
            "multi_lead_tube": np.array(int(any_multi_lead_tube), dtype=np.int8),
            "model_tac": np.array(model_tac, dtype=np.float32),
            "persistence_tac": np.array(persistence_tac, dtype=np.float32),
            "model_corr_map": model_corr,
            "persistence_corr_map": persistence_corr,
        }
        if weekly_aggregate is not None:
            output_prefix = weekly_prefix_used or WEEKLY7_PREFIX
            save_payload.update({
                f"{output_prefix}{key}": value
                for key, value in weekly_aggregate.items()
            })
            if output_prefix == WEEKLY7_PREFIX:
                save_payload.update({
                    "weekly7_n_samples": np.array(weekly_samples, dtype=np.int32),
                    "weekly7_model_tac": np.array(weekly_model_tac, dtype=np.float32),
                    "weekly7_persistence_tac": np.array(weekly_persistence_tac, dtype=np.float32),
                    "weekly7_model_corr_map": weekly_model_corr,
                    "weekly7_persistence_corr_map": weekly_persistence_corr,
                })
            else:
                save_payload.update({
                    "weekly_truth7_n_samples": np.array(weekly_samples, dtype=np.int32),
                    "weekly_truth7_model_tac": np.array(weekly_model_tac, dtype=np.float32),
                    "weekly_truth7_persistence_tac": np.array(weekly_persistence_tac, dtype=np.float32),
                    "weekly_truth7_model_corr_map": weekly_model_corr,
                    "weekly_truth7_persistence_corr_map": weekly_persistence_corr,
                })
        np.savez_compressed(args.output, **save_payload)
        print(f"Saved aggregate stats to {args.output}")


if __name__ == "__main__":
    main()
