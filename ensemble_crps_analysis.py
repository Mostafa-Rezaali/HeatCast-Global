#!/usr/bin/env python3
"""Compute ensemble CRPS and reliability diagnostics.

Without full multi-seed prediction arrays, this script reports the honest
single-member placeholder from existing sample-summary MAE files. When ensemble
prediction files are available, pass --ensemble_glob. Each file should contain:

  pred:  (N,H,W) for one member, or (M,N,H,W) for multiple members
  truth: (N,H,W)
  mask:  (H,W), optional if another file provides it
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from publication_analysis_utils import ensure_dir


def ensemble_crps(ensemble_preds, truth, mask):
    ensemble_preds = np.asarray(ensemble_preds, dtype=np.float32)
    truth = np.asarray(truth, dtype=np.float32)
    mask = np.asarray(mask) > 0.5
    if ensemble_preds.ndim != 4:
        raise ValueError("ensemble_preds must have shape (M,N,H,W)")
    m = ensemble_preds.shape[0]
    abs_diff = np.abs(ensemble_preds - truth[None]).mean(axis=0)
    spread = np.zeros_like(abs_diff, dtype=np.float32)
    pairs = 0
    for i in range(m):
        for j in range(i + 1, m):
            spread += np.abs(ensemble_preds[i] - ensemble_preds[j])
            pairs += 1
    if pairs:
        spread /= float(pairs)
    crps = abs_diff - 0.5 * spread
    return float(np.mean(crps[:, mask]))


def reliability_curve(ensemble_preds, truth, mask, n_bins=10):
    mask = np.asarray(mask) > 0.5
    threshold = np.percentile(truth[:, mask], 90.0)
    probs = np.mean(ensemble_preds > threshold, axis=0)[:, mask].ravel()
    obs = (truth[:, mask].ravel() > threshold).astype(np.float32)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    observed = np.full(n_bins, np.nan, dtype=np.float32)
    forecast = np.full(n_bins, np.nan, dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        if i == n_bins - 1:
            idx = (probs >= edges[i]) & (probs <= edges[i + 1])
        else:
            idx = (probs >= edges[i]) & (probs < edges[i + 1])
        counts[i] = int(idx.sum())
        if counts[i]:
            observed[i] = float(np.mean(obs[idx]))
            forecast[i] = float(np.mean(probs[idx]))
    return centers.astype(np.float32), observed, forecast, counts, float(threshold)


def fold_from_path(path):
    match = re.search(r"cvfold(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else -1


def load_ensemble_prediction_groups(pattern):
    groups = defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        groups[fold_from_path(path)].append(path)
    return dict(groups)


def crps_from_ensemble_files(pattern):
    groups = load_ensemble_prediction_groups(pattern)
    if not groups:
        raise FileNotFoundError(f"No ensemble prediction files matched {pattern!r}")

    crps_per_fold = {}
    all_preds = []
    all_truth = []
    mask = None
    for fold, paths in groups.items():
        members = []
        truth = None
        fold_mask = None
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                pred = np.asarray(data["pred"], dtype=np.float32)
                if pred.ndim == 3:
                    members.append(pred)
                elif pred.ndim == 4:
                    members.extend([pred[i] for i in range(pred.shape[0])])
                else:
                    raise ValueError(f"Unexpected pred shape in {path}: {pred.shape}")
                if truth is None:
                    truth = np.asarray(data["truth"], dtype=np.float32)
                if "mask" in data:
                    fold_mask = np.asarray(data["mask"], dtype=np.uint8)
        if truth is None:
            raise ValueError(f"No truth array found for fold {fold}")
        if fold_mask is None and mask is None:
            raise ValueError(f"No mask array found for fold {fold}; include mask in at least one file.")
        if fold_mask is None:
            fold_mask = mask
        mask = fold_mask
        ens = np.stack(members, axis=0)
        crps_per_fold[fold] = ensemble_crps(ens, truth, fold_mask)
        all_preds.append(ens)
        all_truth.append(truth)

    # Concatenate samples. Assumes all folds have the same member count.
    member_counts = {arr.shape[0] for arr in all_preds}
    if len(member_counts) != 1:
        raise ValueError(f"All folds must have the same ensemble size, got {sorted(member_counts)}")
    stitched_preds = np.concatenate(all_preds, axis=1)
    stitched_truth = np.concatenate(all_truth, axis=0)
    stitched_crps = ensemble_crps(stitched_preds, stitched_truth, mask)
    bins, obs, fcst, counts, threshold = reliability_curve(stitched_preds, stitched_truth, mask)
    return crps_per_fold, stitched_crps, bins, obs, fcst, counts, threshold


def placeholder_from_sample_summaries(pattern):
    crps_per_fold = np.full(5, np.nan, dtype=np.float32)
    persistence_per_fold = np.full(5, np.nan, dtype=np.float32)
    total_model = total_persist = 0.0
    total_n = 0
    for path in sorted(glob.glob(pattern)):
        fold = fold_from_path(path)
        with np.load(path, allow_pickle=True) as data:
            if "mae" not in data:
                continue
            mae = np.asarray(data["mae"], dtype=np.float64)
            mae = mae[np.isfinite(mae)]
            if mae.size:
                crps_per_fold[fold] = float(np.mean(mae))
                total_model += float(np.sum(mae))
                total_n += int(mae.size)
            if "persist_mae" in data:
                pmae = np.asarray(data["persist_mae"], dtype=np.float64)
                pmae = pmae[np.isfinite(pmae)]
                if pmae.size:
                    persistence_per_fold[fold] = float(np.mean(pmae))
                    total_persist += float(np.sum(pmae))
    stitched = total_model / total_n if total_n else float("nan")
    persistence = total_persist / total_n if total_n else float("nan")
    return crps_per_fold, stitched, persistence_per_fold, persistence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble_glob", default=None, help="Optional full ensemble prediction files.")
    parser.add_argument(
        "--summary_glob",
        default="hindcast_paper_data/hindcast_sample_summary_cvfold*_test.npz",
        help="Sample summary files for single-member placeholder CRPS.",
    )
    parser.add_argument("--output_dir", default="paper_figures")
    parser.add_argument("--output", default="ensemble_crps_results.npz")
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    reliability_bins = np.linspace(0.05, 0.95, 10, dtype=np.float32)
    reliability_obs = np.full(10, np.nan, dtype=np.float32)
    reliability_fcst = np.full(10, np.nan, dtype=np.float32)
    reliability_counts = np.zeros(10, dtype=np.int64)
    threshold = np.array(np.nan, dtype=np.float32)
    persistence_crps_stitched = np.array(np.nan, dtype=np.float32)
    persistence_crps_per_fold = np.full(5, np.nan, dtype=np.float32)
    climatology_crps_stitched = np.array(np.nan, dtype=np.float32)

    if args.ensemble_glob:
        fold_dict, stitched, bins, obs, fcst, counts, threshold_value = crps_from_ensemble_files(args.ensemble_glob)
        crps_per_fold = np.full(5, np.nan, dtype=np.float32)
        for fold, value in fold_dict.items():
            if 0 <= fold < len(crps_per_fold):
                crps_per_fold[fold] = value
        crps_stitched = np.array(stitched, dtype=np.float32)
        reliability_bins = bins
        reliability_obs = obs
        reliability_fcst = fcst
        reliability_counts = counts
        threshold = np.array(threshold_value, dtype=np.float32)
        mode = "ensemble"
    else:
        crps_per_fold, stitched, persistence_crps_per_fold, persistence = placeholder_from_sample_summaries(
            args.summary_glob
        )
        crps_stitched = np.array(stitched, dtype=np.float32)
        persistence_crps_stitched = np.array(persistence, dtype=np.float32)
        mode = "single-member-placeholder"

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = out_dir / output_path
    np.savez_compressed(
        output_path,
        mode=np.array(mode),
        crps_per_fold=crps_per_fold.astype(np.float32),
        crps_stitched=crps_stitched,
        persistence_crps_per_fold=persistence_crps_per_fold.astype(np.float32),
        persistence_crps_stitched=persistence_crps_stitched,
        climatology_crps_stitched=climatology_crps_stitched,
        reliability_bins=reliability_bins,
        reliability_observed_freq=reliability_obs,
        reliability_forecast_freq=reliability_fcst,
        reliability_counts=reliability_counts,
        reliability_threshold_z=threshold,
    )

    print("Ensemble CRPS analysis")
    print("======================")
    print(f"Mode: {mode}")
    print(f"CRPS per fold: {np.array2string(crps_per_fold, precision=4)}")
    print(f"Stitched CRPS: {float(crps_stitched):.4f}")
    if np.isfinite(persistence_crps_stitched):
        print(f"Persistence CRPS: {float(persistence_crps_stitched):.4f}")
    if mode != "ensemble":
        print("Reliability is not computed until full ensemble prediction arrays are available.")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

