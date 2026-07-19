#!/usr/bin/env python3
"""Build fold-safe global normalization and window-threshold sidecars lazily.

The normalization sidecar is created by the global training-data factory. This
driver then streams normalized training targets into disk memmaps and computes
week3, week4, and W34 q95/upper-tercile fields in bounded pixel blocks.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import cfm_mesh_train as cfm
from data_pipeline.build_cache import fold_sidecar_path
from global_evaluation import (
    GLOBAL_WINDOWS,
    THRESHOLD_QUANTILES,
    load_fold_window_statistics,
    save_fold_window_statistics,
)
from global_targets import label_to_date


def _center_lead(leads) -> int:
    return int(np.floor(np.median(np.asarray(leads, dtype=np.float64)) + 0.5))


def build_streaming_thresholds(dataset, train_years, work_dir: Path, *, block_pixels: int = 256):
    """Compute monthly window thresholds/base rates without full-array RAM use."""
    leads = tuple(int(value) for value in cfm.prediction_leads(cfm.Config))
    lead_positions = {
        name: tuple(leads.index(int(lead)) for lead in window_leads)
        for name, window_leads in GLOBAL_WINDOWS.items()
    }
    height, width = cfm.Config.IMAGE_SIZE
    cells = height * width
    counts: Dict[Tuple[str, int], int] = {}
    assignments = []
    for init_index in dataset.indices:
        initialization = label_to_date(dataset.date_labels[int(init_index)])
        if initialization.year not in set(int(value) for value in train_years):
            raise AssertionError("Threshold builder received a non-training initialization.")
        sample_assignment = {}
        for name, window_leads in GLOBAL_WINDOWS.items():
            month = (initialization + timedelta(days=_center_lead(window_leads))).month
            key = (name, int(month))
            sample_assignment[name] = key
            counts[key] = counts.get(key, 0) + 1
        assignments.append(sample_assignment)
    work_dir.mkdir(parents=True, exist_ok=True)
    memmaps = {
        key: np.memmap(
            work_dir / f"{key[0]}_month{key[1]:02d}.float32.dat",
            mode="w+",
            dtype=np.float32,
            shape=(count, cells),
        )
        for key, count in counts.items()
    }
    rows = {key: 0 for key in counts}
    for sample_index in range(len(dataset)):
        target = np.asarray(dataset[sample_index][0], dtype=np.float32)
        for name, positions in lead_positions.items():
            key = assignments[sample_index][name]
            memmaps[key][rows[key]] = np.nanmean(target[list(positions)], axis=0).reshape(-1)
            rows[key] += 1
    for values in memmaps.values():
        values.flush()

    thresholds = {
        name: {
            label: np.full((12, height, width), np.nan, dtype=np.float32)
            for label in THRESHOLD_QUANTILES
        }
        for name in GLOBAL_WINDOWS
    }
    base_rates = {
        name: {
            label: np.full((12, height, width), np.nan, dtype=np.float32)
            for label in THRESHOLD_QUANTILES
        }
        for name in GLOBAL_WINDOWS
    }
    for (name, month), values in memmaps.items():
        for start in range(0, cells, int(block_pixels)):
            stop = min(start + int(block_pixels), cells)
            block = np.asarray(values[:, start:stop], dtype=np.float32)
            for label, quantile in THRESHOLD_QUANTILES.items():
                threshold = np.nanquantile(block, float(quantile), axis=0).astype(np.float32)
                rate = np.nanmean(block > threshold[None], axis=0).astype(np.float32)
                thresholds[name][label][month - 1].reshape(-1)[start:stop] = threshold
                base_rates[name][label][month - 1].reshape(-1)[start:stop] = rate
    return thresholds, base_rates, tuple(path for path in work_dir.glob("*.float32.dat"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--fold_years_json", required=True)
    parser.add_argument("--resolution", choices=("1.5deg", "0.25deg"), default="1.5deg")
    parser.add_argument("--block_pixels", type=int, default=256)
    args = parser.parse_args()

    cfm.Config.CV_FOLD_YEARS_PATH = args.fold_years_json
    cfm.configure_domain("global", args.resolution, "climatology_anomaly", cfm.Config)
    cfm.Config.CV_FOLD = int(args.fold)
    cfm.Config.CV_TEST_OFFSETS = (int(args.fold),)
    cfm.Config.CV_VAL_OFFSETS = ((int(args.fold) + 1) % 5,)
    bundle = cfm.prepare_global_training_datasets(rank=0, ddp=False, require_conditions=False)
    destination = fold_sidecar_path(Path(cfm.Config.TRAINING_DATA_PATH), args.fold, "thresholds")
    if destination.exists():
        _, _, saved_train_years = load_fold_window_statistics(destination, args.fold)
        if set(saved_train_years) == set(bundle["train_years"]):
            print(f"Fold {args.fold} thresholds already match approved train years: {destination}")
            return 0
    work_dir = destination.parent / f"fold{args.fold}_threshold_work"
    thresholds, base_rates, work_files = build_streaming_thresholds(
        bundle["datasets"]["train"],
        bundle["train_years"],
        work_dir,
        block_pixels=args.block_pixels,
    )
    save_fold_window_statistics(
        destination,
        thresholds,
        base_rates,
        fold=args.fold,
        train_years=bundle["train_years"],
    )
    for path in work_files:
        path.unlink(missing_ok=True)
    work_dir.rmdir()
    print(f"Saved fold-safe week3/week4/W34 thresholds: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
