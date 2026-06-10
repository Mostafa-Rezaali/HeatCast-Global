#!/usr/bin/env python3
"""Choose the W34 evaluation checkpoint from fold-0 validation window TAC and AUC."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import numpy as np


def read_window_tac(path: Path) -> float:
    with np.load(path, allow_pickle=False) as data:
        for key in ("tube_weekly7_model_tac", "weekly7_model_tac"):
            if key in data:
                return float(np.asarray(data[key]).item())
    raise KeyError(f"{path}: missing tube/weekly window TAC.")


def read_window_auc(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for preferred in ("windowed_mu_sigma_error", "heldout_monotonic_calibrator"):
        for row in rows:
            if row.get("model") == preferred:
                return float(row["roc_auc"])
    raise KeyError(f"{path}: missing windowed analytic/calibrated AUC row.")


def choose_checkpoint(rows: Dict[str, Dict[str, float]], tac_tie_tolerance: float) -> tuple[str, str]:
    tac_delta = rows["best_monitor"]["window_tac"] - rows["best_tac"]["window_tac"]
    if abs(tac_delta) > float(tac_tie_tolerance):
        selected = "best_monitor" if tac_delta > 0.0 else "best_tac"
        return selected, "higher fold-0 validation 14-day-mean TAC"
    selected = max(rows, key=lambda name: rows[name]["window_auc"])
    return selected, "window TAC tied; higher fold-0 validation window AUC"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best_tac_stats", required=True)
    parser.add_argument("--best_monitor_stats", required=True)
    parser.add_argument("--best_tac_results", required=True)
    parser.add_argument("--best_monitor_results", required=True)
    parser.add_argument("--output_choice", default="w34_checkpoint_choice.txt")
    parser.add_argument("--output_csv", default="w34_checkpoint_arbitration.csv")
    parser.add_argument("--tac_tie_tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    rows: Dict[str, Dict[str, float]] = {
        "best_tac": {
            "window_tac": read_window_tac(Path(args.best_tac_stats)),
            "window_auc": read_window_auc(Path(args.best_tac_results)),
        },
        "best_monitor": {
            "window_tac": read_window_tac(Path(args.best_monitor_stats)),
            "window_auc": read_window_auc(Path(args.best_monitor_results)),
        },
    }
    selected, reason = choose_checkpoint(rows, args.tac_tie_tolerance)

    Path(args.output_choice).write_text(f"{selected}\n", encoding="utf-8")
    with Path(args.output_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("checkpoint", "window_tac", "window_auc", "selected", "selection_reason"),
        )
        writer.writeheader()
        for name, values in rows.items():
            writer.writerow({
                "checkpoint": name,
                **values,
                "selected": name == selected,
                "selection_reason": reason if name == selected else "",
            })

    print("W34 fold-0 checkpoint arbitration")
    print("================================")
    for name, values in rows.items():
        print(f"{name:<12} window_TAC={values['window_tac']:.4f} window_AUC={values['window_auc']:.4f}")
    print(f"Selected: {selected} ({reason})")
    print(f"Locked choice written to: {args.output_choice}")


if __name__ == "__main__":
    main()
