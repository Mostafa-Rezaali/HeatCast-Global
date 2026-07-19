"""Validated user-supplied year roles for global five-fold experiments.

The repository intentionally ships no production fold assignment.  This
module turns the approved JSON decision into a strict runtime contract without
inventing train, calibration, test, or ECMWF comparison years.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple


GLOBAL_YEARS: Tuple[int, ...] = tuple(range(1979, 2025))
ROLE_ALIASES = {
    "train_years": "train_years",
    "calibration_years": "calibration_years",
    "val_years": "calibration_years",
    "test_years": "test_years",
}


def _years(values: Sequence[object], label: str) -> Tuple[int, ...]:
    years = tuple(sorted({int(value) for value in values}))
    invalid = sorted(set(years) - set(GLOBAL_YEARS))
    if invalid:
        raise ValueError(f"{label} includes years outside 1979-2024: {invalid}.")
    return years


def validate_fold_table(raw: Mapping[str, object]) -> Dict[int, Dict[str, Tuple[int, ...]]]:
    """Validate five complete, role-disjoint folds and pooled test coverage."""
    records = raw.get("folds")
    if not isinstance(records, list) or len(records) != 5:
        raise ValueError("Global fold JSON must contain exactly five records in 'folds'.")
    folds: Dict[int, Dict[str, Tuple[int, ...]]] = {}
    for record in records:
        if not isinstance(record, Mapping) or "fold" not in record:
            raise ValueError("Each global fold record needs an integer 'fold'.")
        fold = int(record["fold"])
        if fold in folds or fold not in range(5):
            raise ValueError(f"Fold ids must be unique 0..4; found {fold}.")
        roles: Dict[str, Tuple[int, ...]] = {}
        for source, target in ROLE_ALIASES.items():
            if source in record:
                if target in roles:
                    raise ValueError(f"Fold {fold} defines both val_years and calibration_years.")
                roles[target] = _years(record[source], f"fold {fold} {source}")
        missing = sorted({"train_years", "calibration_years", "test_years"} - set(roles))
        if missing:
            raise ValueError(f"Fold {fold} is missing role lists: {missing}.")
        train = set(roles["train_years"])
        calibration = set(roles["calibration_years"])
        test = set(roles["test_years"])
        if train & calibration or train & test or calibration & test:
            raise ValueError(f"Fold {fold} train/calibration/test years overlap.")
        covered = train | calibration | test
        if covered != set(GLOBAL_YEARS):
            raise ValueError(
                f"Fold {fold} must assign every 1979-2024 year exactly once; "
                f"missing={sorted(set(GLOBAL_YEARS) - covered)}."
            )
        folds[fold] = roles
    if set(folds) != set(range(5)):
        raise ValueError("Global fold ids must be exactly 0,1,2,3,4.")
    pooled_test = [year for fold in range(5) for year in folds[fold]["test_years"]]
    if len(pooled_test) != len(set(pooled_test)) or set(pooled_test) != set(GLOBAL_YEARS):
        raise ValueError("Five global test folds must partition 1979-2024 exactly once.")
    return folds


def load_fold_table(path: Path) -> Dict[int, Dict[str, Tuple[int, ...]]]:
    """Load an approved fold JSON and reject missing or malformed decisions."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing approved global fold table: {source}. Resolve TODO(USER) in docs/DECISIONS_NEEDED.md."
        )
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Global fold JSON root must be an object.")
    return validate_fold_table(raw)


def comparison_period(path: Path) -> Tuple[int, ...]:
    """Load the separately approved matched ECMWF comparison years."""
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    values = raw.get("ens_comparison_period") if isinstance(raw, Mapping) else None
    if not isinstance(values, list) or not values:
        raise ValueError(
            "Fold JSON needs non-empty ens_comparison_period after the ECMWF cycle decision is pinned."
        )
    return _years(values, "ens_comparison_period")
