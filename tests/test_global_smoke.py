"""CLI-level end-to-end regression for the true-shape global CPU smoke mode."""

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_global_smoke_cli_runs_two_steps_sampling_and_export():
    completed = subprocess.run(
        [sys.executable, "src/cfm_mesh_train.py", "--smoke_test"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["status"] == "pass"
    assert payload["grid_shape"] == [121, 240]
    assert payload["prediction_shape"] == [1, 14, 121, 240]
    assert payload["train_steps"] == 2
    assert payload["sample_finite"] is True
    assert payload["export_dry_run"] is True
    assert payload["elapsed_seconds"] < 300
