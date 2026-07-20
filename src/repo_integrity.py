#!/usr/bin/env python3
"""Fast, data-free repository contract audit for HeatCast-Global.

This audit protects experiment intent and submission-script consistency. It is
deliberately independent of the external NetCDF datasets and GPU runtime.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


W34_LEADS = tuple(range(15, 29))
MJJAS_MONTHS = (5, 6, 7, 8, 9)
EMAIL = "mostafarezaali@ufl.edu"
CURRENT_SUBMISSIONS = (
    "slurm/submit_ens_stack_opportunity.slurm",
    "slurm/submit_ens_widen_cycles.slurm",
    "slurm/submit_export_w34_stack_netcdf.slurm",
    "slurm/submit_global_data_build.slurm",
    "slurm/submit_global_ens_cycles.slurm",
    "slurm/submit_global_w34_eval_stitch.slurm",
    "slurm/submit_global_w34_tube_all.slurm",
    "slurm/submit_paper_evidence_blocks.slurm",
    "slurm/submit_paper_figures_journal.slurm",
    "slurm/submit_teleconnection_stack_analysis.slurm",
    "slurm/submit_w34_eval_stitch.slurm",
    "slurm/submit_w34_tube_all.slurm",
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def _text(root: Path, relative: str) -> str:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required repository file is missing: {relative}")
    return path.read_text(encoding="utf-8")


def _contains_all(text: str, tokens: Iterable[str]) -> tuple[bool, list[str]]:
    missing = [token for token in tokens if token not in text]
    return not missing, missing


def _shell_csv_variable(text: str, name: str) -> tuple[str, ...]:
    match = re.search(rf"(?m)^{re.escape(name)}=([^\n]+)$", text)
    if match is None:
        return ()
    value = match.group(1).strip().strip("\"'")
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _result(name: str, passed: bool, detail: str) -> CheckResult:
    return CheckResult(name=name, passed=bool(passed), detail=detail)


def _required_tokens_check(
    root: Path,
    name: str,
    relative: str,
    tokens: Iterable[str],
) -> CheckResult:
    text = _text(root, relative)
    passed, missing = _contains_all(text, tokens)
    detail = f"{relative}: all required tokens present" if passed else f"{relative}: missing {missing}"
    return _result(name, passed, detail)


def audit_repository(root: Path) -> list[CheckResult]:
    """Return all fast repository-contract checks."""
    root = root.resolve()
    results: list[CheckResult] = []

    cfm = _text(root, "src/cfm_mesh_train.py")
    exceed = _text(root, "src/exceedance_eval.py")
    ens_ingest = _text(root, "src/ens_ingest.py")
    ens_score = _text(root, "src/ens_score.py")
    ens_target_grid = _text(root, "src/ens_target_grid.py")
    ens_compare = _text(root, "src/ens_compare.py")
    ens_stack = _text(root, "src/ens_heatcast_stack_opportunity.py")
    novelty = _text(root, "src/novelty_analyses.py")
    drivers = _text(root, "src/build_driver_tables.py")
    opportunity = _text(root, "src/forecasts_of_opportunity.py")
    mode = _text(root, "src/mode_dispatch.py")
    mesh = _text(root, "src/mesh_backbone.py")
    init_calendar = _text(root, "src/init_calendar.py")
    spatial_weights = _text(root, "src/spatial_weights.py")
    global_evaluation = _text(root, "src/global_evaluation.py")
    stitch = _text(root, "src/stitch_exceedance_folds.py")
    export = _text(root, "src/export_w34_stack_netcdf.py")
    ens_download = _text(root, "src/download_ecmwf_s2s.py")
    w34_train = _text(root, "slurm/submit_w34_tube_all.slurm")
    w34_eval = _text(root, "slurm/submit_w34_eval_stitch.slurm")

    month_literal = "MJJAS_MONTHS = (5, 6, 7, 8, 9)"
    results.append(_result(
        "target.month_specific_daily_exceedance",
        month_literal in cfm
        and "MJJAS_MONTHS: Tuple[int, ...] = (5, 6, 7, 8, 9)" in init_calendar
        and "from init_calendar import MJJAS_MONTHS, mjjas_mon_thu" in exceed
        and "def build_month_q95" in exceed
        and "truth_z > q95_z" not in exceed
        and "(field_z[valid] > threshold[valid])" in exceed,
        "Shared MJJAS calendar, month-specific q95 builders, and strict daily labels are present",
    ))

    results.append(_result(
        "evaluation.fold_safe_guards",
        all(token in exceed for token in (
            "Calibration/eval year overlap before fitting calibrator.",
            "Leakage check failed: evaluation split overlaps training years.",
            "Disjointness assert failed: calibration split",
            "Leakage assert failed: evaluation target year was in train years.",
        )),
        "Training, calibration, and evaluation overlap guards are present",
    ))

    results.append(_result(
        "distributional.mean_sigma_semantics",
        all(token in mode for token in (
            "mean = persistence + mean_raw",
            "sigma = F.softplus(sigma_raw) + float(floor)",
            "def gaussian_crps(",
        )),
        "CONUS persistence-residual mean and positive softplus sigma semantics remain available",
    ))

    results.append(_required_tokens_check(
        root,
        "global.config_contract",
        "src/cfm_mesh_train.py",
        (
            'DOMAIN = os.environ.get("HEATCAST_DOMAIN", "global")',
            'DATA_ROOT = "/blue/nessie/mostafarezaali/HeatCast-Global/"',
            '"1.5deg": {"shape": (121, 240), "mesh_level": 5}',
            '"0.25deg": {"shape": (721, 1440), "mesh_level": 6}',
            'TARGET_MODES = ("zscore_persistence", "climatology_anomaly")',
            'PREDICT_PERSISTENCE_RESIDUAL = TARGET_MODE == "zscore_persistence"',
            'USE_EXTENDED_GLOBAL_FIELDS = DOMAIN == "conus"',
            'CV_FOLD_YEARS = None  # TODO(USER)',
            'ENS_COMPARISON_PERIOD = None  # TODO(USER)',
            'def configure_domain(',
        ),
    ))

    results.append(_result(
        "global.shared_init_calendar_contract",
        "def mjjas_mon_thu(" in init_calendar
        and "W34_LEADS: Tuple[int, ...] = tuple(range(15, 29))" in init_calendar
        and "require_full_w34: bool = True" in init_calendar
        and "from init_calendar import mjjas_mon_thu" in ens_download
        and "from init_calendar import MJJAS_MONTHS, mjjas_mon_thu" in exceed,
        "Downloader and evaluation import one full-W34-valid MJJAS initialization calendar",
    ))

    results.append(_result(
        "global.area_weight_helper_contract",
        "def area_weights(lat):" in spatial_weights
        and "torch.cos(torch.deg2rad(values))" in spatial_weights
        and "np.cos(np.deg2rad(values))" in spatial_weights
        and "def weighted_spatial_mean(" in spatial_weights,
        "Shared NumPy/PyTorch cosine-latitude weighting helpers are present",
    ))

    results.append(_result(
        "global.area_weighted_training_contract",
        "def _area_weighted_mask(model, mask, reference):" in mode
        and "torch.cos(torch.deg2rad(latitude)).clamp_min(0.0)" in mode
        and "gaussian_crps(pred, sigma, y, mask, model=model)" in mode
        and "weighted_mask = _area_weighted_mask(model, mask, v_pred)" in mode
        and "point_weights = _area_weighted_mask(model, mask, y)[valid]" in mode
        and "def _training_weighted_mask(mask, reference):" in cfm
        and "def _metric_spatial_mean(values, valid):" in cfm,
        "Global CRPS, deterministic, CFM, exceedance, auxiliary, and validation reductions are area weighted",
    ))

    results.append(_required_tokens_check(
        root,
        "global.evaluation_windows_contract",
        "src/global_evaluation.py",
        (
            '"week3": WEEK3_LEADS',
            '"week4": WEEK4_LEADS',
            '"w34": W34_LEADS',
            'THRESHOLD_QUANTILES: Mapping[str, float] = {"upper_tercile": 2.0 / 3.0, "q95": 0.95}',
            "def nh_land_mjjas_mask(",
            "def build_fold_window_thresholds(",
            "def evaluate_global_windows(",
            'row[f"q95_tail_containment_{label}"]',
            'row["monthly_region_breakdowns"]',
            "def year_block_bootstrap(",
        ),
    ))

    results.append(_result(
        "global.evaluation_integration_contract",
        "from global_evaluation import (" in exceed
        and "build_fold_window_thresholds" in exceed
        and "evaluate_global_windows" in exceed
        and "def summarize_global_metric_rows(" in stitch
        and '"global_pooled_year_block_bootstrap.csv"' in stitch,
        "Legacy evaluator exposes the additive global evaluator and stitcher writes all-window year-block summaries",
    ))

    results.append(_required_tokens_check(
        root,
        "global.ens_target_grid_contract",
        "src/ens_target_grid.py",
        (
            "class ENSTargetGrid:",
            "def headline_mask(",
            "self.land_mask & (self.lat[:, None] >= 0.0)",
            "def flattened_area_weights(",
            "def target_grid_for_config(",
            "class LazyGlobalChannel:",
            "def read_pixels_times(",
            "class LazyGlobalTruth(LazyGlobalChannel):",
        ),
    ))

    results.append(_result(
        "global.ens_pipeline_contract",
        all(token in ens_ingest for token in (
            "target_grid_for_config(cfm.Config)",
            "expected_domain=target_grid.domain",
            "expected_resolution=target_grid.resolution",
            "domain=np.array(str(domain))",
            "resolution=np.array(str(resolution))",
        ))
        and all(token in ens_score for token in (
            '"heat_index": LazyGlobalTruth(Path(config.TRAINING_DATA_PATH))',
            'fold_sidecar_path(Path(config.TRAINING_DATA_PATH), int(config.CV_FOLD), "normalization")',
            "load_fold_window_statistics(sidecar, int(cfm.Config.CV_FOLD))",
            '"primary_mask": "NH land, full valid window in MJJAS"',
        ))
        and "weights=metric_weights" in ens_compare
        and "weights=metric_weights" in ens_stack,
        "ENS ingest, fold-safe scoring, comparison, and stacking use the configured global grid and area weights",
    ))

    results.append(_required_tokens_check(
        root,
        "global.novelty_analysis_contract",
        "src/novelty_analyses.py",
        (
            "conditioned-ensemble envelope question",
            "def fit_bayesian_gev(",
            "use_gumbel = values.size < int(minimum_shape_samples)",
            "def gev_envelope_analysis(",
            "def tail_shape_analysis(",
            "def joint_event_probabilities(",
            '"independent_marginals"',
            "def storyline_product(",
            "minimum_members: int = 1000",
            "def assert_fold_safe_exports(",
        ),
    ))

    results.append(_result(
        "global.driver_opportunity_contract",
        '"swvl1_trailing20"' in drivers
        and "LazyGlobalChannel(" in drivers
        and "lazy_reader(pixels, train_t)" in drivers
        and "lazy_reader(pixels, target_t)" in drivers
        and "target_grid.flattened_area_weights(land_mask)" in opportunity
        and "GLOBAL_EXPECTED_YEARS = set(range(1979, 2025))" in opportunity
        and "weights=weights" in opportunity,
        "MJO/ENSO/soil opportunity tables use lazy global soil reads and cosine-latitude weighting",
    ))

    results.append(_required_tokens_check(
        root,
        "global.export_contract",
        "src/export_w34_stack_netcdf.py",
        (
            'if str(getattr(config, "DOMAIN", "conus")) == "global":',
            "grid_for_resolution(str(config.RESOLUTION))",
            "def write_global_hindcast_netcdf(",
            '("anomaly_mean", means[name]',
            '("anomaly_sigma", sigmas[name]',
            'f"prob_{key}"',
            'ds.primary_evaluation = "NH land with full valid window in MJJAS"',
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "global.era5_download_contract",
        "src/data_pipeline/download_era5.py",
        (
            'PREFERRED_DAILY_DATASET = "derived-era5-single-levels-daily-statistics"',
            'HOURLY_SINGLE_LEVEL_DATASET = "reanalysis-era5-single-levels"',
            'PRESSURE_LEVEL_DATASET = "reanalysis-era5-pressure-levels"',
            'CDS_CLIMATE_API_URL = "https://cds.climate.copernicus.eu/api"',
            'YEAR_RANGE: Tuple[int, ...] = tuple(range(1979, 2025))',
            '"data_format": "netcdf"',
            '"download_format": "unarchived"',
            'times=("00:00",)',
            "def validate_download_file(",
            "validate_download_file(partial, task)",
            'target.with_suffix(target.suffix + ".part")',
            'partial.replace(target)',
            '"target_source": task.source_choice',
            "def validate_cds_endpoint(",
            "config_path = validate_cds_endpoint()",
            'ThreadPoolExecutor(max_workers=int(workers))',
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "global.regrid_cache_contract",
        "src/data_pipeline/regrid.py",
        (
            "def conservative_regrid_scipy(",
            "def bilinear_regrid_scipy(",
            "periodic=True",
            "weights_path",
            'method not in ("conservative", "bilinear")',
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "global.lazy_zarr_contract",
        "src/data_pipeline/build_cache.py",
        (
            'chunks=(1, grid.shape[0], grid.shape[1], len(CACHE_CHANNELS))',
            'root = zarr.open_group(str(path), mode="a")',
            'Resume date mismatch at index',
            "class LazyGlobalZarrDataset(Dataset):",
            "self._store = None",
            "def __getstate__(self):",
            "def _ensure_open(self):",
            "root = self._ensure_open()",
            "fold_sidecar_path(",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "global.data_build_submission_contract",
        "slurm/submit_global_data_build.slurm",
        (
            "--mem=500G",
            f"--mail-user={EMAIL}",
            "REPO_DIR=/blue/nessie/mostafarezaali/HeatCast-Global",
            "DATA_ROOT=/blue/nessie/mostafarezaali/HeatCast-Global",
            "unset PYTHONHOME PYTHONPATH",
            "ERA5_CDS_RC=${ERA5_CDS_RC:-$HOME/.cdsapirc-era5}",
            'export CDSAPI_RC="$ERA5_CDS_RC"',
            "ERA5_PRESSURE_DATASET=${ERA5_PRESSURE_DATASET:-reanalysis-era5-pressure-levels}",
            "DOWNLOAD_ONLY=${DOWNLOAD_ONLY:-0}",
            '--data_root "$DATA_ROOT"',
            "data_pipeline.download_era5",
            "data_pipeline.build_cache",
        ),
    ))

    global_data_slurm = _text(root, "slurm/submit_global_data_build.slurm")
    results.append(_result(
        "global.data_build_cpu_only_submission",
        "--gres=" not in global_data_slurm
        and "--partition=hpg-b200" not in global_data_slurm
        and "module load cuda/" not in global_data_slurm,
        "Global ERA5 download/regrid/cache build uses the default CPU partition",
    ))

    results.append(_required_tokens_check(
        root,
        "global.spherical_mesh_contract",
        "src/icosahedral_mesh.py",
        (
            "global_domain=False",
            "if self.global_domain:",
            "self.mesh_vertices = finest_verts",
            'if edge_feature_mode == "xyz":',
            "displacement = mesh_xyz[dst] - grid_xyz[src]",
            '"xyz" if self.global_domain else "latlon"',
            "edge_attr[:, :-1] *= -1",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "global.mesh_factory_contract",
        "src/cfm_mesh_train.py",
        (
            'f"mesh_{config.DOMAIN}_{H}x{W}_level{config.MESH_REFINEMENT_LEVEL}"',
            'if config.DOMAIN == "global":',
            "grid_lon = np.linspace(0.0, 360.0, W, endpoint=False)",
            "mask_raw = None",
            'global_domain=config.DOMAIN == "global"',
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "global.phase_b_memory_flags_contract",
        "src/mesh_backbone.py",
        (
            "gradient_checkpointing=False",
            "self.gradient_checkpointing = bool(gradient_checkpointing)",
            "if self.gradient_checkpointing and self.training and torch.is_grad_enabled():",
            "processor_block, h, e",
            "PeriodicLonConv2d if bool(getattr(mesh, \"global_domain\", False))",
        ),
    ))

    results.append(_result(
        "global.precision_accumulation_contract",
        'VALID_PRECISIONS = ("fp32", "bf16")' in cfm
        and 'GRAD_CHECKPOINT = False' in cfm
        and 'GRAD_ACCUM = 1' in cfm
        and "def optimizer_step_boundary(" in cfm
        and 'enabled=Config.PRECISION == "bf16" and device.type == "cuda"' in cfm
        and "(loss / accumulation_steps).backward()" in cfm
        and "if should_step:" in cfm,
        "Phase A fp32 defaults and config-gated bf16/checkpoint/accumulation paths are explicit",
    ))

    results.append(_required_tokens_check(
        root,
        "global.fold_safe_climatology_contract",
        "src/global_targets.py",
        (
            "DEFAULT_HARMONICS = 4",
            "if parsed.year not in train_year_set:",
            "gram += np.outer(feature, feature)",
            "np.linalg.solve(regularized",
            "def fit_fold_preprocessor_from_zarr(",
            "daily = np.asarray(data[index, :, :, :]",
            '"fold_safe": True',
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "global.input_stack_contract",
        "src/global_dataset.py",
        (
            "GLOBAL_INPUT_CHANNELS: Tuple[str, ...]",
            "VECTOR_INPUT_CHANNELS: Tuple[str, ...]",
            "if len(spatial) != 23:",
            '"global_fields": torch.empty((0,)',
            "class GlobalHeatCastDataset(LazyGlobalZarrDataset):",
            "parse_rmm_components_file",
            "normalize_condition_vectors(",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "global.fold_decision_contract",
        "src/fold_config.py",
        (
            "GLOBAL_YEARS: Tuple[int, ...] = tuple(range(1979, 2025))",
            "Global fold JSON must contain exactly five records",
            "train/calibration/test years overlap",
            "Five global test folds must partition 1979-2024 exactly once",
            "def comparison_period(",
            "ens_comparison_period",
        ),
    ))

    results.append(_result(
        "global.production_training_data_contract",
        "def prepare_global_training_datasets(" in cfm
        and "fit_fold_preprocessor_from_zarr(" in cfm
        and 'GlobalHeatCastDataset(store_path, train_indices' in cfm
        and '"valid_indices_override"' in cfm
        and "Global production requires --fold_years_json" in cfm
        and 'if config.DOMAIN == "global":' in cfm
        and "return self._global_delegate[idx]" in cfm,
        "Production training uses approved folds, worker-lazy zarr datasets, and fold-safe preprocessing",
    ))

    results.append(_required_tokens_check(
        root,
        "global.streaming_sidecar_contract",
        "src/build_global_fold_sidecars.py",
        (
            "Build fold-safe global normalization and window-threshold sidecars lazily",
            "np.memmap(",
            "block_pixels",
            "np.nanquantile(block",
            "save_fold_window_statistics(",
            "require_conditions=False",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "global.cpu_smoke_contract",
        "src/global_smoke_test.py",
        (
            'grid = grid_for_resolution("1.5deg")',
            "for _ in range(2):",
            "sampled_member = mean + sigma * torch.randn_like(mean)",
            "_export_dry_run(",
            '"grid_shape": list(grid.shape)',
            '"train_steps": 2',
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "global.training_submission_contract",
        "slurm/submit_global_w34_tube_all.slurm",
        (
            "--gres=gpu:8",
            "--mem=500G",
            f"--mail-user={EMAIL}",
            "WORK_DIR=/blue/nessie/mostafarezaali/HeatCast-Global/",
            "module load cuda/12.9.1",
            "FOLD_YEARS_JSON=${FOLD_YEARS_JSON:?TODO(USER)",
            "--domain global",
            "--target_mode climatology_anomaly",
            "--prediction_leads \"$LEADS\"",
            "--tube_loss_daily_weight 0.80",
            "--tube_loss_weekly_weight 0.20",
            "--early_stop_metric w34_tac",
            "submit_global_w34_eval_stitch.slurm",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "global.evaluation_submission_contract",
        "slurm/submit_global_w34_eval_stitch.slurm",
        (
            "--gres=gpu:1",
            "--mem=500G",
            f"--mail-user={EMAIL}",
            "WORK_DIR=/blue/nessie/mostafarezaali/HeatCast-Global/",
            "15,16,17,18,19,20,21",
            "22,23,24,25,26,27,28",
            "src/stitch_exceedance_folds.py",
            "src/build_driver_tables.py",
            "src/forecasts_of_opportunity.py",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "global.ens_submission_contract",
        "slurm/submit_global_ens_cycles.slurm",
        (
            "--mem=500G",
            f"--mail-user={EMAIL}",
            "WORK_DIR=/blue/nessie/mostafarezaali/HeatCast-Global/",
            "ECMWF_CYCLE_SPECS=${ECMWF_CYCLE_SPECS:?TODO(USER)",
            "comparison_period",
            "--expected_members \"$MEMBERS\"",
            "--comparison_years \"$COMPARISON_YEARS\"",
            "src/ens_heatcast_stack_opportunity.py",
            "src/export_w34_stack_netcdf.py",
        ),
    ))
    global_ens_submission = _text(root, "slurm/submit_global_ens_cycles.slurm")
    results.append(_result(
        "global.ens_submission_cpu_only",
        "--gres=" not in global_ens_submission
        and "--partition=hpg-b200" not in global_ens_submission
        and "module load cuda/" not in global_ens_submission,
        "Global ENS ingest/scoring/comparison uses the default CPU partition",
    ))

    results.append(_required_tokens_check(
        root,
        "global.runbook_contract",
        "docs/RUNBOOK.md",
        (
            "Resolve the scientific gates",
            "Download fresh ERA5",
            "reanalysis-era5-pressure-levels",
            "submit_global_data_build.slurm",
            "submit_global_w34_tube_all.slurm",
            "submit_global_w34_eval_stitch.slurm",
            "submit_global_ens_cycles.slurm",
            "conditioned-ensemble envelope diagnostic",
            "Phase B (`0.25deg`) checklist",
        ),
    ))

    results.append(_result(
        "distributional.grid_refiner_mean_only",
        "mean_raw + self.grid_refiner(mean_raw)" in mesh
        and "torch.cat([mean_raw + self.grid_refiner(mean_raw), var_raw], dim=1)" in mesh,
        "Grid refiner is applied to the distributional mean while variance bypasses it",
    ))

    train_leads = tuple(int(value) for value in _shell_csv_variable(w34_train, "LEADS"))
    eval_leads = tuple(int(value) for value in _shell_csv_variable(w34_eval, "LEADS"))
    results.append(_result(
        "w34.identical_train_eval_leads",
        train_leads == W34_LEADS and eval_leads == W34_LEADS,
        f"W34 train leads={train_leads}; eval leads={eval_leads}",
    ))

    results.append(_required_tokens_check(
        root,
        "w34.training_contract",
        "slurm/submit_w34_tube_all.slurm",
        (
            "--gres=gpu:8",
            "--mem=500G",
            f"--mail-user={EMAIL}",
            "git pull --ff-only origin main",
            "--multi_lead_tube",
            "--tube_decode_chunk_size 2",
            "--distributional_head",
            "--crps_loss",
            "--sigma_floor 0.1",
            "--early_stop_metric tube_weekly7_tac",
            "--tube_loss_weekly_weight 0.20",
            'sbatch --parsable slurm/submit_w34_eval_stitch.slurm',
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "w34.evaluation_contract",
        "slurm/submit_w34_eval_stitch.slurm",
        (
            "--gres=gpu:1",
            "--mem=500G",
            f"--mail-user={EMAIL}",
            "--target_mode window",
            "--window_leads \"$LEADS\"",
            "--calibration_split val",
            "--eval_split test",
            "--calibrator platt",
            "--save_incremental_arrays",
            "--fit_mode cross_fitted",
            "--tube_decode_chunk_size 2",
            "CHECKPOINT=${CHECKPOINT_OVERRIDE:-best_monitor}",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "opportunity.paired_parent_tests",
        "src/forecasts_of_opportunity.py",
        (
            "def paired_year_block_bootstrap_interactions(",
            "selection_parent",
            "driver_parent",
            "driver_interaction_paired_bootstrap.csv",
            "p_holm_mjo",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "s2s.heatcast_ens_stack_opportunity_contract",
        "src/ens_heatcast_stack_opportunity.py",
        (
            "heatcast_ens_stack",
            "crossfit_excluding_fold",
            "paired_chunk(",
            "merge_cycle_probabilities",
            "init_time_index",
            "heatcast_top10_confidence",
            "opportunity_pair_bootstrap.csv",
            "ThreadPoolExecutor(max_workers=fold_workers)",
            "--fold_workers",
            "robustness_by_month.csv",
            "robustness_by_region.csv",
            "robustness_leave_one_out.csv",
            "Region robustness enabled",
            "--driver_table_dir",
            "driver_pair_bootstrap.csv",
            "driver_pair_parent_bootstrap.csv",
            "Paired driver-stratified Stack-vs-ENS tests",
            "generic_teleconnection",
            "driver_family",
            "alldata",
            "Cross-fit assert: PASS",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "s2s.stack_opportunity_submission_contract",
        "slurm/submit_ens_stack_opportunity.slurm",
        (
            "--mem=500G",
            f"--mail-user={EMAIL}",
            "git pull --ff-only origin main",
            "src/ens_heatcast_stack_opportunity.py",
            "cvfold{F}_ens_w34,cvfold{F}_ens_w34_rt2024",
            "--bootstrap_reps 5000",
            "--max_stack_samples_per_fold 500000",
            "FOLD_WORKERS=${FOLD_WORKERS:-5}",
            "DRIVER_ARGS=()",
            "DRIVER_TABLE_DIR",
            "data_cache/slow_driver_tables_w34_alldata",
            "Missing driver table",
            '--fold_workers "$FOLD_WORKERS"',
            "OMP_NUM_THREADS=1",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "s2s.teleconnection_stack_submission_contract",
        "slurm/submit_teleconnection_stack_analysis.slurm",
        (
            "--mem=500G",
            f"--mail-user={EMAIL}",
            "git pull --ff-only origin main",
            '"$PY" src/repo_integrity.py',
            "TELECONNECTION_INDEX_PATHS=${TELECONNECTION_INDEX_PATHS:?",
            "data_cache/slow_driver_tables_w34_teleconnections",
            "ens_heatcast_stack_opportunity_teleconnections",
            "--teleconnection_index_paths \"$TELECONNECTION_INDEX_PATHS\"",
            "--driver_table_dir \"$DRIVER_DIR\"",
            "--fold_workers \"$FOLD_WORKERS\"",
            "TELECONNECTION STACK ANALYSIS COMPLETE",
        ),
    ))
    tele_submit = _text(root, "slurm/submit_teleconnection_stack_analysis.slurm")
    results.append(_result(
        "s2s.teleconnection_stack_cpu_only_submission",
        "--gres=gpu" not in tele_submit
        and "module load cuda" not in tele_submit
        and "--partition=hpg-b200" not in tele_submit,
        "Teleconnection Stack-vs-ENS postprocessing is CPU-only and does not request B200 GPUs",
    ))
    stack_submit = _text(root, "slurm/submit_ens_stack_opportunity.slurm")
    results.append(_result(
        "s2s.stack_opportunity_cpu_only_submission",
        "--gres=gpu" not in stack_submit
        and "module load cuda" not in stack_submit
        and "--partition=hpg-b200" not in stack_submit,
        "Stack/opportunity paired postprocessing is CPU-only and does not request B200 GPUs",
    ))

    results.append(_required_tokens_check(
        root,
        "paper.evidence_blocks_contract",
        "src/build_paper_evidence_blocks.py",
        (
            "mechanism_block.csv",
            "robustness_block.csv",
            "operational_block.csv",
            "paper_evidence_summary.md",
            "generic teleconnection",
            "AllData",
            "paired_stack_vs_ens_driver",
            "driver_pair_parent_bootstrap.csv",
            "Stack minus ENS delta BSS",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "paper.evidence_blocks_submission_contract",
        "slurm/submit_paper_evidence_blocks.slurm",
        (
            "--mem=16G",
            f"--mail-user={EMAIL}",
            "git pull --ff-only origin main",
            "src/build_paper_evidence_blocks.py",
            "paper_evidence_blocks/window_15-16-17-18-19-20-21-22-23-24-25-26-27-28",
        ),
    ))
    evidence_submit = _text(root, "slurm/submit_paper_evidence_blocks.slurm")
    results.append(_result(
        "paper.evidence_blocks_cpu_only_submission",
        "--gres=gpu" not in evidence_submit
        and "module load cuda" not in evidence_submit
        and "--partition=hpg-b200" not in evidence_submit,
        "Paper evidence block builder is CPU-only and does not request B200 GPUs",
    ))

    results.append(_required_tokens_check(
        root,
        "paper.figures_tables_contract",
        "src/build_paper_figures_tables.py",
        (
            "figure_1_headline_skill",
            "figure_2_headline_stack_minus_ens_ci",
            "figure_3_robustness",
            "figure_4_opportunity_and_driver_tests",
            "figure_5_probabilistic_scorecard",
            "figure_6_probability_threshold_operating_curves",
            "figure_7_opportunity_probability_metrics",
            "methods_text_draft.md",
            "narrative_and_claim_boundaries.md",
            "investigation_record.md",
            "reproducibility_manifest.json",
            "table_1_headline_model_metrics.csv",
            "table_6_operational_metrics.csv",
            "table_7_probability_threshold_operating_points.csv",
            "table_8_opportunity_probability_metrics.csv",
            "hit_rate_0.2",
            "false_alarm_ratio_0.2",
            "Do not say HeatCast alone beats ENS.",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "paper.figures_extended_contract",
        "src/build_paper_figures_extended.py",
        (
            "figure_5_spatial_skill",
            "figure_6_reliability_decomposition",
            "figure_7_case_studies",
            "figure_8_per_lead_profile",
            "figure_9_teleconnection_ranking",
            "figure_9_opportunity_discard_curve",
            "figure_10a_leave_one_out_robustness",
            "table_7_stack_ablation_probability",
            "table_8_per_year_head_to_head",
            "table_9_teleconnection_ranking",
            "table_10_computational_cost_comparison",
            "murphy_decomposition",
            "auc_per_cell",
            "reproducibility_manifest.json",
            "source_entry",
        ),
    ))

    results.append(_required_tokens_check(
        root,
        "paper.figures_journal_submission_contract",
        "slurm/submit_paper_figures_journal.slurm",
        (
            "--mem=64G",
            f"--mail-user={EMAIL}",
            "git pull --ff-only origin main",
            "figure_style.py",
            "src/build_paper_figures_tables.py",
            "src/build_paper_figures_extended.py",
            "--w34_log_glob",
            "OPENBLAS_NUM_THREADS=1",
        ),
    ))
    journal_submit = _text(root, "slurm/submit_paper_figures_journal.slurm")
    results.append(_result(
        "paper.figures_journal_cpu_only_submission",
        "--gres=gpu" not in journal_submit
        and "module load cuda" not in journal_submit
        and "--partition=hpg-b200" not in journal_submit,
        "Journal paper figure builder is CPU-only and does not request B200 GPUs",
    ))

    results.append(_result(
        "s2s.mixed_control_perturbed_grib_contract",
        all(token in ens_ingest for token in (
            'for data_type, default_member in (("cf", 0), ("pf", None)):',
            '"filter_by_keys": {"dataType": data_type}',
            '"indexpath": ""',
            "member_dim = _optional_dimension_name(data, member_candidates)",
            "data = data.expand_dims(member_dim)",
            "np.asarray(data[member_dim].values).reshape(-1)",
            "raw = np.concatenate([group[0] for group in groups], axis=0)",
            "member_values = np.concatenate([group[5] for group in groups])",
        )),
        "ENS ingestion opens and combines control and perturbed GRIB groups explicitly",
    ))

    ens_ingest_submission = _text(root, "slurm/submit_ens_widen_cycles.slurm")
    results.append(_result(
        "s2s.parallel_ingest_contract",
        all(token in ens_ingest for token in (
            "ProcessPoolExecutor",
            "multiprocessing.get_context(\"spawn\")",
            "def ingest_one_init(",
            "def _write_ingested_output(",
            "def validate_ingested_output(",
            "os.replace(temporary_path, output_path)",
            "Removing invalid existing output",
        ))
        and all(token in ens_ingest_submission for token in (
            "--cpus-per-task=32",
            "INGEST_WORKERS=${INGEST_WORKERS:-16}",
            '--workers "$INGEST_WORKERS"',
            "export OMP_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
        )),
        "ENS ingestion uses bounded process parallelism and atomic resume-safe outputs",
    ))

    download_s2s = _text(root, "src/download_ecmwf_s2s.py")
    results.append(_result(
        "s2s.parallel_download_contract",
        "ThreadPoolExecutor(max_workers=int(args.workers))" in download_s2s
        and '"--workers"' in download_s2s
        and 'target.with_suffix(target.suffix + ".part")' in download_s2s
        and "partial.replace(target)" in download_s2s
        and "valid_grib(partial)" in download_s2s,
        "ENS downloading uses bounded parallel requests with validated atomic outputs",
    ))

    results.append(_result(
        "s2s.score_extended_global_contract",
        "def configure_fold(" in ens_score
        and "cfm.apply_extended_global_fields()" in ens_score
        and 'if getattr(config, "DOMAIN", "conus") != "global":' in ens_score
        and "return ee.load_norm_stats()" in ens_score
        and "norm_stats = load_scoring_normalizer(cfm.Config)" in ens_score,
        "ENS scoring preserves CONUS fold stats and loads global fold-safe normalization sidecars",
    ))

    results.append(_result(
        "s2s.score_rejects_corrupt_ingest_contract",
        "valid, reason = validate_ingested_output(" in ens_score
        and "expected_domain=expected_domain" in ens_score
        and "expected_resolution=expected_resolution" in ens_score
        and "Found {len(invalid_files)} invalid ingested ENS outputs" in ens_score
        and "Rerun slurm/submit_ens_widen_cycles.slurm" in ens_score,
        "ENS scoring rejects invalid ingested archives with a repair command",
    ))

    results.append(_result(
        "s2s.score_required_month_coverage_contract",
        "def required_target_months_by_lead(" in ens_score
        and "required_months = required_target_months_by_lead(" in ens_score
        and "for month in required_months[int(lead)]:" in ens_score
        and "int(years[target_t]) not in train_year_set" in ens_score,
        "ENS quantile mappings require only observed valid target months and remain target-year fold safe",
    ))

    results.append(_result(
        "s2s.score_qmap_init_fingerprint_contract",
        "mapping_init_indices=mapping_init_indices" in ens_score
        and '"mapping_init_indices" in data.files' in ens_score
        and 'data["mapping_init_indices"]' in ens_score
        and "np.array_equal(" in ens_score,
        "ENS quantile-mapping caches are invalidated when the exact training initialization set changes",
    ))

    results.append(_result(
        "s2s.score_lightweight_cache_contract",
        "def load_ens_scoring_shared_data(" in ens_score
        and '"heat_index": cache_dir / "heat_index.npy"' in ens_score
        and '"time_values": cache_dir / "time_values.npy"' in ens_score
        and "shared_data = load_ens_scoring_shared_data(cfm.Config)" in ens_score
        and "shared_data = cfm.prepare_shared_data" not in ens_score,
        "Parallel ENS folds use read-only heat/time disk memmaps and skip shared global predictor caches",
    ))

    results.append(_result(
        "s2s.multicycle_widening_contract",
        all(token in ens_score for token in (
            "--rt_tag",
            "downloaded S2S hdate initializations are authoritative",
            "quantile_cache_dir(cache_root, window_leads, rt_tag)",
        ))
        and all(token in ens_ingest for token in (
            "--rt_tag",
            "init_list_{rt_tag}.txt",
            "expected_rt_tag=rt_tag",
        ))
        and all(token in ens_compare for token in (
            "def merge_cycle_probabilities(",
            "def resolve_ens_run_groups(",
            "def per_year_comparison_rows(",
            "Bootstrap blocking assert: PASS (calendar year, never cycle).",
            "ens_heatcast_per_year.csv",
        )),
        "ENS cycles are bias-corrected separately, merged without duplicate-init weighting, and bootstrapped by year",
    ))

    results.append(_required_tokens_check(
        root,
        "s2s.multicycle_submission_contract",
        "slurm/submit_ens_widen_cycles.slurm",
        (
            "--mem=500G",
            "--gres=gpu:1",
            f"--mail-user={EMAIL}",
            "run_cycle \"\"",
            "run_cycle rt2024",
            "cvfold{F}_ens_w34,cvfold{F}_ens_w34_rt2024",
            "WINDOW_LABEL=window_${LEADS//,/-}",
            "Skipping complete ENS score",
            "incremental_arrays/manifest.npz",
            "--emit_per_year",
        ),
    ))

    for relative in CURRENT_SUBMISSIONS:
        text = _text(root, relative)
        results.append(_result(
            f"submission.preflight.{relative}",
            "git pull --ff-only origin main" in text
            and '"$PY" src/repo_integrity.py' in text,
            f"{relative}: pulls current code and runs repository integrity preflight",
        ))

    try:
        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        forbidden_suffixes = (
            ".pth", ".pt", ".npy", ".npz", ".nc", ".pkl", ".grib", ".grib2", ".log", ".err"
        )
        forbidden = sorted(
            path for path in tracked
            if path.lower().endswith(forbidden_suffixes) or ".zarr/" in path.lower()
        )
        results.append(_result(
            "repository.no_tracked_runtime_artifacts",
            not forbidden,
            "No model/data/runtime artifacts tracked" if not forbidden else f"Tracked runtime artifacts: {forbidden}",
        ))
        tracked_submissions = tuple(sorted(path for path in tracked if path.endswith(".slurm")))
        results.append(_result(
            "repository.current_submission_set",
            tracked_submissions == CURRENT_SUBMISSIONS,
            (
                f"Tracked Slurm entry points are current: {tracked_submissions}"
                if tracked_submissions == CURRENT_SUBMISSIONS
                else f"Expected {CURRENT_SUBMISSIONS}; found {tracked_submissions}"
            ),
        ))
    except (OSError, subprocess.CalledProcessError) as exc:
        results.append(_result("repository.no_tracked_runtime_artifacts", False, f"git ls-files failed: {exc}"))

    workflow = _text(root, ".github/workflows/python-package.yml")
    results.append(_result(
        "ci.runs_integrity_and_pytest",
        "python src/repo_integrity.py" in workflow and "pytest" in workflow,
        "GitHub Actions runs the contract audit and pytest",
    ))
    results.append(_result(
        "ci.runs_global_cpu_smoke",
        "python src/cfm_mesh_train.py --smoke_test" in workflow,
        "GitHub Actions runs the true-shape global CPU smoke test",
    ))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    results = audit_repository(args.root)
    passed = sum(result.passed for result in results)
    if args.json:
        print(json.dumps({
            "passed": passed,
            "total": len(results),
            "checks": [asdict(result) for result in results],
        }, indent=2))
    else:
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] {result.name}: {result.detail}")
        print(f"\nRepository integrity: {passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
