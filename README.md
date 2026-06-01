# HeatCast

HeatCast is a GraphCast-style mesh GNN for CONUS warm-season T2max hindcasts.
The current codebase focuses on direct day-15 prediction and experimental
same-initialization multi-lead tube forecasting for `t+12...t+18`.

The repository contains code and job scripts only. Training data, global fields,
checkpoints, caches, hindcast outputs, plots, and proposal/personal artifacts are
intentionally excluded from Git.

## What This Model Does

HeatCast predicts daily PRISM T2max z-score fields over CONUS from local,
global, static, seasonal, and teleconnection predictors.

The main production path is deterministic:

- Input: current and recent local CONUS fields, static features, teleconnection
  indices, and coarse global atmospheric/ocean fields.
- Backbone: GraphCast-style grid-to-mesh, mesh message passing, and mesh-to-grid
  decoder on an icosahedral mesh.
- Output: residual added to day-0 persistence.
- Target: direct daily T2max at `t+15`.
- Verification: daily TAC plus true 7-day mean TAC.

The experimental tube path keeps the same CFM/GNN identity but predicts seven
daily leads from one initialization:

```text
t+12, t+13, t+14, t+15, t+16, t+17, t+18
```

This supports same-init weekly verification by averaging the predicted daily
tube instead of stitching neighboring independent initializations.

## Repository Layout

```text
cfm_mesh_train.py                  Main training, validation, and hindcast export entry point
mesh_backbone.py                   MeshFlowNet encoder-processor-decoder backbone
mode_dispatch.py                   Deterministic and CFM/probabilistic loss/sample dispatch
icosahedral_mesh.py                Icosahedral mesh construction and grid/mesh helpers
stitch_hindcast_tac.py             Combine fold-level TAC sufficient statistics
tac_skill_maps.py                  Generate TAC skill maps and regional summaries
summarize_hindcast_diagnostics.py  Fold, monthly, regional diagnostic tables
compute_baselines.py               Climatology, persistence, MeshFlowNet, optional ridge baselines
export_per_year_stats.py           Per-year sufficient statistics for block bootstrap
bootstrap_significance.py          Year-block bootstrap confidence intervals and p-values
audit_fold2_light.py               Lightweight audit script for weak fold diagnostics
validate_truth_slice.py            Validate suspicious target/truth slices
ensemble_crps_analysis.py          Ensemble CRPS analysis utilities
publication_analysis_utils.py      Shared publication-analysis helpers
submit_weekly7_production.slurm    Production 5-fold daily model with weekly7 monitor
submit_tube7_v1.slurm              Tube7 diagnostic/experimental run
submit_per_year_bootstrap.slurm    Per-year export and bootstrap significance job
submit_multiseed_ensemble.slurm    Multi-seed ensemble training/export job
Model_Inputs.txt                   Input-channel reference
```

## Data Requirements

The code expects external NetCDF files on the HPC filesystem. These are not
included in the repository.

Default paths in `cfm_mesh_train.py`:

```text
/blue/nessie/mostafarezaali/Teleconnection/VDM_Training_Data_Extended_v2.nc
/blue/nessie/mostafarezaali/Teleconnection/Global_Coarse_Conditions_Extended.nc
/blue/nessie/mostafarezaali/Teleconnection/CONUS_topography_ETOPO2022_60s_on_model_grid.nc
```

Default output/cache root:

```text
/blue/nessie/mostafarezaali/Teleconnection/
```

If running somewhere else, update the paths in `Config` inside
`cfm_mesh_train.py` or edit the corresponding `WORK_DIR` and data paths in the
SLURM scripts.

## Inputs

Local CONUS input has 19 channels:

- `t2m_prism[t]`, `t2m_prism[t-1]`, `t2m_prism[t-2]`
- local meteorological predictors at `t`
- topography, latitude, longitude
- day-of-year sine/cosine
- TOA insolation
- land mask

Vector input has 5 teleconnection-index channels.

Global input currently uses 59 coarse/global fields, decomposed into:

- trailing 20-day low-frequency component
- current residual component

This gives 118 global input channels when using one lag and two components.
See `Model_Inputs.txt` for the full channel list.

## Environment

The production scripts are written for UF HiPerGator B200 nodes and assume:

- CUDA module: `cuda/12.9.1`
- Python/torch environment:

```text
/blue/nessie/mostafarezaali/.conda/envs/torch_b200/
```

Core Python dependencies:

- `torch`
- `torchmetrics`
- `numpy`
- `netCDF4`
- `matplotlib`
- `tqdm`
- `scipy`

Optional analysis/figure dependencies used by some scripts:

- `Pillow`
- `reportlab`
- `openpyxl`

## Quick Sanity Check

From the HPC work directory:

```bash
python -m py_compile \
  cfm_mesh_train.py mesh_backbone.py mode_dispatch.py icosahedral_mesh.py \
  stitch_hindcast_tac.py tac_skill_maps.py summarize_hindcast_diagnostics.py \
  compute_baselines.py export_per_year_stats.py bootstrap_significance.py

python -c "import cfm_mesh_train; print('cfm_mesh_train import OK')"
```

## Production Run

The current recommended production workflow is:

```bash
sbatch submit_weekly7_production.slurm
```

This runs all five cross-validation folds with:

```bash
--deterministic
--epochs 30
--batch_size 1
--learning_rate 5e-5
--dropout 0.15
--warmup_epochs 10
--disable_anomaly_corr_loss
--early_stop_metric weekly7_tac
--early_stop_patience 5
--early_stop_min_epoch 12
```

It trains each fold, exports held-out test statistics, stitches the 5-fold
hindcast, and writes diagnostic summaries.

Main output locations:

```text
hindcast_stats/
hindcast_paper_data/
paper_figures/pub_weekly7/
training_metrics/
test_prediction_plots/
```

## Tube7 Experimental Run

The tube experiment predicts a seven-day daily target tube from one
initialization:

```bash
sbatch submit_tube7_v1.slurm
```

By default, the script runs fold 2 only:

```bash
FOLDS=${FOLDS:-"2"}
```

To run other folds:

```bash
FOLDS="0 1 2 3 4" sbatch submit_tube7_v1.slurm
```

The important flags are:

```bash
--multi_lead_tube
--prediction_leads 12,13,14,15,16,17,18
--early_stop_metric tube_weekly7_tac
```

Tube loss:

```text
0.80 * mean_daily_MSE
+0.10 * center_t+15_MSE
+0.10 * same_init_weekly7_MSE
```

Use this as an experimental branch/config, not as a replacement for the
production `pub_weekly7` run until it beats the production reference.

## Hindcast Export

A trained checkpoint can be exported manually:

```bash
torchrun --standalone --nproc_per_node=1 cfm_mesh_train.py \
  --mode export_hindcast \
  --deterministic \
  --cv_fold 0 \
  --run_name cvfold0_pub_weekly7 \
  --checkpoint trained_cfm_direct15_cvfold0_pub_weekly7_best_monitor.pth \
  --hindcast_splits test
```

For tube checkpoints, add:

```bash
--multi_lead_tube \
--prediction_leads 12,13,14,15,16,17,18
```

Use `nproc_per_node=1` for paper-data export. Distributed export skips some
sidecar paper-data products.

## Stitching And Diagnostics

Stitch fold-level statistics:

```bash
python -u stitch_hindcast_tac.py \
  hindcast_stats/hindcast_tac_stats_cvfold*_pub_weekly7_test.npz \
  --output hindcast_stats/stitched_5fold_tac_pub_weekly7.npz
```

Generate weekly7 skill maps:

```bash
python -u tac_skill_maps.py \
  hindcast_stats/hindcast_tac_stats_cvfold*_pub_weekly7_test.npz \
  --metric weekly7 \
  --output_dir paper_figures/pub_weekly7
```

Summarize fold, month, and region diagnostics:

```bash
python -u summarize_hindcast_diagnostics.py \
  hindcast_stats/hindcast_tac_stats_cvfold*_pub_weekly7_test.npz \
  --monthly_glob "hindcast_paper_data/hindcast_monthly_stats_cvfold*_pub_weekly7_test.npz" \
  --region_data paper_figures/pub_weekly7/tac_skill_maps_data.npz
```

Compute baseline table:

```bash
python -u compute_baselines.py \
  hindcast_stats/hindcast_tac_stats_cvfold*_pub_weekly7_test.npz \
  --paper_dir hindcast_paper_data \
  --output_dir paper_figures/pub_weekly7
```

## Bootstrap Significance

Per-year export and year-block bootstrap are heavier than the lightweight map
scripts. Run them on a compute node:

```bash
sbatch submit_per_year_bootstrap.slurm
```

This writes:

```text
hindcast_stats/per_year_tac_stats_pub_weekly7.npz
paper_figures/pub_weekly7/weekly7_bootstrap_results.npz
```

## Current Reference Result

The `pub_weekly7` production run produced:

```text
Daily TAC:       model 0.0938, persistence 0.0767, skill +0.0170
True weekly7 TAC: model 0.1632, persistence 0.1367, skill +0.0265
```

Year-block bootstrap for weekly7 skill:

```text
Skill 95% CI: [-0.0047, +0.0595]
p-value:      0.0471
```

Interpret these as a benchmark-quality result, not a claim that the model wins
in every fold, month, or region. The diagnostics intentionally report weak
folds and seasonal/regional failures.

## Git And Artifact Policy

Do not commit:

- NetCDF data
- checkpoints
- cache files
- `.npy`, `.npz`, `.pkl`
- generated figures
- SLURM logs
- proposal/personal documents
- `node_modules`

The `.gitignore` file is configured to keep the repository code-only.

## Citation / Paper Framing

Paper framing is:

> A transparent CONUS warm-season day-15 and weekly T2max hindcast benchmark
> using a GraphCast-style mesh GNN with global teleconnection-aware predictors.

The strongest claim is not broad operational superiority. The strongest claim is
that the model provides a reproducible neural hindcast benchmark with clear MSE
gains, modest weekly7 TAC skill, and explicit failure analysis by fold, month,
and region.
