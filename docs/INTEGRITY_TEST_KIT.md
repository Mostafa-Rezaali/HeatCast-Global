# HeatCast-Global Repository Integrity Test Kit

This kit checks that the code still implements the intended HeatCast experiment,
not merely that Python files compile.

## Protected Contracts

- Daily exceedance labels use month-specific MJJAS train-year q95 thresholds.
- Five CV test folds are disjoint and cover 1981-2023 exactly once.
- Model-training, calibration, and test roles remain leakage-separated.
- Distributional output means are persistence residuals and sigma is positive.
- Gaussian CRPS matches numerical quadrature.
- The grid refiner changes the mean only, never predicted variance.
- W34 training and evaluation both use leads 15-28.
- W34 uses the distributional CRPS head, bounded decoder memory, and window TAC monitor.
- W34 evaluation saves fold arrays and uses leakage-clean cross-fitted stitching.
- Critical Slurm jobs retain the established memory, GPU, Git-pull, and email settings.
- Runtime model/data artifacts are not accidentally committed.
- Global mode defaults to ERA5 `1.5deg`, climatology anomalies, no separate
  coarse-global stack, and the unchanged 15--28 distributional tube.
- CONUS mode retains the original z-score/persistence-residual and 59-field
  coarse-global path.
- The ECMWF downloader and evaluation import one full-W34-valid MJJAS
  Monday/Thursday initialization calendar.
- Global spatial reductions use the shared normalized cosine-latitude helper.
- ERA5 requests are monthly, resumable, atomic, source-recorded, and blocked
  when an unspecified CDS dataset identifier would otherwise be guessed.
- Conservative target and bilinear predictor regridding have a pure-SciPy
  fallback; zarr uses `time=1` chunks and opens only within Dataset workers.
- Global mesh refinement derives from resolution, keeps the complete sphere,
  uses XYZ node/edge features, connects every grid cell including poles and
  the longitude seam, and supports optional processor checkpointing, bf16, and
  gradient accumulation without changing Phase A defaults.
- Global climatology uses exactly four annual harmonics and training years
  only; the poisoning test protects both coefficients and normalization.
- The 26 fine-grid and 8 vector channels are assembled from worker-local cache
  slices, and CI runs a two-step `121 x 240` distributional CPU smoke test with
  sampling and NetCDF export verification.
- Global training losses and validation summaries use cosine-latitude weights;
  global evaluation applies NH-land/full-window-MJJAS masks and reports week3,
  week4, and W34 TAC, MSE skill, CRPS/CRPSS, q95 and upper-tercile probability
  diagnostics, tail containment, monthly/region breakdowns, and year-block CIs.
- Global NetCDF export derives coordinates from the configured grid and writes
  continuous anomaly means, distributional sigma, truth, and supplied
  exceedance probabilities for all three windows without fixed raster sizes.

## Commands

Fast contract audit, suitable before every commit or Slurm submission:

```bash
python src/repo_integrity.py
```

Machine-readable audit:

```bash
python src/repo_integrity.py --json
```

Complete data-free test kit:

```bash
python -m pytest -q
```

Undefined-name and syntax checks:

```bash
python -m py_compile src/repo_integrity.py src/cfm_mesh_train.py src/mesh_backbone.py src/mode_dispatch.py src/exceedance_eval.py
git ls-files -z '*.py' | xargs -0 python -m flake8 --count --select=E9,F63,F7,F82 --show-source --statistics
```

## What This Kit Does Not Prove

The data-free kit cannot prove that external NetCDF/GRIB files are complete,
that a checkpoint is scientifically skillful, or that a full B200 job fits in
memory. Those require runtime preflight and post-run acceptance checks.

Before a production W34 submission, run the fast audit and then inspect the
first training batch for GPU memory, finite loss, sigma above its floor, and the
expected 14-lead output shape. After training, require the existing leakage,
valid-cell-count, and year-block-bootstrap assertions from the evaluation stack.
