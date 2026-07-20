# HeatCast-Global Production Runbook

This is the production order for the Phase A ERA5 `1.5deg` experiment on UF
HiPerGator. The code clone and configured runtime root are both
`/blue/nessie/mostafarezaali/HeatCast-Global`; generated data directories are
Git-ignored.

## 1. Resolve the scientific gates

Do not submit production training or ECMWF scoring until every applicable item
in `docs/DECISIONS_NEEDED.md` is approved.

Create an external JSON file (do not commit it until its scientific contents
are approved) with:

- exactly five records under `folds`, with integer `fold` values `0` through
  `4`;
- `train_years`, `calibration_years`, and `test_years` in every record;
- disjoint roles within each fold that cover every year 1979–2024 exactly
  once;
- test-year sets that partition 1979–2024 exactly once across the five folds;
- `ens_comparison_period`, containing only the reforecast years supported by
  the pinned ECMWF cycle/version.

`src/fold_config.py` validates this contract and refuses incomplete,
overlapping, or inferred assignments.

Required runtime inputs:

```text
/blue/nessie/mostafarezaali/HeatCast-Global/cache/teleconnection_5.npy
/blue/nessie/mostafarezaali/HeatCast-Global/drivers/rmm.txt
/blue/nessie/mostafarezaali/HeatCast-Global/drivers/nino34.txt
<approved fold-year JSON>
~/.cdsapirc-era5
```

The five-index array must align exactly with the ERA5 cache time axis. The RMM
and Niño files are parsed by the existing driver-table parsers; no alternate
model-only parser is used.

Keep ERA5 and ECMWF S2S credentials separate. Create `~/.cdsapirc-era5` from
the Climate Data Store API profile with:

```yaml
url: https://cds.climate.copernicus.eu/api
key: <CDS personal access token>
```

The data-build Slurm script exports this file through `CDSAPI_RC`. An existing
`~/.cdsapirc` configured with `https://ecds.ecmwf.int/api` may remain unchanged
for the ECMWF ECDS/S2S workflow; that endpoint does not serve ERA5 collection
IDs.

## 2. Local/data-free preflight

From the repository root:

```bash
python src/repo_integrity.py
python -m pytest -q
python src/cfm_mesh_train.py --smoke_test
git status --short
```

Expected: every contract passes, pytest is green, the smoke result reports
`grid_shape=[121,240]`, and Git contains no data/runtime artifacts.

## 3. Download fresh ERA5, then build the cache and fold sidecars

The pressure-level source is pinned to the official CDS identifier
`reanalysis-era5-pressure-levels`. The fresh-data workflow does not inspect or
reuse the original HeatCast archive. Requests cover all days of 1979--2024 and
default to one NetCDF per variable group per year: 231 total tasks instead of
2,761 monthly tasks. Every file is validated before its atomic rename, and
resume skips only files whose task metadata and NetCDF headers agree.

Submit the download alone first:

```bash
cd /blue/nessie/mostafarezaali/HeatCast-Global && sbatch --export=ALL,DOWNLOAD_ONLY=1,BUILD_FOLD_SIDECARS=0 slurm/submit_global_data_build.slurm
```

The downloader uses `DOWNLOAD_CHUNKING=yearly` by default. It runs eight local
worker threads, but admits at most one active
request per CDS dataset. The three ERA5 collections can therefore progress in
parallel without flooding any one dataset queue. Temporary CDS queue-limit
responses are retried automatically with exponential backoff from 60 seconds
to 15 minutes; completed valid files remain resume-safe. `DOWNLOAD_WORKERS`
controls the local task pool, while `DOWNLOAD_PER_DATASET=1` should remain the
production default:

```bash
cd /blue/nessie/mostafarezaali/HeatCast-Global && sbatch --export=ALL,DOWNLOAD_ONLY=1,BUILD_FOLD_SIDECARS=0,DOWNLOAD_WORKERS=8,DOWNLOAD_PER_DATASET=1 slurm/submit_global_data_build.slurm
```

If CDS rejects a particular annual payload for request-size rather than queue
pressure, resubmit with `DOWNLOAD_CHUNKING=monthly`; annual and monthly target
names are distinct, and the cache builder follows the same configured layout.

Raw files and the deterministic request manifest are written below:

```text
/blue/nessie/mostafarezaali/HeatCast-Global/raw/era5/
/blue/nessie/mostafarezaali/HeatCast-Global/manifests/era5_download_tasks.json
```

After the download completes and the fold table is approved, rerun the same
workflow to conservatively regrid Tmax, bilinearly regrid predictors, write
`time=1` zarr chunks, and build the five fold-safe normalization and
week3/week4/W34 threshold sidecars:

```bash
cd /blue/nessie/mostafarezaali/HeatCast-Global && sbatch --export=ALL,FOLD_YEARS_JSON='/absolute/path/fold_years.json',RESOLUTION='1.5deg',BUILD_FOLD_SIDECARS=1 slurm/submit_global_data_build.slurm
```

This is CPU/I/O work. Following the original HeatCast CPU Slurm pattern, it
uses HiPerGator's default CPU partition and requests no GPU. The expected cache
is:

```text
cache/era5_1.5deg.zarr/
cache/era5_1.5deg.zarr.sidecars/fold{0..4}_normalization.npz
cache/era5_1.5deg.zarr.sidecars/fold{0..4}_thresholds.npz
```

Before training, use `src/data_pipeline/check_cache.py` to compare selected
cached slices with their raw downloads. A successful metadata file alone does
not prove scientific correctness.

## 4. Train five global folds

```bash
sbatch --export=ALL,FOLD_YEARS_JSON='/absolute/path/fold_years.json',RESOLUTION='1.5deg',PRECISION='fp32',GRAD_ACCUM=1,GRAD_CHECKPOINT=0 slurm/submit_global_w34_tube_all.slurm
```

The job uses eight B200 GPUs, batch size one per rank, direct leads 15–28, the
unchanged `0.80` daily + `0.20` W34 Gaussian-CRPS tube loss, and early stopping
on area-weighted NH-land `w34_tac`. It resumes fold checkpoints unless
`CLEAN_START=1` is explicitly supplied. When all five folds pass their
completion checks, it submits `submit_global_w34_eval_stitch.slurm`.

For Phase A memory pressure, increase `GRAD_ACCUM` without changing the
effective scientific sample definition. `GRAD_CHECKPOINT=1` and
`PRECISION=bf16` are Phase B-oriented options and remain off by default.

## 5. Export, evaluate, stitch, and verify

The chained evaluation job can also be submitted directly:

```bash
sbatch --export=ALL,FOLD_YEARS_JSON='/absolute/path/fold_years.json',RESOLUTION='1.5deg' slurm/submit_global_w34_eval_stitch.slurm
```

It performs, in order:

1. held-out hindcast export for validation and test roles;
2. fold-safe q95 evaluation for week3, week4, and W34;
3. cross-fitted five-fold stitching and whole-year bootstrap;
4. MJO, ENSO, and lazy trailing-20-day soil-state driver tables;
5. area-weighted forecasts-of-opportunity analysis.

The global evaluator uses only NH land for headline metrics. Its window
thresholds come from the prebuilt training-year sidecars, and its
checkpoint-dependent error spread is accumulated online rather than retaining
all training fields in RAM. Weak folds, months, and regions remain in outputs.

## 6. Ingest and score pinned ECMWF cycles

Download ECMWF S2S files and create each cycle's authoritative
`init_list_<cycle-tag>.txt` using the pinned model version/cycle. The repository
does not guess cycle names or member counts. Supply each as
`cycle_tag:member_count`:

```bash
sbatch --export=ALL,FOLD_YEARS_JSON='/absolute/path/fold_years.json',RESOLUTION='1.5deg',ECMWF_CYCLE_SPECS='<approved-tag>:<approved-members>[,<approved-tag>:<approved-members>]',INGEST_WORKERS=16,FOLD_WORKERS=2 slurm/submit_global_ens_cycles.slurm
```

This CPU workflow:

- regrids every cycle to the configured ERA5 grid with atomic resume checks;
- scores week3, week4, and W34 using cycle-specific quantile mappings;
- restricts calibration/test records to `ens_comparison_period` from the
  approved JSON;
- compares HeatCast and ENS on identical initialization/cell intersections;
- runs the cross-fitted HeatCast+ENS stack and driver robustness analysis;
- writes the grid-parameterized W34 NetCDF export.

No percentage or headline comparison is valid until the matched-init audit,
cycle metadata, cell counts, and comparison years agree in every fold.

## 7. Novelty products and figures

`src/novelty_analyses.py` consumes explicit fold-safe exported NPZ inputs. Run
each analysis on CPU through Slurm, never on the login node:

```text
--analysis gev         conditioned ENS-GEV envelope and reverse miss test
--analysis tail_shape  Gaussian-head versus CFM q95/q99/q99.9 divergence
--analysis joint       coherent regional area events versus ENS/independence
--analysis storyline   1000+ member plausible-worst-case demonstration
```

The GEV result is a conditioned-ensemble envelope diagnostic, not a
climatological return-period estimate. The storyline remains blocked until the
approved case summers are inside test folds. Generated NetCDF, NPZ, figures,
logs, and checkpoints stay outside Git.

Use the existing paper figure/table builders against the global stitched,
matched-ENS, stack, region, month, and novelty outputs. Inspect all regional
and monthly tables before selecting headline panels.

## 8. Phase B (`0.25deg`) checklist

Phase B is a configuration change, not a fork of the implementation:

- rebuild ERA5 at `RESOLUTION=0.25deg` and verify `721 x 1440` metadata;
- confirm mesh refinement resolves to level 6 and rerun connectivity checks;
- rebuild all five normalization/threshold sidecars at the new resolution;
- start with `GRAD_CHECKPOINT=1`, `PRECISION=bf16`, batch size one, and an
  approved `GRAD_ACCUM` value;
- verify one batch, two optimizer steps, validation, checkpoint resume, and
  export before requesting a full five-fold allocation;
- measure host RAM, zarr I/O, GPU peak allocation, and validation wall time;
- re-ingest ENS to the `0.25deg` target; never reuse `1.5deg` archives or
  regridding weights;
- rerun integrity, pytest, slice checks, fold leakage audits, matched-date
  audits, and year-block bootstrap checks;
- retain the 15–28 leads, distributional semantics, sigma floor, loss weights,
  and user-approved fold/cycle decisions unchanged.

## 9. Completion audit

After the final production run:

```bash
python src/repo_integrity.py
python -m pytest -q
git ls-files | grep -E '\.(nc|npy|npz|pkl|grib|grib2|pth|pt|log|err)$' && exit 1 || true
git status --short
```

Archive job IDs and runtime metadata outside Git. A run is complete only when
the five-fold leakage audit, matched initialization audit, area/mask metadata,
all three window summaries, and whole-year confidence intervals are present.
