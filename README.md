# HeatCast-Global

[![License: MIT](https://img.shields.io/github/license/Mostafa-Rezaali/HeatCast-Global)](LICENSE)
[![CI](https://github.com/Mostafa-Rezaali/HeatCast-Global/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/Mostafa-Rezaali/HeatCast-Global/actions/workflows/python-package.yml)
[![Forecast window](https://img.shields.io/badge/W34-leads%20%2B15...%2B28-6A5ACD)](slurm/submit_w34_tube_all.slurm)
[![Domain](https://img.shields.io/badge/default-global%20ERA5-2E8B57)](#domain-modes)

HeatCast-Global extends the
[HeatCast](https://github.com/Mostafa-Rezaali/HeatCast) week-3--4 system from
CONUS/PRISM to global ERA5. Its primary experiment predicts daily maximum 2 m
temperature anomalies for leads `t+15...t+28` and evaluates week 3, week 4,
and W34 heat extremes over Northern Hemisphere land for MJJAS-valid windows.
ECMWF ENS comparisons use matched initializations and retain HeatCast's
fold-safe calibration and year-block bootstrap design.

The reference architecture remains the **HeatCast-W34 Distributional Mesh
GNN**: direct single-pass inference with a Gaussian mean/sigma head, Gaussian
CRPS training, tube attention, and five year-disjoint folds. The CFM sampling
path remains available for full-field ensemble and tail analyses.

This repository is code-only. ERA5, ECMWF S2S data, zarr stores, checkpoints,
figures, logs, and other runtime artifacts are intentionally excluded.

## Domain modes

All global behavior is introduced behind `Config.DOMAIN`:

| Mode | Target and grid | Status |
|---|---|---|
| `global` (default) | ERA5 daily Tmax; Phase A `1.5deg` (`121 x 240`) | Active development target |
| `conus` | Original PRISM T2max z-score/persistence-residual path | Preserved compatibility path |

Phase B changes configuration to ERA5 `0.25deg` (`721 x 1440`) and a finer
mesh. Grid sizes, cache shapes, and connectivity must be derived from config;
Phase B is not a separate model implementation.

## Protected scientific contracts

- Prediction leads remain exactly `15,16,...,28`.
- Tube loss remains `0.80 * daily CRPS + 0.20 * 14-day-mean CRPS`.
- The distributional head emits a per-pixel mean and positive sigma with the
  existing `0.1` normalized-unit floor.
- Cross-validation, calibration, threshold construction, stitching, and
  bootstrap roles remain fold-disjoint.
- Global headline metrics are area weighted, land masked, restricted to the
  Northern Hemisphere, and use initializations whose full W34 window is in
  MJJAS.
- ECMWF ENS comparisons use identical initialization dates and spatial masks.
- Datasets read samples lazily; training never materializes complete arrays in
  parent-process RAM.

## Resolution plan

| Phase | Resolution | Grid | Default mesh refinement | Purpose |
|---|---:|---:|---:|---|
| A | `1.5deg` | `121 x 240` | 5 | Development and first production results |
| B | `0.25deg` | `721 x 1440` | 6 | Config-only high-resolution production path |

## Implementation phases

| Phase | Deliverable | State |
|---:|---|---|
| 0 | Duplicate/rebrand, artifact policy, baseline CI | Complete |
| 1 | Domain config, calendar, area weighting, window math | Planned |
| 2 | ERA5 download/regrid/zarr cache and data-build Slurm | Planned |
| 3 | Global spherical mesh/connectivity and memory flags | Planned |
| 4 | Fold-safe anomaly target, global inputs, CPU smoke test | Planned |
| 5 | Weighted week3/week4/W34 evaluation and export | Planned |
| 6 | Global ENS path and novelty analyses | Planned |
| 7 | Production Slurm workflows, runbook, Phase B checklist | Planned |

Each phase is committed only after the data-free integrity audit and pytest
suite pass. Scientific choices that cannot be inferred are marked
`TODO(USER)` and collected in [docs/DECISIONS_NEEDED.md](docs/DECISIONS_NEEDED.md).

## Quick start

```bash
git clone https://github.com/Mostafa-Rezaali/HeatCast-Global.git
cd HeatCast-Global
python src/repo_integrity.py
python -m pytest -q
```

The original repository remains configured as the `upstream` remote in the
development clone so compatible HeatCast fixes can be cherry-picked.

## Repository layout

```text
src/                    Model, evaluation, ENS, analysis, and data-pipeline code
slurm/                  HiPerGator production workflows
tests/                  Data-free regression and scientific-contract tests
docs/                   Inputs, decisions, integrity guidance, and runbook
matlab/                 Grid-parameterized visualization utilities
```

Important existing modules are preserved rather than rebuilt:

| File | Role |
|---|---|
| `src/cfm_mesh_train.py` | Configuration, data setup, five-fold training, validation, checkpointing |
| `src/mesh_backbone.py` | Grid/mesh encoder, multimesh processor, decoder, tube attention, distributional head |
| `src/mode_dispatch.py` | Deterministic-direct and CFM dispatch plus Gaussian CRPS |
| `src/exceedance_eval.py` | Fold-safe percentile thresholds, calibration, and exceedance evaluation |
| `src/stitch_exceedance_folds.py` | Cross-fitted stitching and year-block bootstrap |
| `src/ens_ingest.py` | Resume-safe ECMWF S2S ingestion and target-grid regridding |
| `src/ens_score.py` | Cycle-aware quantile mapping, fold-safe calibration, and scoring |
| `src/repo_integrity.py` | Fast data-free repository and experiment-contract audit |

The authoritative predictor inventory is
[docs/MODEL_INPUTS.md](docs/MODEL_INPUTS.md). Operational instructions will be
maintained in `docs/RUNBOOK.md` after the production workflows are complete.

## Target environment

Production workflows target UF HiPerGator B200 nodes with `cuda/12.9.1` and
the Python/torchrun executables under
`/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/`. The single global
data root is `/blue/nessie/mostafarezaali/HeatCastGlobal/`; code and Slurm
workflows obtain it from `Config`.

## Verification

Run before every commit and production submission:

```bash
python src/repo_integrity.py
python -m pytest -q
```

CI stays data-free and never downloads ERA5 or ECMWF data. A single-process
CPU synthetic smoke test is added as part of the global training path.

## License

HeatCast-Global is released under the [MIT License](LICENSE).
