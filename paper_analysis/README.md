# Paper Analysis Scripts

Run these after the new 5-fold anomaly + lagged-global jobs finish.

## What can be made from which file

- `stitched_5fold_tac.npz`
  - Spatial TAC maps
  - TAC significance tests
  - Regional TAC tables/profiles

- `hindcast_paper_data/hindcast_sample_summary_*_test.npz`
  - Scatter plots
  - Seasonal/monthly summary plots
  - Per-sample metric tables

- `hindcast_paper_data/hindcast_selected_maps_*_test.npz`
  - Sample prediction maps
  - Extreme heat examples
  - Error distribution and bias/RMSE maps for selected examples

- `paper_inputs/fold_epoch_metrics.csv`
  - Training curves
  - Metrics summary bars
  - Fold consistency tables

## Recommended run order

```bash
python paper_analysis/generate_architecture_diagram.py
python paper_analysis/generate_model_tables.py

python paper_analysis/generate_stitched_maps_tables.py
python paper_analysis/generate_training_figures_tables.py
python paper_analysis/generate_sample_prediction_figures.py
```

The current hindcast export in `cfm_mesh_train.py` now writes the paper sidecar
files by default during `--mode export_hindcast`. To disable that behavior:

```bash
--no_paper_export
```

To increase or reduce selected sample map output size:

```bash
--paper_maps_per_month 8 --paper_heat_maps 12
```

