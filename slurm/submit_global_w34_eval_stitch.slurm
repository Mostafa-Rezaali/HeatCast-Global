#!/bin/bash
#SBATCH --job-name=hc_global_eval
#SBATCH --output=global_w34_eval_stitch_%j.log
#SBATCH --error=global_w34_eval_stitch_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --qos=nessie
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

WORK_DIR=/blue/nessie/mostafarezaali/HeatCast-Global/
PY=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
RESOLUTION=${RESOLUTION:-1.5deg}
FOLD_YEARS_JSON=${FOLD_YEARS_JSON:?TODO(USER): path to approved five-fold year JSON}
RUNS=cvfold0_global_w34_dist_v1,cvfold1_global_w34_dist_v1,cvfold2_global_w34_dist_v1,cvfold3_global_w34_dist_v1,cvfold4_global_w34_dist_v1
FULL_LEADS=15,16,17,18,19,20,21,22,23,24,25,26,27,28
WINDOWS=(
  15,16,17,18,19,20,21
  22,23,24,25,26,27,28
  15,16,17,18,19,20,21,22,23,24,25,26,27,28
)
INPUT_ROOT=exceedance_eval_incremental
DRIVER_DIR=drivers/fold_safe_tables
FORCE_EVAL=${FORCE_EVAL:-0}

cd "$WORK_DIR"
module load cuda/12.9.1
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HEATCAST_DOMAIN=global
export HEATCAST_RESOLUTION="$RESOLUTION"
export HEATCAST_FOLD_YEARS_JSON="$FOLD_YEARS_JSON"

test -s "$FOLD_YEARS_JSON"
test -s "$WORK_DIR/drivers/rmm.txt"
test -s "$WORK_DIR/drivers/nino34.txt"

git pull --ff-only origin main
"$PY" -m py_compile src/repo_integrity.py src/cfm_mesh_train.py src/exceedance_eval.py src/stitch_exceedance_folds.py
"$PY" src/repo_integrity.py

for FOLD in 0 1 2 3 4; do
    RUN=cvfold${FOLD}_global_w34_dist_v1
    test -s "trained_cfm_direct15_${RUN}_best_monitor.pth"
    "$PY" -u src/cfm_mesh_train.py \
      --domain global \
      --resolution "$RESOLUTION" \
      --target_mode climatology_anomaly \
      --fold_years_json "$FOLD_YEARS_JSON" \
      --mode export_hindcast \
      --deterministic \
      --cv_fold "$FOLD" \
      --run_name "$RUN" \
      --multi_lead_tube \
      --prediction_leads "$FULL_LEADS" \
      --distributional_head \
      --crps_loss \
      --hindcast_splits val,test

    for LEADS in "${WINDOWS[@]}"; do
        LABEL=${LEADS//,/-}
        MANIFEST="$INPUT_ROOT/$RUN/test/window_${LABEL}/incremental_arrays/manifest.npz"
        if [ "$FORCE_EVAL" = "0" ] && [ -s "$MANIFEST" ]; then
            echo "Skipping complete HeatCast evaluation fold=$FOLD window=$LABEL"
            continue
        fi
        "$PY" -u src/exceedance_eval.py \
          --run_name "$RUN" \
          --cv_fold "$FOLD" \
          --checkpoint best_monitor \
          --multi_lead_tube \
          --prediction_leads "$FULL_LEADS" \
          --tube_decode_chunk_size 2 \
          --target_mode window \
          --window_leads "$LEADS" \
          --use_model_sigma \
          --calibration_split val \
          --eval_split test \
          --calibrator platt \
          --incremental_skill_diagnostic \
          --save_incremental_arrays \
          --year_bootstrap_reps 1000 \
          --output_dir "$INPUT_ROOT"
    done
done

for LEADS in "${WINDOWS[@]}"; do
    "$PY" -u src/stitch_exceedance_folds.py \
      --runs "$RUNS" \
      --window_leads "$LEADS" \
      --calibrator platt \
      --fit_mode cross_fitted \
      --input_root "$INPUT_ROOT" \
      --bootstrap_reps 5000
done

"$PY" -u src/build_driver_tables.py \
  --domain global \
  --resolution "$RESOLUTION" \
  --rmm_path "$WORK_DIR/drivers/rmm.txt" \
  --nino34_path "$WORK_DIR/drivers/nino34.txt" \
  --input_root "$INPUT_ROOT" \
  --run_names "$RUNS" \
  --window_leads "$FULL_LEADS" \
  --output_dir "$DRIVER_DIR"

"$PY" -u src/forecasts_of_opportunity.py \
  --domain global \
  --resolution "$RESOLUTION" \
  --input_root "$INPUT_ROOT" \
  --run_names "$RUNS" \
  --window_leads "$FULL_LEADS" \
  --driver_table_dir "$DRIVER_DIR" \
  --n_bootstrap 2000

echo "GLOBAL WEEK3, WEEK4, AND W34 EXPORT, STITCH, AND OPPORTUNITY VERIFICATION COMPLETE"
