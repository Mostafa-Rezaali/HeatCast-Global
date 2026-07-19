#!/bin/bash
#SBATCH --job-name=hc_global_w34
#SBATCH --output=global_w34_tube_all_%j.log
#SBATCH --error=global_w34_tube_all_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --qos=nessie
#SBATCH --gres=gpu:8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

WORK_DIR=/blue/nessie/mostafarezaali/HeatCastGlobal/
PY=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
TORCHRUN=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/torchrun
LEADS=15,16,17,18,19,20,21,22,23,24,25,26,27,28
RESOLUTION=${RESOLUTION:-1.5deg}
PRECISION=${PRECISION:-fp32}
GRAD_ACCUM=${GRAD_ACCUM:-1}
GRAD_CHECKPOINT=${GRAD_CHECKPOINT:-0}
CLEAN_START=${CLEAN_START:-0}
FOLD_YEARS_JSON=${FOLD_YEARS_JSON:?TODO(USER): path to approved five-fold year JSON}

cd "$WORK_DIR"
module load cuda/12.9.1
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HEATCAST_DOMAIN=global
export HEATCAST_RESOLUTION="$RESOLUTION"
export HEATCAST_FOLD_YEARS_JSON="$FOLD_YEARS_JSON"

test -s "$FOLD_YEARS_JSON"
test -s "$WORK_DIR/cache/era5_${RESOLUTION}.zarr/heatcast_cache_metadata.json"
test -s "$WORK_DIR/cache/teleconnection_5.npy"
test -s "$WORK_DIR/drivers/rmm.txt"

git pull --ff-only origin main
"$PY" -m py_compile src/repo_integrity.py src/cfm_mesh_train.py src/global_dataset.py src/global_targets.py
"$PY" src/repo_integrity.py

MEMORY_ARGS=(--precision "$PRECISION" --grad_accum "$GRAD_ACCUM")
if [ "$GRAD_CHECKPOINT" = "1" ]; then
    MEMORY_ARGS+=(--grad_checkpoint)
fi

fold_training_complete() {
    local run=$1
    local metrics="training_metrics/fold_epoch_metrics_${run}.csv"
    local best_monitor="trained_cfm_direct15_${run}_best_monitor.pth"
    [ -s "$metrics" ] && [ -s "$best_monitor" ] || return 1
    "$PY" -c "import csv,sys; rows=list(csv.DictReader(open(sys.argv[1], newline='', encoding='utf-8'))); last=rows[-1] if rows else {}; ok=last.get('early_stop_metric')=='w34_tac' and (int(float(last.get('early_stop_failures') or 0))>=5 or int(float(last.get('epoch') or 0))>=30); raise SystemExit(0 if ok else 1)" "$metrics"
}

for FOLD in 0 1 2 3 4; do
    RUN=cvfold${FOLD}_global_w34_dist_v1
    test -s "$WORK_DIR/cache/era5_${RESOLUTION}.zarr.sidecars/fold${FOLD}_normalization.npz"
    test -s "$WORK_DIR/cache/era5_${RESOLUTION}.zarr.sidecars/fold${FOLD}_thresholds.npz"
    if [ "$CLEAN_START" = "0" ] && fold_training_complete "$RUN"; then
        echo "Fold $FOLD is complete; skipping $RUN."
        continue
    fi
    TRAIN_MODE=train
    CHECKPOINT_ARGS=()
    RESUME_CKPT=$(ls -1t checkpoints/${RUN}/checkpoint_epoch_*.pth 2>/dev/null | head -n 1 || true)
    if [ "$CLEAN_START" = "0" ] && [ -n "$RESUME_CKPT" ]; then
        TRAIN_MODE=resume
        CHECKPOINT_ARGS=(--checkpoint "$RESUME_CKPT")
    fi
    "$TORCHRUN" --standalone --nproc_per_node=8 src/cfm_mesh_train.py \
      --domain global \
      --resolution "$RESOLUTION" \
      --target_mode climatology_anomaly \
      --fold_years_json "$FOLD_YEARS_JSON" \
      --mode "$TRAIN_MODE" \
      "${CHECKPOINT_ARGS[@]}" \
      --deterministic \
      --cv_fold "$FOLD" \
      --run_name "$RUN" \
      --multi_lead_tube \
      --prediction_leads "$LEADS" \
      --tube_decode_chunk_size 2 \
      --epochs 30 \
      --batch_size 1 \
      --checkpoint_freq 1 \
      --num_validation_samples 928 \
      --learning_rate 5e-5 \
      --dropout 0.15 \
      --warmup_epochs 10 \
      --disable_anomaly_corr_loss \
      --gradient_loss_weight 0.0 \
      --tube_loss_daily_weight 0.80 \
      --tube_loss_center_weight 0.00 \
      --tube_loss_weekly_weight 0.20 \
      --distributional_head \
      --crps_loss \
      --sigma_floor 0.1 \
      --early_stop_metric w34_tac \
      --early_stop_patience 5 \
      --early_stop_min_epoch 12 \
      --early_stop_min_delta 0.0001 \
      "${MEMORY_ARGS[@]}"
    test -s "trained_cfm_direct15_${RUN}_best_monitor.pth"
    fold_training_complete "$RUN"
done

echo "ALL FIVE GLOBAL FOLDS COMPLETE"
EVAL_JOB_ID=$(sbatch --parsable --export=ALL,FOLD_YEARS_JSON="$FOLD_YEARS_JSON",RESOLUTION="$RESOLUTION" slurm/submit_global_w34_eval_stitch.slurm)
echo "Submitted global evaluation/stitch workflow: $EVAL_JOB_ID"
