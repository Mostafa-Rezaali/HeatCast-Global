#!/bin/bash
#SBATCH --job-name=tube7_v1
#SBATCH --output=tube7_v1_%j.log
#SBATCH --error=tube7_v1_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --qos=nessie
#SBATCH --gres=gpu:8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

module load cuda/12.9.1

unset PYTHONHOME
unset PYTHONPATH

TORCHRUN=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/torchrun
PYTHON_EXE=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python

WORK_DIR=/blue/nessie/mostafarezaali/Teleconnection
SCRIPT=${WORK_DIR}/cfm_mesh_train.py
RUN_SUFFIX=tube7_v1
FOLDS=${FOLDS:-"2"}

cd "$WORK_DIR" || exit 1

$PYTHON_EXE -m py_compile \
  cfm_mesh_train.py mesh_backbone.py mode_dispatch.py icosahedral_mesh.py \
  stitch_hindcast_tac.py tac_skill_maps.py summarize_hindcast_diagnostics.py \
  compute_baselines.py export_per_year_stats.py bootstrap_significance.py || exit 1
$PYTHON_EXE -c "import cfm_mesh_train; print('cfm_mesh_train import OK')" || exit 1

nvidia-smi

export TORCH_NCCL_DUMP_ON_TIMEOUT=0
export TORCH_NCCL_TRACE_BUFFER_SIZE=0
export NCCL_NET_MERGE_LEVEL=LOC
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES

rm -rf /dev/shm/cfm_cache

for FOLD in $FOLDS; do
    RUN_NAME=cvfold${FOLD}_${RUN_SUFFIX}

    echo "============================================================"
    echo "Training tube diagnostic fold $FOLD"
    echo "Run name: $RUN_NAME"
    echo "Prediction leads: 12,13,14,15,16,17,18"
    echo "============================================================"

    rm -rf /dev/shm/cfm_cache
    rm -f ${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}.pth
    rm -f ${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}_best_tac.pth
    rm -f ${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}_best_monitor.pth
    rm -rf ${WORK_DIR}/checkpoints/${RUN_NAME}
    rm -rf ${WORK_DIR}/test_prediction_plots/${RUN_NAME}
    rm -f ${WORK_DIR}/hindcast_stats/hindcast_tac_stats_${RUN_NAME}_test.npz
    rm -f ${WORK_DIR}/training_metrics/fold_epoch_metrics_${RUN_NAME}.csv
    rm -f ${WORK_DIR}/data_cache/norm_stats_direct15_tube12-13-14-15-16-17-18_*.npz
    rm -f ${WORK_DIR}/data_cache/local_daily_climo30_direct15_tube12-13-14-15-16-17-18_*.npy

    $TORCHRUN --standalone --nproc_per_node=8 "$SCRIPT" \
      --mode train \
      --deterministic \
      --multi_lead_tube \
      --prediction_leads 12,13,14,15,16,17,18 \
      --cv_fold "$FOLD" \
      --run_name "$RUN_NAME" \
      --epochs 30 \
      --batch_size 1 \
      --checkpoint_freq 1 \
      --num_validation_samples 1088 \
      --learning_rate 5e-5 \
      --dropout 0.15 \
      --warmup_epochs 10 \
      --disable_anomaly_corr_loss \
      --early_stop_metric tube_weekly7_tac \
      --early_stop_patience 5 \
      --early_stop_min_epoch 12 \
      --early_stop_min_delta 0.0001

    TRAIN_STATUS=$?
    if [ $TRAIN_STATUS -ne 0 ]; then
        echo "Training failed for fold $FOLD with exit code $TRAIN_STATUS"
        exit $TRAIN_STATUS
    fi

    CKPT=${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}_best_monitor.pth
    if [ ! -f "$CKPT" ]; then
        echo "Missing best-monitor checkpoint: $CKPT"
        echo "Falling back to best-TAC checkpoint."
        CKPT=${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}_best_tac.pth
    fi
    if [ ! -f "$CKPT" ]; then
        echo "Missing best-TAC checkpoint; falling back to best-R2 checkpoint."
        CKPT=${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}.pth
    fi
    if [ ! -f "$CKPT" ]; then
        echo "Missing checkpoint after training: $CKPT"
        exit 1
    fi

    echo "============================================================"
    echo "Exporting tube fold $FOLD"
    echo "Export checkpoint: $CKPT"
    echo "============================================================"

    rm -rf /dev/shm/cfm_cache

    $TORCHRUN --standalone --nproc_per_node=1 "$SCRIPT" \
      --mode export_hindcast \
      --deterministic \
      --multi_lead_tube \
      --prediction_leads 12,13,14,15,16,17,18 \
      --cv_fold "$FOLD" \
      --run_name "$RUN_NAME" \
      --checkpoint "$CKPT" \
      --hindcast_splits test

    EXPORT_STATUS=$?
    if [ $EXPORT_STATUS -ne 0 ]; then
        echo "Export failed for fold $FOLD with exit code $EXPORT_STATUS"
        exit $EXPORT_STATUS
    fi
done

STAT_FILES=""
for FOLD in $FOLDS; do
    FILE=hindcast_stats/hindcast_tac_stats_cvfold${FOLD}_${RUN_SUFFIX}_test.npz
    if [ -f "$FILE" ]; then
        STAT_FILES="$STAT_FILES $FILE"
    fi
done
if [ -n "$STAT_FILES" ]; then
    echo "============================================================"
    echo "Stitching available tube folds"
    echo "============================================================"
    $PYTHON_EXE -u stitch_hindcast_tac.py \
      $STAT_FILES \
      --output hindcast_stats/stitched_tac_${RUN_SUFFIX}.npz || exit 1

    echo "============================================================"
    echo "Tube weekly7 diagnostics"
    echo "============================================================"
    $PYTHON_EXE -u tac_skill_maps.py \
      $STAT_FILES \
      --metric weekly7 \
      --output_dir paper_figures/${RUN_SUFFIX} || exit 1
    $PYTHON_EXE -u summarize_hindcast_diagnostics.py \
      $STAT_FILES \
      --monthly_glob "hindcast_paper_data/hindcast_monthly_stats_cvfold*_${RUN_SUFFIX}_test.npz" \
      --region_data paper_figures/${RUN_SUFFIX}/tac_skill_maps_data.npz || exit 1
    $PYTHON_EXE -u compute_baselines.py \
      $STAT_FILES \
      --output_dir paper_figures/${RUN_SUFFIX} || exit 1
fi

echo "============================================================"
echo "TUBE7 V1 COMPLETE"
echo "============================================================"
