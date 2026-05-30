#!/bin/bash
#SBATCH --job-name=weekly7_cv_all
#SBATCH --output=weekly7_cv_all_%j.log
#SBATCH --error=weekly7_cv_all_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=3-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --qos=nessie
#SBATCH --gres=gpu:8
#SBATCH --exclude=c1003a-s15
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

module load cuda/12.9.1

unset PYTHONHOME
unset PYTHONPATH

TORCHRUN=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/torchrun
PYTHON_EXE=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python

WORK_DIR=/blue/nessie/mostafarezaali/Teleconnection
SCRIPT=${WORK_DIR}/cfm_mesh_train.py

cd $WORK_DIR || exit 1

$PYTHON_EXE -m py_compile \
  cfm_mesh_train.py mesh_backbone.py mode_dispatch.py icosahedral_mesh.py \
  stitch_hindcast_tac.py summarize_hindcast_diagnostics.py tac_skill_maps.py \
  compute_baselines.py audit_fold2_light.py publication_analysis_utils.py || exit 1
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

RUN_SUFFIX=pub_weekly7
FOLDS=${FOLDS:-"0 1 2 3 4"}
CLEAN_START=${CLEAN_START:-0}

if [ "$CLEAN_START" = "1" ]; then
    echo "CLEAN_START=1: cleaning old ${RUN_SUFFIX} artifacts for folds: ${FOLDS}"
    for FOLD in $FOLDS; do
        RUN_NAME=cvfold${FOLD}_${RUN_SUFFIX}
        rm -f ${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}.pth
        rm -f ${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}_best_tac.pth
        rm -f ${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}_best_monitor.pth
        rm -rf ${WORK_DIR}/checkpoints/${RUN_NAME}
        rm -f ${WORK_DIR}/hindcast_stats/hindcast_tac_stats_${RUN_NAME}_test.npz
        rm -f ${WORK_DIR}/training_metrics/fold_epoch_metrics_${RUN_NAME}.csv
        rm -rf ${WORK_DIR}/test_prediction_plots/${RUN_NAME}
        rm -f ${WORK_DIR}/hindcast_paper_data/hindcast_*_${RUN_NAME}_test.npz
    done
    rm -f ${WORK_DIR}/hindcast_stats/stitched_5fold_tac_${RUN_SUFFIX}.npz
else
    echo "CLEAN_START=0: preserving existing ${RUN_SUFFIX} checkpoints/exports for resume."
fi
rm -rf /dev/shm/cfm_cache
echo "  Done"

for FOLD in $FOLDS; do
    RUN_NAME=cvfold${FOLD}_${RUN_SUFFIX}
    STAT_FILE=${WORK_DIR}/hindcast_stats/hindcast_tac_stats_${RUN_NAME}_test.npz

    if [ -f "$STAT_FILE" ]; then
        echo "============================================================"
        echo "Fold $FOLD already has export stats: $STAT_FILE"
        echo "Skipping training/export for $RUN_NAME"
        echo "============================================================"
        continue
    fi

    echo "============================================================"
    echo "Training fold $FOLD"
    echo "Run name: $RUN_NAME"
    echo "Primary monitor: true weekly7 TAC"
    echo "============================================================"

    rm -rf /dev/shm/cfm_cache

    TRAIN_MODE=train
    CHECKPOINT_ARG=""
    RESUME_CKPT=$(ls -1t ${WORK_DIR}/checkpoints/${RUN_NAME}/checkpoint_epoch_*.pth 2>/dev/null | head -n 1 || true)
    if [ -z "$RESUME_CKPT" ]; then
        for CANDIDATE in \
            ${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}_best_monitor.pth \
            ${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}_best_tac.pth \
            ${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}.pth
        do
            if [ -f "$CANDIDATE" ]; then
                RESUME_CKPT=$CANDIDATE
                break
            fi
        done
    fi
    if [ -n "$RESUME_CKPT" ]; then
        TRAIN_MODE=resume
        CHECKPOINT_ARG="--checkpoint $RESUME_CKPT"
        echo "Resuming fold $FOLD from checkpoint: $RESUME_CKPT"
    else
        echo "No existing checkpoint found for fold $FOLD; starting fresh."
    fi

    $TORCHRUN --standalone --nproc_per_node=8 "$SCRIPT" \
      --mode "$TRAIN_MODE" \
      $CHECKPOINT_ARG \
      --deterministic \
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
      --early_stop_metric weekly7_tac \
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
        echo "Falling back to best daily-TAC checkpoint."
        CKPT=${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}_best_tac.pth
    fi
    if [ ! -f "$CKPT" ]; then
        echo "Missing best daily-TAC checkpoint. Falling back to best-R2 checkpoint."
        CKPT=${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}.pth
    fi
    if [ ! -f "$CKPT" ]; then
        echo "Missing checkpoint after training: $CKPT"
        exit 1
    fi

    echo "============================================================"
    echo "Starting export for fold $FOLD"
    echo "Export checkpoint: $CKPT"
    echo "============================================================"

    rm -rf /dev/shm/cfm_cache

    $TORCHRUN --standalone --nproc_per_node=1 "$SCRIPT" \
      --mode export_hindcast \
      --deterministic \
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

echo "============================================================"
echo "All folds trained/exported. Stitching 5-fold TAC."
echo "============================================================"

STAT_FILES=""
for FOLD in $FOLDS; do
    FILE=hindcast_stats/hindcast_tac_stats_cvfold${FOLD}_${RUN_SUFFIX}_test.npz
    if [ -f "$FILE" ]; then
        STAT_FILES="$STAT_FILES $FILE"
    else
        echo "Missing fold stat file: $FILE"
        exit 1
    fi
done

$PYTHON_EXE -u stitch_hindcast_tac.py \
  $STAT_FILES \
  --output hindcast_stats/stitched_5fold_tac_${RUN_SUFFIX}.npz

STITCH_STATUS=$?
if [ $STITCH_STATUS -ne 0 ]; then
    echo "Stitching failed with exit code $STITCH_STATUS"
    exit $STITCH_STATUS
fi

echo "============================================================"
echo "Weekly7 diagnostics"
echo "============================================================"

$PYTHON_EXE -u tac_skill_maps.py \
  $STAT_FILES \
  --metric weekly7 \
  --output_dir paper_figures/${RUN_SUFFIX}

$PYTHON_EXE -u summarize_hindcast_diagnostics.py \
  $STAT_FILES \
  --monthly_glob "hindcast_paper_data/hindcast_monthly_stats_cvfold*_${RUN_SUFFIX}_test.npz" \
  --region_data paper_figures/${RUN_SUFFIX}/tac_skill_maps_data.npz

$PYTHON_EXE -u compute_baselines.py \
  $STAT_FILES \
  --paper_dir hindcast_paper_data \
  --output_dir paper_figures/${RUN_SUFFIX}

echo "============================================================"
echo "ALL FOLDS COMPLETE"
echo "============================================================"
