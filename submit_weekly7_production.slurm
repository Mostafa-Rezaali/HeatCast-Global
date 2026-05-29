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

echo "Cleaning old ${RUN_SUFFIX} artifacts only..."
rm -f ${WORK_DIR}/trained_cfm_direct15_cvfold*_${RUN_SUFFIX}.pth
rm -f ${WORK_DIR}/trained_cfm_direct15_cvfold*_${RUN_SUFFIX}_best_tac.pth
rm -f ${WORK_DIR}/trained_cfm_direct15_cvfold*_${RUN_SUFFIX}_best_monitor.pth
rm -rf ${WORK_DIR}/checkpoints/cvfold*_${RUN_SUFFIX}
rm -f ${WORK_DIR}/hindcast_stats/hindcast_tac_stats_cvfold*_${RUN_SUFFIX}_test.npz
rm -f ${WORK_DIR}/hindcast_stats/stitched_5fold_tac_${RUN_SUFFIX}.npz
rm -f ${WORK_DIR}/training_metrics/fold_epoch_metrics_cvfold*_${RUN_SUFFIX}.csv
rm -rf ${WORK_DIR}/test_prediction_plots/cvfold*_${RUN_SUFFIX}
rm -f ${WORK_DIR}/hindcast_paper_data/hindcast_*_cvfold*_${RUN_SUFFIX}_test.npz
rm -rf /dev/shm/cfm_cache
echo "  Done"

for FOLD in 0 1 2 3 4; do
    RUN_NAME=cvfold${FOLD}_${RUN_SUFFIX}

    echo "============================================================"
    echo "Training fold $FOLD"
    echo "Run name: $RUN_NAME"
    echo "Primary monitor: true weekly7 TAC"
    echo "============================================================"

    rm -rf /dev/shm/cfm_cache

    $TORCHRUN --standalone --nproc_per_node=8 "$SCRIPT" \
      --mode train \
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

$PYTHON_EXE -u stitch_hindcast_tac.py \
  hindcast_stats/hindcast_tac_stats_cvfold*_${RUN_SUFFIX}_test.npz \
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
  hindcast_stats/hindcast_tac_stats_cvfold*_${RUN_SUFFIX}_test.npz \
  --metric weekly7 \
  --output_dir paper_figures/${RUN_SUFFIX}

$PYTHON_EXE -u summarize_hindcast_diagnostics.py \
  hindcast_stats/hindcast_tac_stats_cvfold*_${RUN_SUFFIX}_test.npz \
  --monthly_glob "hindcast_paper_data/hindcast_monthly_stats_cvfold*_${RUN_SUFFIX}_test.npz" \
  --region_data paper_figures/${RUN_SUFFIX}/tac_skill_maps_data.npz

$PYTHON_EXE -u compute_baselines.py \
  hindcast_stats/hindcast_tac_stats_cvfold*_${RUN_SUFFIX}_test.npz \
  --paper_dir hindcast_paper_data \
  --output_dir paper_figures/${RUN_SUFFIX}

echo "============================================================"
echo "ALL FOLDS COMPLETE"
echo "============================================================"
