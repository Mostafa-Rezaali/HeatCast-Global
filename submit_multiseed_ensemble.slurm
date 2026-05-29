#!/bin/bash
#SBATCH --job-name=direct15_ensemble_all
#SBATCH --output=direct15_ensemble_all_%j.log
#SBATCH --error=direct15_ensemble_all_%j.err
#SBATCH --account=gis4123
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

cd "$WORK_DIR" || exit 1

$PYTHON_EXE -m py_compile cfm_mesh_train.py mesh_backbone.py mode_dispatch.py icosahedral_mesh.py stitch_hindcast_tac.py || exit 1
$PYTHON_EXE -c "import cfm_mesh_train; print('cfm_mesh_train import OK')" || exit 1

export TORCH_NCCL_DUMP_ON_TIMEOUT=0
export TORCH_NCCL_TRACE_BUFFER_SIZE=0
export NCCL_NET_MERGE_LEVEL=LOC
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES

SEEDS=(42 137 256 512 1024)

for SEED in "${SEEDS[@]}"; do
  for FOLD in 0 1 2 3 4; do
    RUN_NAME=cvfold${FOLD}_seed${SEED}

    echo "============================================================"
    echo "Training ensemble member"
    echo "Fold: $FOLD"
    echo "Seed: $SEED"
    echo "Run name: $RUN_NAME"
    echo "============================================================"

    rm -rf /dev/shm/cfm_cache
    rm -rf "${WORK_DIR}/checkpoints/${RUN_NAME}"
    rm -f "${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}.pth"
    rm -f "${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}_best_tac.pth"
    rm -f "${WORK_DIR}/training_metrics/fold_epoch_metrics_${RUN_NAME}.csv"

    $TORCHRUN --standalone --nproc_per_node=8 "$SCRIPT" \
      --mode train \
      --deterministic \
      --cv_fold "$FOLD" \
      --run_name "$RUN_NAME" \
      --seed "$SEED" \
      --epochs 25 \
      --batch_size 1 \
      --checkpoint_freq 1 \
      --num_validation_samples 1088 \
      --learning_rate 5e-5 \
      --dropout 0.30 \
      --warmup_epochs 10 \
      --early_stop_metric tac \
      --early_stop_patience 4 \
      --early_stop_min_epoch 5 \
      --early_stop_min_delta 0.0001

    TRAIN_STATUS=$?
    if [ $TRAIN_STATUS -ne 0 ]; then
      echo "Training failed for fold $FOLD seed $SEED with exit code $TRAIN_STATUS"
      exit $TRAIN_STATUS
    fi

    CKPT=${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}_best_tac.pth
    if [ ! -f "$CKPT" ]; then
      echo "Missing best-TAC checkpoint: $CKPT"
      CKPT=${WORK_DIR}/trained_cfm_direct15_${RUN_NAME}.pth
    fi
    if [ ! -f "$CKPT" ]; then
      echo "Missing checkpoint after training: $CKPT"
      exit 1
    fi

    echo "Export checkpoint: $CKPT"
    rm -rf /dev/shm/cfm_cache

    $TORCHRUN --standalone --nproc_per_node=1 "$SCRIPT" \
      --mode export_hindcast \
      --deterministic \
      --cv_fold "$FOLD" \
      --run_name "$RUN_NAME" \
      --seed "$SEED" \
      --checkpoint "$CKPT" \
      --hindcast_splits test \
      --hindcast_output_dir "${WORK_DIR}/hindcast_stats/ensemble_members" \
      --hindcast_paper_dir "${WORK_DIR}/hindcast_paper_data/ensemble_members"

    EXPORT_STATUS=$?
    if [ $EXPORT_STATUS -ne 0 ]; then
      echo "Export failed for fold $FOLD seed $SEED with exit code $EXPORT_STATUS"
      exit $EXPORT_STATUS
    fi

    echo "Ensemble member complete: fold=$FOLD seed=$SEED"
  done
done

echo "============================================================"
echo "All ensemble members complete."
echo "============================================================"
