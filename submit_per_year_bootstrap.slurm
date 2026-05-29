#!/bin/bash
#SBATCH --job-name=per_year_bootstrap
#SBATCH --output=per_year_bootstrap_%j.log
#SBATCH --error=per_year_bootstrap_%j.err
#SBATCH --account=gis4123
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --partition=hpg-b200
#SBATCH --qos=nessie
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

module load cuda/12.9.1

unset PYTHONHOME
unset PYTHONPATH

PYTHON_EXE=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
WORK_DIR=/blue/nessie/mostafarezaali/Teleconnection

cd "$WORK_DIR" || exit 1

$PYTHON_EXE -m py_compile \
  cfm_mesh_train.py mesh_backbone.py mode_dispatch.py icosahedral_mesh.py \
  publication_analysis_utils.py export_per_year_stats.py bootstrap_significance.py || exit 1

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES

rm -rf /dev/shm/cfm_cache
RUN_SUFFIX=pub_weekly7

echo "============================================================"
echo "Exporting per-year daily and true-weekly7 TAC sufficient statistics"
echo "============================================================"

$PYTHON_EXE -u export_per_year_stats.py \
  --run_name_pattern cvfold{fold}_${RUN_SUFFIX} \
  --checkpoint_pattern trained_cfm_direct15_{run_name}_best_monitor.pth \
  --output hindcast_stats/per_year_tac_stats_${RUN_SUFFIX}.npz \
  --device cuda

EXPORT_STATUS=$?
if [ $EXPORT_STATUS -ne 0 ]; then
  echo "Per-year export failed with exit code $EXPORT_STATUS"
  exit $EXPORT_STATUS
fi

echo "============================================================"
echo "Running year-block bootstrap significance"
echo "============================================================"

$PYTHON_EXE -u bootstrap_significance.py \
  --input hindcast_stats/per_year_tac_stats_${RUN_SUFFIX}.npz \
  --output_dir paper_figures/${RUN_SUFFIX} \
  --output weekly7_bootstrap_results.npz \
  --metric weekly7 \
  --n_bootstrap 10000 \
  --pixel_stride 4

BOOT_STATUS=$?
if [ $BOOT_STATUS -ne 0 ]; then
  echo "Bootstrap failed with exit code $BOOT_STATUS"
  exit $BOOT_STATUS
fi

echo "============================================================"
echo "Per-year export and bootstrap complete."
echo "============================================================"
