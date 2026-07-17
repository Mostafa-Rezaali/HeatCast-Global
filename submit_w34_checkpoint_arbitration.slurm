#!/bin/bash
#SBATCH --job-name=w34_ckpt_arb
#SBATCH --output=w34_ckpt_arb_%j.log
#SBATCH --error=w34_ckpt_arb_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --qos=nessie
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

WORK_DIR=/blue/nessie/mostafarezaali/Teleconnection
PY=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
TORCHRUN=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/torchrun
LEADS=15,16,17,18,19,20,21,22,23,24,25,26,27,28
RUN=cvfold0_w34_dist_v1
ARB_ROOT=w34_checkpoint_arbitration

cd "$WORK_DIR"
module load cuda/12.9.1
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

git pull --ff-only origin main
"$PY" -m py_compile cfm_mesh_train.py exceedance_eval.py w34_checkpoint_arbitration.py

for SELECTOR in best_tac best_monitor; do
    CKPT=trained_cfm_direct15_${RUN}_${SELECTOR}.pth
    test -f "$CKPT"
    ARB_RUN=cvfold0_w34_arb_${SELECTOR}

    "$TORCHRUN" --standalone --nproc_per_node=1 cfm_mesh_train.py \
      --mode export_hindcast \
      --deterministic \
      --cv_fold 0 \
      --run_name "$ARB_RUN" \
      --multi_lead_tube \
      --prediction_leads "$LEADS" \
      --distributional_head \
      --crps_loss \
      --sigma_floor 0.1 \
      --checkpoint "$CKPT" \
      --hindcast_splits val

    "$PY" -u exceedance_eval.py \
      --run_name "$RUN" \
      --cv_fold 0 \
      --checkpoint "$CKPT" \
      --prediction_leads "$LEADS" \
      --target_mode window \
      --window_leads "$LEADS" \
      --use_model_sigma \
      --calibration_split train \
      --eval_split val \
      --calibrator platt \
      --output_dir "${ARB_ROOT}/${SELECTOR}"
done

"$PY" -u w34_checkpoint_arbitration.py \
  --best_tac_stats hindcast_stats/hindcast_tac_stats_cvfold0_w34_arb_best_tac_val.npz \
  --best_monitor_stats hindcast_stats/hindcast_tac_stats_cvfold0_w34_arb_best_monitor_val.npz \
  --best_tac_results "${ARB_ROOT}/best_tac/${RUN}/val/window_15-16-17-18-19-20-21-22-23-24-25-26-27-28/exceedance_results.csv" \
  --best_monitor_results "${ARB_ROOT}/best_monitor/${RUN}/val/window_15-16-17-18-19-20-21-22-23-24-25-26-27-28/exceedance_results.csv" \
  --output_choice w34_checkpoint_choice.txt \
  --output_csv w34_checkpoint_arbitration.csv
