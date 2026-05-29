#!/bin/bash
#SBATCH --job-name=mesh_gnn_train
#SBATCH --output=mesh_gnn_train_%j.log
#SBATCH --error=mesh_gnn_train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=1-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --qos=nessie
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

module load cuda/12.9.1

TORCHRUN=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/torchrun
PYTHON_EXE=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python

WORK_DIR=/blue/nessie/mostafarezaali/Teleconnection

SCRIPT=${WORK_DIR}/cfm_mesh_train_small_domain.py
CLIM_SCRIPT=${WORK_DIR}/compute_climatology.py
BASELINE_SCRIPT=${WORK_DIR}/compute_baselines.py
MESH_MODULE=${WORK_DIR}/icosahedral_mesh.py
BACKBONE_MODULE=${WORK_DIR}/mesh_backbone.py
DISPATCH_MODULE=${WORK_DIR}/mode_dispatch.py

CLIM_FILE=${WORK_DIR}/data_cache/climatology_daily_1981_2015_sub30_40_-105_-90.npz
OLD_NORM_STATS=${WORK_DIR}/data_cache/norm_stats_v2_sub30_40_-105_-90.npz
ANOM_NORM_STATS=${WORK_DIR}/data_cache/norm_stats_anomaly_v1_sub30_40_-105_-90.npz
RAW_CACHE=${WORK_DIR}/data_cache_sub30_40_-105_-90

cd "${WORK_DIR}"

echo "Using Python at: ${PYTHON_EXE}"
"${PYTHON_EXE}" --version
echo "Using torchrun at: ${TORCHRUN}"
echo "Working directory: ${WORK_DIR}"

# ==============================================================================
# REQUIRED PRE-STEPS
# ==============================================================================
if [ ! -f "${CLIM_FILE}" ]; then
    echo "ERROR: Missing climatology file:"
    echo "  ${CLIM_FILE}"
    echo "Run this first:"
    echo "  python3 -u compute_climatology.py"
    exit 1
fi

# ==============================================================================
# CACHE CLEANUP
# ==============================================================================
echo "Cleaning stale shared-memory and old normalization caches..."
rm -rf /dev/shm/cfm_cache*
rm -f "${OLD_NORM_STATS}"

# Keep the expensive raw data cache by default. Set FORCE_REBUILD_DATA_CACHE=1
# only when the source NetCDF or subdomain slice cache is known stale/corrupt.
if [ "${FORCE_REBUILD_DATA_CACHE:-0}" = "1" ]; then
    echo "FORCE_REBUILD_DATA_CACHE=1, removing raw cache: ${RAW_CACHE}"
    rm -rf "${RAW_CACHE}"
    rm -f "${ANOM_NORM_STATS}"
fi

# ==============================================================================
# AUTO-CLEANUP: Remove non-breaking spaces and convert tabs to 4 spaces
# ==============================================================================
echo "Sanitizing scripts for hidden characters and formatting..."
for f in "${SCRIPT}" "${CLIM_SCRIPT}" "${BASELINE_SCRIPT}" "${MESH_MODULE}" "${BACKBONE_MODULE}" "${DISPATCH_MODULE}"; do
    sed -i 's/\xc2\xa0/ /g' "${f}"
    expand -t 4 "${f}" > "${f}.tmp" && mv "${f}.tmp" "${f}"
done

# ==============================================================================
# FAIL FAST ON SYNTAX ERRORS
# ==============================================================================
echo "Running syntax checks..."
for f in "${SCRIPT}" "${CLIM_SCRIPT}" "${BASELINE_SCRIPT}" "${MESH_MODULE}" "${BACKBONE_MODULE}" "${DISPATCH_MODULE}"; do
    "${PYTHON_EXE}" -c "import py_compile; py_compile.compile('${f}', doraise=True)" || {
        echo "SYNTAX ERROR in ${f}"
        exit 1
    }
    echo "  $(basename "${f}"): OK"
done

nvidia-smi

export TORCH_NCCL_DUMP_ON_TIMEOUT=0
export TORCH_NCCL_TRACE_BUFFER_SIZE=0
export NCCL_NET_MERGE_LEVEL=LOC
export NCCL_IB_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset CUDA_VISIBLE_DEVICES

# ==============================================================================
# STAGE 1: deterministic direct 15-day anomaly prediction
# ==============================================================================
MODE_FLAG="--deterministic"
MODE_NAME="Stage 1 deterministic climatology-anomaly"

# Stage 1 memory-safe mesh defaults for 1x B200.
# Override at submit time only if you intentionally want a different size:
#   MESH_FLAGS="--mesh_level 8 --mesh_rounds 4" sbatch submit_mesh_train.sh
MESH_FLAGS="${MESH_FLAGS:---mesh_level 7 --mesh_rounds 4}"

echo "Starting training..."
echo "Mode: ${MODE_NAME}"
echo "Climatology: ${CLIM_FILE}"
echo "Mesh flags: ${MESH_FLAGS:-<code defaults>}"

"${TORCHRUN}" --standalone --nproc_per_node=1 "${SCRIPT}" --mode train ${MODE_FLAG} ${MESH_FLAGS}

# --- RESUME from checkpoint ---
# "${TORCHRUN}" --standalone --nproc_per_node=1 "${SCRIPT}" --mode resume ${MODE_FLAG} ${MESH_FLAGS} \
#     --checkpoint "${WORK_DIR}/trained_cfm_improved.pth"
