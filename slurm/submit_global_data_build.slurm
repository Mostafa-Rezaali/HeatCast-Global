#!/bin/bash
#SBATCH --job-name=hc_global_data
#SBATCH --output=global_data_build_%j.log
#SBATCH --error=global_data_build_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --qos=nessie
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

WORK_DIR=/blue/nessie/mostafarezaali/HeatCastGlobal/
PY=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
TARGET_SOURCE=${TARGET_SOURCE:-daily_statistics}
RESOLUTION=${RESOLUTION:-1.5deg}
YEARS=${YEARS:-1979-2024}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-4}
: "${ERA5_PRESSURE_DATASET:?TODO(USER): pin and export the pressure-level CDS dataset identifier}"

cd "$WORK_DIR"
module load cuda/12.9.1
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CDS_CONFIG=${CDSAPI_RC:-$HOME/.cdsapirc}
test -s "$CDS_CONFIG" || {
    echo "Missing CDS credentials at $CDS_CONFIG. Resolve TODO(USER) in docs/DECISIONS_NEEDED.md."
    exit 2
}

git pull --ff-only origin main
"$PY" -m py_compile src/repo_integrity.py src/data_pipeline/download_era5.py src/data_pipeline/regrid.py src/data_pipeline/build_cache.py src/data_pipeline/check_cache.py
"$PY" src/repo_integrity.py

(
    cd src
    "$PY" -m data_pipeline.download_era5 \
      --data_root "$WORK_DIR" \
      --years "$YEARS" \
      --target_source "$TARGET_SOURCE" \
      --pressure_dataset "$ERA5_PRESSURE_DATASET" \
      --workers "$DOWNLOAD_WORKERS"
)

(
    cd src
    "$PY" -m data_pipeline.build_cache \
      --data_root "$WORK_DIR" \
      --years "$YEARS" \
      --resolution "$RESOLUTION" \
      --target_source "$TARGET_SOURCE"
)

test -s "$WORK_DIR/cache/era5_${RESOLUTION}.zarr/heatcast_cache_metadata.json"
echo "GLOBAL ERA5 CACHE BUILD COMPLETE: $WORK_DIR/cache/era5_${RESOLUTION}.zarr"
