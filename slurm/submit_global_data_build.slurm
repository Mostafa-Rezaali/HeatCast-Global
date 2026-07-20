#!/bin/bash
#SBATCH --job-name=hc_global_data
#SBATCH --output=global_data_build_%j.log
#SBATCH --error=global_data_build_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

REPO_DIR=/blue/nessie/mostafarezaali/HeatCast-Global
DATA_ROOT=/blue/nessie/mostafarezaali/HeatCast-Global
PY=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
TARGET_SOURCE=${TARGET_SOURCE:-daily_statistics}
RESOLUTION=${RESOLUTION:-1.5deg}
YEARS=${YEARS:-1979-2024}
MONTHS=${MONTHS:-1,2,3,4,5,6,7,8,9,10,11,12}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-4}
BUILD_FOLD_SIDECARS=${BUILD_FOLD_SIDECARS:-1}
DOWNLOAD_ONLY=${DOWNLOAD_ONLY:-0}
ERA5_PRESSURE_DATASET=${ERA5_PRESSURE_DATASET:-reanalysis-era5-pressure-levels}

cd "$REPO_DIR"
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
      --data_root "$DATA_ROOT" \
      --years "$YEARS" \
      --months "$MONTHS" \
      --target_source "$TARGET_SOURCE" \
      --pressure_dataset "$ERA5_PRESSURE_DATASET" \
      --workers "$DOWNLOAD_WORKERS"
)

if [ "$DOWNLOAD_ONLY" = "1" ]; then
    echo "MISSING-ONLY ERA5 DOWNLOAD COMPLETE: $DATA_ROOT"
    exit 0
fi

(
    cd src
    "$PY" -m data_pipeline.build_cache \
      --data_root "$DATA_ROOT" \
      --years "$YEARS" \
      --months "$MONTHS" \
      --resolution "$RESOLUTION" \
      --target_source "$TARGET_SOURCE"
)

test -s "$DATA_ROOT/cache/era5_${RESOLUTION}.zarr/heatcast_cache_metadata.json"
if [ "$BUILD_FOLD_SIDECARS" = "1" ]; then
    FOLD_YEARS_JSON=${FOLD_YEARS_JSON:?TODO(USER): path to approved five-fold year JSON}
    test -s "$FOLD_YEARS_JSON"
    for FOLD in 0 1 2 3 4; do
        "$PY" -u src/build_global_fold_sidecars.py \
          --fold "$FOLD" \
          --fold_years_json "$FOLD_YEARS_JSON" \
          --resolution "$RESOLUTION"
    done
fi
echo "GLOBAL ERA5 CACHE BUILD COMPLETE: $DATA_ROOT/cache/era5_${RESOLUTION}.zarr"
