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
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-8}
DOWNLOAD_CHUNKING=${DOWNLOAD_CHUNKING:-yearly}
DOWNLOAD_PER_DATASET=${DOWNLOAD_PER_DATASET:-1}
DOWNLOAD_MAX_RETRIES=${DOWNLOAD_MAX_RETRIES:-12}
DOWNLOAD_RETRY_BASE_SECONDS=${DOWNLOAD_RETRY_BASE_SECONDS:-60}
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

ERA5_CDS_RC=${ERA5_CDS_RC:-$HOME/.cdsapirc-era5}
export CDSAPI_RC="$ERA5_CDS_RC"
test -s "$CDSAPI_RC" || {
    echo "Missing ERA5 CDS credentials at $CDSAPI_RC. Resolve TODO(USER) in docs/DECISIONS_NEEDED.md."
    exit 2
}
case "$DOWNLOAD_WORKERS" in
    ''|*[!0-9]*|0)
        echo "DOWNLOAD_WORKERS must be a positive integer; got '$DOWNLOAD_WORKERS'."
        exit 2
        ;;
esac
case "$DOWNLOAD_CHUNKING" in
    yearly|monthly) ;;
    *)
        echo "DOWNLOAD_CHUNKING must be yearly or monthly; got '$DOWNLOAD_CHUNKING'."
        exit 2
        ;;
esac
case "$DOWNLOAD_PER_DATASET" in
    ''|*[!0-9]*|0)
        echo "DOWNLOAD_PER_DATASET must be a positive integer; got '$DOWNLOAD_PER_DATASET'."
        exit 2
        ;;
esac
case "$DOWNLOAD_MAX_RETRIES" in
    ''|*[!0-9]*)
        echo "DOWNLOAD_MAX_RETRIES must be a nonnegative integer; got '$DOWNLOAD_MAX_RETRIES'."
        exit 2
        ;;
esac

git pull --ff-only origin main
echo "ERA5 download plan: $DOWNLOAD_CHUNKING files, $DOWNLOAD_WORKERS local workers, $DOWNLOAD_PER_DATASET active request per dataset"
"$PY" -m py_compile src/repo_integrity.py src/data_pipeline/download_era5.py src/data_pipeline/regrid.py src/data_pipeline/build_cache.py src/data_pipeline/check_cache.py
"$PY" src/repo_integrity.py

(
    cd src
    "$PY" -m data_pipeline.download_era5 \
      --data_root "$DATA_ROOT" \
      --years "$YEARS" \
      --months "$MONTHS" \
      --chunking "$DOWNLOAD_CHUNKING" \
      --target_source "$TARGET_SOURCE" \
      --pressure_dataset "$ERA5_PRESSURE_DATASET" \
      --workers "$DOWNLOAD_WORKERS" \
      --per_dataset_workers "$DOWNLOAD_PER_DATASET" \
      --max_retries "$DOWNLOAD_MAX_RETRIES" \
      --retry_base_seconds "$DOWNLOAD_RETRY_BASE_SECONDS"
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
      --chunking "$DOWNLOAD_CHUNKING" \
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
