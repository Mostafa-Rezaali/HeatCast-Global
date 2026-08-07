#!/bin/bash
#SBATCH --job-name=hc_global_data
#SBATCH --output=global_data_build_%j.log
#SBATCH --error=global_data_build_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
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
DOWNLOAD_MAX_RETRIES=${DOWNLOAD_MAX_RETRIES:-12}
DOWNLOAD_RETRY_BASE_SECONDS=${DOWNLOAD_RETRY_BASE_SECONDS:-60}
ARCO_PROGRESS_EVERY=${ARCO_PROGRESS_EVERY:-10}
ARCO_ERA5_STORE=${ARCO_ERA5_STORE:-gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3}
ARCO_GROUPS=${ARCO_GROUPS:-daily_tmax,daily_t2m,single_levels,pressure_geopotential,pressure_850,static}
BUILD_FOLD_SIDECARS=${BUILD_FOLD_SIDECARS:-1}
DOWNLOAD_ONLY=${DOWNLOAD_ONLY:-0}
SKIP_DOWNLOAD=${SKIP_DOWNLOAD:-0}
CACHE_SAMPLE_COUNT=${CACHE_SAMPLE_COUNT:-9}
CACHE_WORKERS=${CACHE_WORKERS:-64}
CACHE_PROGRESS_EVERY=${CACHE_PROGRESS_EVERY:-30}

cd "$REPO_DIR"
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

case "$SKIP_DOWNLOAD" in
    0|1) ;;
    *)
        echo "SKIP_DOWNLOAD must be 0 or 1; got '$SKIP_DOWNLOAD'."
        exit 2
        ;;
esac
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
case "$DOWNLOAD_MAX_RETRIES" in
    ''|*[!0-9]*)
        echo "DOWNLOAD_MAX_RETRIES must be a nonnegative integer; got '$DOWNLOAD_MAX_RETRIES'."
        exit 2
        ;;
esac
case "$CACHE_WORKERS" in
    ''|*[!0-9]*|0)
        echo "CACHE_WORKERS must be a positive integer; got '$CACHE_WORKERS'."
        exit 2
        ;;
esac
case "$CACHE_PROGRESS_EVERY" in
    ''|*[!0-9]*|0)
        echo "CACHE_PROGRESS_EVERY must be a positive integer; got '$CACHE_PROGRESS_EVERY'."
        exit 2
        ;;
esac

git pull --ff-only origin main
"$PY" -m py_compile src/repo_integrity.py src/data_pipeline/download_era5.py src/data_pipeline/download_arco_era5.py src/data_pipeline/regrid.py src/data_pipeline/build_cache.py src/data_pipeline/check_cache.py src/data_pipeline/validate_pipeline.py
"$PY" src/repo_integrity.py

if [ "$SKIP_DOWNLOAD" = "0" ]; then
    "$PY" -c 'import tensorstore' || {
        echo "Missing TensorStore. Install once with: $PY -m pip install tensorstore"
        exit 2
    }
    echo "Google ARCO-ERA5 plan: $DOWNLOAD_CHUNKING files, $DOWNLOAD_WORKERS workers, groups=$ARCO_GROUPS"
    (
        cd src
        "$PY" -m data_pipeline.download_arco_era5 \
          --data_root "$DATA_ROOT" \
          --years "$YEARS" \
          --months "$MONTHS" \
          --chunking "$DOWNLOAD_CHUNKING" \
          --groups "$ARCO_GROUPS" \
          --workers "$DOWNLOAD_WORKERS" \
          --store "$ARCO_ERA5_STORE" \
          --max_retries "$DOWNLOAD_MAX_RETRIES" \
          --retry_base_seconds "$DOWNLOAD_RETRY_BASE_SECONDS" \
          --progress_every "$ARCO_PROGRESS_EVERY"
    )
else
    echo "SKIP_DOWNLOAD=1: using the completed local ERA5 archive without contacting Google Cloud"
fi

(
    cd src
    echo "Validating completed ERA5 manifest before cache construction"
    "$PY" -m data_pipeline.validate_pipeline raw \
      --data_root "$DATA_ROOT" \
      --expected_tasks 231 \
      --report "$DATA_ROOT/manifests/era5_raw_validation.json"
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
      --target_source "$TARGET_SOURCE" \
      --workers "$CACHE_WORKERS" \
      --progress_every "$CACHE_PROGRESS_EVERY"
)

test -s "$DATA_ROOT/cache/era5_${RESOLUTION}.zarr/heatcast_cache_metadata.json"
(
    cd src
    "$PY" -m data_pipeline.validate_pipeline cache \
      --data_root "$DATA_ROOT" \
      --years "$YEARS" \
      --months "$MONTHS" \
      --resolution "$RESOLUTION" \
      --target_source "$TARGET_SOURCE" \
      --sample_count "$CACHE_SAMPLE_COUNT" \
      --report "$DATA_ROOT/cache/era5_${RESOLUTION}_validation.json"
)
if [ "$BUILD_FOLD_SIDECARS" = "1" ]; then
    FOLD_YEARS_JSON=${FOLD_YEARS_JSON:-$REPO_DIR/configs/global_fold_years.json}
    test -s "$FOLD_YEARS_JSON"
    for FOLD in 0 1 2 3 4; do
        "$PY" -u src/build_global_fold_sidecars.py \
          --fold "$FOLD" \
          --fold_years_json "$FOLD_YEARS_JSON" \
          --resolution "$RESOLUTION"
    done
fi
echo "GLOBAL ERA5 CACHE BUILD COMPLETE: $DATA_ROOT/cache/era5_${RESOLUTION}.zarr"
