#!/bin/bash
#SBATCH --job-name=hc_global_hi_data
#SBATCH --output=global_heat_index_build_%j.log
#SBATCH --error=global_heat_index_build_%j.err
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
RESOLUTION=${RESOLUTION:-1.5deg}
YEARS=${YEARS:-1979-2024}
MONTHS=${MONTHS:-1,2,3,4,5,6,7,8,9,10,11,12}
DOWNLOAD_CHUNKING=${DOWNLOAD_CHUNKING:-yearly}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-8}
DOWNLOAD_PER_DATASET=${DOWNLOAD_PER_DATASET:-8}
DOWNLOAD_MAX_RETRIES=${DOWNLOAD_MAX_RETRIES:-12}
DOWNLOAD_RETRY_BASE_SECONDS=${DOWNLOAD_RETRY_BASE_SECONDS:-60}
HEAT_INDEX_WORKERS=${HEAT_INDEX_WORKERS:-64}
HEAT_INDEX_PROGRESS_EVERY=${HEAT_INDEX_PROGRESS_EVERY:-30}
FOLD_YEARS_JSON=${FOLD_YEARS_JSON:-$REPO_DIR/configs/global_fold_years.json}
ERA5_PRESSURE_DATASET=${ERA5_PRESSURE_DATASET:-reanalysis-era5-pressure-levels}

cd "$REPO_DIR"
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HEATCAST_DOMAIN=global
export HEATCAST_RESOLUTION="$RESOLUTION"
export HEATCAST_GLOBAL_TARGET=heat_index
export HEATCAST_FOLD_YEARS_JSON="$FOLD_YEARS_JSON"
export CDSAPI_RC=${ERA5_CDS_RC:-$HOME/.cdsapirc-era5}

test -s "$CDSAPI_RC"
test -s "$DATA_ROOT/cache/era5_${RESOLUTION}.zarr/heatcast_cache_metadata.json"
test -s "$DATA_ROOT/cache/teleconnection_5.npy"
test -s "$DATA_ROOT/drivers/rmm.txt"
test -s "$FOLD_YEARS_JSON"

git pull --ff-only origin main
"$PY" -m py_compile src/heat_index.py src/data_pipeline/download_era5.py src/data_pipeline/build_heat_index_target.py src/build_global_fold_sidecars.py
"$PY" src/repo_integrity.py

echo "Downloading only missing Heat Index source data: 46 annual daily-mean dewpoint files"
(
    cd src
    "$PY" -m data_pipeline.download_era5 \
      --data_root "$DATA_ROOT" \
      --years "$YEARS" \
      --months "$MONTHS" \
      --chunking "$DOWNLOAD_CHUNKING" \
      --target_source daily_statistics \
      --pressure_dataset "$ERA5_PRESSURE_DATASET" \
      --workers "$DOWNLOAD_WORKERS" \
      --per_dataset_workers "$DOWNLOAD_PER_DATASET" \
      --max_retries "$DOWNLOAD_MAX_RETRIES" \
      --retry_base_seconds "$DOWNLOAD_RETRY_BASE_SECONDS" \
      --enable_heat_index

    "$PY" -m data_pipeline.validate_pipeline raw \
      --data_root "$DATA_ROOT" \
      --expected_tasks 277 \
      --report "$DATA_ROOT/manifests/era5_heat_index_raw_validation.json"

    "$PY" -u -m data_pipeline.build_heat_index_target \
      --data_root "$DATA_ROOT" \
      --years "$YEARS" \
      --months "$MONTHS" \
      --chunking "$DOWNLOAD_CHUNKING" \
      --resolution "$RESOLUTION" \
      --workers "$HEAT_INDEX_WORKERS" \
      --progress_every "$HEAT_INDEX_PROGRESS_EVERY"
)

"$PY" -c 'import sys,zarr,numpy as np; p=f"{sys.argv[1]}/cache/era5_{sys.argv[2]}.zarr"; r=zarr.open_group(p,mode="r"); n=len(r["time"]); expected=(n,121,240) if sys.argv[2]=="1.5deg" else (n,721,1440); assert r["heat_index"].shape==expected; assert int(np.count_nonzero(r["heat_index_complete"][:]))==n; assert all(np.isfinite(r["heat_index"][i]).all() for i in (0,n//2,n-1)); print(f"PASS heat_index target: {n}/{n} complete")' "$DATA_ROOT" "$RESOLUTION"

for FOLD in 0 1 2 3 4; do
    rm -f \
      "$DATA_ROOT/cache/era5_${RESOLUTION}.zarr.sidecars/fold${FOLD}_normalization.npz" \
      "$DATA_ROOT/cache/era5_${RESOLUTION}.zarr.sidecars/fold${FOLD}_thresholds.npz"
done
for FOLD in 0 1 2 3 4; do
    "$PY" -u src/build_global_fold_sidecars.py \
      --fold "$FOLD" \
      --fold_years_json "$FOLD_YEARS_JSON" \
      --resolution "$RESOLUTION" \
      --force
done

echo "GLOBAL HEAT-INDEX TARGET AND FOLD SIDECARS COMPLETE"
