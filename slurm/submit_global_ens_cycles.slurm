#!/bin/bash
#SBATCH --job-name=hc_global_ens
#SBATCH --output=global_ens_cycles_%j.log
#SBATCH --error=global_ens_cycles_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mostafarezaali@ufl.edu

set -euo pipefail

WORK_DIR=/blue/nessie/mostafarezaali/HeatCast-Global/
PY=/blue/nessie/mostafarezaali/.conda/envs/torch_b200/bin/python
RESOLUTION=${RESOLUTION:-1.5deg}
INGEST_WORKERS=${INGEST_WORKERS:-16}
FOLD_WORKERS=${FOLD_WORKERS:-2}
FOLD_YEARS_JSON=${FOLD_YEARS_JSON:?TODO(USER): path to approved fold and ENS comparison-year JSON}
ECMWF_CYCLE_SPECS=${ECMWF_CYCLE_SPECS:?TODO(USER): comma-separated cycle_tag:member_count values}
HEATCAST_RUNS=cvfold0_global_w34_dist_v1,cvfold1_global_w34_dist_v1,cvfold2_global_w34_dist_v1,cvfold3_global_w34_dist_v1,cvfold4_global_w34_dist_v1
WINDOWS=(
  15,16,17,18,19,20,21
  22,23,24,25,26,27,28
  15,16,17,18,19,20,21,22,23,24,25,26,27,28
)

cd "$WORK_DIR"
unset PYTHONHOME PYTHONPATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HEATCAST_DOMAIN=global
export HEATCAST_RESOLUTION="$RESOLUTION"
export HEATCAST_FOLD_YEARS_JSON="$FOLD_YEARS_JSON"

test -s "$FOLD_YEARS_JSON"
COMPARISON_YEARS=$("$PY" -c "from fold_config import comparison_period; import sys; print(','.join(map(str, comparison_period(sys.argv[1]))))" "$FOLD_YEARS_JSON")
test -n "$COMPARISON_YEARS"

git pull --ff-only origin main
"$PY" -m py_compile src/repo_integrity.py src/ens_target_grid.py src/ens_ingest.py src/ens_score.py src/ens_compare.py
"$PY" src/repo_integrity.py

IFS=',' read -r -a CYCLE_SPECS <<< "$ECMWF_CYCLE_SPECS"
ENS_TEMPLATES=()
for SPEC in "${CYCLE_SPECS[@]}"; do
    TAG=${SPEC%%:*}
    MEMBERS=${SPEC##*:}
    if [ -z "$TAG" ] || ! [[ "$MEMBERS" =~ ^[0-9]+$ ]]; then
        echo "Invalid ECMWF_CYCLE_SPECS entry '$SPEC'; expected cycle_tag:member_count."
        exit 2
    fi
    INIT_LIST="$WORK_DIR/ens_reforecast/raw/init_list_${TAG}.txt"
    test -s "$INIT_LIST"
    "$PY" -u src/ens_ingest.py \
      --domain global \
      --resolution "$RESOLUTION" \
      --rt_tag "$TAG" \
      --init_list "$INIT_LIST" \
      --variable mx2t6 \
      --max_lead 28 \
      --expected_members "$MEMBERS" \
      --workers "$INGEST_WORKERS"
    ENS_TEMPLATES+=("cvfold{F}_global_ens_w34_${TAG}")
done
ENS_RUNS=$(IFS=','; echo "${ENS_TEMPLATES[*]}")

score_fold_cycle_window() {
    local fold=$1
    local tag=$2
    local leads=$3
    local label=${leads//,/-}
    local run="cvfold${fold}_global_ens_w34_${tag}"
    local output="$WORK_DIR/ens_exceedance_incremental/${run}/test/window_${label}/ens_score_metadata.json"
    if [ -s "$output" ]; then
        echo "Skipping complete ENS score fold=$fold cycle=$tag window=$label"
        return 0
    fi
    "$PY" -u src/ens_score.py \
      --domain global \
      --resolution "$RESOLUTION" \
      --fold_years_json "$FOLD_YEARS_JSON" \
      --comparison_years "$COMPARISON_YEARS" \
      --cv_fold "$fold" \
      --run_name "$run" \
      --rt_tag "$tag" \
      --window_leads "$leads"
}

for LEADS in "${WINDOWS[@]}"; do
    for SPEC in "${CYCLE_SPECS[@]}"; do
        TAG=${SPEC%%:*}
        PIDS=()
        for FOLD in 0 1 2 3 4; do
            score_fold_cycle_window "$FOLD" "$TAG" "$LEADS" &
            PIDS+=("$!")
            if [ "${#PIDS[@]}" -ge "$FOLD_WORKERS" ]; then
                for PID in "${PIDS[@]}"; do wait "$PID"; done
                PIDS=()
            fi
        done
        for PID in "${PIDS[@]}"; do wait "$PID"; done
    done
    LABEL=${LEADS//,/-}
    "$PY" -u src/ens_compare.py \
      --domain global \
      --resolution "$RESOLUTION" \
      --comparison_years "$COMPARISON_YEARS" \
      --heatcast_runs "$HEATCAST_RUNS" \
      --ens_runs "$ENS_RUNS" \
      --window_leads "$LEADS" \
      --heatcast_root exceedance_eval_incremental \
      --ens_root ens_exceedance_incremental \
      --output_dir "ens_head_to_head_${LABEL}" \
      --bootstrap_reps 5000 \
      --emit_per_year
done

W34_LEADS=15,16,17,18,19,20,21,22,23,24,25,26,27,28
"$PY" -u src/ens_heatcast_stack_opportunity.py \
  --domain global \
  --resolution "$RESOLUTION" \
  --heatcast_runs "$HEATCAST_RUNS" \
  --ens_runs "$ENS_RUNS" \
  --window_leads "$W34_LEADS" \
  --heatcast_root exceedance_eval_incremental \
  --ens_root ens_exceedance_incremental \
  --output_dir ens_heatcast_stack_opportunity_global \
  --driver_table_dir drivers/fold_safe_tables \
  --bootstrap_reps 5000 \
  --fold_workers 5 \
  --emit_per_year

mkdir -p exports
"$PY" -u src/export_w34_stack_netcdf.py \
  --domain global \
  --resolution "$RESOLUTION" \
  --output exports/global_w34_heatcast_ens_stack.nc \
  --window_leads "$W34_LEADS" \
  --heatcast_runs "$HEATCAST_RUNS" \
  --ens_runs "$ENS_RUNS" \
  --sort_by target \
  --fold_workers 5 \
  --max_stack_samples_per_fold 300000 \
  --require_continuous_fields

echo "GLOBAL MATCHED ECMWF ENS INGEST, SCORE, AND COMPARISON COMPLETE"
