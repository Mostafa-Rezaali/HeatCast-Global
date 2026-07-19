# Scientific Decisions Needed

This register is the single collection point for every unresolved
`TODO(USER)` in HeatCast-Global. Code paths remain fixture-driven or disabled
until each decision is pinned; no production score should infer these values.

## ECMWF ENS cycle and comparison period

`TODO(USER)`: Pin the ECMWF model version/cycle used for final scoring, confirm
its available reforecast years and ensemble-member counts, and then set the
fixed `ens_comparison_period` to those matched years.

Affected interfaces: ENS download metadata, cycle-specific ingest, the shared
initialization calendar, comparison-period config, and paper tables.

## Cross-validation fold years

`TODO(USER)`: Approve the exact five year-disjoint train/calibration/test fold
table for 1979--2024. Until approved, `Config.CV_FOLD_YEARS` remains `None`;
data-free tests use synthetic fixture years and production scoring must refuse
to infer a table.

Affected interfaces: fold-specific climatology, normalization, percentile
thresholds, hindcast export, and all pooled comparisons.

Production jobs require the approved table through `FOLD_YEARS_JSON` (or
`HEATCAST_FOLD_YEARS_JSON`). `src/fold_config.py` rejects missing roles,
within-fold overlap, incomplete 1979--2024 coverage, or pooled test folds that
do not partition all years exactly once.

## CDS credentials

`TODO(USER)`: Install and validate the CDS API key in `~/.cdsapirc` on
HiPerGator and accept the required ERA5/S2S dataset licenses before running the
download workflow.

This is an operational credential decision; no key or token belongs in Git.

## CDS pressure-level dataset identifier

`TODO(USER)`: Pin the exact CDS dataset identifier used for ERA5 pressure-level
geopotential, temperature, humidity, and wind retrievals. The prompt specifies
the variables and levels but deliberately does not name this dataset; therefore
`download_era5.py` exposes `--pressure_dataset` and production download refuses
to guess it.

## W34 storyline summers

`TODO(USER)`: Select one or two case summers that fall inside approved test
folds for the 1000+ member CFM plausible-worst-case demonstration. These cases
are illustrative figures, not verified headline metrics.

## Heat-index scope

`TODO(USER)`: Decide whether the optional ERA5 dewpoint-derived heat-index
target is in scope for the first paper. `ENABLE_HEAT_INDEX` remains off by
default because enabling it increases target-side acquisition and cache
volume.
