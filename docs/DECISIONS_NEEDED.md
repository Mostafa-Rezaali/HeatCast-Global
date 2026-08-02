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

## Cross-validation fold years -- resolved 2026-07-31

The user approved the interleaved five-fold table committed at
`configs/global_fold_years.json`. Fold `k` tests `years[k::5]`, calibrates on
`years[(k+1)%5::5]`, and trains on all remaining 1979--2024 years.

Affected interfaces: fold-specific climatology, normalization, percentile
thresholds, hindcast export, and all pooled comparisons.

Production jobs default to the approved table and may override it through
`FOLD_YEARS_JSON` (or `HEATCAST_FOLD_YEARS_JSON`). `src/fold_config.py` rejects missing roles,
within-fold overlap, incomplete 1979--2024 coverage, or pooled test folds that
do not partition all years exactly once.

## Global vector indices -- resolved 2026-08-02

The user approved replacing the legacy frozen OMI `PC2Coefficient` with
Niño3.4. The five base indices are ordered PNA, NAO, Niño3.4, PDO, AO; the
separate BOM RMM1, RMM2, and MJO-amplitude channels remain unchanged.

`src/data_pipeline/build_condition_cache.py` downloads official daily CPC
PNA/NAO/AO and monthly NOAA PSL Niño3.4, reuses the original ERSSTv5
`PDO.xlsx`, records source hashes, validates complete 1979--2024 coverage, and
requires the four preserved channels to reproduce the legacy normalized
CondTrain overlap before it writes `cache/teleconnection_5.npy`.

## Isolated CPC daily-index gaps -- resolved 2026-08-02

The user approved retaining the five values absent from the official CPC files
as missing and excluding condition-incomplete initialization rows without
interpolation. The only affected full-W34 MJJAS model initialization is
2003-04-30; it is a Wednesday and therefore is not in the matched ECMWF
Monday/Thursday evaluation calendar. Missing PNA/NAO values on 2006-10-26 and
2007-01-26 fall outside the valid MJJAS initialization season.

## CDS credentials

`TODO(USER)`: Install and validate the ERA5 Climate Data Store API token in
`~/.cdsapirc-era5` on HiPerGator, using
`url: https://cds.climate.copernicus.eu/api`, and accept the required ERA5
dataset licenses before running the download workflow. Keep any existing
`~/.cdsapirc` using `https://ecds.ecmwf.int/api` for the separate ECMWF S2S
workflow.

This is an operational credential decision; no key or token belongs in Git.

## W34 storyline summers

`TODO(USER)`: Select one or two case summers that fall inside approved test
folds for the 1000+ member CFM plausible-worst-case demonstration. These cases
are illustrative figures, not verified headline metrics.

## Heat-index target -- resolved 2026-08-02

The user selected Heat Index instead of Tmax as the global primary target.
The implementation exactly reuses the existing HeatIndex project convention:
relative humidity is derived from ERA5 daily-mean 2 m dewpoint and daily Tmax,
then the NOAA/Steadman simple expression or Rothfusz regression with dry/humid
adjustments is evaluated in Fahrenheit and returned in Celsius. Humidity is not
clipped. Heat Index is computed before conservative regridding. Tmax remains a
lagged predictor, and the CONUS target path remains unchanged.

`TODO(USER)`: Before the matched ECMWF benchmark, pin the S2S surface-humidity
or dewpoint field and its temporal definition needed to calculate member-wise
Heat Index. The existing `mx2t6`-only ENS archive is a Tmax benchmark and must
not be presented as a matched Heat Index benchmark.
