# HeatCast-Global Model Inputs

This file is the authoritative channel contract for both `Config.DOMAIN`
modes. Global mode is the repository default. All time-varying normalization
and climatology statistics are fitted from the active fold's training years
only.

## Global ERA5 mode (default)

The model call remains:

```python
model(x_input, dummy_t, vec_c, global_fields=None)
```

`dummy_t = 0.5` is the fixed direct-inference time embedding. The separate
59-field coarse-global encoder is disabled (`NUM_GLOBAL_CHANNELS = 0`) because
the fine grid is itself global.

### Fine-grid input: 26 channels

`x_input = [x_t, x_t-1, x_t-2, spatial_c]` on the configured `lat x lon` grid.
`A` denotes subtraction of a four-harmonic day-of-year climatology fitted per
grid cell from training years only, followed by fold-safe normalization.

1. ERA5 daily Tmax at `t` (`A`)
2. ERA5 daily Tmax at `t-1` (`A`)
3. ERA5 daily Tmax at `t-2` (`A`)
4. daily-mean `t2m` at `t` (`A`)
5. `swvl1` at `t` (`A`)
6. `swvl1` trailing 20-day mean (`A`)
7. `swvl2` trailing 20-day mean (`A`)
8. SST at `t`, zero-filled where invalid (`A`)
9. SST validity mask
10. z500 at `t` (`A`)
11. z500 trailing low20 component (`A`)
12. MSLP at `t` (`A`)
13. t850 at `t` (`A`)
14. q850 at `t`
15. u850 at `t`
16. v850 at `t`
17. z300 at `t`
18. orography (ERA5 surface geopotential divided by `g`)
19. land-sea mask
20. sine latitude
21. cosine latitude
22. sine longitude
23. cosine longitude
24. day-of-year sine
25. day-of-year cosine
26. daily-mean TOA insolation at `t`, scaled by the solar constant

The external zarr cache stores 17 physical fields. Lagged Tmax and the nine
positional/calendar/insolation channels are assembled lazily per sample. The
store is opened inside each Dataset worker, never in the parent process, and
has `time=1` chunks.

### Vector input: 8 channels

1--5. the existing five configured teleconnection indices
6. MJO RMM1
7. MJO RMM2
8. MJO amplitude

RMM values use the parser shared with `build_driver_tables.py`; the model path
does not implement a second parser. Vector normalization uses only the active
fold's training indices.

### Global target and outputs

- Target: ERA5 daily maximum 2 m temperature for UTC days.
- Default target mode: `climatology_anomaly`.
- Climatology: intercept plus the first four annual sine/cosine harmonics per
  grid cell, fitted from training years only.
- Persistence residual: disabled in the default anomaly mode.
- Leads: exactly `15...28`, emitted as a 14-lead mean/sigma tube.
- Continuous aggregates: week 3 (`15...21`), week 4 (`22...28`), and W34
  (`15...28`).

## CONUS compatibility mode

`DOMAIN=conus` retains the original PRISM target, z-score plus persistence
residual semantics, regional mesh, five vector indices, and separate coarse
global conditioning.

The base CONUS local stack remains the original 19 channels:

1. PRISM T2max at `t`
2. PRISM T2max at `t-1`
3. PRISM T2max at `t-2`
4. geopotential
5. soil moisture
6. sea-level pressure
7. 2 m temperature
8. q850
9. t850
10. u850
11. v850
12. z300
13. topography
14. latitude
15. longitude
16. day-of-year sine
17. day-of-year cosine
18. TOA insolation
19. land mask

The current compatibility implementation can additionally enable six existing
slow local-lag channels (`t2max`, soil moisture, and 2 m temperature at lags 7
and 14), producing 25 deterministic input channels. This is preserved behavior,
not part of the global redesign.

The CONUS coarse-global stack remains 59 ERA5 variables decomposed into
`low20` and `residual` components (118 channels). Its variable inventory and
decomposition code are unchanged and are disabled only when `DOMAIN=global`.
