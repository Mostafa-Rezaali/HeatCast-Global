# MATLAB plotting utilities

This folder contains MATLAB scripts for plotting HeatCast W34 NetCDF exports.

Primary script:

```matlab
make_w34_truth_hindcast_movie('w34_heatcast_ens_stack.nc')
```

The movie shows observed continuous W34 truth on the left and HeatCast continuous
W34 hindcast on the right. Both fields are read from the NetCDF one time slice
at a time, so MATLAB does not need to keep the full 3-D arrays in memory.

Expected NetCDF variables:

- `ground_truth_3d(lat, lon, time)`: continuous observed W34 mean z-score
- `model_output_3d(lat, lon, time)`: continuous HeatCast W34 mean z-score
- `target_date_yyyymmdd(time)`: target window center date labels
- `lat`, `lon`: plotting grids

