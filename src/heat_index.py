"""Exact NOAA/Steadman heat-index calculations reused by HeatCast-Global.

The implementation matches the existing Mostafa-Rezaali/HeatIndex workflow:
relative humidity is derived from daily-mean dewpoint and daily Tmax, then the
simple Steadman expression or Rothfusz regression is applied in Fahrenheit and
returned in degrees Celsius.
"""

from __future__ import annotations

import numpy as np


def saturation_vapor_pressure_hpa(temperature_c) -> np.ndarray:
    """Return saturation vapor pressure in hPa for Celsius temperature."""
    values = np.asarray(temperature_c, dtype=np.float64)
    return 6.112 * np.exp((17.67 * values) / (values + 243.5))


def relative_humidity_from_dewpoint(tmax_c, dewpoint_mean_c) -> np.ndarray:
    """Derive RH (%) at daily Tmax from daily-mean dewpoint without clipping."""
    temperature = np.asarray(tmax_c, dtype=np.float64)
    dewpoint = np.asarray(dewpoint_mean_c, dtype=np.float64)
    if temperature.shape != dewpoint.shape:
        raise ValueError(
            f"Tmax and dewpoint shapes must match, got {temperature.shape} and {dewpoint.shape}."
        )
    return 100.0 * (
        saturation_vapor_pressure_hpa(dewpoint)
        / saturation_vapor_pressure_hpa(temperature)
    )


def heat_index_c(tmax_c, relative_humidity_percent) -> np.ndarray:
    """Return NOAA/Steadman heat index in Celsius, matching HeatIndex exactly."""
    temperature = np.asarray(tmax_c, dtype=np.float64)
    rh = np.asarray(relative_humidity_percent, dtype=np.float64)
    if temperature.shape != rh.shape:
        raise ValueError(
            f"Temperature and RH shapes must match, got {temperature.shape} and {rh.shape}."
        )
    temp_f = (9.0 / 5.0) * temperature + 32.0
    heat_index = np.full(temp_f.shape, np.nan, dtype=np.float64)

    mask_simple = temp_f < 80.0
    heat_index[mask_simple] = 0.5 * (
        temp_f[mask_simple]
        + 61.0
        + ((temp_f[mask_simple] - 68.0) * 1.2)
        + (rh[mask_simple] * 0.094)
    )

    adjustment_low = np.zeros(temp_f.shape, dtype=np.float64)
    adjustment_high = np.zeros(temp_f.shape, dtype=np.float64)
    mask_low = (rh < 13.0) & (temp_f >= 80.0) & (temp_f <= 112.0)
    adjustment_low[mask_low] = -(
        ((13.0 - rh[mask_low]) / 4.0)
        * np.sqrt((17.0 - np.abs(temp_f[mask_low] - 95.0)) / 17.0)
    )
    mask_high = (rh > 85.0) & (temp_f >= 80.0) & (temp_f <= 87.0)
    adjustment_high[mask_high] = ((rh[mask_high] - 85.0) / 10.0) * (
        (87.0 - temp_f[mask_high]) / 5.0
    )

    mask_regression = temp_f >= 80.0
    tf = temp_f[mask_regression]
    humidity = rh[mask_regression]
    rothfusz = (
        -42.379
        + (2.04901523 * tf)
        + (10.14333127 * humidity)
        - (0.22475541 * tf * humidity)
        - (0.00683783 * tf**2)
        - (0.05481717 * humidity**2)
        + (0.00122874 * tf**2 * humidity)
        + (0.00085282 * tf * humidity**2)
        - (0.00000199 * tf**2 * humidity**2)
    )
    heat_index[mask_regression] = (
        rothfusz
        + adjustment_low[mask_regression]
        + adjustment_high[mask_regression]
    )
    return (5.0 / 9.0) * (heat_index - 32.0)


def heat_index_from_tmax_dewpoint_c(tmax_c, dewpoint_mean_c) -> np.ndarray:
    """Compute Celsius heat index from Celsius Tmax and daily-mean dewpoint."""
    rh = relative_humidity_from_dewpoint(tmax_c, dewpoint_mean_c)
    return heat_index_c(tmax_c, rh)
