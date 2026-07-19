"""Global rectilinear regridding with xESMF and data-free SciPy fallbacks.

Tmax targets use first-order conservative cell-overlap weights. Smooth
predictors use periodic bilinear interpolation. Weight matrices can be cached
outside the repository and are independent of field values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True)
class GridSpec:
    """One global latitude-longitude grid in degrees."""

    lat: np.ndarray
    lon: np.ndarray
    resolution: str

    def __post_init__(self):
        lat = np.asarray(self.lat, dtype=np.float64)
        lon = np.asarray(self.lon, dtype=np.float64)
        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError("Grid latitude and longitude must be one-dimensional.")
        if lat.size < 2 or lon.size < 2:
            raise ValueError("Global grids require at least two coordinates per axis.")
        object.__setattr__(self, "lat", lat)
        object.__setattr__(self, "lon", lon)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.lat.size, self.lon.size)


def grid_for_resolution(resolution: str) -> GridSpec:
    """Return the configured Phase A or Phase B global grid."""
    specs = {
        "1.5deg": (121, 240, 1.5),
        "0.25deg": (721, 1440, 0.25),
    }
    if str(resolution) not in specs:
        raise ValueError(f"Unsupported resolution {resolution!r}; expected one of {tuple(specs)}.")
    n_lat, n_lon, spacing = specs[str(resolution)]
    lat = np.linspace(90.0, -90.0, n_lat, dtype=np.float64)
    lon = np.arange(n_lon, dtype=np.float64) * spacing
    return GridSpec(lat=lat, lon=lon, resolution=str(resolution))


def _sorted_unique_longitudes(lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normalized = np.mod(np.asarray(lon, dtype=np.float64), 360.0)
    order = np.argsort(normalized)
    sorted_lon = normalized[order]
    keep = np.ones(sorted_lon.size, dtype=bool)
    keep[1:] = np.diff(sorted_lon) > 1e-10
    return sorted_lon[keep], order[keep]


def _latitude_edges(sorted_lat: np.ndarray) -> np.ndarray:
    edges = np.empty(sorted_lat.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (sorted_lat[:-1] + sorted_lat[1:])
    edges[0] = -90.0
    edges[-1] = 90.0
    return np.clip(edges, -90.0, 90.0)


def _longitude_edges(sorted_lon: np.ndarray) -> np.ndarray:
    edges = np.empty(sorted_lon.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (sorted_lon[:-1] + sorted_lon[1:])
    edges[0] = 0.5 * (sorted_lon[-1] - 360.0 + sorted_lon[0])
    edges[-1] = edges[0] + 360.0
    return edges


def _periodic_segments(start: float, end: float):
    width = float(end - start)
    if width <= 0.0 or width > 360.0 + 1e-8:
        raise ValueError(f"Invalid periodic cell width {width}.")
    left = float(start % 360.0)
    right = left + width
    if right <= 360.0 + 1e-12:
        return ((left, min(right, 360.0)),)
    return ((left, 360.0), (0.0, right - 360.0))


def _interval_overlap(a, b) -> float:
    return max(0.0, min(float(a[1]), float(b[1])) - max(float(a[0]), float(b[0])))


def conservative_weight_matrices(
    source_lat,
    source_lon,
    target_lat,
    target_lon,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return separable first-order spherical cell-overlap weight matrices."""
    source_lat = np.asarray(source_lat, dtype=np.float64)
    target_lat = np.asarray(target_lat, dtype=np.float64)
    src_lat_order = np.argsort(source_lat)
    tgt_lat_order = np.argsort(target_lat)
    src_lat = source_lat[src_lat_order]
    tgt_lat = target_lat[tgt_lat_order]
    if np.any(np.diff(src_lat) <= 0.0) or np.any(np.diff(tgt_lat) <= 0.0):
        raise ValueError("Latitude coordinates must be unique.")

    src_mu_edges = np.sin(np.deg2rad(_latitude_edges(src_lat)))
    tgt_mu_edges = np.sin(np.deg2rad(_latitude_edges(tgt_lat)))
    lat_weights_sorted = np.zeros((tgt_lat.size, src_lat.size), dtype=np.float64)
    for target_index in range(tgt_lat.size):
        target_interval = (tgt_mu_edges[target_index], tgt_mu_edges[target_index + 1])
        target_width = target_interval[1] - target_interval[0]
        for source_index in range(src_lat.size):
            source_interval = (src_mu_edges[source_index], src_mu_edges[source_index + 1])
            lat_weights_sorted[target_index, source_index] = (
                _interval_overlap(target_interval, source_interval) / target_width
            )

    src_lon, src_lon_order = _sorted_unique_longitudes(np.asarray(source_lon, dtype=np.float64))
    tgt_lon, tgt_lon_order = _sorted_unique_longitudes(np.asarray(target_lon, dtype=np.float64))
    if src_lon.size != len(source_lon) or tgt_lon.size != len(target_lon):
        raise ValueError("Longitude coordinates must be unique modulo 360 degrees.")
    src_lon_edges = _longitude_edges(src_lon)
    tgt_lon_edges = _longitude_edges(tgt_lon)
    lon_weights_sorted = np.zeros((tgt_lon.size, src_lon.size), dtype=np.float64)
    for target_index in range(tgt_lon.size):
        target_segments = _periodic_segments(tgt_lon_edges[target_index], tgt_lon_edges[target_index + 1])
        target_width = tgt_lon_edges[target_index + 1] - tgt_lon_edges[target_index]
        for source_index in range(src_lon.size):
            source_segments = _periodic_segments(src_lon_edges[source_index], src_lon_edges[source_index + 1])
            overlap = sum(_interval_overlap(a, b) for a in target_segments for b in source_segments)
            lon_weights_sorted[target_index, source_index] = overlap / target_width

    return lat_weights_sorted, lon_weights_sorted, src_lat_order, src_lon_order


def _load_or_build_conservative_weights(
    source_lat,
    source_lon,
    target_lat,
    target_lon,
    weights_path: Optional[Path],
):
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in (source_lat, source_lon, target_lat, target_lon))
    if weights_path is not None and Path(weights_path).is_file():
        with np.load(weights_path, allow_pickle=False) as cached:
            if all(np.array_equal(cached[name], value) for name, value in zip(
                ("source_lat", "source_lon", "target_lat", "target_lon"), arrays
            )):
                return (
                    cached["lat_weights"], cached["lon_weights"],
                    cached["src_lat_order"], cached["src_lon_order"],
                )
    built = conservative_weight_matrices(*arrays)
    if weights_path is not None:
        path = Path(weights_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        with partial.open("wb") as handle:
            np.savez_compressed(
                handle,
                source_lat=arrays[0], source_lon=arrays[1],
                target_lat=arrays[2], target_lon=arrays[3],
                lat_weights=built[0], lon_weights=built[1],
                src_lat_order=built[2], src_lon_order=built[3],
            )
        partial.replace(path)
    return built


def conservative_regrid_scipy(
    field,
    source_lat,
    source_lon,
    target_lat,
    target_lon,
    *,
    weights_path: Optional[Path] = None,
) -> np.ndarray:
    """Conservatively regrid a 2-D field using cached spherical overlaps."""
    array = np.asarray(field, dtype=np.float64)
    if array.shape != (len(source_lat), len(source_lon)):
        raise ValueError(
            f"Field shape {array.shape} does not match source grid {(len(source_lat), len(source_lon))}."
        )
    lat_w, lon_w, src_lat_order, src_lon_order = _load_or_build_conservative_weights(
        source_lat, source_lon, target_lat, target_lon, weights_path
    )
    ordered = array[np.asarray(src_lat_order)][:, np.asarray(src_lon_order)]
    valid = np.isfinite(ordered).astype(np.float64)
    numerator = lat_w @ np.where(np.isfinite(ordered), ordered, 0.0) @ lon_w.T
    denominator = lat_w @ valid @ lon_w.T
    sorted_output = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0.0,
    )
    target_lat_order = np.argsort(np.asarray(target_lat, dtype=np.float64))
    target_lon_order = np.argsort(np.mod(np.asarray(target_lon, dtype=np.float64), 360.0))
    output = np.empty_like(sorted_output)
    output[np.ix_(target_lat_order, target_lon_order)] = sorted_output
    return output.astype(np.float32)


def bilinear_regrid_scipy(field, source_lat, source_lon, target_lat, target_lon) -> np.ndarray:
    """Periodically bilinear-regrid a smooth 2-D global field."""
    array = np.asarray(field, dtype=np.float64)
    lat_order = np.argsort(np.asarray(source_lat, dtype=np.float64))
    sorted_lat = np.asarray(source_lat, dtype=np.float64)[lat_order]
    sorted_lon, lon_order = _sorted_unique_longitudes(np.asarray(source_lon, dtype=np.float64))
    ordered = array[np.asarray(lat_order)][:, np.asarray(lon_order)]
    extended_lon = np.concatenate((sorted_lon[-1:] - 360.0, sorted_lon, sorted_lon[:1] + 360.0))
    extended = np.concatenate((ordered[:, -1:], ordered, ordered[:, :1]), axis=1)
    interpolator = RegularGridInterpolator(
        (sorted_lat, extended_lon),
        extended,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    target_lat_grid, target_lon_grid = np.meshgrid(
        np.asarray(target_lat, dtype=np.float64),
        np.mod(np.asarray(target_lon, dtype=np.float64), 360.0),
        indexing="ij",
    )
    points = np.column_stack((target_lat_grid.ravel(), target_lon_grid.ravel()))
    return interpolator(points).reshape(target_lat_grid.shape).astype(np.float32)


def _xesmf_regrid(field, source_lat, source_lon, target_lat, target_lon, method, weights_path):
    import xarray as xr
    import xesmf as xe

    source = xr.DataArray(
        np.asarray(field),
        dims=("lat", "lon"),
        coords={"lat": np.asarray(source_lat), "lon": np.asarray(source_lon)},
    )
    target = xr.Dataset(coords={"lat": np.asarray(target_lat), "lon": np.asarray(target_lon)})
    regridder = xe.Regridder(
        source,
        target,
        method,
        periodic=True,
        filename=str(weights_path) if weights_path is not None else None,
        reuse_weights=bool(weights_path is not None and Path(weights_path).exists()),
    )
    return np.asarray(regridder(source).values, dtype=np.float32)


def regrid_field(
    field,
    source_lat,
    source_lon,
    target: GridSpec,
    *,
    method: str,
    weights_path: Optional[Path] = None,
    prefer_xesmf: bool = True,
) -> np.ndarray:
    """Regrid one field, falling back to the pure-SciPy implementation."""
    if method not in ("conservative", "bilinear"):
        raise ValueError("method must be 'conservative' or 'bilinear'.")
    if prefer_xesmf:
        try:
            return _xesmf_regrid(
                field, source_lat, source_lon, target.lat, target.lon, method, weights_path
            )
        except (ImportError, ModuleNotFoundError):
            pass
    if method == "conservative":
        return conservative_regrid_scipy(
            field,
            source_lat,
            source_lon,
            target.lat,
            target.lon,
            weights_path=weights_path,
        )
    return bilinear_regrid_scipy(field, source_lat, source_lon, target.lat, target.lon)
