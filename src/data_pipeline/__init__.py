"""ERA5 acquisition, regridding, cache construction, and validation tools."""

__all__ = (
    "CACHE_CHANNELS",
    "GridSpec",
    "LazyGlobalZarrDataset",
    "grid_for_resolution",
    "regrid_field",
)


def __getattr__(name):
    """Load public helpers lazily so ``python -m`` entry points run once."""
    if name in ("CACHE_CHANNELS", "LazyGlobalZarrDataset"):
        from . import build_cache
        return getattr(build_cache, name)
    if name in ("GridSpec", "grid_for_resolution", "regrid_field"):
        from . import regrid
        return getattr(regrid, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
