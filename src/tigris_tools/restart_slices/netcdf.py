from __future__ import annotations

from pathlib import Path

from .cache import slice_cache_path
from .extract import SliceResult


def write_slice_caches(
    result: SliceResult,
    savdir: str | Path,
    num: int,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write the two datasets using the pyathena_tigris cache schema."""

    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "slice cache output requires xarray; install tigris-tools[slices]"
        ) from exc

    paths: dict[str, Path] = {}
    for axis in ("z", "y"):
        path = slice_cache_path(savdir, axis, num)
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing cache: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        dims = ("y", "x") if axis == "z" else ("z", "x")
        coordinates = {dim: result.coordinates[dim] for dim in dims}
        data_vars = {name: (dims, values) for name, values in result.planes[axis].items()}
        dataset = xr.Dataset(data_vars=data_vars, coords=coordinates, attrs={"time": result.time})
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            dataset.to_netcdf(temporary)
            temporary.replace(path)
        finally:
            dataset.close()
            temporary.unlink(missing_ok=True)
        paths[axis] = path
    return paths
