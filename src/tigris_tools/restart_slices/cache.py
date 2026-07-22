from __future__ import annotations

from pathlib import Path

SLICE_PREFIXES = {"y": "allslc.y", "z": "allslc.z"}
PROJECTION_PREFIXES = {"y": "prj.y", "z": "prj.z"}


def slice_cache_path(
    savdir: str | Path,
    axis: str,
    num: int,
    *,
    filebase: str | None = None,
    outid: int | None = None,
) -> Path:
    """Return the path used by pyathena_tigris ``check_netcdf``.

    ``savdir`` is the simulation analysis directory, not the axis-specific
    subdirectory.  Output numbers use the five-digit formatting of the
    authoritative decorator.
    """

    try:
        prefix = SLICE_PREFIXES[axis]
    except KeyError as exc:
        choices = ", ".join(sorted(SLICE_PREFIXES))
        raise ValueError(f"axis must be one of {choices}; got {axis!r}") from exc
    if num < 0:
        raise ValueError(f"output number must be non-negative; got {num}")
    if outid is not None and outid < 0:
        raise ValueError(f"output ID must be non-negative; got {outid}")

    name = prefix if filebase is None else filebase
    if outid is not None:
        name += f".out{outid:d}"
    return Path(savdir) / prefix / f"{name}.{num:05d}.nc"


def projection_cache_path(
    savdir: str | Path,
    axis: str,
    num: int,
    *,
    filebase: str | None = None,
    outid: int | None = None,
) -> Path:
    """Return the path used by ``SliceProj.get_prj`` and ``check_netcdf``."""

    try:
        prefix = PROJECTION_PREFIXES[axis]
    except KeyError as exc:
        choices = ", ".join(sorted(PROJECTION_PREFIXES))
        raise ValueError(f"axis must be one of {choices}; got {axis!r}") from exc
    if num < 0:
        raise ValueError(f"output number must be non-negative; got {num}")
    if outid is not None and outid < 0:
        raise ValueError(f"output ID must be non-negative; got {outid}")

    name = prefix if filebase is None else filebase
    if outid is not None:
        name += f".out{outid:d}"
    return Path(savdir) / prefix / f"{name}.{num:05d}.nc"


def cache_is_fresh(cache_path: str | Path, source_path: str | Path) -> bool:
    """Match pyathena_tigris: a cache is reusable only when newer than source."""

    cache = Path(cache_path)
    source = Path(source_path)
    return cache.is_file() and cache.stat().st_mtime > source.stat().st_mtime
