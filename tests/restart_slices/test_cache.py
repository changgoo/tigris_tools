import os

import pytest

from tigris_tools.restart_slices.cache import (
    cache_is_fresh,
    projection_cache_path,
    slice_cache_path,
)


def test_slice_cache_path_matches_pyathena_tigris(tmp_path):
    assert slice_cache_path(tmp_path, "z", 25) == (tmp_path / "allslc.z" / "allslc.z.00025.nc")
    assert slice_cache_path(tmp_path, "y", 25, outid=2) == (
        tmp_path / "allslc.y" / "allslc.y.out2.00025.nc"
    )


def test_slice_cache_path_supports_filebase(tmp_path):
    assert slice_cache_path(tmp_path, "z", 3, filebase="central") == (
        tmp_path / "allslc.z" / "central.00003.nc"
    )


def test_projection_cache_path_matches_pyathena_tigris(tmp_path):
    assert projection_cache_path(tmp_path, "z", 25) == tmp_path / "prj.z" / "prj.z.00025.nc"
    assert projection_cache_path(tmp_path, "y", 25, outid=2) == (
        tmp_path / "prj.y" / "prj.y.out2.00025.nc"
    )


@pytest.mark.parametrize("axis", ["x", "Y", ""])
def test_slice_cache_path_rejects_unknown_axis(tmp_path, axis):
    with pytest.raises(ValueError, match="axis must be"):
        slice_cache_path(tmp_path, axis, 0)


def test_cache_is_fresh_uses_strict_timestamp_ordering(tmp_path):
    source = tmp_path / "TIGRESS.00025.rst"
    cache = tmp_path / "allslc.z.00025.nc"
    source.write_bytes(b"source")
    cache.write_bytes(b"cache")

    os.utime(source, (10, 10))
    os.utime(cache, (11, 11))
    assert cache_is_fresh(cache, source)

    os.utime(cache, (10, 10))
    assert not cache_is_fresh(cache, source)
