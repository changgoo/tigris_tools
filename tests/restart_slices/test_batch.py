import os
import struct

from tigris_tools.refine_restart import layout
from tigris_tools.restart_slices.batch import (
    build_batch_plan,
    discover_numbered_restarts,
    discover_particle_files,
    match_particle_file,
)
from tigris_tools.restart_slices.cache import slice_cache_path

PARAMS = b"""<restart>
nint_user_mesh_data = 0
nreal_user_mesh_data = 0
<par_end>
"""


def test_discovery_excludes_repaired_names_and_supports_prefix(tmp_path):
    header = _header(cycle=17, time=3.0)
    _write_restart(tmp_path / "TIGRESS.00021.rst", header)
    _write_restart(tmp_path / "TIGRESS.00021.repaired.rst", header)
    _write_restart(tmp_path / "OTHER.00022.rst", header)

    all_files = discover_numbered_restarts(tmp_path)
    tigress_files = discover_numbered_restarts(tmp_path, prefix="TIGRESS")

    assert [(item.num, item.path.name) for item in all_files] == [
        (21, "TIGRESS.00021.rst"),
        (22, "OTHER.00022.rst"),
    ]
    assert [(item.num, item.path.name) for item in tigress_files] == [
        (21, "TIGRESS.00021.rst")
    ]


def test_particle_matching_uses_cycle_and_time(tmp_path):
    header = _header(cycle=17, time=3.0)
    restart_path = tmp_path / "TIGRESS.00021.rst"
    _write_restart(restart_path, header)
    _write_particle(tmp_path / "TIGRESS.out3.00151.par0.parbin", header)
    _write_particle(tmp_path / "TIGRESS.out3.00152.par0.parbin", _header(cycle=18, time=4.0))

    restart = discover_numbered_restarts(tmp_path)[0]
    particles = discover_particle_files(tmp_path)

    assert match_particle_file(restart, particles).name == "TIGRESS.out3.00151.par0.parbin"


def test_batch_plan_skips_only_when_both_caches_are_fresh(tmp_path):
    header = _header(cycle=17, time=3.0)
    restart_path = tmp_path / "TIGRESS.00021.rst"
    _write_restart(restart_path, header)
    restart = discover_numbered_restarts(tmp_path)[0]
    savdir = tmp_path / "analysis"
    for axis in ("y", "z"):
        cache = slice_cache_path(savdir, axis, 21)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(b"cache")
        os.utime(cache, (restart_path.stat().st_mtime + 10,) * 2)

    plan = build_batch_plan([restart], [], savdir)

    assert plan[0].mode == "fresh"


def _header(*, cycle, time):
    region = layout.RegionSize.from_bounds(
        x1min=0,
        x2min=0,
        x3min=0,
        x1max=1,
        x2max=1,
        x3max=1,
        nx1=1,
        nx2=1,
        nx3=1,
    )
    return layout.MeshHeader(0, 0, region, time, 0.1, cycle)


def _write_restart(path, header):
    path.write_bytes(PARAMS + layout.pack_mesh_header(header))


def _write_particle(path, header):
    path.write_bytes(
        b"# synthetic\n"
        + b"# pid,\n"
        + struct.pack("<ii", 0, 0)
        + struct.pack("<ii", header.nbtotal, header.root_level)
        + layout.pack_region_size(header.mesh_size)
        + struct.pack("<ddi", header.time, header.dt, header.ncycle)
    )
