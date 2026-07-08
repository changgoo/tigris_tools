from tigris_tools.refine_restart.layout import (
    ID_LIST_RECORD_STRUCT,
    LOGICAL_LOCATION_STRUCT,
    MESH_HEADER_STRUCT,
    REGION_SIZE_STRUCT,
    LogicalLocation,
    MeshHeader,
    RegionSize,
    pack_id_record,
    pack_logical_location,
    pack_mesh_header,
    pack_region_size,
    unpack_id_record,
    unpack_logical_location,
    unpack_mesh_header,
    unpack_region_size,
)


def test_struct_sizes_match_restart_writer():
    assert LOGICAL_LOCATION_STRUCT.size == 32
    assert REGION_SIZE_STRUCT.size == 112
    assert MESH_HEADER_STRUCT.size == 140
    assert ID_LIST_RECORD_STRUCT.size == 48


def test_logical_location_round_trip():
    loc = LogicalLocation(lx1=1, lx2=2, lx3=3, level=4)
    assert unpack_logical_location(pack_logical_location(loc)) == loc


def test_region_size_round_trip():
    region = RegionSize.from_bounds(
        x1min=0.0,
        x2min=1.0,
        x3min=2.0,
        x1max=10.0,
        x2max=11.0,
        x3max=12.0,
        nx1=64,
        nx2=32,
        nx3=16,
    )
    assert unpack_region_size(pack_region_size(region)) == region


def test_mesh_header_round_trip():
    region = RegionSize.from_bounds(
        x1min=0.0,
        x2min=0.0,
        x3min=0.0,
        x1max=1.0,
        x2max=2.0,
        x3max=3.0,
        nx1=64,
        nx2=64,
        nx3=512,
    )
    header = MeshHeader(nbtotal=64, root_level=2, mesh_size=region, time=1.25, dt=0.5, ncycle=7)
    assert unpack_mesh_header(pack_mesh_header(header)) == header


def test_id_record_round_trip():
    loc = LogicalLocation(lx1=5, lx2=6, lx3=7, level=3)
    packed = pack_id_record(loc, cost=1.5, byte_size=123456)
    assert unpack_id_record(packed) == (loc, 1.5, 123456)
