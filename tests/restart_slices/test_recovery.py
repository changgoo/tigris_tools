import struct

from tigris_tools.refine_restart import layout
from tigris_tools.refine_restart.reader import read_restart_index
from tigris_tools.restart_slices.recovery import (
    recover_rank_shifted_index,
    recover_rank_shifted_index_from_payload,
)
from tigris_tools.restart_slices.repair import (
    repair_rank_shifted_restart,
    validate_repaired_restart,
)


def test_recover_rank_shifted_index_with_particle_validation(tmp_path):
    restart, particle, canonical_payload_start, sizes = _write_shifted_outputs(tmp_path)

    report = recover_rank_shifted_index(restart, particle)

    assert report.block_shifts == (0, -9, 0, 0)
    assert report.candidate_shifts == (-9, 0)
    assert report.exact_id_records == 3
    assert report.overlap_boundaries == 1
    assert report.gap_boundaries == 1
    assert report.max_overlap_bytes == 9
    assert report.max_gap_bytes == 9
    assert [block.byte_size for block in report.index.blocks] == sizes
    expected_offsets = []
    prefix = 0
    for size, shift in zip(sizes, report.block_shifts):
        expected_offsets.append(canonical_payload_start + prefix + shift)
        prefix += size
    assert [block.file_offset for block in report.index.blocks] == expected_offsets


def test_repair_rank_shifted_restart_writes_normal_layout(tmp_path):
    restart, particle, _canonical_payload_start, sizes = _write_shifted_outputs(tmp_path)
    output = tmp_path / "repaired.rst"

    report = repair_rank_shifted_restart(restart, particle, output)
    repaired = read_restart_index(output)

    assert report.block_shifts == (0, -9, 0, 0)
    assert [block.byte_size for block in repaired.blocks] == sizes
    assert repaired.payload_end == repaired.file_size


def test_recover_rank_shifted_index_from_embedded_particle_counts(tmp_path):
    restart, particle, canonical_payload_start, sizes = _write_shifted_outputs(tmp_path)
    # The synthetic fixture reserves 32 unrelated bytes after its payload.  A
    # production final.rst has only its declared payload/PRNG tail, so trim the
    # fixture to the exact format required by sidecar-free recovery.
    restart.write_bytes(restart.read_bytes()[: canonical_payload_start + sum(sizes)])

    strict = recover_rank_shifted_index(restart, particle)
    inferred = recover_rank_shifted_index_from_payload(restart)

    assert inferred.block_shifts == strict.block_shifts
    assert [block.byte_size for block in inferred.index.blocks] == sizes
    assert [block.file_offset for block in inferred.index.blocks] == [
        block.file_offset for block in strict.index.blocks
    ]


def test_validate_repaired_restart_without_sidecar(tmp_path):
    restart, _particle, canonical_payload_start, sizes = _write_shifted_outputs(tmp_path)
    restart.write_bytes(restart.read_bytes()[: canonical_payload_start + sum(sizes)])
    output = tmp_path / "repaired-without-sidecar.rst"

    repair_rank_shifted_restart(restart, None, output)
    report = validate_repaired_restart(output)

    assert report["blocks"] == 4
    assert report["payload_bytes"] == sum(sizes)


def _write_shifted_outputs(tmp_path):
    params = b"""<mesh>
nx1 = 8
nx2 = 1
nx3 = 1
<meshblock>
nx1 = 2
nx2 = 1
nx3 = 1
<particle1>
type = basic
<configure>
Magnetic_fields = OFF
Cosmic_Ray_Transport = OFF
Number_of_scalars = 0
<restart>
nint_user_mesh_data = 0
nreal_user_mesh_data = 0
magnetic_fields_enabled = 0
cr_enabled = 0
nscalars = 0
nint_user_meshblock_data = 0
nreal_user_meshblock_data = 1
real_user_meshblock_data_size_0 = 40
<par_end>
"""
    region = layout.RegionSize.from_bounds(
        x1min=0.0,
        x2min=0.0,
        x3min=0.0,
        x1max=8.0,
        x2max=1.0,
        x3max=1.0,
        nx1=8,
        nx2=1,
        nx3=1,
    )
    header = layout.MeshHeader(4, 2, region, 3.0, 0.1, 17)
    counts = [0, 1, 0, 2]
    nint, nreal = 2, 7
    particle_records = [_particle_record(npar, nint, nreal, idx) for idx, npar in enumerate(counts)]
    hydro_bytes = layout.bytes_for_real_array(layout.cell_centered_shape(layout.NHYDRO, (2, 1, 1)))
    payloads = [
        bytes([65 + idx]) * hydro_bytes + record + bytes([97 + idx]) * 40
        for idx, record in enumerate(particle_records)
    ]
    sizes = [len(payload) for payload in payloads]
    shifts = [0, -9, 0, 0]
    locations = [layout.LogicalLocation(idx, 0, 0, 2) for idx in range(4)]

    prefix = params + layout.pack_mesh_header(header)
    id_list_start = len(prefix)
    canonical_payload_start = id_list_start + 4 * layout.ID_LIST_RECORD_STRUCT.size
    total_payload = sum(sizes)
    image = bytearray(canonical_payload_start + total_payload + 32)
    image[: len(prefix)] = prefix
    payload_prefix = 0
    for idx, (loc, size, shift, payload) in enumerate(zip(locations, sizes, shifts, payloads)):
        record = layout.pack_id_record(loc, 1.0, size)
        record_start = id_list_start + idx * layout.ID_LIST_RECORD_STRUCT.size + shift
        image[record_start : record_start + len(record)] = record
        payload_start = canonical_payload_start + payload_prefix + shift
        image[payload_start : payload_start + size] = payload
        payload_prefix += size
    restart = tmp_path / "shifted.rst"
    restart.write_bytes(image)

    particle = tmp_path / "particles.parbin"
    particle.write_bytes(
        b"# synthetic\n"
        + b"# pid,\n"
        + struct.pack("<ii", nint, nreal)
        + struct.pack("<ii", header.nbtotal, header.root_level)
        + layout.pack_region_size(region)
        + struct.pack("<ddi", header.time, header.dt, header.ncycle)
        + b"".join(particle_records)
    )
    return restart, particle, canonical_payload_start, sizes


def _particle_record(npar, nint, nreal, block):
    header = struct.pack("<ii", npar, 100 + block)
    ints = struct.pack("<" + "i" * (npar * nint), *range(npar * nint))
    reals = struct.pack("<" + "d" * (npar * nreal), *range(npar * nreal))
    return header + ints + reals
