from __future__ import annotations

import math
import os
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tigris_tools.refine_restart import layout
from tigris_tools.refine_restart.convert import _fixed_payload_sizes, _particle_shape
from tigris_tools.refine_restart.param_block import read_parameter_block
from tigris_tools.refine_restart.reader import RestartFormatError, RestartIndex


@dataclass(frozen=True)
class ParticleSidecar:
    nint: int
    nreal: int
    header: layout.MeshHeader
    records: tuple[bytes, ...]
    counts: tuple[int, ...]


@dataclass(frozen=True)
class RecoveryReport:
    index: RestartIndex
    block_shifts: tuple[int, ...]
    candidate_shifts: tuple[int, ...]
    exact_id_records: int
    overlap_boundaries: int
    gap_boundaries: int
    max_overlap_bytes: int
    max_gap_bytes: int


def recover_rank_shifted_index(
    restart_path: str | Path,
    particle_path: str | Path,
    *,
    max_shift: int = 256,
) -> RecoveryReport:
    """Recover this run's rank-shifted ID table using its paired particle dump.

    The damaged files were written with one rank-dependent header displacement
    per block.  The paired parbin output supplies an independent block particle
    count, and therefore the exact restart payload size for every block.  The
    recovery is accepted only when every embedded particle record matches the
    sidecar byte-for-byte at the inferred payload offset.

    This is intentionally strict and specialized.  It does not guess through
    ambiguous damage and is not used by the normal restart reader.
    """

    restart_path = Path(restart_path)
    sidecar = read_particle_sidecar(particle_path)
    with restart_path.open("rb") as stream:
        params = read_parameter_block(stream)
        header = layout.unpack_mesh_header(
            _read_exact(stream, layout.MESH_HEADER_STRUCT.size, "mesh header")
        )
        user_mesh_data = _read_exact(
            stream,
            _user_mesh_data_size(params),
            "user mesh data",
        )
        id_list_start = stream.tell()

    _validate_sidecar_header(header, sidecar.header)
    locations = _uniform_root_locations(params, header)
    if len(locations) != header.nbtotal:
        raise RestartFormatError(
            "rank-shift recovery currently requires a uniform root grid without AMR"
        )

    particle_nint, particle_nreal = _particle_shape(params)
    if (particle_nint, particle_nreal) != (sidecar.nint, sidecar.nreal):
        raise RestartFormatError(
            "particle schema mismatch: restart expects "
            f"({particle_nint}, {particle_nreal}), sidecar has "
            f"({sidecar.nint}, {sidecar.nreal})"
        )
    base_payload_size, particle_offset = _base_payload_size(params, header)
    particle_stride = sidecar.nint * layout.INT_SIZE + sidecar.nreal * layout.REAL_SIZE
    sizes = tuple(base_payload_size + count * particle_stride for count in sidecar.counts)

    location_to_index = {loc: idx for idx, loc in enumerate(locations)}
    scan_start = max(0, id_list_start - max_shift)
    scan_end = id_list_start + header.nbtotal * layout.ID_LIST_RECORD_STRUCT.size + max_shift
    with restart_path.open("rb") as stream:
        stream.seek(scan_start)
        id_window = stream.read(scan_end - scan_start)

    candidate_shifts, modal_cost = _find_candidate_shifts(
        id_window,
        window_start=scan_start,
        id_list_start=id_list_start,
        location_to_index=location_to_index,
        sizes=sizes,
        root_level=header.root_level,
        max_shift=max_shift,
    )
    if not candidate_shifts:
        raise RestartFormatError("could not find any complete displaced ID records")

    shifts, exact_id_records = _choose_block_shifts(
        id_window,
        window_start=scan_start,
        id_list_start=id_list_start,
        locations=locations,
        sizes=sizes,
        cost=modal_cost,
        candidates=candidate_shifts,
    )
    list_bytes = header.nbtotal * layout.ID_LIST_RECORD_STRUCT.size
    canonical_payload_start = id_list_start + list_bytes
    prefix = 0
    blocks: list[layout.InputBlockDesc] = []
    for loc, size, shift in zip(locations, sizes, shifts):
        blocks.append(
            layout.InputBlockDesc(loc, modal_cost, size, canonical_payload_start + prefix + shift)
        )
        prefix += size

    _validate_particle_records(restart_path, blocks, sidecar.records, particle_offset)
    overlaps = [
        shifts[idx] - shifts[idx + 1]
        for idx in range(len(shifts) - 1)
        if shifts[idx + 1] < shifts[idx]
    ]
    gaps = [
        shifts[idx + 1] - shifts[idx]
        for idx in range(len(shifts) - 1)
        if shifts[idx + 1] > shifts[idx]
    ]
    file_size = restart_path.stat().st_size
    payload_end = max(block.file_offset + block.byte_size for block in blocks)
    index = RestartIndex(
        params=params,
        header=header,
        user_mesh_data=user_mesh_data,
        blocks=blocks,
        payload_start=min(block.file_offset for block in blocks),
        payload_end=payload_end,
        file_size=file_size,
    )
    return RecoveryReport(
        index=index,
        block_shifts=shifts,
        candidate_shifts=candidate_shifts,
        exact_id_records=exact_id_records,
        overlap_boundaries=len(overlaps),
        gap_boundaries=len(gaps),
        max_overlap_bytes=max(overlaps, default=0),
        max_gap_bytes=max(gaps, default=0),
    )


def recover_rank_shifted_index_from_payload(
    restart_path: str | Path,
    *,
    max_shift: int = 256,
) -> RecoveryReport:
    """Recover a displaced restart when no same-cycle particle dump exists.

    This is the fallback needed for ``final.rst`` files written between regular
    particle outputs.  Candidate rank shifts are inferred only from complete
    logical-location/cost prefixes in the ID table.  The payload size of every
    block is then independently reconstructed from its embedded particle count.
    Recovery is accepted only when all 4096-style records select a unique shift,
    every particle-derived block fits in the file, and the resulting payload plus
    the per-rank PRNG state consumes the file exactly.

    A same-cycle particle sidecar remains preferable because it permits a
    byte-for-byte comparison of every embedded particle record.
    """

    restart_path = Path(restart_path)
    with restart_path.open("rb") as stream:
        params = read_parameter_block(stream)
        header = layout.unpack_mesh_header(
            _read_exact(stream, layout.MESH_HEADER_STRUCT.size, "mesh header")
        )
        user_mesh_data = _read_exact(
            stream,
            _user_mesh_data_size(params),
            "user mesh data",
        )
        id_list_start = stream.tell()

    locations = _uniform_root_locations(params, header)
    if len(locations) != header.nbtotal:
        raise RestartFormatError(
            "rank-shift recovery currently requires a uniform root grid without AMR"
        )
    particle_nint, particle_nreal = _particle_shape(params)
    if particle_nreal <= 0:
        raise RestartFormatError(
            "sidecar-free recovery requires embedded particle records"
        )
    base_payload_size, particle_offset = _base_payload_size(params, header)
    particle_stride = particle_nint * layout.INT_SIZE + particle_nreal * layout.REAL_SIZE
    if particle_stride <= 0:
        raise RestartFormatError("invalid embedded particle stride")

    scan_start = max(0, id_list_start - max_shift)
    scan_end = id_list_start + header.nbtotal * layout.ID_LIST_RECORD_STRUCT.size + max_shift
    with restart_path.open("rb") as stream:
        stream.seek(scan_start)
        id_window = stream.read(scan_end - scan_start)

    candidate_shifts, modal_cost = _find_candidate_shifts_from_locations(
        id_window,
        window_start=scan_start,
        id_list_start=id_list_start,
        locations=locations,
        root_level=header.root_level,
        max_shift=max_shift,
    )
    if not candidate_shifts:
        raise RestartFormatError("could not find any complete displaced ID locations")
    shifts = _choose_block_shifts_from_prefixes(
        id_window,
        window_start=scan_start,
        id_list_start=id_list_start,
        locations=locations,
        cost=modal_cost,
        candidates=candidate_shifts,
    )

    list_bytes = header.nbtotal * layout.ID_LIST_RECORD_STRUCT.size
    canonical_payload_start = id_list_start + list_bytes
    file_size = restart_path.stat().st_size
    descriptor = os.open(restart_path, os.O_RDONLY)
    prefix = 0
    exact_id_records = 0
    blocks: list[layout.InputBlockDesc] = []
    try:
        for idx, (loc, shift) in enumerate(zip(locations, shifts)):
            file_offset = canonical_payload_start + prefix + shift
            particle_header = os.pread(
                descriptor,
                8,
                file_offset + particle_offset,
            )
            if len(particle_header) != 8:
                raise EOFError(f"truncated embedded particle header for block {idx}")
            npar, _idmax = struct.unpack("<ii", particle_header)
            if npar < 0:
                raise RestartFormatError(
                    f"negative embedded particle count {npar} for block {idx}"
                )
            byte_size = base_payload_size + npar * particle_stride
            if file_offset < canonical_payload_start - max_shift:
                raise RestartFormatError(f"invalid recovered payload offset for block {idx}")
            if file_offset + byte_size > file_size:
                raise EOFError(
                    f"recovered payload {idx} extends past EOF: "
                    f"end={file_offset + byte_size}, file_size={file_size}"
                )
            blocks.append(layout.InputBlockDesc(loc, modal_cost, byte_size, file_offset))
            record_offset = id_list_start + idx * layout.ID_LIST_RECORD_STRUCT.size + shift
            relative = record_offset - scan_start
            actual = id_window[
                relative : relative + layout.ID_LIST_RECORD_STRUCT.size
            ]
            expected = layout.pack_id_record(loc, modal_cost, byte_size)
            score = sum(left == right for left, right in zip(expected, actual))
            if score < 3 * layout.ID_LIST_RECORD_STRUCT.size // 4:
                raise RestartFormatError(
                    f"payload-derived ID record {idx} has only "
                    f"{score}/{layout.ID_LIST_RECORD_STRUCT.size} matching bytes"
                )
            exact_id_records += score == layout.ID_LIST_RECORD_STRUCT.size
            prefix += byte_size
    finally:
        os.close(descriptor)

    nrandom = params.get_int("random", "nrandom", 0) or 0
    prng_bytes = header.nbtotal * nrandom * layout.PRNG_RECORD_STRUCT.size
    expected_file_size = canonical_payload_start + prefix + shifts[-1] + prng_bytes
    if expected_file_size != file_size:
        raise RestartFormatError(
            "particle-derived payload sizes do not consume the restart exactly: "
            f"expected EOF {expected_file_size}, actual EOF {file_size}"
        )

    overlaps = [
        shifts[idx] - shifts[idx + 1]
        for idx in range(len(shifts) - 1)
        if shifts[idx + 1] < shifts[idx]
    ]
    gaps = [
        shifts[idx + 1] - shifts[idx]
        for idx in range(len(shifts) - 1)
        if shifts[idx + 1] > shifts[idx]
    ]
    payload_end = max(block.file_offset + block.byte_size for block in blocks)
    index = RestartIndex(
        params=params,
        header=header,
        user_mesh_data=user_mesh_data,
        blocks=blocks,
        payload_start=min(block.file_offset for block in blocks),
        payload_end=payload_end,
        file_size=file_size,
    )
    return RecoveryReport(
        index=index,
        block_shifts=shifts,
        candidate_shifts=candidate_shifts,
        exact_id_records=exact_id_records,
        overlap_boundaries=len(overlaps),
        gap_boundaries=len(gaps),
        max_overlap_bytes=max(overlaps, default=0),
        max_gap_bytes=max(gaps, default=0),
    )


def read_particle_sidecar(path: str | Path) -> ParticleSidecar:
    path = Path(path)
    with path.open("rb") as stream:
        try:
            stream.readline().decode("utf-8")
            stream.readline().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RestartFormatError(f"invalid parbin text header in {path}") from exc
        nint, nreal = struct.unpack("<ii", _read_exact(stream, 8, "particle schema"))
        nbtotal, root_level = struct.unpack("<ii", _read_exact(stream, 8, "particle mesh"))
        region = layout.unpack_region_size(
            _read_exact(stream, layout.REGION_SIZE_STRUCT.size, "particle mesh region")
        )
        time, dt, ncycle = struct.unpack("<ddi", _read_exact(stream, 20, "particle time"))
        header = layout.MeshHeader(nbtotal, root_level, region, time, dt, ncycle)
        records: list[bytes] = []
        counts: list[int] = []
        stride = nint * layout.INT_SIZE + nreal * layout.REAL_SIZE
        for idx in range(nbtotal):
            record_header = _read_exact(stream, 8, f"particle block {idx} header")
            npar, _idmax = struct.unpack("<ii", record_header)
            if npar < 0:
                raise RestartFormatError(f"negative particle count in sidecar block {idx}")
            body = _read_exact(stream, npar * stride, f"particle block {idx} data")
            records.append(record_header + body)
            counts.append(npar)
    return ParticleSidecar(nint, nreal, header, tuple(records), tuple(counts))


def _find_candidate_shifts(
    data: bytes,
    *,
    window_start: int,
    id_list_start: int,
    location_to_index: dict[layout.LogicalLocation, int],
    sizes: tuple[int, ...],
    root_level: int,
    max_shift: int,
) -> tuple[tuple[int, ...], float]:
    hits: list[tuple[int, float]] = []
    record_size = layout.ID_LIST_RECORD_STRUCT.size
    for relative in range(len(data) - record_size + 1):
        loc, cost, byte_size = layout.unpack_id_record(data, relative)
        idx = location_to_index.get(loc)
        if idx is None or loc.level != root_level or byte_size != sizes[idx]:
            continue
        if not math.isfinite(cost) or cost <= 0.0:
            continue
        absolute = window_start + relative
        shift = absolute - (id_list_start + idx * record_size)
        if abs(shift) <= max_shift:
            hits.append((shift, cost))
    if not hits:
        return (), 0.0
    shifts = tuple(sorted({shift for shift, _cost in hits}))
    cost = Counter(cost for _shift, cost in hits).most_common(1)[0][0]
    return shifts, cost


def _find_candidate_shifts_from_locations(
    data: bytes,
    *,
    window_start: int,
    id_list_start: int,
    locations: tuple[layout.LogicalLocation, ...],
    root_level: int,
    max_shift: int,
) -> tuple[tuple[int, ...], float]:
    location_to_index = {loc: idx for idx, loc in enumerate(locations)}
    hits: list[tuple[int, float]] = []
    record_size = layout.ID_LIST_RECORD_STRUCT.size
    for relative in range(len(data) - record_size + 1):
        loc, cost, _byte_size = layout.unpack_id_record(data, relative)
        idx = location_to_index.get(loc)
        if idx is None or loc.level != root_level:
            continue
        if not math.isfinite(cost) or cost <= 0.0:
            continue
        absolute = window_start + relative
        shift = absolute - (id_list_start + idx * record_size)
        if abs(shift) <= max_shift:
            hits.append((shift, cost))
    if not hits:
        return (), 0.0
    shifts = tuple(sorted({shift for shift, _cost in hits}))
    cost = Counter(cost for _shift, cost in hits).most_common(1)[0][0]
    return shifts, cost


def _choose_block_shifts_from_prefixes(
    data: bytes,
    *,
    window_start: int,
    id_list_start: int,
    locations: tuple[layout.LogicalLocation, ...],
    cost: float,
    candidates: tuple[int, ...],
) -> tuple[int, ...]:
    shifts: list[int] = []
    record_size = layout.ID_LIST_RECORD_STRUCT.size
    for idx, loc in enumerate(locations):
        expected = layout.pack_logical_location(loc) + struct.pack("<d", cost)
        scored: list[tuple[int, int]] = []
        for shift in candidates:
            relative = id_list_start + idx * record_size + shift - window_start
            actual = data[relative : relative + len(expected)]
            score = sum(left == right for left, right in zip(expected, actual))
            scored.append((score, shift))
        scored.sort(reverse=True)
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            raise RestartFormatError(f"ambiguous displaced ID prefix {idx}")
        if scored[0][0] < 3 * len(expected) // 4:
            raise RestartFormatError(
                f"displaced ID prefix {idx} has only "
                f"{scored[0][0]}/{len(expected)} matching bytes"
            )
        shifts.append(scored[0][1])
    return tuple(shifts)


def _choose_block_shifts(
    data: bytes,
    *,
    window_start: int,
    id_list_start: int,
    locations: tuple[layout.LogicalLocation, ...],
    sizes: tuple[int, ...],
    cost: float,
    candidates: tuple[int, ...],
) -> tuple[tuple[int, ...], int]:
    shifts: list[int] = []
    exact = 0
    record_size = layout.ID_LIST_RECORD_STRUCT.size
    for idx, (loc, size) in enumerate(zip(locations, sizes)):
        expected = layout.pack_id_record(loc, cost, size)
        scored: list[tuple[int, int]] = []
        for shift in candidates:
            relative = id_list_start + idx * record_size + shift - window_start
            actual = data[relative : relative + record_size]
            score = sum(left == right for left, right in zip(expected, actual))
            scored.append((score, shift))
        scored.sort(reverse=True)
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            raise RestartFormatError(f"ambiguous displaced ID record {idx}")
        if scored[0][0] < 3 * record_size // 4:
            raise RestartFormatError(
                f"displaced ID record {idx} has only {scored[0][0]}/{record_size} matching bytes"
            )
        shifts.append(scored[0][1])
        exact += scored[0][0] == record_size
    return tuple(shifts), exact


def _validate_particle_records(
    restart_path: Path,
    blocks: list[layout.InputBlockDesc],
    expected_records: tuple[bytes, ...],
    particle_offset: int,
) -> None:
    descriptor = os.open(restart_path, os.O_RDONLY)
    try:
        for idx, (block, expected) in enumerate(zip(blocks, expected_records)):
            actual = os.pread(
                descriptor,
                len(expected),
                block.file_offset + particle_offset,
            )
            if actual != expected:
                raise RestartFormatError(
                    f"recovered payload offset failed particle validation for block {idx}"
                )
    finally:
        os.close(descriptor)


def _base_payload_size(params, header: layout.MeshHeader) -> tuple[int, int]:
    restart = params.values.get("restart", {})
    configure = params.values.get("configure", {})
    block_size = (
        params.get_int("meshblock", "nx1", header.mesh_size.nx1) or header.mesh_size.nx1,
        params.get_int("meshblock", "nx2", header.mesh_size.nx2) or header.mesh_size.nx2,
        params.get_int("meshblock", "nx3", header.mesh_size.nx3) or header.mesh_size.nx3,
    )
    nscalars = int(restart.get("nscalars", configure.get("Number_of_scalars", 0)))
    ncrg = int(restart.get("ncrg", configure.get("Cosmic_Ray_energy_groups", 1)))
    magnetic = str(restart.get("magnetic_fields_enabled", "false")).lower() in {
        "1",
        "on",
        "true",
    }
    cr_enabled = str(restart.get("cr_enabled", "false")).lower() in {"1", "on", "true"}
    before, after = _fixed_payload_sizes(block_size, nscalars, ncrg, magnetic, cr_enabled)
    tail = _user_meshblock_data_size(params)
    return before + 8 + after + tail, before


def _uniform_root_locations(
    params, header: layout.MeshHeader
) -> tuple[layout.LogicalLocation, ...]:
    block_size = (
        params.get_int("meshblock", "nx1", header.mesh_size.nx1) or header.mesh_size.nx1,
        params.get_int("meshblock", "nx2", header.mesh_size.nx2) or header.mesh_size.nx2,
        params.get_int("meshblock", "nx3", header.mesh_size.nx3) or header.mesh_size.nx3,
    )
    mesh_size = (header.mesh_size.nx1, header.mesh_size.nx2, header.mesh_size.nx3)
    if any(mesh % block != 0 for mesh, block in zip(mesh_size, block_size)):
        raise RestartFormatError("mesh is not divisible by its root meshblock size")
    grid = tuple(mesh // block for mesh, block in zip(mesh_size, block_size))
    active_dims = 3 if grid[2] > 1 else 2 if grid[1] > 1 else 1
    nleaf = 1 << active_dims
    locations: list[layout.LogicalLocation] = []

    def visit(level: int, lx1: int, lx2: int, lx3: int) -> None:
        if level == header.root_level:
            locations.append(layout.LogicalLocation(lx1, lx2, lx3, level))
            return
        level_factor = 1 << (header.root_level - level - 1)
        for child in range(nleaf):
            i = child & 1
            j = (child >> 1) & 1
            k = (child >> 2) & 1
            child_coords = (lx1 * 2 + i, lx2 * 2 + j, lx3 * 2 + k)
            if all(coord * level_factor < limit for coord, limit in zip(child_coords, grid)):
                visit(level + 1, *child_coords)

    visit(0, 0, 0, 0)
    return tuple(locations)


def _user_mesh_data_size(params) -> int:
    restart = params.values.get("restart", {})
    return _listed_array_bytes(restart, "user_mesh_data")


def _user_meshblock_data_size(params) -> int:
    restart = params.values.get("restart", {})
    return _listed_array_bytes(restart, "user_meshblock_data")


def _listed_array_bytes(restart: dict[str, str], stem: str) -> int:
    total = 0
    for kind in ("int", "real"):
        count = int(restart.get(f"n{kind}_{stem}", 0))
        for idx in range(count):
            key = f"{kind}_{stem}_size_{idx}"
            if key not in restart:
                raise RestartFormatError(f"missing <restart>/{key}")
            total += int(restart[key])
    return total


def _validate_sidecar_header(restart: layout.MeshHeader, particle: layout.MeshHeader) -> None:
    if restart.nbtotal != particle.nbtotal or restart.root_level != particle.root_level:
        raise RestartFormatError("restart and particle sidecar mesh headers do not match")
    if restart.ncycle != particle.ncycle:
        raise RestartFormatError(
            f"restart cycle {restart.ncycle} does not match particle cycle {particle.ncycle}"
        )
    tolerance = 1.0e-10 * max(1.0, abs(restart.time))
    if abs(restart.time - particle.time) > tolerance:
        raise RestartFormatError(
            f"restart time {restart.time} does not match particle time {particle.time}"
        )


def _read_exact(stream, size: int, label: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise EOFError(f"truncated {label}: expected {size}, got {len(data)}")
    return data
