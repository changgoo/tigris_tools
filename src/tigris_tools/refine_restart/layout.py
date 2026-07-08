from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable

import numpy as np


INT_SIZE = 4
INT_DTYPE = np.dtype("<i4")
UINT64_DTYPE = np.dtype("<u8")
REAL_SIZE = 8
REAL_DTYPE = np.dtype("<f8")
NHYDRO = 5
NGHOST = 4


LOGICAL_LOCATION_STRUCT = struct.Struct("<qqqi4x")
REGION_SIZE_STRUCT = struct.Struct("<12d3i4x")
MESH_HEADER_STRUCT = struct.Struct("<ii" + ("12d3i4x") + "ddi")
ID_LIST_RECORD_STRUCT = struct.Struct("<qqqi4xdQ")
PRNG_RECORD_STRUCT = struct.Struct("<QQ")


@dataclass(frozen=True)
class LogicalLocation:
    lx1: int
    lx2: int
    lx3: int
    level: int


@dataclass(frozen=True)
class RegionSize:
    x1min: float
    x2min: float
    x3min: float
    x1max: float
    x2max: float
    x3max: float
    x1len: float
    x2len: float
    x3len: float
    x1rat: float
    x2rat: float
    x3rat: float
    nx1: int
    nx2: int
    nx3: int

    @classmethod
    def from_bounds(
        cls,
        *,
        x1min: float,
        x2min: float,
        x3min: float,
        x1max: float,
        x2max: float,
        x3max: float,
        x1rat: float = 1.0,
        x2rat: float = 1.0,
        x3rat: float = 1.0,
        nx1: int,
        nx2: int,
        nx3: int,
    ) -> "RegionSize":
        return cls(
            x1min,
            x2min,
            x3min,
            x1max,
            x2max,
            x3max,
            x1max - x1min,
            x2max - x2min,
            x3max - x3min,
            x1rat,
            x2rat,
            x3rat,
            nx1,
            nx2,
            nx3,
        )


@dataclass(frozen=True)
class MeshHeader:
    nbtotal: int
    root_level: int
    mesh_size: RegionSize
    time: float
    dt: float
    ncycle: int


@dataclass(frozen=True)
class InputBlockDesc:
    loc: LogicalLocation
    cost: float
    byte_size: int
    file_offset: int


@dataclass
class OutputBlockDesc:
    loc: LogicalLocation
    cost: float
    byte_size: int
    parents: list[int]


@dataclass
class ParticleTable:
    npar: int
    idmax: int
    intprop: np.ndarray
    realprop: np.ndarray


@dataclass
class BlockPayload:
    hydro_u: np.ndarray
    field_bx1f: np.ndarray | None = None
    field_bx2f: np.ndarray | None = None
    field_bx3f: np.ndarray | None = None
    particles: list[ParticleTable] | None = None
    cr_u: np.ndarray | None = None
    scalars_s: np.ndarray | None = None
    ublock_int: list[np.ndarray] | None = None
    ublock_real: list[np.ndarray] | None = None


def pack_logical_location(loc: LogicalLocation) -> bytes:
    return LOGICAL_LOCATION_STRUCT.pack(loc.lx1, loc.lx2, loc.lx3, loc.level)


def unpack_logical_location(data: bytes, offset: int = 0) -> LogicalLocation:
    lx1, lx2, lx3, level = LOGICAL_LOCATION_STRUCT.unpack_from(data, offset)
    return LogicalLocation(lx1, lx2, lx3, level)


def pack_region_size(region: RegionSize) -> bytes:
    return REGION_SIZE_STRUCT.pack(
        region.x1min,
        region.x2min,
        region.x3min,
        region.x1max,
        region.x2max,
        region.x3max,
        region.x1len,
        region.x2len,
        region.x3len,
        region.x1rat,
        region.x2rat,
        region.x3rat,
        region.nx1,
        region.nx2,
        region.nx3,
    )


def unpack_region_size(data: bytes, offset: int = 0) -> RegionSize:
    values = REGION_SIZE_STRUCT.unpack_from(data, offset)
    return RegionSize(*values)


def pack_mesh_header(header: MeshHeader) -> bytes:
    r = header.mesh_size
    return MESH_HEADER_STRUCT.pack(
        header.nbtotal,
        header.root_level,
        r.x1min,
        r.x2min,
        r.x3min,
        r.x1max,
        r.x2max,
        r.x3max,
        r.x1len,
        r.x2len,
        r.x3len,
        r.x1rat,
        r.x2rat,
        r.x3rat,
        r.nx1,
        r.nx2,
        r.nx3,
        header.time,
        header.dt,
        header.ncycle,
    )


def unpack_mesh_header(data: bytes, offset: int = 0) -> MeshHeader:
    values = MESH_HEADER_STRUCT.unpack_from(data, offset)
    region = RegionSize(*values[2:17])
    return MeshHeader(values[0], values[1], region, values[17], values[18], values[19])


def pack_id_record(loc: LogicalLocation, cost: float, byte_size: int) -> bytes:
    return ID_LIST_RECORD_STRUCT.pack(loc.lx1, loc.lx2, loc.lx3, loc.level, cost, byte_size)


def unpack_id_record(data: bytes, offset: int = 0) -> tuple[LogicalLocation, float, int]:
    lx1, lx2, lx3, level, cost, byte_size = ID_LIST_RECORD_STRUCT.unpack_from(data, offset)
    return LogicalLocation(lx1, lx2, lx3, level), cost, byte_size


def ghost_cell_shape(block_size: tuple[int, int, int]) -> tuple[int, int, int]:
    nx1, nx2, nx3 = block_size
    ncells1 = nx1 + 2 * NGHOST
    ncells2 = nx2 + 2 * NGHOST if nx2 > 1 else 1
    ncells3 = nx3 + 2 * NGHOST if nx3 > 1 else 1
    return ncells3, ncells2, ncells1


def face_shapes(block_size: tuple[int, int, int]) -> tuple[tuple[int, ...], ...]:
    ncells3, ncells2, ncells1 = ghost_cell_shape(block_size)
    return (
        (ncells3, ncells2, ncells1 + 1),
        (ncells3, ncells2 + 1, ncells1),
        (ncells3 + 1, ncells2, ncells1),
    )


def cell_centered_shape(nvar: int, block_size: tuple[int, int, int]) -> tuple[int, int, int, int]:
    ncells3, ncells2, ncells1 = ghost_cell_shape(block_size)
    return nvar, ncells3, ncells2, ncells1


def bytes_for_real_array(shape: Iterable[int]) -> int:
    return int(np.prod(tuple(shape), dtype=np.int64)) * REAL_SIZE


def bytes_for_int_array(shape: Iterable[int]) -> int:
    return int(np.prod(tuple(shape), dtype=np.int64)) * INT_SIZE


def particle_table_size(npar: int, nint: int, nreal: int) -> int:
    return 2 * INT_SIZE + npar * (nint * INT_SIZE + nreal * REAL_SIZE)


def assert_layout_sizes() -> None:
    if LOGICAL_LOCATION_STRUCT.size != 32:
        raise AssertionError(f"LogicalLocation size {LOGICAL_LOCATION_STRUCT.size} != 32")
    if REGION_SIZE_STRUCT.size != 112:
        raise AssertionError(f"RegionSize size {REGION_SIZE_STRUCT.size} != 112")
    if MESH_HEADER_STRUCT.size != 140:
        raise AssertionError(f"MeshHeader size {MESH_HEADER_STRUCT.size} != 140")
    if ID_LIST_RECORD_STRUCT.size != 48:
        raise AssertionError(f"ID-list record size {ID_LIST_RECORD_STRUCT.size} != 48")


assert_layout_sizes()
