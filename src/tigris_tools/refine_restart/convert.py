from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from . import layout
from .param_block import ParameterBlock
from .reader import RestartReader
from .refine import refine_cell_centered, refine_face_x1, refine_face_x2, refine_face_x3

NCR = 4


@dataclass(frozen=True)
class RestartSchema:
    block_size: tuple[int, int, int]
    nscalars: int
    ncrg: int
    magnetic_fields: bool
    cr_enabled: bool
    particle_nint: int
    particle_nreal: int
    user_meshblock_tail_bytes: int
    prng_record_bytes: int


def refine_restart(
    input_path: str | Path,
    output_path: str | Path,
    factor: int = 2,
    *,
    block_size: tuple[int, int, int] | None = None,
    verbose: int = 0,
) -> None:
    if factor != 2:
        raise NotImplementedError("only --refine 2 is implemented")
    with RestartReader(input_path, verbose=verbose > 0) as reader:
        assert reader.index is not None
        index = reader.index
        schema = infer_schema(reader)
        out_block_size = block_size or schema.block_size
        chunks_per_parent = _chunks_per_parent(schema.block_size, out_block_size, factor)
        out_header = _refined_header(index.header, out_block_size, factor)
        out_params = _patched_params(
            index.params,
            out_header,
            out_block_size,
            schema.prng_record_bytes,
        )
        out_blocks_per_parent = chunks_per_parent[0] * chunks_per_parent[1] * chunks_per_parent[2]
        out_nbtotal = len(index.blocks) * out_blocks_per_parent
        output_path = Path(output_path)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        try:
            with tmp_path.open("wb") as out:
                out.write(out_params.encode("ascii"))
                out.write(layout.pack_mesh_header(out_header))
                out.write(index.user_mesh_data)
                idlist_pos = out.tell()
                out.write(b"\0" * (out_nbtotal * layout.ID_LIST_RECORD_STRUCT.size))
                records = _write_refined_payloads(
                    reader,
                    out,
                    schema,
                    out_block_size,
                    chunks_per_parent,
                    out_header.root_level,
                    factor,
                    verbose,
                )
                _write_prng(out, schema, out_nbtotal)
                out.seek(idlist_pos)
                for loc, size in records:
                    out.write(layout.pack_id_record(loc, 1.0, size))
            tmp_path.replace(output_path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise


def infer_schema(reader: RestartReader) -> RestartSchema:
    assert reader.index is not None
    index = reader.index
    params = index.params
    block_size = (
        params.get_int("meshblock", "nx1", index.header.mesh_size.nx1)
        or index.header.mesh_size.nx1,
        params.get_int("meshblock", "nx2", index.header.mesh_size.nx2)
        or index.header.mesh_size.nx2,
        params.get_int("meshblock", "nx3", index.header.mesh_size.nx3)
        or index.header.mesh_size.nx3,
    )
    configure = params.values.get("configure", {})
    restart = params.values.get("restart", {})
    nscalars = int(restart.get("nscalars", configure.get("Number_of_scalars", 0)))
    ncrg = int(restart.get("ncrg", configure.get("Cosmic_Ray_energy_groups", 1)))
    magnetic_fields = restart.get(
        "magnetic_fields_enabled", configure.get("Magnetic_fields", "OFF")
    ).lower() in {
        "true",
        "on",
        "1",
    }
    cr_enabled = restart.get(
        "cr_enabled", configure.get("Cosmic_Ray_Transport", "OFF")
    ).lower() not in {
        "false",
        "off",
        "none",
        "0",
    }
    particle_nint, particle_nreal = _particle_shape(params)
    fixed_before_particles, fixed_after_particles = _fixed_payload_sizes(
        block_size, nscalars, ncrg, magnetic_fields, cr_enabled
    )
    sample = reader.read_block_bytes(index.blocks[0])
    particle_offset = fixed_before_particles
    npar = struct.unpack_from("<i", sample, particle_offset)[0] if particle_nreal else 0
    particle_bytes = (
        0 if not particle_nreal else 8 + npar * (particle_nint * 4 + particle_nreal * 8)
    )
    tail = len(sample) - fixed_before_particles - particle_bytes - fixed_after_particles
    if tail < 0:
        raise ValueError("payload schema inference failed: negative user MeshBlock tail")
    prng_record_bytes = _infer_prng_record_bytes(index)
    return RestartSchema(
        block_size,
        nscalars,
        ncrg,
        magnetic_fields,
        cr_enabled,
        particle_nint,
        particle_nreal,
        tail,
        prng_record_bytes,
    )


def _write_refined_payloads(
    reader: RestartReader,
    out: BinaryIO,
    schema: RestartSchema,
    out_block_size: tuple[int, int, int],
    chunks_per_parent: tuple[int, int, int],
    out_root_level: int,
    factor: int,
    verbose: int,
) -> list[tuple[layout.LogicalLocation, int]]:
    assert reader.index is not None
    records: list[tuple[layout.LogicalLocation, int]] = []
    for parent_index, block in enumerate(reader.index.blocks):
        payload = _parse_payload(reader.read_block_bytes(block), schema)
        children = _split_refined_payload(
            payload,
            block.loc,
            reader.index.header,
            schema,
            out_block_size,
            chunks_per_parent,
            out_root_level,
            factor,
        )
        for loc, child_bytes in children:
            out.write(child_bytes)
            records.append((loc, len(child_bytes)))
        if verbose and (parent_index + 1) % 16 == 0:
            print(
                f"refine_restart.convert: refined parents={parent_index + 1}/{len(reader.index.blocks)}"
            )
    return records


def _parse_payload(data: bytes, schema: RestartSchema) -> dict[str, object]:
    bs = schema.block_size
    offset = 0
    hydro_shape = layout.cell_centered_shape(layout.NHYDRO, bs)
    hydro = _take_real(data, offset, hydro_shape)
    offset += hydro.nbytes
    bx1 = bx2 = bx3 = None
    if schema.magnetic_fields:
        s1, s2, s3 = layout.face_shapes(bs)
        bx1 = _take_real(data, offset, s1)
        offset += bx1.nbytes
        bx2 = _take_real(data, offset, s2)
        offset += bx2.nbytes
        bx3 = _take_real(data, offset, s3)
        offset += bx3.nbytes
    particles = None
    if schema.particle_nreal:
        npar, idmax = struct.unpack_from("<ii", data, offset)
        offset += 8
        ints = np.frombuffer(
            data, dtype=layout.INT_DTYPE, count=schema.particle_nint * npar, offset=offset
        ).copy()
        ints = ints.reshape(schema.particle_nint, npar)
        offset += ints.nbytes
        reals = np.frombuffer(
            data, dtype=layout.REAL_DTYPE, count=schema.particle_nreal * npar, offset=offset
        ).copy()
        reals = reals.reshape(schema.particle_nreal, npar)
        offset += reals.nbytes
        particles = (npar, idmax, ints, reals)
    cr = None
    if schema.cr_enabled:
        cr = _take_real(data, offset, (schema.ncrg, NCR, *layout.ghost_cell_shape(bs)))
        offset += cr.nbytes
    scalars = None
    if schema.nscalars:
        scalars = _take_real(data, offset, (schema.nscalars, *layout.ghost_cell_shape(bs)))
        offset += scalars.nbytes
    tail = data[offset : offset + schema.user_meshblock_tail_bytes]
    return {
        "hydro": hydro,
        "bx1": bx1,
        "bx2": bx2,
        "bx3": bx3,
        "particles": particles,
        "cr": cr,
        "scalars": scalars,
        "tail": tail,
    }


def _split_refined_payload(
    payload: dict[str, object],
    parent_loc: layout.LogicalLocation,
    header: layout.MeshHeader,
    schema: RestartSchema,
    out_block_size: tuple[int, int, int],
    chunks_per_parent: tuple[int, int, int],
    out_root_level: int,
    factor: int,
):
    refined = {
        "hydro": refine_cell_centered(payload["hydro"], factor),
        "cr": refine_cell_centered(payload["cr"], factor) if payload["cr"] is not None else None,
        "scalars": refine_cell_centered(payload["scalars"], factor)
        if payload["scalars"] is not None
        else None,
        "bx1": refine_face_x1(payload["bx1"], factor) if payload["bx1"] is not None else None,
        "bx2": refine_face_x2(payload["bx2"], factor) if payload["bx2"] is not None else None,
        "bx3": refine_face_x3(payload["bx3"], factor) if payload["bx3"] is not None else None,
    }
    children = []
    parent_bounds = _parent_bounds(parent_loc, header, schema.block_size)
    for cz in range(chunks_per_parent[2]):
        for cy in range(chunks_per_parent[1]):
            for cx in range(chunks_per_parent[0]):
                loc = layout.LogicalLocation(
                    parent_loc.lx1 * chunks_per_parent[0] + cx,
                    parent_loc.lx2 * chunks_per_parent[1] + cy,
                    parent_loc.lx3 * chunks_per_parent[2] + cz,
                    out_root_level,
                )
                bounds = _chunk_bounds(parent_bounds, chunks_per_parent, (cx, cy, cz))
                child_particles = (
                    _child_particles(payload["particles"], bounds)
                    if payload["particles"] is not None
                    else None
                )
                children.append(
                    (
                        loc,
                        _pack_child(
                            refined,
                            child_particles,
                            payload["tail"],
                            schema,
                            out_block_size,
                            (cx, cy, cz),
                        ),
                    )
                )
    return children


def _pack_child(
    refined: dict[str, object],
    particles,
    tail: bytes,
    schema: RestartSchema,
    out_block_size: tuple[int, int, int],
    child: tuple[int, int, int],
) -> bytes:
    chunks = [
        _child_cell(refined["hydro"], out_block_size, child).tobytes(order="C"),
    ]
    if schema.magnetic_fields:
        chunks.extend(
            [
                _child_face(refined["bx1"], out_block_size, child, 2).tobytes(order="C"),
                _child_face(refined["bx2"], out_block_size, child, 1).tobytes(order="C"),
                _child_face(refined["bx3"], out_block_size, child, 0).tobytes(order="C"),
            ]
        )
    if schema.particle_nreal:
        chunks.append(_pack_particles(particles, schema))
    if schema.cr_enabled:
        chunks.append(_child_cell(refined["cr"], out_block_size, child).tobytes(order="C"))
    if schema.nscalars:
        chunks.append(_child_cell(refined["scalars"], out_block_size, child).tobytes(order="C"))
    chunks.append(b"\0" * len(tail))
    return b"".join(chunks)


def _child_cell(array, block_size: tuple[int, int, int], child: tuple[int, int, int]) -> np.ndarray:
    cx, cy, cz = child
    sx = layout.NGHOST * 2 - layout.NGHOST + cx * block_size[0]
    sy = layout.NGHOST * 2 - layout.NGHOST + cy * block_size[1]
    sz = layout.NGHOST * 2 - layout.NGHOST + cz * block_size[2]
    return array[
        ...,
        sz : sz + block_size[2] + 2 * layout.NGHOST,
        sy : sy + block_size[1] + 2 * layout.NGHOST,
        sx : sx + block_size[0] + 2 * layout.NGHOST,
    ]


def _child_face(
    array,
    block_size: tuple[int, int, int],
    child: tuple[int, int, int],
    distinguished_axis: int,
) -> np.ndarray:
    cx, cy, cz = child
    sx = layout.NGHOST * 2 - layout.NGHOST + cx * block_size[0]
    sy = layout.NGHOST * 2 - layout.NGHOST + cy * block_size[1]
    sz = layout.NGHOST * 2 - layout.NGHOST + cz * block_size[2]
    lengths = [
        block_size[2] + 2 * layout.NGHOST,
        block_size[1] + 2 * layout.NGHOST,
        block_size[0] + 2 * layout.NGHOST,
    ]
    lengths[distinguished_axis] += 1
    return array[sz : sz + lengths[0], sy : sy + lengths[1], sx : sx + lengths[2]]


def _child_particles(
    particles,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
):
    npar, idmax, ints, reals = particles
    if npar == 0:
        return npar, idmax, ints[:, :0], reals[:, :0]
    mask = np.ones(npar, dtype=bool)
    for axis, real_idx in enumerate((1, 2, 3)):
        vals = reals[real_idx]
        lo, hi = bounds[axis]
        mask &= (vals >= lo) & (vals < hi)
    return int(mask.sum()), idmax, ints[:, mask], reals[:, mask]


def _parent_bounds(
    parent_loc: layout.LogicalLocation, header: layout.MeshHeader, block_size: tuple[int, int, int]
):
    ms = header.mesh_size
    nrb = (ms.nx1 // block_size[0], ms.nx2 // block_size[1], ms.nx3 // block_size[2])
    lengths = (ms.x1len / nrb[0], ms.x2len / nrb[1], ms.x3len / nrb[2])
    mins = (ms.x1min, ms.x2min, ms.x3min)
    locs = (parent_loc.lx1, parent_loc.lx2, parent_loc.lx3)
    return tuple(
        (mins[i] + locs[i] * lengths[i], mins[i] + (locs[i] + 1) * lengths[i]) for i in range(3)
    )


def _chunk_bounds(parent_bounds, chunks_per_parent, child):
    bounds = []
    for axis in range(3):
        lo, hi = parent_bounds[axis]
        width = (hi - lo) / chunks_per_parent[axis]
        c = child[axis]
        bounds.append((lo + c * width, lo + (c + 1) * width))
    return tuple(bounds)


def _pack_particles(particles, schema: RestartSchema) -> bytes:
    npar, idmax, ints, reals = particles
    return b"".join(
        [
            struct.pack("<ii", npar, idmax),
            ints.astype(layout.INT_DTYPE, copy=False).tobytes(order="C"),
            reals.astype(layout.REAL_DTYPE, copy=False).tobytes(order="C"),
        ]
    )


def _take_real(data: bytes, offset: int, shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape, dtype=np.int64))
    return (
        np.frombuffer(data, dtype=layout.REAL_DTYPE, count=count, offset=offset)
        .copy()
        .reshape(shape)
    )


def _fixed_payload_sizes(block_size, nscalars, ncrg, magnetic_fields, cr_enabled):
    before = layout.bytes_for_real_array(layout.cell_centered_shape(layout.NHYDRO, block_size))
    if magnetic_fields:
        before += sum(layout.bytes_for_real_array(s) for s in layout.face_shapes(block_size))
    after = 0
    if cr_enabled:
        after += layout.bytes_for_real_array((ncrg, NCR, *layout.ghost_cell_shape(block_size)))
    if nscalars:
        after += layout.bytes_for_real_array((nscalars, *layout.ghost_cell_shape(block_size)))
    return before, after


def _particle_shape(params: ParameterBlock) -> tuple[int, int]:
    particle_blocks = [
        v
        for k, v in sorted(params.values.items())
        if k.startswith("particle") and v.get("type", "none") != "none"
    ]
    if not particle_blocks:
        return 0, 0
    p = particle_blocks[0]
    nint = 2
    mesh = params.values.get("mesh", {})
    if mesh.get("ix1_bc") == "shear_periodic" or mesh.get("ox1_bc") == "shear_periodic":
        nint += 1
    nreal = 7
    if p.get("type") == "complex":
        nreal += 4
        if p.get("mass_return", "false").lower() == "true" and float(p.get("fgas", "0")) != 0.0:
            nreal += 1
        if p.get("feedback", "false").lower() == "true":
            nreal += 3
    return nint, nreal


def _patched_params(
    params: ParameterBlock,
    header: layout.MeshHeader,
    block_size: tuple[int, int, int],
    prng_record_bytes: int = 0,
) -> str:
    updates = {
        ("mesh", "nx1"): header.mesh_size.nx1,
        ("mesh", "nx2"): header.mesh_size.nx2,
        ("mesh", "nx3"): header.mesh_size.nx3,
        ("meshblock", "nx1"): block_size[0],
        ("meshblock", "nx2"): block_size[1],
        ("meshblock", "nx3"): block_size[2],
    }
    updates.update(_prng_parameter_updates(prng_record_bytes))
    return params.patched(updates)


def _refined_header(
    header: layout.MeshHeader,
    out_block_size: tuple[int, int, int],
    factor: int,
) -> layout.MeshHeader:
    ms = header.mesh_size
    mesh_size = layout.RegionSize(
        ms.x1min,
        ms.x2min,
        ms.x3min,
        ms.x1max,
        ms.x2max,
        ms.x3max,
        ms.x1len,
        ms.x2len,
        ms.x3len,
        ms.x1rat,
        ms.x2rat,
        ms.x3rat,
        ms.nx1 * factor,
        ms.nx2 * factor,
        ms.nx3 * factor,
    )
    for mesh_n, block_n in zip((mesh_size.nx1, mesh_size.nx2, mesh_size.nx3), out_block_size):
        if mesh_n % block_n != 0:
            raise ValueError(
                f"output mesh size {mesh_n} is not divisible by output block size {block_n}"
            )
    nrb = (
        mesh_size.nx1 // out_block_size[0],
        mesh_size.nx2 // out_block_size[1],
        mesh_size.nx3 // out_block_size[2],
    )
    nbtotal = nrb[0] * nrb[1] * nrb[2]
    root_level = _root_level_for_grid(nrb)
    return layout.MeshHeader(nbtotal, root_level, mesh_size, header.time, header.dt, header.ncycle)


def _chunks_per_parent(
    in_block_size: tuple[int, int, int],
    out_block_size: tuple[int, int, int],
    factor: int,
) -> tuple[int, int, int]:
    chunks = []
    for in_axis, out_axis in zip(in_block_size, out_block_size):
        refined = in_axis * factor
        if refined % out_axis != 0:
            raise ValueError(
                f"target block size {out_block_size} does not divide refined parent block "
                f"{tuple(n * factor for n in in_block_size)}"
            )
        chunks.append(refined // out_axis)
    return tuple(chunks)


def _root_level_for_grid(nrb: tuple[int, int, int]) -> int:
    nbmax = max(nrb)
    level = 0
    while (1 << level) < nbmax:
        level += 1
    return level


def _infer_prng_record_bytes(index) -> int:
    trailing = index.file_size - index.payload_end
    return (
        trailing // len(index.blocks) if trailing > 0 and trailing % len(index.blocks) == 0 else 0
    )


def _write_prng(out: BinaryIO, schema: RestartSchema, out_nbtotal: int) -> None:
    if schema.prng_record_bytes <= 0:
        return
    nprng = schema.prng_record_bytes // layout.PRNG_RECORD_STRUCT.size
    for rank in range(out_nbtotal):
        for p in range(nprng):
            seed = _prng_seed(rank, p)
            out.write(layout.PRNG_RECORD_STRUCT.pack(seed, 0))


def _prng_parameter_updates(prng_record_bytes: int) -> dict[tuple[str, str], object]:
    if prng_record_bytes <= 0:
        return {}
    if prng_record_bytes % layout.PRNG_RECORD_STRUCT.size != 0:
        raise ValueError(
            f"PRNG record size {prng_record_bytes} is not divisible by "
            f"{layout.PRNG_RECORD_STRUCT.size}"
        )
    nprng = prng_record_bytes // layout.PRNG_RECORD_STRUCT.size
    updates: dict[tuple[str, str], object] = {("random", "nrandom"): nprng}
    for index in range(nprng):
        # Keep the textual rank-0 state consistent with the first binary PRNG
        # record.  The 20-digit seed also fixes ParameterDump's column width on
        # every MPI rank, even after the binary state is loaded per rank.
        updates[(f"random{index}", "seed")] = _prng_seed(0, index)
        updates[(f"random{index}", "count")] = 0
    return updates


def _prng_seed(rank: int, index: int) -> int:
    return (0x9E3779B97F4A7C15 * (rank + 1) + index) & 0xFFFFFFFFFFFFFFFF
