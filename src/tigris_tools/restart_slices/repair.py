from __future__ import annotations

import math
import os
import struct
from pathlib import Path
from typing import TextIO

import numpy as np

from tigris_tools.refine_restart import layout
from tigris_tools.refine_restart.convert import (
    RestartSchema,
    _fixed_payload_sizes,
    _parse_payload,
    _patched_params,
    _write_prng,
    infer_schema,
)
from tigris_tools.refine_restart.reader import (
    RestartFormatError,
    RestartReader,
    read_restart_index,
)

from .recovery import (
    RecoveryReport,
    _user_meshblock_data_size,
    recover_rank_shifted_index,
    recover_rank_shifted_index_from_payload,
)


def repair_rank_shifted_restart(
    input_path: str | Path,
    particle_path: str | Path | None,
    output_path: str | Path,
    *,
    verbose: bool = False,
    log: TextIO | None = None,
) -> RecoveryReport:
    """Rewrite a rank-shifted restart into the canonical TIGRESS++ layout.

    The original file is never modified and an existing output is never
    overwritten.  Recovery is first validated against the same-cycle particle
    sidecar when one is available.  Without a sidecar, block sizes are recovered
    from the embedded particle headers and accepted only if they consume the file
    exactly.  The repaired file receives a fixed ID table, contiguous payloads,
    synchronized textual/binary PRNG state, and safe values for bytes that may
    have participated in overlapping MPI writes.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("repair output must differ from the input restart")

    if particle_path is None:
        report = recover_rank_shifted_index_from_payload(input_path)
    else:
        report = recover_rank_shifted_index(input_path, particle_path)
    index = report.index
    params = index.params
    block_size = (
        params.get_int("meshblock", "nx1", index.header.mesh_size.nx1)
        or index.header.mesh_size.nx1,
        params.get_int("meshblock", "nx2", index.header.mesh_size.nx2)
        or index.header.mesh_size.nx2,
        params.get_int("meshblock", "nx3", index.header.mesh_size.nx3)
        or index.header.mesh_size.nx3,
    )
    nrandom = params.get_int("random", "nrandom", 0) or 0
    prng_record_bytes = nrandom * layout.PRNG_RECORD_STRUCT.size
    output_params = _patched_params(
        params,
        index.header,
        block_size,
        prng_record_bytes,
    )
    tail_bytes = _user_meshblock_data_size(params)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        raise FileExistsError(f"refusing to overwrite temporary output: {tmp_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with input_path.open("rb") as source, tmp_path.open("xb") as output:
            output.write(output_params.encode("ascii"))
            output.write(layout.pack_mesh_header(index.header))
            output.write(index.user_mesh_data)
            for block in index.blocks:
                output.write(layout.pack_id_record(block.loc, block.cost, block.byte_size))

            for block_index, block in enumerate(index.blocks):
                source.seek(block.file_offset)
                payload = bytearray(source.read(block.byte_size))
                if len(payload) != block.byte_size:
                    raise EOFError(
                        f"truncated recovered payload {block_index}: "
                        f"expected {block.byte_size}, got {len(payload)}"
                    )
                _repair_overlap_edges(
                    payload,
                    block_index,
                    report.block_shifts,
                    tail_bytes,
                )
                output.write(payload)
                if verbose and (block_index + 1) % 128 == 0:
                    message = f"repair_restart: copied {block_index + 1}/{len(index.blocks)} blocks"
                    print(message, file=log)

            schema = RestartSchema(
                block_size=block_size,
                nscalars=0,
                ncrg=0,
                magnetic_fields=False,
                cr_enabled=False,
                particle_nint=0,
                particle_nreal=0,
                user_meshblock_tail_bytes=tail_bytes,
                prng_record_bytes=prng_record_bytes,
            )
            _write_prng(output, schema, len(index.blocks))
            output.flush()
            os.fsync(output.fileno())
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    repaired = read_restart_index(output_path)
    if repaired.header != index.header or len(repaired.blocks) != len(index.blocks):
        raise RestartFormatError("repaired restart failed structural verification")
    if [block.byte_size for block in repaired.blocks] != [
        block.byte_size for block in index.blocks
    ]:
        raise RestartFormatError("repaired restart block sizes differ from recovered input")
    return report


def validate_repaired_restart(
    restart_path: str | Path,
    particle_path: str | Path | None = None,
    *,
    check_payloads: bool = True,
) -> dict[str, int]:
    """Strictly validate a canonical repaired restart.

    The fixed ID table is checked with the ordinary reader.  When a particle
    sidecar exists, all embedded particle records are compared byte-for-byte;
    otherwise the payload-derived recovery checks exact file consumption.  A
    full validation also reads and parses every meshblock and rejects non-finite
    physical arrays or an inconsistent embedded-particle size.
    """

    restart_path = Path(restart_path)
    index = read_restart_index(restart_path)
    if particle_path is None:
        recovery = recover_rank_shifted_index_from_payload(restart_path)
    else:
        recovery = recover_rank_shifted_index(restart_path, particle_path)
    if set(recovery.block_shifts) != {0}:
        raise RestartFormatError(
            f"repaired restart still has displaced records: {recovery.candidate_shifts}"
        )
    if recovery.exact_id_records != index.header.nbtotal:
        raise RestartFormatError(
            "repaired restart does not have an exact canonical ID record for every block"
        )
    if not check_payloads:
        return {"blocks": len(index.blocks), "payload_bytes": 0, "finite_arrays": 0}

    payload_bytes = 0
    finite_arrays = 0
    with RestartReader(restart_path) as reader:
        assert reader.index is not None
        schema = infer_schema(reader)
        fixed_before, fixed_after = _fixed_payload_sizes(
            schema.block_size,
            schema.nscalars,
            schema.ncrg,
            schema.magnetic_fields,
            schema.cr_enabled,
        )
        particle_stride = (
            schema.particle_nint * layout.INT_SIZE
            + schema.particle_nreal * layout.REAL_SIZE
        )
        for block_number, block in enumerate(reader.index.blocks):
            data = reader.read_block_bytes(block)
            if schema.particle_nreal:
                if len(data) < fixed_before + 8:
                    raise RestartFormatError(
                        f"payload {block_number} is shorter than its particle header"
                    )
                npar = struct.unpack_from("<i", data, fixed_before)[0]
                if npar < 0:
                    raise RestartFormatError(
                        f"payload {block_number} has negative particle count {npar}"
                    )
                particle_bytes = 8 + npar * particle_stride
            else:
                particle_bytes = 0
            expected_size = (
                fixed_before
                + particle_bytes
                + fixed_after
                + schema.user_meshblock_tail_bytes
            )
            if expected_size != block.byte_size:
                raise RestartFormatError(
                    f"payload {block_number} size mismatch: "
                    f"ID={block.byte_size}, schema={expected_size}"
                )
            payload = _parse_payload(data, schema)
            arrays = [
                payload[name]
                for name in ("hydro", "bx1", "bx2", "bx3", "cr", "scalars")
                if payload[name] is not None
            ]
            particles = payload["particles"]
            if particles is not None:
                arrays.extend((particles[2], particles[3]))
            for values in arrays:
                if not np.isfinite(values).all():
                    raise RestartFormatError(
                        f"payload {block_number} contains non-finite array values"
                    )
                finite_arrays += 1
            payload_bytes += len(data)
    return {
        "blocks": len(index.blocks),
        "payload_bytes": payload_bytes,
        "finite_arrays": finite_arrays,
    }


def _repair_overlap_edges(
    payload: bytearray,
    block_index: int,
    shifts: tuple[int, ...],
    tail_bytes: int,
) -> None:
    incoming = 0
    if block_index > 0:
        incoming = max(0, shifts[block_index - 1] - shifts[block_index])
    if incoming:
        affected_reals = math.ceil(incoming / layout.REAL_SIZE)
        replacement_start = affected_reals * layout.REAL_SIZE
        replacement = payload[replacement_start : replacement_start + layout.REAL_SIZE]
        if len(replacement) != layout.REAL_SIZE:
            raise RestartFormatError("payload is too short to repair its leading ghost cells")
        for idx in range(affected_reals):
            start = idx * layout.REAL_SIZE
            payload[start : start + layout.REAL_SIZE] = replacement

    outgoing = 0
    if block_index + 1 < len(shifts):
        outgoing = max(0, shifts[block_index] - shifts[block_index + 1])
    if outgoing:
        affected_bytes = math.ceil(outgoing / layout.REAL_SIZE) * layout.REAL_SIZE
        if affected_bytes > tail_bytes:
            raise RestartFormatError(
                f"overlap requires repairing {affected_bytes} bytes but user tail has "
                f"only {tail_bytes}"
            )
        payload[-affected_bytes:] = b"\0" * affected_bytes
