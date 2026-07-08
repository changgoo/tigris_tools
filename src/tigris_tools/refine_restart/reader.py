from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, TextIO

from . import layout
from .param_block import ParameterBlock, read_parameter_block


@dataclass
class RestartIndex:
    params: ParameterBlock
    header: layout.MeshHeader
    user_mesh_data: bytes
    blocks: list[layout.InputBlockDesc]
    payload_start: int
    payload_end: int
    file_size: int


class RestartReader:
    def __init__(self, path: str | Path, *, verbose: bool = False, log: TextIO | None = None):
        self.path = Path(path)
        self.verbose = verbose
        self.log = log if log is not None else sys.stderr
        self._stream: BinaryIO | None = None
        self.index: RestartIndex | None = None

    def __enter__(self) -> "RestartReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._stream is None:
            self._stream = self.path.open("rb")
            self.index = self._read_index(self._stream)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def read_block_bytes(self, block: layout.InputBlockDesc) -> bytes:
        if self._stream is None:
            raise RuntimeError("reader is not open")
        self._stream.seek(block.file_offset)
        data = self._stream.read(block.byte_size)
        if len(data) != block.byte_size:
            raise EOFError(
                f"truncated block payload at offset {block.file_offset}: "
                f"expected {block.byte_size}, got {len(data)}"
            )
        self._log(
            f"read block loc=({block.loc.level},{block.loc.lx1},{block.loc.lx2},{block.loc.lx3}) "
            f"bytes={block.byte_size} offset={block.file_offset}"
        )
        return data

    def _read_index(self, stream: BinaryIO) -> RestartIndex:
        params = read_parameter_block(stream)
        self._log(f"parameter block bytes={stream.tell()}")
        header_data = _read_exact(stream, layout.MESH_HEADER_STRUCT.size, "mesh header")
        header = layout.unpack_mesh_header(header_data)
        self._log(
            "mesh "
            f"nbtotal={header.nbtotal} root_level={header.root_level} "
            f"nx=({header.mesh_size.nx1},{header.mesh_size.nx2},{header.mesh_size.nx3}) "
            f"time={header.time} dt={header.dt} ncycle={header.ncycle}"
        )
        inferred_user_mesh_size = False
        user_mesh_size = _user_mesh_data_size(params)
        if user_mesh_size is None:
            user_mesh_size = self._infer_user_mesh_data_size(stream, params, header)
            inferred_user_mesh_size = True
        user_mesh_data = _read_exact(stream, user_mesh_size, "user mesh data")
        suffix = " inferred=true" if inferred_user_mesh_size else ""
        self._log(f"user mesh data bytes={user_mesh_size}{suffix}")
        self._log(_restart_summary(params))

        id_records = _read_exact(
            stream,
            header.nbtotal * layout.ID_LIST_RECORD_STRUCT.size,
            "ID list",
        )
        payload_start = stream.tell()
        offset = payload_start
        blocks: list[layout.InputBlockDesc] = []
        for idx in range(header.nbtotal):
            loc, cost, byte_size = layout.unpack_id_record(
                id_records,
                idx * layout.ID_LIST_RECORD_STRUCT.size,
            )
            blocks.append(layout.InputBlockDesc(loc, cost, byte_size, offset))
            offset += byte_size
        file_size = self.path.stat().st_size
        if offset > file_size:
            raise EOFError(
                f"block payloads extend past EOF: expected end {offset}, file size {file_size}"
            )
        if blocks:
            sizes = [block.byte_size for block in blocks]
            self._log(
                "blocks "
                f"count={len(blocks)} payload_bytes={sum(sizes)} "
                f"min={min(sizes)} max={max(sizes)} "
                f"payload_span={payload_start}:{offset} "
                f"trailing_bytes={file_size - offset} file_size={file_size}"
            )
            first = blocks[0].loc
            last = blocks[-1].loc
            self._log(
                "block locations "
                f"first=({first.level},{first.lx1},{first.lx2},{first.lx3}) "
                f"last=({last.level},{last.lx1},{last.lx2},{last.lx3})"
            )
        else:
            self._log(f"blocks count=0 payload_span={payload_start}:{offset} file_size={file_size}")
        return RestartIndex(
            params=params,
            header=header,
            user_mesh_data=user_mesh_data,
            blocks=blocks,
            payload_start=payload_start,
            payload_end=offset,
            file_size=file_size,
        )

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"refine_restart.reader: {message}", file=self.log)

    def _infer_user_mesh_data_size(
        self,
        stream: BinaryIO,
        params: ParameterBlock,
        header: layout.MeshHeader,
    ) -> int:
        # ponytail: old restarts may lack <restart>; scan for the ID list instead of
        # reverse-engineering problem-specific user Mesh data.
        start = stream.tell()
        file_size = self.path.stat().st_size
        max_scan = min(16 * 1024 * 1024, file_size - start)
        data = stream.read(max_scan)
        record_size = layout.ID_LIST_RECORD_STRUCT.size
        list_size = header.nbtotal * record_size
        block_size = (
            params.get_int("meshblock", "nx1", header.mesh_size.nx1) or header.mesh_size.nx1,
            params.get_int("meshblock", "nx2", header.mesh_size.nx2) or header.mesh_size.nx2,
            params.get_int("meshblock", "nx3", header.mesh_size.nx3) or header.mesh_size.nx3,
        )
        nrb = (
            header.mesh_size.nx1 // block_size[0],
            header.mesh_size.nx2 // block_size[1],
            header.mesh_size.nx3 // block_size[2],
        )
        for rel in range(0, len(data) - list_size + 1, 4):
            if _looks_like_id_list(data, rel, header.nbtotal, header.root_level, nrb, file_size, start):
                stream.seek(start)
                return rel
        stream.seek(start)
        raise ValueError("could not infer user Mesh data size: no plausible ID list found")


def read_restart_index(
    path: str | Path,
    *,
    verbose: bool = False,
    log: TextIO | None = None,
) -> RestartIndex:
    with RestartReader(path, verbose=verbose, log=log) as reader:
        assert reader.index is not None
        return reader.index


def _read_exact(stream: BinaryIO, size: int, label: str) -> bytes:
    if size == 0:
        return b""
    data = stream.read(size)
    if len(data) != size:
        raise EOFError(f"truncated {label}: expected {size}, got {len(data)}")
    return data


def _user_mesh_data_size(params: ParameterBlock) -> int | None:
    if "restart" not in params.values:
        return None
    nint = params.get_int("restart", "nint_user_mesh_data", 0) or 0
    nreal = params.get_int("restart", "nreal_user_mesh_data", 0) or 0
    total = 0
    for idx in range(nint):
        total += _required_size(params, "restart", f"int_user_mesh_data_size_{idx}")
    for idx in range(nreal):
        total += _required_size(params, "restart", f"real_user_mesh_data_size_{idx}")
    return total


def _required_size(params: ParameterBlock, block: str, key: str) -> int:
    value = params.get_int(block, key)
    if value is None:
        raise ValueError(f"missing required <{block}>/{key}")
    if value < 0:
        raise ValueError(f"negative size in <{block}>/{key}: {value}")
    return value


def _restart_summary(params: ParameterBlock) -> str:
    restart = params.values.get("restart", {})
    meshblock = params.values.get("meshblock", {})
    keys = {
        "meshblock": f"({meshblock.get('nx1','?')},{meshblock.get('nx2','?')},{meshblock.get('nx3','?')})",
        "mhd": restart.get("magnetic_fields_enabled", "?"),
        "cr": restart.get("cr_enabled", "?"),
        "ncrg": restart.get("ncrg", "1"),
        "nscalars": restart.get("nscalars", "0"),
        "user_meshblock_int": restart.get("nint_user_meshblock_data", "0"),
        "user_meshblock_real": restart.get("nreal_user_meshblock_data", "0"),
    }
    return "restart schema " + " ".join(f"{key}={value}" for key, value in keys.items())


def _looks_like_id_list(
    data: bytes,
    rel: int,
    nbtotal: int,
    root_level: int,
    nrb: tuple[int, int, int],
    file_size: int,
    absolute_start: int,
) -> bool:
    record_size = layout.ID_LIST_RECORD_STRUCT.size
    total_payload = 0
    seen: set[tuple[int, int, int, int]] = set()
    for idx in range(nbtotal):
        loc, cost, byte_size = layout.unpack_id_record(data, rel + idx * record_size)
        if loc.level != root_level:
            return False
        if not (0 <= loc.lx1 < nrb[0] and 0 <= loc.lx2 < nrb[1] and 0 <= loc.lx3 < nrb[2]):
            return False
        key = (loc.level, loc.lx1, loc.lx2, loc.lx3)
        if key in seen:
            return False
        seen.add(key)
        if not math.isfinite(cost) or cost <= 0.0:
            return False
        if byte_size <= 0:
            return False
        total_payload += byte_size
    payload_start = absolute_start + rel + nbtotal * record_size
    return payload_start + total_payload <= file_size
