from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .convert import refine_restart
from .reader import read_restart_index


def _parse_block_size(value: str) -> tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected NX,NY,NZ")
    try:
        parsed = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("block size entries must be integers") from exc
    if any(n <= 0 for n in parsed):
        raise argparse.ArgumentTypeError("block size entries must be positive")
    return parsed  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refine_restart",
        description="Refine or coarsen a TIGRESS++ / Athena++ restart checkpoint.",
    )
    parser.add_argument("input", type=Path, help="input .rst file")
    parser.add_argument("output", type=Path, help="output .rst file")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--refine", type=int, metavar="N", help="uniform refinement factor")
    mode.add_argument("--coarsen", type=int, metavar="N", help="uniform coarsening factor")
    parser.add_argument(
        "--block-size",
        type=_parse_block_size,
        metavar="NX,NY,NZ",
        help="target meshblock active-cell size; default keeps input size",
    )
    parser.add_argument("--seed", type=int, metavar="U64", help="PRNG top seed override")
    parser.add_argument("--verify", action="store_true", help="re-read output and verify structure")
    parser.add_argument("--dry-run", action="store_true", help="plan conversion without writing")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    factor = args.refine if args.refine is not None else args.coarsen
    if factor is None or factor <= 0:
        parser.error("refine/coarsen factor must be positive")

    if args.dry_run:
        index = read_restart_index(args.input, verbose=args.verbose > 0)
        out_mesh = (
            index.header.mesh_size.nx1 * factor,
            index.header.mesh_size.nx2 * factor,
            index.header.mesh_size.nx3 * factor,
        )
        out_block_size = args.block_size or (
            index.params.get_int("meshblock", "nx1", index.header.mesh_size.nx1),
            index.params.get_int("meshblock", "nx2", index.header.mesh_size.nx2),
            index.params.get_int("meshblock", "nx3", index.header.mesh_size.nx3),
        )
        in_block_size = (
            index.params.get_int("meshblock", "nx1", index.header.mesh_size.nx1),
            index.params.get_int("meshblock", "nx2", index.header.mesh_size.nx2),
            index.params.get_int("meshblock", "nx3", index.header.mesh_size.nx3),
        )
        for axis, (mesh_n, in_b, out_b) in enumerate(zip(out_mesh, in_block_size, out_block_size), start=1):
            if mesh_n % out_b != 0:
                parser.exit(2, f"refine_restart: output mesh nx{axis}={mesh_n} is not divisible by block size {out_b}\n")
            if (in_b * factor) % out_b != 0:
                parser.exit(2, f"refine_restart: refined input block axis {axis} size {in_b * factor} is not divisible by output block size {out_b}\n")
        out_blocks = (out_mesh[0] // out_block_size[0]) * (out_mesh[1] // out_block_size[1]) * (out_mesh[2] // out_block_size[2])
        print(
            "dry-run: "
            f"blocks={len(index.blocks)} "
            f"output_blocks={out_blocks} "
            f"mesh=({index.header.mesh_size.nx1},{index.header.mesh_size.nx2},{index.header.mesh_size.nx3}) "
            f"output_mesh=({out_mesh[0]},{out_mesh[1]},{out_mesh[2]}) "
            f"output_block_size=({out_block_size[0]},{out_block_size[1]},{out_block_size[2]}) "
            f"payload_bytes={index.payload_end - index.payload_start}"
        )
        return 0

    if args.coarsen is not None:
        parser.exit(2, "refine_restart: --coarsen is not implemented yet\n")
    refine_restart(args.input, args.output, factor=factor, block_size=args.block_size, verbose=args.verbose)
    return 0
