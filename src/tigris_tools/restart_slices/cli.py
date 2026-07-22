from __future__ import annotations

import argparse
import json
from pathlib import Path

from .extract import extract_central_slices, infer_output_number, validate_slices
from .netcdf import write_slice_caches
from .validation_plot import write_validation_plot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tigris-slices",
        description="Create pyathena-compatible central slices from a TIGRESS++ restart",
    )
    parser.add_argument("restart", type=Path)
    parser.add_argument(
        "--particle", type=Path, help="same-cycle parbin used only for malformed restart recovery"
    )
    parser.add_argument("--savdir", type=Path, required=True)
    parser.add_argument(
        "--num", type=int, help="cache output number (default: infer from restart name)"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-plot", action="store_true", help="do not write the raw-array validation PNG"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    num = args.num if args.num is not None else infer_output_number(args.restart)
    result = extract_central_slices(
        args.restart,
        particle_path=args.particle,
        verbose=args.verbose,
    )
    report = validate_slices(result)
    paths = write_slice_caches(result, args.savdir, num, overwrite=args.overwrite)
    plot_path = None
    if not args.no_plot:
        plot_path = write_validation_plot(result, args.savdir / f"slice_validation.{num:05d}.png")

    summary = {
        "restart": str(args.restart),
        "time": result.time,
        "cycle": result.cycle,
        "selected_coordinates": result.selected_coordinates,
        "recovered": result.recovery is not None,
        "caches": {axis: str(path) for axis, path in paths.items()},
        "plot": str(plot_path) if plot_path is not None else None,
        "fields": report,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
