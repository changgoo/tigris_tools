from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tigris_tools.refine_restart.reader import RestartFormatError, read_restart_index

from .batch import discover_numbered_restarts, discover_particle_files, match_particle_file
from .cache import cache_is_fresh, projection_cache_path, slice_cache_path
from .netcdf import write_projection_caches
from .projection import extract_particles, extract_projections
from .snapshot_plot import snapshot_figure_path, write_snapshot_plot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tigris-projections-all",
        description="Generate projection caches and snapshot figures from numbered restarts",
    )
    parser.add_argument("restart_dir", type=Path)
    parser.add_argument("--savdir", type=Path, required=True)
    parser.add_argument("--figdir", type=Path)
    parser.add_argument("--prefix")
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-snapshot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    figdir = args.figdir or args.savdir / "snapshot"
    restarts = discover_numbered_restarts(
        args.restart_dir, prefix=args.prefix, start=args.start, stop=args.stop
    )
    if not restarts:
        raise SystemExit(f"tigris-projections-all: no numbered restarts in {args.restart_dir}")
    particle_files = discover_particle_files(args.restart_dir, verbose=args.verbose)
    completed = skipped = figures = 0
    failures = []
    for position, item in enumerate(restarts, 1):
        projection_paths = {
            axis: projection_cache_path(args.savdir, axis, item.num) for axis in ("y", "z")
        }
        projections_fresh = not args.overwrite and all(
            cache_is_fresh(path, item.path) for path in projection_paths.values()
        )
        output = snapshot_figure_path(figdir, item.num)
        figure_inputs = [
            *projection_paths.values(),
            *(slice_cache_path(args.savdir, axis, item.num) for axis in ("y", "z")),
        ]
        figure_fresh = (
            not args.overwrite
            and output.is_file()
            and all(path.is_file() and output.stat().st_mtime > path.stat().st_mtime for path in figure_inputs)
        )
        mode = "fresh" if projections_fresh else "generate"
        print(f"[{position}/{len(restarts)}] {item.path.name}: projections={mode}", flush=True)
        if args.dry_run:
            continue
        particle_path = None
        try:
            read_restart_index(item.path)
        except RestartFormatError:
            particle_path = match_particle_file(item, particle_files)
        try:
            result = None
            if projections_fresh:
                skipped += 1
            else:
                result = extract_projections(
                    item.path, particle_path=particle_path, verbose=args.verbose
                )
                projection_paths = write_projection_caches(
                    result, args.savdir, item.num, overwrite=True
                )
                completed += 1
                print(
                    f"  wrote {projection_paths['z']} and {projection_paths['y']}", flush=True
                )
            if not args.no_snapshot and not figure_fresh:
                slices = {
                    axis: slice_cache_path(args.savdir, axis, item.num) for axis in ("y", "z")
                }
                missing = [str(path) for path in slices.values() if not path.is_file()]
                if missing:
                    raise FileNotFoundError(f"snapshot requires slice caches: {missing}")
                particles = (
                    result.particles
                    if result is not None
                    else extract_particles(item.path, particle_path=particle_path)
                )
                write_snapshot_plot(
                    slices["y"],
                    slices["z"],
                    projection_paths["y"],
                    projection_paths["z"],
                    item.path,
                    output,
                    particles=particles,
                )
                figures += 1
                print(f"  wrote {output}", flush=True)
        except Exception as exc:
            failures.append({"restart": str(item.path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if args.fail_fast:
                break
    print(
        json.dumps(
            {
                "found": len(restarts),
                "projections_completed": completed,
                "projections_skipped_fresh": skipped,
                "figures_completed": figures,
                "figdir": str(figdir),
                "dry_run": args.dry_run,
                "failures": failures,
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
