from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from tigris_tools.refine_restart import layout
from tigris_tools.refine_restart.param_block import read_parameter_block
from tigris_tools.refine_restart.reader import RestartFormatError, read_restart_index

from .cache import cache_is_fresh, slice_cache_path
from .extract import extract_central_slices
from .netcdf import write_slice_caches
from .recovery import read_particle_sidecar
from .validation_plot import write_validation_plot


@dataclass(frozen=True)
class RestartFile:
    num: int
    path: Path
    header: layout.MeshHeader


@dataclass(frozen=True)
class ParticleFile:
    path: Path
    header: layout.MeshHeader


@dataclass(frozen=True)
class BatchPlan:
    restart: RestartFile
    particle_path: Path | None
    mode: str


def discover_numbered_restarts(
    directory: str | Path,
    *,
    prefix: str | None = None,
    start: int | None = None,
    stop: int | None = None,
) -> list[RestartFile]:
    """Find only names ending in ``.<digits>.rst``.

    Names such as ``TIGRESS.00039.repaired.rst`` are deliberately excluded.
    """

    directory = Path(directory)
    pattern = re.compile(r"^(?P<prefix>.+)\.(?P<num>\d+)\.rst$")
    results = []
    for path in directory.glob("*.rst"):
        match = pattern.fullmatch(path.name)
        if match is None or (prefix is not None and match.group("prefix") != prefix):
            continue
        num = int(match.group("num"))
        if start is not None and num < start:
            continue
        if stop is not None and num > stop:
            continue
        results.append(RestartFile(num, path, read_restart_header(path)))
    results.sort(key=lambda item: (item.num, item.path.name))
    duplicates = sorted({item.num for item in results if sum(x.num == item.num for x in results) > 1})
    if duplicates:
        raise ValueError(
            "multiple restart prefixes have the same output numbers "
            f"{duplicates}; select one with --prefix"
        )
    return results


def read_restart_header(path: str | Path) -> layout.MeshHeader:
    with Path(path).open("rb") as stream:
        read_parameter_block(stream)
        data = stream.read(layout.MESH_HEADER_STRUCT.size)
    if len(data) != layout.MESH_HEADER_STRUCT.size:
        raise EOFError(f"truncated restart mesh header: {path}")
    return layout.unpack_mesh_header(data)


def discover_particle_files(directory: str | Path, *, verbose: bool = False) -> list[ParticleFile]:
    results = []
    for path in sorted(Path(directory).glob("*.parbin")):
        try:
            sidecar = read_particle_sidecar(path)
        except (EOFError, OSError, ValueError) as exc:
            if verbose:
                print(f"tigris-slices-all: ignoring {path.name}: {exc}", file=sys.stderr)
            continue
        results.append(ParticleFile(path, sidecar.header))
    return results


def match_particle_file(
    restart: RestartFile,
    particles: list[ParticleFile],
) -> Path:
    matches = [item.path for item in particles if _same_state(restart.header, item.header)]
    if not matches:
        raise FileNotFoundError(
            f"no particle sidecar matches cycle={restart.header.ncycle}, "
            f"time={restart.header.time:.17g} for {restart.path.name}"
        )
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"multiple particle sidecars match {restart.path.name}: {names}")
    return matches[0]


def build_batch_plan(
    restarts: list[RestartFile],
    particles: list[ParticleFile],
    savdir: str | Path,
    *,
    overwrite: bool = False,
) -> list[BatchPlan]:
    plans = []
    for restart in restarts:
        caches = [slice_cache_path(savdir, axis, restart.num) for axis in ("y", "z")]
        if not overwrite and all(cache_is_fresh(path, restart.path) for path in caches):
            plans.append(BatchPlan(restart, None, "fresh"))
            continue
        try:
            read_restart_index(restart.path)
        except RestartFormatError:
            particle_path = match_particle_file(restart, particles)
            plans.append(BatchPlan(restart, particle_path, "recovered"))
        else:
            plans.append(BatchPlan(restart, None, "normal"))
    return plans


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tigris-slices-all",
        description="Generate central slice caches from every numbered restart in a directory",
    )
    parser.add_argument("restart_dir", type=Path)
    parser.add_argument("--savdir", type=Path, required=True)
    parser.add_argument("--prefix", help="restart basename before .NNNNN.rst")
    parser.add_argument("--start", type=int, help="first output number, inclusive")
    parser.add_argument("--stop", type=int, help="last output number, inclusive")
    parser.add_argument("--overwrite", action="store_true", help="regenerate fresh caches too")
    parser.add_argument("--dry-run", action="store_true", help="show inputs and sidecars only")
    parser.add_argument(
        "--plot-validation",
        action="store_true",
        help="also write one raw-array validation PNG per restart",
    )
    parser.add_argument("--fail-fast", action="store_true", help="stop at the first failure")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start is not None and args.stop is not None and args.start > args.stop:
        raise SystemExit("tigris-slices-all: --start must not exceed --stop")
    restarts = discover_numbered_restarts(
        args.restart_dir,
        prefix=args.prefix,
        start=args.start,
        stop=args.stop,
    )
    if not restarts:
        raise SystemExit(f"tigris-slices-all: no numbered restart files in {args.restart_dir}")
    particles = discover_particle_files(args.restart_dir, verbose=args.verbose)
    plans = build_batch_plan(restarts, particles, args.savdir, overwrite=args.overwrite)

    completed = 0
    skipped = 0
    failures = []
    for position, plan in enumerate(plans, start=1):
        item = plan.restart
        particle_text = f" particle={plan.particle_path.name}" if plan.particle_path else ""
        print(
            f"[{position}/{len(plans)}] {item.path.name}: {plan.mode}{particle_text}",
            flush=True,
        )
        if plan.mode == "fresh":
            skipped += 1
            continue
        if args.dry_run:
            continue
        try:
            result = extract_central_slices(
                item.path,
                particle_path=plan.particle_path,
                verbose=args.verbose,
            )
            paths = write_slice_caches(result, args.savdir, item.num, overwrite=True)
            if args.plot_validation:
                write_validation_plot(
                    result,
                    args.savdir / f"slice_validation.{item.num:05d}.png",
                )
            completed += 1
            print(
                f"  wrote {paths['z']} and {paths['y']}",
                flush=True,
            )
        except Exception as exc:  # keep an overnight batch moving past one bad checkpoint
            failures.append({"restart": str(item.path), "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if args.fail_fast:
                break

    summary = {
        "found": len(restarts),
        "planned": sum(plan.mode != "fresh" for plan in plans),
        "completed": completed,
        "skipped_fresh": skipped,
        "dry_run": args.dry_run,
        "failures": failures,
    }
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


def _same_state(left: layout.MeshHeader, right: layout.MeshHeader) -> bool:
    if (
        left.nbtotal != right.nbtotal
        or left.root_level != right.root_level
        or left.ncycle != right.ncycle
    ):
        return False
    tolerance = 1.0e-10 * max(1.0, abs(left.time))
    return abs(left.time - right.time) <= tolerance


if __name__ == "__main__":
    raise SystemExit(main())
