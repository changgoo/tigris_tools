from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class MovieSpec:
    kind: str
    frame_glob: str
    output: Path


def movie_specs(
    run_dir: str | Path,
    *,
    movies_dir: str | Path | None = None,
    slice_figdir: str | Path | None = None,
    snapshot_figdir: str | Path | None = None,
    basename: str | None = None,
) -> tuple[MovieSpec, MovieSpec]:
    """Return the CR-slice and snapshot movie paths used by plot_slices.py."""

    run = Path(run_dir)
    name = basename or run.name
    destination = Path(movies_dir) if movies_dir is not None else run / "movies"
    slices = Path(slice_figdir) if slice_figdir is not None else run / "cr_slices"
    snapshots = Path(snapshot_figdir) if snapshot_figdir is not None else run / "snapshot"
    return (
        MovieSpec(
            kind="slices",
            frame_glob=str(slices / f"{name}_*.png"),
            output=destination / f"{name}_cr_slices.mp4",
        ),
        MovieSpec(
            kind="snapshot",
            frame_glob=str(snapshots / "snapshot_*.png"),
            output=destination / f"{name}_snapshot.mp4",
        ),
    )


def ffmpeg_command(
    executable: str,
    spec: MovieSpec,
    *,
    fps_in: int = 15,
    fps_out: int = 15,
) -> list[str]:
    """Build the ffmpeg invocation used by pyathena's make_movie helper."""

    return [
        executable,
        "-y",
        "-r",
        str(fps_in),
        "-f",
        "image2",
        "-pattern_type",
        "glob",
        "-i",
        spec.frame_glob,
        "-r",
        str(fps_out),
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-f",
        "mp4",
        str(spec.output),
    ]


def find_ffmpeg(requested: str) -> str | None:
    executable = shutil.which(requested)
    if executable is not None or requested != "ffmpeg":
        return executable
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    candidate = Path(imageio_ffmpeg.get_ffmpeg_exe())
    return str(candidate) if candidate.is_file() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tigris-make-movies",
        description="Create plot_slices.py-compatible MP4 movies with ffmpeg",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--movies-dir", type=Path, help="default: RUN_DIR/movies")
    parser.add_argument("--slice-figdir", type=Path, help="default: RUN_DIR/cr_slices")
    parser.add_argument("--snapshot-figdir", type=Path, help="default: RUN_DIR/snapshot")
    parser.add_argument("--basename", help="default: RUN_DIR directory name")
    parser.add_argument("--kind", choices=("all", "slices", "snapshot"), default="all")
    parser.add_argument("--fps-in", type=int, default=15)
    parser.add_argument("--fps-out", type=int, default=15)
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable or path")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fps_in <= 0 or args.fps_out <= 0:
        parser.error("--fps-in and --fps-out must be positive")
    if not args.run_dir.is_dir():
        parser.error(f"run directory not found: {args.run_dir}")

    specs = movie_specs(
        args.run_dir,
        movies_dir=args.movies_dir,
        slice_figdir=args.slice_figdir,
        snapshot_figdir=args.snapshot_figdir,
        basename=args.basename,
    )
    selected = [spec for spec in specs if args.kind in ("all", spec.kind)]
    executable = args.ffmpeg if args.dry_run else find_ffmpeg(args.ffmpeg)
    if executable is None:
        parser.error(f"ffmpeg executable not found: {args.ffmpeg}")

    completed = []
    failures = []
    for spec in selected:
        frames = sorted(glob.glob(spec.frame_glob))
        if not frames:
            failures.append({"kind": spec.kind, "error": f"no frames match {spec.frame_glob}"})
            continue
        command = ffmpeg_command(executable, spec, fps_in=args.fps_in, fps_out=args.fps_out)
        print(f"[{spec.kind}] {len(frames)} frames -> {spec.output}", flush=True)
        if args.dry_run:
            print("  " + " ".join(command), flush=True)
            continue
        spec.output.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if result.returncode or not spec.output.is_file() or not spec.output.stat().st_size:
            failures.append(
                {
                    "kind": spec.kind,
                    "error": f"ffmpeg exited with status {result.returncode}",
                    "output": result.stdout,
                }
            )
            continue
        completed.append(str(spec.output))
        print(f"  wrote {spec.output}", flush=True)

    print(
        json.dumps(
            {
                "completed": completed,
                "dry_run": args.dry_run,
                "failures": failures,
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
