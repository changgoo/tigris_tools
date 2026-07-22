from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .batch import (
    RestartFile,
    discover_numbered_restarts,
    discover_particle_files,
    match_particle_file,
    read_restart_header,
)
from .repair import repair_rank_shifted_restart, validate_repaired_restart


@dataclass(frozen=True)
class RepairTarget:
    input: Path
    particle: Path | None
    candidate: Path
    backup: Path


def discover_repair_targets(
    directory: Path,
    *,
    prefix: str,
    include_final: bool,
) -> list[RepairTarget]:
    particles = discover_particle_files(directory)
    numbered = discover_numbered_restarts(directory, prefix=prefix)
    targets = [_target(item.path, match_particle_file(item, particles)) for item in numbered]
    final = directory / f"{prefix}.final.rst"
    if include_final:
        if not final.is_file():
            raise FileNotFoundError(f"final restart does not exist: {final}")
        final_item = RestartFile(-1, final, read_restart_header(final))
        matches = [
            particle.path
            for particle in particles
            if particle.header.ncycle == final_item.header.ncycle
            and abs(particle.header.time - final_item.header.time)
            <= 1.0e-10 * max(1.0, abs(final_item.header.time))
        ]
        if len(matches) > 1:
            raise ValueError(f"multiple particle sidecars match {final}")
        targets.append(_target(final, matches[0] if matches else None))
    return targets


def _target(path: Path, particle: Path | None) -> RepairTarget:
    stem = path.name[: -len(".rst")]
    return RepairTarget(
        input=path,
        particle=particle,
        candidate=path.with_name(f"{stem}.repaired.rst"),
        backup=path.with_name(f"{stem}.corrupt.rst"),
    )


def prepare_targets(targets: list[RepairTarget], manifest: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for position, target in enumerate(targets, start=1):
        print(
            f"[{position}/{len(targets)}] prepare {target.input.name} "
            f"particle={target.particle.name if target.particle else 'embedded-only'}",
            flush=True,
        )
        if target.backup.exists() and not target.candidate.exists():
            report = validate_repaired_restart(
                target.input,
                target.particle,
                check_payloads=False,
            )
            record = _record(target, "already-replaced", report)
        else:
            if target.backup.exists():
                raise FileExistsError(
                    f"backup already exists while candidate is present: {target.backup}"
                )
            if target.candidate.exists():
                print(f"  reusing existing candidate {target.candidate.name}", flush=True)
            else:
                repair_rank_shifted_restart(
                    target.input,
                    target.particle,
                    target.candidate,
                    verbose=True,
                )
            report = validate_repaired_restart(target.candidate, target.particle)
            record = _record(target, "candidate-validated", report)
            print(
                f"  validated blocks={report['blocks']} "
                f"payload_bytes={report['payload_bytes']}",
                flush=True,
            )
        records.append(record)
        _write_manifest(manifest, records, phase="preparing")
    _write_manifest(manifest, records, phase="prepared")
    return records


def replace_targets(
    targets: list[RepairTarget],
    records: list[dict[str, object]],
    manifest: Path,
) -> None:
    incomplete = [
        target
        for target in targets
        if not target.backup.exists() and not target.candidate.is_file()
    ]
    if incomplete:
        raise FileNotFoundError(f"missing repaired candidates: {incomplete}")
    conflicts = [
        target.backup
        for target in targets
        if target.backup.exists() and target.candidate.exists()
    ]
    if conflicts:
        raise FileExistsError(f"backup/candidate conflicts prevent replacement: {conflicts}")

    by_input = {record["input"]: record for record in records}
    for position, target in enumerate(targets, start=1):
        if target.backup.exists() and not target.candidate.exists():
            print(f"[{position}/{len(targets)}] already replaced {target.input.name}", flush=True)
            continue
        print(f"[{position}/{len(targets)}] replace {target.input.name}", flush=True)
        target.input.rename(target.backup)
        try:
            target.candidate.rename(target.input)
            _fsync_directory(target.input.parent)
            report = validate_repaired_restart(
                target.input,
                target.particle,
                check_payloads=False,
            )
        except Exception:
            if target.input.exists() and not target.candidate.exists():
                target.input.rename(target.candidate)
            if target.backup.exists() and not target.input.exists():
                target.backup.rename(target.input)
            _fsync_directory(target.input.parent)
            raise
        record = by_input[str(target.input)]
        record["status"] = "replaced-and-revalidated"
        record["canonical_validation"] = report
        _write_manifest(manifest, records, phase="replacing")
    _write_manifest(manifest, records, phase="complete")


def _record(
    target: RepairTarget,
    status: str,
    validation: dict[str, int],
) -> dict[str, object]:
    paths = {key: str(value) if value is not None else None for key, value in asdict(target).items()}
    return {**paths, "status": status, "candidate_validation": validation}


def _write_manifest(
    path: Path,
    records: list[dict[str, object]],
    *,
    phase: str,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {"phase": phase, "targets": records}
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repair-restarts-all",
        description=(
            "Create and fully validate every repaired restart before atomically "
            "replacing canonical names"
        ),
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prefix", default="TIGRESS")
    parser.add_argument("--include-final", action="store_true")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="after every candidate validates, move originals to .corrupt.rst and replace them",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSON audit manifest (default: DIRECTORY/restart_repair_manifest.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    directory = args.directory.resolve()
    manifest = (
        args.manifest.resolve()
        if args.manifest is not None
        else directory / "restart_repair_manifest.json"
    )
    targets = discover_repair_targets(
        directory,
        prefix=args.prefix,
        include_final=args.include_final,
    )
    if not targets:
        raise SystemExit(f"repair-restarts-all: no numbered restarts in {directory}")
    print(f"repair-restarts-all: targets={len(targets)} manifest={manifest}", flush=True)
    records = prepare_targets(targets, manifest)
    if args.replace:
        replace_targets(targets, records, manifest)
    else:
        print("repair-restarts-all: candidates validated; originals unchanged", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
