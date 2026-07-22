from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .recovery import recover_rank_shifted_index
from .repair import repair_rank_shifted_restart


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="repair-restart",
        description="Repair a rank-shifted TIGRESS++ restart without modifying the original.",
    )
    parser.add_argument("input", type=Path, help="corrupted input .rst")
    parser.add_argument("particle", type=Path, help="same-cycle .parbin sidecar")
    parser.add_argument("output", type=Path, help="new repaired .rst")
    parser.add_argument("--dry-run", action="store_true", help="validate recovery without writing")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        report = recover_rank_shifted_index(args.input, args.particle)
    else:
        report = repair_rank_shifted_restart(
            args.input,
            args.particle,
            args.output,
            verbose=args.verbose,
            log=sys.stderr,
        )
    print(
        "recovery: "
        f"blocks={len(report.index.blocks)} "
        f"exact_id_records={report.exact_id_records} "
        f"shifts={report.candidate_shifts} "
        f"overlaps={report.overlap_boundaries} "
        f"max_overlap={report.max_overlap_bytes} "
        f"gaps={report.gap_boundaries} "
        f"max_gap={report.max_gap_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
