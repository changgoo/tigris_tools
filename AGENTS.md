# Repository Guidelines

## Project Structure & Module Organization

Python code uses a `src/` layout. Restart refinement lives in
`src/tigris_tools/refine_restart/`; slice extraction, repair, caching, and plotting live in
`src/tigris_tools/restart_slices/`. Keep tests in the matching directories under `tests/`, with
files named `test_*.py`. User documentation belongs in `docs/`; `scripts/` contains convenience
launchers, while `pbs/` contains NAS batch-job examples. CLI entry points are declared in
`pyproject.toml`.

## Build, Test, and Development Commands

- `python -m pip install -e ".[dev]"` installs the package, pytest, and Ruff for development.
- `python -m pip install -e ".[dev,slices]"` also installs NetCDF and plotting dependencies.
- `python -m ruff check .` checks imports and undefined names using the repository configuration.
- `python -m pytest -q` runs the fast synthetic-file test suite used by CI.
- `PYTHONPATH=src python -m tigris_tools.refine_restart --help` exercises a CLI without installing.

Use a command's `--dry-run` option before processing production restart files. For example,
`refine-restart INPUT.rst OUT.rst --refine 2 --dry-run -v` validates the planned conversion.

## Coding Style & Naming Conventions

Target Python 3.9 or newer. Use four-space indentation, type hints for public interfaces, and a
100-character line limit. Follow existing names: `snake_case` for functions and modules,
`PascalCase` for classes, and uppercase constants. Keep file-format parsing separate from CLI
argument handling. Run Ruff before submitting; its configured rules enforce Pyflakes (`F`) and
import sorting (`I`).

## Testing Guidelines

Tests use pytest and small synthetic restart or particle files; do not commit large simulation
outputs as fixtures. Add focused tests beside the affected subsystem and name cases descriptively,
such as `test_batch_plan_skips_only_when_both_caches_are_fresh`. Use `tmp_path` for generated
artifacts and `pytest.mark.parametrize` for format variants. The `slow` marker is reserved for tests
requiring a TIGRESS++ executable or real checkpoint. No coverage threshold is configured, but new
branches and failure paths should be exercised.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative subjects such as `Add Ruff linting` and `Implement
refine_restart package`. After completing each task, commit the finished work; stage only files that
belong to that task and keep the commit scoped to one logical change. Pull requests should explain
the motivation and compatibility impact, link relevant issues or design notes, and list the exact
lint/test commands run. Include a dry-run transcript for restart transformations and before/after
images when plotting output changes. Never commit production checkpoints, generated NetCDF caches,
plots, or scheduler logs.
