# tigris_tools

Standalone post-processing utilities for TIGRESS++ / Athena++ simulations.

Independent of the [tigris](https://github.com/PrincetonUniversity/tigris)
repository - these tools consume simulation outputs (restart files, HDF5
dumps, particle catalogs) but do not link against or require a source-tree
build of the simulation code.

## Install

From a checkout:

```sh
python -m pip install -e .
```

For local development without installing:

```sh
PYTHONPATH=src python -m tigris_tools.refine_restart --help
```

## Tools

- **refine_restart** — read a TIGRESS++ restart checkpoint and produce a new
  checkpoint at a different mesh resolution or meshblock size, so that the
  simulation can restart from it without any code changes. See
  [`docs/refine_restart.md`](docs/refine_restart.md) for the full
  description.

Current implemented scope:

- `--refine 2`
- same meshblock size, or a target meshblock size that divides each refined
  input block
- double-precision, Cartesian TIGRESS++ restart files
- hydro, MHD face fields, multigroup CR, passive scalars, complex particles,
  user mesh data, user meshblock data, and PRNG section resizing

Not implemented yet: coarsening and arbitrary block repartitioning across
multiple input parents.

## Usage

Dry-run a conversion and print restart metadata:

```sh
refine-restart INPUT.rst OUT.rst --refine 2 --dry-run -v
```

Refine by 2 while keeping the original meshblock size. For example, a
`64 x 64 x 512` mesh with `32^3` meshblocks becomes
`128 x 128 x 1024` with `32^3` meshblocks:

```sh
refine-restart INPUT.rst OUT.rst --refine 2 -v
```

Refine by 2 but write larger meshblocks. For example, `32^3` input blocks
refined by 2 can be emitted as `64^3` output blocks, preserving the original
number of blocks/ranks:

```sh
refine-restart INPUT.rst OUT.rst --refine 2 --block-size 64,64,64 -v
```

Equivalent module invocation:

```sh
python -m tigris_tools.refine_restart INPUT.rst OUT.rst --refine 2
```

## Development

Run the fast test suite:

```sh
python -m pytest -q
```

The tests build small synthetic restart files; CI does not need large binary
fixtures.

## Layout

```
docs/
  refine_restart.md           # description of the refine_restart tool
  superpowers/specs/          # design specs produced during brainstorming
src/tigris_tools/
  refine_restart/             # Python package and CLI implementation
tests/refine_restart/
  test_*.py                   # synthetic restart and transform checks
```

## License

MIT. See [`LICENSE`](LICENSE).
