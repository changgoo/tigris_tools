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
- **restart_slices** — reconstruct central slice caches
  directly from restart output for `TIGRESS-CR` summary plots. See
  [`docs/restart_slices.md`](docs/restart_slices.md) for the compatibility
  contract and implementation status.

Current implemented scope:

- `--refine 2`
- same meshblock size, or a target meshblock size that divides each refined
  input block
- double-precision, Cartesian TIGRESS++ restart files
- hydro, MHD face fields, multigroup CR, passive scalars, complex particles,
  user mesh data, user meshblock data, and PRNG section resizing

Not implemented yet: coarsening and arbitrary block repartitioning across
multiple input parents.

## TODO and Known Issues

- Implement `--coarsen`.
- Support output meshblocks larger than one refined input block per axis.
  Current `--block-size` must divide `refine_factor * input_block_size` in
  each direction.
- Generalize particle schema detection. Current support covers the TIGRESS++
  complex-particle layouts seen so far, including shear-periodic `ish`.
- Preserve or deliberately reconstruct user MeshBlock data by semantic type.
  Current refinement zeros/copies opaque tail bytes conservatively enough for
  tested restarts, but module-specific counters may need better policies.
- PRNG streams are resized and reseeded, not made statistically identical to
  what an equivalent high-resolution run would have produced.
- Large conversions are serial and write one temporary file next to the
  output. Expect high wall time and enough free space for the full output.
- Face-centered magnetic fields use a simple divergence-preserving refinement
  rule; it is not an interpolative high-order prolongation.
- Restart compatibility is best for double-precision Cartesian TIGRESS++
  builds matching the source checkpoint physics. Older restart metadata can
  be inferred, but unusual build options may need new schema code.
- Always run `--dry-run -v` first, then restart the simulation for a short
  smoke test before using a converted checkpoint for production.

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

Repair a restart affected by rank-dependent writer offsets, using its
same-cycle particle dump as an independent block-boundary check:

```sh
repair-restart INPUT.rst SAME_CYCLE.parbin OUTPUT.repaired.rst --dry-run
repair-restart INPUT.rst SAME_CYCLE.parbin OUTPUT.repaired.rst -v
```

The repair command never overwrites an existing file. See
[`docs/restart_slices.md`](docs/restart_slices.md#repairing-affected-checkpoints)
for the recovery assumptions and native validation procedure.

Create the central `y=0` and `z=0` NetCDF caches and a raw-array validation
plot from a normal restart:

```sh
tigris-slices TIGRESS.00040.rst --savdir /path/to/analysis
```

For one of the historical rank-shifted files, provide its same-cycle parbin:

```sh
tigris-slices TIGRESS.00039.rst \
  --particle TIGRESS.out3.00169.par0.parbin \
  --savdir /path/to/analysis
```

Install the optional NetCDF and plotting dependencies with
`python -m pip install -e '.[slices]'`.

Generate or resume slices for every numbered restart in one run directory:

```sh
run=/nobackup/ckim14/tigress_classic/crmhd_duale-8pc-R16_tall-rst
tigris-slices-all "$run" --prefix TIGRESS --savdir "$run" --dry-run
tigris-slices-all "$run" --prefix TIGRESS --savdir "$run"
```

The batch command matches malformed restarts to parbin sidecars using cycle,
time, and mesh metadata. It ignores noncanonical names such as
`TIGRESS.00039.repaired.rst`, runs sequentially, skips both caches when they
are newer than the restart, and continues past individual failures. Use
`--start`, `--stop`, `--fail-fast`, or `--plot-validation` when needed.

Render the eight-panel `plot_slices_cr` summary directly from the cached
NetCDF slices, without importing pyathena or TIGRESS-CR:

```sh
tigris-plot-slices-all "$run" --prefix TIGRESS --savdir "$run"
```

Figures are written to `$run/cr_slices/<run-name>_NNNN.png`. The command skips
fresh figures, continues past missing cache pairs, and accepts `--start`,
`--stop`, `--figdir`, and `--overwrite`.

Generate the `prj.y` and `prj.z` NetCDF caches and the default 13-panel
`plot_snapshot` figure directly from each restart:

```sh
tigris-projections-all "$run" --prefix TIGRESS --savdir "$run"
```

Projection caches use the same `phase=(whole,hot,wc)` schema and physical units
as `TIGRESS-CR/python/slc_prj.py`. Snapshot figures are written to
`$run/snapshot/snapshot_NNNNN.png`; their particle overlays come from the exact
particle records embedded in each restart. To redraw snapshots from existing
slice and projection caches, use `tigris-plot-snapshots-all`.

A ready-to-submit NAS PBS job is provided at
[`pbs/generate_all_restart_slices.pbs`](pbs/generate_all_restart_slices.pbs):

```sh
qsub /home1/ckim14/tigris_tools/pbs/generate_all_restart_slices.pbs
```

The PBS job runs both cache generation and standalone figure generation. Set
`FIG_DIR` to change the figure directory or `PLOT_OVERWRITE=1` to redraw fresh
figures.

After slice generation finishes, submit the independent resumable projection
job:

```sh
qsub /home1/ckim14/tigris_tools/pbs/generate_all_restart_projections.pbs
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
