# Restart Refinement with `tigris_tools`

This page describes how to use the standalone `refine_restart` utility from
[`tigris_tools`](https://github.com/changgoo/tigris_tools) to post-process a
TIGRESS++ restart file into a higher-resolution restart file.

The tool reads an Athena++/TIGRESS++ `.rst` checkpoint, refines the mesh data,
updates the restart metadata, and writes a new `.rst` file that the existing
TIGRESS++ executable can restart from directly.

## When to Use This

Use `refine_restart` when a run should continue from the same physical state
but at a finer uniform mesh resolution.

Common cases:

- refine `64 x 64 x 512` to `128 x 128 x 1024`
- keep `32^3` meshblocks and increase the MPI rank count
- refine cells by 2 but emit `64^3` meshblocks to keep roughly the same block
  and rank count

This is a post-processing workaround for workflows where runtime AMR/SMR is
not available or not supported by the active TIGRESS++ physics modules.

## Install

Clone and install `tigris_tools`:

```sh
git clone git@github.com:changgoo/tigris_tools.git
cd tigris_tools
python -m pip install -e .
```

For development use without installing:

```sh
PYTHONPATH=src python -m tigris_tools.refine_restart --help
```

## Supported Scope

Current support:

- uniform `--refine 2`
- same meshblock size, or a target meshblock size that divides each refined
  input meshblock
- double-precision Cartesian TIGRESS++ restart files
- hydro conserved variables
- MHD face-centered fields
- multigroup CR restart arrays
- passive scalars
- complex particles, including shear-periodic `ish`
- user Mesh data and user MeshBlock data
- PRNG section resizing/reseeding

Not supported yet:

- `--coarsen`
- non-Cartesian coordinates
- arbitrary repartitioning where one output meshblock spans multiple input
  meshblocks
- fully general particle schemas beyond the TIGRESS++ layouts already tested

## Basic Usage

Always start with a dry run:

```sh
refine-restart INPUT.rst OUT.rst --refine 2 --dry-run -v
```

The dry run prints the parsed restart metadata, block count, target mesh, and
payload size. Check that these values match the intended run before writing a
large output file.

Refine by 2 and keep the input meshblock size:

```sh
refine-restart INPUT.rst OUT.rst --refine 2 -v
```

Example result:

```text
input mesh:       64 x 64 x 512
input blocks:     64 blocks of 32^3
output mesh:     128 x 128 x 1024
output blocks:   512 blocks of 32^3
```

Refine by 2 but write larger meshblocks:

```sh
refine-restart INPUT.rst OUT.rst --refine 2 --block-size 64,64,64 -v
```

Example result:

```text
input mesh:       64 x 64 x 512
input blocks:     64 blocks of 32^3
output mesh:     128 x 128 x 1024
output blocks:    64 blocks of 64^3
```

The second mode is useful when the downstream run should keep the same number
of MeshBlocks or MPI ranks.

## Restarting from the New File

Place the generated restart file where the TIGRESS++ run expects it, then use
the normal restart path. For example, if the converted file is named
`OUT.rst`, configure the run scripts so the executable restarts from that file.

After conversion, run a short smoke test before using the restart for
production:

1. restart from the converted `.rst`
2. advance one or a few cycles
3. confirm no restart warnings, segmentation faults, or NaNs
4. inspect the first output for expected mesh size and physical state

## Implementation Notes

Cell-centered quantities are refined by piecewise-constant volume copy. Each
coarse cell becomes `2 x 2 x 2` fine cells with the same value.

Face-centered magnetic fields are refined with a simple divergence-preserving
rule: transverse directions are repeated and interior faces along the
distinguished axis are linearly interpolated.

Particles are redistributed by physical position into the output MeshBlock
that owns the particle location. Particle values are not modified.

Ghost-zone data are written because Athena++ restart payloads store
ghost-inclusive arrays. Boundary communication after restart will refresh
ghost zones.

## Expected Caveats

- The tool is serial. Large restart files can take minutes and require enough
  free disk space for the complete output file.
- The output is written through a temporary file next to the requested output.
- PRNG streams are resized and reseeded; they are not bitwise equivalent to a
  hypothetical native high-resolution run.
- User MeshBlock data are carried as opaque restart bytes. This has worked for
  tested TIGRESS++ restarts, but module-specific counters may need special
  handling later.
- Restart files from older code versions may lack complete `<restart>`
  metadata. The reader can infer the ID-list offset for known legacy layouts,
  but unusual restart schemas may need code updates.
- The tool assumes `Real=double`.

## Troubleshooting

If dry-run reports impossible block sizes, garbage logical locations, or tiny
payload sizes, the reader likely misidentified the restart schema. Do not use
the output; update `tigris_tools` or inspect the restart layout first.

If the simulation segfaults shortly after restart, first check particle
metadata. A previous failure mode was a wrong particle integer-property count
for shear-periodic runs, which corrupted particle coordinates.

If the simulation loads but later diverges, compare a short low-resolution
restart and refined restart output. The refinement is intentionally simple and
does not perform high-order prolongation.

## Quick Validation Commands

From the `tigris_tools` checkout:

```sh
python -m ruff check .
python -m pytest -q
```

The test suite uses synthetic restart files and does not require large
production checkpoints.
