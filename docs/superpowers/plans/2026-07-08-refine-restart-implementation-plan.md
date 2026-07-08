# `refine_restart` Implementation Plan

**Date:** 2026-07-08
**Spec:** `docs/superpowers/specs/2026-07-07-refine-restart-design.md`
**Source context:** `../tigris/src/outputs/restart.cpp`,
`../tigris/src/mesh/mesh.cpp`, `../tigris/src/mesh/meshblock.cpp`,
`../tigris/src/athena.hpp`, `../tigris/src/particles/particles.cpp`
**Primary deliverable:** `python -m tigris_tools.refine_restart`

## Objective

Implement a standalone Python package that reads a TIGRESS++ / Athena++
restart checkpoint, uniformly refines or coarsens the mesh, optionally
changes the meshblock size, redistributes particles, reseeds PRNG state when
present, and writes a binary-compatible restart file that the simulation can
load without source changes.

The first usable version should support serial streaming I/O, Cartesian
meshes, uniform flat output meshes, piecewise-constant cell-centered
refinement, divergence-preserving face-field refinement, particle
redistribution by position, and synthetic restart-file tests.

## Milestones

### 1. Project scaffold

- Add `pyproject.toml` with a minimal Python package configuration.
- Add package directories:
  - `src/tigris_tools/__init__.py`
  - `src/tigris_tools/refine_restart/__init__.py`
  - module files named in the design spec.
- Add `tests/refine_restart/` and configure pytest markers for slow tests.
- Add console entry point if useful, while keeping
  `python -m tigris_tools.refine_restart` as the canonical invocation.

Acceptance criteria:

- `python -m pytest` discovers the test tree.
- `python -m tigris_tools.refine_restart --help` exits successfully.

### 2. Binary layout foundation

Implement `layout.py` first because every later component depends on exact
byte accounting.

- Define dataclasses for `LogicalLocation`, `RegionSize`, `MeshHeader`,
  `InputBlockDesc`, `OutputBlockDesc`, `ParticleTable`, and `BlockPayload`.
- Assume Athena++ `Real` is 8-byte `double`, matching the target TIGRESS++
  build that will consume the generated restart files.
- Define NumPy dtypes or `struct.Struct` helpers for:
  - `LogicalLocation`
  - `RegionSize`
  - mesh binary header wrapper fields
  - ID-list records
  - PRNG records
- Add explicit size and alignment assertions.
- Add helper functions to compute:
  - ghost-inclusive cell counts
  - face-centered array shapes
  - block payload byte counts
  - particle payload byte counts
  - absolute block payload offsets

Tests:

- `test_layout.py` verifies round-trip pack/unpack for known records.
- Add a skipped or optional C++ probe fixture path for final confirmation
  against Athena++ POD layouts.

Implementation note:

- Treat `RegionSize` layout as the highest-risk open item. Do not proceed
  to real checkpoint writes until a probe-generated fixture confirms field
  order, padding, and total size.
- Source inspection shows `RegionSize` field order is 12 `Real` values
  (`x*min`, `x*max`, `x*len`, `x*rat`) followed by three `int` values;
  `LogicalLocation` is three `int64_t` values followed by one `int`, with
  native C++ alignment.

### 3. Parameter block parser and rewriter

Implement `param_block.py`.

- Read from file start through the `<par_end>` line, enforcing the 40 KB
  scan expectation documented by Athena++.
- Preserve original text and comments as much as practical.
- Parse block/key/value entries needed by the tool:
  - `<mesh>/nx1`, `nx2`, `nx3`
  - `<meshblock>/nx1`, `nx2`, `nx3`
  - `<restart>/*`
  - coordinate-system keys needed to reject non-Cartesian inputs
- Provide typed getters with defaults matching non-strict restart behavior.
- Provide a patch function that updates only required keys and leaves
  unknown restart metadata intact.

Tests:

- Read and rewrite a representative parameter block without unrelated text
  churn.
- Verify missing optional `<restart>` fields warn once and default
  predictably.
- Verify mesh and meshblock dimensions are patched correctly.

### 4. Streaming reader

Implement `reader.py`.

- Open input restart files in binary mode.
- Read sections 1 through 4:
  - parameter block
  - mesh header
  - user mesh data blob
  - ID list
- Compute each block's absolute payload offset from ID-list sizes.
- Provide `read_block_payload(desc)` that:
  - seeks directly to the block payload
  - unpacks hydro, optional GR primitives, magnetic fields, particles, CR,
    scalars, and user meshblock data in schema order
  - validates exact byte consumption against `desc.byte_size`
- Detect whether a PRNG section exists from remaining bytes after block
  payloads and infer `N_prngs` when possible.

Tests:

- Synthetic restart with one block and all optional sections disabled.
- Synthetic restart with magnetic fields, multigroup CR, scalars, user data,
  particles, and PRNG records enabled.
- Truncated payload and bad-size cases fail with actionable offsets.

### 5. Mesh/block planning

Implement `plan.py`.

- Validate CLI and metadata constraints before opening the output path:
  - exactly one of refine or coarsen
  - positive integer factor
  - Cartesian coordinates
  - mesh dimensions divisible by target block size
  - refine/coarsen block-size divisibility rules from the spec
- Build canonical output `LogicalLocation` records for a flat output mesh.
- For flat outputs, keep every output block at `root_level`; represent the
  changed resolution by patching `mesh_size.nx*`, `<mesh>/nx*`, and the
  root-block grid dimensions. Do not encode uniform refinement by raising
  output `loc.level`, because `Mesh::Mesh` treats refined locations in a
  non-multilevel restart as an AMR/static-refinement case and then adjusts
  `root_level`.
- Map each output block to its input parent or parents.
- Assign uniform cost `1.0`.
- Pre-pass particle tables as needed to compute output block byte sizes
  before writing the ID list.
- Return a deterministic ordered list of `OutputBlockDesc`.

Tests:

- Reference case: `64x64x512`, `32^3` blocks refined by 2 into
  `128x128x1024`, `32^3` blocks.
- Same reference case emitted as `64^3` blocks.
- Coarsening mirror cases.
- Invalid divisibility cases fail before output creation.

### 6. Field refinement and coarsening

Implement `refine.py`.

- Cell-centered arrays:
  - refine by repeating along the last three axes
  - coarsen by reshaping into factor-sized groups and averaging
- Face-centered arrays:
  - refine transverse axes by repeat
  - refine distinguished axis by copying parent boundary faces and inserting
    linearly interpolated interior faces
  - coarsen by transverse averaging and distinguished-axis decimation
- Keep routines shape-driven and independent of specific physics fields.

Tests:

- Refine then coarsen exactly recovers piecewise-constant cell-centered
  arrays.
- Constant and linear face fields refine to expected values.
- Divergence-free input remains divergence-free to machine precision.
- Shape tests cover all three face orientations and anisotropic block sizes.

### 7. Particle repartitioning

Implement `particles.py`.

- Parse and pack particle tables without modifying property values.
- Use the base `Particles` schema from `particles.cpp`: integer properties
  start with `pid`, `flag`; real properties start with `mass`, `x1`, `x2`,
  `x3`, `v1`, `v2`, `v3`. Derived particle classes append after these base
  fields, so position indices are `realprop[1:4]` for current TIGRESS++
  particle types.
- Assign particles to output blocks using floor-based physical coordinate
  membership.
- Preserve per-container particle property order within each output block
  deterministically.
- Propagate `idmax`:
  - refine: every child gets the parent `idmax`
  - coarsen: merged block gets the maximum parent `idmax`

Tests:

- Particles on interior cells, block boundaries, and domain boundaries map
  to expected blocks.
- All integer and real properties are preserved byte-for-byte.
- Coarsen merges multiple parent tables and uses max `idmax`.

Risk:

- If future particle classes reorder the base properties, expose explicit
  CLI/schema configuration rather than guessing silently.

### 8. PRNG reseeding

Implement `prng.py`.

- Port the SplitMix64-style `AthenaRandom::HashUniform` finalization from
  `src/utils/random.hpp`.
- Default `top_seed` to input rank 0, PRNG 0 seed when a PRNG section exists.
- Support `--seed U64` override.
- Emit `N_output_blocks * N_prngs` records with count zero.

Tests:

- Regression table generated from a small C++ probe.
- No PRNG input produces no PRNG output.
- `--seed` changes seeds deterministically and leaves counts at zero.

### 9. Writer and two-pass data flow

Implement `writer.py`.

- Open the output path only after reader and plan phases succeed.
- Write sections in exact restart order:
  1. patched parameter block
  2. patched mesh header
  3. copied user mesh data
  4. planned ID list with final byte sizes
  5. transformed block payloads
  6. reseeded PRNG section if present
- Use a temporary output path in the same directory and atomically rename on
  success.
- Delete the temporary file on write failure.
- Cache only parents needed for current output blocks and evict when their
  remaining child/merge uses reach zero.

Tests:

- End-to-end synthetic write can be re-read by `reader.py`.
- ID-list byte sizes match actual emitted payload sizes.
- Payload offsets are contiguous and final file size matches expectation.

### 10. CLI integration

Implement `cli.py` and `__main__.py`.

CLI:

```sh
python -m tigris_tools.refine_restart INPUT.rst OUTPUT.rst \
  (--refine N | --coarsen N) \
  [--block-size NX,NY,NZ] \
  [--seed U64] \
  [--verify] \
  [--dry-run] \
  [-v|--verbose]
```

Behavior:

- Map validation errors to exit code 2.
- Map I/O errors to exit code 3.
- Map output verification failures to exit code 4.
- `--dry-run` prints input mesh, output mesh, output block count, target
  block size, PRNG policy, and estimated output bytes without writing.
- `--verify` re-reads the output and runs lightweight structural checks.

Tests:

- `--help`, invalid argument combinations, dry-run, and successful synthetic
  conversions via subprocess.

### 11. End-to-end verification

Implement synthetic fixture builder in `tests/refine_restart/conftest.py`.

- Generate compact restart files with configurable:
  - mesh size
  - block size
  - physics flags
  - particle containers and counts
  - user mesh and meshblock data
  - PRNG count
- Use fixtures for fast CI tests.

Fast tests:

- Refine then coarsen returns equivalent synthetic data.
- Per-block conservation for hydro, CR, scalars, and face fields.
- Particle sets are preserved globally.

Slow opt-in test:

- Guard with `pytest -m slow` and `TIGRIS_CRMHD_EXE`.
- Produce or consume a small real checkpoint, refine it, restart the
  simulation for one cycle, and assert clean exit plus expected block count.

## Suggested Delivery Order

1. Scaffold package, CLI stub, and tests.
2. Implement parameter-block parsing and binary layout helpers.
3. Build the synthetic restart fixture.
4. Implement reader until synthetic files can be fully decoded.
5. Implement pure array refinement/coarsening routines.
6. Implement planner and particle repartitioning.
7. Implement writer and end-to-end synthetic refine.
8. Add coarsening path.
9. Add PRNG reseeding.
10. Add `--dry-run`, `--verify`, and polished CLI errors.
11. Validate `RegionSize` and PRNG hash against C++ probes.
12. Run the slow load-and-step smoke test with a real TIGRESS++ binary.

## Definition of Done

- `python -m pytest` passes all fast tests.
- `python -m tigris_tools.refine_restart --dry-run` reports a correct plan
  for the documented reference checkpoint shape.
- Synthetic refine and coarsen conversions re-read cleanly and satisfy
  conservation checks.
- A refined checkpoint from a real TIGRESS++ restart loads and advances at
  least one cycle with the target simulation binary.
- Documentation in `docs/refine_restart.md` includes basic install and CLI
  usage once the command is implemented.

## Known Risks

- `RegionSize` and `LogicalLocation` binary layout must be verified against
  the exact compiler/build used by the source checkpoint.
- Particle position-property indexing may not be self-describing in older
  checkpoint schemas.
- User meshblock data semantics are module-specific; the v1 zero/sum policy
  is appropriate for documented TIGRESS++ accumulators but should remain
  clearly documented.
- Real checkpoints may contain restart metadata evolution not represented in
  synthetic fixtures; real-file smoke tests should be added as soon as a
  small checkpoint is available.
