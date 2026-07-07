# `refine_restart` — Design Spec

**Status:** approved for planning
**Date:** 2026-07-07
**Location:** `tigris_tools/src/tigris_tools/refine_restart/`

## 1. Goal

Build a standalone Python tool that reads a TIGRESS++ / Athena++ restart
checkpoint (`.rst`) and writes a new checkpoint at a different mesh
resolution, in the **exact binary layout** the simulation binary expects,
so restart works without any code changes.

The reference use case (from `docs/refine_restart.md`): produce a
`128×128×1024`, 32³-block checkpoint from a `64×64×512`, 32³-block
checkpoint, or the same physical up-res emitted as 64³-blocks so the
downstream run keeps the same rank count.

## 2. Scope

### In scope

- **Uniform integer refinement** by factor `R ∈ {2, 4, 8, ...}` applied to the
  whole mesh. All output blocks live at a single level.
- **Uniform integer coarsening** by factor `R` (mirror of refine).
- **Block-size changes** on the same call. Refine constraint per axis:
  `(R · B_in) % B_out == 0` and `R · B_in ≥ B_out` — one input block yields
  `((R · B_in) / B_out)³` output blocks (≥ 1). Coarsen: mirror; `((R · B_out) / B_in)³`
  input blocks fold into one output block (≥ 1).
- **Particle redistribution** by physical position, using the same block-
  membership rule as the simulation (`floor(MeshCoordsToIndices)`).
- **Fresh PRNG state** sized to the target rank count, seeded via a
  HashUniform-compatible mixer.
- **Streaming per-block I/O** — peak memory bounded by one input block plus
  one output block's arrays.

### Out of scope (v1)

- Non-uniform / SMR-style refinement (only some blocks refine).
- Adaptive prolongation stencils (piecewise-constant volume-copy suffices
  because `blockfft` prohibits mesh refinement at run time; the tool's
  output is always a flat mesh).
- MPI parallelism (data flow is designed to allow it later; not
  implemented in v1).
- Coordinate systems other than Cartesian.
- Preserving statistical reproducibility of the input PRNG streams (we
  reseed).

## 3. Restart file layout (reference)

See `docs/refine_restart.md` for the authoritative description. The tool
must reproduce this layout exactly.

Sections in file order:

1. **`ParameterInput` text block** terminated by `<par_end>` (INI-style,
   ASCII).
2. **Mesh binary header** — rank-0-written; contains `nbtotal, root_level,
   mesh_size, time, dt, ncycle`.
3. **User mesh data blob** — optional, sizes from `<restart>/*_user_mesh_data_size_*`.
4. **Per-block ID list** — one `(LogicalLocation, cost, byte_size)` record
   per block.
5. **Per-block payloads** — concatenated in ID-list order; each contains
   hydro conserved, (GR primitives), face-B, particles, CR, scalars, and
   user meshblock data in that fixed order.
6. **PRNG section** — optional; one record per rank, `Nprngs × (u64 seed,
   u64 count)` per rank.

Two-part metadata surface:

- The `<restart>` block in section 1 carries all field-count schema
  (`nint_user_*`, `nreal_user_*`, `nscalars`, `ncrg`, per-array sizes,
  `magnetic_fields_enabled`, `cr_enabled`, `strict_restart`).
- The `<mesh>/nx*` and `<meshblock>/nx*` blocks in section 1 fix the mesh
  and block dimensions used by the reader when reconstructing MeshBlocks.
- Section 2 carries the binary `RegionSize` and `root_level` used to
  cross-check.

The mesh tree is **not** persisted explicitly — the reader rebuilds it by
inserting each `LogicalLocation` from the ID list.

## 4. Refinement semantics

### Cell-centered fields (hydro `u`, CR `u_cr`, scalars `s`, GR `w`/`w1`)

- **Refine:** `subdivide_cellcentered(a, R)` — `np.repeat` along each of the
  last three axes. Each parent cell contributes `R³` identical child
  cells. Ghost cells are subdivided too; the simulation regenerates them
  on the next boundary exchange.
- **Coarsen:** `coarsen_cellcentered(a, R)` — reshape into `R³`-cell
  blocks and take the mean. Exactly inverts refine (mean of R identical
  values is that value, to machine precision).

### Face-centered magnetic fields (`Bx1f`, `Bx2f`, `Bx3f`)

Each face-centered array has one extra grid point along its distinguished
axis (e.g. `Bx1f.shape == (ncells3, ncells2, ncells1+1)`).

- **Refine (2× illustrated; R× generalises with `R-1` interior faces per
  parent-cell span):**
  - *Transverse axes.* On the two axes perpendicular to the distinguished
    axis, each parent face-value is `np.repeat`-ed R× so the parent's
    single face-value covers `R × R` child sub-faces.
  - *Distinguished axis, outer faces.* Faces at parent-cell boundaries
    map one-to-one to child faces at those same physical positions —
    copied verbatim (after the transverse repeat).
  - *Distinguished axis, interior faces.* Between each adjacent pair of
    parent faces on that axis, insert `R-1` interior faces by **linear
    interpolation** along the distinguished axis:

    ```
    Bx1f_interior[m] = Bx1f_left + (Bx1f_right - Bx1f_left) * m / R
                       for m = 1 .. R-1
    ```

    Uniform on the transverse axes (same value for all `R × R` sub-faces
    at that axial position).

  This scheme preserves `∇·B == 0` in each child cell. Proof (2×):
  the discrete `∇·B` in each child telescopes to the parent's discrete
  `∇·B` divided by the child's cell-widths — so if the parent has zero
  divergence, every child does. Generalises to arbitrary R by the same
  telescoping.

- **Coarsen:** transverse-axis mean; decimation on the distinguished axis
  (keep every R-th face). Preserves `∫B·dA` across each coarse face and
  exactly inverts the refine step for the boundary faces; interior
  parent faces are dropped and their information is discarded (this is
  fundamental to any face-decimation coarsening).

Test invariant (`test_refine.py`): given a divergence-free input, output
is divergence-free cell-by-cell to machine precision.

### Particles (all containers on a block)

Repartition rule (`particles.repartition`):

```
For each particle p in parent block:
    index_i = floor(MeshCoordsToIndices(p.x1, p.x2, p.x3))
    assign p to the child block whose active cell range contains index_i.
```

The active-cell test matches `MeshBlock::CheckInMeshBlock`. Particles
on a shared face resolve to the lower-index block (floor semantics).
Ghost-region particles (`npar_gh_out_`) are not in the file (writer
passes `include_ghost=false`) so nothing to redistribute there.

**`idmax` propagation:** each output block records `max(idmax[parent] for
parent in parents(output))`. For refine, all N children of one parent
share the parent's idmax. For coarsen, the merged output takes the max
across its parents. Prevents ID collisions when new particles spawn after
restart.

Particle field values are preserved verbatim; only ownership changes.

### PRNG state

Emitted only if the input has a non-empty PRNG section (i.e., the running
code had `Mesh::prandomc->GetNumber() > 0`). If the input has no PRNG
section, the output has none — nothing to reseed.

When present: `N_out_ranks = N_output_blocks` (blockfft: one MeshBlock per
rank). `N_prngs` (number of enrolled PRNGs) is carried through from the
input verbatim — it's a build-time / parameter-file property of the
simulation, not something the tool re-derives.

Emit `N_out_ranks × N_prngs × 16 bytes` back-to-back. For each rank `r`
and PRNG index `p`:

```
seed_out[r][p] = hash_uniform_u64(top_seed, keys=(p, r))
count_out[r][p] = 0
```

`top_seed` defaults to the input's rank-0 PRNG-0 seed. `--seed` overrides.

`hash_uniform_u64` mirrors the finalize step of
`AthenaRandom::HashUniform` in `src/utils/random.hpp` (SplitMix64-style
avalanche, ported as a single ~10-line function). A Python port +
regression test against a C++ probe fixture live in `tests/test_prng.py`.

### User meshblock data

- **Refine:** each child block's user_meshblock_data (int and real) is
  zero-initialized. Rationale: tigress_classic uses these for per-block
  scalar accumulators (`cooling`, `faceflux`, `particle`, `minmax`).
  Zeroing kills cumulative counters at the refinement boundary and lets
  the sim recompute min/max diagnostics on the next cycle. Duplicating
  would cause `factor-of-N_children` overcount for cumulative fields.
- **Coarsen:** the output block's arrays are the element-wise sum over
  its input parents. Preserves cumulative counters and gives an
  approximate parent min/max (recomputed next cycle regardless).

### User mesh data (global)

Copied verbatim into the output. Contents are simulation-global
diagnostics that don't depend on the mesh partition.

## 5. Package layout

```
~/tigris_tools/
├── pyproject.toml
├── src/tigris_tools/
│   └── refine_restart/
│       ├── __init__.py
│       ├── layout.py         # numpy dtypes mirroring C++ structs,
│       │                     # byte-offset helpers, particle-pack format
│       ├── param_block.py    # parse/rewrite the <par_end>-terminated INI
│       ├── reader.py         # streaming reader (yield InputBlockDesc,
│       │                     # read_block_payload on demand)
│       ├── refine.py         # cell + face refine / coarsen
│       ├── particles.py      # repartition + idmax
│       ├── prng.py           # HashUniform-compatible reseed
│       ├── plan.py           # build output block list, size each block
│       ├── writer.py         # streaming writer, two-pass
│       └── cli.py            # argparse entry point
└── tests/refine_restart/
    ├── conftest.py           # synthetic .rst builder
    ├── test_layout.py        # struct layouts vs C++ probe fixture
    ├── test_refine.py        # cell + face refine/coarsen invertibility,
    │                         # divergence-free preservation
    ├── test_particles.py     # boundary tie-breaks, idmax
    ├── test_prng.py          # hash_uniform_u64 matches C++
    ├── test_endtoend.py      # synthetic .rst → refine → coarsen ≡ input
    └── fixtures/             # small hand-built .rst files (KB range)
```

CLI entry: `python -m tigris_tools.refine_restart`.

## 6. Data flow

Two passes; peak resident memory bounded by one input block + one output
block worth of arrays.

**Read phase (once):**

1. `param_block.parse(f)` — read INI text up to `<par_end>`; expose the
   `<restart>`, `<mesh>`, `<meshblock>` blocks as dicts.
2. `layout.read_mesh_header(f)` — read the fixed-size binary header.
3. Read the user_mesh_data blob (small; hold in memory).
4. `reader.read_id_list(f, nbtotal)` — read all
   `(LogicalLocation, cost, byte_size)` records; compute each block's
   absolute file offset by cumulative-summing the payload sizes.

**Plan phase (once, in memory):**

5. `plan.build_output_blocks(inputs, refine_factor, in_block_size,
   out_block_size)` returns a list of `OutputBlockDesc`:
   - New `LogicalLocation` list, all at `level = input.root_level` (flat
     mesh at the new resolution).
   - Cost per output block = `1.0` (uniform; the sim's load balancer
     re-derives real costs after the first cycle).
   - `parents: list[int]` — indices into the input block list.
   - Estimated byte size — computed from field shapes plus a particle
     count obtained by pre-reading each parent's particle table once and
     caching the per-child partition.

**Write phase (streaming):**

6. Open output file. Rewrite the parameter block: patch `<mesh>/nx*`,
   `<meshblock>/nx*`, and `<restart>/*_user_meshblock_data_size_*`
   entries; leave everything else verbatim. Write up to and including
   `<par_end>`.
7. Write the new binary header (`nbtotal`, `root_level` = input's,
   `mesh_size` with adjusted `nx*`, unchanged `time`, `dt`, `ncycle`).
8. Write the user_mesh_data blob verbatim.
9. Write the new ID list (`OutputBlockDesc` records in canonical order).
10. Stream block payloads:
    - For each `OutputBlockDesc`, read the parent input block(s) needed
      (already cached from the particle pre-pass), transform the arrays
      (cell + face + particles + user data), pack, write.
    - Evict parent caches once all their children have been emitted.
11. Write the PRNG section (`N_output_blocks` × `Nprngs` × 16 bytes).
12. Close.

**Parallelism note:** every step past #4 partitions cleanly over output
blocks. MPI upgrade replaces the sequential loop in step 10 with an
`MPI_Comm_split` over output block indices and `MPI_File_write_at_all`
calls; no algorithmic change.

## 7. Data structures

Defined in `layout.py`; all mirror the on-disk C++ layout.

```python
@dataclass
class LogicalLocation:
    lx1: int; lx2: int; lx3: int
    level: int

@dataclass
class RegionSize:
    x1min: float; x2min: float; x3min: float
    x1max: float; x2max: float; x3max: float
    x1rat: float; x2rat: float; x3rat: float
    nx1: int; nx2: int; nx3: int

@dataclass
class MeshHeader:
    nbtotal: int; root_level: int
    mesh_size: RegionSize
    time: float; dt: float; ncycle: int

@dataclass
class InputBlockDesc:
    loc: LogicalLocation
    cost: float
    byte_size: int
    file_offset: int

@dataclass
class OutputBlockDesc:
    loc: LogicalLocation
    cost: float
    byte_size: int          # filled in during plan phase (pass 1)
    parents: list[int]      # indices into InputBlockDesc list

@dataclass
class ParticleTable:
    npar: int
    idmax: int
    intprop: np.ndarray     # shape (nint, npar), dtype int32
    realprop: np.ndarray    # shape (nreal, npar), dtype float64

@dataclass
class BlockPayload:
    hydro_u: np.ndarray                   # (NHYDRO, ncells3, ncells2, ncells1)
    field_bx1f: np.ndarray | None         # (ncells3, ncells2, ncells1+1)
    field_bx2f: np.ndarray | None
    field_bx3f: np.ndarray | None
    particles: list[ParticleTable]        # one per enrolled container
    cr_u: np.ndarray | None               # (NCRG, 4, ncells3, ncells2, ncells1)
    scalars_s: np.ndarray | None          # (NSCALARS, ncells3, ncells2, ncells1)
    ublock_int: list[np.ndarray]
    ublock_real: list[np.ndarray]
```

Struct layouts (`LogicalLocation`, `RegionSize`) are declared as
`np.dtype([...], align=True)` with explicit field order matching the C++
class definitions. `test_layout.py` compares against a C++ probe fixture
(binary bytes produced by a tiny C++ program that constructs known
instances) to catch alignment drift.

## 8. CLI

```
python -m tigris_tools.refine_restart INPUT.rst OUTPUT.rst
    (--refine N | --coarsen N)     # positive integer, mutually exclusive
    [--block-size NX,NY,NZ]         # target meshblock size; default = keep input's
    [--seed U64]                    # top-level seed for PRNG reseed;
                                    #   default = input's rank-0 PRNG-0 seed
                                    #   (ignored if input has no PRNG section)
    [--verify]                      # after write, re-read output and run
                                    #   Tier-1 refine⇄coarsen round-trip
    [--dry-run]                     # do pass 1 only; print output plan; no write
    [-v/--verbose]                  # per-block progress logging
```

Exit codes: `0` success, `2` argument / pre-flight validation failure,
`3` I/O error, `4` output-verification failure.

## 9. Error handling

### Pre-flight (before any file write)

Fail with an actionable message if any of these fails:

- `--refine N` and `--coarsen N` both set, or neither set.
- Block-size divisibility: `(R · B_in[i]) % B_out[i] != 0` for any axis.
- Mesh divisibility: `(R · mesh_size.nx[i]) % B_out[i] != 0` for any axis.
- Coarsen block-count divisibility: input's root-grid dimension in each
  axis is not divisible by the required coarsen factor.
- Coordinate system is not Cartesian (parameter block check).

### Runtime

- **Missing `<restart>/*` fields** (older checkpoints): honor
  `strict_restart=false` semantics — warn once, list defaulted fields,
  continue.
- **Field-count mismatches** (NHYDRO / NSCALARS / NCRG) between input's
  `<restart>/*` and the tool's built-in expectation: the tool reads counts
  from the input, so any consistent input works; only structural
  misalignment (e.g. inconsistent user-data-size sums) is fatal.
- **File-truncation or bad offsets** in input: fatal, with the offending
  offset and expected size in the message.

### Output posture

Writer opens `OUTPUT.rst` only after pass 1 succeeds. Any pass-2 failure
deletes the partially-written output before exiting non-zero.

## 10. Verification

Three tiers, all in `tests/refine_restart/`.

### Tier 1 — refine⇄coarsen round-trip byte-check

Fast (~seconds), always in CI. Builds a synthetic `.rst` from raw bytes,
refines by R, coarsens by R, and asserts:

- All array bytes match to zero tolerance (piecewise-constant + volume-
  average is exactly invertible in IEEE 754).
- All particles present with identical field values (per-block ordering
  may differ; test compares as sets keyed by particle ID).
- Parameter-block edits round-trip (input mesh/meshblock dimensions
  recovered).

### Tier 2 — per-block conservation

Fast, always in CI. For each parent/child mapping:

- `Σ_children ρ_c · dV_c == ρ_p · dV_p` for every cell-centered
  quantity (hydro `u`, CR `u_cr`, scalars `s`).
- `Σ_child_faces_on_parent_face Bx1f · dA_c == Bx1f · dA_p`, similarly
  for x2f, x3f.
- `∇·B == 0` cell-by-cell on both input and output when input is
  divergence-free.

### Tier 3 — load-and-step smoke test

Slow, opt-in (`pytest -m slow`, requires `TIGRIS_CRMHD_EXE` env var
pointing at a built simulation binary):

1. Native short run at low resolution, produce `low.rst`.
2. `refine_restart low.rst high.rst --refine 2`.
3. Restart `tigris_crmhd.exe` from `high.rst`, take 1 cycle.
4. Assert exit 0; no NaNs in resulting HDF5/rst; expected `nbtotal`.

Runs on a workstation (single-rank). Not gated by CI.

## 11. Fixtures & test data

- **Synthetic `.rst` builder** in `tests/refine_restart/conftest.py`.
  Generates a valid `.rst` from parameters (nblocks, block_size, physics
  flags, particle counts). Byte-level output, no simulation binary
  required. Handles all sections including the parameter block and PRNG.
- **C++ probe fixture** — a tiny standalone `.cpp` that constructs known
  `LogicalLocation` and `RegionSize` instances and dumps their bytes;
  checked-in as a binary file plus the source. Used only by
  `test_layout.py` and regenerated on demand (not on every test run).
- **Example checkpoint pointer** — `TIGRESS_TEST_RST` env var pointing at
  the on-cluster example in the description. Used only in Tier-3 runs
  and never in CI.

## 12. Open items deferred to implementation

- Exact byte layout of `RegionSize` — pin down with a probe fixture once
  the package is scaffolded; adjust `np.dtype` if compiler alignment
  differs from what a naive Python dtype would give.
- Exact `hash_uniform_u64` mixing constants — port from
  `src/utils/random.hpp` verbatim during implementation and test against
  a C++-generated table.
- Whether to expose an `--allow-non-blockfft` flag that lifts the
  `N_out_ranks == N_output_blocks` constraint for future runs that don't
  use blockfft. Default off for v1.

## 13. Non-goals for this spec

This spec defines what to build and why. Concrete task decomposition,
file-by-file line counts, and test naming go in the implementation plan
(produced by the `writing-plans` skill next).
