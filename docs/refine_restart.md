# Post-Processing Restart Refinement Tool

## Context

This repository is an astrophysical simulation code, Athena++, with a custom
module for CRMHD simulations called **TIGRESS++**. The canonical configure
invocation for a CRMHD run is:

```sh
./configure.py --prob=tigress_classic --nghost=4 -fft -fb --grav=blockfft \
               -mpi -hdf5 -b --cr=mg --flux=hlld --cxx=icpx
```

During a run, the code periodically writes checkpoint (`.rst`) files. This
project builds a **standalone helper** that reads such a checkpoint, refines
(or coarsens) the mesh and its data, and writes a new checkpoint file in the
**exact same binary layout** so the simulation can restart from it without
any code changes.

## Restart File Format

The `.rst` file is a single flat binary stream with the following sections,
in order. All numeric fields are native-endian, native-width (`int` = 32-bit,
`Real` = 64-bit `double` in a standard build, `IOWrapperSizeT` = `uint64`).

### 1. ParameterInput text block

- Written first, by rank 0, by dumping the runtime `ParameterInput` as an
  Athena++ INI-style text section (`src/outputs/restart.cpp` calls
  `pin->ParameterDump(ost)`).
- Terminated by the literal line `<par_end>`.
- On restart, `ParameterInput::LoadFromFile` (`src/parameter_input.cpp`)
  consumes this block up to and including `<par_end>` before the `Mesh`
  restart constructor runs. The scan reads at most 40 KB looking for the
  terminator.
- **Critical for refinement:** `RestartOutput::WriteParameterInput`
  populates a `<restart>` block that carries the reader-side schema:
  `nint_user_mesh_data`, `nreal_user_mesh_data`, their per-array
  byte-sizes, `magnetic_fields_enabled`, `cr_enabled`, `ncrg` (multigroup
  CR), `nscalars`, `nint_user_meshblock_data`, `nreal_user_meshblock_data`,
  and their per-array sizes. The `MeshBlock` restart constructor uses these
  to interpret the binary payload. The `<meshblock>` block (`nx1`, `nx2`,
  `nx3`) determines the block size used when reconstructing MeshBlocks —
  so **changing the meshblock size on refinement is done here, not in the
  binary header**.

### 2. Mesh binary header

Written once by rank 0, `3*sizeof(int) + 2*sizeof(Real) + sizeof(RegionSize)`
bytes in this exact order:

| Field         | Type         | Meaning                                    |
|---------------|--------------|--------------------------------------------|
| `nbtotal`     | `int`        | total number of MeshBlocks in the mesh     |
| `root_level`  | `int`        | refinement level of the root grid          |
| `mesh_size`   | `RegionSize` | full mesh extents + total cells `nx1/2/3`  |
| `time`        | `Real`       | simulation time of the checkpoint          |
| `dt`          | `Real`       | last timestep                              |
| `ncycle`      | `int`        | cycle counter                              |

### 3. User Mesh data (optional)

If `nint_user_mesh_data > 0` or `nreal_user_mesh_data > 0`, a single
contiguous blob follows: all integer arrays first (in enrolled order),
then all real arrays. Each array's byte length is recovered from the
`<restart>` block. Written by rank 0.

### 4. Per-block ID list

For each of `nbtotal` blocks, in global-ID order, one record of
`sizeof(LogicalLocation) + sizeof(double) + sizeof(IOWrapperSizeT)` bytes:

| Field  | Type              | Meaning                                        |
|--------|-------------------|------------------------------------------------|
| `loc`  | `LogicalLocation` | `{level, lx1, lx2, lx3}` in the mesh tree      |
| `cost` | `double`          | load-balance cost weight                       |
| `size` | `IOWrapperSizeT`  | byte length of this block's data blob (§5)     |

The mesh tree is **not written explicitly** — the reader rebuilds it by
inserting each `loc` via `tree.AddMeshBlockWithoutRefine`, then assigns
GIDs by traversal. Each rank pre-computes its own blocks' sizes via
`MeshBlock::GetBlockSizeInBytes`, and rank 0 assembles the full list with
`MPI_Allgatherv`; the list is then written collectively with
`Write_at_all`.

### 5. Per-block data blobs

Written collectively (`Write_at_all`) at offsets defined by the ID list.
Each block's blob concatenates the following, in this exact order — any
section absent from the build is omitted:

1. Hydro conserved variables: `phydro->u` (with ghosts included, as
   `NHYDRO × ncells3 × ncells2 × ncells1` doubles).
2. If `GENERAL_RELATIVITY`: `phydro->w` then `phydro->w1` (primitives at
   current and previous stage).
3. If `MAGNETIC_FIELDS_ENABLED`: face-centered fields `b.x1f`, `b.x2f`,
   `b.x3f` in that order.
4. **For each `Particles` container** attached to the block, packed by
   `Particles::PackParticlesForRestart`:
   - `int npar` — number of particles owned by this block.
   - `int idmax` — highest particle unique ID seen (a per-container
     global-max counter).
   - `nint × int[npar]` — integer property arrays (one contiguous slab per
     property).
   - `nreal × Real[npar]` — real property arrays (same layout).
5. If `CR_ENABLED`: `pcr->u_cr`. For multigroup CR (`--cr=mg`,
   `CR_ENABLED == 2`), the array has shape `NCRG × 4 × ncells3 × ncells2
   × ncells1`; groups are written contiguously and `NCRG` is stored in
   `<restart>/ncrg`.
6. If `NSCALARS > 0`: `pscalars->s`
   (`NSCALARS × ncells3 × ncells2 × ncells1`).
7. User MeshBlock data: all `iuser_meshblock_data[n]` (integer, in order),
   then all `ruser_meshblock_data[n]` (real, in order). Sizes come from
   the `<restart>` block.

Note that ghost cells are included in every array — the ghosts are
regenerated on restart via boundary comms, but the storage is
ghost-inclusive because sizes are computed from `AthenaArray::GetSizeInBytes`.

### 6. Random-number container (optional)

Present only if `Mesh::prandomc->GetNumber() > 0`. Written collectively,
one fixed-size record per rank (`mysize` per rank, but each rank writes
the state of all its enrolled PRNGs). For each `AthenaRandom`: `uint64
seed` followed by `uint64 count`.

### Metadata evolution

The set of `<restart>` fields, particle properties, and even the presence
of some sections (multigroup CR, passive scalars) has grown with the
code. On read, the `MeshBlock` restart constructor is tolerant when
`<restart>/strict_restart = false`: mismatches in `NCRG`, `NSCALARS`, and
user-data sizes emit warnings and skip or truncate rather than aborting.
The refinement tool must either target a specific checkpoint schema
(preferred) or replicate this graceful degradation. It must, at
minimum, preserve every `<restart>/*` key it does not modify.

### Example checkpoint

```
/scratch/gpfs/EOST/changgoo/tigress_classic/crmhd-16pc-b0.1-mhdbc_diode-crbc_lngrad_out-icpx/TIGRESS.final.rst
```

### Implications for the refinement tool

- **Two-part metadata surface.** Some quantities live in the ParameterInput
  text section (`<restart>`, `<meshblock>`, `<mesh>`), others in the binary
  Mesh header. Refinement must update **both**: e.g. bumping `mesh_size.nx*`
  in the header and, if the block size changes, `<meshblock>/nx*` in the
  text section.
- **The mesh tree is implicit.** Only the sorted `LogicalLocation` list needs
  to be emitted; the reader reconstructs the tree from it.
- **Offsets are content-addressed.** Section 5's file offsets are derived from
  the ID-list sizes, so any change to per-block byte counts (e.g. more
  particles per child block after redistribution) requires re-computing every
  downstream offset.
- **Ghost cells are stored.** Refinement can either subdivide only the active
  region and zero the ghosts (they will be overwritten on restart), or
  subdivide the full ghost-inclusive array — the latter is simpler as long as
  the child block's ghost layout stays consistent.

## Goal

Given an existing checkpoint file, produce a new checkpoint that describes
the **same physical domain** at a **different mesh resolution** (or with a
different meshblock size), such that the current simulation binary can
restart from it directly.

### Why Not Use Athena++'s Built-In AMR?

TIGRESS++ CRMHD does not fully support mesh refinement, and several modules
have hard constraints — notably, `blockfft` currently supports only a single
meshblock per MPI rank. A post-processing refinement step lets us
side-step these constraints while still moving between resolutions.

### Concrete Example

- **Input checkpoint:** total mesh `64 × 64 × 512`, meshblocks of `32³`,
  run on 64 ranks.
- **Desired output checkpoint:** total mesh `128 × 128 × 1024`
  (same physical box, 2× per axis), meshblocks of `32³`, to be run on
  512 ranks.

The tool must also support changing the meshblock size across refinement —
e.g., refining `32³` blocks by 2× but writing the result as `64³` blocks so
the new run can reuse the original rank count.

## Requirements

### Refinement Semantics

Prolongation does **not** need to be interpolative. It suffices to
**subdivide each coarse cell into 2×2×2 child cells that carry the parent
cell's value** (piecewise-constant / volume-copy). This guarantees
conservation for cell-centered quantities and keeps face-centered fields
divergence-free trivially.

### Mesh Tree and Metadata

Refinement changes the number of meshblocks, the block-to-rank mapping, and
the mesh tree. All of these must be updated consistently and written in the
exact form expected by the `Mesh` and `MeshBlock` restart constructors —
otherwise the new checkpoint will fail to load.

### Particles

Particle data owned by each coarse meshblock must be **redistributed** into
the corresponding refined meshblocks according to particle position. No
particle values need to change; only their ownership.

### Coarsening

The tool should also support the inverse operation (2× coarsening), useful
for downsampling a high-resolution checkpoint. Coarsening is
volume-averaging for cell-centered quantities and area-averaging for
face-centered fields.

## Design Considerations

### Memory

The simplest approach — read the whole checkpoint into memory, refine, write
it back — is not viable for large runs. Refining a large checkpoint (say,
`512³` at higher resolution) would exceed the memory of a single node.

The tool should stream data **one meshblock at a time** (or a small number
at a time), so that peak memory is bounded by a handful of meshblocks
rather than the whole mesh.

### Parallelism

An MPI-parallel implementation (matching Athena++'s own I/O model) would
let the tool handle arbitrarily large checkpoints. This is a stretch goal;
a serial streaming implementation is the minimum viable version.

### Language

Implementation language is open — Python, Julia, Rust, or C++ are all
candidates. The choice should be driven by:

- Ability to read/write the exact binary layout produced by
  `src/outputs/restart.cpp`.
- Ease of streaming large binary data.
- Availability of MPI bindings (if the parallel version is pursued).

## Deliverables

1. A helper tool that reads an Athena++/TIGRESS++ restart file, refines or
   coarsens it by an integer factor, and writes a new restart file.
2. Support for changing the meshblock size independently of the refinement
   factor.
3. Consistent handling of the mesh tree, block-to-rank mapping, and
   metadata so the new file is a drop-in restart target.
4. Redistribution of particle data across the new meshblock partition.
5. A verification path: restart the current simulation binary from a
   refined checkpoint and confirm the run proceeds without error and with
   the expected physical state.
