# Restart slice summaries

## Goal

`restart_slices` will create the two central-plane NetCDF caches consumed by
`TIGRESS-CR/python/plot_slices.py::plot_slices_cr` directly from a TIGRESS++
restart file.  It is intended for runs that do not write ATHDF snapshots.

The first milestone is deliberately limited to slices:

- `z=0`, stored as `allslc.z`
- `y=0`, stored as `allslc.y`
- the eight default `plot_slices_cr` panels and their vector overlays

Projections and particle overlays required by `plot_snapshot` are a later
milestone.  The restart parser and reconstructed-field code will be shared by
both milestones.

## User interface

The target command is one operation:

```sh
tigris-slices TIGRESS.00025.rst --savdir /path/to/analysis
```

It will write both slice caches and, unless disabled, the summary PNG.  The
converter will stream meshblocks and will not assemble the full three-
dimensional domain.

Malformed historical checkpoints require their same-cycle particle sidecar:

```sh
tigris-slices TIGRESS.00039.rst \
  --particle TIGRESS.out3.00169.par0.parbin \
  --savdir /path/to/analysis
```

The particle file is used to validate recovered byte offsets. It is not used
to construct any fluid or CR field.

For all numbered restarts in a directory, use the resumable batch command:

```sh
run=/nobackup/ckim14/tigress_classic/crmhd_duale-8pc-R16_tall-rst
tigris-slices-all "$run" --prefix TIGRESS --savdir "$run" --dry-run
tigris-slices-all "$run" --prefix TIGRESS --savdir "$run"
```

The first command reports every restart, whether its caches are fresh, and
the independently matched particle sidecar needed for recovery. The second
generates stale or missing pairs sequentially. A failed checkpoint is reported
and does not stop later checkpoints unless `--fail-fast` is supplied.

The standalone package can render the compatible eight-panel summary from the
cache pairs without importing `TIGRESS-CR` or pyathena:

```sh
tigris-plot-slices-all "$run" --prefix TIGRESS --savdir "$run"
```

By default this writes `$run/cr_slices/<run-name>_NNNN.png`. Existing figures
newer than both input caches are skipped. Missing cache pairs are reported while
the command continues to render every available output.

## Projections and snapshots

The combined product command streams every meshblock once and writes both slice
caches and both projection caches while retaining only two-dimensional
accumulators and central-plane tiles in memory:

```sh
tigris-products-all "$run" --prefix TIGRESS --savdir "$run"
```

For output 30 the cache paths are:

```text
<savdir>/allslc.z/allslc.z.00030.nc
<savdir>/allslc.y/allslc.y.00030.nc
<savdir>/prj.z/prj.z.00030.nc
<savdir>/prj.y/prj.y.00030.nc
```

Each dataset has `phase = whole, hot, wc`, a `time` attribute, and the same
available variables as `SliceProj.get_prj`: `Sigma`, mass/thermal/kinetic/metal
fluxes, and CR total/diffusive/advective/streaming energy fluxes. Photochemical
runs additionally receive the component surface densities and emission
measure. Surface quantities are line-of-sight integrals; fluxes are
line-of-sight averages, matching `slc_prj.py`.

Unless the corresponding plotting option is disabled, the same command writes
`<savdir>/cr_slices/<run-name>_NNNN.png` and
`<savdir>/snapshot/snapshot_NNNNN.png`. The snapshot reproduces the default
`plot_snapshot` field layout and uses the exact particles embedded in the
restart rather than selecting a nearby particle output. Existing compatible
caches and figures are skipped unless `--overwrite` is given.

`tigris-projections-all` remains a compatibility alias. When invoked under MPI,
numbered restarts are still processed sequentially to
keep memory bounded. For each restart, ranks open the file read-only and read
disjoint contiguous meshblock ranges. Each rank constructs local y/z
projection accumulators and copies only tiles intersecting the central y/z
planes. `MPI_Reduce` sums every projection field and phase onto rank zero, while
the slice tiles are gathered and assembled there. Only rank zero writes the four
NetCDF files, reads the lightweight embedded-particle records, and renders both
figures. A normal invocation remains a supported one-rank fallback:

```sh
mpiexec -n 8 tigris-products-all "$run" --prefix TIGRESS --savdir "$run"
```

The NAS job for this stage is:

```sh
qsub /home1/ckim14/tigris_tools/pbs/generate_all_restart_projections.pbs
```

After figures are available, reproduce the two movie names used by the original
`plot_slices.py` workflow:

```sh
tigris-make-movies "$run"
```

At the default 15 fps this writes
`<run>/movies/<run-name>_cr_slices.mp4` from `cr_slices/*.png` and
`<run>/movies/<run-name>_snapshot.mp4` from `snapshot/*.png`. The command uses
H.264 with `yuv420p`, uses `ffmpeg` from `PATH` or an installed
`imageio-ffmpeg` fallback, and accepts `--dry-run`, `--kind`, `--fps-in`, and
`--fps-out`.

## Compatibility contract

The cache contract follows `pyathena_tigris.LoadSim.Decorators.check_netcdf`.
For output number 25 the default paths are:

```text
<savdir>/allslc.z/allslc.z.00025.nc
<savdir>/allslc.y/allslc.y.00025.nc
```

An output ID is inserted before the output number, for example
`allslc.z.out2.00025.nc`.  A cache is fresh only when its modification time is
newer than the source restart, matching the existing decorator.

Each dataset will use the same physical cell-center coordinates, variable
names, and `time` attribute as `TIGRESS-CR/python/slc_prj.py::get_slice`.
Fields renamed by that function use the renamed form in NetCDF:

| ATHDF name | Slice-cache name |
| --- | --- |
| `rho` | `density` |
| `press` | `pressure` |
| `vel1`, `vel2`, `vel3` | `velocity1`, `velocity2`, `velocity3` |
| `Bcc1`, `Bcc2`, `Bcc3` | `cell_centered_B1`, `cell_centered_B2`, `cell_centered_B3` |

CR variables retain their ATHDF names, including `0-Ec`, `0-Fc1` through
`0-Fc3`, `0-Sigma_diff1`, `0-Sigma_adv1`, `0-Vs1` through `0-Vs3`, and
`0-Vd1` through `0-Vd3`.

This is data-and-figure compatibility.  The new package will not depend on a
`LoadSimTIGRESSPP` instance.  A later pyathena change can consume the same
files without recreating them.

## Source selection in pyathena

The eventual pyathena integration needs two related changes:

1. `get_data()` selects ATHDF when it exists and the matching restart when it
   does not.
2. The `get_slice(dryrun=True)` freshness path obtains the modification time
   from whichever source was selected.  It must not call
   `load_hdf5(file_only=True)` unconditionally, because that fails before an
   existing restart-derived cache can be opened.

The cache naming helpers in `tigris_tools.restart_slices.cache` are kept free
of xarray and plotting dependencies so the same rules can be reused there.

## Reconstruction contract

The restart stores genuine simulation arrays rather than every output field:

- conserved hydro state `u`
- face-centered magnetic fields
- CR conserved state `u_cr = (Ec, Fc1, Fc2, Fc3)`
- conserved passive scalars
- particles and problem-specific user data

The slice converter must reproduce the C++ output path, not approximate the
missing quantities:

1. Average face fields to cell centers exactly as TIGRESS++ does.
2. Apply the same conserved-to-primitive equation of state, including the
   configured dual-energy behavior.
3. Convert passive conserved scalars to primitive scalar fractions/energies.
4. Reproduce `CosmicRay::UpdateOpacity`, including the CR pressure-gradient
   stencil, self-consistent scattering, ion density, opacity caps, streaming
   velocity, and diffusion velocity.
5. Apply the same floors, parameter values, and unit conventions read from the
   restart parameter block.

Restart ghost zones will be retained while reconstructing a plane so the CR
gradient stencil is correct across meshblock boundaries.  Only active cells
are written to NetCDF.

## Restart integrity issue

The current production files show a nonstandard byte displacement inside the
fixed-width meshblock ID table.  Valid 48-byte records do not remain on a
48-byte stride.

The likely source is `src/outputs/restart.cpp`: each MPI rank serializes a
parameter block containing its random-generator state and computes
`headeroffset` from its local serialized length, while only rank 0 writes the
header.  If those lengths differ, collective ID-table and payload writes use
different base offsets.  The native restart reader and this Python reader both
correctly expect one fixed offset.

The reader therefore validates every ID record and fails explicitly on this
condition.  It will not silently reinterpret or interpolate corrupted data.
Recovery of already-written files uses the matching parbin output as an
independent block-size and byte-offset check:

- generate the uniform root-grid locations in TIGRESS++ Z-order;
- obtain each block's particle count and exact particle bytes from the parbin;
- infer the finite set of rank-dependent header displacements from complete
  ID records;
- score damaged records only against those observed displacements;
- require every embedded particle record to match the sidecar byte-for-byte at
  the recovered payload position;
- reject recovery if any displacement is ambiguous or any particle record
  differs.

For `TIGRESS.00025.rst`, this recovers all 4,096 block offsets.  All 4,096
embedded particle records, including 175 non-empty records, match the same-run
parbin exactly.  The observed shifts are `0`, `-9`, `-18`, `-27`, and `-36`
bytes.  There are 1,787 overlapping rank boundaries (at most 27 bytes) and
1,564 gaps (at most 36 bytes).  End overlaps remain within the 420-byte user
MeshBlock tail.  A possible beginning overlap reaches only the first few
density ghost-cell bytes and does not touch active cells or the central-plane
stencils used by this project.

The same recovery and byte-for-byte particle validation also passes for the
first and latest inspected checkpoints, `TIGRESS.00021.rst` and
`TIGRESS.00039.rst`, with the same displacement pattern.

The sidecar is a recovery aid for these malformed historical files, not a
normal input requirement.  Files produced after the writer fix use the
ordinary restart index and need no parbin.

The writer fix should broadcast the rank-0 serialized header size before any
rank calculates an output offset.  Changing `~/tigris` is outside this package
and will be done only as an explicit change to the simulation source.

### Trigger and compatibility fix

The first refined checkpoint was structurally valid, but its textual
`<randomN>` values still described the input checkpoint while the converter
generated new rank-dependent binary PRNG records.  After loading that file,
`SaveAll` expanded each rank's parameter formatting width according to the
decimal length of its new seeds.  This exposed the rank-local `headeroffset`
bug in the native writer.

`refine_restart` now writes the generated rank-0 PRNG seeds and counts into the
text header as well as the binary section.  The seeds are 20-digit unsigned
values, which pins every random block's `ParameterDump` width to the maximum
possible uint64 width on all ranks.  This prevents current unpatched TIGRESS++
executables from producing rank-dependent offsets after reading a newly
refined checkpoint.

### Repairing affected checkpoints

The non-destructive repair command is:

```sh
repair-restart INPUT.rst SAME_CYCLE.parbin OUTPUT.rst
```

It refuses to overwrite either the input or an existing output.  It writes a
canonical fixed-stride ID table, copies recovered payloads into contiguous
positions, replaces potentially damaged leading density corner-ghost values
with a neighboring intact ghost value, zeros affected trailing diagnostic
values, and writes synchronized fresh PRNG records.

For `TIGRESS.00039.rst`, the repaired copy
`TIGRESS.00039.repaired.rst` passes the ordinary Python restart reader and the
unchanged native mesh test:

```sh
athena -r TIGRESS.00039.repaired.rst -m 4096
```

The native code reconstructs all 4,096 MeshBlocks with one block per rank.
This validates the header and block tree but intentionally stops before a
full payload load.  A short scheduled 4,096-rank restart remains required
before using the repaired checkpoint for production evolution.

## Validation references

The sibling ATHDF run provides the expected schema and a useful
order-of-magnitude reference:

```text
restart: TIGRESS.00025.rst
ATHDF:   TIGRESS.out2.00155.athdf
time:    1025.000050880921 (restart), 1025 (float32 ATHDF attribute)
cycle:   2551035
```

It is not a cell-by-cell gold state: the restart-only continuation has already
diverged from the sibling run, and even the particle outputs differ at the
same nominal time.  Numerical validation of the reconstruction therefore
needs either a valid restart/ATHDF pair from one run or an ATHDF emitted from a
corrected copy of the recovered checkpoint.

Validation proceeds from raw to derived fields:

1. mesh geometry and central-plane coordinates;
2. density, velocity, CR conserved variables, and passive scalars;
3. cell-centered magnetic field and thermal pressure;
4. CR opacity, streaming velocity, and diffusion velocity;
5. complete NetCDF schema and the rendered `plot_slices_cr` figure.

### Checkpoint 39 validation

The implemented pipeline was run on the recovered index for
`TIGRESS.00039.rst` with `TIGRESS.out3.00169.par0.parbin`:

- 568 intersecting MeshBlocks were streamed; the full 3D mesh was never
  assembled.
- The nearest central coordinates match xarray's tie behavior: `y=4 pc` and
  `z=4 pc` on this 8 pc grid.
- `allslc.y` has dimensions `z=2048, x=256`; `allslc.z` has dimensions
  `y=256, x=256`.
- All 25 cache variables are finite in every active cell on both planes.
- Pressure reconstructed from total energy agrees with the independent dual
  internal-energy scalar at median relative errors of `1.0e-6` (`y` slice)
  and `5.2e-6` (`z` slice). The 99th percentiles are `2.1e-5` and `6.2e-4`.
- The unchanged `TIGRESS-CR/python/plot_slices.py::plot_slices_cr` reads the
  two NetCDF files and renders all eight default fields and vector overlays.

This verifies byte offsets, active arrays, cache schema, derived-field calls,
and plotting interoperability for the current checkpoint. The self-consistent
CR formulas are a direct vectorized port of the configured C++ path, but a
cell-by-cell CR comparison still requires a restart and ATHDF emitted by the
same run state.

Comparisons will report maximum absolute and relative errors per field, with
special checks along meshblock boundaries.

## Implementation phases

- [x] Record the slice-cache naming and freshness contract.
- [x] Add early validation for malformed fixed-width restart ID tables.
- [x] Recover and independently validate all block offsets in the existing
  production restart using its same-cycle particle sidecar.
- [x] Implement streaming selection of blocks intersecting `y=0` and `z=0`.
- [x] Port hydro, magnetic, and passive-scalar reconstruction.
- [x] Port the current single-group MHD `DefaultOpacity` CR reconstruction.
- [x] Write xarray NetCDF datasets matching `slc_prj.py`.
- [x] Validate the cache by running the existing `plot_slices_cr` unchanged.
- [ ] Validate against the time-1025 ATHDF reference.
- [ ] Add restart fallback and restart-aware freshness to pyathena.
- [ ] Extend the pipeline to projections and particles for `plot_snapshot`.
