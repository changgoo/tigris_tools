from __future__ import annotations

import csv
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from tigris_tools.refine_restart import layout
from tigris_tools.refine_restart.convert import _fixed_payload_sizes, _parse_payload

from .extract import (
    RestartSource,
    SliceResult,
    _cell_coordinates,
    _field_names,
    _nearest_index,
    _reconstruct_block,
    _require_complete,
    _require_uniform_cartesian,
)

M_H_CGS = 1.6735575e-24
K_BOLTZMANN_CGS = 1.380649e-16
M_SUN_CGS = 1.988409870698051e33
PC_CGS = 3.0856775814913673e18
YEAR_CGS = 365.25 * 24.0 * 3600.0
PHASES = ("whole", "hot", "wc")


@dataclass(frozen=True)
class ProjectionUnits:
    surface_density: float
    mass_flux: float
    energy_flux: float
    temperature_mu: float


@dataclass(frozen=True)
class ParticleData:
    integers: np.ndarray
    reals: np.ndarray

    @classmethod
    def empty(cls, nint: int = 0, nreal: int = 0) -> "ParticleData":
        return cls(np.empty((0, nint), dtype=np.int32), np.empty((0, nreal)))


@dataclass(frozen=True)
class ProjectionResult:
    projections: dict[str, dict[str, dict[str, np.ndarray]]]
    coordinates: dict[str, np.ndarray]
    time: float
    cycle: int
    particles: ParticleData
    recovery: object | None
    slices: SliceResult | None = None


def extract_particles(
    restart_path: str | Path,
    *,
    particle_path: str | Path | None = None,
) -> ParticleData:
    """Read only embedded particle records, without loading mesh arrays."""

    with RestartSource(restart_path, particle_path=particle_path) as source:
        assert source.index is not None and source.schema is not None
        schema = source.schema
        if not schema.particle_nreal:
            return ParticleData.empty()
        before, _after = _fixed_payload_sizes(
            schema.block_size,
            schema.nscalars,
            schema.ncrg,
            schema.magnetic_fields,
            schema.cr_enabled,
        )
        descriptor = os.open(Path(restart_path), os.O_RDONLY)
        integers = []
        reals = []
        try:
            for block in source.index.blocks:
                header = os.pread(descriptor, 8, block.file_offset + before)
                if len(header) != 8:
                    raise EOFError(f"truncated particle header at {block.file_offset + before}")
                npar, _idmax = struct.unpack("<ii", header)
                if npar < 0:
                    raise ValueError(f"negative particle count {npar}")
                if not npar:
                    continue
                int_bytes = npar * schema.particle_nint * layout.INT_SIZE
                real_bytes = npar * schema.particle_nreal * layout.REAL_SIZE
                record = os.pread(
                    descriptor, int_bytes + real_bytes, block.file_offset + before + 8
                )
                if len(record) != int_bytes + real_bytes:
                    raise EOFError("truncated embedded particle record")
                integers.append(
                    np.frombuffer(record, dtype=layout.INT_DTYPE, count=npar * schema.particle_nint)
                    .reshape(schema.particle_nint, npar)
                    .T.copy()
                )
                reals.append(
                    np.frombuffer(
                        record,
                        dtype=layout.REAL_DTYPE,
                        count=npar * schema.particle_nreal,
                        offset=int_bytes,
                    )
                    .reshape(schema.particle_nreal, npar)
                    .T.copy()
                )
        finally:
            os.close(descriptor)
        return ParticleData(
            np.concatenate(integers) if integers else np.empty((0, schema.particle_nint)),
            np.concatenate(reals) if reals else np.empty((0, schema.particle_nreal)),
        )


def projection_units(params) -> ProjectionUnits:
    units = params.values.get("units", {})
    mass_cgs = float(units["mass_cgs"])
    length_cgs = float(units["length_cgs"])
    time_cgs = float(units["time_cgs"])
    density_cgs = mass_cgs / length_cgs**3
    velocity_cgs = length_cgs / time_cgs
    energy_density_cgs = density_cgs * velocity_cgs**2
    return ProjectionUnits(
        surface_density=density_cgs * length_cgs * PC_CGS**2 / M_SUN_CGS,
        mass_flux=density_cgs * velocity_cgs * (1000.0 * PC_CGS) ** 2 * YEAR_CGS
        / M_SUN_CGS,
        energy_flux=energy_density_cgs
        * velocity_cgs
        * (1000.0 * PC_CGS) ** 2
        * YEAR_CGS,
        temperature_mu=velocity_cgs**2 * M_H_CGS / K_BOLTZMANN_CGS,
    )


def derive_projection_fields(
    fields: Mapping[str, np.ndarray],
    units: ProjectionUnits,
    gamma: float,
) -> dict[str, np.ndarray]:
    """Reproduce the per-cell quantities constructed by ``SliceProj.get_prj``."""

    density = fields["density"]
    vz = fields["velocity3"]
    result = {
        "Sigma": density * units.surface_density,
        "mflux": density * vz * units.mass_flux,
        "teflux": gamma / (gamma - 1.0) * fields["pressure"] * vz * units.energy_flux,
        "keflux": (
            0.5
            * density
            * vz
            * sum(fields[f"velocity{i}"] ** 2 for i in range(1, 4))
            * units.energy_flux
        ),
    }
    if "rmetal" in fields:
        result["mZflux"] = density * fields["rmetal"] * vz * units.mass_flux
    if "0-Fc3" in fields:
        result["creflux"] = fields["0-Fc3"] * units.energy_flux
    if "0-Vd3" in fields:
        enthalpy = (4.0 / 3.0) * fields["0-Ec"] * units.energy_flux
        result["creflux_diff"] = enthalpy * fields["0-Vd3"]
        result["creflux_adv"] = enthalpy * vz
        result["creflux_str"] = enthalpy * fields["0-Vs3"]
    if {"xHI", "xH2", "xe"}.issubset(fields):
        result["Sigma_HI"] = density * fields["xHI"] * units.surface_density
        result["Sigma_H2"] = 2.0 * density * fields["xH2"] * units.surface_density
        result["Sigma_HII"] = (
            density * (1.0 - fields["xHI"] - 2.0 * fields["xH2"])
            * units.surface_density
        )
        result["Sigma_EL"] = density * fields["xe"] * units.surface_density
        result["EM"] = (density * fields["xe"]) ** 2
    return result


def gas_temperature(
    fields: Mapping[str, np.ndarray],
    units: ProjectionUnits,
    mu_from_t1: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    t1 = fields["pressure"] / fields["density"] * units.temperature_mu
    if {"xe", "xH2"}.issubset(fields):
        mu = 1.4 / (1.1 + fields["xe"] - fields["xH2"])
    elif mu_from_t1 is None:
        mu = 1.27
    else:
        mu = mu_from_t1(t1)
    return mu * t1


def extract_projections(
    restart_path: str | Path,
    *,
    particle_path: str | Path | None = None,
    verbose: bool = False,
    comm=None,
    include_slices: bool = False,
) -> ProjectionResult:
    """Stream root meshblocks into y/z projections, optionally across MPI ranks.

    Every rank holds a local two-dimensional accumulator and reads a contiguous
    range of the restart payload. The accumulators are summed onto rank zero;
    only rank zero receives populated ``ProjectionResult.projections``.
    """

    restart_path = Path(restart_path)
    rank = 0 if comm is None else comm.Get_rank()
    size = 1 if comm is None else comm.Get_size()
    # The high-level progress report is useful; one reader line per 8 MiB block is not.
    with RestartSource(restart_path, particle_path=particle_path, verbose=False) as source:
        assert source.index is not None and source.schema is not None
        index = source.index
        schema = source.schema
        _require_uniform_cartesian(index, schema)
        coords = _cell_coordinates(index)
        selected = {axis: _nearest_index(coords[axis], 0.0) for axis in ("y", "z")}
        units = projection_units(index.params)
        gamma = index.params.get_float("hydro", "gamma")
        if gamma is None:
            raise ValueError("missing <hydro>/gamma")
        mu_from_t1 = _read_mu_interpolator(restart_path.parent / "cool_ftn.runtime.csv")
        mesh = index.header.mesh_size
        spacing = (mesh.x1len / mesh.nx1, mesh.x2len / mesh.nx2, mesh.x3len / mesh.nx3)
        shapes = {"z": (mesh.nx2, mesh.nx1), "y": (mesh.nx3, mesh.nx1)}
        projections: dict[str, dict[str, dict[str, np.ndarray]]] = {
            axis: {phase: {} for phase in PHASES} for axis in ("z", "y")
        }
        local_slice_tiles = []
        bx, by, bz = schema.block_size
        g = layout.NGHOST
        cr_reconstructor = None
        if schema.cr_enabled:
            from .cr import CRReconstructor

            cr_reconstructor = CRReconstructor(index.params, restart_path.parent)

        blocks = sorted(index.blocks, key=lambda block: block.file_offset)
        if size > len(blocks):
            raise ValueError(f"MPI ranks ({size}) exceed restart meshblocks ({len(blocks)})")
        first, stop = _block_range(len(blocks), rank, size)
        local_blocks = blocks[first:stop]
        local_error = None
        try:
            for number, block in enumerate(local_blocks, start=1):
                payload = _parse_payload(source.read_block_bytes(block), schema)
                reconstructed = _reconstruct_block(
                    payload, index, schema, cr_reconstructor, spacing
                )
                active = {
                    name: values[g : g + bz, g : g + by, g : g + bx]
                    for name, values in reconstructed.items()
                }
                per_cell = derive_projection_fields(active, units, gamma)
                temperature = gas_temperature(active, units, mu_from_t1)
                masks = {
                    "whole": 1.0,
                    "hot": temperature > 2.0e4,
                    "wc": temperature <= 2.0e4,
                }
                _accumulate_block(
                    projections, per_cell, masks, block.loc, schema.block_size, spacing, shapes
                )
                if include_slices:
                    local_slice_tiles.extend(
                        _slice_tiles(active, block.loc, schema.block_size, selected)
                    )
                if verbose and rank == 0 and (
                    number % 128 == 0 or number == len(local_blocks)
                ):
                    print(
                        f"restart_projections: rank=0 blocks={number}/{len(local_blocks)} "
                        f"global={len(blocks)} ranks={size}",
                        flush=True,
                    )
        except Exception as exc:
            local_error = f"rank {rank}: {type(exc).__name__}: {exc}"

        if comm is not None:
            errors = comm.allgather(local_error)
            failures = [error for error in errors if error is not None]
            if failures:
                raise RuntimeError("MPI projection failure: " + "; ".join(failures))
            projections = _reduce_projections(projections, comm)
            gathered_tiles = comm.gather(local_slice_tiles, root=0) if include_slices else None
        elif local_error is not None:
            raise RuntimeError(local_error)
        else:
            gathered_tiles = [local_slice_tiles] if include_slices else None

        slices = None
        if rank == 0 and include_slices:
            assert gathered_tiles is not None
            planes = _assemble_slice_tiles(
                gathered_tiles,
                _field_names(index, schema),
                {"z": shapes["z"], "y": shapes["y"]},
            )
            _require_complete(planes)
            slices = SliceResult(
                planes=planes,
                coordinates=coords,
                time=index.header.time,
                cycle=index.header.ncycle,
                selected_indices=selected,
                selected_coordinates={
                    axis: float(coords[axis][cell]) for axis, cell in selected.items()
                },
                recovery=source.recovery,
            )

        return ProjectionResult(
            projections=projections,
            coordinates=coords,
            time=index.header.time,
            cycle=index.header.ncycle,
            particles=ParticleData.empty(schema.particle_nint, schema.particle_nreal),
            recovery=source.recovery,
            slices=slices,
        )


def _reduce_projections(projections, comm):
    """Sum local projection arrays onto rank zero without a second packed copy."""

    from mpi4py import MPI

    rank = comm.Get_rank()
    field_names = tuple(projections["z"]["whole"])
    all_names = comm.allgather(field_names)
    if any(names != field_names for names in all_names):
        raise RuntimeError(f"MPI ranks reconstructed different projection fields: {all_names}")
    reduced = (
        {axis: {phase: {} for phase in PHASES} for axis in ("z", "y")}
        if rank == 0
        else {}
    )
    for axis in ("z", "y"):
        for phase in PHASES:
            for name in field_names:
                local = np.ascontiguousarray(projections[axis][phase][name])
                output = np.empty_like(local) if rank == 0 else None
                comm.Reduce(local, output, op=MPI.SUM, root=0)
                if rank == 0:
                    reduced[axis][phase][name] = output
    return reduced


def _block_range(count: int, rank: int, size: int) -> tuple[int, int]:
    if size < 1 or not 0 <= rank < size:
        raise ValueError(f"invalid MPI rank/size: rank={rank}, size={size}")
    return count * rank // size, count * (rank + 1) // size


def _slice_tiles(fields, loc, block_size, selected):
    """Copy only central-plane tiles so block-sized 3D arrays can be released."""

    bx, by, bz = block_size
    x0, y0, z0 = loc.lx1 * bx, loc.lx2 * by, loc.lx3 * bz
    tiles = []
    if z0 <= selected["z"] < z0 + bz:
        local_z = selected["z"] - z0
        tiles.append(
            ("z", y0, x0, {name: values[local_z, :, :].copy() for name, values in fields.items()})
        )
    if y0 <= selected["y"] < y0 + by:
        local_y = selected["y"] - y0
        tiles.append(
            ("y", z0, x0, {name: values[:, local_y, :].copy() for name, values in fields.items()})
        )
    return tiles


def _assemble_slice_tiles(tile_groups, field_names, shapes):
    planes = {
        axis: {
            name: np.full(shapes[axis], np.nan, dtype=np.float64) for name in field_names
        }
        for axis in ("z", "y")
    }
    for tiles in tile_groups:
        for axis, row, column, fields in tiles:
            sample = next(iter(fields.values()))
            target = (
                slice(row, row + sample.shape[0]),
                slice(column, column + sample.shape[1]),
            )
            for name, values in fields.items():
                planes[axis][name][target] = values
    return planes


def _accumulate_block(
    projections,
    per_cell,
    masks,
    loc,
    block_size,
    spacing,
    shapes,
) -> None:
    bx, by, bz = block_size
    x0, y0, z0 = loc.lx1 * bx, loc.lx2 * by, loc.lx3 * bz
    targets = {
        "z": (slice(y0, y0 + by), slice(x0, x0 + bx)),
        "y": (slice(z0, z0 + bz), slice(x0, x0 + bx)),
    }
    reductions = {"z": (0, spacing[2]), "y": (1, spacing[1])}
    lengths = {
        "z": shapes["y"][0] * spacing[2],
        "y": shapes["z"][0] * spacing[1],
    }
    for axis, (dimension, cell_width) in reductions.items():
        for phase, mask in masks.items():
            for name, values in per_cell.items():
                output = projections[axis][phase].setdefault(
                    name, np.zeros(shapes[axis], dtype=np.float64)
                )
                factor = cell_width if name.startswith("Sigma") or name == "EM" else (
                    cell_width / lengths[axis]
                )
                output[targets[axis]] += np.sum(values * mask, axis=dimension) * factor


def _read_mu_interpolator(path: Path) -> Callable[[np.ndarray], np.ndarray] | None:
    if not path.is_file():
        return None
    log_t1 = []
    mu = []
    with path.open(newline="") as stream:
        rows = csv.reader(stream)
        next(rows, None)
        for row in rows:
            # Runtime tables include an integer index before the named columns.
            if len(row) >= 3:
                log_t1.append(float(row[-4]))
                mu.append(float(row[-3]))
    x = np.asarray(log_t1)
    y = np.asarray(mu)
    if not len(x):
        return None

    def interpolate(t1: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.interp(np.log10(t1), x, y)

    return interpolate
