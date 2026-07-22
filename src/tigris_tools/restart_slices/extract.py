from __future__ import annotations

import os
import re
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tigris_tools.refine_restart import layout
from tigris_tools.refine_restart.convert import RestartSchema, _parse_payload, infer_schema
from tigris_tools.refine_restart.reader import RestartFormatError, RestartIndex, RestartReader

from .cr import CRReconstructor
from .recovery import RecoveryReport, recover_rank_shifted_index

SCALAR_NAMES = {
    "IDE": "eint",
    "IDS": "entropy",
    "IDZ": "metal",
    "ISN": "SN",
    "IRT": "ret",
    "ICR": "CR",
    "ISCR": "SCR",
    "IHI": "HI",
    "IH2": "H2",
    "IEL": "EL",
}

SCALAR_OUTPUT_NAMES = {"rHI": "xHI", "rH2": "xH2", "rEL": "xe"}


@dataclass(frozen=True)
class SliceResult:
    planes: dict[str, dict[str, np.ndarray]]
    coordinates: dict[str, np.ndarray]
    time: float
    cycle: int
    selected_indices: dict[str, int]
    selected_coordinates: dict[str, float]
    recovery: RecoveryReport | None


class RestartSource(AbstractContextManager["RestartSource"]):
    """Read payloads through either the ordinary or recovered restart index."""

    def __init__(
        self,
        path: str | Path,
        *,
        particle_path: str | Path | None = None,
        verbose: bool = False,
    ) -> None:
        self.path = Path(path)
        self.particle_path = Path(particle_path) if particle_path is not None else None
        self.verbose = verbose
        self.index: RestartIndex | None = None
        self.schema: RestartSchema | None = None
        self.recovery: RecoveryReport | None = None
        self._reader: RestartReader | None = None
        self._descriptor: int | None = None

    def __enter__(self) -> "RestartSource":
        reader = RestartReader(self.path, verbose=self.verbose)
        try:
            reader.open()
        except RestartFormatError:
            reader.close()
            if self.particle_path is None:
                raise RestartFormatError(
                    "restart ID table is malformed; pass the same-cycle particle dump "
                    "with --particle to recover verified payload offsets"
                ) from None
            self.recovery = recover_rank_shifted_index(self.path, self.particle_path)
            self.index = self.recovery.index
            self._descriptor = os.open(self.path, os.O_RDONLY)
        else:
            self._reader = reader
            self.index = reader.index
        self.schema = infer_schema(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._reader is not None:
            self._reader.close()
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None

    def read_block_bytes(self, block: layout.InputBlockDesc) -> bytes:
        if self._reader is not None:
            return self._reader.read_block_bytes(block)
        if self._descriptor is None:
            raise RuntimeError("restart source is not open")
        data = os.pread(self._descriptor, block.byte_size, block.file_offset)
        if len(data) != block.byte_size:
            raise EOFError(
                f"truncated block payload at offset {block.file_offset}: "
                f"expected {block.byte_size}, got {len(data)}"
            )
        return data


def extract_central_slices(
    restart_path: str | Path,
    *,
    particle_path: str | Path | None = None,
    verbose: bool = False,
) -> SliceResult:
    """Stream the meshblocks intersecting the nearest-to-zero y and z planes."""

    with RestartSource(restart_path, particle_path=particle_path, verbose=verbose) as source:
        assert source.index is not None and source.schema is not None
        index = source.index
        schema = source.schema
        _require_uniform_cartesian(index, schema)

        coords = _cell_coordinates(index)
        selected = {axis: _nearest_index(coords[axis], 0.0) for axis in ("y", "z")}
        selected_coords = {axis: float(coords[axis][idx]) for axis, idx in selected.items()}
        shapes = {
            "z": (len(coords["y"]), len(coords["x"])),
            "y": (len(coords["z"]), len(coords["x"])),
        }
        names = _field_names(index, schema)
        planes = {
            axis: {name: np.full(shapes[axis], np.nan, dtype=np.float64) for name in names}
            for axis in ("z", "y")
        }

        block_by_loc = {(b.loc.lx1, b.loc.lx2, b.loc.lx3): b for b in index.blocks}
        bx, by, bz = schema.block_size
        wanted_y = selected["y"] // by
        wanted_z = selected["z"] // bz
        selected_blocks = [
            block for loc, block in block_by_loc.items() if loc[1] == wanted_y or loc[2] == wanted_z
        ]
        selected_blocks.sort(key=lambda block: block.file_offset)
        cr_reconstructor = (
            CRReconstructor(index.params, Path(restart_path).parent) if schema.cr_enabled else None
        )
        mesh = index.header.mesh_size
        spacing = (mesh.x1len / mesh.nx1, mesh.x2len / mesh.nx2, mesh.x3len / mesh.nx3)

        for number, block in enumerate(selected_blocks, start=1):
            payload = _parse_payload(source.read_block_bytes(block), schema)
            fields = _reconstruct_block(payload, index, schema, cr_reconstructor, spacing)
            _copy_intersections(planes, fields, block.loc, schema.block_size, selected)
            if verbose and (number % 32 == 0 or number == len(selected_blocks)):
                print(f"restart_slices: blocks={number}/{len(selected_blocks)}")

        _require_complete(planes)
        return SliceResult(
            planes=planes,
            coordinates=coords,
            time=index.header.time,
            cycle=index.header.ncycle,
            selected_indices=selected,
            selected_coordinates=selected_coords,
            recovery=source.recovery,
        )


def infer_output_number(path: str | Path) -> int:
    match = re.search(r"\.(\d+)\.rst$", Path(path).name)
    if match is None:
        raise ValueError("cannot infer output number; pass --num explicitly")
    return int(match.group(1))


def validate_slices(result: SliceResult) -> dict[str, dict[str, dict[str, float | int]]]:
    report: dict[str, dict[str, dict[str, float | int]]] = {}
    for axis, fields in result.planes.items():
        report[axis] = {}
        for name, values in fields.items():
            finite = np.isfinite(values)
            report[axis][name] = {
                "size": int(values.size),
                "finite": int(finite.sum()),
                "nonfinite": int((~finite).sum()),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
    return report


def _reconstruct_block(
    payload,
    index: RestartIndex,
    schema: RestartSchema,
    cr_reconstructor: CRReconstructor | None,
    spacing: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    hydro = payload["hydro"]
    density = hydro[0]
    density_floor = index.params.get_float("hydro", "dfloor", np.sqrt(1024 * np.finfo(float).tiny))
    density_primitive = np.maximum(density, density_floor)
    velocity = hydro[1:4] / density_primitive

    fields: dict[str, np.ndarray] = {
        "density": density_primitive,
        "velocity1": velocity[0],
        "velocity2": velocity[1],
        "velocity3": velocity[2],
    }
    magnetic_energy = 0.0
    bcc = None
    if schema.magnetic_fields:
        bcc = np.stack(
            [
                0.5 * (payload["bx1"][..., :-1] + payload["bx1"][..., 1:]),
                0.5 * (payload["bx2"][:, :-1, :] + payload["bx2"][:, 1:, :]),
                0.5 * (payload["bx3"][:-1, ...] + payload["bx3"][1:, ...]),
            ]
        )
        magnetic_energy = 0.5 * np.sum(bcc * bcc, axis=0)
        for component in range(3):
            fields[f"cell_centered_B{component + 1}"] = bcc[component]

    gamma = index.params.get_float("hydro", "gamma")
    if gamma is None:
        raise ValueError("missing <hydro>/gamma")
    kinetic_energy = 0.5 * np.sum(hydro[1:4] * hydro[1:4], axis=0) / density_primitive
    pressure = (gamma - 1.0) * (hydro[4] - kinetic_energy - magnetic_energy)
    pressure_floor = index.params.get_float("hydro", "pfloor", np.sqrt(1024 * np.finfo(float).tiny))
    tsq_floor = index.params.get_float("hydro", "tsqfloor", np.sqrt(1024 * np.finfo(float).tiny))
    fields["pressure"] = np.maximum(
        pressure, np.maximum(pressure_floor, tsq_floor * density_primitive)
    )

    if payload["cr"] is not None:
        for group in range(schema.ncrg):
            prefix = f"{group}-"
            fields[prefix + "Ec"] = payload["cr"][group, 0]
            for component in range(3):
                fields[prefix + f"Fc{component + 1}"] = payload["cr"][group, component + 1]
        if cr_reconstructor is not None:
            if schema.ncrg != 1 or bcc is None:
                raise NotImplementedError(
                    "CR output reconstruction currently requires single-group MHD"
                )
            reconstructed_cr = cr_reconstructor.reconstruct(
                density_primitive, fields["pressure"], bcc, payload["cr"][0, 0], spacing
            )
            fields.update({f"0-{name}": values for name, values in reconstructed_cr.items()})

    if payload["scalars"] is not None:
        scalar_floor = index.params.get_float(
            "hydro", "sfloor", np.sqrt(1024 * np.finfo(float).tiny)
        )
        scalar_names = _scalar_names(index, schema.nscalars)
        primitive = (
            np.maximum(payload["scalars"], scalar_floor * density_primitive) / density_primitive
        )
        for scalar_index, name in enumerate(scalar_names):
            raw_name = f"r{name}"
            fields[SCALAR_OUTPUT_NAMES.get(raw_name, raw_name)] = primitive[scalar_index]
    return fields


def _copy_intersections(planes, fields, loc, block_size, selected) -> None:
    nx, ny, nz = block_size
    g = layout.NGHOST
    x0, y0, z0 = loc.lx1 * nx, loc.lx2 * ny, loc.lx3 * nz
    active_x = slice(g, g + nx)
    if z0 <= selected["z"] < z0 + nz:
        local_z = g + selected["z"] - z0
        target = (slice(y0, y0 + ny), slice(x0, x0 + nx))
        for name, values in fields.items():
            planes["z"][name][target] = values[local_z, g : g + ny, active_x]
    if y0 <= selected["y"] < y0 + ny:
        local_y = g + selected["y"] - y0
        target = (slice(z0, z0 + nz), slice(x0, x0 + nx))
        for name, values in fields.items():
            planes["y"][name][target] = values[g : g + nz, local_y, active_x]


def _field_names(index: RestartIndex, schema: RestartSchema) -> list[str]:
    names = ["density", "pressure", "velocity1", "velocity2", "velocity3"]
    if schema.magnetic_fields:
        names.extend(f"cell_centered_B{component}" for component in range(1, 4))
    if schema.cr_enabled:
        for group in range(schema.ncrg):
            names.extend([f"{group}-Ec", *(f"{group}-Fc{component}" for component in range(1, 4))])
            names.extend(
                [
                    f"{group}-Sigma_diff1",
                    f"{group}-Sigma_adv1",
                    *(f"{group}-Vs{component}" for component in range(1, 4)),
                    *(f"{group}-Vd{component}" for component in range(1, 4)),
                ]
            )
    for name in _scalar_names(index, schema.nscalars):
        raw_name = f"r{name}"
        names.append(SCALAR_OUTPUT_NAMES.get(raw_name, raw_name))
    return names


def _scalar_names(index: RestartIndex, count: int) -> list[str]:
    names = [str(number) for number in range(count)]
    indices = index.params.values.get("scalar_indices", {})
    for key, name in SCALAR_NAMES.items():
        if key in indices:
            scalar_index = int(indices[key])
            if 0 <= scalar_index < count:
                names[scalar_index] = name
    return names


def _cell_coordinates(index: RestartIndex) -> dict[str, np.ndarray]:
    mesh = index.header.mesh_size
    bounds = (
        (mesh.x1min, mesh.x1max, mesh.nx1),
        (mesh.x2min, mesh.x2max, mesh.nx2),
        (mesh.x3min, mesh.x3max, mesh.nx3),
    )
    result = {}
    for axis, (lower, upper, count) in zip(("x", "y", "z"), bounds):
        spacing = (upper - lower) / count
        result[axis] = lower + (np.arange(count, dtype=np.float64) + 0.5) * spacing
    return result


def _nearest_index(values: np.ndarray, target: float) -> int:
    distance = np.abs(values - target)
    # xarray/pandas nearest lookup resolves an exact midpoint toward the larger coordinate.
    return int(np.flatnonzero(distance == distance.min())[-1])


def _require_uniform_cartesian(index: RestartIndex, schema: RestartSchema) -> None:
    mesh = index.header.mesh_size
    if any(ratio != 1.0 for ratio in (mesh.x1rat, mesh.x2rat, mesh.x3rat)):
        raise NotImplementedError("restart slice extraction currently requires a uniform grid")
    if index.params.get("configure", "Coordinate_system", "cartesian") != "cartesian":
        raise NotImplementedError(
            "restart slice extraction currently requires Cartesian coordinates"
        )
    if any(block.loc.level != index.header.root_level for block in index.blocks):
        raise NotImplementedError("restart slice extraction does not yet support AMR")
    expected = (
        (mesh.nx1 // schema.block_size[0])
        * (mesh.nx2 // schema.block_size[1])
        * (mesh.nx3 // schema.block_size[2])
    )
    if expected != len(index.blocks):
        raise ValueError(f"expected {expected} uniform root blocks, found {len(index.blocks)}")


def _require_complete(planes: dict[str, dict[str, np.ndarray]]) -> None:
    failures = []
    for axis, fields in planes.items():
        for name, values in fields.items():
            count = int(np.count_nonzero(~np.isfinite(values)))
            if count:
                failures.append(f"{axis}/{name}: {count} non-finite cells")
    if failures:
        raise ValueError("slice reconstruction is incomplete: " + "; ".join(failures))
