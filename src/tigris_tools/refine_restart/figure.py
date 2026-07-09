from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import layout
from .reader import RestartReader
from .refine import coarsen_cell_centered


def require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("install matplotlib to use --figure") from exc


@dataclass(frozen=True)
class SliceSpec:
    axis: int
    label: str
    index: int


@dataclass(frozen=True)
class ParticlePoint:
    x1: float
    x2: float
    x3: float
    owner: int


class ComparisonFigure:
    def __init__(
        self,
        header: layout.MeshHeader,
        out_header: layout.MeshHeader,
        in_block_size: tuple[int, int, int],
        out_block_size: tuple[int, int, int],
        factor: int,
        slice_spec: str = "x3:mid",
    ) -> None:
        self.header = header
        self.out_header = out_header
        self.in_block_size = in_block_size
        self.out_block_size = out_block_size
        self.factor = factor
        self.slice = parse_slice_spec(slice_spec, _mesh_shape(header))
        self.fields: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.particles: list[ParticlePoint] = []

    def add_block(
        self,
        loc: layout.LogicalLocation,
        payload: dict[str, object],
        refined: dict[str, object],
    ) -> None:
        for name, before, after in _available_fields(
            payload,
            refined,
            _cell_widths(self.header),
            _cell_widths(self.out_header),
        ):
            if name not in self.fields:
                self.fields[name] = (
                    np.full(_plane_shape(_mesh_shape(self.header), self.slice.axis), np.nan),
                    np.full(_plane_shape(_mesh_shape(self.out_header), self.slice.axis), np.nan),
                    np.full(_plane_shape(_mesh_shape(self.header), self.slice.axis), np.nan),
                )
            in_plane, out_plane, delta_plane = self.fields[name]
            _copy_slice(
                in_plane,
                before,
                loc,
                self.in_block_size,
                self.in_block_size,
                self.slice.index,
                self.slice.axis,
                active_factor=1,
            )
            _copy_slice(
                out_plane,
                after,
                loc,
                self.in_block_size,
                tuple(n * self.factor for n in self.in_block_size),
                min(
                    self.slice.index * self.factor + self.factor // 2,
                    _mesh_shape(self.out_header)[self.slice.axis] - 1,
                ),
                self.slice.axis,
                active_factor=self.factor,
                global_factor=self.factor,
            )
            coarsened = coarsen_cell_centered(
                _active(after, self.in_block_size, self.factor)[np.newaxis, ...],
                self.factor,
            )[0]
            _copy_slice(
                delta_plane,
                coarsened - _active(before, self.in_block_size, 1),
                loc,
                self.in_block_size,
                self.in_block_size,
                self.slice.index,
                self.slice.axis,
                active_factor=0,
            )
        self.particles.extend(_particles_on_slice(payload, loc, self.header, self.in_block_size, self.slice))

    def save(self, path: str | Path) -> None:
        if not self.fields:
            raise ValueError("no plottable fields found")
        require_matplotlib()
        import matplotlib.pyplot as plt

        names = list(self.fields)
        fig, axes = plt.subplots(
            len(names),
            3,
            figsize=(12, max(3, 3 * len(names))),
            squeeze=False,
            constrained_layout=True,
        )
        for row, name in enumerate(names):
            before, after, delta = self.fields[name]
            finite = np.concatenate([before[np.isfinite(before)], after[np.isfinite(after)]])
            vmin = float(np.min(finite)) if finite.size else None
            vmax = float(np.max(finite)) if finite.size else None
            panels = (
                ("input", before, _mesh_shape(self.header), self.in_block_size, vmin, vmax),
                ("refined", after, _mesh_shape(self.out_header), self.out_block_size, vmin, vmax),
                ("coarsened delta", delta, _mesh_shape(self.header), self.in_block_size, None, None),
            )
            for col, (title, data, mesh_shape, block_size, lo, hi) in enumerate(panels):
                ax = axes[row][col]
                image = ax.imshow(data, origin="lower", vmin=lo, vmax=hi, aspect="auto")
                ax.set_title(f"{name} {title}")
                ax.set_xlabel(_axis_names(self.slice.axis)[0])
                ax.set_ylabel(_axis_names(self.slice.axis)[1])
                _draw_boundaries(ax, mesh_shape, block_size, self.slice.axis)
                if row == 0:
                    _draw_particles(ax, self.particles, self.header, mesh_shape, self.slice)
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(path, dpi=150)
        plt.close(fig)


def write_restart_comparison_figure(
    input_path: str | Path,
    refined_path: str | Path,
    figure_path: str | Path,
    *,
    slice_spec: str = "x3:mid",
    verbose: int = 0,
) -> None:
    require_matplotlib()
    # Local import avoids a module cycle: convert imports this module for in-flight figures.
    from .convert import _parse_payload, infer_schema

    with RestartReader(input_path, verbose=verbose > 0) as before_reader, RestartReader(
        refined_path,
        verbose=verbose > 0,
    ) as after_reader:
        assert before_reader.index is not None
        assert after_reader.index is not None
        before_schema = infer_schema(before_reader)
        after_schema = infer_schema(after_reader)
        factor = _uniform_refinement_factor(before_reader.index.header, after_reader.index.header)
        figure = ComparisonFigure(
            before_reader.index.header,
            after_reader.index.header,
            before_schema.block_size,
            after_schema.block_size,
            factor,
            slice_spec,
        )
        before_fields, particles = _collect_planes(before_reader, before_schema, figure.slice, _parse_payload)
        after_slice = SliceSpec(
            figure.slice.axis,
            figure.slice.label,
            min(
                figure.slice.index * factor + factor // 2,
                _mesh_shape(after_reader.index.header)[figure.slice.axis] - 1,
            ),
        )
        after_fields, _after_particles = _collect_planes(after_reader, after_schema, after_slice, _parse_payload)
        for name in before_fields:
            if name not in after_fields:
                continue
            before = before_fields[name]
            after = after_fields[name]
            figure.fields[name] = (before, after, _coarsen_plane(after, factor) - before)
        figure.particles = particles
        figure.save(figure_path)


def parse_slice_spec(value: str, mesh_shape: tuple[int, int, int]) -> SliceSpec:
    try:
        axis_text, index_text = value.split(":", 1)
    except ValueError as exc:
        raise ValueError("slice must be AXIS:INDEX, for example x3:mid") from exc
    axes = {"x1": 0, "x2": 1, "x3": 2}
    if axis_text not in axes:
        raise ValueError("slice axis must be x1, x2, or x3")
    axis = axes[axis_text]
    size = mesh_shape[axis]
    index = size // 2 if index_text == "mid" else int(index_text)
    if not 0 <= index < size:
        raise ValueError(f"slice index {index} out of range for {axis_text} size {size}")
    return SliceSpec(axis, axis_text, index)


def loc_axis(loc: layout.LogicalLocation, axis: int) -> int:
    return (loc.lx1, loc.lx2, loc.lx3)[axis]


def _available_fields(
    payload: dict[str, object],
    refined: dict[str, object],
    in_dx: tuple[float, float, float],
    out_dx: tuple[float, float, float],
):
    before = dict(_payload_fields(payload, in_dx))
    after = dict(_payload_fields(refined, out_dx))
    for name, before_value in before.items():
        if name in after:
            yield name, before_value, after[name]


def _payload_fields(payload: dict[str, object], dx: tuple[float, float, float]):
    yield "density", payload["hydro"][0]
    yield "velocity_magnitude", _velocity_magnitude(payload["hydro"])
    if payload.get("bx1") is not None:
        yield "magnetic_magnitude", _magnetic_magnitude(payload)
        yield "divB", _divergence_b(payload, dx)
    if payload.get("cr") is not None:
        yield "cr_energy_density", payload["cr"][0, 0]
    if payload.get("scalars") is not None:
        yield "scalar_0", payload["scalars"][0]


def _velocity_magnitude(hydro: np.ndarray) -> np.ndarray:
    density = hydro[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        v2 = (hydro[1] / density) ** 2 + (hydro[2] / density) ** 2 + (hydro[3] / density) ** 2
    return np.sqrt(v2)


def _magnetic_magnitude(payload: dict[str, object]) -> np.ndarray:
    bx1 = _face_to_cell(payload["bx1"], 2)
    bx2 = _face_to_cell(payload["bx2"], 1)
    bx3 = _face_to_cell(payload["bx3"], 0)
    return np.sqrt(bx1 * bx1 + bx2 * bx2 + bx3 * bx3)


def _divergence_b(payload: dict[str, object], dx: tuple[float, float, float]) -> np.ndarray:
    bx1 = payload["bx1"]
    bx2 = payload["bx2"]
    bx3 = payload["bx3"]
    return np.diff(bx1, axis=2) / dx[0] + np.diff(bx2, axis=1) / dx[1] + np.diff(bx3, axis=0) / dx[2]


def _face_to_cell(array: np.ndarray, axis: int) -> np.ndarray:
    left = [slice(None), slice(None), slice(None)]
    right = [slice(None), slice(None), slice(None)]
    left[axis] = slice(None, -1)
    right[axis] = slice(1, None)
    return 0.5 * (array[tuple(left)] + array[tuple(right)])


def _copy_slice(
    plane: np.ndarray,
    data: np.ndarray,
    loc: layout.LogicalLocation,
    in_block_size: tuple[int, int, int],
    data_block_size: tuple[int, int, int],
    global_slice_index: int,
    slice_axis: int,
    *,
    active_factor: int,
    global_factor: int = 1,
) -> None:
    starts = tuple(loc_axis(loc, axis) * in_block_size[axis] * global_factor for axis in range(3))
    sizes = data_block_size
    if not starts[slice_axis] <= global_slice_index < starts[slice_axis] + sizes[slice_axis]:
        return
    local_slice = global_slice_index - starts[slice_axis]
    if active_factor:
        data = _active(data, in_block_size, active_factor)
    slices = [slice(None), slice(None), slice(None)]
    slices[slice_axis] = local_slice
    block_plane = data[tuple(_to_zyx(slices))]
    plane_slices = _plane_slices(starts, sizes, slice_axis)
    plane[plane_slices] = block_plane


def _collect_planes(reader, schema, slice_spec, parse_payload):
    assert reader.index is not None
    fields: dict[str, np.ndarray] = {}
    particles: list[ParticlePoint] = []
    for block in reader.index.blocks:
        payload = parse_payload(reader.read_block_bytes(block), schema)
        for name, data in _payload_fields(payload, _cell_widths(reader.index.header)):
            if name not in fields:
                fields[name] = np.full(_plane_shape(_mesh_shape(reader.index.header), slice_spec.axis), np.nan)
            _copy_slice(
                fields[name],
                data,
                block.loc,
                schema.block_size,
                schema.block_size,
                slice_spec.index,
                slice_spec.axis,
                active_factor=1,
            )
        particles.extend(_particles_on_slice(payload, block.loc, reader.index.header, schema.block_size, slice_spec))
    return fields, particles


def _coarsen_plane(plane: np.ndarray, factor: int) -> np.ndarray:
    height, width = plane.shape
    if height % factor or width % factor:
        raise ValueError(f"refined plane shape {plane.shape} is not divisible by factor {factor}")
    return plane.reshape(height // factor, factor, width // factor, factor).mean(axis=(1, 3))


def _active(array: np.ndarray, block_size: tuple[int, int, int], factor: int) -> np.ndarray:
    starts = [
        layout.NGHOST * factor if block_size[2] > 1 else 0,
        layout.NGHOST * factor if block_size[1] > 1 else 0,
        layout.NGHOST * factor,
    ]
    lengths = [block_size[2] * factor, block_size[1] * factor, block_size[0] * factor]
    return array[
        starts[0] : starts[0] + lengths[0],
        starts[1] : starts[1] + lengths[1],
        starts[2] : starts[2] + lengths[2],
    ]


def _to_zyx(items):
    return (items[2], items[1], items[0])


def _plane_slices(
    starts: tuple[int, int, int],
    sizes: tuple[int, int, int],
    slice_axis: int,
):
    axes = [axis for axis in range(3) if axis != slice_axis]
    return (
        slice(starts[axes[1]], starts[axes[1]] + sizes[axes[1]]),
        slice(starts[axes[0]], starts[axes[0]] + sizes[axes[0]]),
    )


def _plane_shape(mesh_shape: tuple[int, int, int], slice_axis: int) -> tuple[int, int]:
    axes = [axis for axis in range(3) if axis != slice_axis]
    return mesh_shape[axes[1]], mesh_shape[axes[0]]


def _mesh_shape(header: layout.MeshHeader) -> tuple[int, int, int]:
    ms = header.mesh_size
    return ms.nx1, ms.nx2, ms.nx3


def _cell_widths(header: layout.MeshHeader) -> tuple[float, float, float]:
    ms = header.mesh_size
    return ms.x1len / ms.nx1, ms.x2len / ms.nx2, ms.x3len / ms.nx3


def _uniform_refinement_factor(before: layout.MeshHeader, after: layout.MeshHeader) -> int:
    before_shape = _mesh_shape(before)
    after_shape = _mesh_shape(after)
    ratios = []
    for old, new in zip(before_shape, after_shape):
        if new % old:
            raise ValueError(f"refined mesh shape {after_shape} is not divisible by input shape {before_shape}")
        ratios.append(new // old)
    if len(set(ratios)) != 1:
        raise ValueError(f"comparison figure requires uniform refinement, got ratios {tuple(ratios)}")
    return ratios[0]


def _axis_names(slice_axis: int) -> tuple[str, str]:
    axes = [axis for axis in range(3) if axis != slice_axis]
    names = ("x1", "x2", "x3")
    return names[axes[0]], names[axes[1]]


def _draw_boundaries(
    ax,
    mesh_shape: tuple[int, int, int],
    block_size: tuple[int, int, int],
    slice_axis: int,
) -> None:
    axes = [axis for axis in range(3) if axis != slice_axis]
    width = mesh_shape[axes[0]]
    height = mesh_shape[axes[1]]
    for x in range(block_size[axes[0]], width, block_size[axes[0]]):
        ax.axvline(x - 0.5, color="white", linewidth=0.5, alpha=0.8)
    for y in range(block_size[axes[1]], height, block_size[axes[1]]):
        ax.axhline(y - 0.5, color="white", linewidth=0.5, alpha=0.8)


def _draw_particles(
    ax,
    particles: list[ParticlePoint],
    in_header: layout.MeshHeader,
    mesh_shape: tuple[int, int, int],
    slice_spec: SliceSpec,
) -> None:
    if not particles:
        return
    axes = [axis for axis in range(3) if axis != slice_spec.axis]
    ms = in_header.mesh_size
    mins = (ms.x1min, ms.x2min, ms.x3min)
    lens = (ms.x1len, ms.x2len, ms.x3len)
    coords = [(p.x1, p.x2, p.x3) for p in particles]
    xs = [((p[axes[0]] - mins[axes[0]]) / lens[axes[0]]) * mesh_shape[axes[0]] - 0.5 for p in coords]
    ys = [((p[axes[1]] - mins[axes[1]]) / lens[axes[1]]) * mesh_shape[axes[1]] - 0.5 for p in coords]
    owners = [p.owner for p in particles]
    ax.scatter(xs, ys, s=8, marker=".", c=owners, cmap="tab20", alpha=0.85)


def _particles_on_slice(
    payload: dict[str, object],
    loc: layout.LogicalLocation,
    header: layout.MeshHeader,
    block_size: tuple[int, int, int],
    slice_spec: SliceSpec,
) -> list[ParticlePoint]:
    particles = payload.get("particles")
    if particles is None:
        return []
    npar, _idmax, _ints, reals = particles
    if npar == 0:
        return []
    bounds = _block_bounds(header, loc, block_size)
    axis = slice_spec.axis
    block_start = loc_axis(loc, axis) * block_size[axis]
    if not block_start <= slice_spec.index < block_start + block_size[axis]:
        return []
    dx = (bounds[axis][1] - bounds[axis][0]) / block_size[axis]
    center = bounds[axis][0] + (slice_spec.index - block_start + 0.5) * dx
    coords = (reals[1], reals[2], reals[3])
    mask = (coords[axis] >= center - 0.5 * dx) & (coords[axis] < center + 0.5 * dx)
    owner = _owner_id(loc, header, block_size)
    return [
        ParticlePoint(float(x1), float(x2), float(x3), owner)
        for x1, x2, x3 in zip(coords[0][mask], coords[1][mask], coords[2][mask])
    ]


def _owner_id(
    loc: layout.LogicalLocation,
    header: layout.MeshHeader,
    block_size: tuple[int, int, int],
) -> int:
    ms = header.mesh_size
    nx = ms.nx1 // block_size[0]
    ny = ms.nx2 // block_size[1]
    return loc.lx1 + nx * (loc.lx2 + ny * loc.lx3)


def _block_bounds(
    header: layout.MeshHeader,
    loc: layout.LogicalLocation,
    block_size: tuple[int, int, int],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    ms = header.mesh_size
    mesh_shape = _mesh_shape(header)
    mins = (ms.x1min, ms.x2min, ms.x3min)
    lens = (ms.x1len, ms.x2len, ms.x3len)
    starts = (loc.lx1 * block_size[0], loc.lx2 * block_size[1], loc.lx3 * block_size[2])
    return tuple(
        (
            mins[axis] + lens[axis] * starts[axis] / mesh_shape[axis],
            mins[axis] + lens[axis] * (starts[axis] + block_size[axis]) / mesh_shape[axis],
        )
        for axis in range(3)
    )
