from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from tigris_tools.refine_restart.param_block import read_parameter_block

from .batch import discover_numbered_restarts
from .cache import projection_cache_path, slice_cache_path
from .projection import (
    K_BOLTZMANN_CGS,
    M_H_CGS,
    M_SUN_CGS,
    ParticleData,
    ProjectionUnits,
    _read_mu_interpolator,
    extract_particles,
    gas_temperature,
)


@dataclass(frozen=True)
class SnapshotUnits:
    projection: ProjectionUnits
    velocity_kms: float
    pressure_over_kb: float
    magnetic_microgauss: float
    time_myr: float
    mass_msun: float


@dataclass(frozen=True)
class FieldStyle:
    label: str
    cmap: str
    vmin: float
    vmax: float
    logarithmic: bool


FIELD_STYLES = {
    "Sigma": FieldStyle(r"$\Sigma_{\rm gas}\;[M_\odot\,{\rm pc}^{-2}]$", "pink_r", 1e-2, 1e2, True),
    "nH": FieldStyle(r"$n_{\rm H}\;[{\rm cm}^{-3}]$", "rainforest", 1e-5, 1e2, True),
    "T": FieldStyle(r"$T\;[{\rm K}]$", "RdYlBu_r", 1e1, 1e7, True),
    "rret": FieldStyle(r"$f_{\rm ret}$", "bubblegum", 0.0, 1.0, False),
    "pok": FieldStyle(r"$P\;[k_B\,{\rm cm}^{-3}\,{\rm K}]$", "inferno", 1e1, 1e7, True),
    "Bmag": FieldStyle(r"$|\mathbf{B}|\;[\mu{\rm G}]$", "cividis", 1e-1, 1e2, True),
    "vz": FieldStyle(r"$v_z\;[{\rm km\,s}^{-1}]$", "bwr", -200.0, 200.0, False),
}
FIELDS_XY = ("Sigma", "nH", "T", "rret", "pok", "Bmag")
FIELDS_XZ = ("Sigma", "nH", "T", "rret", "pok", "Bmag", "vz")


def read_snapshot_units(restart_path: str | Path) -> SnapshotUnits:
    with Path(restart_path).open("rb") as stream:
        params = read_parameter_block(stream)
    raw = params.values["units"]
    mass_cgs = float(raw["mass_cgs"])
    length_cgs = float(raw["length_cgs"])
    time_cgs = float(raw["time_cgs"])
    density_cgs = mass_cgs / length_cgs**3
    velocity_cgs = length_cgs / time_cgs
    energy_density_cgs = density_cgs * velocity_cgs**2
    seconds_per_myr = 365.25 * 24.0 * 3600.0 * 1e6
    projection = ProjectionUnits(
        surface_density=density_cgs * length_cgs * (3.0856775814913673e18**2) / M_SUN_CGS,
        mass_flux=0.0,
        energy_flux=0.0,
        temperature_mu=velocity_cgs**2 * M_H_CGS / K_BOLTZMANN_CGS,
    )
    return SnapshotUnits(
        projection=projection,
        velocity_kms=velocity_cgs / 1e5,
        pressure_over_kb=energy_density_cgs / K_BOLTZMANN_CGS,
        magnetic_microgauss=np.sqrt(energy_density_cgs) * np.sqrt(4.0 * np.pi) * 1e6,
        time_myr=time_cgs / seconds_per_myr,
        mass_msun=mass_cgs / M_SUN_CGS,
    )


def derive_snapshot_fields(
    data: Mapping[str, object],
    units: SnapshotUnits,
    mu_from_t1=None,
) -> dict[str, object]:
    magnetic_sq = sum(data[f"cell_centered_B{i}"] ** 2 for i in range(1, 4))
    return {
        "nH": data["density"],
        "T": gas_temperature(data, units.projection, mu_from_t1),
        "rret": data["rret"],
        "pok": data["pressure"] * units.pressure_over_kb,
        "Bmag": np.sqrt(magnetic_sq) * units.magnetic_microgauss,
        "vz": data["velocity3"] * units.velocity_kms,
    }


def snapshot_figure_path(figdir: str | Path, num: int) -> Path:
    return Path(figdir) / f"snapshot_{num:05d}.png"


def write_snapshot_plot(
    y_slice: str | Path,
    z_slice: str | Path,
    y_projection: str | Path,
    z_projection: str | Path,
    restart_path: str | Path,
    output_path: str | Path,
    *,
    particles: ParticleData | None = None,
    norm_factor: float = 5.0,
    agemax: float = 40.0,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import xarray as xr
        from matplotlib.colors import LogNorm, Normalize
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    except ImportError as exc:
        raise RuntimeError(
            "snapshot plotting requires matplotlib and xarray; install tigris-tools[slices]"
        ) from exc

    units = read_snapshot_units(restart_path)
    mu_from_t1 = _read_mu_interpolator(Path(restart_path).parent / "cool_ftn.runtime.csv")
    datasets = []
    try:
        for path in (z_slice, y_slice, z_projection, y_projection):
            with xr.open_dataset(path) as stream:
                datasets.append(stream.load())
        xy, xz, prj_xy, prj_xz = datasets
        derived_xy = derive_snapshot_fields(xy, units, mu_from_t1)
        derived_xz = derive_snapshot_fields(xz, units, mu_from_t1)
        lx = _span(xy["x"])
        lz = _span(xz["z"])
        xwidth = 1.5
        ysize = lz / lx * xwidth
        xsize = ysize / len(FIELDS_XY) * 4.0 + len(FIELDS_XZ) * xwidth
        figure = plt.figure(figsize=(xsize, ysize))
        grid = figure.add_gridspec(
            3,
            2 + len(FIELDS_XZ),
            width_ratios=[ysize / 3.0, ysize / 3.0, *([xwidth] * len(FIELDS_XZ))],
            wspace=0.08,
            hspace=0.08,
            left=0.02,
            right=0.99,
            bottom=0.05,
            top=0.97,
        )
        axes_xy = [figure.add_subplot(grid[i % 3, i // 3]) for i in range(len(FIELDS_XY))]
        axes_xz = [figure.add_subplot(grid[:, 2 + i]) for i in range(len(FIELDS_XZ))]
        for index, (axis, field) in enumerate(zip(axes_xy, FIELDS_XY)):
            data = (
                prj_xy[field].sel(phase="whole") if field in prj_xy else derived_xy[field]
            )
            _draw_field(axis, data, field, plt, LogNorm, Normalize)
            if particles is not None and field in {"Sigma", "nH"}:
                _scatter_particles(axis, particles, "z", units, norm_factor, agemax, plt)
            axis.set(xlim=_bounds(xy["x"]), ylim=_bounds(xy["y"]))
            if index == 2:
                axis.set(xlabel="x [pc]", ylabel="y [pc]")
            else:
                axis.axis("off")
        for index, (axis, field) in enumerate(zip(axes_xz, FIELDS_XZ)):
            data = (
                prj_xz[field].sel(phase="whole") if field in prj_xz else derived_xz[field]
            )
            image = _draw_field(axis, data, field, plt, LogNorm, Normalize)
            color_axis = inset_axes(axis, width="80%", height="2%", loc="upper center")
            figure.colorbar(image, cax=color_axis, orientation="horizontal", location="top").set_label(
                FIELD_STYLES[field].label
            )
            if particles is not None and field in {"Sigma", "nH"}:
                _scatter_particles(axis, particles, "y", units, norm_factor, agemax, plt)
            axis.set_aspect("equal", adjustable="box")
            axis.set(xlim=_bounds(xz["x"]), ylim=_bounds(xz["z"]))
            if index == 0:
                axis.set(xlabel="x [pc]", ylabel="z [pc]")
            else:
                axis.axis("off")
        time = float(xy.attrs["time"]) * units.time_myr
        axes_xy[0].annotate(
            f"Model: {Path(restart_path).parent.name}  time={time:6.1f} Myr",
            (0.0, 1.0),
            xycoords="axes fraction",
            va="bottom",
            ha="left",
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        figure.savefig(temporary, format="png", dpi=200, bbox_inches="tight")
        plt.close(figure)
        temporary.replace(output)
        return output
    finally:
        for dataset in datasets:
            dataset.close()


def _draw_field(axis, data, field, plt, log_norm, linear_norm):
    style = FIELD_STYLES[field]
    cmap = _resolve_cmap(style.cmap, plt)
    norm = log_norm(style.vmin, style.vmax) if style.logarithmic else linear_norm(
        style.vmin, style.vmax
    )
    vertical = "z" if "z" in data.dims else "y"
    image = axis.pcolormesh(data["x"], data[vertical], data, cmap=cmap, norm=norm, shading="auto")
    axis.set_aspect("equal", adjustable="box")
    return image


def _resolve_cmap(name: str, plt):
    if name == "rainforest":
        try:
            import colormaps as scientific_colormaps

            return scientific_colormaps.rainforest
        except ImportError:
            return plt.get_cmap("Spectral_r")
    if name == "bubblegum":
        try:
            import cmasher

            return cmasher.bubblegum
        except ImportError:
            return plt.get_cmap("magma")
    return plt.get_cmap(name)


def _scatter_particles(axis, particles, sightline, units, norm_factor, agemax, plt) -> None:
    if not len(particles.reals) or particles.reals.shape[1] < 9:
        return
    real = particles.reals
    integer = particles.integers
    x = real[:, 1]
    y = real[:, 2] if sightline == "z" else real[:, 3]
    mass = real[:, 0] * units.mass_msun
    age = real[:, 8] * units.time_myr
    runaways = mass == 0.0
    clusters = ~runaways
    if np.any(runaways):
        source = runaways & (integer[:, 0] < 0) if integer.shape[1] else np.zeros_like(runaways)
        ordinary = runaways & ~source
        axis.scatter(x[ordinary], y[ordinary], color="k", s=10.0 / norm_factor)
        axis.scatter(x[source], y[source], color="r", marker="*", s=10.0 / norm_factor)
    young = clusters & (age < agemax)
    old = clusters & (age >= agemax) & (age < 40.0)
    axis.scatter(
        x[young],
        y[young],
        s=np.sqrt(mass[young]) / norm_factor,
        c=age[young],
        vmin=0.0,
        vmax=agemax,
        cmap=plt.cm.cool_r,
    )
    axis.scatter(x[old], y[old], s=np.sqrt(mass[old]) / norm_factor, c="grey")


def _span(coordinate) -> float:
    values = np.asarray(coordinate)
    return float(values[-1] - values[0] + np.median(np.diff(values)))


def _bounds(coordinate) -> tuple[float, float]:
    values = np.asarray(coordinate)
    half = 0.5 * float(np.median(np.diff(values)))
    return float(values[0] - half), float(values[-1] + half)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tigris-plot-snapshots-all",
        description="Render standalone plot_snapshot-style figures from cached slices/projections",
    )
    parser.add_argument("restart_dir", type=Path)
    parser.add_argument("--savdir", type=Path, required=True)
    parser.add_argument("--figdir", type=Path)
    parser.add_argument("--prefix")
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    figdir = args.figdir or args.savdir / "snapshot"
    restarts = discover_numbered_restarts(
        args.restart_dir, prefix=args.prefix, start=args.start, stop=args.stop
    )
    completed = skipped = 0
    failures = []
    for position, item in enumerate(restarts, 1):
        output = snapshot_figure_path(figdir, item.num)
        caches = {
            "ys": slice_cache_path(args.savdir, "y", item.num),
            "zs": slice_cache_path(args.savdir, "z", item.num),
            "yp": projection_cache_path(args.savdir, "y", item.num),
            "zp": projection_cache_path(args.savdir, "z", item.num),
        }
        print(f"[{position}/{len(restarts)}] {item.path.name} -> {output.name}", flush=True)
        missing = [str(path) for path in caches.values() if not path.is_file()]
        if missing:
            failures.append({"restart": str(item.path), "error": f"missing caches: {missing}"})
            continue
        newest = max(path.stat().st_mtime for path in caches.values())
        if not args.overwrite and output.is_file() and output.stat().st_mtime > newest:
            skipped += 1
            continue
        try:
            write_snapshot_plot(
                caches["ys"],
                caches["zs"],
                caches["yp"],
                caches["zp"],
                item.path,
                output,
                particles=extract_particles(item.path),
            )
            completed += 1
        except Exception as exc:
            failures.append({"restart": str(item.path), "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps({"found": len(restarts), "completed": completed, "skipped_fresh": skipped, "figdir": str(figdir), "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
