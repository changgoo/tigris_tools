from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from tigris_tools.refine_restart.param_block import read_parameter_block

from .batch import discover_numbered_restarts
from .cache import slice_cache_path
from .cr import K_BOLTZMANN_CGS


@dataclass(frozen=True)
class PlotUnits:
    velocity_kms: float
    cr_flux_velocity_kms: float
    pressure_over_kb: float
    time_myr: float
    sigma_factor: float


@dataclass(frozen=True)
class FieldStyle:
    label: str
    unit: str
    cmap: str
    vmin: float
    vmax: float


FIELD_STYLES = {
    "sigma_para": FieldStyle(
        r"$\sigma_\parallel$", r"$[{\rm cm^{-2}\,s}]$", "viridis", 6.0e-30, 1.5e-27
    ),
    "vmag": FieldStyle(
        r"$|\mathbf{v}|$", r"$[{\rm km\,s^{-1}}]$", "lapaz", 1.0, 500.0
    ),
    "VAi_mag": FieldStyle(
        r"$|v_{\rm A,i}|$", r"$[{\rm km\,s^{-1}}]$", "lapaz", 1.0, 500.0
    ),
    "Vcr_mag": FieldStyle(
        r"$|v_{\rm eff}|$", r"$[{\rm km\,s^{-1}}]$", "lapaz", 1.0, 500.0
    ),
    "pok_cr": FieldStyle(
        r"$P_{\rm cr}$", r"$[k_B\,{\rm cm^{-3}\,K}]$", "plasma", 5.0e1, 5.0e4
    ),
    "pok_trbz": FieldStyle(
        r"$P_{\rm turb,z}$", r"$[k_B\,{\rm cm^{-3}\,K}]$", "plasma", 5.0e1, 5.0e4
    ),
    "pok": FieldStyle(
        r"$P$", r"$[k_B\,{\rm cm^{-3}\,K}]$", "plasma", 5.0e1, 5.0e4
    ),
    "pok_mag": FieldStyle(
        r"$P_{\rm mag}$", r"$[k_B\,{\rm cm^{-3}\,K}]$", "plasma", 5.0e1, 5.0e4
    ),
}

DEFAULT_FIELDS = tuple(FIELD_STYLES)
VECTOR_PREFIXES = {
    "vmag": "velocity",
    "VAi_mag": "0-Vs",
    "Vcr_mag": "0-Fc",
    "pok_mag": "cell_centered_B",
}


def read_plot_units(restart_path: str | Path) -> PlotUnits:
    with Path(restart_path).open("rb") as stream:
        params = read_parameter_block(stream)
    units = params.values.get("units", {})
    mass_cgs = float(units["mass_cgs"])
    length_cgs = float(units["length_cgs"])
    time_cgs = float(units["time_cgs"])
    density_cgs = mass_cgs / length_cgs**3
    velocity_cgs = length_cgs / time_cgs
    energy_density_cgs = density_cgs * velocity_cgs**2
    seconds_per_myr = 365.25 * 24.0 * 3600.0 * 1.0e6
    vmax_cgs = float(params.values["cr"]["vmax"])
    vmax_code = vmax_cgs / velocity_cgs
    diffusivity_cgs = length_cgs**2 / time_cgs
    return PlotUnits(
        velocity_kms=velocity_cgs / 1.0e5,
        cr_flux_velocity_kms=vmax_code * velocity_cgs / 1.0e5,
        pressure_over_kb=energy_density_cgs / K_BOLTZMANN_CGS,
        time_myr=time_cgs / seconds_per_myr,
        sigma_factor=1.0 / (vmax_code * diffusivity_cgs),
    )


def derive_summary_fields(
    data: Mapping[str, object],
    units: PlotUnits,
) -> dict[str, object]:
    velocity_sq = sum(data[f"velocity{component}"] ** 2 for component in range(1, 4))
    streaming_sq = sum(data[f"0-Vs{component}"] ** 2 for component in range(1, 4))
    flux_sq = sum(data[f"0-Fc{component}"] ** 2 for component in range(1, 4))
    magnetic_sq = sum(
        data[f"cell_centered_B{component}"] ** 2 for component in range(1, 4)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        vcr_magnitude = (
            np.sqrt(flux_sq)
            * units.cr_flux_velocity_kms
            / ((4.0 / 3.0) * data["0-Ec"])
        )
    return {
        "sigma_para": data["0-Sigma_diff1"] * units.sigma_factor,
        "vmag": np.sqrt(velocity_sq) * units.velocity_kms,
        "VAi_mag": np.sqrt(streaming_sq) * units.velocity_kms,
        "Vcr_mag": vcr_magnitude,
        "pok_cr": data["0-Ec"] * (1.0 / 3.0) * units.pressure_over_kb,
        "pok_trbz": (
            data["density"] * data["velocity3"] ** 2 * units.pressure_over_kb
        ),
        "pok": data["pressure"] * units.pressure_over_kb,
        # Match the current pyathena/TIGRESS-CR derived-field implementation.
        "pok_mag": np.sqrt(magnetic_sq) * units.pressure_over_kb,
    }


def write_summary_plot(
    y_cache: str | Path,
    z_cache: str | Path,
    restart_path: str | Path,
    output_path: str | Path,
    *,
    fields: Sequence[str] = DEFAULT_FIELDS,
    kpc: bool = False,
    vectors: bool = True,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import xarray as xr
        from matplotlib.colors import LogNorm
    except ImportError as exc:
        raise RuntimeError(
            "summary plotting requires matplotlib and xarray; install tigris-tools[slices]"
        ) from exc

    unknown = sorted(set(fields) - set(FIELD_STYLES))
    if unknown:
        raise ValueError(f"unsupported summary fields: {unknown}")
    units = read_plot_units(restart_path)
    with xr.open_dataset(y_cache) as stream:
        xz = stream.load()
    with xr.open_dataset(z_cache) as stream:
        xy = stream.load()
    try:
        xz_fields = derive_summary_fields(xz, units)
        xy_fields = derive_summary_fields(xy, units)
        scale = 1.0e-3 if kpc else 1.0
        xz_ratio = _coordinate_span(xz["z"]) / _coordinate_span(xz["x"])
        figure, axes = plt.subplots(
            2,
            len(fields),
            sharex="col",
            sharey="row",
            figsize=(1.5 * len(fields), 1.5 * (1.0 + xz_ratio)),
            gridspec_kw={
                "height_ratios": [xz_ratio, 1.0],
                "wspace": 0.0,
                "hspace": 0.0,
            },
            constrained_layout=True,
        )
        axes = np.asarray(axes).reshape(2, len(fields))
        for row, (dataset, derived, vertical) in enumerate(
            ((xz, xz_fields, "z"), (xy, xy_fields, "y"))
        ):
            xcoord = np.asarray(dataset["x"]) * scale
            vcoord = np.asarray(dataset[vertical]) * scale
            for column, field in enumerate(fields):
                panel = axes[row, column]
                style = FIELD_STYLES[field]
                image = panel.pcolormesh(
                    xcoord,
                    vcoord,
                    np.asarray(derived[field]),
                    shading="auto",
                    rasterized=True,
                    cmap=_resolve_cmap(style.cmap),
                    norm=LogNorm(style.vmin, style.vmax),
                )
                panel.set_aspect("equal", adjustable="box")
                panel.axis("off")
                if row == 0:
                    colorbar = figure.colorbar(
                        image,
                        ax=panel,
                        orientation="horizontal",
                        location="top",
                        pad=0.01,
                        shrink=0.8,
                        aspect=10,
                    )
                    colorbar.set_label(f"{style.label}\n{style.unit}", size="large")
                    panel.axhline(0.0, color="k", linestyle=":", linewidth=0.7)
                if vectors and field in VECTOR_PREFIXES:
                    _add_streamlines(
                        panel,
                        dataset,
                        xcoord,
                        vcoord,
                        vertical,
                        VECTOR_PREFIXES[field],
                        scale,
                        silver=field == "pok_mag",
                        density_factor=0.5 if field == "VAi_mag" else 1.0,
                    )
        time_code = float(xy.attrs["time"])
        axes[0, 0].annotate(
            f"t={time_code * units.time_myr:.2f} Myr",
            (0.1, 0.99),
            ha="left",
            va="top",
            xycoords="axes fraction",
            fontsize="large",
            bbox={"boxstyle": "round,pad=0.2", "fc": "w", "ec": "k", "lw": 1},
        )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        try:
            figure.savefig(
                temporary,
                format=output.suffix.lstrip("."),
                dpi=200,
                bbox_inches="tight",
            )
            temporary.replace(output)
        finally:
            plt.close(figure)
            temporary.unlink(missing_ok=True)
        return output
    finally:
        xz.close()
        xy.close()


def _coordinate_span(coordinate) -> float:
    values = np.asarray(coordinate)
    if values.size < 2:
        return 1.0
    spacing = float(np.median(np.diff(values)))
    return float(values[-1] - values[0] + spacing)


def _resolve_cmap(name: str):
    if name != "lapaz":
        return name
    try:
        import colormaps as scientific_colormaps
    except ImportError:
        return "viridis"
    return scientific_colormaps.lapaz


def _add_streamlines(
    panel,
    dataset,
    xcoord,
    vertical_coord,
    vertical: str,
    prefix: str,
    scale: float,
    *,
    silver: bool,
    density_factor: float,
) -> None:
    vertical_component = 3 if vertical == "z" else 2
    horizontal = np.asarray(dataset[f"{prefix}1"])
    vertical_values = np.asarray(dataset[f"{prefix}{vertical_component}"])
    if not np.isfinite(horizontal).all() or not np.isfinite(vertical_values).all():
        return
    density = (
        (1.0 * density_factor, 3.0 * density_factor)
        if vertical == "z"
        else 0.7 * density_factor
    )
    panel.streamplot(
        xcoord,
        vertical_coord,
        horizontal * scale,
        vertical_values * scale,
        color="silver" if silver else "k",
        density=density,
        linewidth=0.5,
        arrowsize=0.7,
    )


def figure_path(figdir: str | Path, basename: str, num: int) -> Path:
    return Path(figdir) / f"{basename}_{num:04d}.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tigris-plot-slices-all",
        description="Render plot_slices_cr-compatible summaries from standalone slice caches",
    )
    parser.add_argument("restart_dir", type=Path)
    parser.add_argument("--savdir", type=Path, required=True, help="directory containing allslc.*")
    parser.add_argument("--figdir", type=Path, help="default: SAVDIR/cr_slices")
    parser.add_argument("--prefix", default="TIGRESS", help="numbered restart prefix")
    parser.add_argument("--figure-prefix", help="default: restart directory basename")
    parser.add_argument("--start", type=int)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-vectors", action="store_true")
    parser.add_argument("--kpc", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start is not None and args.stop is not None and args.start > args.stop:
        raise SystemExit("tigris-plot-slices-all: --start must not exceed --stop")
    restart_dir = args.restart_dir.resolve()
    savdir = args.savdir.resolve()
    figdir = args.figdir.resolve() if args.figdir else savdir / "cr_slices"
    basename = args.figure_prefix or restart_dir.name
    restarts = discover_numbered_restarts(
        restart_dir,
        prefix=args.prefix,
        start=args.start,
        stop=args.stop,
    )
    failures: list[dict[str, object]] = []
    completed = 0
    skipped = 0
    for position, restart in enumerate(restarts, start=1):
        y_cache = slice_cache_path(savdir, "y", restart.num)
        z_cache = slice_cache_path(savdir, "z", restart.num)
        output = figure_path(figdir, basename, restart.num)
        print(f"[{position}/{len(restarts)}] {restart.path.name} -> {output.name}", flush=True)
        missing = [str(path) for path in (y_cache, z_cache) if not path.is_file()]
        if missing:
            failures.append({"num": restart.num, "missing_caches": missing})
            print(f"  MISSING: {', '.join(missing)}", flush=True)
            continue
        newest_input = max(y_cache.stat().st_mtime, z_cache.stat().st_mtime)
        if not args.overwrite and output.is_file() and output.stat().st_mtime > newest_input:
            skipped += 1
            print("  fresh", flush=True)
            continue
        try:
            write_summary_plot(
                y_cache,
                z_cache,
                restart.path,
                output,
                kpc=args.kpc,
                vectors=not args.no_vectors,
            )
            completed += 1
            print(f"  wrote {output}", flush=True)
        except Exception as exc:
            failures.append({"num": restart.num, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
    print(
        json.dumps(
            {
                "found": len(restarts),
                "completed": completed,
                "skipped_fresh": skipped,
                "figdir": str(figdir),
                "failures": failures,
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
