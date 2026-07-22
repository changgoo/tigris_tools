from __future__ import annotations

from pathlib import Path

import numpy as np

from .extract import SliceResult


def write_validation_plot(result: SliceResult, path: str | Path) -> Path:
    """Render stored/reconstructed arrays before CR transport reconstruction."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, SymLogNorm
    except ImportError as exc:
        raise RuntimeError("slice validation plotting requires matplotlib") from exc

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        ("density", lambda d: d["density"], True),
        ("pressure", lambda d: d["pressure"], True),
        ("|velocity|", lambda d: _magnitude(d, "velocity"), True),
        ("|B|", lambda d: _magnitude(d, "cell_centered_B"), True),
        ("0-Ec", lambda d: d["0-Ec"], True),
        ("|0-Fc|", lambda d: _magnitude(d, "0-Fc"), True),
        ("rCR", lambda d: d.get("rCR"), False),
        ("reint", lambda d: d.get("reint"), True),
    ]
    fields = [
        (name, function, log)
        for name, function, log in fields
        if function(result.planes["z"]) is not None
    ]
    figure, axes = plt.subplots(
        2, len(fields), figsize=(2.2 * len(fields), 6.0), constrained_layout=True
    )
    for row, axis in enumerate(("y", "z")):
        plane = result.planes[axis]
        horizontal = result.coordinates["x"]
        vertical_name = "z" if axis == "y" else "y"
        vertical = result.coordinates[vertical_name]
        extent = (horizontal[0], horizontal[-1], vertical[0], vertical[-1])
        for column, (name, function, take_log) in enumerate(fields):
            values = function(plane)
            panel = axes[row, column]
            norm = _norm(values, take_log, LogNorm, SymLogNorm)
            image = panel.imshow(
                values, origin="lower", extent=extent, aspect="auto", norm=norm, cmap="viridis"
            )
            figure.colorbar(image, ax=panel, location="top", fraction=0.06, pad=0.02)
            if row == 0:
                panel.set_title(name)
            if column == 0:
                panel.set_ylabel(vertical_name)
            panel.set_xlabel("x")
    figure.suptitle(f"restart cycle {result.cycle}, time={result.time:.9g}")
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def _magnitude(fields: dict[str, np.ndarray], prefix: str) -> np.ndarray | None:
    names = [f"{prefix}{component}" for component in range(1, 4)]
    if not all(name in fields for name in names):
        return None
    return np.sqrt(sum(fields[name] ** 2 for name in names))


def _norm(values, take_log, log_norm, symlog_norm):
    finite = values[np.isfinite(values)]
    if take_log:
        positive = finite[finite > 0]
        if positive.size:
            low, high = np.percentile(positive, (1, 99))
            if high > low > 0:
                return log_norm(vmin=low, vmax=high)
    scale = float(np.percentile(np.abs(finite), 99)) if finite.size else 1.0
    if np.nanmin(finite) < 0 < np.nanmax(finite) and scale > 0:
        return symlog_norm(linthresh=scale * 1.0e-4, vmin=-scale, vmax=scale)
    return None
