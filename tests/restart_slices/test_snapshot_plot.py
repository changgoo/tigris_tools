import numpy as np

from tigris_tools.restart_slices.projection import ProjectionUnits
from tigris_tools.restart_slices.snapshot_plot import (
    SnapshotUnits,
    _grid_rectangles,
    derive_snapshot_fields,
    snapshot_figure_path,
)


def test_derive_snapshot_fields_uses_tigress_cr_units():
    data = {
        "density": np.array([2.0]),
        "pressure": np.array([3.0]),
        "rret": np.array([0.25]),
        "velocity3": np.array([4.0]),
        "cell_centered_B1": np.array([3.0]),
        "cell_centered_B2": np.array([4.0]),
        "cell_centered_B3": np.array([0.0]),
    }
    units = SnapshotUnits(
        projection=ProjectionUnits(0.0, 0.0, 0.0, 10.0),
        velocity_kms=2.0,
        pressure_over_kb=5.0,
        magnetic_microgauss=7.0,
        time_myr=1.0,
        mass_msun=1.0,
    )

    result = derive_snapshot_fields(data, units)

    np.testing.assert_allclose(result["nH"], 2.0)
    np.testing.assert_allclose(result["T"], 1.27 * 15.0)
    np.testing.assert_allclose(result["rret"], 0.25)
    np.testing.assert_allclose(result["pok"], 15.0)
    np.testing.assert_allclose(result["Bmag"], 35.0)
    np.testing.assert_allclose(result["vz"], 8.0)


def test_snapshot_figure_path_matches_plot_snapshot(tmp_path):
    assert snapshot_figure_path(tmp_path, 30) == tmp_path / "snapshot_00030.png"


def test_snapshot_grids_leave_original_label_gutter():
    xy, xz = _grid_rectangles(12.0, 18.5, 6, 7, 1.5)

    assert np.isclose(xz[0] - (xy[0] + xy[2]), 0.05)
