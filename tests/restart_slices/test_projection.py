import numpy as np

from tigris_tools.restart_slices.projection import (
    ProjectionUnits,
    _accumulate_block,
    _block_range,
    derive_projection_fields,
)


def test_derive_projection_fields_matches_slc_prj_conventions():
    fields = {
        "density": np.array([[[2.0]]]),
        "pressure": np.array([[[3.0]]]),
        "velocity1": np.array([[[4.0]]]),
        "velocity2": np.array([[[5.0]]]),
        "velocity3": np.array([[[6.0]]]),
        "rmetal": np.array([[[0.25]]]),
        "0-Ec": np.array([[[7.0]]]),
        "0-Fc3": np.array([[[8.0]]]),
        "0-Vd3": np.array([[[9.0]]]),
        "0-Vs3": np.array([[[10.0]]]),
    }
    units = ProjectionUnits(11.0, 12.0, 13.0, 14.0)

    result = derive_projection_fields(fields, units, gamma=5.0 / 3.0)

    np.testing.assert_allclose(result["Sigma"], 22.0)
    np.testing.assert_allclose(result["mflux"], 144.0)
    np.testing.assert_allclose(result["teflux"], 585.0)
    np.testing.assert_allclose(result["keflux"], 2 * 6 * (4**2 + 5**2 + 6**2) / 2 * 13)
    np.testing.assert_allclose(result["mZflux"], 36.0)
    np.testing.assert_allclose(result["creflux"], 104.0)
    np.testing.assert_allclose(result["creflux_diff"], 4 / 3 * 7 * 9 * 13)
    np.testing.assert_allclose(result["creflux_adv"], 4 / 3 * 7 * 6 * 13)
    np.testing.assert_allclose(result["creflux_str"], 4 / 3 * 7 * 10 * 13)


def test_accumulate_block_integrates_surface_fields_and_averages_fluxes():
    projections = {
        axis: {phase: {} for phase in ("whole", "hot", "wc")} for axis in ("z", "y")
    }
    values = np.arange(8, dtype=float).reshape(2, 2, 2) + 1
    per_cell = {"Sigma": values, "mflux": values}
    masks = {"whole": 1.0, "hot": values > 4, "wc": values <= 4}
    loc = type("Location", (), {"lx1": 0, "lx2": 0, "lx3": 0})()

    _accumulate_block(
        projections,
        per_cell,
        masks,
        loc,
        (2, 2, 2),
        (1.0, 2.0, 3.0),
        {"z": (2, 2), "y": (2, 2)},
    )

    np.testing.assert_allclose(projections["z"]["whole"]["Sigma"], values.sum(0) * 3)
    np.testing.assert_allclose(projections["z"]["whole"]["mflux"], values.mean(0))
    np.testing.assert_allclose(projections["y"]["whole"]["Sigma"], values.sum(1) * 2)
    np.testing.assert_allclose(projections["y"]["whole"]["mflux"], values.mean(1))


def test_mpi_block_ranges_are_contiguous_balanced_and_complete():
    ranges = [_block_range(4096, rank, 7) for rank in range(7)]

    assert ranges[0][0] == 0
    assert ranges[-1][1] == 4096
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert max(stop - start for start, stop in ranges) - min(
        stop - start for start, stop in ranges
    ) <= 1
