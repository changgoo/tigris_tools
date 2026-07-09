import numpy as np

from tigris_tools.refine_restart import layout
from tigris_tools.refine_restart.figure import ComparisonFigure, parse_slice_spec
from tigris_tools.refine_restart.refine import (
    refine_cell_centered,
    refine_face_x1,
    refine_face_x2,
    refine_face_x3,
)


def test_parse_slice_spec_midpoint():
    spec = parse_slice_spec("x3:mid", (8, 10, 12))

    assert spec.axis == 2
    assert spec.index == 6


def test_comparison_figure_collects_default_fields_and_particles():
    region = layout.RegionSize.from_bounds(
        x1min=0.0,
        x2min=0.0,
        x3min=0.0,
        x1max=1.0,
        x2max=1.0,
        x3max=1.0,
        nx1=2,
        nx2=2,
        nx3=2,
    )
    header = layout.MeshHeader(1, 0, region, 0.0, 0.1, 1)
    out_region = layout.RegionSize.from_bounds(
        x1min=0.0,
        x2min=0.0,
        x3min=0.0,
        x1max=1.0,
        x2max=1.0,
        x3max=1.0,
        nx1=4,
        nx2=4,
        nx3=4,
    )
    out_header = layout.MeshHeader(1, 0, out_region, 0.0, 0.1, 1)
    hydro = np.zeros(layout.cell_centered_shape(layout.NHYDRO, (2, 2, 2)))
    bx1 = np.zeros(layout.face_shapes((2, 2, 2))[0])
    bx2 = np.zeros(layout.face_shapes((2, 2, 2))[1])
    bx3 = np.zeros(layout.face_shapes((2, 2, 2))[2])
    cr = np.zeros((1, 4, *layout.ghost_cell_shape((2, 2, 2))))
    scalars = np.zeros((1, *layout.ghost_cell_shape((2, 2, 2))))
    hydro[0, 4:6, 4:6, 4:6] = np.arange(8).reshape(2, 2, 2) + 1.0
    hydro[1, 4:6, 4:6, 4:6] = 2.0 * hydro[0, 4:6, 4:6, 4:6]
    hydro[2, 4:6, 4:6, 4:6] = 3.0 * hydro[0, 4:6, 4:6, 4:6]
    hydro[3, 4:6, 4:6, 4:6] = 4.0 * hydro[0, 4:6, 4:6, 4:6]
    bx1[4:6, 4:6, 4:7] = 1.0
    bx2[4:6, 4:7, 4:6] = 2.0
    bx3[4:7, 4:6, 4:6] = 3.0
    cr[0, 0, 4:6, 4:6, 4:6] = 10.0
    scalars[0, 4:6, 4:6, 4:6] = 2.0
    payload = {
        "hydro": hydro,
        "bx1": bx1,
        "bx2": bx2,
        "bx3": bx3,
        "cr": cr,
        "scalars": scalars,
        "particles": (
            1,
            1,
            np.zeros((2, 1), dtype=np.int32),
            np.array([[0.0], [0.5], [0.5], [0.75]]),
        ),
    }
    refined = {
        "hydro": refine_cell_centered(hydro, 2),
        "bx1": refine_face_x1(bx1, 2),
        "bx2": refine_face_x2(bx2, 2),
        "bx3": refine_face_x3(bx3, 2),
        "cr": refine_cell_centered(cr, 2),
        "scalars": refine_cell_centered(scalars, 2),
    }
    figure = ComparisonFigure(header, out_header, (2, 2, 2), (4, 4, 4), 2, "x3:1")

    figure.add_block(layout.LogicalLocation(0, 0, 0, 0), payload, refined)

    assert set(figure.fields) == {
        "density",
        "velocity_magnitude",
        "magnetic_magnitude",
        "divB",
        "cr_energy_density",
        "scalar_0",
    }
    before, after, delta = figure.fields["density"]
    assert before.shape == (2, 2)
    assert after.shape == (4, 4)
    assert np.allclose(delta, 0.0)
    velocity, _refined_velocity, _velocity_delta = figure.fields["velocity_magnitude"]
    assert np.allclose(velocity, np.sqrt(29.0))
    magnetic, _refined_magnetic, _magnetic_delta = figure.fields["magnetic_magnitude"]
    assert np.allclose(magnetic, np.sqrt(14.0))
    divb, _refined_divb, _divb_delta = figure.fields["divB"]
    assert np.allclose(divb, 0.0)
    assert len(figure.particles) == 1
    assert figure.particles[0].owner == 0
