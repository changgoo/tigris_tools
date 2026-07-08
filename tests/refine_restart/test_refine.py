import numpy as np

from tigris_tools.refine_restart.refine import (
    coarsen_cell_centered,
    coarsen_face_x1,
    coarsen_face_x2,
    coarsen_face_x3,
    refine_cell_centered,
    refine_face_x1,
    refine_face_x2,
    refine_face_x3,
)


def test_cell_centered_refine_then_coarsen():
    array = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)
    refined = refine_cell_centered(array, 2)
    assert refined.shape == (2, 6, 8, 10)
    assert np.array_equal(coarsen_cell_centered(refined, 2), array)


def test_face_refine_then_coarsen_boundary_faces():
    x1 = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5)
    x2 = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5)
    x3 = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5)

    assert np.allclose(coarsen_face_x1(refine_face_x1(x1, 2), 2), x1)
    assert np.allclose(coarsen_face_x2(refine_face_x2(x2, 2), 2), x2)
    assert np.allclose(coarsen_face_x3(refine_face_x3(x3, 2), 2), x3)


def test_face_refinement_interpolates_distinguished_axis():
    face = np.array([[[0.0, 2.0, 4.0]]])
    refined = refine_face_x1(face, 2)
    assert refined.shape == (2, 2, 5)
    assert np.allclose(refined[0, 0, :], [0.0, 1.0, 2.0, 3.0, 4.0])
