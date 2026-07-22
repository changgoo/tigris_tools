import numpy as np

from tigris_tools.restart_slices.summary_plot import (
    PlotUnits,
    derive_summary_fields,
    figure_path,
)


def test_derive_summary_fields_matches_tigress_cr_conventions():
    data = {
        "density": np.array([7.0]),
        "pressure": np.array([11.0]),
        "velocity1": np.array([3.0]),
        "velocity2": np.array([4.0]),
        "velocity3": np.array([2.0]),
        "0-Vs1": np.array([0.0]),
        "0-Vs2": np.array([6.0]),
        "0-Vs3": np.array([8.0]),
        "0-Fc1": np.array([3.0]),
        "0-Fc2": np.array([4.0]),
        "0-Fc3": np.array([0.0]),
        "0-Ec": np.array([2.0]),
        "0-Sigma_diff1": np.array([13.0]),
        "cell_centered_B1": np.array([0.0]),
        "cell_centered_B2": np.array([5.0]),
        "cell_centered_B3": np.array([12.0]),
    }
    units = PlotUnits(
        velocity_kms=2.0,
        cr_flux_velocity_kms=6.0,
        pressure_over_kb=3.0,
        time_myr=4.0,
        sigma_factor=5.0,
    )

    fields = derive_summary_fields(data, units)

    np.testing.assert_allclose(fields["sigma_para"], [65.0])
    np.testing.assert_allclose(fields["vmag"], [2.0 * np.sqrt(29.0)])
    np.testing.assert_allclose(fields["VAi_mag"], [20.0])
    np.testing.assert_allclose(fields["Vcr_mag"], [11.25])
    np.testing.assert_allclose(fields["pok_cr"], [2.0])
    np.testing.assert_allclose(fields["pok_trbz"], [84.0])
    np.testing.assert_allclose(fields["pok"], [33.0])
    np.testing.assert_allclose(fields["pok_mag"], [39.0])


def test_figure_path_uses_plot_slices_cr_name(tmp_path):
    assert figure_path(tmp_path, "model", 21) == tmp_path / "model_0021.png"
