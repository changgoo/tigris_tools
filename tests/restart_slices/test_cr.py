from pathlib import Path

import numpy as np

from tigris_tools.refine_restart.param_block import parse_parameter_text
from tigris_tools.restart_slices.cr import CRReconstructor


def _params():
    return parse_parameter_text(
        """<units>
mass_cgs = 4.91769147364387e31
length_cgs = 3.0856776e18
time_cgs = 30856776000000
mean_mass_per_hydrogen = 2.38871337414e-24
<cr>
vmax = 2e9
vs_flag = 1
self_consistent_flag = 0
valfven_flag = 0
perp_diff_flag = 1
sigma = 1e-25
max_opacity = 1e10
perp_to_par_diff = 10
ecfloor = 1e-18
<cooling>
coolftn_file = cooling.txt
<par_end>
"""
    )


def _write_cooling_table(directory: Path) -> None:
    (directory / "cooling.txt").write_text(
        """# test table
4 3
# bounds
1 9
# logT mu cool heat
1 1.3 0 0
5 1.2 0 0
9 0.6 0 0
"""
    )


def test_constant_opacity_streaming_and_diffusion_are_finite(tmp_path):
    _write_cooling_table(tmp_path)
    reconstruct = CRReconstructor(_params(), tmp_path)
    shape = (8, 8, 8)
    density = np.full(shape, 2.0)
    pressure = np.full(shape, 3.0)
    magnetic = np.zeros((3, *shape))
    magnetic[0] = 4.0
    ecr = np.broadcast_to(np.arange(shape[2], dtype=float), shape).copy() + 10.0

    fields = reconstruct.reconstruct(density, pressure, magnetic, ecr, (1.0, 1.0, 1.0))
    active = (slice(1, -1), slice(1, -1), slice(1, -1))

    assert np.all(fields["Sigma_diff1"] == reconstruct.sigma)
    assert np.all(fields["Vs1"][active] < 0.0)
    assert np.all(fields["Vs2"][active] == 0.0)
    assert np.all(fields["Vs3"][active] == 0.0)
    for values in fields.values():
        assert np.isfinite(values[active]).all()
