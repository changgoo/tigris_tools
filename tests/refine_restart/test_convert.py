from io import BytesIO

import numpy as np

from tigris_tools.refine_restart import layout
from tigris_tools.refine_restart.convert import (
    RestartSchema,
    _particle_shape,
    _patched_params,
    _prng_seed,
    _write_prng,
    refine_restart,
)
from tigris_tools.refine_restart.param_block import parse_parameter_text
from tigris_tools.refine_restart.reader import read_restart_index


def test_particle_shape_includes_shear_integer_property():
    params = parse_parameter_text(
        """<mesh>
ix1_bc = shear_periodic
ox1_bc = shear_periodic
<particle1>
type = complex
feedback = true
mass_return = true
fgas = 0.9
<par_end>
"""
    )

    assert _particle_shape(params) == (3, 15)


def test_generated_prng_header_matches_rank_zero_binary_state():
    params = parse_parameter_text(
        """<mesh>
nx1 = 2
nx2 = 1
nx3 = 1
<meshblock>
nx1 = 2
nx2 = 1
nx3 = 1
<random>
nrandom = 3
<random0>
name = feedback
seed = 2
count = 0
<random1>
name = ia_runaway
seed = 3
count = 0
<random2>
name = perturbation
seed = 1
count = 100
<par_end>
"""
    )
    region = layout.RegionSize.from_bounds(
        x1min=0,
        x2min=0,
        x3min=0,
        x1max=1,
        x2max=1,
        x3max=1,
        nx1=2,
        nx2=1,
        nx3=1,
    )
    header = layout.MeshHeader(1, 0, region, 0, 0, 0)
    prng_bytes = 3 * layout.PRNG_RECORD_STRUCT.size
    patched = parse_parameter_text(_patched_params(params, header, (2, 1, 1), prng_bytes))

    for index in range(3):
        seed = patched.get_int(f"random{index}", "seed")
        assert seed == _prng_seed(0, index)
        assert len(str(seed)) == 20
        assert patched.get_int(f"random{index}", "count") == 0

    schema = RestartSchema((2, 1, 1), 0, 1, False, False, 0, 0, 0, prng_bytes)
    stream = BytesIO()
    _write_prng(stream, schema, 1)
    records = np.frombuffer(stream.getvalue(), dtype=layout.UINT64_DTYPE).reshape(3, 2)
    assert records[:, 0].tolist() == [_prng_seed(0, index) for index in range(3)]
    assert records[:, 1].tolist() == [0, 0, 0]


def test_refine_can_change_meshblock_size(tmp_path):
    params = b"""<mesh>
nx1 = 2
nx2 = 2
nx3 = 2
<meshblock>
nx1 = 2
nx2 = 2
nx3 = 2
<configure>
Magnetic_fields = OFF
Cosmic_Ray_Transport = OFF
Number_of_scalars = 0
<par_end>
"""
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
    payload = np.arange(
        np.prod(layout.cell_centered_shape(layout.NHYDRO, (2, 2, 2))),
        dtype=np.float64,
    ).tobytes()
    src = tmp_path / "in.rst"
    dst = tmp_path / "out.rst"
    src.write_bytes(
        params
        + layout.pack_mesh_header(header)
        + layout.pack_id_record(layout.LogicalLocation(0, 0, 0, 0), 1.0, len(payload))
        + payload
    )

    refine_restart(src, dst, factor=2, block_size=(4, 4, 4))
    index = read_restart_index(dst)

    assert index.header.nbtotal == 1
    assert index.header.root_level == 0
    assert (index.header.mesh_size.nx1, index.header.mesh_size.nx2, index.header.mesh_size.nx3) == (
        4,
        4,
        4,
    )
    assert index.params.get_int("meshblock", "nx1") == 4
    assert index.blocks[0].loc == layout.LogicalLocation(0, 0, 0, 0)
