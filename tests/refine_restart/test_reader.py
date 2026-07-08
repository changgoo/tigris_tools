from io import StringIO

from tigris_tools.refine_restart.layout import (
    LogicalLocation,
    MeshHeader,
    RegionSize,
    pack_id_record,
    pack_mesh_header,
)
from tigris_tools.refine_restart.reader import read_restart_index


def test_read_restart_index(tmp_path):
    path, header, user_mesh_data, payload0 = _write_tiny_restart(tmp_path)

    index = read_restart_index(path)

    assert index.header == header
    assert index.user_mesh_data == user_mesh_data
    assert len(index.blocks) == 2
    assert index.blocks[0].file_offset == index.payload_start
    assert index.blocks[1].file_offset == index.payload_start + len(payload0)
    assert index.payload_end == path.stat().st_size


def test_verbose_restart_index_logs_summary(tmp_path):
    path, _header, _user_mesh_data, _payload0 = _write_tiny_restart(tmp_path)
    log = StringIO()

    read_restart_index(path, verbose=True, log=log)

    output = log.getvalue()
    assert "mesh nbtotal=2 root_level=1 nx=(4,4,4)" in output
    assert "restart schema" in output
    assert "blocks count=2 payload_bytes=24 min=11 max=13" in output


def test_read_restart_index_infers_legacy_user_mesh_data(tmp_path):
    path, header, user_mesh_data, payload0 = _write_tiny_restart(
        tmp_path,
        include_restart_block=False,
        mesh_nx=(8, 4, 4),
        user_mesh_data=b"u" * 40,
    )

    index = read_restart_index(path)

    assert index.header == header
    assert index.user_mesh_data == user_mesh_data
    assert index.blocks[0].file_offset == index.payload_start
    assert index.blocks[1].file_offset == index.payload_start + len(payload0)


def _write_tiny_restart(
    tmp_path,
    *,
    include_restart_block=True,
    mesh_nx=(4, 4, 4),
    user_mesh_data=b"iiiirrrrrrrr",
):
    params = f"""<mesh>
nx1 = {mesh_nx[0]}
nx2 = {mesh_nx[1]}
nx3 = {mesh_nx[2]}
<meshblock>
nx1 = 4
nx2 = 4
nx3 = 4
""".encode("ascii")
    if include_restart_block:
        params += f"""<restart>
nint_user_mesh_data = 1
nreal_user_mesh_data = 1
int_user_mesh_data_size_0 = 4
real_user_mesh_data_size_0 = {len(user_mesh_data) - 4}
""".encode("ascii")
    params += b"<par_end>\n"
    region = RegionSize.from_bounds(
        x1min=0.0,
        x2min=0.0,
        x3min=0.0,
        x1max=1.0,
        x2max=1.0,
        x3max=1.0,
        nx1=mesh_nx[0],
        nx2=mesh_nx[1],
        nx3=mesh_nx[2],
    )
    header = MeshHeader(nbtotal=2, root_level=1, mesh_size=region, time=2.0, dt=0.25, ncycle=9)
    payload0 = b"a" * 11
    payload1 = b"b" * 13
    path = tmp_path / "tiny.rst"
    path.write_bytes(
        params
        + pack_mesh_header(header)
        + user_mesh_data
        + pack_id_record(LogicalLocation(0, 0, 0, 1), 1.0, len(payload0))
        + pack_id_record(LogicalLocation(1, 0, 0, 1), 2.0, len(payload1))
        + payload0
        + payload1
    )
    return path, header, user_mesh_data, payload0
