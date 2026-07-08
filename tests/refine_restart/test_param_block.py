from tigris_tools.refine_restart.param_block import parse_parameter_text


def test_parse_and_patch_parameter_block():
    text = """<mesh>
nx1 = 64
nx2 = 64
nx3 = 512

<meshblock>
nx1 = 32
nx2 = 32
nx3 = 32
<par_end>
"""
    params = parse_parameter_text(text)
    assert params.get_int("mesh", "nx3") == 512

    patched = params.patched(
        {
            ("mesh", "nx1"): 128,
            ("mesh", "nx2"): 128,
            ("mesh", "nx3"): 1024,
            ("meshblock", "nx1"): 64,
        }
    )
    reparsed = parse_parameter_text(patched)
    assert reparsed.get_int("mesh", "nx1") == 128
    assert reparsed.get_int("mesh", "nx2") == 128
    assert reparsed.get_int("mesh", "nx3") == 1024
    assert reparsed.get_int("meshblock", "nx1") == 64
    assert patched.endswith("<par_end>\n")
