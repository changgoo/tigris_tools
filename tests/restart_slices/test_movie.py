from tigris_tools.restart_slices.movie import ffmpeg_command, find_ffmpeg, main, movie_specs


def test_movie_specs_follow_plot_slices_naming(tmp_path):
    run = tmp_path / "model"

    slices, snapshot = movie_specs(run)

    assert slices.frame_glob == str(run / "cr_slices/model_*.png")
    assert slices.output == run / "movies/model_cr_slices.mp4"
    assert snapshot.frame_glob == str(run / "snapshot/snapshot_*.png")
    assert snapshot.output == run / "movies/model_snapshot.mp4"


def test_ffmpeg_command_matches_pyathena_encoding(tmp_path):
    slices, _ = movie_specs(tmp_path / "model")

    command = ffmpeg_command("/usr/bin/ffmpeg", slices, fps_in=3, fps_out=12)

    assert command[:5] == ["/usr/bin/ffmpeg", "-y", "-r", "3", "-f"]
    assert command[command.index("-i") + 1] == slices.frame_glob
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-vcodec") + 1] == "libx264"
    assert command[command.index("-r", 4) + 1] == "12"
    assert command[-1] == str(slices.output)


def test_find_ffmpeg_honors_an_explicit_missing_command():
    assert find_ffmpeg("definitely-not-an-ffmpeg-command") is None


def test_movie_dry_run_discovers_both_frame_sets(tmp_path, capsys):
    run = tmp_path / "model"
    (run / "cr_slices").mkdir(parents=True)
    (run / "snapshot").mkdir()
    (run / "cr_slices/model_0001.png").write_bytes(b"frame")
    (run / "snapshot/snapshot_00001.png").write_bytes(b"frame")

    assert main([str(run), "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert "model_cr_slices.mp4" in output
    assert "model_snapshot.mp4" in output
