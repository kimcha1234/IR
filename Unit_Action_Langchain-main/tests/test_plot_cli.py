import json
from pathlib import Path

from robot_drawing_planner import plot_cli


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_plot_cli_plots_square_example_to_temp_png(tmp_path, capsys):
    out = tmp_path / "square_plan.png"

    result = plot_cli.main(["examples/square_plan.json", "--out", str(out)])
    captured = capsys.readouterr()

    assert result == 0
    assert captured.err == ""
    assert captured.out == f"Saved plan plot to {out}\n"
    assert out.exists()
    assert out.read_bytes()[:8] == PNG_SIGNATURE


def test_plot_cli_omitted_out_creates_default_png_path(tmp_path, capsys):
    plan_path = tmp_path / "circle_plan.json"
    plan_path.write_text(
        json.dumps(json.loads(Path("examples/circle_plan.json").read_text(encoding="utf-8"))),
        encoding="utf-8",
    )

    result = plot_cli.main([str(plan_path)])
    captured = capsys.readouterr()

    expected = plan_path.with_suffix(".png")
    assert result == 0
    assert captured.out == f"Saved plan plot to {expected}\n"
    assert expected.exists()
    assert expected.read_bytes()[:8] == PNG_SIGNATURE


def test_plot_cli_exits_nonzero_for_missing_file(capsys):
    result = plot_cli.main(["does/not/exist.json"])
    captured = capsys.readouterr()

    assert result != 0
    assert "Plan JSON file not found" in captured.err
    assert captured.out == ""


def test_plot_cli_does_not_require_openai_api_key(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = tmp_path / "square_no_key.png"

    result = plot_cli.main(["examples/square_plan.json", "--out", str(out)])
    captured = capsys.readouterr()

    assert result == 0
    assert "OPENAI_API_KEY" not in captured.err
    assert out.exists()


def test_plot_cli_supports_show_pen_up(tmp_path, capsys):
    out = tmp_path / "letter_with_pen_up.png"

    result = plot_cli.main(
        [
            "examples/letter_A_plan.json",
            "--out",
            str(out),
            "--show-pen-up",
            "--no-labels",
            "--title",
            "Letter A",
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert captured.err == ""
    assert out.exists()


def test_plot_cli_output_png_has_correct_signature(tmp_path, capsys):
    out = tmp_path / "circle_plan.png"

    result = plot_cli.main(
        ["examples/circle_plan.json", "--out", str(out), "--arc-samples", "32", "--dpi", "120"]
    )
    capsys.readouterr()

    assert result == 0
    assert out.read_bytes()[:8] == PNG_SIGNATURE
