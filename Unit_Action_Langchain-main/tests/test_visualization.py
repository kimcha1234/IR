import math
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from robot_drawing_planner.visualization import (
    extract_draw_segments,
    load_drawing_plan,
    plot_drawing_plan,
    sample_arc,
    sample_line,
    save_plan_plot,
)


def test_sample_line_returns_start_and_end():
    points = sample_line({"x": 0.0, "y": 0.0}, {"x": 0.1, "y": -0.1})

    assert points[0] == (0.0, 0.0)
    assert points[-1] == (0.1, -0.1)


def test_sample_arc_full_circle_returns_closed_points():
    points = sample_arc(
        center={"x": 0.0, "y": 0.0},
        radius_m=0.05,
        start_angle_rad=0.0,
        end_angle_rad=2.0 * math.pi,
        direction="ccw",
        samples=64,
    )

    assert points[0][0] == pytest.approx(points[-1][0])
    assert points[0][1] == pytest.approx(points[-1][1])


def test_extract_draw_segments_for_square_example():
    plan = load_drawing_plan("examples/square_plan.json")

    segments = extract_draw_segments(plan)

    assert len(segments) == 4
    assert {segment["type"] for segment in segments} == {"line"}
    assert all(len(segment["points"]) == 2 for segment in segments)


def test_extract_draw_segments_for_circle_example():
    plan = load_drawing_plan("examples/circle_plan.json")

    segments = extract_draw_segments(plan)

    assert len(segments) == 1
    assert segments[0]["type"] == "arc"
    assert len(segments[0]["points"]) == 96


def test_save_plan_plot_creates_png_file_for_square_plan(tmp_path):
    plan = load_drawing_plan("examples/square_plan.json")
    out = tmp_path / "square_plot.png"

    result = save_plan_plot(plan, out)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_saved_png_file_has_png_signature(tmp_path):
    plan = load_drawing_plan("examples/square_plan.json")
    out = tmp_path / "square_plot.png"

    save_plan_plot(plan, out)

    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_drawing_plan_returns_fig_and_ax():
    plan = load_drawing_plan("examples/square_plan.json")

    fig, ax = plot_drawing_plan(plan)

    try:
        assert fig is not None
        assert ax is not None
        assert ax.get_xlabel() == "board x [m]"
        assert ax.get_ylabel() == "board y [m]"
    finally:
        plt.close(fig)


def test_invalid_json_file_raises_clear_error(tmp_path):
    bad = tmp_path / "bad_plan.json"
    bad.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_drawing_plan(bad)


def test_actions_without_strokes_reconstruct_draw_segments():
    plan = {
        "schema_version": "1.0",
        "source_command": "action-only plan",
        "strokes": [],
        "actions": [
            {
                "name": "draw_line",
                "frame": "board",
                "stroke_id": "line_from_action",
                "params": {
                    "start": {"x": 0.0, "y": 0.0, "z": 0.0, "unit": "m"},
                    "end": {"x": 0.05, "y": 0.0, "z": 0.0, "unit": "m"},
                },
            },
            {
                "name": "draw_arc",
                "frame": "board",
                "stroke_id": "arc_from_action",
                "params": {
                    "center": {"x": 0.0, "y": 0.0, "z": 0.0, "unit": "m"},
                    "radius_m": 0.05,
                    "start_angle_rad": 0.0,
                    "end_angle_rad": math.pi,
                    "direction": "ccw",
                },
            },
        ],
        "diagnostics": {"validation_ok": True},
    }

    segments = extract_draw_segments(plan, arc_samples=12)

    assert [segment["type"] for segment in segments] == ["line", "arc"]
    assert segments[0]["points"] == [(0.0, 0.0), (0.05, 0.0)]
    assert len(segments[1]["points"]) == 12


def test_load_drawing_plan_requires_strokes_or_actions(tmp_path):
    bad = Path(tmp_path) / "not_a_plan.json"
    bad.write_text('{"schema_version": "1.0"}', encoding="utf-8")

    with pytest.raises(ValueError, match="strokes.*actions"):
        load_drawing_plan(bad)
