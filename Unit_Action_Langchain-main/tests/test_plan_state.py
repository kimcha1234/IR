import math

import pytest

from robot_drawing_planner.config import PlannerConfig
from robot_drawing_planner.plan_state import PlanBuilder
from robot_drawing_planner.schemas import PrimitiveAction


ALLOWED_PRIMITIVES = {
    "move_to_start",
    "align_pen_orientation",
    "pen_down",
    "draw_line",
    "draw_arc",
    "pen_up",
}

FINISH_WITH_PEN_DOWN_MESSAGE = (
    "finish_plan requires pen_state == 'up'; call pen_up before finish_plan."
)


def build_square_plan(config: PlannerConfig | None = None) -> PlanBuilder:
    builder = PlanBuilder.begin_plan(
        "draw a square with unit action tools",
        config=config,
    )
    builder.move_to_start(-0.05, 0.05)
    builder.align_pen_orientation()
    builder.pen_down()
    builder.draw_line_to(0.05, 0.05)
    builder.draw_line_to(0.05, -0.05)
    builder.draw_line_to(-0.05, -0.05)
    builder.draw_line_to(-0.05, 0.05)
    builder.pen_up()
    return builder


def test_valid_square_sequence_creates_eight_primitive_actions():
    builder = build_square_plan()
    summary = builder.finish_plan()

    assert summary["finished"] is True
    assert summary["errors"] == []
    assert len(builder.actions) == 8
    assert [action.name for action in builder.actions] == [
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_line",
        "draw_line",
        "draw_line",
        "draw_line",
        "pen_up",
    ]
    assert len(builder.strokes) == 4


def test_draw_line_to_before_pen_down_produces_error():
    builder = PlanBuilder.begin_plan("bad line")
    builder.move_to_start(0.0, 0.0)
    summary = builder.draw_line_to(0.1, 0.0)

    assert "draw_line_to requires pen_state == 'down'." in summary["failed_calls"]
    assert summary["errors"] == []
    assert len(builder.strokes) == 0
    assert [action.name for action in builder.actions] == ["move_to_start"]


def test_move_to_start_while_pen_is_down_produces_error():
    builder = PlanBuilder.begin_plan("bad move")
    builder.move_to_start(0.0, 0.0)
    builder.pen_down()
    summary = builder.move_to_start(0.1, 0.1)

    assert "move_to_start requires pen_state == 'up'." in summary["failed_calls"]
    assert summary["errors"] == []
    assert [action.name for action in builder.actions] == ["move_to_start", "pen_down"]
    assert builder.current_position.x == 0.0
    assert builder.current_position.y == 0.0


def test_finish_plan_without_strokes_fails():
    builder = PlanBuilder.begin_plan("empty plan")
    summary = builder.finish_plan()

    assert summary["finished"] is False
    assert "finish_plan requires at least one drawable stroke." in summary["failed_calls"]
    assert summary["errors"] == []
    assert builder.actions == []


def test_finish_plan_with_pen_down_fails_without_marking_finished():
    builder = PlanBuilder.begin_plan("unsafe finish")
    builder.move_to_start(0.0, 0.0)
    builder.pen_down()
    builder.draw_line_to(0.05, 0.0)

    summary = builder.finish_plan()

    assert summary["finished"] is False
    assert summary["pen_state"] == "down"
    assert FINISH_WITH_PEN_DOWN_MESSAGE in summary["failed_calls"]
    assert summary["errors"] == []


def test_pen_up_after_drawing_succeeds():
    builder = PlanBuilder.begin_plan("line")
    builder.move_to_start(0.0, 0.0)
    builder.pen_down()
    builder.draw_line_to(0.1, 0.0)
    summary = builder.pen_up()

    assert summary["pen_state"] == "up"
    assert summary["errors"] == []
    assert builder.actions[-1].name == "pen_up"


def test_no_action_name_outside_allowed_robot_primitives():
    builder = build_square_plan()

    assert all(isinstance(action, PrimitiveAction) for action in builder.actions)
    assert {action.name for action in builder.actions}.issubset(ALLOWED_PRIMITIVES)


def test_agentic_square_action_params_include_handoff_3d_metadata():
    config = PlannerConfig(
        hover_height_m=0.04,
        drawing_z_m=0.002,
        default_speed_m_s=0.05,
        pen_down_speed_m_s=0.012,
        pen_up_speed_m_s=0.024,
    )
    builder = build_square_plan(config=config)

    move, _align, pen_down, draw_line, *_middle, pen_up = builder.actions

    assert move.params["target"]["z"] == pytest.approx(config.hover_height_m)
    assert move.params["target"]["unit"] == "m"
    assert move.params["hover_height_m"] == pytest.approx(config.hover_height_m)
    assert move.params["note"] == (
        "free-space move; kinematics module converts board frame to base frame"
    )
    assert pen_down.params["target"]["z"] == pytest.approx(config.drawing_z_m)
    assert pen_down.params["approach_axis"] == "-z"
    assert pen_down.params["speed_m_s"] == pytest.approx(config.pen_down_speed_m_s)
    assert draw_line.params["start"]["z"] == pytest.approx(config.drawing_z_m)
    assert draw_line.params["end"]["z"] == pytest.approx(config.drawing_z_m)
    assert draw_line.params["speed_m_s"] == pytest.approx(config.default_speed_m_s)
    assert pen_up.params["lift_height_m"] == pytest.approx(config.hover_height_m)
    assert pen_up.params["speed_m_s"] == pytest.approx(config.pen_up_speed_m_s)


def test_move_to_start_outside_board_fails_without_action():
    builder = PlanBuilder.begin_plan("bad outside move")

    summary = builder.move_to_start(2.0, 0.0)

    assert summary["number_of_actions"] == 0
    assert builder.actions == []
    assert builder.current_position is None
    assert summary["errors"] == []
    assert any("outside the board bounds" in error for error in summary["failed_calls"])


def test_draw_line_to_outside_board_fails_without_stroke_or_action():
    builder = PlanBuilder.begin_plan("bad outside line")
    builder.move_to_start(0.0, 0.0)
    builder.pen_down()
    action_count_before = len(builder.actions)
    stroke_count_before = len(builder.strokes)

    summary = builder.draw_line_to(2.0, 0.0)

    assert len(builder.actions) == action_count_before
    assert len(builder.strokes) == stroke_count_before
    assert builder.current_position.x == 0.0
    assert builder.current_position.y == 0.0
    assert summary["errors"] == []
    assert any("outside the board bounds" in error for error in summary["failed_calls"])


def test_draw_arc_outside_board_fails_without_stroke_or_action():
    builder = PlanBuilder.begin_plan("bad outside arc")
    builder.move_to_start(0.0, 0.0)
    builder.pen_down()
    action_count_before = len(builder.actions)
    stroke_count_before = len(builder.strokes)

    summary = builder.draw_arc(
        center_x=0.24,
        center_y=0.0,
        radius_m=0.05,
        start_angle_rad=0.0,
        end_angle_rad=1.0,
        direction="ccw",
    )

    assert len(builder.actions) == action_count_before
    assert len(builder.strokes) == stroke_count_before
    assert summary["errors"] == []
    assert any("arc bounding box is outside" in error for error in summary["failed_calls"])


def test_draw_arc_wrong_start_position_fails_without_stroke_or_action():
    builder = PlanBuilder.begin_plan("bad arc start")
    builder.move_to_start(0.0, 0.0)
    builder.pen_down()
    action_count_before = len(builder.actions)
    stroke_count_before = len(builder.strokes)

    summary = builder.draw_arc(
        center_x=0.0,
        center_y=0.0,
        radius_m=0.05,
        start_angle_rad=0.0,
        end_angle_rad=math.pi,
        direction="ccw",
    )

    assert len(builder.actions) == action_count_before
    assert len(builder.strokes) == stroke_count_before
    assert builder.current_position.x == pytest.approx(0.0)
    assert builder.current_position.y == pytest.approx(0.0)
    assert summary["errors"] == []
    assert any("expected arc start is (0.05, 0.0)" in error for error in summary["failed_calls"])
    assert any("move_to_start(x=0.05, y=0.0)" in error for error in summary["failed_calls"])


def test_draw_arc_matching_start_position_succeeds():
    builder = PlanBuilder.begin_plan("good arc start")
    builder.move_to_start(0.05, 0.0)
    builder.pen_down()

    summary = builder.draw_arc(
        center_x=0.0,
        center_y=0.0,
        radius_m=0.05,
        start_angle_rad=0.0,
        end_angle_rad=math.pi,
        direction="ccw",
    )

    assert summary["errors"] == []
    assert len(builder.strokes) == 1
    assert builder.actions[-1].name == "draw_arc"


def test_full_circle_leaves_current_position_at_original_arc_start():
    builder = PlanBuilder.begin_plan("circle")
    builder.move_to_start(0.05, 0.0)
    builder.pen_down()

    summary = builder.draw_arc(
        center_x=0.0,
        center_y=0.0,
        radius_m=0.05,
        start_angle_rad=0.0,
        end_angle_rad=2.0 * math.pi,
        direction="ccw",
    )

    assert summary["current_position"]["x"] == pytest.approx(0.05)
    assert summary["current_position"]["y"] == pytest.approx(0.0)
    assert builder.current_position.x == pytest.approx(0.05)
    assert builder.current_position.y == pytest.approx(0.0)


def test_failed_out_of_board_attempt_does_not_poison_repaired_plan():
    builder = PlanBuilder.begin_plan("repair outside line")
    builder.move_to_start(0.0, 0.0)
    builder.pen_down()
    failed = builder.draw_line_to(2.0, 0.0)
    assert failed["errors"] == []
    assert failed["failed_calls"]

    builder.draw_line_to(0.05, 0.0)
    builder.pen_up()
    summary = builder.finish_plan()

    assert summary["finished"] is True
    assert summary["errors"] == []
    assert summary["failed_calls"]
    assert [action.name for action in builder.actions] == [
        "move_to_start",
        "pen_down",
        "draw_line",
        "pen_up",
    ]
