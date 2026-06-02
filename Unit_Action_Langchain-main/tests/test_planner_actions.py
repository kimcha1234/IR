import json

import pytest

from robot_drawing_planner.config import PlannerConfig
from robot_drawing_planner.llm_client import parse_command_with_llm
from robot_drawing_planner.planner import (
    build_plan_from_parsed_goal,
    plan_drawing,
    plan_from_goal,
    plan_from_text,
)
from robot_drawing_planner.schemas import Measurement, ParsedGoal, Point2D


class FakeParser:
    def __init__(self, goal):
        self.goal = goal
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.goal


class FakeLLM:
    def __init__(self, result):
        self.result = result
        self.structured_calls = []
        self.invoke_calls = []

    def with_structured_output(self, schema, method):
        self.structured_calls.append((schema, method))
        return self

    def invoke(self, messages):
        self.invoke_calls.append(messages)
        return self.result


def test_square_plan_has_single_continuous_contour_action_sequence():
    plan = plan_from_goal(
        ParsedGoal(
            shape_type="square",
            side_length=Measurement(value=10, unit="cm"),
            raw_command="draw a square",
        )
    )
    action_names = [action.name for action in plan.actions]
    assert action_names == [
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_line",
        "draw_line",
        "draw_line",
        "draw_line",
        "pen_up",
    ]


def test_triangle_plan_has_three_draw_lines():
    plan = plan_from_goal(
        ParsedGoal(
            shape_type="triangle",
            side_length=Measurement(value=8, unit="cm"),
            raw_command="draw a triangle",
        )
    )
    action_names = [action.name for action in plan.actions]
    assert action_names.count("draw_line") == 3
    assert action_names == [
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_line",
        "draw_line",
        "draw_line",
        "pen_up",
    ]


def test_circle_plan_has_one_draw_arc():
    plan = plan_from_goal(
        ParsedGoal(
            shape_type="circle",
            radius=Measurement(value=5, unit="cm"),
            raw_command="draw a circle",
        )
    )
    action_names = [action.name for action in plan.actions]
    assert action_names.count("draw_arc") == 1
    assert action_names == [
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_arc",
        "pen_up",
    ]


def test_letter_a_has_three_independent_line_groups():
    plan = plan_from_goal(
        ParsedGoal(
            shape_type="letter",
            letter="A",
            size=Measurement(value=8, unit="cm"),
            raw_command="draw a capital A",
        )
    )
    action_names = [action.name for action in plan.actions]
    assert action_names == [
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_line",
        "pen_up",
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_line",
        "pen_up",
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_line",
        "pen_up",
    ]
    assert action_names.count("pen_down") == 3
    assert action_names.count("draw_line") == 3
    assert action_names.count("pen_up") == 3


def test_unsupported_star_returns_empty_actions():
    parsed = ParsedGoal.model_construct(
        shape_type="star",
        center=None,
        radius=None,
        side_length=None,
        size=Measurement(value=5, unit="cm"),
        orientation_deg=0.0,
        letter=None,
        position_hint=None,
        raw_command="draw a star",
    )
    plan = build_plan_from_parsed_goal(parsed)
    assert plan.actions == []
    assert plan.strokes == []
    assert plan.diagnostics["validation_ok"] is False
    assert any("Unsupported shape_type" in error for error in plan.diagnostics["errors"])


def test_out_of_board_circle_returns_empty_actions():
    plan = plan_from_goal(
        ParsedGoal(
            shape_type="circle",
            center=Point2D(x=0.24, y=0.0),
            radius=Measurement(value=5, unit="cm"),
            raw_command="draw a circle near the edge",
        ),
        board=None,
    )
    assert plan.actions == []
    assert plan.diagnostics["validation_ok"] is False
    assert any("validation failed" in error for error in plan.diagnostics["errors"])


def test_action_names_do_not_include_internal_pipeline_names():
    plan = plan_from_goal(
        ParsedGoal(
            shape_type="square",
            side_length=Measurement(value=10, unit="cm"),
            raw_command="draw a square",
        )
    )
    forbidden = {
        "parse_goal",
        "normalize_goal",
        "generate_strokes",
        "validate_goal",
        "verify_done",
    }
    assert not any(action.name in forbidden for action in plan.actions)


def test_action_params_include_planner_handoff_metadata():
    config = PlannerConfig(
        hover_height_m=0.04,
        drawing_z_m=0.0,
        default_speed_m_s=0.03,
        pen_down_speed_m_s=0.01,
        pen_up_speed_m_s=0.02,
    )
    plan = build_plan_from_parsed_goal(
        ParsedGoal(
            shape_type="circle",
            radius=Measurement(value=5, unit="cm"),
            raw_command="draw a circle",
        ),
        config=config,
    )
    move, align, pen_down, draw_arc, pen_up = plan.actions
    assert move.params["target"]["z"] == pytest.approx(0.04)
    assert move.params["note"] == (
        "free-space move; kinematics module converts board frame to base frame"
    )
    assert align.params["mode"] == "normal_to_board"
    assert align.params["note"] == "orientation constraint only; no IK computed here"
    assert pen_down.params["speed_m_s"] == pytest.approx(0.01)
    assert draw_arc.params["sampling_hint"] == (
        "kinematics module should sample circular Cartesian waypoints"
    )
    assert pen_up.params["lift_height_m"] == pytest.approx(0.04)


def test_plan_from_text_uses_fake_parser_without_live_api():
    fake = FakeParser(
        {
            "shape_type": "letter",
            "letter": "A",
            "size": {"value": 8, "unit": "cm"},
            "raw_command": "draw a capital A",
        }
    )
    plan = plan_from_text("draw a capital A", parser=fake)
    assert fake.calls
    assert plan.goal.shape_type == "letter"
    assert plan.goal.letter == "A"
    assert [action.name for action in plan.actions].count("draw_line") == 3


def test_plan_drawing_uses_llm_argument():
    command = "Draw a circle with radius 5 cm"
    fake = FakeLLM(
        ParsedGoal(
            shape_type="circle",
            radius=Measurement(value=5, unit="cm"),
            raw_command=command,
        )
    )
    plan = plan_drawing(command, llm=fake)
    assert fake.structured_calls == [(ParsedGoal, "json_schema")]
    assert plan.goal.shape_type == "circle"
    assert [action.name for action in plan.actions].count("draw_arc") == 1


def test_korean_square_command_parses_with_fake_llm():
    command = "중앙에 한 변 10cm 네모"
    fake = FakeLLM(
        ParsedGoal(
            shape_type="square",
            center=None,
            side_length=Measurement(value=10, unit="cm"),
            position_hint="중앙",
            raw_command=command,
        )
    )
    parsed = parse_command_with_llm(command, llm=fake)

    assert parsed.shape_type == "square"
    assert parsed.side_length == Measurement(value=10, unit="cm")
    assert parsed.center is None
    assert parsed.position_hint == "중앙"
    assert parsed.raw_command == command
    assert fake.structured_calls == [(ParsedGoal, "json_schema")]


def test_english_circle_radius_command_parses_with_fake_llm_dict_result():
    command = "Draw a circle with radius 5 cm"
    fake = FakeLLM(
        {
            "shape_type": "circle",
            "center": None,
            "radius": {"value": 5, "unit": "cm"},
            "raw_command": command,
        }
    )
    parsed = parse_command_with_llm(command, llm=fake)

    assert parsed.shape_type == "circle"
    assert parsed.radius == Measurement(value=5, unit="cm")
    assert parsed.raw_command == command


def test_missing_api_key_raises_only_when_live_llm_requested(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    command = "Draw a circle with radius 5 cm"
    fake = FakeLLM(
        ParsedGoal(
            shape_type="circle",
            radius=Measurement(value=5, unit="cm"),
            raw_command=command,
        )
    )

    parsed = parse_command_with_llm(command, llm=fake)
    assert parsed.shape_type == "circle"

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        parse_command_with_llm(command)


def test_plan_json_contains_no_low_level_robot_fields():
    plan = plan_from_goal(
        ParsedGoal(
            shape_type="triangle",
            side_length=Measurement(value=8, unit="cm"),
            raw_command="draw a triangle",
        )
    )
    payload = json.dumps(plan.model_dump(mode="json"))
    forbidden = ["joint_angles", "jacobian", "isaac_command", "trajectory"]
    assert not any(word in payload for word in forbidden)


def test_empty_command_rejected_before_parser_call():
    fake = FakeParser(
        {
            "shape_type": "circle",
            "size": {"value": 5, "unit": "cm"},
            "raw_command": "draw a circle",
        }
    )
    with pytest.raises(ValueError, match="must not be empty"):
        plan_from_text("   ", parser=fake)
    assert fake.calls == []

