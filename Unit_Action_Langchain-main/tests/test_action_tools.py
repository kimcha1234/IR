import json

from robot_drawing_planner.action_tools import UnitActionToolset, create_unit_action_tools

FINISH_WITH_PEN_DOWN_MESSAGE = (
    "finish_plan requires pen_state == 'up'; call pen_up before finish_plan."
)


def invoke(toolset: UnitActionToolset, name: str, payload: dict | None = None):
    return toolset.tool_by_name(name).invoke(payload or {})


def run_square_sequence(toolset: UnitActionToolset):
    assert invoke(toolset, "begin_plan", {"source_command": "draw a square"})["ok"] is True
    assert invoke(toolset, "move_to_start", {"x": -0.05, "y": 0.05})["ok"] is True
    assert invoke(toolset, "align_pen_orientation")["ok"] is True
    assert invoke(toolset, "pen_down")["ok"] is True
    assert invoke(toolset, "draw_line_to", {"x": 0.05, "y": 0.05})["ok"] is True
    assert invoke(toolset, "draw_line_to", {"x": 0.05, "y": -0.05})["ok"] is True
    assert invoke(toolset, "draw_line_to", {"x": -0.05, "y": -0.05})["ok"] is True
    assert invoke(toolset, "draw_line_to", {"x": -0.05, "y": 0.05})["ok"] is True
    assert invoke(toolset, "pen_up")["ok"] is True


def test_begin_plan_then_square_sequence_direct_tool_invocation():
    toolset, tools = create_unit_action_tools()
    assert {tool.name for tool in tools} == {
        "begin_plan",
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_line_to",
        "draw_arc",
        "pen_up",
        "check_plan",
        "finish_plan",
    }

    run_square_sequence(toolset)

    assert toolset.builder is not None
    assert len(toolset.builder.actions) == 8
    assert len(toolset.builder.strokes) == 4


def test_check_plan_returns_ok_true_for_valid_square():
    toolset = UnitActionToolset()
    run_square_sequence(toolset)

    feedback = invoke(toolset, "check_plan")

    assert feedback["ok"] is True
    assert feedback["message"] == "Plan checked."
    assert feedback["current_position"] == {"x": -0.05, "y": 0.05, "unit": "m"}
    assert feedback["pen_state"] == "up"
    assert feedback["action_count"] == 8
    assert feedback["warnings"] == []
    assert feedback["errors"] == []
    assert feedback["failed_calls"] == []


def test_finish_plan_returns_serializable_plan_payload():
    toolset = UnitActionToolset()
    run_square_sequence(toolset)
    invoke(toolset, "check_plan")

    feedback = invoke(toolset, "finish_plan")

    assert feedback["ok"] is True
    assert feedback["message"] == "Plan finished."
    assert feedback["action_count"] == 8
    assert feedback["plan"]["source_command"] == "draw a square"
    assert feedback["plan"]["finished"] is True
    assert len(feedback["plan"]["actions"]) == 8
    assert len(feedback["plan"]["strokes"]) == 4
    json.dumps(feedback["plan"])


def test_finish_plan_requires_check_plan_before_finish():
    toolset = UnitActionToolset()
    run_square_sequence(toolset)

    feedback = invoke(toolset, "finish_plan")

    assert feedback["ok"] is False
    assert "requires check_plan" in feedback["message"]


def test_finish_plan_with_pen_down_returns_false_without_plan_payload():
    toolset = UnitActionToolset()
    invoke(toolset, "begin_plan", {"source_command": "unsafe finish"})
    invoke(toolset, "move_to_start", {"x": 0.0, "y": 0.0})
    invoke(toolset, "pen_down")
    invoke(toolset, "draw_line_to", {"x": 0.05, "y": 0.0})
    invoke(toolset, "check_plan")

    feedback = invoke(toolset, "finish_plan")

    assert feedback["ok"] is False
    assert feedback["message"] == FINISH_WITH_PEN_DOWN_MESSAGE
    assert feedback["pen_state"] == "down"
    assert feedback["errors"] == [FINISH_WITH_PEN_DOWN_MESSAGE]
    assert "plan" not in feedback
    assert toolset.builder is not None
    assert toolset.builder.finished is False


def test_invalid_pen_state_returns_ok_false():
    toolset = UnitActionToolset()
    invoke(toolset, "begin_plan", {"source_command": "bad plan"})
    invoke(toolset, "move_to_start", {"x": 0.0, "y": 0.0})

    feedback = invoke(toolset, "draw_line_to", {"x": 0.1, "y": 0.0})

    assert feedback["ok"] is False
    assert "draw_line_to requires pen_state == 'down'." == feedback["message"]
    assert feedback["action_count"] == 1
    assert feedback["errors"] == ["draw_line_to requires pen_state == 'down'."]
    assert feedback["failed_calls"] == ["draw_line_to requires pen_state == 'down'."]


def test_move_to_start_outside_board_returns_false_and_keeps_action_count_zero():
    toolset = UnitActionToolset()
    invoke(toolset, "begin_plan", {"source_command": "bad outside move"})

    feedback = invoke(toolset, "move_to_start", {"x": 2.0, "y": 0.0})

    assert feedback["ok"] is False
    assert feedback["action_count"] == 0
    assert "outside the board bounds" in feedback["message"]
    assert any("outside the board bounds" in error for error in feedback["errors"])
    assert toolset.builder is not None
    assert toolset.builder.actions == []


def test_draw_line_to_outside_board_returns_false_without_new_stroke():
    toolset = UnitActionToolset()
    invoke(toolset, "begin_plan", {"source_command": "bad outside line"})
    invoke(toolset, "move_to_start", {"x": 0.0, "y": 0.0})
    invoke(toolset, "pen_down")
    assert toolset.builder is not None
    stroke_count_before = len(toolset.builder.strokes)
    action_count_before = len(toolset.builder.actions)

    feedback = invoke(toolset, "draw_line_to", {"x": 2.0, "y": 0.0})

    assert feedback["ok"] is False
    assert len(toolset.builder.strokes) == stroke_count_before
    assert len(toolset.builder.actions) == action_count_before
    assert any("outside the board bounds" in error for error in feedback["errors"])


def test_draw_arc_outside_board_returns_false_without_new_stroke():
    toolset = UnitActionToolset()
    invoke(toolset, "begin_plan", {"source_command": "bad outside arc"})
    invoke(toolset, "move_to_start", {"x": 0.0, "y": 0.0})
    invoke(toolset, "pen_down")
    assert toolset.builder is not None
    stroke_count_before = len(toolset.builder.strokes)
    action_count_before = len(toolset.builder.actions)

    feedback = invoke(
        toolset,
        "draw_arc",
        {
            "center_x": 0.24,
            "center_y": 0.0,
            "radius_m": 0.05,
            "start_angle_rad": 0.0,
            "end_angle_rad": 1.0,
            "direction": "ccw",
        },
    )

    assert feedback["ok"] is False
    assert len(toolset.builder.strokes) == stroke_count_before
    assert len(toolset.builder.actions) == action_count_before
    assert any("arc bounding box is outside" in error for error in feedback["errors"])


def test_draw_arc_wrong_start_position_returns_false_without_new_stroke():
    toolset = UnitActionToolset()
    invoke(toolset, "begin_plan", {"source_command": "bad arc start"})
    invoke(toolset, "move_to_start", {"x": 0.0, "y": 0.0})
    invoke(toolset, "pen_down")
    assert toolset.builder is not None
    stroke_count_before = len(toolset.builder.strokes)
    action_count_before = len(toolset.builder.actions)

    feedback = invoke(
        toolset,
        "draw_arc",
        {
            "center_x": 0.0,
            "center_y": 0.0,
            "radius_m": 0.05,
            "start_angle_rad": 0.0,
            "end_angle_rad": 1.0,
            "direction": "ccw",
        },
    )

    assert feedback["ok"] is False
    assert len(toolset.builder.strokes) == stroke_count_before
    assert len(toolset.builder.actions) == action_count_before
    assert any("expected arc start is (0.05, 0.0)" in error for error in feedback["errors"])
    assert any("move_to_start(x=0.05, y=0.0)" in error for error in feedback["errors"])


def test_repaired_out_of_board_attempt_can_finish_successfully():
    toolset = UnitActionToolset()
    invoke(toolset, "begin_plan", {"source_command": "repair outside line"})
    invoke(toolset, "move_to_start", {"x": 0.0, "y": 0.0})
    invoke(toolset, "pen_down")
    failed = invoke(toolset, "draw_line_to", {"x": 2.0, "y": 0.0})
    assert failed["ok"] is False

    assert invoke(toolset, "draw_line_to", {"x": 0.05, "y": 0.0})["ok"] is True
    assert invoke(toolset, "pen_up")["ok"] is True
    checked = invoke(toolset, "check_plan")
    assert checked["ok"] is True
    assert checked["errors"] == []
    assert checked["failed_calls"]

    finished = invoke(toolset, "finish_plan")
    assert finished["ok"] is True
    assert finished["plan"]["finished"] is True
    assert finished["plan"]["diagnostics"]["errors"] == []
    assert finished["plan"]["diagnostics"]["failed_calls"]


def test_tool_descriptions_are_explicit_about_scope():
    toolset = UnitActionToolset()
    for tool in toolset.tools():
        description = tool.description
        assert "symbolic robot primitive actions" in description
        assert "does not move the robot" in description
        assert "does not compute IK, FK, Jacobians" in description
        assert "Isaac Sim commands" in description
    assert "must call check_plan before finish_plan" in toolset.tool_by_name(
        "finish_plan"
    ).description
