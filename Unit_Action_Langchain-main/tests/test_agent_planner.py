import math

from langchain_core.messages import AIMessage

from robot_drawing_planner.agent_planner import (
    build_agent_system_prompt,
    plan_drawing_agentic,
)
from robot_drawing_planner.config import PlannerConfig
from robot_drawing_planner.planner import plan_from_goal
from robot_drawing_planner.schemas import Measurement, ParsedGoal

FINISH_WITH_PEN_DOWN_MESSAGE = (
    "finish_plan requires pen_state == 'up'; call pen_up before finish_plan."
)


class FakeToolCallingLLM:
    def __init__(self, tool_call_batches):
        self.tool_call_batches = list(tool_call_batches)
        self.bound_tools = None
        self.calls = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if self.tool_call_batches:
            batch = self.tool_call_batches.pop(0)
        else:
            batch = []
        return AIMessage(content="", tool_calls=batch)


def tc(name, args=None, call_id=None):
    return {
        "name": name,
        "args": args or {},
        "id": call_id or f"{name}_id",
    }


def square_batches():
    return [
        [tc("begin_plan", {"source_command": "draw square"})],
        [tc("move_to_start", {"x": -0.05, "y": 0.05})],
        [tc("align_pen_orientation")],
        [tc("pen_down")],
        [tc("draw_line_to", {"x": 0.05, "y": 0.05})],
        [tc("draw_line_to", {"x": 0.05, "y": -0.05})],
        [tc("draw_line_to", {"x": -0.05, "y": -0.05})],
        [tc("draw_line_to", {"x": -0.05, "y": 0.05})],
        [tc("pen_up")],
        [tc("check_plan")],
        [tc("finish_plan")],
    ]


def test_agentic_prompt_contains_board_scale_sequence_and_arc_guidance():
    prompt = build_agent_system_prompt(PlannerConfig())

    assert "coordinates are meters" in prompt
    assert "current board x range is [-0.25, 0.25]" in prompt
    assert "current board y range is [-0.175, 0.175]" in prompt
    assert "Do not use pixel-like or arbitrary coordinates such as 1, 2, 3" in prompt
    assert "pen_up before finish_plan" in prompt
    assert "arc start point must match current pen position" in prompt
    assert "multiple Unit Action tools in a single LLM response" in prompt
    assert "Finish within the tool budget" in prompt
    assert "check_plan, then finish_plan immediately" in prompt
    assert "simple body rectangle, triangular roof, and optional door" in prompt


def test_agentic_prompt_uses_actual_configured_board_range():
    prompt = build_agent_system_prompt(
        PlannerConfig(board_width_m=0.4, board_height_m=0.2)
    )

    assert "current board x range is [-0.2, 0.2]" in prompt
    assert "current board y range is [-0.1, 0.1]" in prompt


def test_plan_drawing_agentic_sends_strengthened_prompt_to_llm():
    llm = FakeToolCallingLLM(square_batches())

    plan_drawing_agentic("draw a square", llm=llm)

    prompt = llm.calls[0][0].content
    assert "coordinates are meters" in prompt
    assert "begin_plan first" in prompt
    assert "move_to_start before pen_down" in prompt
    assert "pen_up before finish_plan" in prompt
    assert "For house/star/smiley/letters" in prompt


def test_fake_llm_calls_tools_for_square():
    llm = FakeToolCallingLLM(square_batches())

    plan = plan_drawing_agentic("draw a square", llm=llm)

    assert plan.diagnostics["mode"] == "agentic_unit_action_tools"
    assert plan.diagnostics["validation_ok"] is True
    assert plan.goal.shape_type == "custom"
    assert [action.name for action in plan.actions] == [
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_line",
        "draw_line",
        "draw_line",
        "draw_line",
        "pen_up",
    ]
    assert len(plan.strokes) == 4
    assert llm.bound_tools is not None


def test_fake_llm_calls_tools_for_circle():
    llm = FakeToolCallingLLM(
        [
            [tc("begin_plan", {"source_command": "draw circle"})],
            [tc("move_to_start", {"x": 0.05, "y": 0.0})],
            [tc("align_pen_orientation")],
            [tc("pen_down")],
            [
                tc(
                    "draw_arc",
                    {
                        "center_x": 0.0,
                        "center_y": 0.0,
                        "radius_m": 0.05,
                        "start_angle_rad": 0.0,
                        "end_angle_rad": 2.0 * math.pi,
                        "direction": "ccw",
                    },
                )
            ],
            [tc("pen_up")],
            [tc("check_plan")],
            [tc("finish_plan")],
        ]
    )

    plan = plan_drawing_agentic("draw a circle", llm=llm)

    assert plan.diagnostics["validation_ok"] is True
    assert [action.name for action in plan.actions] == [
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_arc",
        "pen_up",
    ]
    assert len(plan.strokes) == 1
    assert plan.strokes[0].type == "arc"


def test_agentic_and_template_square_share_handoff_param_keys():
    agentic_plan = plan_drawing_agentic(
        "draw a square",
        llm=FakeToolCallingLLM(square_batches()),
    )
    template_plan = plan_from_goal(
        ParsedGoal(
            shape_type="square",
            side_length=Measurement(value=10, unit="cm"),
            raw_command="draw a square",
        )
    )

    agentic_by_name = _first_action_by_name(agentic_plan.actions)
    template_by_name = _first_action_by_name(template_plan.actions)

    for action_name in [
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_line",
        "pen_up",
    ]:
        assert set(agentic_by_name[action_name].params) == set(
            template_by_name[action_name].params
        )


def test_agentic_and_template_circle_share_draw_arc_handoff_param_keys():
    agentic_plan = plan_drawing_agentic(
        "draw a circle",
        llm=FakeToolCallingLLM(
            [
                [tc("begin_plan", {"source_command": "draw circle"})],
                [tc("move_to_start", {"x": 0.05, "y": 0.0})],
                [tc("align_pen_orientation")],
                [tc("pen_down")],
                [
                    tc(
                        "draw_arc",
                        {
                            "center_x": 0.0,
                            "center_y": 0.0,
                            "radius_m": 0.05,
                            "start_angle_rad": 0.0,
                            "end_angle_rad": 2.0 * math.pi,
                            "direction": "ccw",
                        },
                    )
                ],
                [tc("pen_up")],
                [tc("check_plan")],
                [tc("finish_plan")],
            ]
        ),
    )
    template_plan = plan_from_goal(
        ParsedGoal(
            shape_type="circle",
            radius=Measurement(value=5, unit="cm"),
            raw_command="draw a circle",
        )
    )

    agentic_arc = _first_action_by_name(agentic_plan.actions)["draw_arc"]
    template_arc = _first_action_by_name(template_plan.actions)["draw_arc"]

    assert set(agentic_arc.params) == set(template_arc.params)
    assert agentic_arc.params["center"]["z"] == template_arc.params["center"]["z"]


def test_fake_llm_calls_tools_for_house_like_shape():
    llm = FakeToolCallingLLM(
        [
            [tc("begin_plan", {"source_command": "draw a house"})],
            [tc("move_to_start", {"x": -0.05, "y": -0.05})],
            [tc("align_pen_orientation")],
            [tc("pen_down")],
            [tc("draw_line_to", {"x": 0.05, "y": -0.05})],
            [tc("draw_line_to", {"x": 0.05, "y": 0.05})],
            [tc("draw_line_to", {"x": -0.05, "y": 0.05})],
            [tc("draw_line_to", {"x": -0.05, "y": -0.05})],
            [tc("pen_up")],
            [tc("move_to_start", {"x": -0.06, "y": 0.05})],
            [tc("pen_down")],
            [tc("draw_line_to", {"x": 0.0, "y": 0.10})],
            [tc("draw_line_to", {"x": 0.06, "y": 0.05})],
            [tc("draw_line_to", {"x": -0.06, "y": 0.05})],
            [tc("pen_up")],
            [tc("check_plan")],
            [tc("finish_plan")],
        ]
    )

    plan = plan_drawing_agentic("draw a house-like shape", llm=llm)

    assert plan.diagnostics["validation_ok"] is True
    assert len(plan.strokes) == 7
    assert [action.name for action in plan.actions].count("draw_line") == 7
    assert plan.goal.shape_type == "custom"


def test_fake_llm_repairs_invalid_draw_line_before_pen_down_after_feedback():
    llm = FakeToolCallingLLM(
        [
            [tc("begin_plan", {"source_command": "repair a line"})],
            [tc("draw_line_to", {"x": 0.1, "y": 0.0})],
            [tc("begin_plan", {"source_command": "repair a line"})],
            [tc("move_to_start", {"x": 0.0, "y": 0.0})],
            [tc("align_pen_orientation")],
            [tc("pen_down")],
            [tc("draw_line_to", {"x": 0.1, "y": 0.0})],
            [tc("pen_up")],
            [tc("check_plan")],
            [tc("finish_plan")],
        ]
    )

    plan = plan_drawing_agentic("repair a line", llm=llm)

    assert plan.diagnostics["validation_ok"] is True
    assert plan.actions
    assert [action.name for action in plan.actions] == [
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_line",
        "pen_up",
    ]
    assert any(
        "draw_line_to requires pen_state" in message.content
        for call_messages in llm.calls
        for message in call_messages
        if hasattr(message, "content")
    )


def test_fake_llm_repairs_out_of_board_line_after_feedback():
    llm = FakeToolCallingLLM(
        [
            [tc("begin_plan", {"source_command": "repair outside line"})],
            [tc("move_to_start", {"x": 0.0, "y": 0.0})],
            [tc("align_pen_orientation")],
            [tc("pen_down")],
            [tc("draw_line_to", {"x": 2.0, "y": 0.0})],
            [tc("draw_line_to", {"x": 0.05, "y": 0.0})],
            [tc("pen_up")],
            [tc("check_plan")],
            [tc("finish_plan")],
        ]
    )

    plan = plan_drawing_agentic("repair outside line", llm=llm)

    assert plan.diagnostics["validation_ok"] is True
    assert plan.diagnostics["errors"] == []
    assert any("outside the board bounds" in call for call in plan.diagnostics["failed_calls"])
    assert [action.name for action in plan.actions] == [
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_line",
        "pen_up",
    ]
    assert len(plan.strokes) == 1
    assert plan.strokes[0].end.x == 0.05
    assert any(
        "outside the board bounds" in message.content
        for call_messages in llm.calls
        for message in call_messages
        if hasattr(message, "content")
    )


def test_fake_llm_repairs_finish_plan_with_pen_down_after_feedback():
    llm = FakeToolCallingLLM(
        [
            [tc("begin_plan", {"source_command": "repair unsafe finish"})],
            [tc("move_to_start", {"x": 0.0, "y": 0.0})],
            [tc("align_pen_orientation")],
            [tc("pen_down")],
            [tc("draw_line_to", {"x": 0.05, "y": 0.0})],
            [tc("check_plan")],
            [tc("finish_plan")],
            [tc("pen_up")],
            [tc("check_plan")],
            [tc("finish_plan")],
        ]
    )

    plan = plan_drawing_agentic("repair unsafe finish", llm=llm)

    assert plan.diagnostics["validation_ok"] is True
    assert plan.diagnostics["errors"] == []
    assert FINISH_WITH_PEN_DOWN_MESSAGE in plan.diagnostics["failed_calls"]
    assert [action.name for action in plan.actions] == [
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_line",
        "pen_up",
    ]
    assert any(
        FINISH_WITH_PEN_DOWN_MESSAGE in message.content
        for call_messages in llm.calls
        for message in call_messages
        if hasattr(message, "content")
    )


def test_fake_llm_repairs_invalid_arc_start_after_feedback():
    llm = FakeToolCallingLLM(
        [
            [tc("begin_plan", {"source_command": "repair bad circle start"})],
            [tc("move_to_start", {"x": 0.0, "y": 0.0})],
            [tc("align_pen_orientation")],
            [tc("pen_down")],
            [
                tc(
                    "draw_arc",
                    {
                        "center_x": 0.0,
                        "center_y": 0.0,
                        "radius_m": 0.05,
                        "start_angle_rad": 0.0,
                        "end_angle_rad": 2.0 * math.pi,
                        "direction": "ccw",
                    },
                )
            ],
            [tc("begin_plan", {"source_command": "repair bad circle start"})],
            [tc("move_to_start", {"x": 0.05, "y": 0.0})],
            [tc("align_pen_orientation")],
            [tc("pen_down")],
            [
                tc(
                    "draw_arc",
                    {
                        "center_x": 0.0,
                        "center_y": 0.0,
                        "radius_m": 0.05,
                        "start_angle_rad": 0.0,
                        "end_angle_rad": 2.0 * math.pi,
                        "direction": "ccw",
                    },
                )
            ],
            [tc("pen_up")],
            [tc("check_plan")],
            [tc("finish_plan")],
        ]
    )

    plan = plan_drawing_agentic("repair bad circle start", llm=llm)

    assert plan.diagnostics["validation_ok"] is True
    assert plan.diagnostics["errors"] == []
    assert [action.name for action in plan.actions] == [
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_arc",
        "pen_up",
    ]
    assert len(plan.strokes) == 1
    assert plan.strokes[0].type == "arc"
    assert any(
        "expected arc start is (0.05, 0.0)" in message.content
        and "move_to_start(x=0.05, y=0.0)" in message.content
        for call_messages in llm.calls
        for message in call_messages
        if hasattr(message, "content")
    )


def test_unrepaired_finish_plan_with_pen_down_exceeds_max_steps_with_empty_actions():
    llm = FakeToolCallingLLM(
        [
            [tc("begin_plan", {"source_command": "never repair unsafe finish"})],
            [tc("move_to_start", {"x": 0.0, "y": 0.0})],
            [tc("pen_down")],
            [tc("draw_line_to", {"x": 0.05, "y": 0.0})],
            [tc("check_plan")],
            [tc("finish_plan")],
        ]
    )

    plan = plan_drawing_agentic("never repair unsafe finish", llm=llm, max_steps=6)

    assert plan.actions == []
    assert plan.diagnostics["validation_ok"] is False
    assert FINISH_WITH_PEN_DOWN_MESSAGE in plan.diagnostics["errors"]
    assert any("max_llm_steps 6 exceeded" in error for error in plan.diagnostics["errors"])
    assert FINISH_WITH_PEN_DOWN_MESSAGE in plan.diagnostics["failed_calls"]


def test_unrepaired_out_of_board_attempt_exceeds_max_steps_with_empty_actions():
    llm = FakeToolCallingLLM(
        [
            [tc("begin_plan", {"source_command": "never repair outside line"})],
            [tc("move_to_start", {"x": 0.0, "y": 0.0})],
            [tc("pen_down")],
            [tc("draw_line_to", {"x": 2.0, "y": 0.0})],
            [tc("draw_line_to", {"x": 2.0, "y": 0.0})],
        ]
    )

    plan = plan_drawing_agentic("never repair outside line", llm=llm, max_steps=5)

    assert plan.actions == []
    assert plan.strokes == []
    assert plan.diagnostics["validation_ok"] is False
    assert any("max_llm_steps 5 exceeded" in error for error in plan.diagnostics["errors"])
    assert any("outside the board bounds" in call for call in plan.diagnostics["failed_calls"])


def test_max_steps_failure_returns_diagnostics_error():
    llm = FakeToolCallingLLM([[tc("begin_plan", {"source_command": "never finish"})]])

    plan = plan_drawing_agentic("never finish", llm=llm, max_steps=2)

    assert plan.actions == []
    assert plan.diagnostics["validation_ok"] is False
    assert any("max_llm_steps 2 exceeded" in error for error in plan.diagnostics["errors"])


def test_default_budget_diagnostics_are_recorded():
    plan = plan_drawing_agentic(
        "draw a square",
        llm=FakeToolCallingLLM(square_batches()),
    )

    assert plan.diagnostics["validation_ok"] is True
    assert plan.diagnostics["max_llm_steps"] == 80
    assert plan.diagnostics["max_tool_calls"] == 200
    assert plan.diagnostics["llm_step_count"] == 11
    assert plan.diagnostics["tool_call_count"] == 11


def test_max_steps_argument_still_limits_llm_loop_for_compatibility():
    llm = FakeToolCallingLLM([[tc("begin_plan", {"source_command": "never finish"})]])

    plan = plan_drawing_agentic("never finish", llm=llm, max_steps=2)

    assert plan.diagnostics["validation_ok"] is False
    assert plan.diagnostics["max_llm_steps"] == 2
    assert plan.diagnostics["llm_step_count"] == 2
    assert any("max_llm_steps 2 exceeded" in error for error in plan.diagnostics["errors"])


def test_max_tool_calls_stops_execution_even_with_llm_budget_remaining():
    llm = FakeToolCallingLLM(
        [
            [
                tc("begin_plan", {"source_command": "too many tools"}),
                tc("move_to_start", {"x": 0.0, "y": 0.0}),
                tc("pen_down"),
                tc("draw_line_to", {"x": 0.05, "y": 0.0}),
            ],
        ]
    )

    plan = plan_drawing_agentic(
        "too many tools",
        llm=llm,
        max_llm_steps=10,
        max_tool_calls=3,
    )

    assert plan.actions == []
    assert plan.diagnostics["validation_ok"] is False
    assert plan.diagnostics["max_llm_steps"] == 10
    assert plan.diagnostics["max_tool_calls"] == 3
    assert plan.diagnostics["llm_step_count"] == 1
    assert plan.diagnostics["tool_call_count"] == 3
    assert any("max_tool_calls 3 exceeded" in error for error in plan.diagnostics["errors"])


def test_auto_finalize_valid_pen_up_plan_after_llm_budget():
    llm = FakeToolCallingLLM(
        [
            [
                tc("begin_plan", {"source_command": "short line"}),
                tc("move_to_start", {"x": 0.0, "y": 0.0}),
                tc("align_pen_orientation"),
                tc("pen_down"),
                tc("draw_line_to", {"x": 0.05, "y": 0.0}),
                tc("pen_up"),
            ],
        ]
    )

    plan = plan_drawing_agentic("short line", llm=llm, max_llm_steps=1)

    assert plan.diagnostics["validation_ok"] is True
    assert plan.actions
    assert len(plan.strokes) == 1
    assert any("auto-finalized" in warning for warning in plan.diagnostics["warnings"])
    assert [action.name for action in plan.actions][-1] == "pen_up"


def test_auto_finalize_can_be_disabled_for_valid_partial_plan():
    llm = FakeToolCallingLLM(
        [
            [
                tc("begin_plan", {"source_command": "short line"}),
                tc("move_to_start", {"x": 0.0, "y": 0.0}),
                tc("pen_down"),
                tc("draw_line_to", {"x": 0.05, "y": 0.0}),
                tc("pen_up"),
            ],
        ]
    )

    plan = plan_drawing_agentic(
        "short line",
        llm=llm,
        max_llm_steps=1,
        auto_finalize_if_valid=False,
    )

    assert plan.actions == []
    assert plan.diagnostics["validation_ok"] is False
    assert plan.diagnostics["partial_preview_available"] is True
    assert plan.diagnostics["partial_stroke_count"] == 1
    assert plan.diagnostics["partial_plan_is_not_executable"] is True


def test_pen_state_down_cannot_auto_finalize():
    llm = FakeToolCallingLLM(
        [
            [
                tc("begin_plan", {"source_command": "unsafe line"}),
                tc("move_to_start", {"x": 0.0, "y": 0.0}),
                tc("pen_down"),
                tc("draw_line_to", {"x": 0.05, "y": 0.0}),
            ],
        ]
    )

    plan = plan_drawing_agentic("unsafe line", llm=llm, max_llm_steps=1)

    assert plan.actions == []
    assert plan.diagnostics["validation_ok"] is False
    assert plan.diagnostics["partial_preview_available"] is True
    assert plan.diagnostics["partial_stroke_count"] == 1
    assert plan.diagnostics["partial_plan_is_not_executable"] is True


def test_no_strokes_cannot_auto_finalize():
    llm = FakeToolCallingLLM(
        [
            [
                tc("begin_plan", {"source_command": "no strokes"}),
                tc("move_to_start", {"x": 0.0, "y": 0.0}),
            ],
        ]
    )

    plan = plan_drawing_agentic("no strokes", llm=llm, max_llm_steps=1)

    assert plan.actions == []
    assert plan.diagnostics["validation_ok"] is False
    assert plan.diagnostics["partial_preview_available"] is False
    assert plan.diagnostics["partial_stroke_count"] == 0
    assert plan.diagnostics["partial_plan_is_not_executable"] is False


def test_agentic_output_actions_come_from_tool_calls_not_template_compiler():
    llm = FakeToolCallingLLM(
        [
            [tc("begin_plan", {"source_command": "custom rectangle"})],
            [tc("move_to_start", {"x": -0.03, "y": 0.02})],
            [tc("align_pen_orientation")],
            [tc("pen_down")],
            [tc("draw_line_to", {"x": 0.07, "y": 0.02})],
            [tc("pen_up")],
            [tc("check_plan")],
            [tc("finish_plan")],
        ]
    )

    plan = plan_drawing_agentic("custom rectangle", llm=llm)

    assert plan.diagnostics["mode"] == "agentic_unit_action_tools"
    assert plan.goal.assumptions == [
        "Agentic mode does not use deterministic ParsedGoal templates."
    ]
    assert len(plan.actions) == 5
    assert len(plan.strokes) == 1
    assert plan.strokes[0].start.x == -0.03
    assert plan.strokes[0].end.x == 0.07


def _first_action_by_name(actions):
    by_name = {}
    for action in actions:
        by_name.setdefault(action.name, action)
    return by_name
