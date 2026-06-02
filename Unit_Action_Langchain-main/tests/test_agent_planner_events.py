import json

from langchain_core.messages import AIMessage

from robot_drawing_planner.agent_events import AgentRunResult
from robot_drawing_planner.agent_planner import plan_drawing_agentic
from robot_drawing_planner.schemas import DrawingPlan


class FakeToolCallingLLM:
    def __init__(self, tool_call_batches):
        self.tool_call_batches = list(tool_call_batches)
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
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


def test_agentic_collect_events_returns_agent_run_result_for_square():
    result = plan_drawing_agentic(
        "draw a square",
        llm=FakeToolCallingLLM(square_batches()),
        collect_events=True,
    )

    assert isinstance(result, AgentRunResult)
    assert result.command == "draw a square"
    assert result.plan.diagnostics["validation_ok"] is True
    assert result.events


def test_agentic_events_include_required_event_types():
    result = plan_drawing_agentic(
        "draw a square",
        llm=FakeToolCallingLLM(square_batches()),
        collect_events=True,
    )

    event_types = [event.event_type for event in result.events]
    assert "user_request" in event_types
    assert any(
        event.event_type == "tool_call" and event.tool_name == "move_to_start"
        for event in result.events
    )
    assert "tool_result" in event_types
    assert "plan_finished" in event_types


def test_agentic_tool_call_events_preserve_fake_llm_order():
    result = plan_drawing_agentic(
        "draw a square",
        llm=FakeToolCallingLLM(square_batches()),
        collect_events=True,
    )

    tool_names = [
        event.tool_name for event in result.events if event.event_type == "tool_call"
    ]

    assert tool_names == [
        "begin_plan",
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_line_to",
        "draw_line_to",
        "draw_line_to",
        "draw_line_to",
        "pen_up",
        "check_plan",
        "finish_plan",
    ]


def test_agentic_collect_events_false_still_returns_drawing_plan():
    plan = plan_drawing_agentic(
        "draw a square",
        llm=FakeToolCallingLLM(square_batches()),
        collect_events=False,
    )

    assert isinstance(plan, DrawingPlan)
    assert plan.diagnostics["validation_ok"] is True


def test_agentic_event_callback_receives_events_without_changing_return_type():
    seen = []

    plan = plan_drawing_agentic(
        "draw a square",
        llm=FakeToolCallingLLM(square_batches()),
        event_callback=seen.append,
    )

    assert isinstance(plan, DrawingPlan)
    assert seen
    assert seen[0].event_type == "user_request"
    assert any(event.event_type == "plan_finished" for event in seen)


def test_agentic_events_do_not_contain_openai_api_key_label():
    result = plan_drawing_agentic(
        "draw a square",
        llm=FakeToolCallingLLM(square_batches()),
        collect_events=True,
    )

    payload = json.dumps(
        [event.model_dump(mode="json") for event in result.events],
        ensure_ascii=False,
    )
    assert "OPENAI_API_KEY" not in payload


def test_agentic_error_event_metadata_includes_budget_values():
    result = plan_drawing_agentic(
        "never finish",
        llm=FakeToolCallingLLM([[tc("begin_plan", {"source_command": "never finish"})]]),
        max_llm_steps=1,
        max_tool_calls=5,
        collect_events=True,
    )

    error_events = [event for event in result.events if event.event_type == "error"]

    assert error_events
    metadata = error_events[-1].metadata
    assert metadata["max_llm_steps"] == 1
    assert metadata["max_tool_calls"] == 5
    assert metadata["llm_step_count"] == 1
    assert metadata["tool_call_count"] == 1


def test_agentic_auto_finalize_events_include_auto_finish_plan():
    result = plan_drawing_agentic(
        "short line",
        llm=FakeToolCallingLLM(
            [
                [
                    tc("begin_plan", {"source_command": "short line"}),
                    tc("move_to_start", {"x": 0.0, "y": 0.0}),
                    tc("pen_down"),
                    tc("draw_line_to", {"x": 0.05, "y": 0.0}),
                    tc("pen_up"),
                ],
            ]
        ),
        max_llm_steps=1,
        collect_events=True,
    )

    auto_finish_calls = [
        event
        for event in result.events
        if event.event_type == "tool_call"
        and event.tool_name == "finish_plan"
        and event.metadata.get("auto") is True
    ]
    plan_finished_events = [
        event for event in result.events if event.event_type == "plan_finished"
    ]

    assert result.plan.diagnostics["validation_ok"] is True
    assert auto_finish_calls
    assert plan_finished_events[-1].metadata["auto_finalized"] is True
    assert any(
        event.event_type == "tool_call"
        and event.tool_name == "check_plan"
        and event.metadata.get("auto") is True
        for event in result.events
    )
