import pytest
from pydantic import ValidationError

from robot_drawing_planner.agent_events import (
    AgentRunEvent,
    AgentRunResult,
    event_to_chat_markdown,
    make_event,
)
from robot_drawing_planner.schemas import DrawingPlan, NormalizedGoal, Point2D


def minimal_plan() -> DrawingPlan:
    return DrawingPlan(
        source_command="draw a line",
        goal=NormalizedGoal(
            shape_type="custom",
            center=Point2D(x=0.0, y=0.0),
            radius_m=None,
            side_length_m=None,
            size_m=0.1,
            orientation_rad=0.0,
            letter=None,
            assumptions=[],
            warnings=[],
        ),
        strokes=[],
        actions=[],
        diagnostics={"validation_ok": True},
    )


def test_agent_run_event_instantiates():
    event = make_event(
        event_index=0,
        step_index=0,
        event_type="tool_call",
        tool_name="move_to_start",
        tool_args={"x": 0.0, "y": 0.0},
        message="Calling move_to_start.",
        ok=None,
    )

    assert event.event_index == 0
    assert event.event_type == "tool_call"
    assert event.tool_args == {"x": 0.0, "y": 0.0}
    assert "T" in event.timestamp


def test_agent_run_result_instantiates_with_minimal_drawing_plan():
    event = make_event(0, "user_request", "draw a line")
    result = AgentRunResult(
        command="draw a line",
        plan=minimal_plan(),
        events=[event],
        plan_json_path="outputs/plan.json",
        plot_png_path="outputs/plan.png",
        events_json_path="outputs/plan_events.json",
    )

    assert result.command == "draw a line"
    assert result.plan.diagnostics["validation_ok"] is True
    assert result.events == [event]
    assert result.events_json_path == "outputs/plan_events.json"


def test_event_to_chat_markdown_formats_tool_call():
    event = make_event(
        event_index=1,
        event_type="tool_call",
        tool_name="move_to_start",
        tool_args={"x": 0.05, "y": 0.0},
        message="Calling move_to_start.",
    )

    text = event_to_chat_markdown(event)

    assert text.startswith("🔧 Tool Call: move_to_start")
    assert '"x": 0.05' in text
    assert '"y": 0.0' in text


def test_event_to_chat_markdown_formats_failed_tool_result():
    event = make_event(
        event_index=2,
        event_type="tool_result",
        tool_result={"ok": False, "message": "outside board"},
        message="outside board",
        ok=False,
    )

    text = event_to_chat_markdown(event)

    assert text.startswith("❌ Tool Result: ok=False")
    assert "message: outside board" in text


def test_agent_run_event_rejects_extra_fields():
    with pytest.raises(ValidationError):
        AgentRunEvent(
            event_index=0,
            step_index=None,
            event_type="user_request",
            message="draw",
            timestamp="2026-05-13T00:00:00+00:00",
            unexpected="nope",
        )


def test_agent_run_event_rejects_non_json_serializable_metadata():
    with pytest.raises(ValidationError, match="JSON-serializable"):
        make_event(
            event_index=0,
            event_type="tool_call",
            message="bad args",
            tool_args={"bad": object()},
        )
