import json

from langchain_core.messages import AIMessage

from robot_drawing_planner.agent_events import AgentRunResult
from robot_drawing_planner.demo_core import run_demo_request, safe_slug


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


def one_line_batches():
    return [
        [tc("begin_plan", {"source_command": "draw a short line"})],
        [tc("move_to_start", {"x": 0.0, "y": 0.0})],
        [tc("align_pen_orientation")],
        [tc("pen_down")],
        [tc("draw_line_to", {"x": 0.05, "y": 0.0})],
        [tc("pen_up")],
        [tc("check_plan")],
        [tc("finish_plan")],
    ]


def test_run_demo_request_no_api_generates_plan_json_and_plot_png(tmp_path):
    result = run_demo_request(
        "중앙에 한 변 10cm짜리 네모를 그려줘",
        mode="no-api",
        out_dir=tmp_path,
    )

    assert isinstance(result, AgentRunResult)
    assert result.plan_json_path is not None
    assert result.plot_png_path is not None
    assert result.events_json_path is not None
    assert result.plan.diagnostics["validation_ok"] is True

    plan_path = tmp_path / result.plan_json_path.split("/")[-1]
    plot_path = tmp_path / result.plot_png_path.split("/")[-1]
    events_path = tmp_path / result.events_json_path.split("/")[-1]
    assert plan_path.exists()
    assert plot_path.exists()
    assert events_path.exists()


def test_run_demo_request_events_include_plot_generated(tmp_path):
    result = run_demo_request(
        "중앙에 반지름 5cm짜리 원을 그려줘",
        mode="no-api",
        out_dir=tmp_path,
    )

    assert any(event.event_type == "plot_generated" for event in result.events)


def test_run_demo_request_png_signature_is_valid(tmp_path):
    result = run_demo_request(
        "중앙에 한 변 10cm짜리 세모를 그려줘",
        mode="no-api",
        out_dir=tmp_path,
    )

    assert result.plot_png_path is not None
    assert open(result.plot_png_path, "rb").read(8) == b"\x89PNG\r\n\x1a\n"


def test_run_demo_request_plan_json_is_valid_json(tmp_path):
    result = run_demo_request(
        "중앙에 크기 10cm인 글자 A를 써줘",
        mode="no-api",
        out_dir=tmp_path,
        create_plot=False,
    )

    assert result.plan_json_path is not None
    data = json.loads(open(result.plan_json_path, encoding="utf-8").read())
    assert data["schema_version"] == "1.0"
    assert data["actions"]


def test_run_demo_request_events_json_is_valid_json(tmp_path):
    result = run_demo_request(
        "중앙에 한 변 10cm짜리 네모를 그려줘",
        mode="no-api",
        out_dir=tmp_path,
        create_plot=False,
    )

    assert result.events_json_path is not None
    data = json.loads(open(result.events_json_path, encoding="utf-8").read())
    assert data["command"] == "중앙에 한 변 10cm짜리 네모를 그려줘"
    assert data["mode"] == "no-api"
    assert data["plan_json_path"] == result.plan_json_path
    assert data["plot_png_path"] is None
    assert data["events"]


def test_run_demo_request_agentic_fake_llm_returns_tool_events(tmp_path):
    result = run_demo_request(
        "draw a short line",
        mode="agentic",
        out_dir=tmp_path,
        llm=FakeToolCallingLLM(one_line_batches()),
        model_name="ignored-when-fake-llm-is-passed",
        request_timeout_s=120.0,
        create_plot=False,
    )

    assert result.plan.diagnostics["validation_ok"] is True
    assert any(event.event_type == "tool_call" for event in result.events)
    assert any(event.event_type == "tool_result" for event in result.events)
    assert any(
        event.event_type == "tool_call" and event.tool_name == "draw_line_to"
        for event in result.events
    )


def test_run_demo_request_agentic_passes_budget_values(tmp_path):
    result = run_demo_request(
        "draw a short line",
        mode="agentic",
        out_dir=tmp_path,
        llm=FakeToolCallingLLM(one_line_batches()),
        max_llm_steps=91,
        max_tool_calls=92,
        create_plot=False,
    )

    assert result.plan.diagnostics["validation_ok"] is True
    assert result.plan.diagnostics["max_llm_steps"] == 91
    assert result.plan.diagnostics["max_tool_calls"] == 92


def test_run_demo_request_agentic_events_json_contains_tool_events(tmp_path):
    result = run_demo_request(
        "draw a short line",
        mode="agentic",
        out_dir=tmp_path,
        llm=FakeToolCallingLLM(one_line_batches()),
        create_plot=False,
    )

    assert result.events_json_path is not None
    text = open(result.events_json_path, encoding="utf-8").read()
    data = json.loads(text)
    event_types = [event["event_type"] for event in data["events"]]
    assert "tool_call" in event_types
    assert "tool_result" in event_types
    assert "OPENAI_API_KEY" not in text


def test_run_demo_request_agentic_streams_events_via_callback(tmp_path):
    streamed = []
    snapshots = []

    result = run_demo_request(
        "draw a short line",
        mode="agentic",
        out_dir=tmp_path,
        llm=FakeToolCallingLLM(one_line_batches()),
        create_plot=True,
        event_callback=streamed.append,
        plan_snapshot_callback=snapshots.append,
    )

    assert result.plan.diagnostics["validation_ok"] is True
    assert any(event.event_type == "tool_call" for event in streamed)
    assert any(event.event_type == "tool_result" for event in streamed)
    assert streamed[-1].event_type == "plot_generated"
    assert snapshots
    assert any(len(snapshot.strokes) == 1 for snapshot in snapshots)


def test_safe_slug_removes_problematic_characters():
    slug = safe_slug("../Draw A: square? 10cm짜리!!")

    assert slug == "draw-a-square-10cm"
    assert "/" not in slug
    assert "." not in slug
    assert "?" not in slug


def test_run_demo_request_no_api_does_not_require_live_openai(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = run_demo_request(
        "중앙에 한 변 10cm짜리 네모를 그려줘",
        mode="no-api",
        out_dir=tmp_path,
        create_plot=False,
    )

    assert result.plan.diagnostics["validation_ok"] is True
    assert result.plan_json_path is not None
