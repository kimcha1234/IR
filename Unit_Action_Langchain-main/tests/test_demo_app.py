import importlib
import inspect
import sys

from robot_drawing_planner.agent_events import make_event


def import_demo_app_fresh(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    sys.modules.pop("robot_drawing_planner.demo_app", None)
    return importlib.import_module("robot_drawing_planner.demo_app")


def test_import_demo_app_without_openai_api_key(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)

    assert module.MISSING_API_KEY_MESSAGE == (
        "OPENAI_API_KEY is not set. Make sure it is exported in ~/.zshrc."
    )


def test_event_to_streamlit_message_formats_tool_call(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)
    event = make_event(
        event_index=0,
        event_type="tool_call",
        message="Calling move_to_start",
        tool_name="move_to_start",
        tool_args={"x": 0.0, "y": 0.0},
    )

    role, markdown = module.event_to_streamlit_message(event)

    assert role == "assistant"
    assert "move_to_start" in markdown
    assert '"x": 0.0' in markdown


def test_event_to_streamlit_message_formats_user_request(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)
    event = make_event(
        event_index=0,
        event_type="user_request",
        message="draw a square",
    )

    role, markdown = module.event_to_streamlit_message(event)

    assert role == "user"
    assert "draw a square" in markdown


def test_load_bytes_reads_temp_file(tmp_path, monkeypatch):
    module = import_demo_app_fresh(monkeypatch)
    path = tmp_path / "payload.bin"
    path.write_bytes(b"demo-bytes")

    assert module.load_bytes(path) == b"demo-bytes"


def test_main_function_exists(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)

    assert callable(module.main)


def test_demo_app_exposes_default_gui_model(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)

    assert module.DEFAULT_GUI_MODEL


def test_demo_app_uses_1000_max_tool_call_rounds(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)

    assert module.MAX_AGENTIC_TOOL_CALL_ROUNDS == 1000
    assert module.DEFAULT_AGENTIC_LLM_STEPS == 80
    assert module.DEFAULT_AGENTIC_TOOL_CALLS == 200


def test_demo_app_exposes_budget_controls(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)
    source = inspect.getsource(module.main)

    assert "Max LLM steps" in source
    assert "Max tool calls" in source
    assert "Open-ended prompts may need more LLM/tool-call budget." in source
    assert "max_llm_steps=int(max_llm_steps)" in source
    assert "max_tool_calls=int(max_tool_calls)" in source


def test_demo_app_model_choice_helpers(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)

    choices = module._model_choices_with_default(["gpt-4o-mini", module.DEFAULT_GUI_MODEL])

    assert choices[0] == module.DEFAULT_GUI_MODEL
    assert choices.count(module.DEFAULT_GUI_MODEL) == 1
    assert module._model_index(choices, "gpt-4o-mini") == choices.index("gpt-4o-mini")
    assert module._model_index(choices, "missing-model") == 0


def test_active_command_html_is_sticky_and_escapes_text(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)

    css = module._demo_css()
    html = module._active_command_html('<script>alert("x")</script>')

    assert "position: sticky" in css
    assert "Current request" in html
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_scroll_to_bottom_helpers_reference_anchor(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)

    anchor = module._chat_bottom_anchor_html()
    script = module._scroll_chat_to_bottom_script()

    assert module.CHAT_BOTTOM_ANCHOR_ID in anchor
    assert module.CHAT_BOTTOM_ANCHOR_ID in script
    assert "scrollIntoView" in script


def test_demo_app_event_key_is_stable(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)
    event = make_event(
        event_index=3,
        event_type="tool_result",
        message="draw_line appended.",
        tool_name="draw_line_to",
        ok=True,
    )

    assert module._event_key(event) == (
        3,
        "tool_result",
        "draw_line_to",
        "draw_line appended.",
    )


def test_tool_event_block_markdown_groups_llm_tool_and_status(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)
    llm_event = make_event(
        event_index=0,
        event_type="llm_message",
        message="LLM response requested 1 tool call(s).",
        metadata={"tool_call_count": 1},
    )
    tool_call_event = make_event(
        event_index=1,
        event_type="tool_call",
        message="Calling unit-action tool 'begin_plan'.",
        tool_name="begin_plan",
        tool_args={
            "source_command": "Draw a circle with center at (0,0) radius 0.05 meters"
        },
    )
    tool_result_event = make_event(
        event_index=2,
        event_type="tool_result",
        message="Plan started.",
        tool_name="begin_plan",
        ok=True,
    )

    markdown = module.tool_event_block_markdown(
        llm_event,
        tool_call_event,
        tool_result_event,
    )

    assert "LLM response requested" not in markdown
    assert "### 🔧 Tool Call: `begin_plan`" in markdown
    assert "#### Parameters" in markdown
    assert "source_command" in markdown
    assert "Draw a circle with center at (0,0) radius 0.05 meters" in markdown
    assert "✅ **Status:** ok=True · Plan started." in markdown


def test_live_plot_figure_returns_matplotlib_figure(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)
    plan = {
        "strokes": [
            {
                "type": "line",
                "stroke_id": "stroke_001",
                "start": {"x": 0.0, "y": 0.0, "unit": "m"},
                "end": {"x": 0.05, "y": 0.0, "unit": "m"},
            }
        ],
        "actions": [],
    }

    fig = module._live_plot_figure(plan, show_pen_up_moves=False)

    try:
        assert fig.axes
        assert fig.axes[0].get_title() == "Live planned board-frame path"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_demo_app_result_record_supports_events_json_path(monkeypatch, tmp_path):
    module = import_demo_app_fresh(monkeypatch)
    event_path = tmp_path / "events.json"
    event_path.write_text("{}", encoding="utf-8")
    record = {
        "kind": "result",
        "plan_json_path": None,
        "plot_png_path": None,
        "events_json_path": str(event_path),
        "plan": {},
    }

    assert record["events_json_path"] == str(event_path)
    assert module.load_bytes(record["events_json_path"]) == b"{}"


def test_classify_plan_result_marks_valid_plan_as_final(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)
    status = module.classify_plan_result(
        {
            "strokes": [
                {
                    "type": "line",
                    "stroke_id": "stroke_001",
                    "start": {"x": 0.0, "y": 0.0, "unit": "m"},
                    "end": {"x": 0.05, "y": 0.0, "unit": "m"},
                }
            ],
            "diagnostics": {"validation_ok": True},
        }
    )

    assert status["validation_ok"] is True
    assert "final planned" in status["summary"]
    assert status["image_caption"] == "Final planned board-frame path"


def test_classify_plan_result_marks_invalid_strokes_as_partial_preview(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)
    plan = {
        "strokes": [
            {
                "type": "line",
                "stroke_id": "stroke_001",
                "start": {"x": 0.0, "y": 0.0, "unit": "m"},
                "end": {"x": 0.05, "y": 0.0, "unit": "m"},
            }
        ],
        "diagnostics": {
            "validation_ok": False,
            "errors": ["Agentic planning failed: max_llm_steps 1 exceeded."],
            "failed_calls": ["draw_line_to rejected: outside board"],
            "partial_preview_available": True,
        },
    }

    status = module.classify_plan_result(plan)
    diagnostics_text = module.result_diagnostics_markdown(plan)

    assert status["validation_ok"] is False
    assert status["partial_preview"] is True
    assert "partial preview" in status["summary"]
    assert status["image_caption"] == "Partial planned-path preview — not executable"
    assert "diagnostics.errors" in diagnostics_text
    assert "diagnostics.failed_calls" in diagnostics_text
    assert "diagnostics.partial_preview_available" in diagnostics_text


def test_plan_plot_caption_uses_final_or_partial_text(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)

    assert (
        module._plan_plot_caption(
            {"strokes": [], "diagnostics": {"validation_ok": True}}
        )
        == "Final planned board-frame path"
    )
    assert (
        module._plan_plot_caption(
            {
                "strokes": [
                    {
                        "type": "line",
                        "stroke_id": "stroke_001",
                        "start": {"x": 0.0, "y": 0.0, "unit": "m"},
                        "end": {"x": 0.05, "y": 0.0, "unit": "m"},
                    }
                ],
                "diagnostics": {"validation_ok": False},
            }
        )
        == "Partial planned-path preview — not executable"
    )


def test_demo_app_import_does_not_execute_planner(monkeypatch):
    import robot_drawing_planner.demo_core as demo_core

    def fail_if_called(*args, **kwargs):
        raise AssertionError("planner should not run during demo_app import")

    monkeypatch.setattr(demo_core, "run_demo_request", fail_if_called)
    module = import_demo_app_fresh(monkeypatch)

    assert module.run_demo_request is fail_if_called
