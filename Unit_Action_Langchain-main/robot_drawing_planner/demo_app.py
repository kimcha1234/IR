"""Streamlit chatbot demo for the agentic robot drawing planner.

The app visualizes planned board-frame primitive paths only. It does not
execute a robot or compute IK, FK, Jacobians, joint commands, torque, dynamics,
or Isaac Sim execution.
"""

from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from robot_drawing_planner.agent_events import (
    AgentRunEvent,
    event_to_chat_markdown,
    make_event,
)
from robot_drawing_planner.agent_planner import (
    DEFAULT_AGENTIC_LLM_STEPS,
    DEFAULT_AGENTIC_TOOL_CALLS,
    MAX_AGENTIC_TOOL_CALL_ROUNDS,
)
from robot_drawing_planner.config import DEFAULT_MODEL, DEFAULT_TIMEOUT_SECONDS
from robot_drawing_planner.demo_core import run_demo_request
from robot_drawing_planner.model_catalog import (
    DEFAULT_MODEL_CHOICES,
    get_planner_model_choices,
)
from robot_drawing_planner.schemas import DrawingPlan
from robot_drawing_planner.visualization import plot_drawing_plan

MISSING_API_KEY_MESSAGE = (
    "OPENAI_API_KEY is not set. Make sure it is exported in ~/.zshrc."
)
DEFAULT_GUI_MODEL = os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
CHAT_BOTTOM_ANCHOR_ID = "demo-chat-bottom-anchor"


def load_bytes(path: str | Path) -> bytes:
    """Read a file as bytes for Streamlit downloads and images."""

    return Path(path).read_bytes()


def event_to_streamlit_message(event: AgentRunEvent) -> tuple[str, str]:
    """Return a Streamlit chat role and markdown body for an event."""

    role = "user" if event.event_type == "user_request" else "assistant"
    return role, event_to_chat_markdown(event)


def render_event(event: AgentRunEvent) -> None:
    """Render one sanitized agent event as a chat/timeline message."""

    role, markdown = event_to_streamlit_message(event)
    with st.chat_message(role):
        if event.event_type == "tool_result" and event.ok is False:
            st.error(markdown)
        else:
            st.markdown(markdown)


def tool_event_block_markdown(
    llm_event: AgentRunEvent | None,
    tool_call_event: AgentRunEvent,
    tool_result_event: AgentRunEvent,
) -> str:
    """Return a grouped LLM/tool-call/tool-result block for GUI display."""

    args = json.dumps(
        tool_call_event.tool_args or {},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    status_icon = "✅" if tool_result_event.ok else "❌"
    return "\n\n".join(
        [
            f"### 🔧 Tool Call: `{tool_call_event.tool_name}`",
            "#### Parameters",
            f"```json\n{args}\n```",
            (
                f"{status_icon} **Status:** ok={tool_result_event.ok} · "
                f"{tool_result_event.message}"
            ),
        ]
    )


def render_tool_event_block(
    llm_event: AgentRunEvent | None,
    tool_call_event: AgentRunEvent,
    tool_result_event: AgentRunEvent,
) -> None:
    """Render one grouped tool-call block."""

    with st.chat_message("assistant"):
        with st.container(border=True):
            st.markdown(
                tool_event_block_markdown(
                    llm_event,
                    tool_call_event,
                    tool_result_event,
                )
            )


def _demo_css() -> str:
    return """
<style>
.demo-active-command {
  position: sticky;
  top: 0;
  z-index: 50;
  padding: 12px 14px;
  margin: 0 0 12px 0;
  border: 1px solid rgba(49, 51, 63, 0.18);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
}
.demo-active-command-label {
  display: block;
  margin-bottom: 4px;
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  color: #666;
}
.demo-active-command-text {
  font-size: 1rem;
  font-weight: 650;
  line-height: 1.35;
}
</style>
"""


def _active_command_html(command: str) -> str:
    return (
        '<div class="demo-active-command">'
        '<span class="demo-active-command-label">Current request</span>'
        f'<span class="demo-active-command-text">{escape(command)}</span>'
        "</div>"
    )


def _render_active_command(command: str | None) -> None:
    if command:
        st.markdown(_active_command_html(command), unsafe_allow_html=True)


def _chat_bottom_anchor_html() -> str:
    return f'<div id="{CHAT_BOTTOM_ANCHOR_ID}" style="height: 1px;"></div>'


def _scroll_chat_to_bottom_script() -> str:
    return f"""
<script>
const target = window.parent.document.getElementById("{CHAT_BOTTOM_ANCHOR_ID}");
if (target) {{
  target.scrollIntoView({{behavior: "smooth", block: "end"}});
}}
</script>
"""


def _scroll_chat_to_bottom() -> None:
    components.html(_scroll_chat_to_bottom_script(), height=0, width=0)


def main() -> None:
    """Run the Streamlit chatbot demo."""

    st.set_page_config(page_title="LLM Robot Drawing Planner Demo", layout="wide")
    st.markdown(_demo_css(), unsafe_allow_html=True)
    st.title("LLM Robot Drawing Planner Demo")

    with st.sidebar:
        mode = st.selectbox("Mode", ["agentic", "template", "no-api"], index=0)
        model_choices = st.session_state.setdefault(
            "demo_model_choices",
            _model_choices_with_default(DEFAULT_MODEL_CHOICES),
        )
        if st.button("Refresh models from OpenAI API"):
            if not os.getenv("OPENAI_API_KEY"):
                st.warning(MISSING_API_KEY_MESSAGE)
            else:
                model_choices = _model_choices_with_default(get_planner_model_choices())
                st.session_state["demo_model_choices"] = model_choices
                st.success(f"Loaded {len(model_choices)} model choices.")
        selected_model = st.selectbox(
            "ChatGPT model",
            model_choices,
            index=_model_index(model_choices, DEFAULT_GUI_MODEL),
        )
        custom_model = st.text_input("Custom model override", value="")
        model_name = custom_model.strip() or selected_model
        max_llm_steps = st.number_input(
            "Max LLM steps",
            min_value=1,
            max_value=MAX_AGENTIC_TOOL_CALL_ROUNDS,
            value=DEFAULT_AGENTIC_LLM_STEPS,
        )
        max_tool_calls = st.number_input(
            "Max tool calls",
            min_value=1,
            max_value=MAX_AGENTIC_TOOL_CALL_ROUNDS,
            value=DEFAULT_AGENTIC_TOOL_CALLS,
        )
        request_timeout_s = st.number_input(
            "Request timeout seconds",
            min_value=1.0,
            max_value=600.0,
            value=float(DEFAULT_TIMEOUT_SECONDS),
            step=5.0,
        )
        show_pen_up_moves = st.checkbox("Show pen-up moves", value=False)
        show_raw_json = st.checkbox("Show raw JSON", value=False)
        out_dir = st.text_input("Output directory", value="outputs/demo")
        st.caption("Model temperature is fixed at 0 for deterministic planning.")
        st.caption("Open-ended prompts may need more LLM/tool-call budget.")
        st.caption(
            "Refresh uses OpenAI's Models API; it exposes basic id/owner metadata, "
            "not full tool-calling capability guarantees."
        )
        st.caption("Agentic GUI events stream by default as each tool call completes.")
        st.info(
            "This demo visualizes planned primitive paths only. "
            "It does not execute a robot."
        )

    history = st.session_state.setdefault("demo_chat_history", [])
    chat_col, plot_col = st.columns([0.58, 0.42], gap="large")

    with chat_col:
        st.subheader("Planner timeline")
        chat_scroll = st.container(height=720)
        with chat_scroll:
            _render_active_command(st.session_state.get("demo_active_command"))
            for record in history:
                _render_history_record(record, show_raw_json=show_raw_json)
            live_container = st.container()
            st.markdown(_chat_bottom_anchor_html(), unsafe_allow_html=True)

    with plot_col:
        st.subheader("Live planned path")
        plot_placeholder = st.empty()
        latest_plan = st.session_state.get("demo_latest_live_plan")
        if latest_plan:
            _render_live_plot(
                plot_placeholder,
                latest_plan,
                show_pen_up_moves=show_pen_up_moves,
                caption=_plan_plot_caption(latest_plan),
            )
        else:
            plot_placeholder.info("The planned board-frame path will update here.")

    submitted_command = st.chat_input("Describe what the robot should draw")
    if submitted_command:
        st.session_state["demo_pending_command"] = submitted_command
        st.session_state["demo_active_command"] = submitted_command
        st.rerun()

    command = st.session_state.pop("demo_pending_command", None)
    if not command:
        return

    user_record = {"kind": "user", "content": command}
    history.append(user_record)
    with live_container:
        _render_history_record(user_record, show_raw_json=show_raw_json)
    _scroll_chat_to_bottom()
    st.session_state["demo_latest_live_plan"] = None
    plot_placeholder.info("Waiting for the first planned stroke...")

    if mode == "agentic" and not os.getenv("OPENAI_API_KEY"):
        error_event = make_event(
            event_index=0,
            event_type="error",
            message=MISSING_API_KEY_MESSAGE,
            ok=False,
            metadata={"mode": mode},
        )
        event_record = {"kind": "event", "event": error_event.model_dump(mode="json")}
        history.append(event_record)
        with live_container:
            _render_history_record(event_record, show_raw_json=show_raw_json)
        _scroll_chat_to_bottom()
        return

    streamed_event_keys: set[tuple[int, str, str | None, str]] = set()
    pending_llm_event: AgentRunEvent | None = None
    pending_tool_call_event: AgentRunEvent | None = None

    def stream_plan_snapshot(plan: DrawingPlan) -> None:
        plan_payload = plan.model_dump(mode="json")
        st.session_state["demo_latest_live_plan"] = plan_payload
        _render_live_plot(
            plot_placeholder,
            plan_payload,
            show_pen_up_moves=show_pen_up_moves,
        )

    def stream_event(event: AgentRunEvent) -> None:
        nonlocal pending_llm_event, pending_tool_call_event
        if event.event_type == "user_request":
            return
        if event.event_type == "llm_message":
            pending_llm_event = event
            return
        if event.event_type == "tool_call":
            pending_tool_call_event = event
            return
        if event.event_type == "tool_result" and pending_tool_call_event is not None:
            for grouped_event in [pending_llm_event, pending_tool_call_event, event]:
                if grouped_event is not None:
                    streamed_event_keys.add(_event_key(grouped_event))
            event_record = {
                "kind": "tool_block",
                "llm_event": (
                    pending_llm_event.model_dump(mode="json")
                    if pending_llm_event is not None
                    else None
                ),
                "tool_call_event": pending_tool_call_event.model_dump(mode="json"),
                "tool_result_event": event.model_dump(mode="json"),
            }
            pending_tool_call_event = None
            history.append(event_record)
            with live_container:
                _render_history_record(event_record, show_raw_json=show_raw_json)
            _scroll_chat_to_bottom()
            return

        streamed_event_keys.add(_event_key(event))
        event_record = {"kind": "event", "event": event.model_dump(mode="json")}
        history.append(event_record)
        with live_container:
            _render_history_record(event_record, show_raw_json=show_raw_json)
        _scroll_chat_to_bottom()

    try:
        with st.spinner("Planning symbolic unit actions..."):
            result = run_demo_request(
                command,
                mode=mode,
                out_dir=out_dir,
                model_name=model_name.strip() or None,
                request_timeout_s=float(request_timeout_s),
                max_llm_steps=int(max_llm_steps),
                max_tool_calls=int(max_tool_calls),
                create_plot=True,
                show_pen_up_moves=show_pen_up_moves,
                event_callback=stream_event,
                plan_snapshot_callback=stream_plan_snapshot,
            )
    except Exception as exc:
        error_event = make_event(
            event_index=0,
            event_type="error",
            message=str(exc),
            ok=False,
            metadata={"mode": mode},
        )
        event_record = {"kind": "event", "event": error_event.model_dump(mode="json")}
        history.append(event_record)
        with live_container:
            _render_history_record(event_record, show_raw_json=show_raw_json)
        _scroll_chat_to_bottom()
        return

    for event in result.events:
        if event.event_type == "user_request" or _event_key(event) in streamed_event_keys:
            continue
        event_record = {"kind": "event", "event": event.model_dump(mode="json")}
        history.append(event_record)
        with live_container:
            _render_history_record(event_record, show_raw_json=show_raw_json)
        _scroll_chat_to_bottom()

    result_record = {
        "kind": "result",
        "plan_json_path": result.plan_json_path,
        "plot_png_path": result.plot_png_path,
        "events_json_path": result.events_json_path,
        "plan": result.plan.model_dump(mode="json"),
    }
    st.session_state["demo_latest_live_plan"] = result_record["plan"]
    _render_live_plot(
        plot_placeholder,
        result_record["plan"],
        show_pen_up_moves=show_pen_up_moves,
        caption=_plan_plot_caption(result_record["plan"]),
    )
    history.append(result_record)
    with live_container:
        _render_history_record(result_record, show_raw_json=show_raw_json)
    st.session_state["demo_active_command"] = None
    _scroll_chat_to_bottom()


def _event_key(event: AgentRunEvent) -> tuple[int, str, str | None, str]:
    return (event.event_index, event.event_type, event.tool_name, event.message)


def _model_choices_with_default(choices: list[str]) -> list[str]:
    merged = []
    for model_id in [DEFAULT_GUI_MODEL, *choices]:
        if model_id and model_id not in merged:
            merged.append(model_id)
    return merged


def _model_index(choices: list[str], preferred: str) -> int:
    try:
        return choices.index(preferred)
    except ValueError:
        return 0


def _live_plot_figure(plan: DrawingPlan | dict[str, Any], show_pen_up_moves: bool):
    fig, _ax = plot_drawing_plan(
        plan,
        title="Live planned board-frame path",
        show_pen_up_moves=show_pen_up_moves,
        show_action_labels=True,
    )
    return fig


def _render_live_plot(
    placeholder: Any,
    plan: DrawingPlan | dict[str, Any],
    show_pen_up_moves: bool,
    caption: str = "Updates as unit-action tool calls add planned strokes.",
) -> None:
    from matplotlib import pyplot as plt

    fig = _live_plot_figure(plan, show_pen_up_moves=show_pen_up_moves)
    try:
        with placeholder.container():
            st.pyplot(fig, clear_figure=True)
            st.caption(caption)
    finally:
        plt.close(fig)


def _render_history_record(record: dict[str, Any], show_raw_json: bool) -> None:
    kind = record.get("kind")
    if kind == "user":
        with st.chat_message("user"):
            st.markdown(str(record.get("content", "")))
    elif kind == "event":
        render_event(AgentRunEvent.model_validate(record["event"]))
    elif kind == "tool_block":
        llm_payload = record.get("llm_event")
        render_tool_event_block(
            AgentRunEvent.model_validate(llm_payload) if llm_payload else None,
            AgentRunEvent.model_validate(record["tool_call_event"]),
            AgentRunEvent.model_validate(record["tool_result_event"]),
        )
    elif kind == "result":
        _render_result_record(record, show_raw_json=show_raw_json)


def _render_result_record(record: dict[str, Any], show_raw_json: bool) -> None:
    plan_json_path = record.get("plan_json_path")
    plot_png_path = record.get("plot_png_path")
    events_json_path = record.get("events_json_path")
    plan_payload = record.get("plan") or {}
    status = classify_plan_result(plan_payload)

    with st.chat_message("assistant"):
        if status["validation_ok"]:
            st.success("Final planned board-frame path is ready.")
        else:
            st.error("Planning did not produce an executable final plan.")
            if status["has_strokes"]:
                st.warning(
                    "The image below is a partial preview only and should not be sent to the robot."
                )
        diagnostics_text = result_diagnostics_markdown(plan_payload)
        if diagnostics_text:
            st.markdown(diagnostics_text)
        if plot_png_path and Path(plot_png_path).exists():
            st.caption(
                f"{status['image_caption']} is shown in the right-side live plot panel."
            )
        if plan_json_path and Path(plan_json_path).exists():
            st.download_button(
                "Download plan JSON",
                data=load_bytes(plan_json_path),
                file_name=Path(plan_json_path).name,
                mime="application/json",
            )
        if plot_png_path and Path(plot_png_path).exists():
            st.download_button(
                "Download plot PNG",
                data=load_bytes(plot_png_path),
                file_name=Path(plot_png_path).name,
                mime="image/png",
            )
        if events_json_path and Path(events_json_path).exists():
            st.download_button(
                "Download event log JSON",
                data=load_bytes(events_json_path),
                file_name=Path(events_json_path).name,
                mime="application/json",
            )
        if show_raw_json:
            with st.expander("DrawingPlan JSON"):
                st.json(plan_payload)
        st.caption(
            "Planned primitive path only; no robot execution, IK, FK, Jacobians, "
            "joint commands, torque, dynamics, or Isaac Sim execution."
        )


def classify_plan_result(plan_payload: DrawingPlan | dict[str, Any]) -> dict[str, Any]:
    """Classify a rendered result as executable, partial preview, or failed empty."""

    payload = (
        plan_payload.model_dump(mode="json")
        if isinstance(plan_payload, DrawingPlan)
        else dict(plan_payload or {})
    )
    diagnostics = payload.get("diagnostics") or {}
    validation_ok = bool(diagnostics.get("validation_ok"))
    has_strokes = bool(payload.get("strokes") or [])
    partial_preview = (not validation_ok) and has_strokes
    if validation_ok:
        image_caption = "Final planned board-frame path"
        summary = "final planned board-frame path"
    elif partial_preview:
        image_caption = "Partial planned-path preview — not executable"
        summary = "partial preview only; not executable"
    else:
        image_caption = "No executable planned path"
        summary = "planning failed without drawable preview"
    return {
        "validation_ok": validation_ok,
        "has_strokes": has_strokes,
        "partial_preview": partial_preview,
        "image_caption": image_caption,
        "summary": summary,
    }


def _plan_plot_caption(plan_payload: DrawingPlan | dict[str, Any]) -> str:
    status = classify_plan_result(plan_payload)
    diagnostics = (
        plan_payload.diagnostics
        if isinstance(plan_payload, DrawingPlan)
        else (plan_payload.get("diagnostics") or {})
    )
    if "validation_ok" not in diagnostics:
        return "Updates as unit-action tool calls add planned strokes."
    return str(status["image_caption"])


def result_diagnostics_markdown(plan_payload: DrawingPlan | dict[str, Any]) -> str:
    """Return a concise diagnostics block for the GUI result message."""

    payload = (
        plan_payload.model_dump(mode="json")
        if isinstance(plan_payload, DrawingPlan)
        else dict(plan_payload or {})
    )
    diagnostics = payload.get("diagnostics") or {}
    lines: list[str] = []

    errors = diagnostics.get("errors") or []
    if errors:
        lines.append("**diagnostics.errors**")
        lines.extend(f"- {error}" for error in errors)

    failed_calls = diagnostics.get("failed_calls") or []
    if failed_calls:
        if lines:
            lines.append("")
        lines.append("**diagnostics.failed_calls**")
        lines.extend(f"- {call}" for call in failed_calls)

    if "partial_preview_available" in diagnostics:
        if lines:
            lines.append("")
        lines.append(
            "**diagnostics.partial_preview_available:** "
            f"`{diagnostics['partial_preview_available']}`"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
