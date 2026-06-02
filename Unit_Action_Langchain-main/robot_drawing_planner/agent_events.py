"""Structured event models for chatbot-style agent planner demos."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from robot_drawing_planner.schemas import DrawingPlan

AgentEventType = Literal[
    "user_request",
    "llm_message",
    "tool_call",
    "tool_result",
    "plan_finished",
    "plot_generated",
    "error",
]


class AgentRunEvent(BaseModel):
    """One sanitized event in a planner demo timeline."""

    model_config = ConfigDict(extra="forbid")

    event_index: int = Field(ge=0, description="Monotonic event index.")
    step_index: int | None = Field(default=None, description="LLM loop step index.")
    event_type: AgentEventType = Field(description="Timeline event type.")
    tool_name: str | None = Field(default=None, description="Tool name for tool events.")
    tool_args: dict[str, Any] | None = Field(
        default=None,
        description="JSON-serializable tool arguments.",
    )
    tool_result: dict[str, Any] | None = Field(
        default=None,
        description="JSON-serializable tool result.",
    )
    message: str = Field(description="Human-readable sanitized event message.")
    ok: bool | None = Field(default=None, description="Whether the event succeeded.")
    timestamp: str = Field(description="ISO-format timestamp.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-serializable event metadata.",
    )

    @field_validator("tool_args", "tool_result", "metadata")
    @classmethod
    def require_json_serializable_dict(
        cls,
        value: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        try:
            json.dumps(value, ensure_ascii=False)
        except TypeError as exc:
            raise ValueError("event dictionaries must be JSON-serializable") from exc
        return value


class AgentRunResult(BaseModel):
    """Sanitized result bundle for the chatbot-style GUI demo."""

    model_config = ConfigDict(extra="forbid")

    command: str = Field(description="Original natural-language command.")
    plan: DrawingPlan = Field(description="Final DrawingPlan JSON model.")
    events: list[AgentRunEvent] = Field(description="Sanitized agent timeline events.")
    plan_json_path: str | None = Field(
        default=None,
        description="Optional saved plan JSON path.",
    )
    plot_png_path: str | None = Field(
        default=None,
        description="Optional saved plot PNG path.",
    )
    events_json_path: str | None = Field(
        default=None,
        description="Optional saved sanitized event log JSON path.",
    )


def make_event(
    event_index: int,
    event_type: AgentEventType,
    message: str,
    step_index: int | None = None,
    tool_name: str | None = None,
    tool_args: dict[str, Any] | None = None,
    tool_result: dict[str, Any] | None = None,
    ok: bool | None = None,
    metadata: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> AgentRunEvent:
    """Create a sanitized event with an ISO timestamp."""

    return AgentRunEvent(
        event_index=event_index,
        step_index=step_index,
        event_type=event_type,
        tool_name=tool_name,
        tool_args=tool_args,
        tool_result=tool_result,
        message=message,
        ok=ok,
        timestamp=timestamp or datetime.now(UTC).isoformat(),
        metadata=metadata or {},
    )


def event_to_chat_markdown(event: AgentRunEvent) -> str:
    """Format one event for display in a chat/timeline UI."""

    if event.event_type == "tool_call":
        args = json.dumps(event.tool_args or {}, ensure_ascii=False, sort_keys=True)
        return f"🔧 Tool Call: {event.tool_name}\nargs: {args}"
    if event.event_type == "tool_result":
        marker = "✅" if event.ok else "❌"
        return f"{marker} Tool Result: ok={event.ok}\nmessage: {event.message}"
    if event.event_type == "user_request":
        return f"User request: {event.message}"
    if event.event_type == "llm_message":
        return f"LLM: {event.message}"
    if event.event_type == "plan_finished":
        return f"Plan finished: {event.message}"
    if event.event_type == "plot_generated":
        return f"Plot generated: {event.message}"
    if event.event_type == "error":
        return f"Error: {event.message}"
    return event.message
