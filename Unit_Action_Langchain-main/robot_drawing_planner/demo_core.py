"""Core helpers for the chatbot-style planner demo.

This module orchestrates planner calls, JSON export, and planned-path plotting.
It does not compute IK, FK, Jacobians, joint commands, torques, robot dynamics,
or Isaac Sim execution.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from robot_drawing_planner.agent_events import AgentRunEvent, AgentRunResult, make_event
from robot_drawing_planner.agent_planner import (
    DEFAULT_AGENTIC_LLM_STEPS,
    DEFAULT_AGENTIC_TOOL_CALLS,
    MAX_AGENTIC_TOOL_CALL_ROUNDS,
    plan_drawing_agentic,
)
from robot_drawing_planner.cli import demo_parse_command
from robot_drawing_planner.config import DEFAULT_CONFIG, PlannerConfig
from robot_drawing_planner.llm_client import get_llm
from robot_drawing_planner.planner import build_plan_from_parsed_goal, plan_drawing
from robot_drawing_planner.schemas import DrawingPlan
from robot_drawing_planner.visualization import save_plan_plot


def run_demo_request(
    command: str,
    mode: Literal["agentic", "template", "no-api"] = "agentic",
    out_dir: str | Path = "outputs/demo",
    config: PlannerConfig | None = None,
    llm: Any | None = None,
    model_name: str | None = None,
    request_timeout_s: float | None = None,
    max_steps: int | None = None,
    max_llm_steps: int = DEFAULT_AGENTIC_LLM_STEPS,
    max_tool_calls: int = DEFAULT_AGENTIC_TOOL_CALLS,
    create_plot: bool = True,
    show_pen_up_moves: bool = False,
    event_callback: Callable[[AgentRunEvent], None] | None = None,
    plan_snapshot_callback: Callable[[DrawingPlan], None] | None = None,
) -> AgentRunResult:
    """Run a planner demo request and save plan JSON plus optional plot PNG."""

    planner_config = config or DEFAULT_CONFIG
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    base_name = f"{timestamp}-{safe_slug(command)}"
    plan_path = output_dir / f"{base_name}.json"
    plot_path = output_dir / f"{base_name}.png"
    events_path = output_dir / f"{base_name}_events.json"

    if mode == "agentic":
        live_llm = (
            llm
            if llm is not None
            else get_llm(model_name, timeout_seconds=request_timeout_s)
        )
        agentic_result = plan_drawing_agentic(
            command,
            config=planner_config,
            llm=live_llm,
            max_steps=max_steps,
            max_llm_steps=max_llm_steps,
            max_tool_calls=max_tool_calls,
            event_callback=event_callback,
            collect_events=True,
            plan_snapshot_callback=plan_snapshot_callback,
        )
        if not isinstance(agentic_result, AgentRunResult):
            raise TypeError("agentic planner did not return AgentRunResult")
        plan = agentic_result.plan
        events = list(agentic_result.events)
    elif mode == "template":
        live_llm = (
            llm
            if llm is not None
            else get_llm(model_name, timeout_seconds=request_timeout_s)
        )
        plan = plan_drawing(command, config=planner_config, llm=live_llm)
        events = _baseline_events(command, plan, mode="template")
    elif mode == "no-api":
        parsed_goal = demo_parse_command(command)
        plan = build_plan_from_parsed_goal(parsed_goal, config=planner_config)
        events = _baseline_events(command, plan, mode="no-api")
    else:
        raise ValueError(f"Unsupported demo mode: {mode}")

    if mode != "agentic" and event_callback is not None:
        for event in events:
            event_callback(event)
    if mode != "agentic" and plan_snapshot_callback is not None:
        plan_snapshot_callback(plan)

    write_plan_json(plan, plan_path)
    plot_png_path: str | None = None
    if create_plot:
        saved_plot = save_plan_plot(
            plan,
            plot_path,
            config=planner_config,
            title="Planned board-frame path",
            show_pen_up_moves=show_pen_up_moves,
        )
        plot_png_path = str(saved_plot)
        events.append(
            make_event(
                event_index=len(events),
                event_type="plot_generated",
                message=f"Saved planned-path plot to {saved_plot}.",
                ok=True,
                metadata={"plot_png_path": str(saved_plot)},
            )
        )
        if event_callback is not None:
            event_callback(events[-1])

    write_events_json(
        command=command,
        mode=mode,
        events=events,
        path=events_path,
        plan_json_path=str(plan_path),
        plot_png_path=plot_png_path,
    )

    return AgentRunResult(
        command=command,
        plan=plan,
        events=events,
        plan_json_path=str(plan_path),
        plot_png_path=plot_png_path,
        events_json_path=str(events_path),
    )


def safe_slug(text: str, max_len: int = 48) -> str:
    """Create a conservative filename slug from user text."""

    lowered = text.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered)
    slug = re.sub(r"-+", "-", slug).strip("-")
    if not slug:
        slug = "request"
    return slug[:max_len].strip("-") or "request"


def write_plan_json(plan: DrawingPlan, path: str | Path) -> Path:
    """Write a DrawingPlan JSON file and return its path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(plan.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_events_json(
    command: str,
    mode: str,
    events: list[AgentRunEvent],
    path: str | Path,
    plan_json_path: str | None,
    plot_png_path: str | None,
) -> Path:
    """Write sanitized agent event log JSON and return its path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "command": command,
        "mode": mode,
        "plan_json_path": plan_json_path,
        "plot_png_path": plot_png_path,
        "events": [event.model_dump(mode="json") for event in events],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _baseline_events(
    command: str,
    plan: DrawingPlan,
    mode: Literal["template", "no-api"],
) -> list[AgentRunEvent]:
    validation_ok = bool(plan.diagnostics.get("validation_ok"))
    return [
        make_event(
            event_index=0,
            event_type="user_request",
            message=command,
            ok=True,
            metadata={"mode": mode},
        ),
        make_event(
            event_index=1,
            event_type="plan_finished",
            message=f"{mode} planner produced a DrawingPlan.",
            ok=validation_ok,
            metadata={
                "mode": mode,
                "action_count": len(plan.actions),
                "stroke_count": len(plan.strokes),
            },
        ),
    ]
