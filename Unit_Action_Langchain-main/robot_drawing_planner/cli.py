"""Command line interface for generating drawing plans."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from robot_drawing_planner.agent_events import AgentRunEvent
from robot_drawing_planner.agent_planner import (
    DEFAULT_AGENTIC_LLM_STEPS,
    DEFAULT_AGENTIC_TOOL_CALLS,
    MAX_AGENTIC_TOOL_CALL_ROUNDS,
    plan_drawing_agentic,
)
from robot_drawing_planner.config import DEFAULT_TIMEOUT_SECONDS, PlannerConfig
from robot_drawing_planner.llm_client import get_llm
from robot_drawing_planner.planner import build_plan_from_parsed_goal, plan_drawing
from robot_drawing_planner.schemas import Measurement, ParsedGoal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a natural language drawing command into robot primitive JSON. "
            "Default mode uses LLM agentic planning over Unit Action tools."
        )
    )
    parser.add_argument("command", help="Natural language drawing command.")
    parser.add_argument("--out", help="Optional path where JSON plan will be written.")
    parser.add_argument(
        "--model",
        "--chatgpt-version",
        dest="model",
        help="Optional OpenAI/ChatGPT model version override.",
    )
    parser.add_argument(
        "--max-steps",
        "--max-llm-steps",
        dest="max_llm_steps",
        type=int,
        default=DEFAULT_AGENTIC_LLM_STEPS,
        help=(
            "Maximum LLM response loop steps for agentic mode "
            f"(1-{MAX_AGENTIC_TOOL_CALL_ROUNDS}, default: "
            f"{DEFAULT_AGENTIC_LLM_STEPS})."
        ),
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=DEFAULT_AGENTIC_TOOL_CALLS,
        help=(
            "Maximum unit-action tool calls for agentic mode "
            f"(1-{MAX_AGENTIC_TOOL_CALL_ROUNDS}, default: "
            f"{DEFAULT_AGENTIC_TOOL_CALLS})."
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "Per-request OpenAI timeout in seconds for live agentic/template modes "
            f"(default: {DEFAULT_TIMEOUT_SECONDS:g})."
        ),
    )
    parser.add_argument(
        "--stream-events",
        action="store_true",
        help=(
            "Print agentic planning events to stderr as each tool call/result completes. "
            "JSON plan output still goes to stdout or --out."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["agentic", "template"],
        default="agentic",
        help=(
            "Planning mode. Default 'agentic' uses LLM agentic planning over Unit Action "
            "tools. 'template' uses the old ParsedGoal/template compiler baseline."
        ),
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help=(
            "Use deterministic demo fallback for development/testing only; this never "
            "calls OpenAI and does not exercise production agentic planning."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument("--config", help="Optional JSON PlannerConfig file.")
    parser.add_argument(
        "--plot-out",
        help="Optional path where a planned-path plot image will be written.",
    )
    parser.add_argument(
        "--out-only",
        action="store_true",
        help="Write JSON to --out without printing it to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        is_agentic_mode = not args.no_api and args.mode == "agentic"
        if is_agentic_mode:
            if args.max_llm_steps < 1 or args.max_llm_steps > MAX_AGENTIC_TOOL_CALL_ROUNDS:
                raise ValueError(
                    f"--max-steps/--max-llm-steps must be between 1 and "
                    f"{MAX_AGENTIC_TOOL_CALL_ROUNDS}."
                )
            if args.max_tool_calls < 1 or args.max_tool_calls > MAX_AGENTIC_TOOL_CALL_ROUNDS:
                raise ValueError(
                    f"--max-tool-calls must be between 1 and {MAX_AGENTIC_TOOL_CALL_ROUNDS}."
                )
        if args.request_timeout <= 0:
            raise ValueError("--request-timeout must be positive.")
        config = _load_config(args.config)
        if args.no_api:
            parsed = demo_parse_command(args.command)
            plan = build_plan_from_parsed_goal(parsed, config=config)
        elif args.mode == "template":
            llm = get_llm(args.model, timeout_seconds=args.request_timeout)
            plan = plan_drawing(args.command, config=config, llm=llm)
        else:
            llm = get_llm(args.model, timeout_seconds=args.request_timeout)
            plan = plan_drawing_agentic(
                args.command,
                config=config,
                llm=llm,
                max_llm_steps=args.max_llm_steps,
                max_tool_calls=args.max_tool_calls,
                event_callback=(
                    _print_event_to_stderr if args.stream_events else None
                ),
            )

        json_text = json.dumps(
            plan.model_dump(mode="json"),
            indent=2 if args.pretty else None,
            ensure_ascii=False,
        )
        if args.out:
            output_path = Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json_text + "\n", encoding="utf-8")
        if args.plot_out:
            import matplotlib

            matplotlib.use("Agg", force=True)
            from robot_drawing_planner.visualization import save_plan_plot

            save_plan_plot(plan, args.plot_out, config=config)
        if not args.out_only:
            print(json_text)
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _print_event_to_stderr(event: AgentRunEvent) -> None:
    print(_event_to_terminal_line(event), file=sys.stderr, flush=True)


def _event_to_terminal_line(event: AgentRunEvent) -> str:
    """Format one planner event as a single stderr line."""

    if event.event_type == "tool_call":
        args = json.dumps(event.tool_args or {}, ensure_ascii=False, sort_keys=True)
        return f"[tool_call] {event.tool_name} args={args}"
    if event.event_type == "tool_result":
        return f"[tool_result] {event.tool_name} ok={event.ok} message={_one_line(event.message)}"
    if event.event_type == "llm_message":
        count = event.metadata.get("tool_call_count")
        return f"[llm_message] tool_calls={count} message={_one_line(event.message)}"
    if event.event_type == "plan_finished":
        return f"[plan_finished] {_one_line(event.message)}"
    if event.event_type == "error":
        return f"[error] {_one_line(event.message)}"
    return f"[{event.event_type}] {_one_line(event.message)}"


def _one_line(value: Any) -> str:
    return " ".join(str(value).split())


def demo_parse_command(command: str) -> ParsedGoal:
    """Small deterministic parser for tests and demos; production uses LangChain."""

    text = command.strip()
    lowered = text.lower()
    shape_type = _infer_shape_type(text)
    measurement = _extract_measurement(text)
    letter = _extract_letter(text) if shape_type == "letter" else None
    position_hint = _extract_position_hint(text)

    kwargs: dict[str, Any] = {
        "shape_type": shape_type,
        "center": None,
        "radius": None,
        "side_length": None,
        "size": None,
        "orientation_deg": 0.0,
        "letter": letter,
        "position_hint": position_hint,
        "raw_command": command,
    }

    if shape_type == "circle":
        if measurement is None:
            kwargs["radius"] = Measurement(value=5, unit="cm")
        elif _mentions_radius(text):
            kwargs["radius"] = measurement
        elif _mentions_diameter_or_size(text):
            kwargs["size"] = measurement
        else:
            kwargs["radius"] = measurement
    elif shape_type in {"square", "triangle"}:
        if measurement is None:
            kwargs["side_length"] = Measurement(value=10, unit="cm")
        elif _mentions_size(text) and not _mentions_side_length(text):
            kwargs["size"] = measurement
        else:
            kwargs["side_length"] = measurement
    elif shape_type == "letter":
        kwargs["size"] = measurement or Measurement(value=10, unit="cm")

    return ParsedGoal(**kwargs)


def _load_config(path: str | None) -> PlannerConfig:
    if path is None:
        return PlannerConfig()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return PlannerConfig.model_validate(payload)


def _infer_shape_type(command: str) -> str:
    lowered = command.lower()
    if any(token in command for token in ["원", "동그라미"]) or "circle" in lowered:
        return "circle"
    if any(token in command for token in ["네모", "사각형", "정사각형"]) or "square" in lowered:
        return "square"
    if any(token in command for token in ["세모", "삼각형", "정삼각형"]) or "triangle" in lowered:
        return "triangle"
    if any(token in command for token in ["글자", "문자", "알파벳"]) or re.search(r"\bletter\b", lowered):
        return "letter"
    if re.search(r"\b[AHLTO]\b", command.upper()):
        return "letter"
    raise ValueError("Could not infer supported shape in --no-api mode.")


def _extract_letter(command: str) -> str:
    match = re.search(r"(?<![A-Z])([AHLTO])(?![A-Z])", command.upper())
    if match:
        return match.group(1)
    for letter in ["A", "H", "L", "T", "O"]:
        if f"letter {letter}".lower() in command.lower():
            return letter
    raise ValueError("--no-api letter commands must include one of A, H, L, T, O.")


def _extract_position_hint(command: str) -> str | None:
    lowered = command.lower()
    if "중앙" in command:
        return "중앙"
    if "center" in lowered:
        return "center"
    if "middle" in lowered:
        return "middle"
    return None


def _extract_measurement(command: str) -> Measurement | None:
    match = re.search(
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m)(?=$|[\s,.;:)\]]|[가-힣])",
        command,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = float(match.group("value"))
    unit = match.group("unit").lower()
    if unit not in {"m", "cm", "mm"}:
        return None
    return Measurement(value=value, unit=unit)


def _mentions_radius(command: str) -> bool:
    lowered = command.lower()
    return "반지름" in command or "radius" in lowered


def _mentions_diameter_or_size(command: str) -> bool:
    lowered = command.lower()
    return any(token in command for token in ["지름", "크기"]) or any(
        token in lowered for token in ["diameter", "size"]
    )


def _mentions_side_length(command: str) -> bool:
    lowered = command.lower()
    return any(token in command for token in ["한 변", "변 길이"]) or any(
        token in lowered for token in ["side length", "side"]
    )


def _mentions_size(command: str) -> bool:
    lowered = command.lower()
    return "크기" in command or "size" in lowered


if __name__ == "__main__":
    raise SystemExit(main())
