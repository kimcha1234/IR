"""CLI orchestration for natural-language command to Isaac execution."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from llm_isaac_bridge.isaac_command import build_isaac_command, run_isaac_command
from llm_isaac_bridge.paths import (
    DEFAULT_ISAAC_OUTPUT_DIR,
    DEFAULT_LLM_ROOT,
    DEFAULT_PLAN_DIR,
    DEFAULT_PLANNER_CONFIG,
    DEFAULT_REPORT_DIR,
    ISAACLAB_ROOT,
)
from llm_isaac_bridge.planner import run_llm_planner
from llm_isaac_bridge.validation import validate_plan_file, write_validation_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an LLM DrawingPlan, validate it, and optionally run Isaac JADE."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--command", help="Natural-language drawing command.")
    source.add_argument("--plan", help="Existing DrawingPlan JSON file to validate/run.")
    parser.add_argument("--planner-mode", choices=["no-api", "template", "agentic"], default="no-api")
    parser.add_argument("--planner-python", help="Python executable used for the LLM planner subprocess.")
    parser.add_argument("--llm-root", default=str(DEFAULT_LLM_ROOT), help="Path to Unit_Action_Langchain-main.")
    parser.add_argument("--planner-config", default=str(DEFAULT_PLANNER_CONFIG), help="Planner config JSON.")
    parser.add_argument("--model", help="OpenAI model override for template/agentic modes.")
    parser.add_argument("--max-llm-steps", type=int, help="Agentic planner LLM step budget.")
    parser.add_argument("--max-tool-calls", type=int, help="Agentic planner tool-call budget.")
    parser.add_argument("--request-timeout", type=float, help="LLM request timeout in seconds.")
    parser.add_argument("--output-root", default=None, help="Root folder for generated plan/report/Isaac outputs.")
    parser.add_argument("--plan-out", help="Explicit generated plan JSON path.")
    parser.add_argument("--report-out", help="Explicit validation report JSON path.")
    parser.add_argument("--isaac-out", help="Explicit Isaac output folder.")
    parser.add_argument("--isaaclab-root", default=str(ISAACLAB_ROOT), help="Path to IsaacLab root.")
    parser.add_argument("--mode", choices=["hover", "contact"], default="contact", help="JADE Isaac execution mode.")
    parser.add_argument("--setup-only", action="store_true", help="Pass --setup-only to run_jade_isaac.py.")
    parser.add_argument("--execute-isaac", action="store_true", help="Actually launch Isaac after validation.")
    parser.add_argument("--print-isaac-command", action="store_true", help="Print the Isaac command even if not executing.")
    parser.add_argument("--stream-planner-events", action="store_true", help="Forward planner event stream to stderr.")
    parser.add_argument("--debug-draw", action="store_true", help="Pass --debug-draw to the Isaac runner.")
    parser.add_argument(
        "--start-delay-s",
        type=_nonnegative_float,
        default=0.0,
        help="Pass a start delay to the Isaac runner so screen recording can be prepared.",
    )
    parser.add_argument("--keep-open", action="store_true", help="Keep Isaac Sim open after execution finishes.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    slug = _slug(args.command or Path(args.plan).stem)
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else None

    plan_path = Path(args.plan).expanduser().resolve() if args.plan else None
    if plan_path is None:
        plan_path = (
            Path(args.plan_out).expanduser().resolve()
            if args.plan_out
            else _rooted(output_root, DEFAULT_PLAN_DIR) / f"{stamp}_{slug}.json"
        )
        result = run_llm_planner(
            args.command,
            output_path=plan_path,
            planner_mode=args.planner_mode,
            llm_root=args.llm_root,
            planner_config=args.planner_config,
            planner_python=args.planner_python,
            model=args.model,
            max_llm_steps=args.max_llm_steps,
            max_tool_calls=args.max_tool_calls,
            request_timeout_s=args.request_timeout,
            stream_events=args.stream_planner_events,
        )
        if result.returncode != 0:
            print("[ERROR] LLM planner failed.", file=sys.stderr)
            print(result.stderr or result.stdout, file=sys.stderr)
            return result.returncode or 1
        print(f"[INFO] plan generated: {result.output_path}", flush=True)

    report = validate_plan_file(plan_path, planner_config=args.planner_config)
    report_path = (
        Path(args.report_out).expanduser().resolve()
        if args.report_out
        else _rooted(output_root, DEFAULT_REPORT_DIR) / f"{stamp}_{slug}_validation.json"
    )
    write_validation_report(report, report_path)
    print(f"[INFO] validation report: {report_path}", flush=True)
    if report.warnings:
        for warning in report.warnings:
            print(f"[WARNING] {warning}", flush=True)
    if not report.ok:
        print("[ERROR] plan validation failed:", file=sys.stderr)
        for error in report.errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    print(
        f"[INFO] validation ok: actions={report.action_count}, "
        f"drawable_actions={report.drawable_action_count}",
        flush=True,
    )

    isaac_out = (
        Path(args.isaac_out).expanduser().resolve()
        if args.isaac_out
        else _rooted(output_root, DEFAULT_ISAAC_OUTPUT_DIR) / f"{stamp}_{slug}_{args.mode}"
    )
    isaac_command = build_isaac_command(
        plan_path=plan_path,
        output_dir=isaac_out,
        mode=args.mode,
        isaaclab_root=args.isaaclab_root,
        setup_only=args.setup_only,
        extra_args=_isaac_extra_args(args),
    )
    if args.print_isaac_command or not args.execute_isaac:
        print("[INFO] Isaac command:", flush=True)
        print(isaac_command.shell_text(), flush=True)
    if not args.execute_isaac:
        print("[INFO] Isaac execution was not launched. Add --execute-isaac to run it.", flush=True)
        return 0

    completed = run_isaac_command(isaac_command)
    print(completed.stdout, end="")
    return int(completed.returncode)


def _slug(text: str) -> str:
    lowered = text.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:48].strip("-") or "request"


def _rooted(output_root: Path | None, default_path: Path) -> Path:
    if output_root is None:
        return default_path
    suffix = default_path.name
    return output_root / suffix


def _isaac_extra_args(args: argparse.Namespace) -> list[str]:
    extra: list[str] = []
    if args.debug_draw:
        extra.append("--debug-draw")
    if args.start_delay_s > 0.0:
        extra.extend(["--start-delay-s", str(float(args.start_delay_s))])
    if args.keep_open:
        extra.append("--keep-open")
    return extra


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
