"""Prepare or run a fixed API-free demo command suite."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_isaac_bridge.isaac_command import build_isaac_command, run_isaac_command
from llm_isaac_bridge.paths import BRIDGE_ROOT, DEFAULT_PLANNER_CONFIG, ISAACLAB_ROOT
from llm_isaac_bridge.planner import run_llm_planner
from llm_isaac_bridge.validation import validate_plan_file, write_validation_report

DEFAULT_DEMO_COMMANDS = BRIDGE_ROOT / "demo_commands.json"
DEFAULT_DEMO_OUTPUT_ROOT = BRIDGE_ROOT.parent / "outputs" / "demo_showcase"


@dataclass(frozen=True)
class DemoCaseResult:
    """One prepared demo case."""

    case_id: str
    title: str
    command: str
    plan_path: str
    validation_report_path: str
    isaac_output_dir: str
    isaac_command: str
    validation_ok: bool
    errors: list[str]
    warnings: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands", default=str(DEFAULT_DEMO_COMMANDS), help="Demo command JSON file.")
    parser.add_argument("--output-root", default=str(DEFAULT_DEMO_OUTPUT_ROOT), help="Output root for plans/reports/runs.")
    parser.add_argument("--case", action="append", default=None, help="Run only matching case id. Can be repeated.")
    parser.add_argument("--mode", choices=["hover", "contact"], default="contact", help="JADE Isaac mode.")
    parser.add_argument("--planner-mode", choices=["no-api", "template", "agentic"], default="no-api")
    parser.add_argument("--planner-python", help="Python executable for template/agentic planner subprocess.")
    parser.add_argument("--planner-config", default=str(DEFAULT_PLANNER_CONFIG), help="Planner/board config JSON.")
    parser.add_argument("--isaaclab-root", default=str(ISAACLAB_ROOT), help="Path to IsaacLab root.")
    parser.add_argument("--execute-isaac", action="store_true", help="Launch Isaac for each selected case.")
    parser.add_argument("--setup-only", action="store_true", help="Pass --setup-only to run_jade_isaac.py.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = _load_cases(args.commands)
    selected = set(args.case or [])
    if selected:
        cases = [case for case in cases if case["id"] in selected]
    if not cases:
        print("[ERROR] no demo cases selected.", file=sys.stderr)
        return 2

    output_root = Path(args.output_root).expanduser().resolve()
    plan_dir = output_root / "plans"
    report_dir = output_root / "reports"
    run_dir = output_root / "isaac_runs"
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[DemoCaseResult] = []
    for case in cases:
        case_id = str(case["id"])
        command = str(case["command"])
        title = str(case.get("title", case_id))
        plan_path = plan_dir / f"{case_id}.json"
        report_path = report_dir / f"{case_id}_validation.json"
        isaac_out = run_dir / case_id

        planner_result = run_llm_planner(
            command,
            output_path=plan_path,
            planner_mode=args.planner_mode,
            planner_python=args.planner_python,
            planner_config=args.planner_config,
        )
        if planner_result.returncode != 0:
            print(f"[ERROR] planner failed for {case_id}: {planner_result.stderr or planner_result.stdout}", file=sys.stderr)
            return planner_result.returncode or 1

        report = validate_plan_file(plan_path, planner_config=args.planner_config)
        write_validation_report(report, report_path)
        isaac_command = build_isaac_command(
            plan_path=plan_path,
            output_dir=isaac_out,
            mode=args.mode,
            isaaclab_root=args.isaaclab_root,
            setup_only=args.setup_only,
        )
        result = DemoCaseResult(
            case_id=case_id,
            title=title,
            command=command,
            plan_path=str(plan_path),
            validation_report_path=str(report_path),
            isaac_output_dir=str(isaac_out),
            isaac_command=isaac_command.shell_text(),
            validation_ok=report.ok,
            errors=list(report.errors),
            warnings=list(report.warnings),
        )
        results.append(result)
        status = "OK" if report.ok else "FAIL"
        print(f"[{status}] {case_id}: {command}", flush=True)
        print(f"  plan: {plan_path}", flush=True)
        print(f"  report: {report_path}", flush=True)
        print(f"  isaac: {isaac_command.shell_text()}", flush=True)
        if not report.ok:
            for error in report.errors:
                print(f"  error: {error}", flush=True)
            return 2

        if args.execute_isaac:
            completed = run_isaac_command(isaac_command)
            print(completed.stdout, end="")
            if completed.returncode != 0:
                return int(completed.returncode)

    _write_manifest(output_root / "manifest.json", results)
    _write_command_script(output_root / "isaac_commands.sh", results)
    print(f"[INFO] manifest: {output_root / 'manifest.json'}", flush=True)
    print(f"[INFO] command script: {output_root / 'isaac_commands.sh'}", flush=True)
    return 0


def _load_cases(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = payload["cases"] if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise ValueError("Demo commands JSON must contain a list or a {'cases': [...]} object.")
    for case in cases:
        if not isinstance(case, dict) or "id" not in case or "command" not in case:
            raise ValueError("Each demo case must contain 'id' and 'command'.")
    return cases


def _write_manifest(path: Path, results: list[DemoCaseResult]) -> None:
    path.write_text(
        json.dumps({"cases": [asdict(result) for result in results]}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_command_script(path: Path, results: list[DemoCaseResult]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for result in results:
        lines.append(f"# {result.case_id}: {result.command}")
        lines.append(result.isaac_command)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


if __name__ == "__main__":
    raise SystemExit(main())
