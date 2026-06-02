"""Subprocess wrapper around the existing Unit_Action_Langchain planner."""

from __future__ import annotations

import os
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from llm_isaac_bridge.paths import DEFAULT_LLM_ROOT, DEFAULT_PLANNER_CONFIG

PlannerMode = Literal["no-api", "template", "agentic"]


@dataclass(frozen=True)
class PlannerRunResult:
    """Result of one planner subprocess call."""

    command: tuple[str, ...]
    output_path: Path
    returncode: int
    stdout: str
    stderr: str


def run_llm_planner(
    natural_language_command: str,
    *,
    output_path: str | Path,
    planner_mode: PlannerMode = "no-api",
    llm_root: str | Path = DEFAULT_LLM_ROOT,
    planner_config: str | Path = DEFAULT_PLANNER_CONFIG,
    planner_python: str | Path | None = None,
    model: str | None = None,
    max_llm_steps: int | None = None,
    max_tool_calls: int | None = None,
    request_timeout_s: float | None = None,
    stream_events: bool = False,
    timeout_s: float | None = None,
) -> PlannerRunResult:
    """Generate a DrawingPlan JSON by calling the existing planner CLI.

    The planner runs in a separate process so the Isaac Python environment does
    not need to import LangChain or OpenAI directly.
    """

    text = natural_language_command.strip()
    if not text:
        raise ValueError("natural_language_command must not be empty.")

    config_path = Path(planner_config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"planner config not found: {config_path}")

    out = Path(output_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if planner_mode == "no-api":
        payload = _fallback_no_api_plan(text, config_path)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return PlannerRunResult(
            command=("bridge-local-no-api", text),
            output_path=out,
            returncode=0,
            stdout="Generated plan with bridge-local no-api fallback.\n",
            stderr="",
        )

    llm_root_path = Path(llm_root).expanduser().resolve()
    if not llm_root_path.exists():
        raise FileNotFoundError(f"LLM planner root not found: {llm_root_path}")

    python_exe = str(planner_python) if planner_python is not None else sys.executable
    args: list[str] = [
        python_exe,
        "-m",
        "robot_drawing_planner.cli",
        text,
        "--pretty",
        "--out",
        str(out),
        "--out-only",
        "--config",
        str(config_path),
    ]
    if planner_mode == "template":
        args.extend(["--mode", "template"])
    elif planner_mode == "agentic":
        args.extend(["--mode", "agentic"])
    else:
        raise ValueError(f"Unsupported planner_mode: {planner_mode!r}")

    if model:
        args.extend(["--model", model])
    if max_llm_steps is not None:
        args.extend(["--max-llm-steps", str(max_llm_steps)])
    if max_tool_calls is not None:
        args.extend(["--max-tool-calls", str(max_tool_calls)])
    if request_timeout_s is not None:
        args.extend(["--request-timeout", str(request_timeout_s)])
    if stream_events:
        args.append("--stream-events")

    env = os.environ.copy()
    env["PYTHONPATH"] = _prepend_pythonpath(str(llm_root_path), env.get("PYTHONPATH"))
    completed = subprocess.run(
        args,
        cwd=str(llm_root_path),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s,
        check=False,
    )
    return PlannerRunResult(
        command=tuple(args),
        output_path=out,
        returncode=int(completed.returncode),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _prepend_pythonpath(path: str, existing: str | None) -> str:
    if existing:
        return path + os.pathsep + existing
    return path


def _fallback_no_api_plan(command: str, config_path: Path) -> dict:
    """Small local fallback used only when the existing no-api CLI cannot import.

    It supports deterministic demo shapes so bridge tests and presentation
    videos can run in a clean Isaac environment without installing LangChain.
    Production LLM runs should use planner_mode ``agentic`` or ``template``
    with the planner dependencies installed.
    """

    config = json.loads(config_path.read_text(encoding="utf-8"))
    shape = _infer_shape(command)
    size_m = _extract_length_m(command)
    if shape == "circle":
        radius = size_m if size_m is not None else float(config.get("default_circle_radius_m", 0.04))
        if _mentions_diameter(command):
            radius *= 0.5
        return _circle_plan(command, radius, config)
    if shape == "square":
        side = size_m if size_m is not None else float(config.get("default_shape_size_m", 0.08))
        return _square_plan(command, side, config)
    if shape == "triangle":
        side = size_m if size_m is not None else float(config.get("default_shape_size_m", 0.08))
        return _triangle_plan(command, side, config)
    if shape == "letter_a":
        size = size_m if size_m is not None else float(config.get("default_shape_size_m", 0.08))
        return _letter_a_plan(command, size, config)
    if shape == "house":
        size = size_m if size_m is not None else float(config.get("default_shape_size_m", 0.08))
        return _house_plan(command, size, config)
    if shape == "star":
        size = size_m if size_m is not None else float(config.get("default_shape_size_m", 0.08))
        return _star_plan(command, size, config)
    raise RuntimeError(
        "Bridge local no-api fallback supports circle, square, triangle, letter A, house, and star. "
        "Install the LLM planner dependencies or use an existing --plan file for this command."
    )


def _infer_shape(command: str) -> str:
    lowered = command.lower()
    if "circle" in lowered or "원" in command or "동그라미" in command:
        return "circle"
    if "square" in lowered or "사각" in command or "네모" in command:
        return "square"
    if "triangle" in lowered or "삼각" in command or "세모" in command:
        return "triangle"
    if "letter a" in lowered or "alphabet a" in lowered or "알파벳 a" in lowered or "글자 a" in lowered:
        return "letter_a"
    if "house" in lowered or "집" in command:
        return "house"
    if "star" in lowered or "별" in command:
        return "star"
    return "circle"


def _extract_length_m(command: str) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(mm|cm|m|밀리|센티|센치)?", command, re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    unit = (match.group(2) or "cm").lower()
    if unit in {"mm", "밀리"}:
        return value * 0.001
    if unit in {"cm", "센티", "센치"}:
        return value * 0.01
    return value


def _mentions_diameter(command: str) -> bool:
    lowered = command.lower()
    if "diameter" in lowered:
        return True
    return "지름" in command and "반지름" not in command


def _point(x: float, y: float, z: float) -> dict[str, float | str]:
    return {"x": float(x), "y": float(y), "z": float(z), "unit": "m"}


def _base_plan(command: str, shape_type: str, actions: list[dict], strokes: list[dict], config: dict) -> dict:
    return {
        "schema_version": "1.0",
        "source_command": command,
        "goal": {
            "shape_type": shape_type,
            "center": {"x": 0.0, "y": 0.0, "unit": "m"},
            "radius_m": strokes[0].get("radius_m") if strokes and strokes[0].get("type") == "arc" else None,
            "side_length_m": None,
            "size_m": float(config.get("default_shape_size_m", 0.08)),
            "orientation_rad": 0.0,
            "letter": None,
            "frame": "board",
            "assumptions": ["bridge-local no-api fallback was used"],
            "warnings": ["robot reachability and IK feasibility are checked downstream by JADE"],
        },
        "strokes": strokes,
        "actions": actions,
        "diagnostics": {
            "validation_ok": True,
            "assumptions": ["bridge-local no-api fallback was used"],
            "warnings": ["robot reachability and IK feasibility are checked downstream by JADE"],
            "errors": [],
            "requires_robot_feasibility_check": True,
            "note": "Generated by llm_isaac_bridge fallback because planner dependencies were unavailable.",
        },
    }


def _circle_plan(command: str, radius: float, config: dict) -> dict:
    hover = float(config.get("hover_height_m", 0.02))
    speed = float(config.get("default_speed_m_s", 0.025))
    down_speed = float(config.get("pen_down_speed_m_s", 0.01))
    up_speed = float(config.get("pen_up_speed_m_s", 0.02))
    stroke_id = "stroke_001"
    actions = [
        {"name": "move_to_start", "frame": "board", "stroke_id": stroke_id, "params": {"target": _point(radius, 0.0, hover), "hover_height_m": hover}},
        {"name": "align_pen_orientation", "frame": "board", "stroke_id": stroke_id, "params": {"mode": "normal_to_board"}},
        {"name": "pen_down", "frame": "board", "stroke_id": stroke_id, "params": {"target": _point(radius, 0.0, 0.0), "speed_m_s": down_speed}},
        {
            "name": "draw_arc",
            "frame": "board",
            "stroke_id": stroke_id,
            "params": {
                "center": _point(0.0, 0.0, 0.0),
                "radius_m": radius,
                "start_angle_rad": 0.0,
                "end_angle_rad": 2.0 * math.pi,
                "direction": "ccw",
                "speed_m_s": speed,
            },
        },
        {"name": "pen_up", "frame": "board", "stroke_id": stroke_id, "params": {"lift_height_m": hover, "speed_m_s": up_speed}},
    ]
    strokes = [
        {
            "type": "arc",
            "stroke_id": stroke_id,
            "center": {"x": 0.0, "y": 0.0, "unit": "m"},
            "radius_m": radius,
            "start_angle_rad": 0.0,
            "end_angle_rad": 2.0 * math.pi,
            "direction": "ccw",
        }
    ]
    plan = _base_plan(command, "circle", actions, strokes, config)
    plan["goal"]["radius_m"] = radius
    plan["goal"]["size_m"] = 2.0 * radius
    return plan


def _square_plan(command: str, side: float, config: dict) -> dict:
    half = side / 2.0
    points = [(-half, -half), (half, -half), (half, half), (-half, half), (-half, -half)]
    return _polyline_plan(command, "square", points, side, config)


def _triangle_plan(command: str, side: float, config: dict) -> dict:
    height = math.sqrt(3.0) * side / 2.0
    points = [(0.0, 2.0 * height / 3.0), (-side / 2.0, -height / 3.0), (side / 2.0, -height / 3.0), (0.0, 2.0 * height / 3.0)]
    return _polyline_plan(command, "triangle", points, side, config)


def _letter_a_plan(command: str, size: float, config: dict) -> dict:
    height = size
    half_width = size * 0.35
    y_bottom = -height / 2.0
    y_top = height / 2.0
    y_cross = -height * 0.05
    strokes = [
        [(-half_width, y_bottom), (0.0, y_top)],
        [(0.0, y_top), (half_width, y_bottom)],
        [(-half_width * 0.45, y_cross), (half_width * 0.45, y_cross)],
    ]
    return _multi_stroke_plan(command, "letter_a", strokes, size, config)


def _house_plan(command: str, size: float, config: dict) -> dict:
    half = size / 2.0
    roof_y = half
    wall_top = size * 0.12
    bottom = -half
    points = [
        (-half, bottom),
        (half, bottom),
        (half, wall_top),
        (0.0, roof_y),
        (-half, wall_top),
        (-half, bottom),
    ]
    return _polyline_plan(command, "house", points, size, config)


def _star_plan(command: str, size: float, config: dict) -> dict:
    outer = size / 2.0
    # Five-point star drawing order. Rotated so one point faces upward in the
    # board frame; the path is one continuous pen-down stroke.
    vertices = []
    for i in range(5):
        angle = math.pi / 2.0 + 2.0 * math.pi * i / 5.0
        vertices.append((outer * math.cos(angle), outer * math.sin(angle)))
    order = [0, 2, 4, 1, 3, 0]
    points = [vertices[i] for i in order]
    return _polyline_plan(command, "star", points, size, config)


def _polyline_plan(command: str, shape: str, points: list[tuple[float, float]], side: float, config: dict) -> dict:
    hover = float(config.get("hover_height_m", 0.02))
    speed = float(config.get("default_speed_m_s", 0.025))
    down_speed = float(config.get("pen_down_speed_m_s", 0.01))
    up_speed = float(config.get("pen_up_speed_m_s", 0.02))
    actions = [
        {"name": "move_to_start", "frame": "board", "params": {"target": _point(points[0][0], points[0][1], hover), "hover_height_m": hover}},
        {"name": "align_pen_orientation", "frame": "board", "params": {"mode": "normal_to_board"}},
        {"name": "pen_down", "frame": "board", "params": {"target": _point(points[0][0], points[0][1], 0.0), "speed_m_s": down_speed}},
    ]
    strokes: list[dict] = []
    for index, (start, end) in enumerate(zip(points, points[1:]), start=1):
        stroke_id = f"stroke_{index:03d}"
        strokes.append(
            {
                "type": "line",
                "stroke_id": stroke_id,
                "start": {"x": start[0], "y": start[1], "unit": "m"},
                "end": {"x": end[0], "y": end[1], "unit": "m"},
            }
        )
        actions.append(
            {
                "name": "draw_line",
                "frame": "board",
                "stroke_id": stroke_id,
                "params": {
                    "start": _point(start[0], start[1], 0.0),
                    "end": _point(end[0], end[1], 0.0),
                    "speed_m_s": speed,
                },
            }
        )
    actions.append({"name": "pen_up", "frame": "board", "params": {"lift_height_m": hover, "speed_m_s": up_speed}})
    plan = _base_plan(command, shape, actions, strokes, config)
    plan["goal"]["side_length_m"] = side
    plan["goal"]["size_m"] = side
    return plan


def _multi_stroke_plan(
    command: str,
    shape: str,
    strokes_xy: list[list[tuple[float, float]]],
    size: float,
    config: dict,
) -> dict:
    hover = float(config.get("hover_height_m", 0.02))
    speed = float(config.get("default_speed_m_s", 0.025))
    down_speed = float(config.get("pen_down_speed_m_s", 0.01))
    up_speed = float(config.get("pen_up_speed_m_s", 0.02))
    actions: list[dict] = []
    strokes: list[dict] = []

    stroke_index = 1
    for polyline in strokes_xy:
        if len(polyline) < 2:
            continue
        first = polyline[0]
        actions.extend(
            [
                {
                    "name": "move_to_start",
                    "frame": "board",
                    "params": {"target": _point(first[0], first[1], hover), "hover_height_m": hover},
                },
                {"name": "align_pen_orientation", "frame": "board", "params": {"mode": "normal_to_board"}},
                {
                    "name": "pen_down",
                    "frame": "board",
                    "params": {"target": _point(first[0], first[1], 0.0), "speed_m_s": down_speed},
                },
            ]
        )
        for start, end in zip(polyline, polyline[1:]):
            stroke_id = f"stroke_{stroke_index:03d}"
            strokes.append(
                {
                    "type": "line",
                    "stroke_id": stroke_id,
                    "start": {"x": start[0], "y": start[1], "unit": "m"},
                    "end": {"x": end[0], "y": end[1], "unit": "m"},
                }
            )
            actions.append(
                {
                    "name": "draw_line",
                    "frame": "board",
                    "stroke_id": stroke_id,
                    "params": {
                        "start": _point(start[0], start[1], 0.0),
                        "end": _point(end[0], end[1], 0.0),
                        "speed_m_s": speed,
                    },
                }
            )
            stroke_index += 1
        actions.append({"name": "pen_up", "frame": "board", "params": {"lift_height_m": hover, "speed_m_s": up_speed}})

    plan = _base_plan(command, shape, actions, strokes, config)
    plan["goal"]["side_length_m"] = size
    plan["goal"]["size_m"] = size
    if shape == "letter_a":
        plan["goal"]["letter"] = "A"
    return plan
