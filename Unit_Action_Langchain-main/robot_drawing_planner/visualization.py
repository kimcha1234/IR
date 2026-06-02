"""Matplotlib visualization for planned drawing primitive JSON.

This module plots planned board-frame strokes only. It does not compute or
visualize robot execution, IK, FK, Jacobians, joint commands, torques, dynamics,
trajectory timing, or Isaac Sim behavior.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from robot_drawing_planner.config import PlannerConfig
from robot_drawing_planner.schemas import (
    ArcStroke,
    DrawingPlan,
    LineStroke,
    PrimitiveAction,
)


def load_plan_json(path: str | Path) -> dict[str, Any]:
    """Read a DrawingPlan JSON file and return the raw dictionary."""

    plan_path = Path(path)
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in plan file '{plan_path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Plan JSON '{plan_path}' must contain a JSON object.")
    return payload


def load_drawing_plan(path: str | Path) -> DrawingPlan | dict[str, Any]:
    """Load a plan as DrawingPlan when possible, otherwise return a raw dict."""

    plan_path = Path(path)
    try:
        text = plan_path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in plan file '{plan_path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Plan JSON '{plan_path}' must contain a JSON object.")
    if "strokes" not in payload and "actions" not in payload:
        raise ValueError(
            f"Plan JSON '{plan_path}' must contain at least one of 'strokes' or 'actions'."
        )
    try:
        return DrawingPlan.model_validate_json(text)
    except Exception:
        return payload


def point2d_from_any(value: dict[str, Any]) -> tuple[float, float]:
    """Return board-frame x/y from a dict, ignoring z when present."""

    if "x" not in value or "y" not in value:
        raise ValueError(f"Point must include x and y fields: {value!r}")
    return float(value["x"]), float(value["y"])


def sample_line(
    start: dict[str, Any],
    end: dict[str, Any],
    samples: int = 2,
) -> list[tuple[float, float]]:
    """Sample a straight line in board-frame meters."""

    count = max(2, int(samples))
    x0, y0 = point2d_from_any(start)
    x1, y1 = point2d_from_any(end)
    if count == 2:
        return [(x0, y0), (x1, y1)]
    return [
        (
            x0 + (x1 - x0) * i / (count - 1),
            y0 + (y1 - y0) * i / (count - 1),
        )
        for i in range(count)
    ]


def sample_arc(
    center: dict[str, Any],
    radius_m: float,
    start_angle_rad: float,
    end_angle_rad: float,
    direction: str = "ccw",
    samples: int = 96,
) -> list[tuple[float, float]]:
    """Sample a circular arc in board-frame meters.

    The result is a visualization sample of symbolic geometry, not a robot
    trajectory sample or timing plan.
    """

    if radius_m <= 0:
        raise ValueError("radius_m must be positive.")
    count = max(2, int(samples))
    cx, cy = point2d_from_any(center)
    start = float(start_angle_rad)
    end = float(end_angle_rad)
    normalized_direction = direction.lower()
    if normalized_direction not in {"cw", "ccw"}:
        raise ValueError("direction must be 'cw' or 'ccw'.")

    delta = end - start
    full_circle = math.isclose(abs(delta), 2.0 * math.pi, rel_tol=0.0, abs_tol=1e-12)
    if normalized_direction == "ccw":
        if full_circle:
            end = start + 2.0 * math.pi
        elif end < start:
            end += 2.0 * math.pi
    else:
        if full_circle:
            end = start - 2.0 * math.pi
        elif end > start:
            end -= 2.0 * math.pi

    return [
        (
            cx + radius_m * math.cos(start + (end - start) * i / (count - 1)),
            cy + radius_m * math.sin(start + (end - start) * i / (count - 1)),
        )
        for i in range(count)
    ]


def extract_draw_segments(
    plan: DrawingPlan | dict[str, Any],
    arc_samples: int = 96,
) -> list[dict[str, Any]]:
    """Extract drawable pen-down line/arc segments from strokes or actions."""

    payload = _plan_to_dict(plan)
    strokes = payload.get("strokes") or []
    if strokes:
        return _segments_from_strokes(strokes, arc_samples=arc_samples)
    return _segments_from_actions(payload.get("actions") or [], arc_samples=arc_samples)


def extract_pen_up_moves(plan: DrawingPlan | dict[str, Any]) -> list[list[tuple[float, float]]]:
    """Reconstruct optional free-space move_to_start paths from action sequence."""

    payload = _plan_to_dict(plan)
    moves: list[list[tuple[float, float]]] = []
    current_position: tuple[float, float] | None = None
    pen_state = "up"
    for action in payload.get("actions") or []:
        item = _as_dict(action)
        name = item.get("name")
        params = _as_dict(item.get("params") or {})
        if name == "move_to_start":
            target = point2d_from_any(_as_dict(params.get("target") or {}))
            if current_position is not None and pen_state == "up":
                moves.append([current_position, target])
            current_position = target
        elif name == "pen_down":
            target = params.get("target")
            if target is not None:
                current_position = point2d_from_any(_as_dict(target))
            pen_state = "down"
        elif name == "draw_line":
            current_position = point2d_from_any(_as_dict(params.get("end") or {}))
        elif name == "draw_arc":
            current_position = _arc_end_point_from_params(params)
        elif name == "pen_up":
            pen_state = "up"
    return moves


def plot_drawing_plan(
    plan: DrawingPlan | dict[str, Any],
    config: PlannerConfig | None = None,
    title: str | None = None,
    show_board: bool = True,
    show_action_labels: bool = True,
    show_pen_up_moves: bool = False,
    arc_samples: int = 96,
    figsize: tuple[float, float] = (7.0, 5.0),
) -> tuple[object, object]:
    """Plot a planned board-frame drawing path and return ``(fig, ax)``."""

    planner_config = config or PlannerConfig()
    segments = extract_draw_segments(plan, arc_samples=arc_samples)
    pen_up_moves = extract_pen_up_moves(plan) if show_pen_up_moves else []
    fig, ax = plt.subplots(figsize=figsize)

    half_w = planner_config.board_width_m / 2.0
    half_h = planner_config.board_height_m / 2.0
    if show_board:
        board = Rectangle(
            (-half_w, -half_h),
            planner_config.board_width_m,
            planner_config.board_height_m,
            fill=False,
            linewidth=1.4,
            edgecolor="black",
        )
        ax.add_patch(board)
        ax.axhline(0.0, color="0.85", linewidth=0.8)
        ax.axvline(0.0, color="0.85", linewidth=0.8)

    for move in pen_up_moves:
        xs = [point[0] for point in move]
        ys = [point[1] for point in move]
        ax.plot(xs, ys, color="0.55", linestyle="--", linewidth=1.0, alpha=0.75)

    all_points: list[tuple[float, float]] = []
    for index, segment in enumerate(segments, start=1):
        points = segment["points"]
        if not points:
            continue
        all_points.extend(points)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        ax.plot(xs, ys, linewidth=2.0)
        if show_action_labels:
            label = segment.get("stroke_id") or str(index)
            ax.text(xs[0], ys[0], str(label), fontsize=8, ha="left", va="bottom")

    if all_points:
        ax.scatter([all_points[0][0]], [all_points[0][1]], color="green", s=34, zorder=4)
        ax.scatter([all_points[-1][0]], [all_points[-1][1]], color="red", s=34, zorder=4)

    out_of_board = _contains_out_of_board(all_points, planner_config)
    plot_title = title or "Planned drawing path"
    if out_of_board:
        plot_title = f"{plot_title} (contains out-of-board coordinates)"
        ax.text(
            0.01,
            0.99,
            "contains out-of-board coordinates",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="darkred",
        )

    ax.set_title(plot_title)
    ax.set_xlabel("board x [m]")
    ax.set_ylabel("board y [m]")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_aspect("equal", adjustable="box")
    _set_limits(ax, all_points, half_w, half_h)
    return fig, ax


def save_plan_plot(
    plan: DrawingPlan | dict[str, Any],
    out_path: str | Path,
    config: PlannerConfig | None = None,
    title: str | None = None,
    show_board: bool = True,
    show_action_labels: bool = True,
    show_pen_up_moves: bool = False,
    arc_samples: int = 96,
    dpi: int = 200,
) -> Path:
    """Save a plan plot as PNG/SVG/PDF according to the output suffix."""

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, _ax = plot_drawing_plan(
        plan,
        config=config,
        title=title,
        show_board=show_board,
        show_action_labels=show_action_labels,
        show_pen_up_moves=show_pen_up_moves,
        arc_samples=arc_samples,
    )
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return output_path


def _plan_to_dict(plan: DrawingPlan | dict[str, Any]) -> dict[str, Any]:
    if isinstance(plan, DrawingPlan):
        return plan.model_dump(mode="json")
    if isinstance(plan, dict):
        return plan
    if hasattr(plan, "model_dump"):
        return plan.model_dump(mode="json")
    raise ValueError(f"Unsupported plan object: {type(plan).__name__}")


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (LineStroke, ArcStroke, PrimitiveAction, DrawingPlan)):
        return value.model_dump(mode="json")
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise ValueError(f"Expected dictionary-like value, got {type(value).__name__}.")


def _segments_from_strokes(
    strokes: list[Any],
    arc_samples: int,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for index, stroke in enumerate(strokes, start=1):
        item = _as_dict(stroke)
        stroke_type = item.get("type")
        stroke_id = item.get("stroke_id")
        if stroke_type == "line":
            points = sample_line(_as_dict(item["start"]), _as_dict(item["end"]))
        elif stroke_type == "arc":
            points = sample_arc(
                _as_dict(item["center"]),
                float(item["radius_m"]),
                float(item["start_angle_rad"]),
                float(item["end_angle_rad"]),
                str(item.get("direction", "ccw")),
                samples=arc_samples,
            )
        else:
            continue
        segments.append(
            {
                "type": stroke_type,
                "stroke_id": stroke_id,
                "points": points,
                "label": str(stroke_id or index),
            }
        )
    return segments


def _segments_from_actions(
    actions: list[Any],
    arc_samples: int,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for index, action in enumerate(actions, start=1):
        item = _as_dict(action)
        name = item.get("name")
        params = _as_dict(item.get("params") or {})
        stroke_id = item.get("stroke_id")
        if name == "draw_line":
            points = sample_line(_as_dict(params["start"]), _as_dict(params["end"]))
            segment_type = "line"
        elif name == "draw_arc":
            points = sample_arc(
                _as_dict(params["center"]),
                float(params["radius_m"]),
                float(params["start_angle_rad"]),
                float(params["end_angle_rad"]),
                str(params.get("direction", "ccw")),
                samples=arc_samples,
            )
            segment_type = "arc"
        else:
            continue
        segments.append(
            {
                "type": segment_type,
                "stroke_id": stroke_id,
                "points": points,
                "label": str(stroke_id or index),
            }
        )
    return segments


def _arc_end_point_from_params(params: dict[str, Any]) -> tuple[float, float]:
    center = _as_dict(params.get("center") or {})
    radius_m = float(params["radius_m"])
    start_angle = float(params["start_angle_rad"])
    end_angle = float(params["end_angle_rad"])
    if math.isclose(
        abs(end_angle - start_angle),
        2.0 * math.pi,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        angle = start_angle
    else:
        angle = end_angle
    cx, cy = point2d_from_any(center)
    return cx + radius_m * math.cos(angle), cy + radius_m * math.sin(angle)


def _contains_out_of_board(
    points: list[tuple[float, float]],
    config: PlannerConfig,
) -> bool:
    half_w = config.board_width_m / 2.0
    half_h = config.board_height_m / 2.0
    return any(x < -half_w or x > half_w or y < -half_h or y > half_h for x, y in points)


def _set_limits(
    ax: Any,
    points: list[tuple[float, float]],
    half_w: float,
    half_h: float,
) -> None:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min = min([-half_w, *xs])
    x_max = max([half_w, *xs])
    y_min = min([-half_h, *ys])
    y_max = max([half_h, *ys])
    span = max(x_max - x_min, y_max - y_min, 1e-6)
    pad = max(0.02, span * 0.06)
    ax.set_xlim(x_min - pad, x_max + pad)
    ax.set_ylim(y_min - pad, y_max + pad)
