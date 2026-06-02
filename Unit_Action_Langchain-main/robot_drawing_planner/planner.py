"""High-level conversion from natural language to primitive action JSON."""

from __future__ import annotations

import math
from typing import Any

from robot_drawing_planner.config import DEFAULT_CONFIG, PlannerConfig
from robot_drawing_planner.geometry import generate_strokes, stroke_end_point, stroke_start_point
from robot_drawing_planner.llm_client import parse_command_with_llm
from robot_drawing_planner.schemas import (
    ArcStroke,
    Board,
    DrawingPlan,
    LineStroke,
    NormalizedGoal,
    ParsedGoal,
    Point2D,
    Point3D,
    PrimitiveAction,
    Stroke,
    ValidationErrorReport,
)
from robot_drawing_planner.validators import (
    IK_FEASIBILITY_WARNING,
    PlannerValidationError,
    normalize_goal,
    validate_no_low_level_robot_fields,
    validate_normalized_goal,
    validate_strokes_inside_board,
)

PLANNER_SCOPE_NOTE = (
    "This planner does not compute IK, joint angles, Jacobians, or Isaac Sim commands."
)


def strokes_to_actions(
    strokes: list[Stroke],
    config: PlannerConfig,
) -> list[PrimitiveAction]:
    """Convert deterministic strokes into robot-level primitive actions."""

    if not strokes:
        return []
    if _is_single_arc(strokes):
        return _actions_for_arc(strokes[0], config)
    if _is_closed_contiguous_line_contour(strokes):
        return _actions_for_continuous_line_contour(strokes, config)

    actions: list[PrimitiveAction] = []
    for stroke in strokes:
        if isinstance(stroke, LineStroke):
            actions.extend(_actions_for_independent_line(stroke, config))
        elif isinstance(stroke, ArcStroke):
            actions.extend(_actions_for_arc(stroke, config))
        else:
            raise TypeError(f"Unknown stroke type: {type(stroke).__name__}")
    return actions


def build_plan_from_parsed_goal(
    parsed_goal: ParsedGoal,
    config: PlannerConfig | None = None,
) -> DrawingPlan:
    """Build a machine-readable drawing plan from an already parsed goal."""

    planner_config = config or DEFAULT_CONFIG
    diagnostics = _base_diagnostics()

    try:
        goal = normalize_goal(parsed_goal, planner_config)
    except PlannerValidationError as exc:
        diagnostics["validation_ok"] = False
        diagnostics["errors"].append(f"validation failed: {exc}")
        return _drawing_plan(
            parsed_goal=parsed_goal,
            goal=_fallback_goal(parsed_goal, planner_config, [str(exc)]),
            strokes=[],
            actions=[],
            diagnostics=diagnostics,
        )

    goal_report = validate_normalized_goal(goal, planner_config)
    _merge_report(diagnostics, goal_report)
    diagnostics["assumptions"] = list(goal.assumptions)

    strokes: list[Stroke] = []
    actions: list[PrimitiveAction] = []
    if goal_report.ok:
        try:
            strokes = generate_strokes(goal)
        except ValueError as exc:
            diagnostics["validation_ok"] = False
            diagnostics["errors"].append(f"validation failed: {exc}")
        else:
            stroke_report = validate_strokes_inside_board(strokes, planner_config)
            _merge_report(diagnostics, stroke_report)
            if stroke_report.ok:
                actions = strokes_to_actions(strokes, planner_config)

    if diagnostics["errors"]:
        diagnostics["validation_ok"] = False
        actions = []

    plan = _drawing_plan(
        parsed_goal=parsed_goal,
        goal=goal,
        strokes=strokes,
        actions=actions,
        diagnostics=diagnostics,
    )
    validate_no_low_level_robot_fields(plan.model_dump(mode="json"))
    return plan


def plan_drawing(
    command: str,
    config: PlannerConfig | None = None,
    llm: Any | None = None,
) -> DrawingPlan:
    """Parse a natural-language command and build a drawing plan."""

    parsed_goal = parse_command_with_llm(command, llm=llm)
    return build_plan_from_parsed_goal(parsed_goal, config=config)


def plan_drawing_from_goal(
    parsed_goal: ParsedGoal,
    config: PlannerConfig | None = None,
) -> DrawingPlan:
    """Compatibility-friendly wrapper for parsed-goal planning."""

    return build_plan_from_parsed_goal(parsed_goal, config=config)


def plan_from_text(
    command: str,
    parser: Any | None = None,
    board: Board | None = None,
) -> DrawingPlan:
    """Backward-compatible text planning wrapper."""

    return plan_drawing(command, config=_config_from_board(board), llm=parser)


def plan_from_goal(parsed: ParsedGoal, board: Board | None = None) -> DrawingPlan:
    """Backward-compatible parsed-goal planning wrapper."""

    return build_plan_from_parsed_goal(parsed, config=_config_from_board(board))


def _actions_for_continuous_line_contour(
    strokes: list[Stroke],
    config: PlannerConfig,
) -> list[PrimitiveAction]:
    line_strokes = [stroke for stroke in strokes if isinstance(stroke, LineStroke)]
    start = line_strokes[0].start
    final = line_strokes[-1].end
    actions = [
        _move_to_start(line_strokes[0].stroke_id, start, config),
        _align_pen_orientation(line_strokes[0].stroke_id),
        _pen_down(line_strokes[0].stroke_id, start, config),
    ]
    for stroke in line_strokes:
        actions.append(_draw_line(stroke, config))
    actions.append(_pen_up(line_strokes[-1].stroke_id, config, final))
    return actions


def _actions_for_independent_line(
    stroke: LineStroke,
    config: PlannerConfig,
) -> list[PrimitiveAction]:
    return [
        _move_to_start(stroke.stroke_id, stroke.start, config),
        _align_pen_orientation(stroke.stroke_id),
        _pen_down(stroke.stroke_id, stroke.start, config),
        _draw_line(stroke, config),
        _pen_up(stroke.stroke_id, config, stroke.end),
    ]


def _actions_for_arc(stroke: Stroke, config: PlannerConfig) -> list[PrimitiveAction]:
    if not isinstance(stroke, ArcStroke):
        raise TypeError("_actions_for_arc requires ArcStroke")
    start = stroke_start_point(stroke)
    end = stroke_end_point(stroke)
    return [
        _move_to_start(stroke.stroke_id, start, config),
        _align_pen_orientation(stroke.stroke_id),
        _pen_down(stroke.stroke_id, start, config),
        _draw_arc(stroke, config),
        _pen_up(stroke.stroke_id, config, end),
    ]


def _move_to_start(
    stroke_id: str,
    point: Point2D,
    config: PlannerConfig,
) -> PrimitiveAction:
    return PrimitiveAction(
        name="move_to_start",
        stroke_id=stroke_id,
        params={
            "target": _point3d(point, config.hover_height_m),
            "hover_height_m": config.hover_height_m,
            "note": "free-space move; kinematics module converts board frame to base frame",
        },
    )


def _align_pen_orientation(stroke_id: str) -> PrimitiveAction:
    return PrimitiveAction(
        name="align_pen_orientation",
        stroke_id=stroke_id,
        params={
            "mode": "normal_to_board",
            "board_normal_axis": "+z",
            "pen_axis_target": "-z",
            "note": "orientation constraint only; no IK computed here",
        },
    )


def _pen_down(
    stroke_id: str,
    point: Point2D,
    config: PlannerConfig,
) -> PrimitiveAction:
    return PrimitiveAction(
        name="pen_down",
        stroke_id=stroke_id,
        params={
            "target": _point3d(point, config.drawing_z_m),
            "approach_axis": "-z",
            "speed_m_s": config.pen_down_speed_m_s,
        },
    )


def _draw_line(stroke: LineStroke, config: PlannerConfig) -> PrimitiveAction:
    return PrimitiveAction(
        name="draw_line",
        stroke_id=stroke.stroke_id,
        params={
            "start": _point3d(stroke.start, config.drawing_z_m),
            "end": _point3d(stroke.end, config.drawing_z_m),
            "speed_m_s": config.default_speed_m_s,
            "sampling_hint": "kinematics module should interpolate Cartesian waypoints",
        },
    )


def _draw_arc(stroke: ArcStroke, config: PlannerConfig) -> PrimitiveAction:
    return PrimitiveAction(
        name="draw_arc",
        stroke_id=stroke.stroke_id,
        params={
            "center": _point3d(stroke.center, config.drawing_z_m),
            "radius_m": stroke.radius_m,
            "start_angle_rad": stroke.start_angle_rad,
            "end_angle_rad": stroke.end_angle_rad,
            "direction": stroke.direction,
            "speed_m_s": config.default_speed_m_s,
            "sampling_hint": "kinematics module should sample circular Cartesian waypoints",
        },
    )


def _pen_up(
    stroke_id: str,
    config: PlannerConfig,
    _point: Point2D,
) -> PrimitiveAction:
    return PrimitiveAction(
        name="pen_up",
        stroke_id=stroke_id,
        params={
            "lift_height_m": config.hover_height_m,
            "speed_m_s": config.pen_up_speed_m_s,
        },
    )


def _point3d(point: Point2D, z: float) -> dict[str, float | str]:
    return Point3D(x=point.x, y=point.y, z=z).model_dump(mode="json")


def _is_single_arc(strokes: list[Stroke]) -> bool:
    return len(strokes) == 1 and isinstance(strokes[0], ArcStroke)


def _is_closed_contiguous_line_contour(strokes: list[Stroke]) -> bool:
    if not strokes or not all(isinstance(stroke, LineStroke) for stroke in strokes):
        return False
    line_strokes = [stroke for stroke in strokes if isinstance(stroke, LineStroke)]
    if len(line_strokes) < 3:
        return False
    for current, next_stroke in zip(line_strokes, line_strokes[1:]):
        if not _points_close(current.end, next_stroke.start):
            return False
    return _points_close(line_strokes[-1].end, line_strokes[0].start)


def _points_close(a: Point2D, b: Point2D, tolerance: float = 1e-9) -> bool:
    return math.isclose(a.x, b.x, abs_tol=tolerance) and math.isclose(
        a.y, b.y, abs_tol=tolerance
    )


def _base_diagnostics() -> dict[str, Any]:
    return {
        "validation_ok": True,
        "assumptions": [],
        "warnings": [IK_FEASIBILITY_WARNING],
        "errors": [],
        "requires_robot_feasibility_check": True,
        "note": PLANNER_SCOPE_NOTE,
    }


def _merge_report(diagnostics: dict[str, Any], report: ValidationErrorReport) -> None:
    diagnostics["validation_ok"] = diagnostics["validation_ok"] and report.ok
    diagnostics["errors"].extend(report.errors)
    for warning in report.warnings:
        if warning not in diagnostics["warnings"]:
            diagnostics["warnings"].append(warning)


def _drawing_plan(
    parsed_goal: ParsedGoal,
    goal: NormalizedGoal,
    strokes: list[Stroke],
    actions: list[PrimitiveAction],
    diagnostics: dict[str, Any],
) -> DrawingPlan:
    return DrawingPlan(
        source_command=getattr(parsed_goal, "raw_command", ""),
        goal=goal,
        strokes=strokes,
        actions=actions,
        diagnostics=diagnostics,
    )


def _fallback_goal(
    parsed_goal: ParsedGoal,
    config: PlannerConfig,
    warnings: list[str],
) -> NormalizedGoal:
    center = getattr(parsed_goal, "center", None) or config.default_center
    return NormalizedGoal.model_construct(
        shape_type=getattr(parsed_goal, "shape_type", "circle"),
        center=center,
        radius_m=None,
        side_length_m=None,
        size_m=None,
        orientation_rad=math.radians(getattr(parsed_goal, "orientation_deg", 0.0)),
        letter=getattr(parsed_goal, "letter", None),
        frame="board",
        assumptions=[],
        warnings=[IK_FEASIBILITY_WARNING, *warnings],
    )


def _config_from_board(board: Board | None) -> PlannerConfig:
    if board is None:
        return DEFAULT_CONFIG
    return PlannerConfig(board_width_m=board.width_m, board_height_m=board.height_m)

