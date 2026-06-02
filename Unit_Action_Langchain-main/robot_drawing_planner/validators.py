"""Validation helpers for planner inputs and generated strokes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from robot_drawing_planner.config import PlannerConfig
from robot_drawing_planner.geometry import stroke_start_point
from robot_drawing_planner.schemas import (
    ArcStroke,
    Board,
    LineStroke,
    Measurement,
    NormalizedGoal,
    ParsedGoal,
    Point2D,
    Stroke,
    ValidationErrorReport,
)
from robot_drawing_planner.units import measurement_to_meters

SUPPORTED_LETTERS = {"A", "H", "L", "T", "O"}
IK_FEASIBILITY_WARNING = (
    "robot reachability and IK feasibility are not checked by the LangChain planner"
)
LOW_LEVEL_ROBOT_FIELDS = {
    "joint_angle",
    "joint_angles",
    "joint_positions",
    "ik",
    "fk",
    "jacobian",
    "isaac_command",
    "isaac_sim_command",
    "trajectory",
    "workspace_feasible",
    "end_effector_pose",
}


class PlannerValidationError(ValueError):
    """Raised when a drawing goal is outside this module's scope."""


def normalize_goal(parsed: ParsedGoal, config: PlannerConfig) -> NormalizedGoal:
    """Normalize a parsed goal to planner-level board units."""

    if parsed.shape_type not in {"circle", "square", "triangle", "letter"}:
        raise PlannerValidationError(f"Unsupported shape_type '{parsed.shape_type}'.")

    assumptions: list[str] = []
    warnings = [IK_FEASIBILITY_WARNING]

    center = parsed.center
    if center is None:
        center = config.default_center
        assumptions.append("center was not specified; default board center was used")
    if parsed.position_hint:
        assumptions.append(f"position_hint preserved: {parsed.position_hint}")

    radius_m = _measurement_to_meters(parsed.radius)
    side_length_m = _measurement_to_meters(parsed.side_length)
    size_m = _measurement_to_meters(parsed.size)
    letter = parsed.letter

    if parsed.shape_type == "circle":
        if radius_m is None:
            if size_m is not None:
                radius_m = size_m / 2.0
            else:
                radius_m = config.default_circle_radius_m
                assumptions.append("circle radius was not specified; default circle radius was used")
        elif size_m is not None:
            warnings.append("circle radius and size were both specified; radius was used")
        if radius_m <= 0:
            raise PlannerValidationError("Circle radius must be positive.")
        if size_m is None:
            size_m = radius_m * 2.0
    elif parsed.shape_type in {"square", "triangle"}:
        if side_length_m is None:
            if size_m is not None:
                side_length_m = size_m
            else:
                side_length_m = config.default_shape_size_m
                assumptions.append(
                    f"{parsed.shape_type} side_length was not specified; "
                    "default shape size was used"
                )
        elif size_m is not None:
            warnings.append(
                f"{parsed.shape_type} side_length and size were both specified; "
                "side_length was used"
            )
        if side_length_m <= 0:
            raise PlannerValidationError(
                f"{parsed.shape_type.capitalize()} side_length must be positive."
            )
        if size_m is None:
            size_m = side_length_m
    elif parsed.shape_type == "letter":
        letter = _normalize_letter(letter)
        if size_m is None:
            size_m = config.default_shape_size_m
            assumptions.append("letter size was not specified; default shape size was used")
        if size_m <= 0:
            raise PlannerValidationError("Letter size must be positive.")
        if letter == "O":
            radius_m = size_m / 2.0

    if letter is not None and parsed.shape_type != "letter":
        warnings.append("letter ignored because shape_type is not letter")
        letter = None

    return NormalizedGoal(
        shape_type=parsed.shape_type,
        center=center,
        radius_m=radius_m,
        side_length_m=side_length_m,
        size_m=size_m,
        orientation_rad=math.radians(parsed.orientation_deg),
        letter=letter,
        assumptions=assumptions,
        warnings=warnings,
    )


def validate_normalized_goal(
    goal: NormalizedGoal,
    config: PlannerConfig,
) -> ValidationErrorReport:
    """Validate a normalized goal without doing robot feasibility checks."""

    errors: list[str] = []
    warnings = list(goal.warnings)
    if IK_FEASIBILITY_WARNING not in warnings:
        warnings.append(IK_FEASIBILITY_WARNING)

    if goal.shape_type == "circle":
        if goal.radius_m is None or goal.radius_m <= 0:
            errors.append("Circle radius_m must be positive.")
    elif goal.shape_type in {"square", "triangle"}:
        if goal.side_length_m is None or goal.side_length_m <= 0:
            errors.append(f"{goal.shape_type.capitalize()} side_length_m must be positive.")
    elif goal.shape_type == "letter":
        if goal.letter is None:
            errors.append("Letter goals require a letter value.")
        elif len(goal.letter) != 1:
            errors.append("Letter goals require exactly one character.")
        elif goal.letter.upper() not in SUPPORTED_LETTERS:
            errors.append(
                f"Unsupported letter '{goal.letter}'. Supported letters: "
                f"{', '.join(sorted(SUPPORTED_LETTERS))}."
            )
        if goal.size_m is None or goal.size_m <= 0:
            errors.append("Letter size_m must be positive.")

    if config.board_width_m <= 0 or config.board_height_m <= 0:
        errors.append("Planner board dimensions must be positive.")

    return ValidationErrorReport(ok=not errors, errors=errors, warnings=warnings)


def validate_strokes_inside_board(
    strokes: list[Stroke],
    config: PlannerConfig,
) -> ValidationErrorReport:
    """Validate that deterministic strokes fit inside planner board bounds."""

    errors: list[str] = []
    warnings = [IK_FEASIBILITY_WARNING]
    for stroke in strokes:
        if isinstance(stroke, LineStroke):
            _append_point_error(stroke.start, config, stroke.stroke_id, "start", errors)
            _append_point_error(stroke.end, config, stroke.stroke_id, "end", errors)
        elif isinstance(stroke, ArcStroke):
            _append_arc_bbox_errors(stroke, config, errors)
        else:
            errors.append(f"validation failed: unknown stroke type {type(stroke).__name__}")
    return ValidationErrorReport(ok=not errors, errors=errors, warnings=warnings)


def validate_strokes_within_board(strokes: list[Stroke], board: Board) -> None:
    """Compatibility wrapper that raises when strokes exceed board boundaries."""

    config = PlannerConfig(board_width_m=board.width_m, board_height_m=board.height_m)
    report = validate_strokes_inside_board(strokes, config)
    if not report.ok:
        raise PlannerValidationError("; ".join(report.errors))


def point_is_inside_board(point: Point2D, config: PlannerConfig, tolerance: float = 1e-9) -> bool:
    """Return whether a point lies within the planner board."""

    half_width = config.board_width_m / 2.0
    half_height = config.board_height_m / 2.0
    return (
        -half_width - tolerance <= point.x <= half_width + tolerance
        and -half_height - tolerance <= point.y <= half_height + tolerance
    )


def validate_no_low_level_robot_fields(payload: Any) -> None:
    """Reject IK, FK, Jacobian, Isaac Sim, or trajectory-like fields."""

    for key, value in _walk_mapping(payload):
        if key in LOW_LEVEL_ROBOT_FIELDS:
            raise PlannerValidationError(
                f"Low-level robot field '{key}' is outside planner scope."
            )
        if isinstance(value, str) and value.strip().lower() in LOW_LEVEL_ROBOT_FIELDS:
            raise PlannerValidationError(
                f"Low-level robot value '{value}' is outside planner scope."
            )


def _normalize_letter(letter: str | None) -> str:
    if letter is None:
        raise PlannerValidationError("Letter goals require a letter value.")
    normalized = letter.strip().upper()
    if len(normalized) != 1:
        raise PlannerValidationError("Letter goals require exactly one character.")
    if normalized not in SUPPORTED_LETTERS:
        raise PlannerValidationError(
            f"Unsupported letter '{normalized}'. Supported letters: "
            f"{', '.join(sorted(SUPPORTED_LETTERS))}."
        )
    return normalized


def _measurement_to_meters(measurement: Measurement | None) -> float | None:
    if measurement is None:
        return None
    try:
        return measurement_to_meters(measurement)
    except ValueError as exc:
        raise PlannerValidationError(str(exc)) from exc


def _append_point_error(
    point: Point2D,
    config: PlannerConfig,
    stroke_id: str,
    point_name: str,
    errors: list[str],
) -> None:
    if point_is_inside_board(point, config):
        return
    errors.append(
        "validation failed: "
        f"{stroke_id} {point_name} point ({point.x}, {point.y}) is outside board bounds "
        f"x=[{-config.board_width_m / 2}, {config.board_width_m / 2}], "
        f"y=[{-config.board_height_m / 2}, {config.board_height_m / 2}]"
    )


def _append_arc_bbox_errors(stroke: ArcStroke, config: PlannerConfig, errors: list[str]) -> None:
    min_x = stroke.center.x - stroke.radius_m
    max_x = stroke.center.x + stroke.radius_m
    min_y = stroke.center.y - stroke.radius_m
    max_y = stroke.center.y + stroke.radius_m
    half_width = config.board_width_m / 2.0
    half_height = config.board_height_m / 2.0
    if min_x < -half_width or max_x > half_width or min_y < -half_height or max_y > half_height:
        errors.append(
            "validation failed: "
            f"{stroke.stroke_id} arc bounding box "
            f"x=[{min_x}, {max_x}], y=[{min_y}, {max_y}] is outside board bounds "
            f"x=[{-half_width}, {half_width}], y=[{-half_height}, {half_height}]"
        )


def _walk_mapping(payload: Any) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key).strip().lower()
            items.append((key, value))
            items.extend(_walk_mapping(value))
    elif isinstance(payload, list):
        for value in payload:
            items.extend(_walk_mapping(value))
    return items
