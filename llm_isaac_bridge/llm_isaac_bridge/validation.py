"""Safety validation for LLM DrawingPlan files before Isaac execution."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from llm_isaac_bridge.paths import DEFAULT_PLANNER_CONFIG

SUPPORTED_ACTIONS = {
    "move_to_start",
    "align_pen_orientation",
    "pen_down",
    "draw_line",
    "draw_line_to",
    "draw_arc",
    "pen_up",
}

DRAW_ACTIONS = {"draw_line", "draw_line_to", "draw_arc"}
UNIT_SCALE = {"m": 1.0, "cm": 0.01, "mm": 0.001}


@dataclass(frozen=True)
class ValidationConfig:
    """Bridge-level validation limits matched to the current Isaac scene."""

    board_width_m: float = 0.30
    board_height_m: float = 0.30
    hover_height_m: float = 0.02
    drawing_z_m: float = 0.0
    max_speed_m_s: float = 0.10
    max_draw_speed_m_s: float = 0.04
    point_tolerance_m: float = 1e-5
    z_warning_tolerance_m: float = 0.015


@dataclass(frozen=True)
class ValidationReport:
    """Machine-readable result of DrawingPlan validation."""

    ok: bool
    errors: list[str]
    warnings: list[str]
    action_count: int
    drawable_action_count: int
    source_command: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_plan_file(
    path: str | Path,
    *,
    planner_config: str | Path = DEFAULT_PLANNER_CONFIG,
) -> ValidationReport:
    """Load and validate a DrawingPlan JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = load_validation_config(planner_config)
    return validate_plan_payload(payload, config=config)


def validate_plan_payload(payload: Mapping[str, Any], *, config: ValidationConfig) -> ValidationReport:
    """Validate one planner payload before it is given to Isaac."""

    errors: list[str] = []
    warnings: list[str] = []
    source_command = str(payload.get("source_command", ""))

    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, Mapping) and diagnostics.get("validation_ok") is False:
        errors.append("planner diagnostics.validation_ok is false.")
        for item in diagnostics.get("errors", []) or []:
            errors.append(f"planner error: {item}")
        for item in diagnostics.get("warnings", []) or []:
            warnings.append(f"planner warning: {item}")

    actions_raw = payload.get("actions")
    if not isinstance(actions_raw, list) or not actions_raw:
        return ValidationReport(
            ok=False,
            errors=errors + ["DrawingPlan requires a non-empty actions list."],
            warnings=warnings,
            action_count=0,
            drawable_action_count=0,
            source_command=source_command,
        )

    pen_state = "up"
    current_xy: tuple[float, float] | None = None
    drawable_count = 0

    for index, action in enumerate(actions_raw):
        label = f"actions[{index}]"
        if not isinstance(action, Mapping):
            errors.append(f"{label} must be an object.")
            continue
        name = str(action.get("name", action.get("type", "")))
        if name not in SUPPORTED_ACTIONS:
            errors.append(f"{label} has unsupported action name {name!r}.")
            continue
        frame = str(action.get("frame", "board"))
        if frame != "board":
            errors.append(f"{label} frame must be 'board', got {frame!r}.")
        params = action.get("params") or _params_from_legacy_action(action)
        if not isinstance(params, Mapping):
            errors.append(f"{label}.params must be an object.")
            continue

        if name == "move_to_start":
            if pen_state != "up":
                errors.append(f"{label} move_to_start requires pen_state == 'up'.")
            point = _required_point(params, "target", label, errors)
            if point is not None:
                _check_point_inside_board(point, config, label, errors)
                _warn_z(point[2], config.hover_height_m, label, "hover height", config, warnings)
                current_xy = (point[0], point[1])
            _check_speed(params, label, config.max_speed_m_s, errors)

        elif name == "align_pen_orientation":
            continue

        elif name == "pen_down":
            if pen_state != "up":
                errors.append(f"{label} pen_down requires pen_state == 'up'.")
            point = _optional_point(params, "target", label, errors)
            if point is not None:
                _check_point_inside_board(point, config, label, errors)
                _warn_z(point[2], config.drawing_z_m, label, "drawing z", config, warnings)
                current_xy = (point[0], point[1])
            elif current_xy is None:
                errors.append(f"{label} pen_down without target requires a previous current position.")
            _check_speed(params, label, config.max_speed_m_s, errors)
            pen_state = "down"

        elif name in {"draw_line", "draw_line_to"}:
            if pen_state != "down":
                errors.append(f"{label} {name} requires pen_state == 'down'.")
            key = "end" if name == "draw_line" else "target"
            if name == "draw_line":
                start = _optional_point(params, "start", label, errors)
                if start is not None and current_xy is not None:
                    _check_xy_close(current_xy, (start[0], start[1]), label, "draw_line start", config, errors)
            end = _required_point(params, key, label, errors)
            if end is not None:
                _check_point_inside_board(end, config, label, errors)
                _warn_z(end[2], config.drawing_z_m, label, "drawing z", config, warnings)
                current_xy = (end[0], end[1])
            _check_speed(params, label, config.max_draw_speed_m_s, errors)
            drawable_count += 1

        elif name == "draw_arc":
            if pen_state != "down":
                errors.append(f"{label} draw_arc requires pen_state == 'down'.")
            center = _required_point(params, "center", label, errors)
            radius = _required_positive_float(params, "radius_m", label, errors)
            start_angle = _required_float(params, "start_angle_rad", label, errors)
            end_angle = _required_float(params, "end_angle_rad", label, errors)
            direction = str(params.get("direction", ""))
            if direction not in {"cw", "ccw"}:
                errors.append(f"{label}.params.direction must be 'cw' or 'ccw'.")
            if center is not None and radius is not None:
                _check_arc_inside_board(center, radius, config, label, errors)
                _warn_z(center[2], config.drawing_z_m, label, "drawing z", config, warnings)
            if center is not None and radius is not None and start_angle is not None and current_xy is not None:
                expected = (
                    center[0] + radius * math.cos(start_angle),
                    center[1] + radius * math.sin(start_angle),
                )
                _check_xy_close(current_xy, expected, label, "draw_arc start", config, errors)
            if center is not None and radius is not None and end_angle is not None:
                current_xy = (
                    center[0] + radius * math.cos(end_angle),
                    center[1] + radius * math.sin(end_angle),
                )
            _check_speed(params, label, config.max_draw_speed_m_s, errors)
            drawable_count += 1

        elif name == "pen_up":
            if pen_state != "down":
                errors.append(f"{label} pen_up requires pen_state == 'down'.")
            lift_height = params.get("lift_height_m")
            lift_distance = params.get("lift_distance_m", params.get("lift_distance"))
            if lift_height is not None and float(lift_height) <= config.drawing_z_m:
                errors.append(f"{label}.params.lift_height_m must be above drawing_z_m.")
            if lift_distance is not None and float(lift_distance) <= 0.0:
                errors.append(f"{label}.params.lift_distance_m must be positive.")
            _check_speed(params, label, config.max_speed_m_s, errors)
            pen_state = "up"

    if drawable_count == 0:
        errors.append("DrawingPlan must contain at least one drawable action.")
    if pen_state != "up":
        errors.append("DrawingPlan finished with pen_state == 'down'; add pen_up before execution.")

    return ValidationReport(
        ok=not errors,
        errors=errors,
        warnings=warnings,
        action_count=len(actions_raw),
        drawable_action_count=drawable_count,
        source_command=source_command,
    )


def write_validation_report(report: ValidationReport, path: str | Path) -> Path:
    """Write validation report JSON."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def load_validation_config(path: str | Path = DEFAULT_PLANNER_CONFIG) -> ValidationConfig:
    """Load bridge validation limits from the planner config file."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ValidationConfig(
        board_width_m=float(raw.get("board_width_m", ValidationConfig.board_width_m)),
        board_height_m=float(raw.get("board_height_m", ValidationConfig.board_height_m)),
        hover_height_m=float(raw.get("hover_height_m", ValidationConfig.hover_height_m)),
        drawing_z_m=float(raw.get("drawing_z_m", ValidationConfig.drawing_z_m)),
        max_draw_speed_m_s=min(
            0.04,
            max(float(raw.get("default_speed_m_s", 0.025)) * 1.6, 0.025),
        ),
    )


def _params_from_legacy_action(action: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the legacy JADE action format enough for validation."""

    name = str(action.get("type", ""))
    params: dict[str, Any] = {}
    if "speed" in action:
        params["speed_m_s"] = action["speed"]
    if name == "move_to_start" and "target" in action:
        params["target"] = _legacy_point(action["target"])
    elif name == "pen_down":
        if "target" in action:
            params["target"] = _legacy_point(action["target"])
        if "target_z" in action:
            params["target_z_m"] = float(action["target_z"]) * UNIT_SCALE[str(action.get("unit", "m"))]
    elif name == "draw_arc":
        if "center" in action:
            params["center"] = _legacy_point(action["center"])
        if "radius" in action:
            params["radius_m"] = action["radius"]
        if "radius_m" in action:
            params["radius_m"] = action["radius_m"]
        if "theta_start_deg" in action:
            params["start_angle_rad"] = math.radians(float(action["theta_start_deg"]))
        if "theta_end_deg" in action:
            params["end_angle_rad"] = math.radians(float(action["theta_end_deg"]))
        params["direction"] = action.get("direction", "cw" if action.get("clockwise") else "ccw")
    elif name == "pen_up":
        if "lift_distance" in action:
            params["lift_distance_m"] = action["lift_distance"]
        if "lift_height_m" in action:
            params["lift_height_m"] = action["lift_height_m"]
    return params


def _legacy_point(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (list, tuple)) and len(value) in {2, 3}:
        return {
            "x": value[0],
            "y": value[1],
            "z": value[2] if len(value) == 3 else 0.0,
            "unit": "m",
        }
    return {"x": 0.0, "y": 0.0, "z": 0.0, "unit": "m"}


def _required_point(
    params: Mapping[str, Any],
    key: str,
    label: str,
    errors: list[str],
) -> tuple[float, float, float] | None:
    point = _optional_point(params, key, label, errors)
    if point is None:
        errors.append(f"{label}.params.{key} is required.")
    return point


def _optional_point(
    params: Mapping[str, Any],
    key: str,
    label: str,
    errors: list[str],
) -> tuple[float, float, float] | None:
    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        errors.append(f"{label}.params.{key} must be a point object.")
        return None
    try:
        unit = str(value.get("unit", "m"))
        scale = UNIT_SCALE[unit]
        return (
            float(value["x"]) * scale,
            float(value["y"]) * scale,
            float(value.get("z", 0.0)) * scale,
        )
    except KeyError as exc:
        errors.append(f"{label}.params.{key} missing field {exc.args[0]!r}.")
    except (TypeError, ValueError):
        errors.append(f"{label}.params.{key} contains non-numeric coordinates.")
    return None


def _required_float(params: Mapping[str, Any], key: str, label: str, errors: list[str]) -> float | None:
    try:
        return float(params[key])
    except KeyError:
        errors.append(f"{label}.params.{key} is required.")
    except (TypeError, ValueError):
        errors.append(f"{label}.params.{key} must be numeric.")
    return None


def _required_positive_float(params: Mapping[str, Any], key: str, label: str, errors: list[str]) -> float | None:
    value = _required_float(params, key, label, errors)
    if value is not None and value <= 0.0:
        errors.append(f"{label}.params.{key} must be positive.")
    return value


def _check_point_inside_board(
    point: tuple[float, float, float],
    config: ValidationConfig,
    label: str,
    errors: list[str],
) -> None:
    half_x = config.board_width_m / 2.0
    half_y = config.board_height_m / 2.0
    if not (-half_x <= point[0] <= half_x and -half_y <= point[1] <= half_y):
        errors.append(
            f"{label} point ({point[0]:.4f}, {point[1]:.4f}) is outside board "
            f"x=[{-half_x:.4f},{half_x:.4f}], y=[{-half_y:.4f},{half_y:.4f}]."
        )


def _check_arc_inside_board(
    center: tuple[float, float, float],
    radius_m: float,
    config: ValidationConfig,
    label: str,
    errors: list[str],
) -> None:
    half_x = config.board_width_m / 2.0
    half_y = config.board_height_m / 2.0
    if center[0] - radius_m < -half_x or center[0] + radius_m > half_x:
        errors.append(f"{label} arc x range is outside board bounds.")
    if center[1] - radius_m < -half_y or center[1] + radius_m > half_y:
        errors.append(f"{label} arc y range is outside board bounds.")


def _check_xy_close(
    actual: tuple[float, float],
    expected: tuple[float, float],
    label: str,
    what: str,
    config: ValidationConfig,
    errors: list[str],
) -> None:
    error = math.hypot(actual[0] - expected[0], actual[1] - expected[1])
    if error > config.point_tolerance_m:
        errors.append(
            f"{label} {what} does not match current position: "
            f"error={error:.6f} m."
        )


def _warn_z(
    z_value: float,
    expected: float,
    label: str,
    what: str,
    config: ValidationConfig,
    warnings: list[str],
) -> None:
    if abs(z_value - expected) > config.z_warning_tolerance_m:
        warnings.append(f"{label} z={z_value:.4f} m differs from expected {what} {expected:.4f} m.")


def _check_speed(
    params: Mapping[str, Any],
    label: str,
    max_speed_m_s: float,
    errors: list[str],
) -> None:
    if "speed_m_s" not in params:
        return
    try:
        speed = float(params["speed_m_s"])
    except (TypeError, ValueError):
        errors.append(f"{label}.params.speed_m_s must be numeric.")
        return
    if speed <= 0.0:
        errors.append(f"{label}.params.speed_m_s must be positive.")
    if speed > max_speed_m_s:
        errors.append(f"{label}.params.speed_m_s={speed:.4f} exceeds limit {max_speed_m_s:.4f}.")
