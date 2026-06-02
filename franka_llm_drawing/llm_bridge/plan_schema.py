"""Small dataclasses for the LLM planner handoff format.

The existing LLM planner may use Pydantic internally. This offline control
package keeps its boundary lightweight and only depends on a JSON-compatible
mapping with an ``actions`` list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

SUPPORTED_ACTION_NAMES = {
    "move_to_start",
    "align_pen_orientation",
    "pen_down",
    "draw_line_to",
    "draw_line",
    "draw_arc",
    "pen_up",
}

UNIT_TO_METERS = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
}


@dataclass(frozen=True)
class Point3D:
    """A point expressed in SI meters after parsing."""

    x: float
    y: float
    z: float = 0.0
    unit: str = "m"

    def to_array(self) -> np.ndarray:
        """Return the point as a shape ``(3,)`` NumPy array in meters."""

        scale = unit_scale(self.unit)
        return np.array([self.x, self.y, self.z], dtype=float) * scale


@dataclass(frozen=True)
class PrimitiveAction:
    """One symbolic drawing primitive from the LLM planner."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    frame: str = "board"
    stroke_id: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PrimitiveAction":
        if "name" in data:
            name = str(data["name"])
            params = data.get("params") or {}
        elif "type" in data:
            name = str(data["type"])
            params = _params_from_jade_action(name, data)
        else:
            raise ValueError("PrimitiveAction requires a 'name' or 'type' field.")
        if not isinstance(params, Mapping):
            raise ValueError("PrimitiveAction.params must be a mapping.")
        stroke_id = data.get("stroke_id", data.get("id"))
        return cls(
            name=name,
            params=dict(params),
            frame=str(data.get("frame", "board")),
            stroke_id=None if stroke_id is None else str(stroke_id),
        )


@dataclass(frozen=True)
class DrawingPlan:
    """A JSON handoff plan containing board-frame primitive actions."""

    actions: list[PrimitiveAction]
    schema_version: str = "1.0"
    source_command: str = ""
    goal: dict[str, Any] | None = None
    strokes: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DrawingPlan":
        actions_raw = data.get("actions")
        if not isinstance(actions_raw, list):
            raise ValueError("DrawingPlan JSON requires an 'actions' list.")
        return cls(
            actions=[PrimitiveAction.from_mapping(item) for item in actions_raw],
            schema_version=str(data.get("schema_version", "1.0")),
            source_command=str(data.get("source_command", "")),
            goal=dict(data["goal"]) if isinstance(data.get("goal"), Mapping) else None,
            strokes=[dict(item) for item in data.get("strokes", [])],
            diagnostics=(
                dict(data["diagnostics"])
                if isinstance(data.get("diagnostics"), Mapping)
                else {}
            ),
        )


def unit_scale(unit: str) -> float:
    """Return a multiplier from the given length unit to meters."""

    try:
        return UNIT_TO_METERS[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported length unit: {unit!r}.") from exc


def point3d_from_mapping(data: Mapping[str, Any], default_z: float = 0.0) -> Point3D:
    """Parse a JSON point mapping into a ``Point3D``.

    Missing ``z`` is accepted because some planner internals represent strokes
    as 2D board coordinates.
    """

    if "x" not in data or "y" not in data:
        raise ValueError("Point mapping requires 'x' and 'y'.")
    return Point3D(
        x=float(data["x"]),
        y=float(data["y"]),
        z=float(data.get("z", default_z)),
        unit=str(data.get("unit", "m")),
    )


def point3d_from_sequence(data: Any, default_z: float = 0.0, unit: str = "m") -> Point3D:
    """Parse a ``[x, y]`` or ``[x, y, z]`` JSON point into ``Point3D``."""

    if not isinstance(data, (list, tuple)) or len(data) not in {2, 3}:
        raise ValueError("Point sequence must be [x, y] or [x, y, z].")
    return Point3D(
        x=float(data[0]),
        y=float(data[1]),
        z=float(data[2]) if len(data) == 3 else float(default_z),
        unit=unit,
    )


def _point_param(value: Any, default_z: float = 0.0, unit: str = "m") -> dict[str, Any]:
    if isinstance(value, Mapping):
        point = point3d_from_mapping(value, default_z=default_z)
    else:
        point = point3d_from_sequence(value, default_z=default_z, unit=unit)
    return {"x": point.x, "y": point.y, "z": point.z, "unit": point.unit}


def _speed_param(data: Mapping[str, Any]) -> dict[str, Any]:
    if "speed_m_s" in data:
        return {"speed_m_s": float(data["speed_m_s"])}
    if "speed" in data:
        return {"speed_m_s": float(data["speed"])}
    return {}


def _params_from_jade_action(name: str, data: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the JADE markdown action schema into the internal schema."""

    unit = str(data.get("unit", data.get("units", "m")))
    params: dict[str, Any] = {}
    params.update(_speed_param(data))

    if name == "move_to_start":
        if "target" not in data:
            raise ValueError("move_to_start requires 'target'.")
        params["target"] = _point_param(data["target"], unit=unit)
        return params

    if name == "pen_down":
        if "target" in data:
            params["target"] = _point_param(data["target"], unit=unit)
        if "target_z" in data:
            params["target_z_m"] = float(data["target_z"]) * unit_scale(unit)
        return params

    if name == "draw_line":
        if "end" not in data:
            raise ValueError("draw_line requires 'end'.")
        if "start" in data:
            params["start"] = _point_param(data["start"], unit=unit)
        params["end"] = _point_param(data["end"], unit=unit)
        return params

    if name == "draw_arc":
        if "center" not in data:
            raise ValueError("draw_arc requires 'center'.")
        params["center"] = _point_param(data["center"], unit=unit)
        if "radius_m" in data:
            params["radius_m"] = float(data["radius_m"])
        elif "radius" in data:
            params["radius_m"] = float(data["radius"]) * unit_scale(unit)
        else:
            raise ValueError("draw_arc requires 'radius' or 'radius_m'.")
        if "theta_start_deg" in data:
            params["start_angle_rad"] = math.radians(float(data["theta_start_deg"]))
        elif "start_angle_rad" in data:
            params["start_angle_rad"] = float(data["start_angle_rad"])
        else:
            raise ValueError("draw_arc requires theta_start_deg or start_angle_rad.")
        if "theta_end_deg" in data:
            params["end_angle_rad"] = math.radians(float(data["theta_end_deg"]))
        elif "end_angle_rad" in data:
            params["end_angle_rad"] = float(data["end_angle_rad"])
        else:
            raise ValueError("draw_arc requires theta_end_deg or end_angle_rad.")
        clockwise = bool(data.get("clockwise", False))
        params["direction"] = str(data.get("direction", "cw" if clockwise else "ccw"))
        return params

    if name == "pen_up":
        if "lift_height_m" in data:
            params["lift_height_m"] = float(data["lift_height_m"])
        elif "lift_distance" in data:
            params["lift_distance_m"] = float(data["lift_distance"]) * unit_scale(unit)
        return params

    raise ValueError(f"Unsupported JADE action type: {name!r}.")
