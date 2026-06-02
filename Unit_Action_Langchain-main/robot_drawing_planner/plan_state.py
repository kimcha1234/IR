"""Stateful symbolic plan builder for agentic unit-action planning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from robot_drawing_planner.config import PlannerConfig
from robot_drawing_planner.schemas import (
    ArcStroke,
    Direction,
    LineStroke,
    Point2D,
    Point3D,
    PrimitiveAction,
    Stroke,
)

PenState = Literal["up", "down"]


@dataclass
class PlanBuilder:
    """Mutable state that LangChain unit-action tools append to.

    This builder records symbolic primitive actions only. It does not compute
    IK, FK, Jacobians, joint commands, trajectory samples, or Isaac Sim commands.
    """

    source_command: str
    current_position: Point2D | None = None
    pen_state: PenState = "up"
    actions: list[PrimitiveAction] = field(default_factory=list)
    strokes: list[Stroke] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    failed_calls: list[str] = field(default_factory=list)
    finished: bool = False
    frame: Literal["board"] = "board"
    config: PlannerConfig = field(default_factory=PlannerConfig)

    @classmethod
    def begin_plan(
        cls,
        source_command: str,
        config: PlannerConfig | None = None,
    ) -> "PlanBuilder":
        """Create a fresh plan-builder state for a user command."""

        return cls(source_command=source_command, config=config or PlannerConfig())

    def move_to_start(self, x: float, y: float) -> dict[str, Any]:
        """Append a symbolic move_to_start primitive if the pen is up."""

        if self.pen_state != "up":
            return self._record_failed_call("move_to_start requires pen_state == 'up'.")
        target = Point2D(x=x, y=y)
        if not self._point_inside_board(target):
            return self._record_failed_call(self._point_outside_message(target, "move_to_start"))
        self.current_position = target
        self.actions.append(
            PrimitiveAction(
                name="move_to_start",
                frame=self.frame,
                stroke_id=None,
                params={
                    "target": self._point3d(target, self.config.hover_height_m),
                    "hover_height_m": self.config.hover_height_m,
                    "note": "free-space move; kinematics module converts board frame to base frame",
                },
            )
        )
        return self.check_plan()

    def align_pen_orientation(self) -> dict[str, Any]:
        """Append a symbolic board-normal orientation primitive."""

        self.actions.append(
            PrimitiveAction(
                name="align_pen_orientation",
                frame=self.frame,
                stroke_id=None,
                params={
                    "mode": "normal_to_board",
                    "board_normal_axis": "+z",
                    "pen_axis_target": "-z",
                    "note": "orientation constraint only; no IK computed here",
                },
            )
        )
        return self.check_plan()

    def pen_down(self) -> dict[str, Any]:
        """Append a pen_down primitive at the current board position."""

        if self.current_position is None:
            return self._record_failed_call("pen_down requires current_position is not None.")
        if self.pen_state == "down":
            return self._record_failed_call("pen_down requires pen_state == 'up'.")
        self.pen_state = "down"
        self.actions.append(
            PrimitiveAction(
                name="pen_down",
                frame=self.frame,
                stroke_id=None,
                params={
                    "target": self._point3d(self.current_position, self.config.drawing_z_m),
                    "approach_axis": "-z",
                    "speed_m_s": self.config.pen_down_speed_m_s,
                },
            )
        )
        return self.check_plan()

    def draw_line_to(self, x: float, y: float) -> dict[str, Any]:
        """Append a draw_line primitive from current_position to the given point."""

        if self.pen_state != "down":
            return self._record_failed_call("draw_line_to requires pen_state == 'down'.")
        if self.current_position is None:
            return self._record_failed_call("draw_line_to requires current_position is not None.")
        start = self.current_position
        end = Point2D(x=x, y=y)
        if not self._point_inside_board(end):
            return self._record_failed_call(self._point_outside_message(end, "draw_line_to"))
        stroke_id = self._next_stroke_id()
        stroke = LineStroke(stroke_id=stroke_id, start=start, end=end)
        self.strokes.append(stroke)
        self.actions.append(
            PrimitiveAction(
                name="draw_line",
                frame=self.frame,
                stroke_id=stroke_id,
                params={
                    "start": self._point3d(start, self.config.drawing_z_m),
                    "end": self._point3d(end, self.config.drawing_z_m),
                    "speed_m_s": self.config.default_speed_m_s,
                    "sampling_hint": "kinematics module should interpolate Cartesian waypoints",
                },
            )
        )
        self.current_position = end
        return self.check_plan()

    def draw_arc(
        self,
        center_x: float,
        center_y: float,
        radius_m: float,
        start_angle_rad: float,
        end_angle_rad: float,
        direction: Direction,
    ) -> dict[str, Any]:
        """Append a draw_arc primitive.

        The arc is symbolic geometry for the downstream kinematics module; this
        method does not sample trajectories.
        """

        if self.pen_state != "down":
            return self._record_failed_call("draw_arc requires pen_state == 'down'.")
        if radius_m <= 0:
            return self._record_failed_call("draw_arc requires radius_m > 0.")
        if self.current_position is None:
            return self._record_failed_call("draw_arc requires current_position is not None.")
        center = Point2D(x=center_x, y=center_y)
        if not self._arc_inside_board(center, radius_m):
            return self._record_failed_call(
                self._arc_outside_message(center, radius_m, "draw_arc")
            )
        expected_start = self._arc_start_point(center, radius_m, start_angle_rad)
        if not self._points_close(self.current_position, expected_start, tolerance=1e-6):
            return self._record_failed_call(
                self._arc_start_mismatch_message(self.current_position, expected_start)
            )
        stroke_id = self._next_stroke_id()
        stroke = ArcStroke(
            stroke_id=stroke_id,
            center=center,
            radius_m=radius_m,
            start_angle_rad=start_angle_rad,
            end_angle_rad=end_angle_rad,
            direction=direction,
        )
        self.strokes.append(stroke)
        self.actions.append(
            PrimitiveAction(
                name="draw_arc",
                frame=self.frame,
                stroke_id=stroke_id,
                params={
                    "center": self._point3d(center, self.config.drawing_z_m),
                    "radius_m": radius_m,
                    "start_angle_rad": start_angle_rad,
                    "end_angle_rad": end_angle_rad,
                    "direction": direction,
                    "speed_m_s": self.config.default_speed_m_s,
                    "sampling_hint": "kinematics module should sample circular Cartesian waypoints",
                },
            )
        )
        self.current_position = self._arc_end_point(stroke)
        return self.check_plan()

    def pen_up(self) -> dict[str, Any]:
        """Append a pen_up primitive."""

        if self.pen_state != "down":
            return self._record_failed_call("pen_up requires pen_state == 'down'.")
        self.pen_state = "up"
        self.actions.append(
            PrimitiveAction(
                name="pen_up",
                frame=self.frame,
                stroke_id=None,
                params={
                    "lift_height_m": self.config.hover_height_m,
                    "speed_m_s": self.config.pen_up_speed_m_s,
                },
            )
        )
        return self.check_plan()

    def check_plan(self) -> dict[str, Any]:
        """Return a machine-readable summary of builder state."""

        return {
            "current_position": (
                self.current_position.model_dump(mode="json")
                if self.current_position is not None
                else None
            ),
            "pen_state": self.pen_state,
            "number_of_actions": len(self.actions),
            "number_of_strokes": len(self.strokes),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "failed_calls": list(self.failed_calls),
            "finished": self.finished,
        }

    def finish_plan(self) -> dict[str, Any]:
        """Mark the plan complete if at least one drawable stroke exists."""

        if not self.strokes:
            return self._record_failed_call(
                "finish_plan requires at least one drawable stroke."
            )
        if self.pen_state == "down":
            return self._record_failed_call(
                "finish_plan requires pen_state == 'up'; call pen_up before finish_plan."
            )
        self.finished = True
        return self.check_plan()

    def _record_error(self, message: str) -> dict[str, Any]:
        self.errors.append(message)
        return self.check_plan()

    def _record_failed_call(self, message: str) -> dict[str, Any]:
        self.failed_calls.append(message)
        return self.check_plan()

    @staticmethod
    def _point3d(point: Point2D, z: float) -> dict[str, float | str]:
        return Point3D(x=point.x, y=point.y, z=z).model_dump(mode="json")

    @staticmethod
    def _arc_start_point(center: Point2D, radius_m: float, start_angle_rad: float) -> Point2D:
        return Point2D(
            x=center.x + radius_m * math.cos(start_angle_rad),
            y=center.y + radius_m * math.sin(start_angle_rad),
        )

    @staticmethod
    def _points_close(a: Point2D, b: Point2D, tolerance: float) -> bool:
        return math.isclose(a.x, b.x, abs_tol=tolerance) and math.isclose(
            a.y,
            b.y,
            abs_tol=tolerance,
        )

    def _point_inside_board(self, point: Point2D) -> bool:
        x_min, x_max, y_min, y_max = self._board_bounds()
        return x_min <= point.x <= x_max and y_min <= point.y <= y_max

    def _arc_inside_board(self, center: Point2D, radius_m: float) -> bool:
        x_min, x_max, y_min, y_max = self._board_bounds()
        return (
            center.x - radius_m >= x_min
            and center.x + radius_m <= x_max
            and center.y - radius_m >= y_min
            and center.y + radius_m <= y_max
        )

    def _board_bounds(self) -> tuple[float, float, float, float]:
        half_width = self.config.board_width_m / 2.0
        half_height = self.config.board_height_m / 2.0
        return -half_width, half_width, -half_height, half_height

    def _point_outside_message(self, point: Point2D, tool_name: str) -> str:
        x_min, x_max, y_min, y_max = self._board_bounds()
        return (
            f"{tool_name} target ({point.x}, {point.y}) is outside the board bounds: "
            f"x must be in [{x_min}, {x_max}], y must be in [{y_min}, {y_max}]."
        )

    def _arc_outside_message(self, center: Point2D, radius_m: float, tool_name: str) -> str:
        x_min, x_max, y_min, y_max = self._board_bounds()
        return (
            f"{tool_name} arc bounding box is outside the board bounds: "
            f"center=({center.x}, {center.y}), radius_m={radius_m}; "
            f"x range [{center.x - radius_m}, {center.x + radius_m}] must fit in "
            f"[{x_min}, {x_max}], y range [{center.y - radius_m}, {center.y + radius_m}] "
            f"must fit in [{y_min}, {y_max}]."
        )

    @staticmethod
    def _arc_start_mismatch_message(current: Point2D, expected_start: Point2D) -> str:
        return (
            "draw_arc requires current_position to match the arc start point within "
            f"tolerance 1e-06. Current position is ({current.x}, {current.y}); "
            f"expected arc start is ({expected_start.x}, {expected_start.y}). "
            "Call pen_up if the pen is down, then "
            f"move_to_start(x={expected_start.x}, y={expected_start.y}), pen_down, "
            "then draw_arc with the same arc parameters."
        )

    def _next_stroke_id(self) -> str:
        return f"stroke_{len(self.strokes) + 1:03d}"

    @staticmethod
    def _arc_end_point(stroke: ArcStroke) -> Point2D:
        if math.isclose(
            abs(stroke.end_angle_rad - stroke.start_angle_rad),
            2.0 * math.pi,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return Point2D(
                x=stroke.center.x + stroke.radius_m * math.cos(stroke.start_angle_rad),
                y=stroke.center.y + stroke.radius_m * math.sin(stroke.start_angle_rad),
            )
        return Point2D(
            x=stroke.center.x + stroke.radius_m * math.cos(stroke.end_angle_rad),
            y=stroke.center.y + stroke.radius_m * math.sin(stroke.end_angle_rad),
        )
