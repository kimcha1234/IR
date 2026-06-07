"""Isaac viewport debug visualization helpers for drawing runs.

This module is intentionally optional. It imports the Isaac Debug Draw API only
when a visualizer is created, so offline tests and the core controller can run
without Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


Color = tuple[float, float, float, float]


@dataclass(frozen=True)
class DebugDrawConfig:
    """Runtime-only viewport visualization settings."""

    enabled: bool = False
    update_stride: int = 5
    planned_path_stride: int = 5
    actual_path_max_points: int = 2500
    draw_planned_path: bool = True
    draw_actual_path: bool = True
    draw_contact_force: bool = False
    draw_tip_frame: bool = True
    planned_color: Color = (0.10, 0.45, 1.0, 0.85)
    actual_color: Color = (1.0, 0.12, 0.08, 0.95)
    force_color: Color = (1.0, 0.75, 0.05, 0.95)
    x_axis_color: Color = (1.0, 0.05, 0.05, 0.95)
    y_axis_color: Color = (0.05, 0.85, 0.10, 0.95)
    z_axis_color: Color = (0.10, 0.35, 1.0, 0.95)
    planned_line_width: float = 2.0
    actual_line_width: float = 3.0
    actual_width_from_force: bool = True
    actual_force_width_per_n: float = 5.0
    actual_max_line_width: float = 12.0
    frame_line_width: float = 3.0
    force_min_line_width: float = 2.0
    force_width_per_n: float = 2.0
    force_max_line_width: float = 12.0
    force_scale_m_per_n: float = 0.04
    force_max_length_m: float = 0.12
    force_min_draw_n: float = 0.02
    frame_axis_length_m: float = 0.045


class IsaacDebugDrawVisualizer:
    """Draw planned/actual trajectory, contact force, and pen-tip frame.

    The class keeps all drawing state local and redraws its own lines at a
    fixed stride. It does not write commands back into the robot or controller.
    """

    def __init__(self, draw_interface: Any, config: DebugDrawConfig | None = None) -> None:
        self.draw = draw_interface
        self.config = config or DebugDrawConfig(enabled=True)
        self._planned_points: list[tuple[float, float, float]] = []
        self._actual_points: list[tuple[float, float, float]] = []
        self._actual_widths: list[float] = []
        self._last_force_segment: tuple[tuple[float, float, float], tuple[float, float, float], float] | None = None
        self._last_frame_segments: list[tuple[tuple[float, float, float], tuple[float, float, float], Color]] = []

    def set_planned_path(self, points: np.ndarray) -> None:
        """Store a subsampled planned path for repeated viewport redraws."""

        if not self.config.enabled or not self.config.draw_planned_path:
            return
        self._planned_points = _subsample_points(points, max(1, int(self.config.planned_path_stride)))
        self.redraw()

    def update(
        self,
        *,
        step_index: int,
        actual_tip_transform: np.ndarray,
        measured_normal_force_n: float | None,
        board_normal_base: np.ndarray,
        record_actual_path: bool = True,
    ) -> None:
        """Update actual path history and live overlays."""

        if not self.config.enabled:
            return
        stride = max(1, int(self.config.update_stride))
        if int(step_index) % stride != 0:
            return

        transform = np.asarray(actual_tip_transform, dtype=float)
        if transform.shape != (4, 4):
            raise ValueError(f"actual_tip_transform must have shape (4, 4), got {transform.shape}.")
        origin = _as_point(transform[:3, 3])
        if self.config.draw_actual_path and record_actual_path:
            self._actual_points.append(origin)
            self._actual_widths.append(self._actual_line_width(measured_normal_force_n))
            max_points = max(2, int(self.config.actual_path_max_points))
            if len(self._actual_points) > max_points:
                self._actual_points = self._actual_points[-max_points:]
                self._actual_widths = self._actual_widths[-max_points:]

        self._last_force_segment = None
        if self.config.draw_contact_force and measured_normal_force_n is not None:
            self._last_force_segment = self._force_segment(origin, measured_normal_force_n, board_normal_base)

        self._last_frame_segments = []
        if self.config.draw_tip_frame:
            self._last_frame_segments = self._frame_segments(transform)

        self.redraw()

    def redraw(self) -> None:
        """Redraw all current lines."""

        if not self.config.enabled:
            return
        starts: list[tuple[float, float, float]] = []
        ends: list[tuple[float, float, float]] = []
        colors: list[Color] = []
        widths: list[float] = []

        if self.config.draw_planned_path:
            _append_polyline(
                starts,
                ends,
                colors,
                widths,
                self._planned_points,
                self.config.planned_color,
                self.config.planned_line_width,
            )
        if self.config.draw_actual_path:
            _append_polyline_variable_width(
                starts,
                ends,
                colors,
                widths,
                self._actual_points,
                self.config.actual_color,
                self._actual_widths,
                self.config.actual_line_width,
            )
        if self._last_force_segment is not None:
            start, end, width = self._last_force_segment
            starts.append(start)
            ends.append(end)
            colors.append(self.config.force_color)
            widths.append(width)
        for start, end, color in self._last_frame_segments:
            starts.append(start)
            ends.append(end)
            colors.append(color)
            widths.append(float(self.config.frame_line_width))

        if hasattr(self.draw, "clear_lines"):
            self.draw.clear_lines()
        if starts:
            self.draw.draw_lines(starts, ends, colors, widths)

    def clear(self) -> None:
        """Clear viewport lines owned through the Debug Draw interface."""

        if hasattr(self.draw, "clear_lines"):
            self.draw.clear_lines()

    def _actual_line_width(self, measured_normal_force_n: float | None) -> float:
        base_width = float(self.config.actual_line_width)
        if not self.config.actual_width_from_force or measured_normal_force_n is None:
            return base_width
        force_n = max(0.0, abs(float(measured_normal_force_n)))
        return min(base_width + force_n * float(self.config.actual_force_width_per_n), float(self.config.actual_max_line_width))

    def _force_segment(
        self,
        origin: tuple[float, float, float],
        measured_normal_force_n: float,
        board_normal_base: np.ndarray,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], float] | None:
        force_n = abs(float(measured_normal_force_n))
        if force_n < float(self.config.force_min_draw_n):
            return None
        normal = _normalized(board_normal_base)
        length = min(force_n * float(self.config.force_scale_m_per_n), float(self.config.force_max_length_m))
        start = np.asarray(origin, dtype=float)
        end = start + normal * length
        width = min(
            float(self.config.force_min_line_width) + force_n * float(self.config.force_width_per_n),
            float(self.config.force_max_line_width),
        )
        return _as_point(start), _as_point(end), width

    def _frame_segments(
        self,
        transform: np.ndarray,
    ) -> list[tuple[tuple[float, float, float], tuple[float, float, float], Color]]:
        origin = np.asarray(transform[:3, 3], dtype=float)
        rotation = np.asarray(transform[:3, :3], dtype=float)
        axis_length = float(self.config.frame_axis_length_m)
        colors = [self.config.x_axis_color, self.config.y_axis_color, self.config.z_axis_color]
        segments = []
        for axis_index, color in enumerate(colors):
            end = origin + axis_length * rotation[:, axis_index]
            segments.append((_as_point(origin), _as_point(end), color))
        return segments


def create_isaac_debug_draw_visualizer(
    config: DebugDrawConfig,
    *,
    log_fn: Callable[[str], None] | None = None,
) -> IsaacDebugDrawVisualizer | None:
    """Create a visualizer if Isaac Debug Draw is available."""

    if not config.enabled:
        return None
    try:
        from isaacsim.util.debug_draw import _debug_draw

        draw = _debug_draw.acquire_debug_draw_interface()
    except Exception as exc:  # pragma: no cover - depends on Isaac runtime.
        if log_fn is not None:
            log_fn(f"Debug Draw unavailable; viewport visualization disabled. Reason: {exc}")
        return None
    return IsaacDebugDrawVisualizer(draw, config)


def _subsample_points(points: np.ndarray, stride: int) -> list[tuple[float, float, float]]:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {arr.shape}.")
    if arr.shape[0] == 0:
        return []
    selected = arr[:: max(1, int(stride))]
    if not np.allclose(selected[-1], arr[-1]):
        selected = np.vstack([selected, arr[-1]])
    return [_as_point(point) for point in selected]


def _append_polyline(
    starts: list[tuple[float, float, float]],
    ends: list[tuple[float, float, float]],
    colors: list[Color],
    widths: list[float],
    points: list[tuple[float, float, float]],
    color: Color,
    width: float,
) -> None:
    if len(points) < 2:
        return
    for start, end in zip(points[:-1], points[1:]):
        starts.append(start)
        ends.append(end)
        colors.append(color)
        widths.append(float(width))


def _append_polyline_variable_width(
    starts: list[tuple[float, float, float]],
    ends: list[tuple[float, float, float]],
    colors: list[Color],
    widths: list[float],
    points: list[tuple[float, float, float]],
    color: Color,
    point_widths: list[float],
    fallback_width: float,
) -> None:
    if len(points) < 2:
        return
    widths_arr = [float(value) for value in point_widths]
    if len(widths_arr) != len(points):
        widths_arr = [float(fallback_width)] * len(points)
    for i, (start, end) in enumerate(zip(points[:-1], points[1:])):
        starts.append(start)
        ends.append(end)
        colors.append(color)
        widths.append(0.5 * (widths_arr[i] + widths_arr[i + 1]))


def _as_point(value: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"point must have shape (3,), got {arr.shape}.")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _normalized(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError("normal vector must be nonzero.")
    return arr / norm
