"""Deterministic stroke geometry generation."""

from __future__ import annotations

import math

from robot_drawing_planner.schemas import ArcStroke, LineStroke, NormalizedGoal, Point2D, Stroke


def rotate_point_around_center(point: Point2D, center: Point2D, angle_rad: float) -> Point2D:
    """Rotate a board-frame point around a board-frame center."""

    translated_x = point.x - center.x
    translated_y = point.y - center.y
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return Point2D(
        x=center.x + translated_x * cos_a - translated_y * sin_a,
        y=center.y + translated_x * sin_a + translated_y * cos_a,
    )


def square_vertices(
    center: Point2D,
    side_length_m: float,
    orientation_rad: float,
) -> list[Point2D]:
    """Return four square vertices in deterministic drawing order."""

    _require_positive(side_length_m, "side_length_m")
    half = side_length_m / 2.0
    vertices = [
        Point2D(x=center.x - half, y=center.y + half),
        Point2D(x=center.x + half, y=center.y + half),
        Point2D(x=center.x + half, y=center.y - half),
        Point2D(x=center.x - half, y=center.y - half),
    ]
    return [rotate_point_around_center(vertex, center, orientation_rad) for vertex in vertices]


def triangle_vertices(
    center: Point2D,
    side_length_m: float,
    orientation_rad: float,
) -> list[Point2D]:
    """Return equilateral triangle vertices in deterministic drawing order."""

    _require_positive(side_length_m, "side_length_m")
    height = math.sqrt(3.0) * side_length_m / 2.0
    vertices = [
        Point2D(x=center.x, y=center.y + 2.0 * height / 3.0),
        Point2D(x=center.x + side_length_m / 2.0, y=center.y - height / 3.0),
        Point2D(x=center.x - side_length_m / 2.0, y=center.y - height / 3.0),
    ]
    return [rotate_point_around_center(vertex, center, orientation_rad) for vertex in vertices]


def circle_arc(center: Point2D, radius_m: float) -> ArcStroke:
    """Return one counterclockwise full-circle arc stroke."""

    _require_positive(radius_m, "radius_m")
    return ArcStroke(
        stroke_id="stroke_001",
        center=center,
        radius_m=radius_m,
        start_angle_rad=0.0,
        end_angle_rad=2.0 * math.pi,
        direction="ccw",
    )


def strokes_for_shape(goal: NormalizedGoal) -> list[Stroke]:
    """Generate deterministic board-frame strokes for a normalized goal."""

    if goal.shape_type == "circle":
        if goal.radius_m is None:
            raise ValueError("Circle goals require radius_m.")
        return [circle_arc(goal.center, goal.radius_m)]
    if goal.shape_type == "square":
        if goal.side_length_m is None:
            raise ValueError("Square goals require side_length_m.")
        return _closed_polyline(square_vertices(goal.center, goal.side_length_m, goal.orientation_rad))
    if goal.shape_type == "triangle":
        if goal.side_length_m is None:
            raise ValueError("Triangle goals require side_length_m.")
        return _closed_polyline(
            triangle_vertices(goal.center, goal.side_length_m, goal.orientation_rad)
        )
    if goal.shape_type == "letter":
        return strokes_for_letter(goal)
    raise ValueError(f"Unsupported drawing shape: {goal.shape_type}")


def strokes_for_letter(goal: NormalizedGoal) -> list[Stroke]:
    """Generate deterministic single-line strokes for supported uppercase letters."""

    if goal.size_m is None:
        raise ValueError("Letter goals require size_m.")

    letter = (goal.letter or "").upper()
    if letter == "A":
        return _letter_a(goal)
    if letter == "H":
        return _letter_h(goal)
    if letter == "L":
        return _letter_l(goal)
    if letter == "T":
        return _letter_t(goal)
    if letter == "O":
        return [circle_arc(goal.center, goal.size_m / 2.0)]
    raise ValueError(
        f"Unsupported letter '{goal.letter}'. Supported letters: A, H, L, T, O."
    )


def generate_strokes(goal: NormalizedGoal) -> list[Stroke]:
    """Backward-compatible alias for deterministic stroke generation."""

    return strokes_for_shape(goal)


def stroke_start_point(stroke: Stroke) -> Point2D:
    """Return the first point of a line or arc stroke."""

    if isinstance(stroke, LineStroke):
        return stroke.start
    return _arc_point(stroke, stroke.start_angle_rad)


def stroke_end_point(stroke: Stroke) -> Point2D:
    """Return the last point of a line or arc stroke."""

    if isinstance(stroke, LineStroke):
        return stroke.end
    if math.isclose(
        abs(stroke.end_angle_rad - stroke.start_angle_rad),
        2.0 * math.pi,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return stroke_start_point(stroke)
    return _arc_point(stroke, stroke.end_angle_rad)


def _letter_box(goal: NormalizedGoal) -> tuple[float, float, float, float]:
    size = goal.size_m
    if size is None:
        raise ValueError("Letter goals require size_m.")
    half = size / 2.0
    return -half, half, -half, half


def _letter_point(goal: NormalizedGoal, x_offset: float, y_offset: float) -> Point2D:
    local_point = Point2D(x=goal.center.x + x_offset, y=goal.center.y + y_offset)
    return rotate_point_around_center(local_point, goal.center, goal.orientation_rad)


def _letter_a(goal: NormalizedGoal) -> list[Stroke]:
    left, right, bottom, top = _letter_box(goal)
    return [
        _line("stroke_001", _letter_point(goal, left, bottom), _letter_point(goal, 0.0, top)),
        _line("stroke_002", _letter_point(goal, 0.0, top), _letter_point(goal, right, bottom)),
        _line("stroke_003", _letter_point(goal, left, 0.0), _letter_point(goal, right, 0.0)),
    ]


def _letter_h(goal: NormalizedGoal) -> list[Stroke]:
    left, right, bottom, top = _letter_box(goal)
    return [
        _line("stroke_001", _letter_point(goal, left, top), _letter_point(goal, left, bottom)),
        _line("stroke_002", _letter_point(goal, right, top), _letter_point(goal, right, bottom)),
        _line("stroke_003", _letter_point(goal, left, 0.0), _letter_point(goal, right, 0.0)),
    ]


def _letter_l(goal: NormalizedGoal) -> list[Stroke]:
    left, right, bottom, top = _letter_box(goal)
    return [
        _line("stroke_001", _letter_point(goal, left, top), _letter_point(goal, left, bottom)),
        _line("stroke_002", _letter_point(goal, left, bottom), _letter_point(goal, right, bottom)),
    ]


def _letter_t(goal: NormalizedGoal) -> list[Stroke]:
    left, right, bottom, top = _letter_box(goal)
    return [
        _line("stroke_001", _letter_point(goal, left, top), _letter_point(goal, right, top)),
        _line("stroke_002", _letter_point(goal, 0.0, top), _letter_point(goal, 0.0, bottom)),
    ]


def _closed_polyline(points: list[Point2D]) -> list[Stroke]:
    strokes: list[Stroke] = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        strokes.append(_line(f"stroke_{index + 1:03d}", start, end))
    return strokes


def _line(stroke_id: str, start: Point2D, end: Point2D) -> LineStroke:
    return LineStroke(stroke_id=stroke_id, start=start, end=end)


def _arc_point(stroke: ArcStroke, angle_rad: float) -> Point2D:
    return Point2D(
        x=stroke.center.x + stroke.radius_m * math.cos(angle_rad),
        y=stroke.center.y + stroke.radius_m * math.sin(angle_rad),
    )


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")

