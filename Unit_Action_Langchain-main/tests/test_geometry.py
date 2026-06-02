import math

import pytest

from robot_drawing_planner.geometry import (
    circle_arc,
    rotate_point_around_center,
    square_vertices,
    strokes_for_shape,
)
from robot_drawing_planner.schemas import ArcStroke, LineStroke, NormalizedGoal, Point2D


def goal(shape_type, size_m=0.10, letter=None):
    return NormalizedGoal(
        shape_type=shape_type,
        letter=letter,
        radius_m=size_m / 2.0 if shape_type == "circle" else None,
        side_length_m=size_m if shape_type in {"square", "triangle"} else None,
        size_m=size_m,
        orientation_rad=0.0,
        center=Point2D(x=0.0, y=0.0),
        assumptions=[],
        warnings=[],
    )


def test_rotate_point_around_center_rotates_deterministically():
    rotated = rotate_point_around_center(
        Point2D(x=1.0, y=0.0),
        Point2D(x=0.0, y=0.0),
        math.pi / 2.0,
    )
    assert rotated.x == pytest.approx(0.0)
    assert rotated.y == pytest.approx(1.0)


def test_square_vertices_close_back_to_start_through_final_stroke():
    square_goal = goal("square")
    vertices = square_vertices(square_goal.center, square_goal.side_length_m, 0.0)
    strokes = strokes_for_shape(square_goal)

    assert len(vertices) == 4
    assert strokes[-1].end == strokes[0].start
    assert strokes[0].start == vertices[0]
    assert strokes[-1].end == vertices[0]


def test_circle_generates_one_full_arc():
    strokes = strokes_for_shape(goal("circle"))
    assert len(strokes) == 1
    assert isinstance(strokes[0], ArcStroke)
    assert strokes[0].radius_m == pytest.approx(0.05)
    assert strokes[0].start_angle_rad == pytest.approx(0.0)
    assert strokes[0].end_angle_rad == pytest.approx(2.0 * math.pi)
    assert strokes[0].direction == "ccw"


def test_circle_arc_uses_required_angles_and_direction():
    arc = circle_arc(Point2D(x=0.0, y=0.0), 0.05)
    assert arc.stroke_id == "stroke_001"
    assert arc.start_angle_rad == pytest.approx(0.0)
    assert arc.end_angle_rad == pytest.approx(2.0 * math.pi)
    assert arc.direction == "ccw"


def test_square_generates_four_lines():
    strokes = strokes_for_shape(goal("square"))
    assert len(strokes) == 4
    assert all(isinstance(stroke, LineStroke) for stroke in strokes)
    assert strokes[0].start.x == pytest.approx(-0.05)
    assert strokes[0].start.y == pytest.approx(0.05)
    assert strokes[-1].end == strokes[0].start


def test_triangle_generates_three_lines():
    strokes = strokes_for_shape(goal("triangle"))
    assert len(strokes) == 3
    assert all(isinstance(stroke, LineStroke) for stroke in strokes)
    assert strokes[-1].end == strokes[0].start


@pytest.mark.parametrize("letter,expected_count", [("A", 3), ("H", 3), ("L", 2), ("T", 2)])
def test_supported_line_letters(letter, expected_count):
    strokes = strokes_for_shape(goal("letter", letter=letter.lower()))
    assert len(strokes) == expected_count
    assert all(isinstance(stroke, LineStroke) for stroke in strokes)
    assert [stroke.stroke_id for stroke in strokes] == [
        f"stroke_{index:03d}" for index in range(1, expected_count + 1)
    ]


def test_letter_o_uses_circular_arc():
    strokes = strokes_for_shape(goal("letter", letter="O"))
    assert len(strokes) == 1
    assert isinstance(strokes[0], ArcStroke)
    assert strokes[0].radius_m == pytest.approx(0.05)


def test_unsupported_letter_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported letter"):
        strokes_for_shape(goal("letter", letter="B"))
