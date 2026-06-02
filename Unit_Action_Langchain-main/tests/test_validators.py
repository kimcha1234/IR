import math

import pytest
from pydantic import ValidationError

from robot_drawing_planner.config import PlannerConfig
from robot_drawing_planner.geometry import generate_strokes
from robot_drawing_planner.schemas import (
    ArcStroke,
    DrawingPlan,
    LineStroke,
    Measurement,
    NormalizedGoal,
    ParsedGoal,
    Point2D,
    Point3D,
    PrimitiveAction,
    ValidationErrorReport,
)
from robot_drawing_planner.validators import (
    IK_FEASIBILITY_WARNING,
    PlannerValidationError,
    normalize_goal,
    validate_normalized_goal,
    validate_no_low_level_robot_fields,
    validate_strokes_inside_board,
)


CONFIG = PlannerConfig()


def test_instantiates_every_schema_model():
    point_2d = Point2D(x=0.0, y=0.0)
    point_3d = Point3D(x=0.0, y=0.0, z=0.1)
    measurement = Measurement(value=5.0, unit="cm")
    parsed = ParsedGoal(
        shape_type="circle",
        center=point_2d,
        radius=measurement,
        raw_command="draw a circle",
    )
    normalized = NormalizedGoal(
        shape_type="circle",
        center=point_2d,
        radius_m=0.05,
        side_length_m=None,
        size_m=0.10,
        orientation_rad=0.0,
        letter=None,
        assumptions=[],
        warnings=[],
    )
    line = LineStroke(stroke_id="stroke_line", start=point_2d, end=Point2D(x=0.1, y=0.0))
    arc = ArcStroke(
        stroke_id="stroke_arc",
        center=point_2d,
        radius_m=0.05,
        start_angle_rad=0.0,
        end_angle_rad=math.pi,
        direction="ccw",
    )
    action = PrimitiveAction(
        name="draw_line",
        stroke_id=line.stroke_id,
        params={"start": line.start.model_dump(), "end": line.end.model_dump()},
    )
    plan = DrawingPlan(
        source_command=parsed.raw_command,
        goal=normalized,
        strokes=[line, arc],
        actions=[action],
        diagnostics={"point_3d_unit": point_3d.unit},
    )
    report = ValidationErrorReport(ok=True, errors=[], warnings=[])

    assert plan.schema_version == "1.0"
    assert report.ok is True


def test_measurement_rejects_non_positive_values():
    with pytest.raises(ValidationError):
        Measurement(value=0, unit="cm")


def test_point2d_rejects_extra_fields():
    with pytest.raises(ValidationError):
        Point2D(x=0.0, y=0.0, z=0.0)


def test_primitive_action_rejects_invalid_name():
    with pytest.raises(ValidationError):
        PrimitiveAction(name="solve_ik", params={})


def test_parsed_goal_rejects_unsupported_shape():
    with pytest.raises(ValidationError):
        ParsedGoal(shape_type="star", size=Measurement(value=5, unit="cm"), raw_command="draw a star")


def test_normalize_goal_rejects_unsupported_letter():
    with pytest.raises(PlannerValidationError, match="Unsupported letter"):
        normalize_goal(
            ParsedGoal(
                shape_type="letter",
                letter="B",
                size=Measurement(value=5, unit="cm"),
                raw_command="draw B",
            ),
            CONFIG,
        )


def test_missing_center_uses_default_center_assumption():
    goal = normalize_goal(
        ParsedGoal(
            shape_type="circle",
            raw_command="draw a circle",
        ),
        CONFIG,
    )
    assert goal.center == CONFIG.default_center
    assert "center was not specified; default board center was used" in goal.assumptions


def test_normalize_goal_converts_cm_to_m():
    goal = normalize_goal(
        ParsedGoal(
            shape_type="square",
            center=Point2D(x=0.10, y=-0.05),
            side_length=Measurement(value=10, unit="cm"),
            raw_command="draw a square",
        ),
        CONFIG,
    )
    assert goal.size_m == pytest.approx(0.10)
    assert goal.side_length_m == pytest.approx(0.10)
    assert goal.center.x == pytest.approx(0.10)
    assert goal.center.y == pytest.approx(-0.05)


def test_circle_size_is_interpreted_as_diameter():
    goal = normalize_goal(
        ParsedGoal(
            shape_type="circle",
            size=Measurement(value=20, unit="cm"),
            raw_command="draw a 20cm diameter circle",
        ),
        CONFIG,
    )

    assert goal.radius_m == pytest.approx(0.10)
    assert goal.size_m == pytest.approx(0.20)


def test_circle_radius_wins_over_size_with_warning():
    goal = normalize_goal(
        ParsedGoal(
            shape_type="circle",
            radius=Measurement(value=5, unit="cm"),
            size=Measurement(value=20, unit="cm"),
            raw_command="draw a circle with radius and size",
        ),
        CONFIG,
    )

    assert goal.radius_m == pytest.approx(0.05)
    assert any("radius and size" in warning for warning in goal.warnings)


def test_square_and_triangle_size_is_interpreted_as_side_length():
    square = normalize_goal(
        ParsedGoal(
            shape_type="square",
            size=Measurement(value=20, unit="cm"),
            raw_command="draw a 20cm square",
        ),
        CONFIG,
    )
    triangle = normalize_goal(
        ParsedGoal(
            shape_type="triangle",
            size=Measurement(value=12, unit="cm"),
            raw_command="draw a 12cm triangle",
        ),
        CONFIG,
    )

    assert square.side_length_m == pytest.approx(0.20)
    assert square.size_m == pytest.approx(0.20)
    assert triangle.side_length_m == pytest.approx(0.12)
    assert triangle.size_m == pytest.approx(0.12)


def test_side_length_wins_over_size_with_warning():
    goal = normalize_goal(
        ParsedGoal(
            shape_type="square",
            side_length=Measurement(value=10, unit="cm"),
            size=Measurement(value=20, unit="cm"),
            raw_command="draw a square with side length and size",
        ),
        CONFIG,
    )

    assert goal.side_length_m == pytest.approx(0.10)
    assert any("side_length and size" in warning for warning in goal.warnings)


def test_negative_size_error():
    parsed = ParsedGoal.model_construct(
        shape_type="square",
        center=Point2D(x=0.0, y=0.0),
        side_length=Measurement.model_construct(value=-5, unit="cm"),
        size=None,
        radius=None,
        orientation_deg=0.0,
        letter=None,
        position_hint=None,
        raw_command="draw a square",
    )
    with pytest.raises(PlannerValidationError, match="positive"):
        normalize_goal(parsed, CONFIG)


def test_board_boundary_violation_report_says_validation_failed():
    goal = normalize_goal(
        ParsedGoal(
            shape_type="square",
            side_length=Measurement(value=50, unit="cm"),
            raw_command="draw a square",
        ),
        CONFIG,
    )
    strokes = generate_strokes(goal)
    report = validate_strokes_inside_board(strokes, CONFIG)
    assert report.ok is False
    assert any("validation failed" in error for error in report.errors)


def test_ik_feasibility_warning_exists():
    goal = normalize_goal(
        ParsedGoal(shape_type="circle", raw_command="draw a circle"),
        CONFIG,
    )
    goal_report = validate_normalized_goal(goal, CONFIG)
    strokes_report = validate_strokes_inside_board(generate_strokes(goal), CONFIG)
    assert IK_FEASIBILITY_WARNING in goal.warnings
    assert IK_FEASIBILITY_WARNING in goal_report.warnings
    assert IK_FEASIBILITY_WARNING in strokes_report.warnings


def test_validate_normalized_goal_reports_negative_constructed_radius():
    goal = NormalizedGoal.model_construct(
        shape_type="circle",
        center=Point2D(x=0.0, y=0.0),
        radius_m=-0.01,
        side_length_m=None,
        size_m=None,
        orientation_rad=0.0,
        letter=None,
        frame="board",
        assumptions=[],
        warnings=[],
    )
    report = validate_normalized_goal(goal, CONFIG)
    assert report.ok is False
    assert any("positive" in error for error in report.errors)


def test_no_low_level_robot_fields_rejects_ik_claims():
    with pytest.raises(PlannerValidationError, match="outside planner scope"):
        validate_no_low_level_robot_fields({"actions": [{"ik": "solved"}]})
