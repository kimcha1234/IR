import pytest
from pydantic import ValidationError

from robot_drawing_planner.config import DEFAULT_CONFIG, PlannerConfig
from robot_drawing_planner.schemas import Measurement, Point2D
from robot_drawing_planner.units import coordinate_to_meters, measurement_to_meters, to_meters


def test_to_meters_supported_units():
    assert to_meters(1, "m") == pytest.approx(1.0)
    assert to_meters(12, "cm") == pytest.approx(0.12)
    assert to_meters(25, "mm") == pytest.approx(0.025)


def test_measurement_to_meters_supported_units():
    assert measurement_to_meters(Measurement(value=1, unit="m")) == pytest.approx(1.0)
    assert measurement_to_meters(Measurement(value=12, unit="cm")) == pytest.approx(0.12)
    assert measurement_to_meters(Measurement(value=25, unit="mm")) == pytest.approx(0.025)


def test_coordinate_to_meters_allows_signed_values():
    assert coordinate_to_meters(-10, "cm") == pytest.approx(-0.10)


def test_to_meters_rejects_non_positive_size():
    with pytest.raises(ValueError, match="positive"):
        to_meters(0, "cm")


def test_measurement_model_rejects_non_positive_values():
    with pytest.raises(ValidationError):
        Measurement(value=0, unit="cm")


def test_measurement_to_meters_rejects_non_positive_constructed_value():
    measurement = Measurement.model_construct(value=0, unit="cm")
    with pytest.raises(ValueError, match="positive"):
        measurement_to_meters(measurement)


def test_to_meters_rejects_unsupported_unit():
    with pytest.raises(ValueError, match="Unsupported unit"):
        to_meters(1, "inch")


def test_default_planner_config_values():
    config = PlannerConfig()
    assert config.board_width_m == pytest.approx(0.50)
    assert config.board_height_m == pytest.approx(0.35)
    assert config.default_center == Point2D(x=0.0, y=0.0)
    assert config.default_shape_size_m == pytest.approx(0.10)
    assert config.default_circle_radius_m == pytest.approx(0.05)
    assert config.hover_height_m == pytest.approx(0.03)
    assert config.drawing_z_m == pytest.approx(0.0)
    assert config.default_speed_m_s == pytest.approx(0.03)
    assert config.pen_down_speed_m_s == pytest.approx(0.01)
    assert config.pen_up_speed_m_s == pytest.approx(0.02)
    assert config.arc_default_direction == "ccw"


def test_default_config_board_convention_values():
    board = DEFAULT_CONFIG.board()
    assert board.width_m == pytest.approx(0.50)
    assert board.height_m == pytest.approx(0.35)
    assert -board.width_m / 2 == pytest.approx(-0.25)
    assert board.width_m / 2 == pytest.approx(0.25)
    assert -board.height_m / 2 == pytest.approx(-0.175)
    assert board.height_m / 2 == pytest.approx(0.175)
