import numpy as np

from franka_llm_drawing.trajectory.time_scaling import (
    cubic_time_scaling,
    cubic_time_scaling_derivative,
    cubic_time_scaling_third_derivative,
    quintic_time_scaling,
    quintic_time_scaling_derivative,
    quintic_time_scaling_second_derivative,
    quintic_time_scaling_third_derivative,
)
from franka_llm_drawing.trajectory import TrajectoryLimits, duration_for_path_length, sample_line


def test_cubic_scaling_endpoints() -> None:
    assert cubic_time_scaling(0.0, 2.0) == 0.0
    assert cubic_time_scaling(2.0, 2.0) == 1.0


def test_cubic_scaling_is_monotonic() -> None:
    values = [cubic_time_scaling(t, 1.0) for t in np.linspace(0.0, 1.0, 51)]

    assert all(a <= b for a, b in zip(values, values[1:]))


def test_cubic_derivative_zero_at_endpoints() -> None:
    assert np.isclose(cubic_time_scaling_derivative(0.0, 1.0), 0.0)
    assert np.isclose(cubic_time_scaling_derivative(1.0, 1.0), 0.0)


def test_quintic_scaling_boundary_velocity_and_acceleration() -> None:
    assert np.isclose(quintic_time_scaling(0.0, 1.0), 0.0)
    assert np.isclose(quintic_time_scaling(1.0, 1.0), 1.0)
    assert np.isclose(quintic_time_scaling_derivative(0.0, 1.0), 0.0)
    assert np.isclose(quintic_time_scaling_derivative(1.0, 1.0), 0.0)
    assert np.isclose(quintic_time_scaling_second_derivative(0.0, 1.0), 0.0)
    assert np.isclose(quintic_time_scaling_second_derivative(1.0, 1.0), 0.0)


def test_time_scaling_jerk_is_available_for_planning_limits() -> None:
    assert np.isclose(cubic_time_scaling_third_derivative(0.0, 2.0), -1.5)
    assert np.isclose(quintic_time_scaling_third_derivative(0.0, 2.0), 7.5)


def test_retiming_respects_peak_velocity_limit_for_quintic_line() -> None:
    duration = duration_for_path_length(
        0.10,
        1.0,
        0.01,
        time_scaling="quintic",
        limits=TrajectoryLimits(max_linear_speed_m_s=0.05),
    )

    assert duration >= 3.75


def test_sample_line_retiming_reduces_peak_speed() -> None:
    samples = sample_line(
        np.array([0.0, 0.0, 0.0]),
        np.array([0.10, 0.0, 0.0]),
        speed_m_s=1.0,
        dt=0.01,
        time_scaling="quintic",
        limits=TrajectoryLimits(max_linear_speed_m_s=0.05),
    )
    peak_speed = max(float(np.linalg.norm(sample.linear_velocity)) for sample in samples)

    assert peak_speed <= 0.0501
