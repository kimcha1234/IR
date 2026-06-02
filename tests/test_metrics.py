import numpy as np

from franka_llm_drawing.evaluation.metrics import (
    contact_maintenance_ratio,
    force_overshoot,
    force_rmse,
    ik_failure_rate,
    max_xy_error,
    min_singular_value,
    orientation_angle_error_deg,
    xy_rmse,
    z_rmse,
)


def test_position_error_metrics() -> None:
    desired = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    actual = np.array([[0.0, 0.0, 0.5], [2.0, 0.0, 1.5]])

    assert np.isclose(xy_rmse(desired, actual), np.sqrt(0.5))
    assert np.isclose(max_xy_error(desired, actual), 1.0)
    assert np.isclose(z_rmse(desired, actual), 0.5)


def test_force_metrics() -> None:
    desired = np.array([1.0, 1.0, 1.0])
    actual = np.array([0.0, 1.0, 2.0])

    assert np.isclose(force_rmse(desired, actual), np.sqrt(2.0 / 3.0))
    assert np.isclose(contact_maintenance_ratio(actual, 0.5), 2.0 / 3.0)
    assert np.isclose(force_overshoot(1.0, actual), 1.0)


def test_orientation_and_singularity_metrics() -> None:
    errors = orientation_angle_error_deg(
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        np.array([0.0, 0.0, 1.0]),
    )

    assert np.allclose(errors, np.array([0.0, 90.0]))
    assert np.isclose(min_singular_value([np.eye(3)]), 1.0)
    assert np.isclose(ik_failure_rate([True, False, True]), 1.0 / 3.0)
