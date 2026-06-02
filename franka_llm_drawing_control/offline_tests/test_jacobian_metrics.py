import numpy as np

from franka_llm_drawing.jade import compute_jacobian_metrics
from franka_llm_drawing.jade.jacobian_metrics import FRANKA_JOINT_LOWER, FRANKA_JOINT_UPPER


def test_jacobian_metrics_translation_singular_values() -> None:
    J = np.zeros((6, 7))
    J[:3, :3] = np.diag([0.4, 0.2, 0.1])
    q = 0.5 * (FRANKA_JOINT_LOWER + FRANKA_JOINT_UPPER)

    metrics = compute_jacobian_metrics(J, q, stroke_id="s0", waypoint_index=3)

    assert metrics.stroke_id == "s0"
    assert metrics.waypoint_index == 3
    assert np.isclose(metrics.sigma_min, 0.1)
    assert np.isclose(metrics.condition_number, 4.0)
    assert metrics.joint_limit_margin > 0.0


def test_joint_limit_margin_can_warn_near_limit() -> None:
    J = np.zeros((6, 7))
    J[:3, :3] = np.eye(3)
    q = 0.5 * (FRANKA_JOINT_LOWER + FRANKA_JOINT_UPPER)
    q[0] += 0.01
    q[0] = FRANKA_JOINT_LOWER[0] + 0.01

    metrics = compute_jacobian_metrics(J, q)

    assert np.isclose(metrics.joint_limit_margin, 0.01)
