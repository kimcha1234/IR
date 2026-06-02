import numpy as np

from franka_llm_drawing.config import ThresholdConfig
from franka_llm_drawing.jade import compute_jacobian_metrics, recommend_motion_policy, validate_stroke
from franka_llm_drawing.jade.jacobian_metrics import FRANKA_JOINT_LOWER, FRANKA_JOINT_UPPER

Q_MID = 0.5 * (FRANKA_JOINT_LOWER + FRANKA_JOINT_UPPER)


def test_validate_stroke_ok() -> None:
    J = np.zeros((6, 7))
    J[:3, :3] = np.diag([0.4, 0.3, 0.2])
    metrics = [compute_jacobian_metrics(J, Q_MID, stroke_id="s0", waypoint_index=i) for i in range(3)]

    report = validate_stroke(metrics)

    assert report.status == "OK"
    assert not report.fail_waypoints


def test_validate_stroke_fail_on_singularity() -> None:
    J = np.zeros((6, 7))
    J[:3, :3] = np.diag([0.4, 0.3, 0.001])
    metrics = [compute_jacobian_metrics(J, Q_MID, stroke_id="s0", waypoint_index=0)]

    report = validate_stroke(metrics)

    assert report.status == "FAIL"
    assert report.fail_waypoints == [0]


def test_adaptive_policy_increases_damping_on_warning() -> None:
    J = np.zeros((6, 7))
    J[:3, :3] = np.diag([0.4, 0.3, 0.02])
    metrics = compute_jacobian_metrics(J, Q_MID)

    policy = recommend_motion_policy(metrics, ThresholdConfig())

    assert policy.status == "WARNING"
    assert policy.lambda_dls > 0.02
    assert policy.speed_scale < 1.0
