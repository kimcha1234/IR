import numpy as np

from franka_llm_drawing.controllers import DifferentialIKConfig, DifferentialIKController
from franka_llm_drawing.controllers.pose_error import axis_alignment_error, orientation_error_axis_angle


def test_diff_ik_output_shape_and_diagnostics() -> None:
    controller = DifferentialIKController(DifferentialIKConfig(max_delta_q=0.1))
    q = np.zeros(7)
    J = np.zeros((6, 7))
    J[:6, :6] = np.eye(6)

    delta_q, diagnostics = controller.compute_delta_q(
        q,
        J,
        np.array([0.01, 0.0, 0.0]),
        np.eye(3),
        np.zeros(3),
        np.eye(3),
    )

    assert delta_q.shape == (7,)
    assert diagnostics.singular_values.shape == (6,)
    assert diagnostics.delta_q_norm <= 0.1


def test_diff_ik_handles_near_singular_matrix_without_nan() -> None:
    controller = DifferentialIKController(DifferentialIKConfig(damping=0.1))
    q = np.zeros(7)
    J = np.zeros((6, 7))
    J[0, 0] = 1e-9

    delta_q, diagnostics = controller.compute_delta_q(
        q,
        J,
        np.array([1.0, 0.0, 0.0]),
        np.eye(3),
        np.zeros(3),
        np.eye(3),
    )

    assert np.isfinite(delta_q).all()
    assert np.isfinite(diagnostics.condition_number)


def test_orientation_error_handles_pi_rotation() -> None:
    R_des = np.diag([1.0, -1.0, -1.0])

    error = orientation_error_axis_angle(R_des, np.eye(3))

    assert np.isclose(np.linalg.norm(error), np.pi)


def test_axis_alignment_error_ignores_roll_about_axis() -> None:
    R_roll = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    error = axis_alignment_error(np.eye(3)[:, 2], R_roll[:, 2])

    assert np.allclose(error, np.zeros(3))
