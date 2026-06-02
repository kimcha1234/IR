import numpy as np

from franka_llm_drawing.sim.isaac_backend import _jacobian_world_to_root


def test_jacobian_world_to_root_rotates_linear_and_angular_rows() -> None:
    theta = np.pi / 2.0
    root_quat_wxyz = np.array([np.cos(theta / 2.0), 0.0, 0.0, np.sin(theta / 2.0)])
    jacobian_w = np.zeros((6, 1))
    jacobian_w[:3, 0] = [1.0, 0.0, 0.0]
    jacobian_w[3:, 0] = [0.0, 1.0, 0.0]

    jacobian_root = _jacobian_world_to_root(jacobian_w, root_quat_wxyz)

    np.testing.assert_allclose(jacobian_root[:3, 0], [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(jacobian_root[3:, 0], [1.0, 0.0, 0.0], atol=1e-12)
