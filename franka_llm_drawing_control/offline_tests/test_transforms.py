import numpy as np

from franka_llm_drawing.frames import (
    make_transform,
    rotation_from_board_normal,
    samples_to_cartesian_trajectory,
    tip_pose_to_ee_pose,
    transform_point,
    transform_pose,
)
from franka_llm_drawing.frames.transforms import invert_transform
from franka_llm_drawing.trajectory.path_primitives import PoseSample


def test_inverse_transform_composes_to_identity() -> None:
    R = rotation_from_board_normal(np.array([0.0, 0.0, 1.0]))
    T = make_transform(R, np.array([0.5, -0.1, 0.2]))

    assert np.allclose(invert_transform(T) @ T, np.eye(4))


def test_transform_point_identity() -> None:
    p = np.array([1.0, 2.0, 3.0])

    assert np.allclose(transform_point(np.eye(4), p), p)


def test_pose_composition_consistency() -> None:
    T_a_b = make_transform(np.eye(3), np.array([1.0, 0.0, 0.0]))
    T_b_c = make_transform(np.eye(3), np.array([0.0, 2.0, 0.0]))

    assert np.allclose(transform_pose(T_a_b, T_b_c)[:3, 3], np.array([1.0, 2.0, 0.0]))


def test_rotation_from_board_normal_is_orthonormal() -> None:
    R = rotation_from_board_normal(np.array([0.0, 0.0, 2.0]))

    assert np.allclose(R.T @ R, np.eye(3))
    assert np.allclose(R[:, 2], np.array([0.0, 0.0, 1.0]))


def test_pen_tip_offset_compensation() -> None:
    T_base_tip = make_transform(np.eye(3), np.array([0.5, 0.0, 0.3]))
    T_ee_tip = make_transform(np.eye(3), np.array([0.0, 0.0, 0.1]))

    T_base_ee = tip_pose_to_ee_pose(T_base_tip, T_ee_tip)

    assert np.allclose(T_base_ee[:3, 3], np.array([0.5, 0.0, 0.2]))


def test_samples_to_cartesian_trajectory() -> None:
    sample = PoseSample(
        t=0.0,
        position=np.array([0.1, 0.0, 0.0]),
        rotation=np.eye(3),
        pen_contact_desired=True,
        source_action="draw_line",
    )
    T_base_board = make_transform(np.eye(3), np.array([0.5, 0.0, 0.2]))
    T_ee_tip = make_transform(np.eye(3), np.array([0.0, 0.0, 0.1]))

    points = samples_to_cartesian_trajectory([sample], T_base_board, T_ee_tip)

    assert len(points) == 1
    assert np.allclose(points[0].p_base_tip, np.array([0.6, 0.0, 0.2]))
    assert np.allclose(points[0].T_base_ee[:3, 3], np.array([0.6, 0.0, 0.1]))
    assert np.allclose(points[0].jacobian_body_tip_translation, np.array([0.0, 0.0, 0.1]))


def test_samples_to_cartesian_trajectory_can_use_separate_jacobian_offset() -> None:
    sample = PoseSample(
        t=0.0,
        position=np.zeros(3),
        rotation=np.eye(3),
    )
    T_base_board = np.eye(4)
    T_ee_tip = make_transform(np.eye(3), np.array([0.0, 0.0, 0.2]))
    jacobian_offset = np.array([0.0, 0.0, 0.14])

    points = samples_to_cartesian_trajectory(
        [sample],
        T_base_board,
        T_ee_tip,
        jacobian_tip_translation_m=jacobian_offset,
    )

    assert np.allclose(points[0].T_base_ee[:3, 3], np.array([0.0, 0.0, -0.2]))
    assert np.allclose(points[0].jacobian_body_tip_translation, jacobian_offset)


def test_samples_to_cartesian_trajectory_with_downward_pen_orientation() -> None:
    sample = PoseSample(
        t=0.0,
        position=np.array([0.0, 0.0, 0.02]),
        rotation=np.eye(3),
    )
    T_base_board = make_transform(np.eye(3), np.array([0.5, 0.0, 0.202]))
    T_ee_tip = make_transform(np.eye(3), np.array([0.0, 0.0, 0.15]))
    R_base_tip = np.diag([1.0, -1.0, -1.0])

    points = samples_to_cartesian_trajectory(
        [sample],
        T_base_board,
        T_ee_tip,
        R_base_tip_override=R_base_tip,
    )

    assert np.allclose(points[0].p_base_tip, np.array([0.5, 0.0, 0.222]))
    assert np.allclose(points[0].T_base_ee[:3, 3], np.array([0.5, 0.0, 0.372]))
