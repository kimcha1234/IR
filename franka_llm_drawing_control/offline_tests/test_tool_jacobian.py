import numpy as np

from franka_llm_drawing.config import ExecutorConfig, JadeConfig, ThresholdConfig
from franka_llm_drawing.controllers.tool_jacobian import shift_spatial_jacobian_to_tool
from franka_llm_drawing.frames import make_transform, tip_pose_to_ee_pose
from franka_llm_drawing.frames.pen_tool_offset import CartesianTrajectoryPoint
from franka_llm_drawing.jade.adaptive_dls_executor import (
    AdaptiveDLSExecutor,
    compute_adaptive_dls_delta_q,
    target_tool_transform,
)
from franka_llm_drawing.jade.reports import MotionPolicy
from franka_llm_drawing.robot.interfaces import RobotState


def test_shift_spatial_jacobian_accounts_for_tool_offset() -> None:
    J_body = np.zeros((6, 1))
    J_body[5, 0] = 1.0

    J_tool = shift_spatial_jacobian_to_tool(
        J_body,
        np.eye(3),
        np.array([1.0, 0.0, 0.0]),
    )

    assert np.allclose(J_tool[:3, 0], np.array([0.0, 1.0, 0.0]))
    assert np.allclose(J_tool[3:, 0], np.array([0.0, 0.0, 1.0]))


def test_adaptive_dls_uses_pen_tip_pose_error() -> None:
    T_body_tip = make_transform(np.eye(3), np.array([1.0, 0.0, 0.0]))
    T_base_tip = make_transform(np.eye(3), np.array([1.0, 0.1, 0.0]))
    T_base_body = tip_pose_to_ee_pose(T_base_tip, T_body_tip)
    target = CartesianTrajectoryPoint(
        t=0.0,
        p_base_tip=T_base_tip[:3, 3],
        R_base_tip=T_base_tip[:3, :3],
        T_base_tip=T_base_tip,
        T_base_ee=T_base_body,
        pen_state="up",
        source_action_name="test",
    )
    state = RobotState(
        q=np.zeros(1),
        qd=np.zeros(1),
        ee_position=np.zeros(3),
        ee_rotation=np.eye(3),
    )
    J_tip = np.zeros((6, 1))
    J_tip[1, 0] = 1.0
    policy = MotionPolicy(
        speed_scale=1.0,
        lambda_dls=0.01,
        max_qdot=100.0,
        max_delta_q=1.0,
        status="OK",
        reason="test",
    )

    delta_q = compute_adaptive_dls_delta_q(
        state,
        J_tip,
        target,
        policy,
        position_gain=1.0,
        orientation_gain=0.0,
        dt=0.02,
    )

    assert delta_q[0] > 0.0
    assert np.allclose(target_tool_transform(target), T_body_tip)


def test_executor_uses_separate_jacobian_tip_translation() -> None:
    T_body_tip = make_transform(np.eye(3), np.array([1.0, 0.0, 0.0]))
    T_base_tip = make_transform(np.eye(3), np.array([1.0, 0.1, 0.0]))
    T_base_body = tip_pose_to_ee_pose(T_base_tip, T_body_tip)
    target = CartesianTrajectoryPoint(
        t=0.0,
        p_base_tip=T_base_tip[:3, 3],
        R_base_tip=T_base_tip[:3, :3],
        T_base_tip=T_base_tip,
        T_base_ee=T_base_body,
        pen_state="up",
        source_action_name="test",
        jacobian_body_tip_translation=np.array([1.0, 0.0, 0.0]),
    )
    backend = _RecordingBackend()
    config = JadeConfig(
        thresholds=ThresholdConfig(
            sigma_min_fail=-1.0,
            condition_fail=float("inf"),
            joint_margin_fail_rad=-1.0,
        ),
        executor=ExecutorConfig(command_integration_enabled=True),
    )

    AdaptiveDLSExecutor(config).step(backend, target, dt=0.02)

    assert backend.last_q_target is not None
    assert backend.last_q_target[0] > 0.0


def test_executor_integrates_command_when_backend_lags() -> None:
    target = _identity_tool_target(np.array([0.01, 0.0, 0.0]))
    backend = _LaggingBackend()
    config = JadeConfig(
        thresholds=ThresholdConfig(
            sigma_min_fail=-1.0,
            condition_fail=float("inf"),
            joint_margin_fail_rad=-1.0,
        ),
        executor=ExecutorConfig(command_integration_enabled=True),
    )
    executor = AdaptiveDLSExecutor(config)

    first = executor.step(backend, target, dt=0.02).q_target.copy()
    second = executor.step(backend, target, dt=0.02).q_target.copy()

    assert second[0] > first[0]
    assert np.linalg.norm(second - backend.get_state().q) <= config.executor.max_command_tracking_error_rad


def test_adaptive_dls_feedforward_tracks_target_motion() -> None:
    previous_target = _identity_tool_target(np.array([0.0, 0.0, 0.0]))
    target = _identity_tool_target(np.array([0.01, 0.0, 0.0]))
    state = RobotState(
        q=np.zeros(7),
        qd=np.zeros(7),
        ee_position=np.zeros(3),
        ee_rotation=np.eye(3),
    )
    J_tip = np.zeros((6, 7))
    J_tip[0, 0] = 1.0
    policy = MotionPolicy(
        speed_scale=1.0,
        lambda_dls=0.01,
        max_qdot=100.0,
        max_delta_q=1.0,
        status="OK",
        reason="test",
    )

    delta_q = compute_adaptive_dls_delta_q(
        state,
        J_tip,
        target,
        policy,
        position_gain=0.0,
        orientation_gain=0.0,
        feedforward_gain=1.0,
        previous_target=previous_target,
        dt=0.02,
    )

    assert delta_q[0] > 0.005


def test_posture_weights_can_bias_pen_roll_joint() -> None:
    target = _identity_tool_target(np.zeros(3))
    state = RobotState(
        q=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        qd=np.zeros(7),
        ee_position=np.zeros(3),
        ee_rotation=np.eye(3),
    )
    policy = MotionPolicy(
        speed_scale=1.0,
        lambda_dls=0.01,
        max_qdot=1000.0,
        max_delta_q=10.0,
        status="OK",
        reason="test",
    )

    delta_q = compute_adaptive_dls_delta_q(
        state,
        np.zeros((6, 7)),
        target,
        policy,
        position_gain=0.0,
        orientation_gain=0.0,
        posture_gain=1.0,
        posture_target_rad=tuple(np.zeros(7)),
        posture_weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0),
        dt=0.02,
    )

    assert np.isclose(delta_q[6], -4.0)


def test_locked_joint_is_removed_from_dls_solution() -> None:
    target = _identity_tool_target(np.array([0.01, 0.0, 0.0]))
    state = RobotState(
        q=np.zeros(7),
        qd=np.zeros(7),
        ee_position=np.zeros(3),
        ee_rotation=np.eye(3),
    )
    J_tip = np.zeros((6, 7))
    J_tip[0, 0] = 1.0
    J_tip[0, 6] = 10.0
    policy = MotionPolicy(
        speed_scale=1.0,
        lambda_dls=0.01,
        max_qdot=1000.0,
        max_delta_q=10.0,
        status="OK",
        reason="test",
    )

    delta_q = compute_adaptive_dls_delta_q(
        state,
        J_tip,
        target,
        policy,
        position_gain=1.0,
        orientation_gain=0.0,
        locked_joint_indices=(6,),
        dt=0.02,
    )

    assert delta_q[0] > 0.0
    assert delta_q[6] == 0.0


def _identity_tool_target(position: np.ndarray) -> CartesianTrajectoryPoint:
    T_base_tip = make_transform(np.eye(3), np.asarray(position, dtype=float))
    return CartesianTrajectoryPoint(
        t=0.0,
        p_base_tip=T_base_tip[:3, 3],
        R_base_tip=T_base_tip[:3, :3],
        T_base_tip=T_base_tip,
        T_base_ee=T_base_tip,
        pen_state="up",
        source_action_name="test",
    )


class _RecordingBackend:
    def __init__(self) -> None:
        self.last_q_target: np.ndarray | None = None

    @property
    def num_joints(self) -> int:
        return 7

    def get_state(self) -> RobotState:
        return RobotState(
            q=np.zeros(7),
            qd=np.zeros(7),
            ee_position=np.zeros(3),
            ee_rotation=np.eye(3),
        )

    def get_end_effector_jacobian(self) -> np.ndarray:
        J = np.zeros((6, 7))
        J[5, 0] = 1.0
        return J

    def get_gravity_torque(self):
        return None

    def get_measured_normal_force(self):
        return None

    def send_joint_position_target(self, q_target: np.ndarray) -> None:
        self.last_q_target = np.asarray(q_target, dtype=float)

    def send_joint_torque_command(self, tau_cmd: np.ndarray) -> None:
        raise NotImplementedError

    def step(self) -> None:
        pass


class _LaggingBackend(_RecordingBackend):
    def get_state(self) -> RobotState:
        return RobotState(
            q=np.zeros(7),
            qd=np.zeros(7),
            ee_position=np.zeros(3),
            ee_rotation=np.eye(3),
        )

    def get_end_effector_jacobian(self) -> np.ndarray:
        J = np.zeros((6, 7))
        J[:6, :6] = np.eye(6)
        return J
