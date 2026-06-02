"""Adaptive damped least-squares executor for JADE."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from franka_llm_drawing.config import JadeConfig
from franka_llm_drawing.controllers.pose_error import axis_alignment_error, orientation_error_axis_angle
from franka_llm_drawing.controllers.tool_jacobian import shift_spatial_jacobian_to_tool
from franka_llm_drawing.frames import ee_pose_to_tip_pose, make_transform
from franka_llm_drawing.frames.pen_tool_offset import CartesianTrajectoryPoint
from franka_llm_drawing.frames.transforms import invert_transform, transform_pose
from franka_llm_drawing.jade.adaptive_policy import recommend_motion_policy
from franka_llm_drawing.jade.jacobian_metrics import (
    FRANKA_JOINT_LOWER,
    FRANKA_JOINT_UPPER,
    JacobianMetrics,
    compute_jacobian_metrics,
)
from franka_llm_drawing.jade.reports import MotionPolicy
from franka_llm_drawing.robot.interfaces import RobotBackend, RobotState


@dataclass(frozen=True)
class DLSExecutionResult:
    q_target: np.ndarray
    delta_q: np.ndarray
    metrics: JacobianMetrics
    policy: MotionPolicy
    skipped: bool = False


class AdaptiveDLSExecutor:
    """Compute and send 7-DOF Franka arm joint targets from Cartesian targets."""

    def __init__(self, config: JadeConfig | None = None, *, fail_on_unsafe: bool = False) -> None:
        self.config = config or JadeConfig()
        self.fail_on_unsafe = bool(fail_on_unsafe)
        self._previous_target: CartesianTrajectoryPoint | None = None
        self._q_command: np.ndarray | None = None

    def reset(self) -> None:
        """Clear trajectory and integrated command history."""

        self._previous_target = None
        self._q_command = None

    def step(
        self,
        backend: RobotBackend,
        target: CartesianTrajectoryPoint,
        *,
        waypoint_index: int = 0,
        dt: float | None = None,
    ) -> DLSExecutionResult:
        state = backend.get_state()
        J_body = backend.get_end_effector_jacobian()
        T_body_tip = target_tool_transform(target)
        jacobian_tip_translation = target.jacobian_body_tip_translation
        if jacobian_tip_translation is None:
            jacobian_tip_translation = T_body_tip[:3, 3]
        J_tip = shift_spatial_jacobian_to_tool(
            J_body,
            state.ee_rotation,
            jacobian_tip_translation,
        )
        metrics = compute_jacobian_metrics(
            J_tip,
            state.q,
            stroke_id=target.stroke_id or target.source_action_name or "unknown",
            waypoint_index=waypoint_index,
            qdot_estimate=state.qd,
        )
        policy = recommend_motion_policy(
            metrics,
            self.config.thresholds,
            self.config.adaptive_policy,
            self.config.executor,
        )
        if self.fail_on_unsafe and policy.status == "FAIL":
            raise RuntimeError(f"Unsafe JADE waypoint {waypoint_index}: {policy.reason}")
        if policy.status == "FAIL":
            self._previous_target = target
            self._q_command = state.q.copy()
            return DLSExecutionResult(
                q_target=state.q.copy(),
                delta_q=np.zeros_like(state.q),
                metrics=metrics,
                policy=policy,
                skipped=True,
            )

        delta_q = compute_adaptive_dls_delta_q(
            state,
            J_tip,
            target,
            policy,
            position_gain=self.config.executor.position_gain,
            position_axis_weights=self.config.executor.position_axis_weights,
            orientation_gain=self.config.executor.orientation_gain,
            position_task_weight=self.config.executor.position_task_weight,
            orientation_task_weight=self.config.executor.orientation_task_weight,
            feedforward_gain=self.config.executor.feedforward_gain,
            orientation_mode=self.config.executor.orientation_mode,
            posture_gain=self.config.executor.posture_gain,
            posture_target_rad=self.config.executor.posture_target_rad,
            posture_weights=self.config.executor.posture_weights,
            locked_joint_indices=self.config.executor.locked_joint_indices,
            previous_target=self._previous_target,
            dt=self.config.sampling.dt if dt is None else dt,
        )
        if self.config.executor.command_integration_enabled:
            if self._q_command is None or self._q_command.shape != state.q.shape:
                self._q_command = state.q.copy()
            q_target = self._q_command + delta_q
            q_target = _limit_command_tracking_error(
                q_target,
                state.q,
                self.config.executor.max_command_tracking_error_rad,
            )
        else:
            q_target = state.q + delta_q
        q_target = _apply_locked_joint_targets(
            q_target,
            self.config.executor.locked_joint_indices,
            self.config.executor.posture_target_rad,
        )
        q_target = _clip_to_joint_limits(q_target)
        delta_q = q_target - state.q
        self._q_command = q_target.copy()
        backend.send_joint_position_target(q_target)
        self._previous_target = target
        return DLSExecutionResult(
            q_target=q_target,
            delta_q=delta_q,
            metrics=metrics,
            policy=policy,
            skipped=False,
        )


def compute_adaptive_dls_delta_q(
    state: RobotState,
    jacobian_6xn: np.ndarray,
    target: CartesianTrajectoryPoint,
    policy: MotionPolicy,
    *,
    position_gain: float,
    orientation_gain: float,
    position_axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    position_task_weight: float = 1.0,
    orientation_task_weight: float = 1.0,
    feedforward_gain: float = 0.0,
    orientation_mode: str = "axis",
    dt: float,
    posture_gain: float = 0.0,
    posture_target_rad: tuple[float, ...] | None = None,
    posture_weights: tuple[float, ...] | None = None,
    locked_joint_indices: tuple[int, ...] = (),
    previous_target: CartesianTrajectoryPoint | None = None,
) -> np.ndarray:
    """Return one bounded DLS joint increment for the pen-tip task frame."""

    q = np.asarray(state.q, dtype=float)
    J = np.asarray(jacobian_6xn, dtype=float)
    if J.shape != (6, q.size):
        raise ValueError(f"jacobian_6xn must have shape (6, {q.size}), got {J.shape}.")
    T_body_tip = target_tool_transform(target)
    T_base_tip_actual = ee_pose_to_tip_pose(
        make_transform(state.ee_rotation, state.ee_position),
        T_body_tip,
    )
    dx = _tip_pose_error_6d(
        target.T_base_tip[:3, 3],
        target.T_base_tip[:3, :3],
        T_base_tip_actual[:3, 3],
        T_base_tip_actual[:3, :3],
        position_weight=position_gain * policy.speed_scale,
        position_axis_weights=position_axis_weights,
        orientation_weight=orientation_gain * policy.speed_scale,
        orientation_mode=orientation_mode,
    )
    if previous_target is not None and feedforward_gain != 0.0:
        dx = dx + _target_feedforward_6d(
            previous_target,
            target,
            position_weight=feedforward_gain * policy.speed_scale,
            position_axis_weights=position_axis_weights,
            orientation_weight=feedforward_gain * policy.speed_scale,
            orientation_mode=orientation_mode,
        )
    locked = _locked_joint_indices(locked_joint_indices, q.size)
    active_mask = np.ones(q.size, dtype=bool)
    active_mask[list(locked)] = False
    J_active = J[:, active_mask]
    task_weights = _task_weights(position_task_weight, orientation_task_weight)
    J_weighted = task_weights[:, None] * J_active
    dx_weighted = task_weights * dx
    lhs = J_weighted @ J_weighted.T + (policy.lambda_dls**2) * np.eye(6)
    try:
        dls_inverse = J_weighted.T @ np.linalg.solve(lhs, np.eye(6))
    except np.linalg.LinAlgError:
        dls_inverse = J_weighted.T @ np.linalg.pinv(lhs)
    delta_q_active = dls_inverse @ dx_weighted
    if posture_gain > 0.0:
        nullspace = np.eye(J_active.shape[1]) - dls_inverse @ J_weighted
        posture_delta = _posture_delta(
            q,
            posture_gain,
            posture_target_rad,
            posture_weights,
        )
        delta_q_active = delta_q_active + nullspace @ posture_delta[active_mask]
    delta_q = np.zeros_like(q)
    delta_q[active_mask] = delta_q_active
    delta_q = _clip_norm(delta_q, policy.max_delta_q)
    max_step_from_velocity = max(float(policy.max_qdot) * max(float(dt), 1e-12), 0.0)
    return _clip_norm(delta_q, max_step_from_velocity)


def target_tool_transform(target: CartesianTrajectoryPoint) -> np.ndarray:
    """Return ``T_body_tip`` encoded by the target's tip and body poses."""

    return transform_pose(invert_transform(target.T_base_ee), target.T_base_tip)


def _tip_pose_error_6d(
    p_des: np.ndarray,
    R_des: np.ndarray,
    p_cur: np.ndarray,
    R_cur: np.ndarray,
    *,
    position_weight: float,
    position_axis_weights: tuple[float, float, float],
    orientation_weight: float,
    orientation_mode: str,
) -> np.ndarray:
    pos_err = (
        float(position_weight)
        * _position_axis_weights(position_axis_weights)
        * (np.asarray(p_des, dtype=float) - np.asarray(p_cur, dtype=float))
    )
    mode = str(orientation_mode).lower()
    if mode == "none":
        ori_err = np.zeros(3)
    elif mode == "axis":
        ori_err = axis_alignment_error(
            np.asarray(R_des, dtype=float)[:, 2],
            np.asarray(R_cur, dtype=float)[:, 2],
        )
    elif mode == "full":
        ori_err = orientation_error_axis_angle(R_des, R_cur)
    else:
        raise ValueError("orientation_mode must be one of: axis, full, none.")
    return np.concatenate([pos_err, float(orientation_weight) * ori_err])


def _target_feedforward_6d(
    previous_target: CartesianTrajectoryPoint,
    target: CartesianTrajectoryPoint,
    *,
    position_weight: float,
    position_axis_weights: tuple[float, float, float],
    orientation_weight: float,
    orientation_mode: str,
) -> np.ndarray:
    position_delta = _position_axis_weights(position_axis_weights) * (
        np.asarray(target.T_base_tip[:3, 3], dtype=float)
        - np.asarray(
            previous_target.T_base_tip[:3, 3],
            dtype=float,
        )
    )
    mode = str(orientation_mode).lower()
    if mode == "none":
        orientation_delta = np.zeros(3)
    elif mode == "axis":
        orientation_delta = axis_alignment_error(
            np.asarray(target.T_base_tip[:3, :3], dtype=float)[:, 2],
            np.asarray(previous_target.T_base_tip[:3, :3], dtype=float)[:, 2],
        )
    elif mode == "full":
        orientation_delta = orientation_error_axis_angle(
            np.asarray(target.T_base_tip[:3, :3], dtype=float),
            np.asarray(previous_target.T_base_tip[:3, :3], dtype=float),
        )
    else:
        raise ValueError("orientation_mode must be one of: axis, full, none.")
    return np.concatenate(
        [
            float(position_weight) * position_delta,
            float(orientation_weight) * orientation_delta,
        ]
    )


def _posture_delta(
    q: np.ndarray,
    gain: float,
    target_rad: tuple[float, ...] | None,
    weights: tuple[float, ...] | None = None,
) -> np.ndarray:
    q_arr = np.asarray(q, dtype=float)
    if target_rad is None:
        target = 0.5 * (FRANKA_JOINT_LOWER + FRANKA_JOINT_UPPER)
    else:
        target = np.asarray(target_rad, dtype=float)
        if target.shape != q_arr.shape:
            raise ValueError(f"posture_target_rad must have shape {q_arr.shape}, got {target.shape}.")
    delta = target - q_arr
    if weights is not None:
        weight_arr = np.asarray(weights, dtype=float)
        if weight_arr.shape != q_arr.shape:
            raise ValueError(f"posture_weights must have shape {q_arr.shape}, got {weight_arr.shape}.")
        delta = weight_arr * delta
    return float(gain) * delta


def _position_axis_weights(weights: tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"position_axis_weights must have shape (3,), got {arr.shape}.")
    return arr


def _task_weights(position_task_weight: float, orientation_task_weight: float) -> np.ndarray:
    pos = float(position_task_weight)
    ori = float(orientation_task_weight)
    if pos < 0.0 or ori < 0.0:
        raise ValueError("task weights must be non-negative.")
    if pos == 0.0 and ori == 0.0:
        raise ValueError("At least one task weight must be positive.")
    return np.array([pos, pos, pos, ori, ori, ori], dtype=float)


def _locked_joint_indices(indices: tuple[int, ...], num_joints: int) -> tuple[int, ...]:
    locked = tuple(sorted(set(int(index) for index in indices)))
    for index in locked:
        if index < 0 or index >= num_joints:
            raise ValueError(f"locked_joint_indices must be in [0, {num_joints - 1}], got {index}.")
    if len(locked) >= num_joints:
        raise ValueError("At least one joint must remain active.")
    return locked


def _apply_locked_joint_targets(
    q_target: np.ndarray,
    locked_joint_indices: tuple[int, ...],
    posture_target_rad: tuple[float, ...] | None,
) -> np.ndarray:
    q = np.asarray(q_target, dtype=float).copy()
    locked = _locked_joint_indices(locked_joint_indices, q.size)
    if not locked or posture_target_rad is None:
        return q
    posture_target = np.asarray(posture_target_rad, dtype=float)
    if posture_target.shape != q.shape:
        raise ValueError(f"posture_target_rad must have shape {q.shape}, got {posture_target.shape}.")
    q[list(locked)] = posture_target[list(locked)]
    return q


def _limit_command_tracking_error(
    q_target: np.ndarray,
    q_actual: np.ndarray,
    max_error_rad: float | None,
) -> np.ndarray:
    if max_error_rad is None:
        return np.asarray(q_target, dtype=float).copy()
    limit = float(max_error_rad)
    if limit <= 0.0:
        return np.asarray(q_actual, dtype=float).copy()
    target = np.asarray(q_target, dtype=float)
    actual = np.asarray(q_actual, dtype=float)
    error = target - actual
    return actual + _clip_norm(error, limit)


def _clip_to_joint_limits(q_target: np.ndarray, margin_rad: float = 1e-4) -> np.ndarray:
    q = np.asarray(q_target, dtype=float).copy()
    if q.shape != FRANKA_JOINT_LOWER.shape:
        return q
    margin = max(float(margin_rad), 0.0)
    return np.clip(q, FRANKA_JOINT_LOWER + margin, FRANKA_JOINT_UPPER - margin)


def _clip_norm(value: np.ndarray, max_norm: float | None) -> np.ndarray:
    if max_norm is None:
        return value
    limit = float(max_norm)
    if limit <= 0.0:
        return np.zeros_like(value)
    norm = float(np.linalg.norm(value))
    if norm > limit:
        return value * (limit / norm)
    return value
