"""Pure math hybrid position-force controller."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from franka_llm_drawing.controllers.nullspace import compute_nullspace_torque
from franka_llm_drawing.controllers.pose_error import orientation_error_axis_angle
from franka_llm_drawing.frames import make_transform, tip_pose_to_ee_pose
from franka_llm_drawing.frames.pen_tool_offset import CartesianTrajectoryPoint


@dataclass(frozen=True)
class HybridPositionForceConfig:
    kp_pos_xy: float = 150.0
    kd_pos_xy: float = 20.0
    kp_z_when_not_in_contact: float = 100.0
    kd_z_when_not_in_contact: float = 10.0
    desired_normal_force_n: float = 1.0
    kp_force: float = 5.0
    ki_force: float = 0.0
    kd_force: float = 0.0
    max_force_cmd_n: float = 20.0
    max_torque_nm: float = 30.0
    orientation_kp: float = 20.0
    orientation_kd: float = 3.0
    nullspace_kp: float = 5.0
    nullspace_kd: float = 1.0
    use_gravity_compensation: bool = True
    normal_force_positive_when_pressing: bool = True
    contact_threshold_n: float = 0.1
    nullspace_damping: float = 0.05


@dataclass(frozen=True)
class HybridControllerDiagnostics:
    task_wrench: np.ndarray
    motion_wrench: np.ndarray
    force_wrench: np.ndarray
    orientation_wrench: np.ndarray
    tau_task: np.ndarray
    tau_null: np.ndarray
    tau_cmd_raw: np.ndarray
    tau_cmd_clipped: np.ndarray
    normal_force_error: float
    contact_active: bool


@dataclass(frozen=True)
class NormalForceAdmittanceConfig:
    """Position-level normal-force loop for the Isaac joint-position backend.

    XY and pen-axis orientation remain position tasks. The paper-normal
    component is adjusted from normal-force error, which gives a stable hybrid
    position-force behavior without switching the Franka asset into raw torque
    control.
    """

    enabled: bool = True
    desired_normal_force_n: float = 1.0
    contact_threshold_n: float = 0.05
    kp_offset_m_per_n: float = 0.0015
    ki_offset_m_per_n_s: float = 0.00035
    kd_offset_m_s_per_n: float = 0.0
    max_press_offset_m: float = 0.004
    max_lift_offset_m: float = 0.025
    max_offset_step_m: float = 0.0015
    force_filter_alpha: float = 0.35


@dataclass(frozen=True)
class NormalForceAdmittanceDiagnostics:
    desired_normal_force_n: float
    measured_normal_force_n: float
    filtered_normal_force_n: float
    normal_force_error_n: float
    normal_offset_m: float
    contact_active: bool
    force_control_active: bool


class NormalForceAdmittanceController:
    """Adjust the commanded tip height to regulate paper-normal contact force."""

    def __init__(self, config: NormalForceAdmittanceConfig | None = None) -> None:
        self.config = config or NormalForceAdmittanceConfig()
        self._normal_offset_m = 0.0
        self._force_error_integral = 0.0
        self._previous_force_error: float | None = None
        self._filtered_force: float | None = None

    def reset(self) -> None:
        self._normal_offset_m = 0.0
        self._force_error_integral = 0.0
        self._previous_force_error = None
        self._filtered_force = None

    def update(
        self,
        target: CartesianTrajectoryPoint,
        *,
        T_ee_tip: np.ndarray,
        board_normal_base: np.ndarray,
        measured_normal_force_n: float | None,
        dt: float,
    ) -> tuple[CartesianTrajectoryPoint, NormalForceAdmittanceDiagnostics]:
        """Return a target whose normal coordinate is force-feedback adjusted."""

        measured = 0.0 if measured_normal_force_n is None else max(float(measured_normal_force_n), 0.0)
        alpha = float(np.clip(self.config.force_filter_alpha, 0.0, 1.0))
        if self._filtered_force is None:
            self._filtered_force = measured
        else:
            self._filtered_force = alpha * measured + (1.0 - alpha) * self._filtered_force

        nominal_desired_force = (
            float(target.desired_normal_force_n)
            if target.desired_normal_force_n is not None
            else float(self.config.desired_normal_force_n)
        )
        contact_active = self._filtered_force >= float(self.config.contact_threshold_n)
        pen_up_release = target.source_action_name == "pen_up"
        guarded_pen_down_contact = target.source_action_name == "pen_down" and contact_active
        force_target_desired = (target.pen_contact_desired or guarded_pen_down_contact) and not pen_up_release
        desired_force = 0.0 if pen_up_release else nominal_desired_force if force_target_desired else 0.0
        force_active = bool(
            self.config.enabled
            and (force_target_desired or contact_active or pen_up_release)
        )
        if not force_active:
            self._normal_offset_m = 0.0
            self._force_error_integral = 0.0
            self._previous_force_error = None
            diagnostics = NormalForceAdmittanceDiagnostics(
                desired_normal_force_n=0.0,
                measured_normal_force_n=measured,
                filtered_normal_force_n=float(self._filtered_force),
                normal_force_error_n=0.0,
                normal_offset_m=0.0,
                contact_active=contact_active,
                force_control_active=False,
            )
            return target, diagnostics

        error = desired_force - float(self._filtered_force)
        dt_s = max(float(dt), 1e-12)
        if (
            self._previous_force_error is not None
            and error != 0.0
            and np.sign(error) != np.sign(self._previous_force_error)
        ):
            self._force_error_integral = 0.0
        self._force_error_integral += error * dt_s
        self._force_error_integral = _clip_integral_for_offset(
            self._force_error_integral,
            self.config.ki_offset_m_per_n_s,
            self.config.max_press_offset_m,
            self.config.max_lift_offset_m,
        )
        derivative = 0.0
        if self._previous_force_error is not None:
            derivative = (error - self._previous_force_error) / dt_s
        self._previous_force_error = error

        raw_offset = -(
            self.config.kp_offset_m_per_n * error
            + self.config.ki_offset_m_per_n_s * self._force_error_integral
            + self.config.kd_offset_m_s_per_n * derivative
        )
        raw_offset = float(
            np.clip(raw_offset, -abs(self.config.max_press_offset_m), abs(self.config.max_lift_offset_m))
        )
        step = float(abs(self.config.max_offset_step_m))
        self._normal_offset_m += float(np.clip(raw_offset - self._normal_offset_m, -step, step))

        normal = _normalized(board_normal_base, "board_normal_base")
        adjusted_position = np.asarray(target.p_base_tip, dtype=float) + self._normal_offset_m * normal
        adjusted_T_base_tip = make_transform(target.R_base_tip, adjusted_position)
        adjusted_target = replace(
            target,
            p_base_tip=adjusted_position,
            T_base_tip=adjusted_T_base_tip,
            T_base_ee=tip_pose_to_ee_pose(adjusted_T_base_tip, T_ee_tip),
        )
        diagnostics = NormalForceAdmittanceDiagnostics(
            desired_normal_force_n=desired_force,
            measured_normal_force_n=measured,
            filtered_normal_force_n=float(self._filtered_force),
            normal_force_error_n=error,
            normal_offset_m=float(self._normal_offset_m),
            contact_active=contact_active,
            force_control_active=True,
        )
        return adjusted_target, diagnostics


class HybridPositionForceController:
    """Task-space controller that splits tangent motion and normal force."""

    def __init__(self, config: HybridPositionForceConfig | None = None) -> None:
        self.config = config or HybridPositionForceConfig()
        self._force_error_integral = 0.0
        self._previous_force_error: float | None = None

    def reset(self) -> None:
        """Clear integral and derivative history."""

        self._force_error_integral = 0.0
        self._previous_force_error = None

    def compute_torque(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        jacobian_6xn: np.ndarray,
        p_des_base: np.ndarray,
        R_des_base: np.ndarray,
        p_cur_base: np.ndarray,
        R_cur_base: np.ndarray,
        v_cur_base: np.ndarray | None,
        w_cur_base: np.ndarray | None,
        board_normal_base: np.ndarray,
        measured_normal_force_n: float | None,
        gravity_torque: np.ndarray | None = None,
        q_nominal: np.ndarray | None = None,
        dt: float = 0.01,
        contact_active: bool = False,
    ) -> tuple[np.ndarray, HybridControllerDiagnostics]:
        """Compute a joint torque command from task-space errors."""

        q_arr = _vector(q, "q")
        qd_arr = _vector(qd, "qd")
        if q_arr.shape != qd_arr.shape:
            raise ValueError("q and qd must have the same shape.")
        J = np.asarray(jacobian_6xn, dtype=float)
        if J.shape != (6, q_arr.size):
            raise ValueError(f"jacobian_6xn must have shape (6, {q_arr.size}).")

        p_des = _vector3(p_des_base, "p_des_base")
        p_cur = _vector3(p_cur_base, "p_cur_base")
        R_des = _rotation(R_des_base, "R_des_base")
        R_cur = _rotation(R_cur_base, "R_cur_base")
        normal = _normalized(board_normal_base, "board_normal_base")
        tangent_projector = np.eye(3) - np.outer(normal, normal)
        v_cur = np.zeros(3) if v_cur_base is None else _vector3(v_cur_base, "v_cur_base")
        w_cur = np.zeros(3) if w_cur_base is None else _vector3(w_cur_base, "w_cur_base")

        measured_force = 0.0 if measured_normal_force_n is None else float(measured_normal_force_n)
        effective_contact = bool(
            contact_active or measured_force >= self.config.contact_threshold_n
        )

        position_error = p_des - p_cur
        tangent_error = tangent_projector @ position_error
        tangent_velocity = tangent_projector @ v_cur
        motion_linear = (
            self.config.kp_pos_xy * tangent_error - self.config.kd_pos_xy * tangent_velocity
        )

        normal_motion = np.zeros(3)
        force_linear = np.zeros(3)
        force_error = self.config.desired_normal_force_n - measured_force

        if effective_contact:
            self._force_error_integral += force_error * max(float(dt), 0.0)
            force_derivative = 0.0
            if self._previous_force_error is not None and dt > 0.0:
                force_derivative = (force_error - self._previous_force_error) / float(dt)
            self._previous_force_error = force_error
            force_mag = (
                self.config.desired_normal_force_n
                + self.config.kp_force * force_error
                + self.config.ki_force * self._force_error_integral
                + self.config.kd_force * force_derivative
            )
            force_mag = float(
                np.clip(force_mag, -self.config.max_force_cmd_n, self.config.max_force_cmd_n)
            )
            press_direction = -normal if self.config.normal_force_positive_when_pressing else normal
            force_linear = force_mag * press_direction
        else:
            self._previous_force_error = None
            normal_error = float(position_error @ normal)
            normal_velocity = float(v_cur @ normal)
            normal_motion = (
                self.config.kp_z_when_not_in_contact * normal_error
                - self.config.kd_z_when_not_in_contact * normal_velocity
            ) * normal

        orientation_error = orientation_error_axis_angle(R_des, R_cur)
        orientation_linear = (
            self.config.orientation_kp * orientation_error - self.config.orientation_kd * w_cur
        )

        motion_wrench = np.concatenate([motion_linear + normal_motion, np.zeros(3)])
        force_wrench = np.concatenate([force_linear, np.zeros(3)])
        orientation_wrench = np.concatenate([np.zeros(3), orientation_linear])
        task_wrench = motion_wrench + force_wrench + orientation_wrench
        tau_task = J.T @ task_wrench

        if q_nominal is None:
            tau_null = np.zeros_like(q_arr)
        else:
            tau_null, _ = compute_nullspace_torque(
                q_arr,
                qd_arr,
                J,
                _vector(q_nominal, "q_nominal"),
                kp=self.config.nullspace_kp,
                kd=self.config.nullspace_kd,
                damping=self.config.nullspace_damping,
                max_torque_nm=self.config.max_torque_nm,
            )

        tau_cmd_raw = tau_task + tau_null
        if self.config.use_gravity_compensation and gravity_torque is not None:
            gravity = _vector(gravity_torque, "gravity_torque")
            if gravity.shape != q_arr.shape:
                raise ValueError("gravity_torque must have the same shape as q.")
            tau_cmd_raw = tau_cmd_raw + gravity

        limit = abs(float(self.config.max_torque_nm))
        tau_cmd_clipped = np.clip(tau_cmd_raw, -limit, limit)
        diagnostics = HybridControllerDiagnostics(
            task_wrench=task_wrench,
            motion_wrench=motion_wrench,
            force_wrench=force_wrench,
            orientation_wrench=orientation_wrench,
            tau_task=tau_task,
            tau_null=tau_null,
            tau_cmd_raw=tau_cmd_raw,
            tau_cmd_clipped=tau_cmd_clipped,
            normal_force_error=float(force_error),
            contact_active=effective_contact,
        )
        return tau_cmd_clipped, diagnostics


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a vector, got shape {arr.shape}.")
    return arr


def _vector3(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


def _rotation(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {arr.shape}.")
    return arr


def _normalized(value: np.ndarray, name: str) -> np.ndarray:
    arr = _vector3(value, name)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError(f"{name} must be nonzero.")
    return arr / norm


def _clip_integral_for_offset(
    integral: float,
    ki_offset_m_per_n_s: float,
    max_press_offset_m: float,
    max_lift_offset_m: float,
) -> float:
    ki = abs(float(ki_offset_m_per_n_s))
    if ki <= 0.0:
        return 0.0
    lower = -abs(float(max_lift_offset_m)) / ki
    upper = abs(float(max_press_offset_m)) / ki
    return float(np.clip(integral, lower, upper))
