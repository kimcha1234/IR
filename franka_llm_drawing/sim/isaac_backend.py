"""Minimal Isaac Lab adapter for the JADE executor.

This module intentionally avoids importing Isaac Lab at module import time.
The runner passes Isaac objects in after ``AppLauncher`` has started the
simulator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from franka_llm_drawing.robot.interfaces import RobotState


class IsaacFrankaBackend:
    """RobotBackend adapter around an Isaac Lab Franka Articulation.

    Only ``panda_joint1`` through ``panda_joint7`` are exposed as command
    targets. Finger joints may exist in the articulation but are deliberately
    ignored by this backend.
    """

    def __init__(
        self,
        *,
        robot: Any,
        sim: Any,
        joint_ids: list[int],
        body_id: int,
        ee_jacobi_idx: int,
        torch_module: Any,
        subtract_frame_transforms: Any,
        contact_sensor: Any | None = None,
        board_normal_base: np.ndarray | None = None,
    ) -> None:
        if len(joint_ids) != 7:
            raise ValueError("IsaacFrankaBackend must receive exactly 7 arm joint ids.")
        self.robot = robot
        self.sim = sim
        self.joint_ids = list(joint_ids)
        self.body_id = int(body_id)
        self.ee_jacobi_idx = int(ee_jacobi_idx)
        self.torch = torch_module
        self.subtract_frame_transforms = subtract_frame_transforms
        self.contact_sensor = contact_sensor
        self.board_normal_base = (
            np.asarray(board_normal_base, dtype=float).copy()
            if board_normal_base is not None
            else np.array([0.0, 0.0, 1.0], dtype=float)
        )

    @property
    def num_joints(self) -> int:
        return 7

    def get_state(self) -> RobotState:
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        joint_vel = self.robot.data.joint_vel[:, self.joint_ids]
        ee_pose_w = self.robot.data.body_pose_w[:, self.body_id]
        root_pose_w = self.robot.data.root_pose_w
        ee_pos_b, ee_quat_b = self.subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )
        ee_vel_w = self.robot.data.body_vel_w[:, self.body_id]
        root_vel_w = self.robot.data.root_vel_w
        R_root_world = _quat_wxyz_to_matrix(_to_numpy(root_pose_w[0, 3:7])).T
        ee_linear_velocity = R_root_world @ _to_numpy(ee_vel_w[0, 0:3] - root_vel_w[0, 0:3])
        ee_angular_velocity = R_root_world @ _to_numpy(ee_vel_w[0, 3:6] - root_vel_w[0, 3:6])
        normal_force = self.get_measured_normal_force()
        measured_wrench = None
        if normal_force is not None:
            measured_wrench = np.concatenate([normal_force * self.board_normal_base, np.zeros(3)])
        return RobotState(
            q=_to_numpy(joint_pos[0]),
            qd=_to_numpy(joint_vel[0]),
            ee_position=_to_numpy(ee_pos_b[0]),
            ee_rotation=_quat_wxyz_to_matrix(_to_numpy(ee_quat_b[0])),
            ee_linear_velocity=ee_linear_velocity,
            ee_angular_velocity=ee_angular_velocity,
            measured_wrench=measured_wrench,
        )

    def get_end_effector_jacobian(self) -> np.ndarray:
        jacobian_w = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.joint_ids]
        return _jacobian_world_to_root(
            _to_numpy(jacobian_w[0]),
            _to_numpy(self.robot.data.root_pose_w[0, 3:7]),
        )

    def get_gravity_torque(self) -> np.ndarray | None:
        if not hasattr(self.robot.root_physx_view, "get_gravity_compensation_forces"):
            return None
        gravity = self.robot.root_physx_view.get_gravity_compensation_forces()[:, self.joint_ids]
        return _to_numpy(gravity[0])

    def get_measured_normal_force(self) -> float | None:
        force_w = self.get_contact_force_world()
        if force_w is None:
            return None
        R_world_root = _quat_wxyz_to_matrix(_to_numpy(self.robot.data.root_pose_w[0, 3:7]))
        normal_w = R_world_root @ _normalize(self.board_normal_base)
        # ContactSensor reports the force vector in world coordinates. We log
        # the paper-normal magnitude, using abs() to stay robust to the sensor
        # body's force sign convention.
        return abs(float(np.dot(force_w, normal_w)))

    def get_contact_force_world(self) -> np.ndarray | None:
        if self.contact_sensor is None:
            return None
        data = self.contact_sensor.data
        force = _force_vector_from_contact_tensor(getattr(data, "force_matrix_w_history", None), history=True)
        if force is not None:
            return force
        force = _force_vector_from_contact_tensor(getattr(data, "force_matrix_w", None), history=False)
        if force is not None:
            return force
        force = _force_vector_from_contact_tensor(getattr(data, "net_forces_w_history", None), history=True)
        if force is not None:
            return force
        return _force_vector_from_contact_tensor(getattr(data, "net_forces_w", None), history=False)

    def send_joint_position_target(self, q_target: np.ndarray) -> None:
        q = np.asarray(q_target, dtype=float)
        if q.shape != (7,):
            raise ValueError(f"q_target must have shape (7,), got {q.shape}.")
        q_tensor = self.torch.tensor(q, dtype=self.torch.float32, device=self.robot.device).unsqueeze(0)
        self.robot.set_joint_position_target(q_tensor, joint_ids=self.joint_ids)

    def send_joint_torque_command(self, tau_cmd: np.ndarray) -> None:
        tau = np.asarray(tau_cmd, dtype=float)
        if tau.shape != (7,):
            raise ValueError(f"tau_cmd must have shape (7,), got {tau.shape}.")
        tau_tensor = self.torch.tensor(tau, dtype=self.torch.float32, device=self.robot.device).unsqueeze(0)
        self.robot.set_joint_effort_target(tau_tensor, joint_ids=self.joint_ids)

    def step(self) -> None:
        self.robot.write_data_to_sim()
        self.sim.step()
        self.robot.update(self.sim.get_physics_dt())
        if self.contact_sensor is not None:
            self.contact_sensor.update(self.sim.get_physics_dt())


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value, dtype=float).copy()


def _force_vector_from_contact_tensor(value: Any, *, history: bool) -> np.ndarray | None:
    if value is None:
        return None
    arr = _to_numpy(value)
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    if arr.ndim < 3:
        return None
    # Drop batch dimension. This runner uses one scene, not vectorized envs.
    arr = arr[0]
    if history:
        arr = np.mean(arr, axis=0)
    # Remaining shapes are either (bodies, filters, 3) or (bodies, 3).
    if arr.ndim == 3:
        return np.sum(arr, axis=(0, 1))
    if arr.ndim == 2:
        return np.sum(arr, axis=0)
    return None


def _normalize(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError("normal vector must be nonzero.")
    return arr / norm


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    quat = np.asarray(q, dtype=float)
    if quat.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {quat.shape}.")
    w, x, y, z = quat / max(np.linalg.norm(quat), 1e-12)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _jacobian_world_to_root(jacobian_w: np.ndarray, root_quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert Isaac's world-frame geometric Jacobian into the robot root frame."""

    J = np.asarray(jacobian_w, dtype=float).copy()
    if J.ndim != 2 or J.shape[0] != 6:
        raise ValueError(f"jacobian_w must have shape (6, n), got {J.shape}.")
    R_world_root = _quat_wxyz_to_matrix(np.asarray(root_quat_wxyz, dtype=float))
    R_root_world = R_world_root.T
    J[:3, :] = R_root_world @ J[:3, :]
    J[3:, :] = R_root_world @ J[3:, :]
    return J
