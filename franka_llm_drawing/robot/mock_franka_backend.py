"""Mock Franka backend for offline tests and controller plumbing."""

from __future__ import annotations

import numpy as np

from franka_llm_drawing.robot.interfaces import RobotState


class MockFrankaBackend:
    """A shape-correct backend that stores commands for inspection.

    This is not a physical or kinematic Franka model. It only provides stable
    dimensions, simple state updates, and a deterministic 6 x 7 Jacobian so the
    offline pipeline can be tested before Isaac is connected.
    """

    def __init__(
        self,
        q0: np.ndarray | None = None,
        dt: float = 0.01,
        measured_normal_force_n: float | None = None,
    ) -> None:
        self._dt = float(dt)
        self._q = np.zeros(7) if q0 is None else _vector(q0, "q0").copy()
        if self._q.shape != (7,):
            raise ValueError("MockFrankaBackend expects 7 joints.")
        self._qd = np.zeros(7)
        self._ee_position = np.zeros(3)
        self._ee_rotation = np.eye(3)
        self._ee_linear_velocity = np.zeros(3)
        self._ee_angular_velocity = np.zeros(3)
        self._last_position_target: np.ndarray | None = None
        self._last_torque_command: np.ndarray | None = None
        self._measured_normal_force_n = measured_normal_force_n

    @property
    def num_joints(self) -> int:
        return 7

    @property
    def last_position_target(self) -> np.ndarray | None:
        return None if self._last_position_target is None else self._last_position_target.copy()

    @property
    def last_torque_command(self) -> np.ndarray | None:
        return None if self._last_torque_command is None else self._last_torque_command.copy()

    def get_state(self) -> RobotState:
        return RobotState(
            q=self._q.copy(),
            qd=self._qd.copy(),
            ee_position=self._ee_position.copy(),
            ee_rotation=self._ee_rotation.copy(),
            ee_linear_velocity=self._ee_linear_velocity.copy(),
            ee_angular_velocity=self._ee_angular_velocity.copy(),
            measured_wrench=None,
        )

    def set_state(self, state: RobotState) -> None:
        """Set the mock state explicitly from a test or example."""

        self._q = _vector(state.q, "state.q").copy()
        self._qd = _vector(state.qd, "state.qd").copy()
        self._ee_position = _vector3(state.ee_position, "state.ee_position").copy()
        self._ee_rotation = np.asarray(state.ee_rotation, dtype=float).copy()
        if self._ee_rotation.shape != (3, 3):
            raise ValueError("state.ee_rotation must have shape (3, 3).")
        self._ee_linear_velocity = (
            np.zeros(3)
            if state.ee_linear_velocity is None
            else _vector3(state.ee_linear_velocity, "state.ee_linear_velocity").copy()
        )
        self._ee_angular_velocity = (
            np.zeros(3)
            if state.ee_angular_velocity is None
            else _vector3(state.ee_angular_velocity, "state.ee_angular_velocity").copy()
        )

    def get_end_effector_jacobian(self) -> np.ndarray:
        J = np.zeros((6, 7))
        J[:6, :6] = np.eye(6)
        J[0, 6] = 0.1
        J[5, 6] = 0.1
        return J

    def get_gravity_torque(self) -> np.ndarray | None:
        return np.zeros(7)

    def get_measured_normal_force(self) -> float | None:
        return self._measured_normal_force_n

    def set_measured_normal_force(self, force_n: float | None) -> None:
        self._measured_normal_force_n = None if force_n is None else float(force_n)

    def send_joint_position_target(self, q_target: np.ndarray) -> None:
        target = _vector(q_target, "q_target")
        if target.shape != (7,):
            raise ValueError("q_target must have shape (7,).")
        self._last_position_target = target.copy()

    def send_joint_torque_command(self, tau_cmd: np.ndarray) -> None:
        tau = _vector(tau_cmd, "tau_cmd")
        if tau.shape != (7,):
            raise ValueError("tau_cmd must have shape (7,).")
        self._last_torque_command = tau.copy()

    def step(self) -> None:
        """Apply the stored command with a deliberately simple mock update."""

        if self._last_position_target is not None:
            q_prev = self._q.copy()
            self._q = self._last_position_target.copy()
            self._qd = (self._q - q_prev) / max(self._dt, 1e-12)
            self._ee_linear_velocity = (self._q[:3] - q_prev[:3]) / max(self._dt, 1e-12)
            self._ee_position = self._q[:3].copy()
        elif self._last_torque_command is not None:
            self._qd = self._qd + self._last_torque_command * self._dt
            self._q = self._q + self._qd * self._dt
            self._ee_position = self._q[:3].copy()


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
