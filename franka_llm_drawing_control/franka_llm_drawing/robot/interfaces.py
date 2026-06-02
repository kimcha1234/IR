"""Backend protocol shared by mock and future Isaac implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RobotState:
    """Robot state snapshot.

    ``ee_position`` and ``ee_rotation`` are expressed in the backend's robot
    base frame. ``measured_wrench`` is a six-vector whose frame convention must
    be documented by the backend.
    """

    q: np.ndarray
    qd: np.ndarray
    ee_position: np.ndarray
    ee_rotation: np.ndarray
    ee_linear_velocity: np.ndarray | None = None
    ee_angular_velocity: np.ndarray | None = None
    measured_wrench: np.ndarray | None = None


class RobotBackend(Protocol):
    """Minimal robot API required by the offline controllers."""

    @property
    def num_joints(self) -> int:
        ...

    def get_state(self) -> RobotState:
        ...

    def get_end_effector_jacobian(self) -> np.ndarray:
        """Return a 6 x n geometric Jacobian."""

    def get_gravity_torque(self) -> np.ndarray | None:
        ...

    def get_measured_normal_force(self) -> float | None:
        ...

    def send_joint_position_target(self, q_target: np.ndarray) -> None:
        ...

    def send_joint_torque_command(self, tau_cmd: np.ndarray) -> None:
        ...

    def step(self) -> None:
        ...
