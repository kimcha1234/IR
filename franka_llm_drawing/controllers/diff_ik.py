"""Pure Python damped least-squares Differential IK fallback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from franka_llm_drawing.controllers.pose_error import pose_error_6d


@dataclass(frozen=True)
class DifferentialIKConfig:
    damping: float = 0.05
    position_gain: float = 1.0
    orientation_gain: float = 0.5
    max_delta_q: float = 0.05
    max_qd: float | None = None


@dataclass(frozen=True)
class DifferentialIKDiagnostics:
    singular_values: np.ndarray
    condition_number: float
    delta_q_norm: float


class DifferentialIKController:
    """Damped least-squares Cartesian pose correction controller."""

    def __init__(self, config: DifferentialIKConfig | None = None) -> None:
        self.config = config or DifferentialIKConfig()

    def compute_delta_q(
        self,
        q: np.ndarray,
        jacobian_6xn: np.ndarray,
        p_des: np.ndarray,
        R_des: np.ndarray,
        p_cur: np.ndarray,
        R_cur: np.ndarray,
    ) -> tuple[np.ndarray, DifferentialIKDiagnostics]:
        """Return a bounded joint increment and diagnostic singularity data."""

        q_arr = _vector(q, "q")
        J = _jacobian(jacobian_6xn, q_arr.size)
        dx = pose_error_6d(
            p_des,
            R_des,
            p_cur,
            R_cur,
            position_weight=self.config.position_gain,
            orientation_weight=self.config.orientation_gain,
        )
        damping = max(float(self.config.damping), 0.0)
        lhs = J @ J.T + (damping**2) * np.eye(6)
        try:
            delta_q = J.T @ np.linalg.solve(lhs, dx)
        except np.linalg.LinAlgError:
            delta_q = J.T @ np.linalg.pinv(lhs) @ dx

        delta_q = _clip_norm(delta_q, self.config.max_delta_q)
        singular_values = np.linalg.svd(J, compute_uv=False)
        sigma_max = float(singular_values[0]) if singular_values.size else 0.0
        sigma_min = float(singular_values[-1]) if singular_values.size else 0.0
        condition_number = sigma_max / max(sigma_min, 1e-12)
        diagnostics = DifferentialIKDiagnostics(
            singular_values=singular_values,
            condition_number=condition_number,
            delta_q_norm=float(np.linalg.norm(delta_q)),
        )
        return delta_q, diagnostics

    def compute_q_target(
        self,
        q: np.ndarray,
        jacobian_6xn: np.ndarray,
        p_des: np.ndarray,
        R_des: np.ndarray,
        p_cur: np.ndarray,
        R_cur: np.ndarray,
    ) -> tuple[np.ndarray, DifferentialIKDiagnostics]:
        """Return ``q + delta_q`` for a position-controlled backend."""

        q_arr = _vector(q, "q")
        delta_q, diagnostics = self.compute_delta_q(
            q_arr,
            jacobian_6xn,
            p_des,
            R_des,
            p_cur,
            R_cur,
        )
        return q_arr + delta_q, diagnostics


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


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a vector, got shape {arr.shape}.")
    return arr


def _jacobian(value: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (6, n):
        raise ValueError(f"jacobian_6xn must have shape (6, {n}), got {arr.shape}.")
    return arr
