"""Optional nullspace posture control for redundant manipulators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NullspaceDiagnostics:
    torque_norm: float
    projector_rank: int


def damped_pseudoinverse(J: np.ndarray, damping: float = 0.05) -> np.ndarray:
    """Return a damped least-squares pseudo-inverse for ``J``."""

    J_arr = np.asarray(J, dtype=float)
    if J_arr.ndim != 2:
        raise ValueError("J must be a matrix.")
    lam = max(float(damping), 0.0)
    rows = J_arr.shape[0]
    return J_arr.T @ np.linalg.inv(J_arr @ J_arr.T + (lam**2) * np.eye(rows))


def compute_nullspace_torque(
    q: np.ndarray,
    qd: np.ndarray,
    jacobian_6xn: np.ndarray,
    q_nominal: np.ndarray,
    kp: float,
    kd: float,
    damping: float = 0.05,
    max_torque_nm: float | None = None,
) -> tuple[np.ndarray, NullspaceDiagnostics]:
    """Compute ``N.T @ (kp * (q_nominal - q) - kd * qd)``."""

    q_arr = _vector(q, "q")
    qd_arr = _vector(qd, "qd")
    q_nom = _vector(q_nominal, "q_nominal")
    if q_arr.shape != qd_arr.shape or q_arr.shape != q_nom.shape:
        raise ValueError("q, qd, and q_nominal must have the same shape.")
    J = np.asarray(jacobian_6xn, dtype=float)
    if J.shape != (6, q_arr.size):
        raise ValueError(f"jacobian_6xn must have shape (6, {q_arr.size}).")
    J_pinv = damped_pseudoinverse(J, damping=damping)
    N = np.eye(q_arr.size) - J_pinv @ J
    tau = N.T @ (float(kp) * (q_nom - q_arr) - float(kd) * qd_arr)
    if max_torque_nm is not None:
        limit = abs(float(max_torque_nm))
        tau = np.clip(tau, -limit, limit)
    return tau, NullspaceDiagnostics(
        torque_norm=float(np.linalg.norm(tau)),
        projector_rank=int(np.linalg.matrix_rank(N)),
    )


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a vector, got shape {arr.shape}.")
    return arr
