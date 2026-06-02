"""Pure evaluation metrics for drawing and controller logs."""

from __future__ import annotations

import numpy as np


def xy_rmse(p_des: np.ndarray, p_act: np.ndarray) -> float:
    des, act = _paired_points(p_des, p_act)
    err = des[:, :2] - act[:, :2]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def max_xy_error(p_des: np.ndarray, p_act: np.ndarray) -> float:
    des, act = _paired_points(p_des, p_act)
    err = np.linalg.norm(des[:, :2] - act[:, :2], axis=1)
    return float(np.max(err))


def z_rmse(p_des: np.ndarray, p_act: np.ndarray) -> float:
    des, act = _paired_points(p_des, p_act)
    err = des[:, 2] - act[:, 2]
    return float(np.sqrt(np.mean(err**2)))


def force_rmse(f_des: np.ndarray, f_act: np.ndarray) -> float:
    des = np.asarray(f_des, dtype=float)
    act = np.asarray(f_act, dtype=float)
    if des.shape != act.shape:
        raise ValueError("f_des and f_act must have the same shape.")
    return float(np.sqrt(np.mean((des - act) ** 2)))


def contact_maintenance_ratio(f_normal: np.ndarray, threshold_n: float) -> float:
    forces = np.asarray(f_normal, dtype=float)
    if forces.size == 0:
        raise ValueError("f_normal must not be empty.")
    return float(np.mean(forces >= float(threshold_n)))


def force_overshoot(f_des: float, f_act: np.ndarray) -> float:
    forces = np.asarray(f_act, dtype=float)
    if forces.size == 0:
        raise ValueError("f_act must not be empty.")
    return float(max(0.0, np.max(forces) - float(f_des)))


def orientation_angle_error_deg(z_pen: np.ndarray, n_board: np.ndarray) -> np.ndarray:
    """Return angle between pen z-axis vectors and board normal in degrees."""

    z = _as_vector_batch(z_pen, "z_pen")
    n = np.asarray(n_board, dtype=float)
    if n.shape == (3,):
        n = np.repeat(n[None, :], z.shape[0], axis=0)
    elif n.shape != z.shape:
        raise ValueError("n_board must have shape (3,) or match z_pen.")
    z_unit = z / np.linalg.norm(z, axis=1, keepdims=True)
    n_unit = n / np.linalg.norm(n, axis=1, keepdims=True)
    dots = np.sum(z_unit * n_unit, axis=1)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def min_singular_value(jacobians: list[np.ndarray]) -> float:
    if not jacobians:
        raise ValueError("jacobians must not be empty.")
    values = [np.linalg.svd(np.asarray(J, dtype=float), compute_uv=False)[-1] for J in jacobians]
    return float(np.min(values))


def ik_failure_rate(success_flags: list[bool]) -> float:
    if not success_flags:
        raise ValueError("success_flags must not be empty.")
    return float(1.0 - np.mean(np.asarray(success_flags, dtype=bool)))


def _paired_points(p_des: np.ndarray, p_act: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    des = np.asarray(p_des, dtype=float)
    act = np.asarray(p_act, dtype=float)
    if des.shape != act.shape:
        raise ValueError("p_des and p_act must have the same shape.")
    if des.ndim != 2 or des.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3).")
    if des.shape[0] == 0:
        raise ValueError("positions must not be empty.")
    return des, act


def _as_vector_batch(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == (3,):
        return arr[None, :]
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    raise ValueError(f"{name} must have shape (3,) or (N, 3).")
