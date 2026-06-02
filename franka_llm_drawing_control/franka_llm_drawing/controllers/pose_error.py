"""Pose error utilities for Cartesian control."""

from __future__ import annotations

import numpy as np


def orientation_error_axis_angle(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """Return a 3D log-map orientation error vector.

    The error is computed from ``R_err = R_des @ R_cur.T`` and is suitable as a
    small-angle angular correction term in the parent frame.
    """

    R_d = _rotation(R_des, "R_des")
    R_c = _rotation(R_cur, "R_cur")
    R_err = R_d @ R_c.T
    skew_vec = np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    )
    cos_theta = float(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        return 0.5 * skew_vec
    sin_theta = float(np.sin(theta))
    if abs(sin_theta) < 1e-9:
        axis = _axis_for_pi_rotation(R_err)
        return theta * axis
    return theta / (2.0 * sin_theta) * skew_vec


def axis_alignment_error(axis_des: np.ndarray, axis_cur: np.ndarray) -> np.ndarray:
    """Return a rotation vector that aligns ``axis_cur`` to ``axis_des``.

    This leaves rotation about the axis unconstrained, which is useful for a
    pen: the drawing task needs the pen axis normal to the paper, not a fixed
    roll angle around the pen.
    """

    desired = _normalized_vector(axis_des, "axis_des")
    current = _normalized_vector(axis_cur, "axis_cur")
    cross = np.cross(current, desired)
    sin_theta = float(np.linalg.norm(cross))
    cos_theta = float(np.clip(np.dot(current, desired), -1.0, 1.0))
    if sin_theta < 1e-9:
        if cos_theta > 0.0:
            return np.zeros(3)
        axis = _orthogonal_axis(current)
        return np.pi * axis
    theta = float(np.arctan2(sin_theta, cos_theta))
    return theta * cross / sin_theta


def pose_error_6d(
    p_des: np.ndarray,
    R_des: np.ndarray,
    p_cur: np.ndarray,
    R_cur: np.ndarray,
    position_weight: float = 1.0,
    orientation_weight: float = 1.0,
) -> np.ndarray:
    """Return weighted ``[position_error, orientation_error]`` with shape ``(6,)``."""

    p_d = _vector3(p_des, "p_des")
    p_c = _vector3(p_cur, "p_cur")
    pos_err = float(position_weight) * (p_d - p_c)
    ori_err = float(orientation_weight) * orientation_error_axis_angle(R_des, R_cur)
    return np.concatenate([pos_err, ori_err])


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


def _normalized_vector(value: np.ndarray, name: str) -> np.ndarray:
    arr = _vector3(value, name)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError(f"{name} must be nonzero.")
    return arr / norm


def _orthogonal_axis(value: np.ndarray) -> np.ndarray:
    v = _normalized_vector(value, "value")
    candidate = np.array([1.0, 0.0, 0.0])
    if abs(float(candidate @ v)) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    axis = candidate - float(candidate @ v) * v
    return axis / max(float(np.linalg.norm(axis)), 1e-12)


def _axis_for_pi_rotation(R_err: np.ndarray) -> np.ndarray:
    diagonal = np.diag(R_err)
    axis = np.zeros(3)
    index = int(np.argmax(diagonal))
    axis[index] = np.sqrt(max(diagonal[index] + 1.0, 0.0) / 2.0)
    if axis[index] < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    for j in range(3):
        if j != index:
            axis[j] = R_err[j, index] / (2.0 * axis[index])
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return axis / norm
