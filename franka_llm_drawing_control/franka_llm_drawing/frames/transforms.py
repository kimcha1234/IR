"""Homogeneous transform helpers.

Frame names follow the convention ``T_a_b`` maps coordinates from frame ``b``
into frame ``a``.
"""

from __future__ import annotations

import numpy as np


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Create a 4x4 homogeneous transform from ``R`` and ``p``."""

    R = _rotation(rotation)
    p = _translation(translation)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Return the inverse of a rigid 4x4 transform."""

    T_arr = _transform(T)
    R = T_arr[:3, :3]
    p = T_arr[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to one 3D point."""

    T_arr = _transform(T)
    point = _translation(p)
    return T_arr[:3, :3] @ point + T_arr[:3, 3]


def transform_pose(T_a_b: np.ndarray, T_b_c: np.ndarray) -> np.ndarray:
    """Compose transforms so the result maps frame ``c`` into frame ``a``."""

    return _transform(T_a_b) @ _transform(T_b_c)


def rotation_from_board_normal(
    board_normal_base: np.ndarray,
    preferred_x_axis_base: np.ndarray | None = None,
) -> np.ndarray:
    """Return a right-handed rotation whose local +z follows the board normal."""

    z_axis = _normalized(board_normal_base, "board_normal_base")
    if preferred_x_axis_base is None:
        candidate = np.array([1.0, 0.0, 0.0])
        if abs(float(candidate @ z_axis)) > 0.95:
            candidate = np.array([0.0, 1.0, 0.0])
    else:
        candidate = _translation(preferred_x_axis_base)
    x_projected = candidate - float(candidate @ z_axis) * z_axis
    x_axis = _normalized(x_projected, "preferred_x_axis_base")
    y_axis = np.cross(z_axis, x_axis)
    y_axis = _normalized(y_axis, "computed_y_axis")
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def _transform(T: np.ndarray) -> np.ndarray:
    arr = np.asarray(T, dtype=float)
    if arr.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {arr.shape}.")
    return arr


def _rotation(R: np.ndarray) -> np.ndarray:
    arr = np.asarray(R, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {arr.shape}.")
    return arr


def _translation(p: np.ndarray) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"translation must have shape (3,), got {arr.shape}.")
    return arr


def _normalized(v: np.ndarray, name: str) -> np.ndarray:
    arr = _translation(v)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError(f"{name} must be nonzero.")
    return arr / norm
