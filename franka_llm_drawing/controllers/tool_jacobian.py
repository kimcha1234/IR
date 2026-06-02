"""Spatial Jacobian helpers for tools offset from the controlled body."""

from __future__ import annotations

import numpy as np


def shift_spatial_jacobian_to_tool(
    jacobian_body_6xn: np.ndarray,
    R_base_body: np.ndarray,
    p_body_tool: np.ndarray,
) -> np.ndarray:
    """Return the spatial Jacobian at a rigidly attached tool point.

    The input Jacobian is ordered as ``[linear_velocity; angular_velocity]`` in
    the base frame. ``p_body_tool`` is the tool point translation expressed in
    the body frame. The angular rows are unchanged, while the linear rows are
    shifted by ``omega x r``.
    """

    J = np.asarray(jacobian_body_6xn, dtype=float)
    if J.ndim != 2 or J.shape[0] != 6:
        raise ValueError(f"jacobian_body_6xn must have shape (6, n), got {J.shape}.")
    R = np.asarray(R_base_body, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"R_base_body must have shape (3, 3), got {R.shape}.")
    p = np.asarray(p_body_tool, dtype=float)
    if p.shape != (3,):
        raise ValueError(f"p_body_tool must have shape (3,), got {p.shape}.")

    r_base = R @ p
    shifted = J.copy()
    shifted[:3, :] = J[:3, :] - _skew(r_base) @ J[3:, :]
    return shifted


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=float,
    )
