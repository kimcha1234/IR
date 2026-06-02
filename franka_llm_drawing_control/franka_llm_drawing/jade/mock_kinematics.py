"""Deterministic mock Jacobian/state helpers for offline JADE validation."""

from __future__ import annotations

import numpy as np

FRANKA_NOMINAL_Q = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741])


def estimate_mock_q_from_tip_position(p_base_tip: np.ndarray) -> np.ndarray:
    """Return a smooth pseudo joint vector for offline feasibility reports.

    This is not Franka forward kinematics. It only gives the offline validator
    a deterministic, bounded q-vector until Isaac PhysX provides the real state.
    """

    p = np.asarray(p_base_tip, dtype=float)
    if p.shape != (3,):
        raise ValueError(f"p_base_tip must have shape (3,), got {p.shape}.")
    q = FRANKA_NOMINAL_Q.copy()
    q += np.array(
        [
            0.40 * (p[1]),
            0.15 * (p[0] - 0.5),
            -0.25 * (p[1]),
            0.10 * (p[2] - 0.35),
            0.20 * (p[0] - 0.5),
            -0.12 * (p[1]),
            0.10 * np.sin(8.0 * p[0]),
        ]
    )
    return q


def mock_franka_jacobian(q: np.ndarray) -> np.ndarray:
    """Return a full-rank, shape-correct 6x7 Jacobian for offline checks."""

    q_arr = np.asarray(q, dtype=float)
    if q_arr.shape != (7,):
        raise ValueError(f"q must have shape (7,), got {q_arr.shape}.")
    J = np.array(
        [
            [0.27, -0.06, 0.21, -0.03, 0.12, 0.02, 0.04],
            [0.03, 0.30, 0.05, -0.18, 0.02, 0.09, -0.03],
            [0.08, 0.03, 0.24, 0.06, -0.15, 0.05, 0.06],
            [0.00, 0.12, 0.02, 0.20, 0.01, -0.04, 0.10],
            [0.15, 0.00, -0.05, 0.02, 0.18, 0.03, -0.02],
            [0.01, -0.10, 0.02, 0.06, -0.01, 0.17, 0.08],
        ],
        dtype=float,
    )
    modulation = 1.0 + 0.04 * np.sin(q_arr)
    return J * modulation[None, :]
