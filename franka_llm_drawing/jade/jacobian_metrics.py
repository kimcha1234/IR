"""Jacobian and joint-limit metrics used by JADE."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FRANKA_JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
FRANKA_JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


@dataclass(frozen=True)
class JointLimits:
    lower: np.ndarray = field(default_factory=lambda: FRANKA_JOINT_LOWER.copy())
    upper: np.ndarray = field(default_factory=lambda: FRANKA_JOINT_UPPER.copy())

    def margin(self, q: np.ndarray) -> float:
        q_arr = np.asarray(q, dtype=float)
        if q_arr.shape != self.lower.shape:
            raise ValueError(f"q must have shape {self.lower.shape}, got {q_arr.shape}.")
        return float(np.min(np.minimum(q_arr - self.lower, self.upper - q_arr)))


@dataclass(frozen=True)
class JacobianMetrics:
    stroke_id: str
    waypoint_index: int
    sigma_min: float
    sigma_max: float
    condition_number: float
    manipulability: float
    joint_limit_margin: float
    q_norm: float
    qdot_norm_est: float | None = None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "stroke_id": self.stroke_id,
            "waypoint_index": self.waypoint_index,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
            "condition_number": self.condition_number,
            "manipulability": self.manipulability,
            "joint_limit_margin": self.joint_limit_margin,
            "q_norm": self.q_norm,
            "qdot_norm_est": self.qdot_norm_est,
        }


def compute_jacobian_metrics(
    jacobian: np.ndarray,
    q: np.ndarray | None = None,
    *,
    stroke_id: str | None = None,
    waypoint_index: int = 0,
    joint_limits: JointLimits | None = None,
    qdot_estimate: np.ndarray | None = None,
    use_translation_rows: bool = True,
) -> JacobianMetrics:
    """Compute singularity and joint-limit metrics from a 3xN or 6xN Jacobian."""

    J = np.asarray(jacobian, dtype=float)
    if J.ndim != 2 or J.shape[1] == 0 or J.shape[0] not in {3, 6}:
        raise ValueError("jacobian must have shape (3, n) or (6, n).")
    J_metric = J[:3, :] if use_translation_rows and J.shape[0] >= 3 else J
    singular_values = np.linalg.svd(J_metric, compute_uv=False)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    sigma_min = float(singular_values[-1]) if singular_values.size else 0.0
    condition_number = sigma_max / max(sigma_min, 1e-12)
    manipulability = float(np.prod(np.maximum(singular_values, 0.0)))
    if q is None:
        joint_limit_margin = float("inf")
        q_norm = 0.0
    else:
        q_arr = np.asarray(q, dtype=float)
        joint_limit_margin = (joint_limits or JointLimits()).margin(q_arr)
        q_norm = float(np.linalg.norm(q_arr))
    qdot_norm = None if qdot_estimate is None else float(np.linalg.norm(np.asarray(qdot_estimate, dtype=float)))
    return JacobianMetrics(
        stroke_id="unknown" if stroke_id is None else str(stroke_id),
        waypoint_index=int(waypoint_index),
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        condition_number=condition_number,
        manipulability=manipulability,
        joint_limit_margin=joint_limit_margin,
        q_norm=q_norm,
        qdot_norm_est=qdot_norm,
    )
