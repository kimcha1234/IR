"""Simple trajectory retiming helpers.

The LLM planner gives geometric unit actions and nominal speeds. This module
turns those into controller-friendly segment durations by enforcing peak
velocity, acceleration, and jerk limits for the scalar path parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryLimits:
    max_linear_speed_m_s: float | None = None
    max_linear_accel_m_s2: float | None = None
    max_linear_jerk_m_s3: float | None = None


def duration_for_path_length(
    path_length_m: float,
    nominal_speed_m_s: float,
    dt_s: float,
    *,
    time_scaling: str,
    limits: TrajectoryLimits | None = None,
) -> float:
    """Return a segment duration that satisfies scalar path limits."""

    length = max(float(path_length_m), 0.0)
    speed = _positive(nominal_speed_m_s, "nominal_speed_m_s")
    dt = _positive(dt_s, "dt_s")
    if length < 1e-12:
        return dt

    duration = max(length / speed, dt)
    limits = limits or TrajectoryLimits()
    v_coeff, a_coeff, j_coeff = time_scaling_peak_coefficients(time_scaling)

    if limits.max_linear_speed_m_s is not None:
        duration = max(duration, v_coeff * length / _positive(limits.max_linear_speed_m_s, "max_linear_speed_m_s"))
    if limits.max_linear_accel_m_s2 is not None:
        duration = max(
            duration,
            math.sqrt(a_coeff * length / _positive(limits.max_linear_accel_m_s2, "max_linear_accel_m_s2")),
        )
    if limits.max_linear_jerk_m_s3 is not None:
        duration = max(
            duration,
            (j_coeff * length / _positive(limits.max_linear_jerk_m_s3, "max_linear_jerk_m_s3")) ** (1.0 / 3.0),
        )
    return duration


def time_scaling_peak_coefficients(method: str) -> tuple[float, float, float]:
    """Return peak ``s_dot*T``, ``s_ddot*T^2``, and ``s_dddot*T^3``."""

    if method == "cubic":
        return 1.5, 6.0, 12.0
    if method == "quintic":
        return 1.875, 10.0 / math.sqrt(3.0), 60.0
    raise ValueError("time_scaling must be 'cubic' or 'quintic'.")


def _positive(value: float, name: str) -> float:
    value_f = float(value)
    if value_f <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value_f
