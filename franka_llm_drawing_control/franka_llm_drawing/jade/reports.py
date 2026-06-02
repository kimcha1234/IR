"""Report dataclasses for JADE validation and execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Status = Literal["OK", "WARNING", "FAIL"]


@dataclass(frozen=True)
class MotionPolicy:
    speed_scale: float
    lambda_dls: float
    max_qdot: float
    max_delta_q: float
    status: Status
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FeasibilityReport:
    stroke_id: str
    status: Status
    min_manipulability: float
    max_condition_number: float
    min_sigma: float
    min_joint_limit_margin: float
    recommended_speed_scale: float
    recommended_damping: float
    warning_waypoints: list[int] = field(default_factory=list)
    fail_waypoints: list[int] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
