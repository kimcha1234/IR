"""CSV execution logger for JADE planned-vs-actual analysis."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExecutionLogRow:
    t: float
    stroke_id: str
    action_type: str
    desired_position: np.ndarray
    actual_position: np.ndarray
    q: np.ndarray
    qdot: np.ndarray
    sigma_min: float
    condition_number: float
    manipulability: float
    lambda_dls: float
    speed_scale: float
    status: str
    joint_limit_margin: float | None = None
    tip_orientation_error_deg: float | None = None
    commanded_position: np.ndarray | None = None
    desired_normal_force_n: float | None = None
    measured_normal_force_n: float | None = None
    filtered_normal_force_n: float | None = None
    normal_force_error_n: float | None = None
    normal_force_offset_m: float | None = None
    force_control_active: bool | None = None
    contact_active: bool | None = None
    policy_reason: str = ""

    @property
    def error(self) -> np.ndarray:
        return np.asarray(self.desired_position, dtype=float) - np.asarray(self.actual_position, dtype=float)

    @property
    def error_norm(self) -> float:
        return float(np.linalg.norm(self.error))


class ExecutionLogger:
    def __init__(self) -> None:
        self.rows: list[ExecutionLogRow] = []

    def append(self, row: ExecutionLogRow) -> None:
        if np.asarray(row.q).shape != (7,):
            raise ValueError("q must have shape (7,).")
        if np.asarray(row.qdot).shape != (7,):
            raise ValueError("qdot must have shape (7,).")
        self.rows.append(row)

    def desired_positions(self) -> np.ndarray:
        return np.asarray([row.desired_position for row in self.rows], dtype=float)

    def actual_positions(self) -> np.ndarray:
        return np.asarray([row.actual_position for row in self.rows], dtype=float)

    def times(self) -> np.ndarray:
        return np.asarray([row.t for row in self.rows], dtype=float)

    def error_norms(self) -> np.ndarray:
        return np.asarray([row.error_norm for row in self.rows], dtype=float)

    def write_csv(self, path: str | Path) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames())
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row_to_dict(row))


def fieldnames() -> list[str]:
    return [
        "t",
        "stroke_id",
        "action_type",
        "x_des",
        "y_des",
        "z_des",
        "x_act",
        "y_act",
        "z_act",
        "x_cmd",
        "y_cmd",
        "z_cmd",
        "error_x",
        "error_y",
        "error_z",
        "error_norm",
        *[f"q{i}" for i in range(1, 8)],
        *[f"qdot{i}" for i in range(1, 8)],
        "sigma_min",
        "condition_number",
        "manipulability",
        "lambda_dls",
        "speed_scale",
        "joint_limit_margin",
        "tip_orientation_error_deg",
        "desired_normal_force_n",
        "measured_normal_force_n",
        "filtered_normal_force_n",
        "normal_force_error_n",
        "normal_force_offset_m",
        "force_control_active",
        "contact_active",
        "status",
        "policy_reason",
    ]


def row_to_dict(row: ExecutionLogRow) -> dict[str, float | str]:
    desired = np.asarray(row.desired_position, dtype=float)
    actual = np.asarray(row.actual_position, dtype=float)
    commanded = (
        np.asarray(row.commanded_position, dtype=float)
        if row.commanded_position is not None
        else np.full(3, np.nan, dtype=float)
    )
    q = np.asarray(row.q, dtype=float)
    qdot = np.asarray(row.qdot, dtype=float)
    error = row.error
    out: dict[str, float | str] = {
        "t": row.t,
        "stroke_id": row.stroke_id,
        "action_type": row.action_type,
        "x_des": desired[0],
        "y_des": desired[1],
        "z_des": desired[2],
        "x_act": actual[0],
        "y_act": actual[1],
        "z_act": actual[2],
        "x_cmd": _optional_float(commanded[0]) if np.isfinite(commanded[0]) else "",
        "y_cmd": _optional_float(commanded[1]) if np.isfinite(commanded[1]) else "",
        "z_cmd": _optional_float(commanded[2]) if np.isfinite(commanded[2]) else "",
        "error_x": error[0],
        "error_y": error[1],
        "error_z": error[2],
        "error_norm": row.error_norm,
        "sigma_min": row.sigma_min,
        "condition_number": row.condition_number,
        "manipulability": row.manipulability,
        "lambda_dls": row.lambda_dls,
        "speed_scale": row.speed_scale,
        "joint_limit_margin": _optional_float(row.joint_limit_margin),
        "tip_orientation_error_deg": _optional_float(row.tip_orientation_error_deg),
        "desired_normal_force_n": _optional_float(row.desired_normal_force_n),
        "measured_normal_force_n": _optional_float(row.measured_normal_force_n),
        "filtered_normal_force_n": _optional_float(row.filtered_normal_force_n),
        "normal_force_error_n": _optional_float(row.normal_force_error_n),
        "normal_force_offset_m": _optional_float(row.normal_force_offset_m),
        "force_control_active": _optional_bool(row.force_control_active),
        "contact_active": _optional_bool(row.contact_active),
        "status": row.status,
        "policy_reason": row.policy_reason,
    }
    out.update({f"q{i + 1}": float(q[i]) for i in range(7)})
    out.update({f"qdot{i + 1}": float(qdot[i]) for i in range(7)})
    return out


def _optional_float(value: float | None) -> float | str:
    return "" if value is None else float(value)


def _optional_bool(value: bool | None) -> str:
    return "" if value is None else str(bool(value))
