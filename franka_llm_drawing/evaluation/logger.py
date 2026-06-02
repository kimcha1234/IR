"""Small in-memory logger for offline control runs."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class LogRecord:
    t: float
    desired_position: np.ndarray
    actual_position: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    torque_command: np.ndarray | None = None
    normal_force_n: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


class TrajectoryLogger:
    """Collect records and optionally write a compact CSV file."""

    def __init__(self) -> None:
        self.records: list[LogRecord] = []

    def append(self, record: LogRecord) -> None:
        self.records.append(record)

    def desired_positions(self) -> np.ndarray:
        return np.array([record.desired_position for record in self.records], dtype=float)

    def actual_positions(self) -> np.ndarray:
        return np.array([record.actual_position for record in self.records], dtype=float)

    def write_csv(self, path: str | Path) -> None:
        with Path(path).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "x_des", "y_des", "z_des", "x_act", "y_act", "z_act"])
            for record in self.records:
                writer.writerow(
                    [
                        record.t,
                        *record.desired_position.tolist(),
                        *record.actual_position.tolist(),
                    ]
                )
