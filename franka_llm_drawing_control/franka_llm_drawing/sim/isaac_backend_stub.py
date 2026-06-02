"""Isaac backend placeholder.

This module intentionally avoids top-level Isaac imports. Fill it in only after
the USD scene, prim paths, end-effector frame, and contact sensors are known.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IsaacBackendConfig:
    usd_path: str | None = None
    robot_prim_path: str | None = None
    ee_frame_name: str | None = None
    board_prim_path: str | None = None
    pen_tip_frame_name: str | None = None
    contact_sensor_path: str | None = None


class IsaacFrankaBackend:
    """Placeholder for future Isaac Lab / Isaac Sim integration."""

    def __init__(self, config: IsaacBackendConfig) -> None:
        self.config = config
        raise NotImplementedError(
            "IsaacFrankaBackend requires the final USD scene and Isaac environment. "
            "Use MockFrankaBackend for offline tests."
        )
