"""Runtime configuration for the planner."""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field

from robot_drawing_planner.schemas import Board, Direction, Point2D

DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 2


class PlannerConfig(BaseModel):
    """Planner-level defaults and board convention.

    Board frame convention:
    - Origin is at the board center.
    - X range is [-board_width_m / 2, board_width_m / 2].
    - Y range is [-board_height_m / 2, board_height_m / 2].
    - z = 0 is the drawing surface.

    These bounds are only for planner-level boundary validation, not robot
    reachability, IK feasibility, or Isaac Sim execution.
    """

    model_config = ConfigDict(extra="forbid")

    board_width_m: float = Field(default=0.50, gt=0, description="Board width in meters.")
    board_height_m: float = Field(default=0.35, gt=0, description="Board height in meters.")
    default_center: Point2D = Field(
        default_factory=lambda: Point2D(x=0.0, y=0.0),
        description="Default board-frame drawing center in meters.",
    )
    default_shape_size_m: float = Field(
        default=0.10,
        gt=0,
        description="Default nominal shape size in meters.",
    )
    default_circle_radius_m: float = Field(
        default=0.05,
        gt=0,
        description="Default circle radius in meters.",
    )
    hover_height_m: float = Field(
        default=0.03,
        gt=0,
        description="Planner hint for pen hover height above the board.",
    )
    drawing_z_m: float = Field(
        default=0.0,
        description="Drawing surface z coordinate in the board frame.",
    )
    default_speed_m_s: float = Field(
        default=0.03,
        gt=0,
        description="Default drawing primitive speed in meters per second.",
    )
    pen_down_speed_m_s: float = Field(
        default=0.01,
        gt=0,
        description="Pen-down primitive speed in meters per second.",
    )
    pen_up_speed_m_s: float = Field(
        default=0.02,
        gt=0,
        description="Pen-up primitive speed in meters per second.",
    )
    arc_default_direction: Direction = Field(
        default="ccw",
        description="Default direction for generated arcs.",
    )

    def board(self) -> Board:
        """Return the corresponding planner board dimensions."""

        return Board(width_m=self.board_width_m, height_m=self.board_height_m)


DEFAULT_CONFIG = PlannerConfig()
DEFAULT_BOARD = DEFAULT_CONFIG.board()


def get_openai_model() -> str:
    """Return the configured OpenAI model name."""

    return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)


def require_openai_api_key() -> str:
    """Return OPENAI_API_KEY or raise the project-required message."""

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Make sure it is exported in ~/.zshrc."
        )
    return key
