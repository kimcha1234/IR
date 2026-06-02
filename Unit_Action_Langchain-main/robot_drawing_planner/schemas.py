"""Strict Pydantic schemas for drawing goals and primitive plans."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

ShapeType: TypeAlias = Literal["circle", "square", "triangle", "letter", "custom"]
PrimitiveActionName: TypeAlias = Literal[
    "move_to_start",
    "align_pen_orientation",
    "pen_down",
    "draw_line",
    "draw_arc",
    "pen_up",
]
FrameName: TypeAlias = Literal["board"]
Unit: TypeAlias = Literal["m", "cm", "mm"]
Direction: TypeAlias = Literal["cw", "ccw"]


class StrictBaseModel(BaseModel):
    """Base model for planner schemas that reject unknown fields."""

    model_config = ConfigDict(extra="forbid")


class Point2D(StrictBaseModel):
    """A 2D point in the board frame."""

    x: float = Field(description="X coordinate on the drawing board.")
    y: float = Field(description="Y coordinate on the drawing board.")
    unit: Literal["m"] = Field(default="m", description="Point unit; always meters.")


class Point3D(StrictBaseModel):
    """A 3D point in meters for future handoff metadata."""

    x: float = Field(description="X coordinate in meters.")
    y: float = Field(description="Y coordinate in meters.")
    z: float = Field(description="Z coordinate in meters.")
    unit: Literal["m"] = Field(default="m", description="Point unit; always meters.")


class Measurement(StrictBaseModel):
    """A positive scalar length with an explicit unit."""

    value: float = Field(gt=0, description="Positive measurement value.")
    unit: Unit = Field(description="Measurement unit.")


class ParsedGoal(StrictBaseModel):
    """Structured goal returned by LangChain structured output."""

    shape_type: ShapeType = Field(description="Requested drawable shape type.")
    center: Point2D | None = Field(
        default=None,
        description="Optional board-frame center point in meters.",
    )
    radius: Measurement | None = Field(
        default=None,
        description="Circle radius, if the command specifies one.",
    )
    side_length: Measurement | None = Field(
        default=None,
        description="Square or triangle side length, if specified.",
    )
    size: Measurement | None = Field(
        default=None,
        description="Generic positive size; diameter for circles, height for letters.",
    )
    orientation_deg: float = Field(
        default=0,
        description="Optional planar orientation in degrees.",
    )
    letter: str | None = Field(
        default=None,
        description="Requested letter when shape_type is letter.",
    )
    position_hint: str | None = Field(
        default=None,
        description="Natural-language position hint preserved for diagnostics.",
    )
    raw_command: str = Field(description="Original user drawing command.")

    @field_validator("letter")
    @classmethod
    def normalize_letter(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        return normalized or None

    @field_validator("raw_command")
    @classmethod
    def require_raw_command(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("raw_command must not be empty.")
        return value


class NormalizedGoal(StrictBaseModel):
    """Validated drawing goal normalized to board-frame meters and radians."""

    shape_type: ShapeType = Field(description="Validated shape type.")
    center: Point2D = Field(description="Board-frame center point in meters.")
    radius_m: float | None = Field(
        default=None,
        gt=0,
        description="Circle radius in meters, when applicable.",
    )
    side_length_m: float | None = Field(
        default=None,
        gt=0,
        description="Square or triangle side length in meters, when applicable.",
    )
    size_m: float | None = Field(
        default=None,
        gt=0,
        description="Generic normalized size in meters.",
    )
    orientation_rad: float = Field(description="Planar orientation in radians.")
    letter: str | None = Field(default=None, description="Validated supported letter.")
    frame: Literal["board"] = Field(default="board", description="Drawing frame.")
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made during normalization.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal normalization warnings.",
    )


class LineStroke(StrictBaseModel):
    """A deterministic straight-line stroke in the board frame."""

    type: Literal["line"] = Field(default="line", description="Stroke type.")
    stroke_id: str = Field(description="Stable stroke identifier.")
    start: Point2D = Field(description="Line start point.")
    end: Point2D = Field(description="Line end point.")


class ArcStroke(StrictBaseModel):
    """A deterministic circular arc stroke in the board frame."""

    type: Literal["arc"] = Field(default="arc", description="Stroke type.")
    stroke_id: str = Field(description="Stable stroke identifier.")
    center: Point2D = Field(description="Arc center point.")
    radius_m: float = Field(gt=0, description="Arc radius in meters.")
    start_angle_rad: float = Field(description="Arc start angle in radians.")
    end_angle_rad: float = Field(description="Arc end angle in radians.")
    direction: Direction = Field(description="Arc direction, clockwise or counterclockwise.")


Stroke: TypeAlias = Annotated[Union[LineStroke, ArcStroke], Field(discriminator="type")]


class PrimitiveAction(StrictBaseModel):
    """Robot-level primitive action, not a low-level robot command."""

    name: PrimitiveActionName = Field(description="Primitive action name.")
    frame: FrameName = Field(default="board", description="Reference frame.")
    stroke_id: str | None = Field(
        default=None,
        description="Related stroke identifier, if this action belongs to a stroke.",
    )
    params: dict[str, Any] = Field(description="Primitive action parameters.")


class DrawingPlan(StrictBaseModel):
    """Complete handoff plan for the downstream kinematics module."""

    schema_version: str = Field(default="1.0", description="Plan schema version.")
    source_command: str = Field(description="Original natural-language command.")
    goal: NormalizedGoal = Field(description="Validated normalized drawing goal.")
    strokes: list[Stroke] = Field(description="Deterministic board-frame strokes.")
    actions: list[PrimitiveAction] = Field(description="Robot-level primitive actions.")
    diagnostics: dict[str, Any] = Field(description="Planner diagnostics and metadata.")


class ValidationErrorReport(StrictBaseModel):
    """Structured validation report for caller-facing diagnostics."""

    ok: bool = Field(description="Whether validation succeeded.")
    errors: list[str] = Field(description="Validation errors.")
    warnings: list[str] = Field(description="Validation warnings.")


class Board(StrictBaseModel):
    """Known planar drawing board dimensions used for boundary validation."""

    width_m: float = Field(default=0.40, gt=0, description="Board width in meters.")
    height_m: float = Field(default=0.30, gt=0, description="Board height in meters.")
    frame: FrameName = Field(default="board", description="Board frame name.")
