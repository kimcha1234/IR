"""LangChain unit-action tools backed by PlanBuilder state."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import StructuredTool

from robot_drawing_planner.config import PlannerConfig
from robot_drawing_planner.plan_state import PlanBuilder

TOOL_SCOPE_NOTE = (
    "This tool appends symbolic robot primitive actions to a plan. "
    "It does not move the robot. It does not compute IK, FK, Jacobians, "
    "joint commands, trajectory samples, or Isaac Sim commands."
)


class UnitActionToolset:
    """Stateful LangChain tool wrapper for agentic unit-action planning."""

    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()
        self.builder: PlanBuilder | None = None
        self._checked_since_last_mutation = False

    def tools(self) -> list[StructuredTool]:
        """Return LangChain tools that share this toolset's PlanBuilder state."""

        return [
            StructuredTool.from_function(
                func=self.begin_plan,
                name="begin_plan",
                description=(
                    "Begin a new symbolic robot drawing plan for the user's command. "
                    f"{TOOL_SCOPE_NOTE} Call this before any other unit-action tool."
                ),
            ),
            StructuredTool.from_function(
                func=self.move_to_start,
                name="move_to_start",
                description=(
                    "Append move_to_start at board-frame x/y meters while the pen is up. "
                    f"{TOOL_SCOPE_NOTE}"
                ),
            ),
            StructuredTool.from_function(
                func=self.align_pen_orientation,
                name="align_pen_orientation",
                description=(
                    "Append a symbolic normal-to-board pen orientation constraint. "
                    f"{TOOL_SCOPE_NOTE}"
                ),
            ),
            StructuredTool.from_function(
                func=self.pen_down,
                name="pen_down",
                description=(
                    "Append pen_down at the current board-frame position. "
                    f"{TOOL_SCOPE_NOTE}"
                ),
            ),
            StructuredTool.from_function(
                func=self.draw_line_to,
                name="draw_line_to",
                description=(
                    "Append draw_line from current_position to board-frame x/y meters. "
                    f"{TOOL_SCOPE_NOTE}"
                ),
            ),
            StructuredTool.from_function(
                func=self.draw_arc,
                name="draw_arc",
                description=(
                    "Append draw_arc with board-frame center, radius, angles, and direction. "
                    f"{TOOL_SCOPE_NOTE}"
                ),
            ),
            StructuredTool.from_function(
                func=self.pen_up,
                name="pen_up",
                description=(
                    "Append pen_up after drawing. "
                    f"{TOOL_SCOPE_NOTE}"
                ),
            ),
            StructuredTool.from_function(
                func=self.check_plan,
                name="check_plan",
                description=(
                    "Inspect the current symbolic plan state and validation feedback. "
                    f"{TOOL_SCOPE_NOTE} The LLM must call check_plan before finish_plan."
                ),
            ),
            StructuredTool.from_function(
                func=self.finish_plan,
                name="finish_plan",
                description=(
                    "Finish the symbolic plan and return a serializable plan payload. "
                    f"{TOOL_SCOPE_NOTE} The LLM must call check_plan before finish_plan."
                ),
            ),
        ]

    def begin_plan(self, source_command: str) -> dict[str, Any]:
        """Begin a fresh PlanBuilder state."""

        self.builder = PlanBuilder.begin_plan(source_command, config=self.config)
        self._checked_since_last_mutation = False
        return self._feedback(True, "Plan started.")

    def move_to_start(self, x: float, y: float) -> dict[str, Any]:
        """Append move_to_start at board-frame meters."""

        builder = self._builder_or_none()
        if builder is None:
            return self._feedback(False, "begin_plan must be called before move_to_start.")
        before_errors = len(builder.errors)
        before_failed = len(builder.failed_calls)
        builder.move_to_start(x, y)
        self._checked_since_last_mutation = False
        return self._feedback_from_error_delta(
            before_errors,
            before_failed,
            "move_to_start appended.",
        )

    def align_pen_orientation(self) -> dict[str, Any]:
        """Append align_pen_orientation."""

        builder = self._builder_or_none()
        if builder is None:
            return self._feedback(False, "begin_plan must be called before align_pen_orientation.")
        before_errors = len(builder.errors)
        before_failed = len(builder.failed_calls)
        builder.align_pen_orientation()
        self._checked_since_last_mutation = False
        return self._feedback_from_error_delta(
            before_errors,
            before_failed,
            "align_pen_orientation appended.",
        )

    def pen_down(self) -> dict[str, Any]:
        """Append pen_down."""

        builder = self._builder_or_none()
        if builder is None:
            return self._feedback(False, "begin_plan must be called before pen_down.")
        before_errors = len(builder.errors)
        before_failed = len(builder.failed_calls)
        builder.pen_down()
        self._checked_since_last_mutation = False
        return self._feedback_from_error_delta(
            before_errors,
            before_failed,
            "pen_down appended.",
        )

    def draw_line_to(self, x: float, y: float) -> dict[str, Any]:
        """Append draw_line to board-frame x/y meters."""

        builder = self._builder_or_none()
        if builder is None:
            return self._feedback(False, "begin_plan must be called before draw_line_to.")
        before_errors = len(builder.errors)
        before_failed = len(builder.failed_calls)
        builder.draw_line_to(x, y)
        self._checked_since_last_mutation = False
        return self._feedback_from_error_delta(
            before_errors,
            before_failed,
            "draw_line appended.",
        )

    def draw_arc(
        self,
        center_x: float,
        center_y: float,
        radius_m: float,
        start_angle_rad: float,
        end_angle_rad: float,
        direction: Literal["cw", "ccw"],
    ) -> dict[str, Any]:
        """Append draw_arc symbolic geometry."""

        builder = self._builder_or_none()
        if builder is None:
            return self._feedback(False, "begin_plan must be called before draw_arc.")
        before_errors = len(builder.errors)
        before_failed = len(builder.failed_calls)
        builder.draw_arc(
            center_x=center_x,
            center_y=center_y,
            radius_m=radius_m,
            start_angle_rad=start_angle_rad,
            end_angle_rad=end_angle_rad,
            direction=direction,
        )
        self._checked_since_last_mutation = False
        return self._feedback_from_error_delta(
            before_errors,
            before_failed,
            "draw_arc appended.",
        )

    def pen_up(self) -> dict[str, Any]:
        """Append pen_up."""

        builder = self._builder_or_none()
        if builder is None:
            return self._feedback(False, "begin_plan must be called before pen_up.")
        before_errors = len(builder.errors)
        before_failed = len(builder.failed_calls)
        builder.pen_up()
        self._checked_since_last_mutation = False
        return self._feedback_from_error_delta(
            before_errors,
            before_failed,
            "pen_up appended.",
        )

    def check_plan(self) -> dict[str, Any]:
        """Return concise structured feedback for current PlanBuilder state."""

        builder = self._builder_or_none()
        if builder is None:
            return self._feedback(False, "begin_plan must be called before check_plan.")
        self._checked_since_last_mutation = True
        return self._feedback(not builder.errors, "Plan checked.")

    def finish_plan(self) -> dict[str, Any]:
        """Finish plan and return a serializable plan payload."""

        builder = self._builder_or_none()
        if builder is None:
            return self._feedback(False, "begin_plan must be called before finish_plan.")
        if not self._checked_since_last_mutation:
            return self._feedback(
                False,
                "finish_plan requires check_plan to be called after the last plan change.",
            )
        before_errors = len(builder.errors)
        before_failed = len(builder.failed_calls)
        builder.finish_plan()
        new_errors = [
            *builder.errors[before_errors:],
            *builder.failed_calls[before_failed:],
        ]
        ok = not new_errors and not builder.errors and builder.finished
        message = "Plan finished." if ok else (new_errors[-1] if new_errors else builder.errors[-1])
        feedback = self._feedback(ok, message, visible_errors=new_errors or None)
        if ok:
            feedback["plan"] = self._plan_payload()
        return feedback

    def tool_by_name(self, name: str) -> StructuredTool:
        """Return one tool by name for tests and simple orchestration."""

        tools = {tool.name: tool for tool in self.tools()}
        return tools[name]

    def _builder_or_none(self) -> PlanBuilder | None:
        return self.builder

    def _require_builder(self) -> PlanBuilder:
        if self.builder is None:
            raise RuntimeError("begin_plan must be called before unit-action tools.")
        return self.builder

    def _feedback_from_error_delta(
        self,
        previous_error_count: int,
        previous_failed_count: int,
        success_message: str,
    ) -> dict[str, Any]:
        builder = self._require_builder()
        new_errors = [
            *builder.errors[previous_error_count:],
            *builder.failed_calls[previous_failed_count:],
        ]
        if new_errors:
            return self._feedback(False, new_errors[-1], visible_errors=new_errors)
        return self._feedback(True, success_message)

    def _feedback(
        self,
        ok: bool,
        message: str,
        visible_errors: list[str] | None = None,
    ) -> dict[str, Any]:
        builder = self.builder
        if builder is None:
            return {
                "ok": ok,
                "message": message,
                "current_position": None,
                "pen_state": "up",
                "action_count": 0,
                "warnings": [],
                "errors": visible_errors or ([message] if not ok else []),
                "failed_calls": [],
            }
        summary = builder.check_plan()
        return {
            "ok": ok,
            "message": message,
            "current_position": summary["current_position"],
            "pen_state": summary["pen_state"],
            "action_count": summary["number_of_actions"],
            "warnings": summary["warnings"],
            "errors": visible_errors if visible_errors is not None else summary["errors"],
            "failed_calls": summary["failed_calls"],
        }

    def _plan_payload(self) -> dict[str, Any]:
        builder = self._require_builder()
        return {
            "schema_version": "1.0",
            "source_command": builder.source_command,
            "frame": builder.frame,
            "finished": builder.finished,
            "strokes": [stroke.model_dump(mode="json") for stroke in builder.strokes],
            "actions": [action.model_dump(mode="json") for action in builder.actions],
            "diagnostics": {
                "warnings": list(builder.warnings),
                "errors": list(builder.errors),
                "failed_calls": list(builder.failed_calls),
                "note": (
                    "This agentic unit-action planner produces symbolic primitive actions only; "
                    "it does not compute IK, FK, Jacobians, joint commands, trajectory samples, "
                    "or Isaac Sim commands."
                ),
            },
        }


def create_unit_action_tools(
    config: PlannerConfig | None = None,
) -> tuple[UnitActionToolset, list[StructuredTool]]:
    """Create a stateful unit-action toolset and its LangChain tools."""

    toolset = UnitActionToolset(config=config)
    return toolset, toolset.tools()
