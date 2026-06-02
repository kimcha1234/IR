from __future__ import annotations

import math
import unittest

from llm_isaac_bridge.validation import ValidationConfig, validate_plan_payload


def circle_plan(radius: float = 0.04) -> dict:
    return {
        "schema_version": "1.0",
        "source_command": "draw a circle",
        "diagnostics": {"validation_ok": True, "errors": [], "warnings": []},
        "actions": [
            {
                "name": "move_to_start",
                "frame": "board",
                "params": {"target": {"x": radius, "y": 0.0, "z": 0.02, "unit": "m"}},
            },
            {"name": "align_pen_orientation", "frame": "board", "params": {}},
            {
                "name": "pen_down",
                "frame": "board",
                "params": {"target": {"x": radius, "y": 0.0, "z": 0.0, "unit": "m"}},
            },
            {
                "name": "draw_arc",
                "frame": "board",
                "params": {
                    "center": {"x": 0.0, "y": 0.0, "z": 0.0, "unit": "m"},
                    "radius_m": radius,
                    "start_angle_rad": 0.0,
                    "end_angle_rad": 2.0 * math.pi,
                    "direction": "ccw",
                    "speed_m_s": 0.025,
                },
            },
            {"name": "pen_up", "frame": "board", "params": {"lift_height_m": 0.02}},
        ],
    }


class PlanValidationTest(unittest.TestCase):
    def test_valid_circle_plan_passes(self) -> None:
        report = validate_plan_payload(circle_plan(), config=ValidationConfig())
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(report.drawable_action_count, 1)

    def test_outside_board_fails(self) -> None:
        plan = circle_plan(radius=0.20)
        report = validate_plan_payload(plan, config=ValidationConfig())
        self.assertFalse(report.ok)
        self.assertTrue(any("outside board" in item for item in report.errors))

    def test_arc_start_mismatch_fails(self) -> None:
        plan = circle_plan(radius=0.04)
        plan["actions"][2]["params"]["target"]["x"] = 0.0
        report = validate_plan_payload(plan, config=ValidationConfig())
        self.assertFalse(report.ok)
        self.assertTrue(any("draw_arc start" in item for item in report.errors))

    def test_pen_must_finish_up(self) -> None:
        plan = circle_plan()
        plan["actions"] = plan["actions"][:-1]
        report = validate_plan_payload(plan, config=ValidationConfig())
        self.assertFalse(report.ok)
        self.assertTrue(any("pen_state" in item for item in report.errors))


if __name__ == "__main__":
    unittest.main()
