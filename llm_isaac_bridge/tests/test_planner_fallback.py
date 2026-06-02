from __future__ import annotations

import unittest

from llm_isaac_bridge.paths import DEFAULT_PLANNER_CONFIG
from llm_isaac_bridge.planner import _fallback_no_api_plan


class PlannerFallbackTest(unittest.TestCase):
    def test_korean_radius_is_not_treated_as_diameter(self) -> None:
        plan = _fallback_no_api_plan("중앙에 반지름 4cm짜리 원을 그려줘", DEFAULT_PLANNER_CONFIG)
        self.assertAlmostEqual(plan["goal"]["radius_m"], 0.04)

    def test_korean_diameter_is_halved(self) -> None:
        plan = _fallback_no_api_plan("중앙에 지름 4cm짜리 원을 그려줘", DEFAULT_PLANNER_CONFIG)
        self.assertAlmostEqual(plan["goal"]["radius_m"], 0.02)

    def test_square_fallback_has_four_line_actions(self) -> None:
        plan = _fallback_no_api_plan("중앙에 한 변 6cm짜리 사각형을 그려줘", DEFAULT_PLANNER_CONFIG)
        draw_lines = [action for action in plan["actions"] if action["name"] == "draw_line"]
        self.assertEqual(len(draw_lines), 4)

    def test_letter_a_fallback_has_three_strokes(self) -> None:
        plan = _fallback_no_api_plan("중앙에 알파벳 A를 8cm 크기로 그려줘", DEFAULT_PLANNER_CONFIG)
        draw_lines = [action for action in plan["actions"] if action["name"] == "draw_line"]
        pen_ups = [action for action in plan["actions"] if action["name"] == "pen_up"]
        self.assertEqual(len(draw_lines), 3)
        self.assertEqual(len(pen_ups), 3)
        self.assertEqual(plan["goal"]["letter"], "A")

    def test_house_fallback_has_outline(self) -> None:
        plan = _fallback_no_api_plan("중앙에 8cm 집 모양을 그려줘", DEFAULT_PLANNER_CONFIG)
        draw_lines = [action for action in plan["actions"] if action["name"] == "draw_line"]
        self.assertEqual(len(draw_lines), 5)

    def test_star_fallback_has_five_segments(self) -> None:
        plan = _fallback_no_api_plan("중앙에 8cm 별을 그려줘", DEFAULT_PLANNER_CONFIG)
        draw_lines = [action for action in plan["actions"] if action["name"] == "draw_line"]
        self.assertEqual(len(draw_lines), 5)


if __name__ == "__main__":
    unittest.main()
