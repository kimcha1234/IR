import os

import pytest

from robot_drawing_planner.llm_client import parse_goal


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE_OPENAI_SMOKE") != "1",
        reason="live OpenAI smoke test is skipped by default",
    ),
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY is not set",
    ),
]


def test_live_openai_structured_output_smoke():
    parsed = parse_goal("draw a 2 cm circle at the center")
    assert parsed.shape_type == "circle"
    assert parsed.radius is not None or parsed.size is not None
