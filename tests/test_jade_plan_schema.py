import numpy as np

from franka_llm_drawing.llm_bridge.plan_schema import DrawingPlan
from franka_llm_drawing.trajectory import sample_drawing_plan


def test_jade_type_schema_is_normalized() -> None:
    plan = DrawingPlan.from_mapping(
        {
            "actions": [
                {"id": "a0", "type": "move_to_start", "target": [0.05, 0.0, 0.02], "speed": 0.1},
                {"id": "a1", "type": "pen_down", "target_z": 0.0, "speed": 0.02},
                {
                    "id": "a2",
                    "type": "draw_arc",
                    "center": [0.0, 0.0, 0.0],
                    "radius": 0.05,
                    "theta_start_deg": 0.0,
                    "theta_end_deg": 90.0,
                    "clockwise": False,
                    "speed": 0.03,
                },
                {"id": "a3", "type": "pen_up", "lift_distance": 0.02, "speed": 0.1},
            ]
        }
    )

    assert plan.actions[0].name == "move_to_start"
    assert plan.actions[2].params["direction"] == "ccw"
    assert plan.actions[2].stroke_id == "a2"


def test_jade_schema_samples_with_quintic() -> None:
    plan = {
        "actions": [
            {"id": "start", "type": "move_to_start", "target": [0.0, 0.0, 0.02]},
            {"id": "down", "type": "pen_down", "target_z": 0.0},
            {"id": "line", "type": "draw_line", "end": [0.05, 0.0, 0.0]},
            {"id": "up", "type": "pen_up", "lift_distance": 0.02},
        ]
    }

    samples = sample_drawing_plan(plan, 0.01, 0.03, 0.02, 0.0, time_scaling="quintic")

    assert np.allclose(samples[0].position, np.array([0.0, 0.0, 0.02]))
    assert np.allclose(samples[-1].position[2], 0.02)
    assert any(sample.stroke_id == "line" for sample in samples)
