import numpy as np

from franka_llm_drawing.trajectory.path_primitives import (
    normalized_arc_delta,
    sample_arc,
    sample_line,
)


def test_line_starts_and_ends_at_expected_points() -> None:
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([0.1, 0.0, 0.0])

    samples = sample_line(start, end, speed_m_s=0.05, dt=0.01)

    assert np.allclose(samples[0].position, start)
    assert np.allclose(samples[-1].position, end)
    assert all(sample.pen_contact_desired for sample in samples)


def test_arc_points_lie_on_radius() -> None:
    center = np.array([0.1, -0.2, 0.0])
    radius = 0.05

    samples = sample_arc(center, radius, 0.0, np.pi, "ccw", speed_m_s=0.03, dt=0.01)
    distances = [np.linalg.norm(sample.position[:2] - center[:2]) for sample in samples]

    assert np.allclose(distances, radius)
    assert np.allclose(samples[0].position, np.array([0.15, -0.2, 0.0]))
    assert np.allclose(samples[-1].position, np.array([0.05, -0.2, 0.0]))


def test_arc_direction_changes_travel_sign() -> None:
    assert normalized_arc_delta(0.0, np.pi / 2.0, "ccw") > 0.0
    assert normalized_arc_delta(0.0, np.pi / 2.0, "cw") < 0.0

    ccw = sample_arc(np.zeros(3), 1.0, 0.0, np.pi / 2.0, "ccw", 1.0, 0.05)
    cw = sample_arc(np.zeros(3), 1.0, 0.0, np.pi / 2.0, "cw", 1.0, 0.05)

    assert ccw[1].position[1] > 0.0
    assert cw[1].position[1] < 0.0
