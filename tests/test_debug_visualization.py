import numpy as np

from franka_llm_drawing.sim.debug_visualization import DebugDrawConfig, IsaacDebugDrawVisualizer


class FakeDebugDraw:
    def __init__(self) -> None:
        self.clear_count = 0
        self.calls = []

    def clear_lines(self) -> None:
        self.clear_count += 1

    def draw_lines(self, starts, ends, colors, widths) -> None:
        self.calls.append((list(starts), list(ends), list(colors), list(widths)))


def test_debug_visualizer_draws_planned_path_actual_path_force_width_and_frame() -> None:
    fake = FakeDebugDraw()
    visualizer = IsaacDebugDrawVisualizer(
        fake,
        DebugDrawConfig(
            enabled=True,
            update_stride=1,
            planned_path_stride=1,
            frame_axis_length_m=0.1,
            force_scale_m_per_n=0.05,
            force_min_draw_n=0.01,
        ),
    )
    planned = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.1, 0.1, 0.0],
        ]
    )

    visualizer.set_planned_path(planned)
    visualizer.update(
        step_index=0,
        actual_tip_transform=np.eye(4),
        measured_normal_force_n=1.0,
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        record_actual_path=True,
    )
    second_tip = np.eye(4)
    second_tip[:3, 3] = np.array([0.05, 0.0, 0.0])
    visualizer.update(
        step_index=1,
        actual_tip_transform=second_tip,
        measured_normal_force_n=2.0,
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        record_actual_path=True,
    )

    starts, ends, colors, widths = fake.calls[-1]
    assert len(starts) == len(ends) == len(colors) == len(widths)
    assert len(starts) == 6  # two planned segments, one actual segment, three frame axes
    assert starts[2] == (0.0, 0.0, 0.0)
    assert ends[2] == (0.05, 0.0, 0.0)
    assert widths[2] > 3.0
    assert np.allclose(ends[-3], (0.15, 0.0, 0.0))
    assert np.allclose(ends[-2], (0.05, 0.1, 0.0))
    assert np.allclose(ends[-1], (0.05, 0.0, 0.1))


def test_debug_visualizer_stride_skips_updates_without_touching_draw() -> None:
    fake = FakeDebugDraw()
    visualizer = IsaacDebugDrawVisualizer(fake, DebugDrawConfig(enabled=True, update_stride=3))

    visualizer.update(
        step_index=1,
        actual_tip_transform=np.eye(4),
        measured_normal_force_n=None,
        board_normal_base=np.array([0.0, 0.0, 1.0]),
    )

    assert fake.calls == []
    assert fake.clear_count == 0


def test_debug_visualizer_does_not_connect_precontact_motion_to_drawing_path() -> None:
    fake = FakeDebugDraw()
    visualizer = IsaacDebugDrawVisualizer(
        fake,
        DebugDrawConfig(
            enabled=True,
            update_stride=1,
            draw_planned_path=False,
            draw_tip_frame=False,
        ),
    )
    precontact = np.eye(4)
    precontact[:3, 3] = np.array([-1.0, 0.0, 0.0])
    draw_first = np.eye(4)
    draw_first[:3, 3] = np.array([0.0, 0.0, 0.0])
    draw_second = np.eye(4)
    draw_second[:3, 3] = np.array([0.1, 0.0, 0.0])

    visualizer.update(
        step_index=0,
        actual_tip_transform=precontact,
        measured_normal_force_n=None,
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        record_actual_path=False,
    )
    visualizer.update(
        step_index=1,
        actual_tip_transform=draw_first,
        measured_normal_force_n=1.0,
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        record_actual_path=True,
    )
    visualizer.update(
        step_index=2,
        actual_tip_transform=draw_second,
        measured_normal_force_n=1.0,
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        record_actual_path=True,
    )

    starts, ends, _, _ = fake.calls[-1]
    assert starts == [(0.0, 0.0, 0.0)]
    assert ends == [(0.1, 0.0, 0.0)]
