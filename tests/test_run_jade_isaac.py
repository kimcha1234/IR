import numpy as np

from franka_llm_drawing.frames import CartesianTrajectoryPoint, make_transform
from franka_llm_drawing.llm_bridge.plan_schema import DrawingPlan, PrimitiveAction
from examples.run_jade_isaac import (
    _TangentialPositionIntegralController,
    _apply_command_position_offset,
    _status_with_orientation_requirement,
    scale_plan_speeds,
)


def test_scale_plan_speeds_scales_only_speed_params() -> None:
    plan = DrawingPlan(
        actions=[
            PrimitiveAction("move_to_start", {"speed_m_s": 0.08, "name": "kept"}),
            PrimitiveAction("pen_up", {"lift_distance_m": 0.02}),
        ]
    )

    scaled = scale_plan_speeds(plan, 0.5)

    assert scaled.actions[0].params["speed_m_s"] == 0.04
    assert scaled.actions[0].params["name"] == "kept"
    assert scaled.actions[1].params["lift_distance_m"] == 0.02
    assert plan.actions[0].params["speed_m_s"] == 0.08


def test_orientation_requirement_can_raise_status() -> None:
    status, reason = _status_with_orientation_requirement(
        "OK",
        "kinematics ok",
        orientation_error_deg=12.0,
        warning_deg=3.0,
        fail_deg=8.0,
    )

    assert status == "FAIL"
    assert "tip_orientation_error=12.00deg" in reason


def test_apply_command_position_offset_updates_tip_and_ee_pose() -> None:
    T = make_transform(np.eye(3), np.array([0.5, 0.0, 0.2]))
    target = CartesianTrajectoryPoint(
        t=0.0,
        p_base_tip=T[:3, 3],
        R_base_tip=T[:3, :3],
        T_base_tip=T,
        T_base_ee=T,
        pen_state="down",
        source_action_name="draw_arc",
    )

    shifted = _apply_command_position_offset(target, (0.005, 0.0, 0.0), np.eye(4))

    assert np.allclose(shifted.p_base_tip, np.array([0.505, 0.0, 0.2]))
    assert np.allclose(shifted.T_base_tip[:3, 3], shifted.p_base_tip)
    assert np.allclose(shifted.T_base_ee[:3, 3], shifted.p_base_tip)
    assert np.allclose(target.p_base_tip, np.array([0.5, 0.0, 0.2]))


def test_tangential_integral_changes_only_board_tangent_axes() -> None:
    T = make_transform(np.eye(3), np.array([0.5, 0.0, 0.2]))
    target = CartesianTrajectoryPoint(
        t=0.0,
        p_base_tip=T[:3, 3],
        R_base_tip=T[:3, :3],
        T_base_tip=T,
        T_base_ee=T,
        pen_state="down",
        source_action_name="draw_arc",
        pen_contact_desired=True,
    )
    controller = _TangentialPositionIntegralController(
        enabled=True,
        gain=1.0,
        leak_per_s=0.0,
        max_offset_m=0.1,
    )

    shifted = controller.update(
        target,
        actual_tip_position=np.array([0.49, -0.02, 0.1]),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        T_ee_tip=np.eye(4),
        dt=0.1,
    )

    assert shifted.p_base_tip[0] > target.p_base_tip[0]
    assert shifted.p_base_tip[1] > target.p_base_tip[1]
    assert shifted.p_base_tip[2] == target.p_base_tip[2]


def test_tangential_integral_smoothly_releases_during_pen_up() -> None:
    T = make_transform(np.eye(3), np.array([0.5, 0.0, 0.2]))
    draw_target = CartesianTrajectoryPoint(
        t=0.0,
        p_base_tip=T[:3, 3],
        R_base_tip=T[:3, :3],
        T_base_tip=T,
        T_base_ee=T,
        pen_state="down",
        source_action_name="draw_arc",
        pen_contact_desired=True,
    )
    pen_up_target = CartesianTrajectoryPoint(
        t=0.1,
        p_base_tip=T[:3, 3],
        R_base_tip=T[:3, :3],
        T_base_tip=T,
        T_base_ee=T,
        pen_state="up",
        source_action_name="pen_up",
        pen_contact_desired=False,
    )
    move_target = CartesianTrajectoryPoint(
        t=0.2,
        p_base_tip=T[:3, 3],
        R_base_tip=T[:3, :3],
        T_base_tip=T,
        T_base_ee=T,
        pen_state="up",
        source_action_name="move_to_start",
        pen_contact_desired=False,
    )
    controller = _TangentialPositionIntegralController(
        enabled=True,
        gain=1.0,
        leak_per_s=0.0,
        max_offset_m=0.1,
    )

    draw_shifted = controller.update(
        draw_target,
        actual_tip_position=np.array([0.49, -0.02, 0.2]),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        T_ee_tip=np.eye(4),
        dt=0.1,
    )
    pen_up_shifted = controller.update(
        pen_up_target,
        actual_tip_position=np.array([0.5, 0.0, 0.2]),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        T_ee_tip=np.eye(4),
        dt=0.1,
    )
    move_shifted = controller.update(
        move_target,
        actual_tip_position=np.array([0.5, 0.0, 0.2]),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        T_ee_tip=np.eye(4),
        dt=0.1,
    )
    next_draw = controller.update(
        draw_target,
        actual_tip_position=draw_target.p_base_tip,
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        T_ee_tip=np.eye(4),
        dt=0.1,
    )

    assert pen_up_shifted.p_base_tip[0] > pen_up_target.p_base_tip[0]
    assert pen_up_shifted.p_base_tip[1] > pen_up_target.p_base_tip[1]
    assert pen_up_shifted.p_base_tip[2] == pen_up_target.p_base_tip[2]
    assert np.linalg.norm(pen_up_shifted.p_base_tip - pen_up_target.p_base_tip) < np.linalg.norm(
        draw_shifted.p_base_tip - draw_target.p_base_tip
    )
    assert np.allclose(move_shifted.p_base_tip, move_target.p_base_tip)
    assert np.allclose(next_draw.p_base_tip, draw_target.p_base_tip)
