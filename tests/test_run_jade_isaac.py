import numpy as np

from franka_llm_drawing.evaluation import ExecutionLogger, ExecutionLogRow
from franka_llm_drawing.config import ExecutorConfig
from franka_llm_drawing.frames import CartesianTrajectoryPoint, make_transform
from franka_llm_drawing.llm_bridge.plan_schema import DrawingPlan, PrimitiveAction
from examples.run_jade_isaac import (
    _TangentialPositionIntegralController,
    _apply_command_position_offset,
    _drawing_rows,
    _force_tracking_summary,
    _select_lookahead_target,
    _servo_iterations_for_target,
    _startup_safe_approach_targets,
    _status_with_orientation_requirement,
    _trajectory_waypoint_index,
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


def test_startup_safe_approach_targets_lift_and_descend_before_first_target() -> None:
    T = make_transform(np.eye(3), np.array([0.5, 0.0, 0.22]))
    current_T = make_transform(np.eye(3), np.array([0.48, -0.01, 0.215]))
    first_target = CartesianTrajectoryPoint(
        t=0.0,
        p_base_tip=T[:3, 3],
        R_base_tip=T[:3, :3],
        T_base_tip=T,
        T_base_ee=T,
        pen_state="up",
        source_action_name="move_to_start",
        pen_contact_desired=False,
    )

    targets = _startup_safe_approach_targets(
        first_target,
        height_m=0.05,
        lift_steps=2,
        lateral_steps=2,
        descent_steps=2,
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        T_ee_tip=np.eye(4),
        current_T_base_tip=current_T,
    )

    assert len(targets) == 6
    assert [target.source_action_name for target in targets] == [
        "startup_lift",
        "startup_lift",
        "startup_lateral",
        "startup_lateral",
        "startup_descent",
        "startup_descent",
    ]
    assert all(not target.pen_contact_desired for target in targets)
    assert all(target.desired_normal_force_n is None for target in targets)
    assert np.allclose(targets[0].p_base_tip, np.array([0.48, -0.01, 0.24]))
    assert np.allclose(targets[1].p_base_tip, np.array([0.48, -0.01, 0.265]))
    assert np.allclose(targets[2].p_base_tip, np.array([0.49, -0.005, 0.2675]))
    assert np.allclose(targets[3].p_base_tip, np.array([0.5, 0.0, 0.27]))
    assert np.allclose(targets[4].p_base_tip, np.array([0.5, 0.0, 0.245]))
    assert np.allclose(targets[5].p_base_tip, np.array([0.5, 0.0, 0.22]))
    assert np.allclose(first_target.p_base_tip, np.array([0.5, 0.0, 0.22]))


def test_startup_safe_approach_is_disabled_when_height_or_steps_are_zero() -> None:
    T = make_transform(np.eye(3), np.array([0.5, 0.0, 0.22]))
    first_target = CartesianTrajectoryPoint(
        t=0.0,
        p_base_tip=T[:3, 3],
        R_base_tip=T[:3, :3],
        T_base_tip=T,
        T_base_ee=T,
        pen_state="up",
        source_action_name="move_to_start",
        pen_contact_desired=False,
    )

    targets = _startup_safe_approach_targets(
        first_target,
        height_m=0.0,
        lift_steps=100,
        lateral_steps=100,
        descent_steps=80,
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        T_ee_tip=np.eye(4),
    )

    assert targets == []


def test_trajectory_waypoint_index_excludes_startup_and_settle_targets() -> None:
    assert _trajectory_waypoint_index(0, pre_trajectory_count=180, trajectory_count=10) == 0
    assert _trajectory_waypoint_index(179, pre_trajectory_count=180, trajectory_count=10) == 0
    assert _trajectory_waypoint_index(180, pre_trajectory_count=180, trajectory_count=10) == 0
    assert _trajectory_waypoint_index(181, pre_trajectory_count=180, trajectory_count=10) == 1
    assert _trajectory_waypoint_index(999, pre_trajectory_count=180, trajectory_count=10) == 9


def test_servo_iterations_preserve_default_differential_mode() -> None:
    target = _trajectory_target("draw_line", "stroke_1", x=0.0)

    iterations = _servo_iterations_for_target(
        target,
        ExecutorConfig(tracking_mode="differential", iterative_servo_iterations=5),
    )

    assert iterations == 1


def test_iterative_servo_can_be_limited_to_drawing_contact_targets() -> None:
    draw_target = _trajectory_target("draw_line", "stroke_1", x=0.0)
    move_target = _trajectory_target("move_to_start", "unknown", x=0.0)
    config = ExecutorConfig(
        tracking_mode="iterative_servo",
        iterative_servo_iterations=3,
        iterative_servo_drawing_only=True,
    )

    assert _servo_iterations_for_target(draw_target, config) == 3
    assert _servo_iterations_for_target(move_target, config) == 1


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


def test_draw_only_force_summary_excludes_non_drawing_contact() -> None:
    logger = ExecutionLogger()
    logger.append(_execution_row("move_to_start", measured_force=40.0))
    logger.append(_execution_row("draw_line", measured_force=3.0))
    logger.append(_execution_row("pen_up", measured_force=10.0))

    all_force = _force_tracking_summary(logger)
    draw_force = _force_tracking_summary(logger, rows=_drawing_rows(logger))

    assert all_force["active_samples"] == 3
    assert all_force["measured_max_n"] == 40.0
    assert draw_force["active_samples"] == 1
    assert draw_force["measured_mean_n"] == 3.0
    assert draw_force["measured_max_n"] == 3.0


def test_lookahead_selects_future_target_within_same_stroke() -> None:
    targets = [
        _trajectory_target("draw_line", "stroke_1", x=0.00),
        _trajectory_target("draw_line", "stroke_1", x=0.01),
        _trajectory_target("draw_line", "stroke_1", x=0.02),
        _trajectory_target("draw_line", "stroke_2", x=0.03),
    ]

    selected = _select_lookahead_target(
        targets,
        0,
        lookahead_steps=3,
        drawing_only=True,
        same_stroke_only=True,
    )

    assert selected.stroke_id == "stroke_1"
    assert selected.p_base_tip[0] == 0.02


def test_lookahead_does_not_shift_non_drawing_action_when_drawing_only() -> None:
    targets = [
        _trajectory_target("move_to_start", "unknown", x=0.00),
        _trajectory_target("draw_line", "stroke_1", x=0.01),
    ]

    selected = _select_lookahead_target(
        targets,
        0,
        lookahead_steps=1,
        drawing_only=True,
        same_stroke_only=True,
    )

    assert selected is targets[0]


def _trajectory_target(action_type: str, stroke_id: str, *, x: float) -> CartesianTrajectoryPoint:
    T = make_transform(np.eye(3), np.array([x, 0.0, 0.2]))
    return CartesianTrajectoryPoint(
        t=0.0,
        p_base_tip=T[:3, 3],
        R_base_tip=T[:3, :3],
        T_base_tip=T,
        T_base_ee=T,
        pen_state="down" if action_type.startswith("draw") else "up",
        source_action_name=action_type,
        stroke_id=stroke_id,
        pen_contact_desired=action_type.startswith("draw"),
    )


def _execution_row(action_type: str, *, measured_force: float) -> ExecutionLogRow:
    desired = np.array([0.5, 0.0, 0.2])
    actual = np.array([0.49, 0.0, 0.2])
    return ExecutionLogRow(
        t=0.0,
        stroke_id="test",
        action_type=action_type,
        desired_position=desired,
        actual_position=actual,
        commanded_position=desired,
        q=np.zeros(7),
        qdot=np.zeros(7),
        sigma_min=0.1,
        condition_number=3.0,
        manipulability=0.1,
        lambda_dls=0.02,
        speed_scale=1.0,
        status="OK",
        joint_limit_margin=1.0,
        tip_orientation_error_deg=0.0,
        desired_normal_force_n=1.0,
        measured_normal_force_n=measured_force,
        filtered_normal_force_n=measured_force,
        normal_force_error_n=1.0 - measured_force,
        normal_force_offset_m=0.0,
        force_control_active=True,
        contact_active=measured_force > 0.05,
    )
