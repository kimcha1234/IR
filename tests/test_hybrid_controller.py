import numpy as np

from franka_llm_drawing.controllers import (
    HybridPositionForceConfig,
    HybridPositionForceController,
    NormalForceAdmittanceConfig,
    NormalForceAdmittanceController,
)
from franka_llm_drawing.frames import CartesianTrajectoryPoint, make_transform


def _jacobian() -> np.ndarray:
    J = np.zeros((6, 7))
    J[:6, :6] = np.eye(6)
    return J


def _target(*, contact: bool = True, desired_force: float | None = 1.0) -> CartesianTrajectoryPoint:
    T = make_transform(np.eye(3), np.array([0.5, 0.0, 0.0]))
    return CartesianTrajectoryPoint(
        t=0.0,
        p_base_tip=T[:3, 3].copy(),
        R_base_tip=T[:3, :3].copy(),
        T_base_tip=T,
        T_base_ee=T,
        pen_state="down" if contact else "up",
        source_action_name="draw_line" if contact else "move_to_start",
        stroke_id="test",
        pen_contact_desired=contact,
        desired_normal_force_n=desired_force,
    )


def _pen_down_target() -> CartesianTrajectoryPoint:
    target = _target(contact=False, desired_force=None)
    return CartesianTrajectoryPoint(
        t=target.t,
        p_base_tip=target.p_base_tip,
        R_base_tip=target.R_base_tip,
        T_base_tip=target.T_base_tip,
        T_base_ee=target.T_base_ee,
        pen_state="up",
        source_action_name="pen_down",
        stroke_id=target.stroke_id,
        pen_contact_desired=False,
        desired_normal_force_n=None,
    )


def _pen_up_target() -> CartesianTrajectoryPoint:
    target = _target(contact=False, desired_force=None)
    return CartesianTrajectoryPoint(
        t=target.t,
        p_base_tip=target.p_base_tip,
        R_base_tip=target.R_base_tip,
        T_base_tip=target.T_base_tip,
        T_base_ee=target.T_base_ee,
        pen_state="up",
        source_action_name="pen_up",
        stroke_id=target.stroke_id,
        pen_contact_desired=False,
        desired_normal_force_n=None,
    )


def test_hybrid_torque_shape() -> None:
    controller = HybridPositionForceController()

    tau, diagnostics = controller.compute_torque(
        q=np.zeros(7),
        qd=np.zeros(7),
        jacobian_6xn=_jacobian(),
        p_des_base=np.array([0.1, 0.0, 0.0]),
        R_des_base=np.eye(3),
        p_cur_base=np.zeros(3),
        R_cur_base=np.eye(3),
        v_cur_base=None,
        w_cur_base=None,
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=None,
    )

    assert tau.shape == (7,)
    assert diagnostics.task_wrench.shape == (6,)


def test_no_contact_mode_uses_z_position_behavior() -> None:
    controller = HybridPositionForceController()

    _, diagnostics = controller.compute_torque(
        q=np.zeros(7),
        qd=np.zeros(7),
        jacobian_6xn=_jacobian(),
        p_des_base=np.array([0.0, 0.0, -0.01]),
        R_des_base=np.eye(3),
        p_cur_base=np.zeros(3),
        R_cur_base=np.eye(3),
        v_cur_base=np.zeros(3),
        w_cur_base=np.zeros(3),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=0.0,
        contact_active=False,
    )

    assert diagnostics.contact_active is False
    assert diagnostics.motion_wrench[2] < 0.0
    assert np.allclose(diagnostics.force_wrench, np.zeros(6))


def test_contact_mode_uses_normal_force_error() -> None:
    controller = HybridPositionForceController()

    _, diagnostics = controller.compute_torque(
        q=np.zeros(7),
        qd=np.zeros(7),
        jacobian_6xn=_jacobian(),
        p_des_base=np.zeros(3),
        R_des_base=np.eye(3),
        p_cur_base=np.zeros(3),
        R_cur_base=np.eye(3),
        v_cur_base=np.zeros(3),
        w_cur_base=np.zeros(3),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=0.2,
        contact_active=True,
    )

    assert diagnostics.contact_active is True
    assert diagnostics.normal_force_error > 0.0
    assert diagnostics.force_wrench[2] < 0.0


def test_torque_clipping() -> None:
    controller = HybridPositionForceController(
        HybridPositionForceConfig(kp_pos_xy=1e6, max_torque_nm=0.5)
    )

    tau, diagnostics = controller.compute_torque(
        q=np.zeros(7),
        qd=np.zeros(7),
        jacobian_6xn=_jacobian(),
        p_des_base=np.array([1.0, 0.0, 0.0]),
        R_des_base=np.eye(3),
        p_cur_base=np.zeros(3),
        R_cur_base=np.eye(3),
        v_cur_base=np.zeros(3),
        w_cur_base=np.zeros(3),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=0.0,
    )

    assert np.max(np.abs(tau)) <= 0.5
    assert np.max(np.abs(diagnostics.tau_cmd_raw)) > 0.5


def test_normal_force_admittance_presses_down_when_force_is_low() -> None:
    controller = NormalForceAdmittanceController(
        NormalForceAdmittanceConfig(kp_offset_m_per_n=0.001, max_offset_step_m=0.01)
    )

    adjusted, diagnostics = controller.update(
        _target(contact=True, desired_force=1.0),
        T_ee_tip=np.eye(4),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=0.0,
        dt=0.02,
    )

    assert diagnostics.force_control_active is True
    assert diagnostics.normal_force_error_n > 0.0
    assert adjusted.p_base_tip[2] < 0.0


def test_normal_force_admittance_lifts_when_force_is_high() -> None:
    controller = NormalForceAdmittanceController(
        NormalForceAdmittanceConfig(kp_offset_m_per_n=0.001, max_offset_step_m=0.01)
    )

    adjusted, diagnostics = controller.update(
        _target(contact=True, desired_force=1.0),
        T_ee_tip=np.eye(4),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=2.0,
        dt=0.02,
    )

    assert diagnostics.normal_force_error_n < 0.0
    assert adjusted.p_base_tip[2] > 0.0


def test_normal_force_admittance_does_not_adjust_pen_up_target() -> None:
    controller = NormalForceAdmittanceController()
    target = _target(contact=False, desired_force=None)

    adjusted, diagnostics = controller.update(
        target,
        T_ee_tip=np.eye(4),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=0.0,
        dt=0.02,
    )

    assert diagnostics.force_control_active is False
    assert np.allclose(adjusted.p_base_tip, target.p_base_tip)


def test_normal_force_admittance_releases_unwanted_contact() -> None:
    controller = NormalForceAdmittanceController(
        NormalForceAdmittanceConfig(kp_offset_m_per_n=0.001, max_offset_step_m=0.01)
    )

    adjusted, diagnostics = controller.update(
        _target(contact=False, desired_force=None),
        T_ee_tip=np.eye(4),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=2.0,
        dt=0.02,
    )

    assert diagnostics.force_control_active is True
    assert diagnostics.desired_normal_force_n == 0.0
    assert adjusted.p_base_tip[2] > 0.0


def test_normal_force_admittance_keeps_release_active_for_pen_up() -> None:
    controller = NormalForceAdmittanceController(
        NormalForceAdmittanceConfig(
            kp_offset_m_per_n=0.001,
            max_offset_step_m=0.001,
            force_filter_alpha=1.0,
        )
    )
    draw_target = _target(contact=True, desired_force=1.0)
    controller.update(
        draw_target,
        T_ee_tip=np.eye(4),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=5.0,
        dt=0.02,
    )
    controller.update(
        draw_target,
        T_ee_tip=np.eye(4),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=5.0,
        dt=0.02,
    )

    adjusted, diagnostics = controller.update(
        _pen_up_target(),
        T_ee_tip=np.eye(4),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=0.0,
        dt=0.02,
    )

    assert diagnostics.force_control_active is True
    assert diagnostics.desired_normal_force_n == 0.0
    assert adjusted.p_base_tip[2] > _pen_up_target().p_base_tip[2]


def test_normal_force_admittance_guards_pen_down_after_contact() -> None:
    controller = NormalForceAdmittanceController(
        NormalForceAdmittanceConfig(kp_offset_m_per_n=0.001, max_offset_step_m=0.01)
    )

    adjusted, diagnostics = controller.update(
        _pen_down_target(),
        T_ee_tip=np.eye(4),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=2.0,
        dt=0.02,
    )

    assert diagnostics.force_control_active is True
    assert diagnostics.desired_normal_force_n == 1.0
    assert diagnostics.normal_force_error_n < 0.0
    assert adjusted.p_base_tip[2] > 0.0


def test_normal_force_admittance_unwinds_lift_bias_when_contact_force_drops() -> None:
    controller = NormalForceAdmittanceController(
        NormalForceAdmittanceConfig(
            kp_offset_m_per_n=0.001,
            ki_offset_m_per_n_s=0.02,
            max_offset_step_m=0.01,
            force_filter_alpha=1.0,
        )
    )
    target = _target(contact=True, desired_force=1.0)

    lifted, high_force = controller.update(
        target,
        T_ee_tip=np.eye(4),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=5.0,
        dt=0.5,
    )
    recovered, low_force = controller.update(
        target,
        T_ee_tip=np.eye(4),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=0.0,
        dt=0.5,
    )

    assert high_force.normal_offset_m > 0.0
    assert low_force.normal_force_error_n > 0.0
    assert recovered.p_base_tip[2] < lifted.p_base_tip[2]
