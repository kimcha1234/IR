"""Run JADE hover tracing in Isaac Lab with the local Franka drawing USD."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from franka_llm_drawing.config import FRANKA_ARM_JOINT_NAMES, load_frames_config, load_isaac_scene_config, load_jade_config
from franka_llm_drawing.controllers import (
    NormalForceAdmittanceConfig,
    NormalForceAdmittanceController,
    NormalForceAdmittanceDiagnostics,
)
from franka_llm_drawing.evaluation import ExecutionLogger, ExecutionLogRow, write_jade_plots
from franka_llm_drawing.evaluation.metrics import max_xy_error, xy_rmse, z_rmse
from franka_llm_drawing.frames import ee_pose_to_tip_pose, make_transform, samples_to_cartesian_trajectory, tip_pose_to_ee_pose
from franka_llm_drawing.jade import AdaptiveDLSExecutor
from franka_llm_drawing.llm_bridge import load_plan_json
from franka_llm_drawing.sim import IsaacFrankaBackend
from franka_llm_drawing.trajectory import sample_drawing_plan
from franka_llm_drawing.trajectory.path_primitives import PoseSample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(PROJECT_ROOT / "examples" / "sample_plans" / "circle.json"))
    parser.add_argument("--frames", default=str(PROJECT_ROOT / "configs" / "frames.yaml"))
    parser.add_argument("--jade", default=str(PROJECT_ROOT / "configs" / "jade.yaml"))
    parser.add_argument("--scene", default=str(PROJECT_ROOT / "configs" / "isaac_scene.yaml"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "jade_isaac"))
    parser.add_argument("--mode", choices=["hover", "contact"], default="hover")
    parser.add_argument(
        "--path-speed-scale",
        type=float,
        default=1.0,
        help="Multiply all plan action speeds before sampling. Use <1.0 for tighter tracking.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Limit trajectory samples for quick smoke tests.")
    parser.add_argument("--settle-steps", type=int, default=150)
    parser.add_argument("--hold-steps", type=int, default=60)
    parser.add_argument("--disable-force-control", action="store_true", help="Disable contact-mode normal-force feedback.")
    parser.add_argument("--force-target-n", type=float, default=None, help="Override contact-mode desired normal force.")
    parser.add_argument("--keep-open", action="store_true", help="Keep Isaac Sim open after the trajectory finishes.")
    parser.add_argument("--setup-only", action="store_true", help="Load USD and robot, then exit before control loop.")
    parser.add_argument("--skip-close", action="store_true", help="Exit without calling SimulationApp.close().")
    return parser


def main() -> None:
    parser = build_parser()
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        print(
            "Isaac Lab modules are not importable in this Python session.\n"
            "Run this script through Isaac Lab, for example:\n"
            "  cd /home/kimchangyeol/IsaacLab\n"
            "  ./isaaclab.sh -p /home/kimchangyeol/IsaacLab/IR/examples/run_jade_isaac.py --mode hover --plan /home/kimchangyeol/IsaacLab/IR/examples/sample_plans/circle.json\n"
            "If conda base is active and Isaac imports still fail, run 'conda deactivate' first and retry.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        run_with_isaac(args, simulation_app)
    finally:
        if not args.setup_only and not args.skip_close:
            simulation_app.close()


def run_with_isaac(args: argparse.Namespace, simulation_app) -> None:
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import subtract_frame_transforms
    from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

    _log("loading JADE configs and drawing plan")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_frames_config(args.frames)
    jade = load_jade_config(args.jade)
    scene_cfg = load_isaac_scene_config(args.scene, project_root=PROJECT_ROOT)
    if scene_cfg.joint_names != FRANKA_ARM_JOINT_NAMES:
        raise RuntimeError("JADE Isaac backend must command only panda_joint1 through panda_joint7.")

    plan = load_plan_json(args.plan)
    if args.path_speed_scale <= 0.0:
        raise ValueError("--path-speed-scale must be positive.")
    if args.path_speed_scale != 1.0:
        plan = scale_plan_speeds(plan, args.path_speed_scale)
    desired_normal_force_n = (
        float(args.force_target_n)
        if args.force_target_n is not None
        else float(jade.force_control.desired_normal_force_n)
    )
    if desired_normal_force_n <= 0.0:
        raise ValueError("Desired normal force must be positive.")
    force_control_enabled = bool(args.mode == "contact" and jade.force_control.enabled and not args.disable_force_control)
    draw_height = frames.hover_height_m if args.mode == "hover" else frames.draw_height_m
    samples = sample_drawing_plan(
        plan,
        dt=jade.sampling.dt,
        default_speed_m_s=jade.sampling.default_speed_m_s,
        hover_height_m=frames.hover_height_m,
        draw_height_m=draw_height,
        default_normal_force_n=None if args.mode == "hover" else desired_normal_force_n,
        time_scaling=jade.sampling.time_scaling,
        max_linear_speed_m_s=jade.sampling.max_linear_speed_m_s,
        max_linear_accel_m_s2=jade.sampling.max_linear_accel_m_s2,
        max_linear_jerk_m_s3=jade.sampling.max_linear_jerk_m_s3,
        contact_force_ramp_duration_s=jade.sampling.contact_force_ramp_duration_s,
        pen_down_settle_duration_s=0.0 if args.mode == "hover" else jade.sampling.pen_down_settle_duration_s,
    )
    if args.mode == "hover":
        samples = force_hover_samples(samples, frames.hover_height_m)
    cartesian = samples_to_cartesian_trajectory(
        samples,
        frames.T_base_board,
        frames.T_ee_tip,
        R_base_tip_override=frames.R_base_tip,
        jacobian_tip_translation_m=frames.jacobian_tip_translation_m,
    )
    if args.max_samples > 0:
        cartesian = cartesian[: args.max_samples]
    if not cartesian:
        raise RuntimeError("No trajectory samples were generated.")

    _log(f"opening USD stage: {scene_cfg.usd_path}")
    if not sim_utils.open_stage(scene_cfg.usd_path):
        raise RuntimeError(f"Could not open USD stage: {scene_cfg.usd_path}")
    if force_control_enabled:
        _activate_contact_reporting(scene_cfg.robot_prim_path)
    sim_device = args.device if scene_cfg.use_gpu else "cpu"
    _log(f"creating SimulationContext with device={sim_device}")
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=scene_cfg.physics_dt, device=sim_device))
    sim.set_camera_view([1.2, -1.2, 0.85], [0.5, 0.0, 0.25])

    _log(f"binding existing Franka articulation at {scene_cfg.robot_prim_path}")
    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
    robot_cfg.prim_path = scene_cfg.robot_prim_path
    robot_cfg.spawn = None
    if force_control_enabled:
        _apply_contact_mode_actuator_gains(robot_cfg, jade.force_control)
        _log(
            "using contact-mode joint-position gains "
            f"stiffness={jade.force_control.contact_joint_stiffness:.1f}, "
            f"damping={jade.force_control.contact_joint_damping:.1f}"
        )
    robot = Articulation(cfg=robot_cfg)

    _log("resetting simulation and robot buffers")
    sim.reset()
    robot.reset()
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_ids = _resolve_arm_joint_ids(robot, scene_cfg.joint_names)
    if scene_cfg.initial_joint_positions_rad is not None:
        initial_q = torch.tensor(
            scene_cfg.initial_joint_positions_rad,
            dtype=joint_pos.dtype,
            device=joint_pos.device,
        )
        joint_pos[:, joint_ids] = initial_q.unsqueeze(0)
        joint_vel[:, joint_ids] = 0.0
        _log(f"applying configured initial arm posture: {list(scene_cfg.initial_joint_positions_rad)}")
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()
    robot.reset()

    _log("resolving Franka end-effector body")
    body_id = _resolve_body_id(robot, scene_cfg.ee_frame_name)
    ee_jacobi_idx = body_id - 1 if robot.is_fixed_base else body_id
    _log(f"resolved joint ids={joint_ids}, ee body={scene_cfg.ee_frame_name}[{body_id}], jacobian index={ee_jacobi_idx}")
    contact_sensor = None
    if force_control_enabled:
        contact_sensor = _create_contact_sensor(scene_cfg)
        _log(
            "enabled normal-force feedback with contact sensor "
            f"{scene_cfg.contact_sensor_prim_path}, target={desired_normal_force_n:.3f} N"
        )
    backend = IsaacFrankaBackend(
        robot=robot,
        sim=sim,
        joint_ids=joint_ids,
        body_id=body_id,
        ee_jacobi_idx=ee_jacobi_idx,
        torch_module=torch,
        subtract_frame_transforms=subtract_frame_transforms,
        contact_sensor=contact_sensor,
        board_normal_base=frames.board_normal_base,
    )
    if args.setup_only:
        print("[INFO] setup-only check completed; exiting before control loop.", flush=True)
        return
    executor = AdaptiveDLSExecutor(jade, fail_on_unsafe=False)
    tangent_integral = _TangentialPositionIntegralController(
        enabled=jade.executor.tangent_integral_enabled,
        gain=jade.executor.tangent_integral_gain,
        leak_per_s=jade.executor.tangent_integral_leak_per_s,
        max_offset_m=jade.executor.max_tangent_integral_offset_m,
    )
    force_controller = (
        NormalForceAdmittanceController(_normal_force_config(jade.force_control, desired_normal_force_n))
        if force_control_enabled
        else None
    )
    logger = ExecutionLogger()
    targets = [cartesian[0]] * max(args.settle_steps, 0) + cartesian + [cartesian[-1]] * max(args.hold_steps, 0)
    substeps = max(1, int(round(scene_cfg.control_dt / scene_cfg.physics_dt)))
    print("[INFO] JADE Isaac backend command joints:", ", ".join(scene_cfg.joint_names), flush=True)
    print(f"[INFO] USD: {scene_cfg.usd_path}", flush=True)
    if force_control_enabled:
        print(f"[INFO] normal force target: {desired_normal_force_n:.3f} N", flush=True)
    print(f"[INFO] trajectory samples: {len(cartesian)}, total control targets: {len(targets)}", flush=True)

    for index, target in enumerate(targets):
        if not simulation_app.is_running():
            break
        if index == 0 or index == args.settle_steps or index == len(targets) - 1:
            print(f"[INFO] control target {index + 1}/{len(targets)}", flush=True)
        command_target = _apply_command_position_offset(
            target,
            jade.executor.command_position_offset_m,
            frames.T_ee_tip,
        )
        pre_state = backend.get_state()
        pre_actual_tip = ee_pose_to_tip_pose(
            make_transform(pre_state.ee_rotation, pre_state.ee_position),
            frames.T_ee_tip,
        )
        command_target = tangent_integral.update(
            command_target,
            actual_tip_position=pre_actual_tip[:3, 3],
            board_normal_base=frames.board_normal_base,
            T_ee_tip=frames.T_ee_tip,
            dt=scene_cfg.control_dt,
        )
        force_diag: NormalForceAdmittanceDiagnostics | None = None
        if force_controller is not None:
            command_target, force_diag = force_controller.update(
                command_target,
                T_ee_tip=frames.T_ee_tip,
                board_normal_base=frames.board_normal_base,
                measured_normal_force_n=backend.get_measured_normal_force(),
                dt=scene_cfg.control_dt,
            )
        result = executor.step(
            backend,
            command_target,
            waypoint_index=min(index, len(cartesian) - 1),
            dt=scene_cfg.control_dt,
        )
        for _ in range(substeps):
            backend.step()
        state = backend.get_state()
        logged_normal_force = backend.get_measured_normal_force()
        actual_tip = ee_pose_to_tip_pose(make_transform(state.ee_rotation, state.ee_position), frames.T_ee_tip)
        tip_orientation_error_deg = _axis_alignment_error_deg(
            target.R_base_tip[:, 2],
            actual_tip[:3, :3][:, 2],
        )
        status, policy_reason = _status_with_orientation_requirement(
            result.policy.status,
            result.policy.reason,
            tip_orientation_error_deg,
            jade.thresholds.tip_orientation_warning_deg,
            jade.thresholds.tip_orientation_fail_deg,
        )
        logged_force_error = None
        logged_contact_active = None
        logged_force = None
        if force_diag is not None:
            logged_force = force_diag.measured_normal_force_n if logged_normal_force is None else logged_normal_force
            logged_force_error = (
                force_diag.desired_normal_force_n - logged_force
                if force_diag.force_control_active
                else 0.0
            )
            logged_contact_active = logged_force >= jade.force_control.contact_threshold_n
        logger.append(
            ExecutionLogRow(
                t=index * scene_cfg.control_dt,
                stroke_id=target.stroke_id or "unknown",
                action_type=target.source_action_name or "unknown",
                desired_position=target.p_base_tip,
                actual_position=actual_tip[:3, 3],
                commanded_position=command_target.p_base_tip,
                q=state.q,
                qdot=state.qd,
                sigma_min=result.metrics.sigma_min,
                condition_number=result.metrics.condition_number,
                manipulability=result.metrics.manipulability,
                lambda_dls=result.policy.lambda_dls,
                speed_scale=result.policy.speed_scale,
                status=status,
                joint_limit_margin=result.metrics.joint_limit_margin,
                tip_orientation_error_deg=tip_orientation_error_deg,
                desired_normal_force_n=force_diag.desired_normal_force_n if force_diag is not None else None,
                measured_normal_force_n=logged_force,
                filtered_normal_force_n=force_diag.filtered_normal_force_n if force_diag is not None else None,
                normal_force_error_n=logged_force_error,
                normal_force_offset_m=force_diag.normal_offset_m if force_diag is not None else None,
                force_control_active=force_diag.force_control_active if force_diag is not None else None,
                contact_active=logged_contact_active,
                policy_reason=policy_reason,
            )
        )

    logger.write_csv(out_dir / "execution_log.csv")
    write_jade_plots(logger, out_dir)
    desired = logger.desired_positions()
    actual = logger.actual_positions()
    summary = {
        "mode": args.mode,
        "samples_logged": len(logger.rows),
        "command_joint_names": list(scene_cfg.joint_names),
        "xy_rmse_m": xy_rmse(desired, actual),
        "max_xy_error_m": max_xy_error(desired, actual),
        "z_rmse_m": z_rmse(desired, actual),
    }
    settle_count = min(max(args.settle_steps, 0), len(logger.rows))
    hold_count = min(max(args.hold_steps, 0), max(0, len(logger.rows) - settle_count))
    draw_start = settle_count
    draw_stop = max(draw_start, len(logger.rows) - hold_count)
    if settle_count:
        summary["settle_mean_error_m"] = _mean_position_error(desired[:settle_count], actual[:settle_count])
    if draw_stop > draw_start:
        summary["draw_mean_error_m"] = _mean_position_error(desired[draw_start:draw_stop], actual[draw_start:draw_stop])
    orientation_errors = np.asarray(
        [row.tip_orientation_error_deg for row in logger.rows if row.tip_orientation_error_deg is not None],
        dtype=float,
    )
    if orientation_errors.size:
        summary["tip_orientation_rmse_deg"] = float(np.sqrt(np.mean(np.square(orientation_errors))))
        summary["tip_orientation_max_error_deg"] = float(np.max(orientation_errors))
    joint_margins = np.asarray(
        [row.joint_limit_margin for row in logger.rows if row.joint_limit_margin is not None],
        dtype=float,
    )
    if joint_margins.size:
        summary["min_joint_limit_margin_rad"] = float(np.min(joint_margins))
    force_summary = _force_tracking_summary(logger)
    if force_summary:
        summary["normal_force"] = force_summary
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] xy_rmse_m: {summary['xy_rmse_m']:.6f}", flush=True)
    print(f"[INFO] max_xy_error_m: {summary['max_xy_error_m']:.6f}", flush=True)
    print(f"[INFO] outputs: {out_dir}", flush=True)

    while args.keep_open and simulation_app.is_running():
        sim.step()


def force_hover_samples(samples: list[PoseSample], hover_height_m: float) -> list[PoseSample]:
    output: list[PoseSample] = []
    for sample in samples:
        position = sample.position.copy()
        position[2] = float(hover_height_m)
        output.append(
            replace(
                sample,
                position=position,
                pen_contact_desired=False,
                desired_normal_force_n=None,
            )
        )
    return output


def scale_plan_speeds(plan, speed_scale: float):
    """Return a copy of the drawing plan with primitive speeds scaled."""

    scaled_actions = []
    for action in plan.actions:
        params = dict(action.params)
        if "speed_m_s" in params:
            params["speed_m_s"] = float(params["speed_m_s"]) * float(speed_scale)
        scaled_actions.append(replace(action, params=params))
    return replace(plan, actions=scaled_actions)


def _normal_force_config(config, desired_normal_force_n: float) -> NormalForceAdmittanceConfig:
    return NormalForceAdmittanceConfig(
        enabled=bool(config.enabled),
        desired_normal_force_n=float(desired_normal_force_n),
        contact_threshold_n=float(config.contact_threshold_n),
        kp_offset_m_per_n=float(config.kp_offset_m_per_n),
        ki_offset_m_per_n_s=float(config.ki_offset_m_per_n_s),
        kd_offset_m_s_per_n=float(config.kd_offset_m_s_per_n),
        max_press_offset_m=float(config.max_press_offset_m),
        max_lift_offset_m=float(config.max_lift_offset_m),
        max_offset_step_m=float(config.max_offset_step_m),
        force_filter_alpha=float(config.force_filter_alpha),
    )


def _apply_command_position_offset(target, offset_m: tuple[float, float, float], T_ee_tip: np.ndarray):
    offset = np.asarray(offset_m, dtype=float)
    if offset.shape != (3,):
        raise ValueError("executor.command_position_offset_m must contain exactly 3 values.")
    if float(np.linalg.norm(offset)) < 1e-12:
        return target
    adjusted_position = np.asarray(target.p_base_tip, dtype=float) + offset
    adjusted_T_base_tip = make_transform(target.R_base_tip, adjusted_position)
    return replace(
        target,
        p_base_tip=adjusted_position,
        T_base_tip=adjusted_T_base_tip,
        T_base_ee=tip_pose_to_ee_pose(adjusted_T_base_tip, T_ee_tip),
    )


class _TangentialPositionIntegralController:
    """Small paper-tangent integral correction for residual XY tracking error."""

    def __init__(
        self,
        *,
        enabled: bool,
        gain: float,
        leak_per_s: float,
        max_offset_m: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.gain = float(gain)
        self.leak_per_s = max(float(leak_per_s), 0.0)
        self.max_offset_m = max(float(max_offset_m), 0.0)
        self._integral = np.zeros(3, dtype=float)

    def reset(self) -> None:
        self._integral[:] = 0.0

    def update(
        self,
        target,
        *,
        actual_tip_position: np.ndarray,
        board_normal_base: np.ndarray,
        T_ee_tip: np.ndarray,
        dt: float,
    ):
        if not self.enabled or self.gain <= 0.0 or self.max_offset_m <= 0.0:
            return target
        dt_s = max(float(dt), 1e-12)
        normal = _normalized(board_normal_base)
        tangent_projector = np.eye(3) - np.outer(normal, normal)
        if not target.pen_contact_desired:
            if target.source_action_name == "pen_up":
                release_leak_per_s = max(self.leak_per_s, 2.0)
                self._integral *= max(0.0, 1.0 - release_leak_per_s * dt_s)
                return self._apply_offset(target, tangent_projector, T_ee_tip)
            self.reset()
            return target

        error = np.asarray(target.p_base_tip, dtype=float) - np.asarray(actual_tip_position, dtype=float)
        tangent_error = tangent_projector @ error
        self._integral = max(0.0, 1.0 - self.leak_per_s * dt_s) * self._integral + tangent_error * dt_s
        return self._apply_offset(target, tangent_projector, T_ee_tip)

    def _apply_offset(self, target, tangent_projector: np.ndarray, T_ee_tip: np.ndarray):
        offset = tangent_projector @ (self.gain * self._integral)
        offset = _clip_norm(offset, self.max_offset_m)
        if self.gain > 1e-12:
            self._integral = offset / self.gain
        if float(np.linalg.norm(offset)) < 1e-12:
            return target
        adjusted_position = np.asarray(target.p_base_tip, dtype=float) + offset
        adjusted_T_base_tip = make_transform(target.R_base_tip, adjusted_position)
        return replace(
            target,
            p_base_tip=adjusted_position,
            T_base_tip=adjusted_T_base_tip,
            T_base_ee=tip_pose_to_ee_pose(adjusted_T_base_tip, T_ee_tip),
        )


def _clip_norm(value: np.ndarray, max_norm: float) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    limit = float(max_norm)
    if limit <= 0.0:
        return np.zeros_like(arr)
    norm = float(np.linalg.norm(arr))
    if norm > limit:
        return arr * (limit / norm)
    return arr


def _normalized(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError("normal vector must be nonzero.")
    return arr / norm


def _status_with_orientation_requirement(
    base_status: str,
    base_reason: str,
    orientation_error_deg: float,
    warning_deg: float,
    fail_deg: float,
) -> tuple[str, str]:
    if not np.isfinite(orientation_error_deg):
        return base_status, base_reason
    orientation_status = "OK"
    if orientation_error_deg >= float(fail_deg):
        orientation_status = "FAIL"
    elif orientation_error_deg >= float(warning_deg):
        orientation_status = "WARNING"
    if orientation_status == "OK":
        return base_status, base_reason
    reason = f"tip_orientation_error={orientation_error_deg:.2f}deg"
    if base_reason:
        reason = f"{base_reason}; {reason}"
    return _worst_status(base_status, orientation_status), reason


def _worst_status(left: str, right: str) -> str:
    rank = {"OK": 0, "WARNING": 1, "FAIL": 2}
    return left if rank.get(left, 0) >= rank.get(right, 0) else right


def _apply_contact_mode_actuator_gains(robot_cfg, force_config) -> None:
    stiffness = float(force_config.contact_joint_stiffness)
    damping = float(force_config.contact_joint_damping)
    if stiffness <= 0.0 or damping < 0.0:
        raise ValueError("contact_joint_stiffness must be positive and contact_joint_damping must be non-negative.")
    for actuator_name in ("panda_shoulder", "panda_forearm"):
        actuator_cfg = robot_cfg.actuators.get(actuator_name)
        if actuator_cfg is None:
            raise RuntimeError(f"Missing Franka actuator config: {actuator_name}")
        actuator_cfg.stiffness = stiffness
        actuator_cfg.damping = damping


def _activate_contact_reporting(robot_prim_path: str) -> None:
    from isaaclab.sim.schemas import activate_contact_sensors

    activate_contact_sensors(robot_prim_path, threshold=0.0)
    _log(f"activated PhysX contact reporting under {robot_prim_path}")


def _create_contact_sensor(scene_cfg):
    from isaaclab.sensors import ContactSensor, ContactSensorCfg

    filter_paths = list(scene_cfg.contact_filter_prim_paths)
    try:
        return _initialize_contact_sensor(
            ContactSensor,
            ContactSensorCfg,
            scene_cfg.contact_sensor_prim_path,
            filter_paths,
        )
    except Exception as exc:
        if not filter_paths:
            raise
        _log(
            "filtered contact sensor initialization failed; retrying without "
            f"contact filter. Original error: {exc}"
        )
        return _initialize_contact_sensor(
            ContactSensor,
            ContactSensorCfg,
            scene_cfg.contact_sensor_prim_path,
            [],
        )


def _initialize_contact_sensor(ContactSensor, ContactSensorCfg, prim_path: str, filter_paths: list[str]):
    sensor_cfg = ContactSensorCfg(
        prim_path=prim_path,
        update_period=0.0,
        history_length=4,
        debug_vis=False,
        filter_prim_paths_expr=filter_paths,
    )
    sensor = ContactSensor(sensor_cfg)
    if not sensor.is_initialized:
        sensor._initialize_impl()
        sensor._is_initialized = True
    sensor.reset()
    return sensor


def _resolve_arm_joint_ids(robot, joint_names: tuple[str, ...]) -> list[int]:
    available = list(robot.joint_names)
    print(f"[INFO][JADE] available joints: {available}", flush=True)
    missing = [name for name in joint_names if name not in available]
    if missing:
        raise RuntimeError(f"Missing required Franka arm joints: {missing}. Available joints: {available}")
    joint_ids = [available.index(name) for name in joint_names]
    if len(joint_ids) != 7:
        raise RuntimeError(f"Expected 7 Franka arm joint ids, got {joint_ids}.")
    return joint_ids


def _resolve_body_id(robot, body_name: str) -> int:
    available = list(robot.body_names)
    print(f"[INFO][JADE] available bodies: {available}", flush=True)
    if body_name not in available:
        raise RuntimeError(f"Missing end-effector body {body_name!r}. Available bodies: {available}")
    return int(available.index(body_name))


def _log(message: str) -> None:
    print(f"[INFO][JADE] {message}", flush=True)


def _axis_alignment_error_deg(desired_axis: np.ndarray, actual_axis: np.ndarray) -> float:
    desired = np.asarray(desired_axis, dtype=float)
    actual = np.asarray(actual_axis, dtype=float)
    desired_norm = float(np.linalg.norm(desired))
    actual_norm = float(np.linalg.norm(actual))
    if desired_norm < 1e-12 or actual_norm < 1e-12:
        return float("nan")
    cosine = float(np.clip(np.dot(desired, actual) / (desired_norm * actual_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _mean_position_error(desired: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    error = np.asarray(desired, dtype=float) - np.asarray(actual, dtype=float)
    mean_error = np.mean(error, axis=0)
    rmse_error = np.sqrt(np.mean(np.square(error), axis=0))
    return {
        "x_des_minus_act": float(mean_error[0]),
        "y_des_minus_act": float(mean_error[1]),
        "z_des_minus_act": float(mean_error[2]),
        "x_rmse": float(rmse_error[0]),
        "y_rmse": float(rmse_error[1]),
        "z_rmse": float(rmse_error[2]),
    }


def _force_tracking_summary(logger: ExecutionLogger) -> dict[str, float | int]:
    rows = [
        row
        for row in logger.rows
        if row.force_control_active
        and row.desired_normal_force_n is not None
        and row.measured_normal_force_n is not None
    ]
    if not rows:
        return {}
    desired = np.asarray([row.desired_normal_force_n for row in rows], dtype=float)
    measured = np.asarray([row.measured_normal_force_n for row in rows], dtype=float)
    error = desired - measured
    offsets = np.asarray(
        [0.0 if row.normal_force_offset_m is None else row.normal_force_offset_m for row in rows],
        dtype=float,
    )
    contacts = np.asarray([bool(row.contact_active) for row in rows], dtype=bool)
    return {
        "active_samples": int(len(rows)),
        "contact_ratio": float(np.mean(contacts)),
        "desired_mean_n": float(np.mean(desired)),
        "measured_mean_n": float(np.mean(measured)),
        "measured_max_n": float(np.max(measured)),
        "force_rmse_n": float(np.sqrt(np.mean(np.square(error)))),
        "force_mean_error_n": float(np.mean(error)),
        "offset_min_m": float(np.min(offsets)),
        "offset_max_m": float(np.max(offsets)),
    }


if __name__ == "__main__":
    main()
