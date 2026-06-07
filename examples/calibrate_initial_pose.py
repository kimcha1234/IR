"""Find a Franka initial posture for the local drawing USD.

The calibration target is the first drawing waypoint with the pen tip hovering
above the paper and the pen axis aligned with the paper normal. The script
uses Isaac's actual articulation FK/Jacobian, including the configured
``T_ee_tip`` tool transform, then prints a 7-joint arm posture that can be
copied into ``configs/isaac_scene.yaml``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from franka_llm_drawing.config import (
    FRANKA_ARM_JOINT_NAMES,
    load_frames_config,
    load_isaac_scene_config,
    load_jade_config,
)
from franka_llm_drawing.controllers.pose_error import axis_alignment_error
from franka_llm_drawing.controllers.tool_jacobian import shift_spatial_jacobian_to_tool
from franka_llm_drawing.frames import ee_pose_to_tip_pose, make_transform, samples_to_cartesian_trajectory
from franka_llm_drawing.jade.jacobian_metrics import FRANKA_JOINT_LOWER, FRANKA_JOINT_UPPER, JointLimits
from franka_llm_drawing.llm_bridge import load_plan_json
from franka_llm_drawing.sim import IsaacFrankaBackend
from franka_llm_drawing.trajectory import sample_drawing_plan
from franka_llm_drawing.trajectory.path_primitives import PoseSample


@dataclass(frozen=True)
class Evaluation:
    q: np.ndarray
    cost: float
    position_error_m: float
    xy_error_m: float
    z_error_m: float
    axis_error_deg: float
    joint_limit_margin_rad: float
    tip_position: np.ndarray
    tip_axis: np.ndarray


@dataclass
class IsaacCalibrationContext:
    robot: object
    sim: object
    backend: IsaacFrankaBackend
    joint_ids: list[int]
    full_joint_pos: object
    full_joint_vel: object
    frames: object
    target_position: np.ndarray
    target_axis: np.ndarray
    posture_target: np.ndarray
    torch: object


def build_parser() -> argparse.ArgumentParser:
    default_seed_log = PROJECT_ROOT / "outputs" / "jade_isaac_axis_posture_orientation_check" / "execution_log.csv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(PROJECT_ROOT / "examples" / "sample_plans" / "circle.json"))
    parser.add_argument("--frames", default=str(PROJECT_ROOT / "configs" / "frames.yaml"))
    parser.add_argument("--jade", default=str(PROJECT_ROOT / "configs" / "jade.yaml"))
    parser.add_argument("--scene", default=str(PROJECT_ROOT / "configs" / "isaac_scene.yaml"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "initial_pose_calibration"))
    parser.add_argument(
        "--clearance-m",
        type=float,
        default=None,
        help="Tip hover height above the paper in board coordinates. Defaults to frames.yaml board.hover_z_m.",
    )
    parser.add_argument("--ik-steps", type=int, default=90)
    parser.add_argument("--random-restarts", type=int, default=12)
    parser.add_argument("--random-noise-rad", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--ik-step-rad", type=float, default=0.08)
    parser.add_argument("--orientation-weight", type=float, default=0.45)
    parser.add_argument("--posture-gain", type=float, default=0.06)
    parser.add_argument("--limit-margin-rad", type=float, default=0.12)
    parser.add_argument("--extra-seed-q", action="append", default=[], help="Comma-separated 7 arm joint values.")
    parser.add_argument(
        "--seed-log",
        default=str(default_seed_log) if default_seed_log.exists() else "",
        help="Optional execution_log.csv used to seed the calibration from previous runs.",
    )
    parser.add_argument("--seed-log-limit", type=int, default=12)
    parser.add_argument(
        "--write-scene-config",
        action="store_true",
        help="Replace backend.initial_joint_positions_rad in --scene with the best result.",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="Keep Isaac Sim open after calibration so the best posture can be inspected.",
    )
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
            "  ./isaaclab.sh -p /home/kimchangyeol/IsaacLab/IR/examples/calibrate_initial_pose.py --headless",
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
        if not args.skip_close and not args.keep_open:
            simulation_app.close()


def run_with_isaac(args: argparse.Namespace, simulation_app) -> None:
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import subtract_frame_transforms
    from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames_config(args.frames)
    jade = load_jade_config(args.jade)
    scene_cfg = load_isaac_scene_config(args.scene, project_root=PROJECT_ROOT)
    if scene_cfg.joint_names != FRANKA_ARM_JOINT_NAMES:
        raise RuntimeError("Calibration must command only panda_joint1 through panda_joint7.")

    target = _build_hover_target(args, frames, jade)
    print(f"[INFO][CALIB] opening USD stage: {scene_cfg.usd_path}", flush=True)
    if not sim_utils.open_stage(scene_cfg.usd_path):
        raise RuntimeError(f"Could not open USD stage: {scene_cfg.usd_path}")

    sim_device = args.device if scene_cfg.use_gpu else "cpu"
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=scene_cfg.physics_dt, device=sim_device))
    sim.set_camera_view([1.2, -1.2, 0.85], [0.5, 0.0, 0.25])

    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
    robot_cfg.prim_path = scene_cfg.robot_prim_path
    robot_cfg.spawn = None
    robot = Articulation(cfg=robot_cfg)

    sim.reset()
    robot.reset()
    joint_ids = _resolve_arm_joint_ids(robot, scene_cfg.joint_names)
    body_id = _resolve_body_id(robot, scene_cfg.ee_frame_name)
    ee_jacobi_idx = body_id - 1 if robot.is_fixed_base else body_id
    backend = IsaacFrankaBackend(
        robot=robot,
        sim=sim,
        joint_ids=joint_ids,
        body_id=body_id,
        ee_jacobi_idx=ee_jacobi_idx,
        torch_module=torch,
        subtract_frame_transforms=subtract_frame_transforms,
    )
    posture_target = _posture_target(scene_cfg, jade)
    context = IsaacCalibrationContext(
        robot=robot,
        sim=sim,
        backend=backend,
        joint_ids=joint_ids,
        full_joint_pos=robot.data.default_joint_pos.clone(),
        full_joint_vel=robot.data.default_joint_vel.clone(),
        frames=frames,
        target_position=target.p_base_tip,
        target_axis=target.R_base_tip[:, 2],
        posture_target=posture_target,
        torch=torch,
    )

    seed_qs = _collect_seed_qs(args, scene_cfg, jade, robot, joint_ids, target)
    print(f"[INFO][CALIB] target tip position: {_format_vector(context.target_position)}", flush=True)
    print(f"[INFO][CALIB] target tip axis: {_format_vector(context.target_axis)}", flush=True)
    print(f"[INFO][CALIB] seed count: {len(seed_qs)}", flush=True)
    results = _run_calibration(args, context, seed_qs)
    best = min(results, key=lambda item: item.cost)

    _apply_q(context, best.q)
    _write_outputs(out_dir, best, results, args)
    if args.write_scene_config:
        _replace_initial_joint_positions(Path(args.scene), best.q)
        print(f"[INFO][CALIB] updated scene config: {args.scene}", flush=True)

    print("[INFO][CALIB] best initial_joint_positions_rad:", _format_yaml_list(best.q), flush=True)
    print(
        "[INFO][CALIB] errors: "
        f"pos={best.position_error_m:.5f} m, xy={best.xy_error_m:.5f} m, "
        f"z={best.z_error_m:.5f} m, axis={best.axis_error_deg:.2f} deg, "
        f"margin={best.joint_limit_margin_rad:.3f} rad",
        flush=True,
    )
    print(f"[INFO][CALIB] outputs: {out_dir}", flush=True)

    while args.keep_open and simulation_app.is_running():
        sim.step()


def _build_hover_target(args: argparse.Namespace, frames, jade):
    plan = load_plan_json(args.plan)
    clearance_m = frames.hover_height_m if args.clearance_m is None else float(args.clearance_m)
    samples = sample_drawing_plan(
        plan,
        dt=jade.sampling.dt,
        default_speed_m_s=jade.sampling.default_speed_m_s,
        hover_height_m=clearance_m,
        draw_height_m=clearance_m,
        default_normal_force_n=None,
        time_scaling=jade.sampling.time_scaling,
        max_linear_speed_m_s=jade.sampling.max_linear_speed_m_s,
        max_linear_accel_m_s2=jade.sampling.max_linear_accel_m_s2,
        max_linear_jerk_m_s3=jade.sampling.max_linear_jerk_m_s3,
        contact_force_ramp_duration_s=jade.sampling.contact_force_ramp_duration_s,
        pen_down_settle_duration_s=0.0,
        corner_dwell_duration_s=0.0,
    )
    samples = _force_hover_samples(samples, clearance_m)
    cartesian = samples_to_cartesian_trajectory(
        samples,
        frames.T_base_board,
        frames.T_ee_tip,
        R_base_tip_override=frames.R_base_tip,
        jacobian_tip_translation_m=frames.jacobian_tip_translation_m,
    )
    if not cartesian:
        raise RuntimeError("No drawing target was generated from the plan.")
    return cartesian[0]


def _run_calibration(
    args: argparse.Namespace,
    context: IsaacCalibrationContext,
    seed_qs: list[np.ndarray],
) -> list[Evaluation]:
    rng = np.random.default_rng(args.seed)
    starts = list(seed_qs)
    base_seeds = seed_qs if seed_qs else [context.posture_target]
    for _ in range(max(0, args.random_restarts)):
        base = base_seeds[int(rng.integers(0, len(base_seeds)))]
        noise = rng.normal(0.0, args.random_noise_rad, size=7)
        starts.append(_clip_to_limits(base + noise, args.limit_margin_rad))

    results: list[Evaluation] = []
    for index, seed_q in enumerate(starts, start=1):
        print(f"[INFO][CALIB] refining seed {index}/{len(starts)}", flush=True)
        results.append(_refine_seed(args, context, seed_q))
    return results


def _refine_seed(
    args: argparse.Namespace,
    context: IsaacCalibrationContext,
    seed_q: np.ndarray,
) -> Evaluation:
    q = _clip_to_limits(seed_q, args.limit_margin_rad)
    best = _evaluate_q(context, q)
    for _ in range(max(0, args.ik_steps)):
        evaluation = _evaluate_q(context, q)
        if evaluation.cost < best.cost:
            best = evaluation
        if (
            evaluation.position_error_m < 0.003
            and abs(evaluation.z_error_m) < 0.002
            and evaluation.axis_error_deg < 5.0
        ):
            break

        state = context.backend.get_state()
        jacobian_body = context.backend.get_end_effector_jacobian()
        jacobian_tip = shift_spatial_jacobian_to_tool(
            jacobian_body,
            state.ee_rotation,
            context.frames.jacobian_tip_translation_m,
        )
        T_base_tip = ee_pose_to_tip_pose(
            make_transform(state.ee_rotation, state.ee_position),
            context.frames.T_ee_tip,
        )
        pos_error = context.target_position - T_base_tip[:3, 3]
        ori_error = axis_alignment_error(context.target_axis, T_base_tip[:3, :3][:, 2])
        weighted_j = jacobian_tip.copy()
        weighted_j[3:, :] *= float(args.orientation_weight)
        error = np.concatenate([pos_error, float(args.orientation_weight) * ori_error])
        delta_q = _damped_least_squares(weighted_j, error, float(args.damping))
        if args.posture_gain > 0.0:
            pinv = _damped_pseudo_inverse(weighted_j, float(args.damping))
            null_projector = np.eye(7) - pinv @ weighted_j
            delta_q += null_projector @ (
                float(args.posture_gain) * (context.posture_target - state.q)
            )
        delta_q = _limit_norm(delta_q, float(args.ik_step_rad))
        candidate_q = _clip_to_limits(state.q + delta_q, args.limit_margin_rad)
        candidate_eval = _evaluate_q(context, candidate_q)
        if candidate_eval.cost <= evaluation.cost * 1.02:
            q = candidate_q
        else:
            q = _clip_to_limits(state.q + 0.5 * delta_q, args.limit_margin_rad)
    final_eval = _evaluate_q(context, q)
    return min([best, final_eval], key=lambda item: item.cost)


def _evaluate_q(context: IsaacCalibrationContext, q: np.ndarray) -> Evaluation:
    _apply_q(context, q)
    state = context.backend.get_state()
    T_base_tip = ee_pose_to_tip_pose(
        make_transform(state.ee_rotation, state.ee_position),
        context.frames.T_ee_tip,
    )
    tip_position = T_base_tip[:3, 3].copy()
    tip_axis = T_base_tip[:3, :3][:, 2].copy()
    pos_error_vec = context.target_position - tip_position
    xy_error = float(np.linalg.norm(pos_error_vec[:2]))
    z_error = float(pos_error_vec[2])
    pos_error = float(np.linalg.norm(pos_error_vec))
    axis_error = _axis_alignment_error_deg(context.target_axis, tip_axis)
    margin = JointLimits().margin(q)
    low_margin_penalty = max(0.0, 0.12 - margin)
    cost = (
        4.0 * xy_error
        + 6.0 * abs(z_error)
        + 0.35 * math.radians(axis_error)
        + 8.0 * low_margin_penalty * low_margin_penalty
        + 0.01 * float(np.linalg.norm(q - context.posture_target))
    )
    return Evaluation(
        q=q.copy(),
        cost=float(cost),
        position_error_m=pos_error,
        xy_error_m=xy_error,
        z_error_m=z_error,
        axis_error_deg=axis_error,
        joint_limit_margin_rad=float(margin),
        tip_position=tip_position,
        tip_axis=tip_axis,
    )


def _apply_q(context: IsaacCalibrationContext, q: np.ndarray) -> None:
    q_tensor = context.torch.tensor(q, dtype=context.full_joint_pos.dtype, device=context.full_joint_pos.device)
    joint_pos = context.full_joint_pos.clone()
    joint_vel = context.full_joint_vel.clone()
    joint_pos[:, context.joint_ids] = q_tensor.unsqueeze(0)
    joint_vel[:, context.joint_ids] = 0.0
    context.robot.write_joint_state_to_sim(joint_pos, joint_vel)
    context.robot.set_joint_position_target(joint_pos)
    context.robot.write_data_to_sim()
    context.sim.forward()
    context.robot.update(0.0)


def _damped_least_squares(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    J = np.asarray(jacobian, dtype=float)
    e = np.asarray(error, dtype=float)
    return J.T @ np.linalg.solve(J @ J.T + damping * damping * np.eye(J.shape[0]), e)


def _damped_pseudo_inverse(jacobian: np.ndarray, damping: float) -> np.ndarray:
    J = np.asarray(jacobian, dtype=float)
    return J.T @ np.linalg.inv(J @ J.T + damping * damping * np.eye(J.shape[0]))


def _collect_seed_qs(args, scene_cfg, jade, robot, joint_ids: list[int], target) -> list[np.ndarray]:
    seeds: list[np.ndarray] = []
    if scene_cfg.initial_joint_positions_rad is not None:
        seeds.append(np.asarray(scene_cfg.initial_joint_positions_rad, dtype=float))
    if jade.executor.posture_target_rad is not None:
        seeds.append(np.asarray(jade.executor.posture_target_rad, dtype=float))
    seeds.append(robot.data.default_joint_pos[0, joint_ids].detach().cpu().numpy().astype(float))
    for raw in args.extra_seed_q:
        seeds.append(_parse_q(raw))
    seed_log = Path(args.seed_log) if args.seed_log else None
    if seed_log is not None and seed_log.exists():
        seeds.extend(_load_seed_log(seed_log, target.p_base_tip, max(0, int(args.seed_log_limit))))
    return _dedupe_seed_qs(seeds)


def _load_seed_log(path: Path, target_position: np.ndarray, limit: int) -> list[np.ndarray]:
    if limit <= 0:
        return []
    rows: list[tuple[float, np.ndarray]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                q = np.asarray([float(row[f"q{i}"]) for i in range(1, 8)], dtype=float)
                actual = np.asarray([float(row["x_act"]), float(row["y_act"]), float(row["z_act"])], dtype=float)
                orientation = float(row.get("tip_orientation_error_deg") or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            score = float(np.linalg.norm(actual - target_position)) + 0.01 * math.radians(orientation)
            rows.append((score, q))
    rows.sort(key=lambda item: item[0])
    return [q for _, q in rows[:limit]]


def _posture_target(scene_cfg, jade) -> np.ndarray:
    if scene_cfg.initial_joint_positions_rad is not None:
        return np.asarray(scene_cfg.initial_joint_positions_rad, dtype=float)
    if jade.executor.posture_target_rad is not None:
        return np.asarray(jade.executor.posture_target_rad, dtype=float)
    return 0.5 * (FRANKA_JOINT_LOWER + FRANKA_JOINT_UPPER)


def _force_hover_samples(samples: list[PoseSample], hover_height_m: float) -> list[PoseSample]:
    output: list[PoseSample] = []
    for sample in samples:
        output.append(
            PoseSample(
                t=sample.t,
                position=np.asarray([sample.position[0], sample.position[1], hover_height_m], dtype=float),
                rotation=sample.rotation,
                pen_contact_desired=False,
                desired_normal_force_n=None,
                source_action=sample.source_action,
                stroke_id=sample.stroke_id,
            )
        )
    return output


def _write_outputs(out_dir: Path, best: Evaluation, results: list[Evaluation], args: argparse.Namespace) -> None:
    sorted_results = sorted(results, key=lambda item: item.cost)
    with (out_dir / "initial_pose_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "cost",
                "position_error_m",
                "xy_error_m",
                "z_error_m",
                "axis_error_deg",
                "joint_limit_margin_rad",
                "x_tip",
                "y_tip",
                "z_tip",
                "axis_x",
                "axis_y",
                "axis_z",
                "q1",
                "q2",
                "q3",
                "q4",
                "q5",
                "q6",
                "q7",
            ]
        )
        for rank, item in enumerate(sorted_results, start=1):
            writer.writerow(
                [
                    rank,
                    item.cost,
                    item.position_error_m,
                    item.xy_error_m,
                    item.z_error_m,
                    item.axis_error_deg,
                    item.joint_limit_margin_rad,
                    *item.tip_position.tolist(),
                    *item.tip_axis.tolist(),
                    *item.q.tolist(),
                ]
            )
    summary = {
        "initial_joint_positions_rad": [float(value) for value in best.q],
        "cost": best.cost,
        "position_error_m": best.position_error_m,
        "xy_error_m": best.xy_error_m,
        "z_error_m": best.z_error_m,
        "axis_error_deg": best.axis_error_deg,
        "joint_limit_margin_rad": best.joint_limit_margin_rad,
        "tip_position": [float(value) for value in best.tip_position],
        "tip_axis": [float(value) for value in best.tip_axis],
        "args": {
            "clearance_m": args.clearance_m,
            "ik_steps": args.ik_steps,
            "random_restarts": args.random_restarts,
            "random_noise_rad": args.random_noise_rad,
            "damping": args.damping,
            "ik_step_rad": args.ik_step_rad,
            "orientation_weight": args.orientation_weight,
            "posture_gain": args.posture_gain,
            "limit_margin_rad": args.limit_margin_rad,
        },
    }
    (out_dir / "best_initial_pose.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _replace_initial_joint_positions(scene_path: Path, q: np.ndarray) -> None:
    text = scene_path.read_text(encoding="utf-8")
    replacement = f"initial_joint_positions_rad: {_format_yaml_list(q)}"
    pattern = r"initial_joint_positions_rad:\s*\[[^\]]*\]"
    new_text, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not find initial_joint_positions_rad in {scene_path}.")
    scene_path.write_text(new_text, encoding="utf-8")


def _resolve_arm_joint_ids(robot, joint_names: tuple[str, ...]) -> list[int]:
    available = list(robot.joint_names)
    missing = [name for name in joint_names if name not in available]
    if missing:
        raise RuntimeError(f"Missing required Franka arm joints: {missing}. Available joints: {available}")
    joint_ids = [available.index(name) for name in joint_names]
    if len(joint_ids) != 7:
        raise RuntimeError(f"Expected 7 Franka arm joint ids, got {joint_ids}.")
    return joint_ids


def _resolve_body_id(robot, body_name: str) -> int:
    available = list(robot.body_names)
    if body_name not in available:
        raise RuntimeError(f"Missing end-effector body {body_name!r}. Available bodies: {available}")
    return int(available.index(body_name))


def _axis_alignment_error_deg(desired_axis: np.ndarray, actual_axis: np.ndarray) -> float:
    desired = np.asarray(desired_axis, dtype=float)
    actual = np.asarray(actual_axis, dtype=float)
    cosine = float(
        np.clip(
            np.dot(desired, actual) / max(float(np.linalg.norm(desired) * np.linalg.norm(actual)), 1e-12),
            -1.0,
            1.0,
        )
    )
    return float(np.degrees(np.arccos(cosine)))


def _parse_q(raw: str) -> np.ndarray:
    values = [float(item) for item in re.split(r"[\s,]+", raw.strip().strip("[]")) if item]
    if len(values) != 7:
        raise ValueError(f"Expected 7 joint values, got {len(values)} from {raw!r}.")
    return np.asarray(values, dtype=float)


def _dedupe_seed_qs(seeds: list[np.ndarray]) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for seed in seeds:
        q = np.asarray(seed, dtype=float)
        if q.shape != (7,):
            continue
        if not any(np.linalg.norm(q - existing) < 1e-6 for existing in output):
            output.append(q)
    return output


def _clip_to_limits(q: np.ndarray, margin: float) -> np.ndarray:
    lower = FRANKA_JOINT_LOWER + float(margin)
    upper = FRANKA_JOINT_UPPER - float(margin)
    return np.clip(np.asarray(q, dtype=float), lower, upper)


def _limit_norm(value: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= max_norm:
        return value
    return value * (max_norm / max(norm, 1e-12))


def _format_yaml_list(q: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in q) + "]"


def _format_vector(value: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(item):.5f}" for item in value) + "]"


if __name__ == "__main__":
    main()
