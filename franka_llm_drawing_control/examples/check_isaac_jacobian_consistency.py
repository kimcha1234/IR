"""Check Isaac Jacobian signs against finite-difference pen-tip motion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from franka_llm_drawing.config import FRANKA_ARM_JOINT_NAMES, load_frames_config, load_isaac_scene_config
from franka_llm_drawing.controllers.pose_error import orientation_error_axis_angle
from franka_llm_drawing.controllers.tool_jacobian import shift_spatial_jacobian_to_tool
from franka_llm_drawing.frames import ee_pose_to_tip_pose, make_transform
from franka_llm_drawing.sim import IsaacFrankaBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default=str(PROJECT_ROOT / "configs" / "frames.yaml"))
    parser.add_argument("--scene", default=str(PROJECT_ROOT / "configs" / "isaac_scene.yaml"))
    parser.add_argument("--eps", type=float, default=1.0e-4)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "jacobian_consistency.json"))
    parser.add_argument("--skip-close", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError:
        print("Run this script through Isaac Lab ./isaaclab.sh -p.", file=sys.stderr)
        raise SystemExit(2)

    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        run_check(args)
    finally:
        if not args.skip_close:
            simulation_app.close()


def run_check(args: argparse.Namespace) -> None:
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.utils.math import subtract_frame_transforms
    from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

    frames = load_frames_config(args.frames)
    scene_cfg = load_isaac_scene_config(args.scene, project_root=PROJECT_ROOT)
    if scene_cfg.joint_names != FRANKA_ARM_JOINT_NAMES:
        raise RuntimeError("Jacobian check must command only panda_joint1 through panda_joint7.")
    if not sim_utils.open_stage(scene_cfg.usd_path):
        raise RuntimeError(f"Could not open USD stage: {scene_cfg.usd_path}")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=scene_cfg.physics_dt, device="cpu"))
    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
    robot_cfg.prim_path = scene_cfg.robot_prim_path
    robot_cfg.spawn = None
    robot = Articulation(cfg=robot_cfg)
    sim.reset()
    robot.reset()

    joint_ids = _resolve_arm_joint_ids(robot, scene_cfg.joint_names)
    body_id = _resolve_body_id(robot, scene_cfg.ee_frame_name)
    ee_jacobi_idx = body_id - 1 if robot.is_fixed_base else body_id
    q0 = np.asarray(scene_cfg.initial_joint_positions_rad, dtype=float)
    backend = IsaacFrankaBackend(
        robot=robot,
        sim=sim,
        joint_ids=joint_ids,
        body_id=body_id,
        ee_jacobi_idx=ee_jacobi_idx,
        torch_module=torch,
        subtract_frame_transforms=subtract_frame_transforms,
    )
    _apply_q(robot, sim, joint_ids, torch, q0)
    state = backend.get_state()
    J_body = backend.get_end_effector_jacobian()
    J_tip = shift_spatial_jacobian_to_tool(J_body, state.ee_rotation, frames.jacobian_tip_translation_m)

    rows = []
    eps = float(args.eps)
    for column, joint_name in enumerate(scene_cfg.joint_names):
        q_minus = q0.copy()
        q_plus = q0.copy()
        q_minus[column] -= eps
        q_plus[column] += eps
        T_minus = _tip_pose_for_q(robot, sim, joint_ids, torch, backend, frames.T_ee_tip, q_minus)
        T_plus = _tip_pose_for_q(robot, sim, joint_ids, torch, backend, frames.T_ee_tip, q_plus)
        fd_linear = (T_plus[:3, 3] - T_minus[:3, 3]) / (2.0 * eps)
        fd_angular = orientation_error_axis_angle(T_plus[:3, :3], T_minus[:3, :3]) / (2.0 * eps)
        fd = np.concatenate([fd_linear, fd_angular])
        jac = J_tip[:, column]
        rows.append(
            {
                "joint": joint_name,
                "linear_cosine": _cosine(fd_linear, jac[:3]),
                "angular_cosine": _cosine(fd_angular, jac[3:]),
                "linear_fd": fd_linear.tolist(),
                "linear_jacobian": jac[:3].tolist(),
                "angular_fd": fd_angular.tolist(),
                "angular_jacobian": jac[3:].tolist(),
                "linear_error_norm": float(np.linalg.norm(fd_linear - jac[:3])),
                "angular_error_norm": float(np.linalg.norm(fd_angular - jac[3:])),
            }
        )

    _apply_q(robot, sim, joint_ids, torch, q0)
    result = {
        "usd_path": scene_cfg.usd_path,
        "joint_names": list(scene_cfg.joint_names),
        "eps": eps,
        "rows": rows,
        "min_linear_cosine": float(min(row["linear_cosine"] for row in rows)),
        "min_angular_cosine": float(min(row["angular_cosine"] for row in rows)),
        "max_linear_error_norm": float(max(row["linear_error_norm"] for row in rows)),
        "max_angular_error_norm": float(max(row["angular_error_norm"] for row in rows)),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def _tip_pose_for_q(robot, sim, joint_ids, torch, backend, T_ee_tip: np.ndarray, q: np.ndarray) -> np.ndarray:
    _apply_q(robot, sim, joint_ids, torch, q)
    state = backend.get_state()
    return ee_pose_to_tip_pose(make_transform(state.ee_rotation, state.ee_position), T_ee_tip)


def _apply_q(robot, sim, joint_ids: list[int], torch, q: np.ndarray) -> None:
    q_tensor = torch.tensor(q, dtype=robot.data.default_joint_pos.dtype, device=robot.device)
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, joint_ids] = q_tensor.unsqueeze(0)
    joint_vel[:, joint_ids] = 0.0
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()
    sim.forward()
    robot.update(0.0)


def _resolve_arm_joint_ids(robot, joint_names: tuple[str, ...]) -> list[int]:
    available = list(robot.joint_names)
    missing = [name for name in joint_names if name not in available]
    if missing:
        raise RuntimeError(f"Missing required Franka arm joints: {missing}. Available joints: {available}")
    return [available.index(name) for name in joint_names]


def _resolve_body_id(robot, body_name: str) -> int:
    available = list(robot.body_names)
    if body_name not in available:
        raise RuntimeError(f"Missing body {body_name!r}. Available bodies: {available}")
    return int(available.index(body_name))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-12)
    return float(np.dot(a, b) / denom)


if __name__ == "__main__":
    main()
