"""Small YAML-subset config loader used by the offline JADE scripts.

PyYAML is intentionally optional so the core package can run in a clean Isaac
Lab Python environment with only NumPy installed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from franka_llm_drawing.frames import make_transform

FRANKA_ARM_JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))


@dataclass(frozen=True)
class FramesConfig:
    T_base_board: np.ndarray
    T_ee_tip: np.ndarray
    jacobian_tip_translation_m: np.ndarray
    R_base_tip: np.ndarray
    board_normal_base: np.ndarray
    hover_height_m: float = 0.02
    draw_height_m: float = 0.0


@dataclass(frozen=True)
class SamplingConfig:
    dt: float = 0.02
    max_segment_length_m: float = 0.005
    time_scaling: str = "quintic"
    default_speed_m_s: float = 0.03
    hover_speed_m_s: float = 0.08
    draw_speed_m_s: float = 0.025
    max_linear_speed_m_s: float | None = 0.025
    max_linear_accel_m_s2: float | None = 0.08
    max_linear_jerk_m_s3: float | None = 0.40
    contact_force_ramp_duration_s: float = 0.0
    pen_down_settle_duration_s: float = 0.0
    corner_dwell_duration_s: float = 0.0


@dataclass(frozen=True)
class ThresholdConfig:
    sigma_min_warning: float = 0.03
    sigma_min_fail: float = 0.01
    condition_warning: float = 100.0
    condition_fail: float = 300.0
    manipulability_warning: float = 0.02
    joint_margin_warning_rad: float = 0.15
    joint_margin_fail_rad: float = 0.05
    tip_orientation_warning_deg: float = 3.0
    tip_orientation_fail_deg: float = 8.0


@dataclass(frozen=True)
class AdaptivePolicyConfig:
    lambda_min: float = 0.02
    lambda_max: float = 0.20
    speed_scale_min: float = 0.20
    speed_scale_warning: float = 0.50


@dataclass(frozen=True)
class ExecutorConfig:
    max_qdot_rad_s: float = 5.0
    max_delta_q_rad: float = 0.12
    tracking_mode: str = "differential"
    iterative_servo_iterations: int = 1
    iterative_servo_drawing_only: bool = True
    command_integration_enabled: bool = False
    max_command_tracking_error_rad: float = 0.35
    startup_lift_height_m: float = 0.0
    startup_lift_steps: int = 0
    startup_lateral_steps: int = 0
    startup_descent_steps: int = 0
    lookahead_time_s: float = 0.0
    lookahead_drawing_only: bool = True
    lookahead_same_stroke_only: bool = True
    tangent_integral_enabled: bool = True
    tangent_integral_gain: float = 0.20
    tangent_integral_leak_per_s: float = 0.20
    max_tangent_integral_offset_m: float = 0.008
    position_gain: float = 4.0
    position_axis_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    orientation_gain: float = 4.0
    position_task_weight: float = 1.0
    orientation_task_weight: float = 12.0
    feedforward_gain: float = 0.0
    command_position_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_mode: str = "axis"
    posture_gain: float = 0.0
    posture_target_rad: tuple[float, ...] | None = None
    posture_weights: tuple[float, ...] | None = None
    locked_joint_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class ForceControlConfig:
    enabled: bool = True
    phase_gating_enabled: bool = False
    unwanted_contact_release_enabled: bool = True
    desired_normal_force_n: float = 1.0
    contact_threshold_n: float = 0.05
    kp_offset_m_per_n: float = 0.0015
    ki_offset_m_per_n_s: float = 0.00035
    kd_offset_m_s_per_n: float = 0.0
    max_press_offset_m: float = 0.004
    max_lift_offset_m: float = 0.025
    max_offset_step_m: float = 0.0015
    max_drawing_press_offset_m: float = 0.004
    max_drawing_lift_offset_m: float = 0.008
    max_drawing_offset_step_m: float = 0.0005
    max_release_lift_offset_m: float = 0.025
    max_release_offset_step_m: float = 0.0015
    max_unwanted_contact_lift_m: float = 0.025
    max_unwanted_contact_offset_step_m: float = 0.0015
    force_filter_alpha: float = 0.35
    contact_joint_stiffness: float = 400.0
    contact_joint_damping: float = 80.0


@dataclass(frozen=True)
class JadeConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    adaptive_policy: AdaptivePolicyConfig = field(default_factory=AdaptivePolicyConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    force_control: ForceControlConfig = field(default_factory=ForceControlConfig)


@dataclass(frozen=True)
class IsaacSceneConfig:
    usd_path: str
    robot_prim_path: str = "/World/Franka"
    board_prim_path: str = "/World/Paper"
    board_frame_prim_path: str = "/World/BoardFrame"
    ee_frame_name: str = "panda_link7"
    pen_tip_prim_path: str = "/World/Franka/panda_link7/panda_link8/panda_hand/PenTool/PenTipFrame"
    contact_sensor_prim_path: str = "/World/Franka/panda_link7"
    contact_filter_prim_paths: tuple[str, ...] = ("/World/Paper",)
    physics_dt: float = 0.005
    control_dt: float = 0.02
    use_gpu: bool = False
    command_mode: str = "joint_position"
    joint_names: tuple[str, ...] = FRANKA_ARM_JOINT_NAMES
    initial_joint_positions_rad: tuple[float, ...] | None = None


def load_frames_config(path: str | Path) -> FramesConfig:
    data = load_yaml_subset(path)
    T_base_board_data = data.get("T_base_board", {})
    T_ee_tip_data = data.get("T_ee_tip", {})
    board_data = data.get("board", {})
    return FramesConfig(
        T_base_board=make_transform(
            np.asarray(T_base_board_data.get("rotation", np.eye(3)), dtype=float),
            np.asarray(T_base_board_data.get("translation_m", [0.5, 0.0, 0.202]), dtype=float),
        ),
        T_ee_tip=make_transform(
            np.asarray(T_ee_tip_data.get("rotation", np.eye(3)), dtype=float),
            np.asarray(T_ee_tip_data.get("translation_m", [0.0, 0.0, 0.207000001]), dtype=float),
        ),
        jacobian_tip_translation_m=np.asarray(
            T_ee_tip_data.get(
                "jacobian_translation_m",
                T_ee_tip_data.get("translation_m", [0.0, 0.0, 0.207000001]),
            ),
            dtype=float,
        ),
        R_base_tip=np.asarray(
            data.get(
                "R_base_tip",
                [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            ),
            dtype=float,
        ),
        board_normal_base=np.asarray(data.get("board_normal_base", [0.0, 0.0, 1.0]), dtype=float),
        hover_height_m=float(board_data.get("hover_z_m", data.get("hover_height_m", 0.02))),
        draw_height_m=float(board_data.get("drawing_z_m", data.get("draw_height_m", 0.0))),
    )


def load_jade_config(path: str | Path) -> JadeConfig:
    data = load_yaml_subset(path)
    jade = data.get("jade", data)
    sampling = jade.get("sampling", {})
    thresholds = jade.get("thresholds", {})
    adaptive = jade.get("adaptive_policy", {})
    executor = jade.get("executor", {})
    force_control = jade.get("force_control", {})
    return JadeConfig(
        sampling=SamplingConfig(
            dt=float(sampling.get("dt", SamplingConfig.dt)),
            max_segment_length_m=float(
                sampling.get("max_segment_length_m", SamplingConfig.max_segment_length_m)
            ),
            time_scaling=str(sampling.get("time_scaling", SamplingConfig.time_scaling)),
            default_speed_m_s=float(
                sampling.get("default_speed_m_s", SamplingConfig.default_speed_m_s)
            ),
            hover_speed_m_s=float(sampling.get("hover_speed_m_s", SamplingConfig.hover_speed_m_s)),
            draw_speed_m_s=float(sampling.get("draw_speed_m_s", SamplingConfig.draw_speed_m_s)),
            max_linear_speed_m_s=_optional_float_value(
                sampling.get("max_linear_speed_m_s", SamplingConfig.max_linear_speed_m_s)
            ),
            max_linear_accel_m_s2=_optional_float_value(
                sampling.get("max_linear_accel_m_s2", SamplingConfig.max_linear_accel_m_s2)
            ),
            max_linear_jerk_m_s3=_optional_float_value(
                sampling.get("max_linear_jerk_m_s3", SamplingConfig.max_linear_jerk_m_s3)
            ),
            contact_force_ramp_duration_s=float(
                sampling.get("contact_force_ramp_duration_s", SamplingConfig.contact_force_ramp_duration_s)
            ),
            pen_down_settle_duration_s=float(
                sampling.get("pen_down_settle_duration_s", SamplingConfig.pen_down_settle_duration_s)
            ),
            corner_dwell_duration_s=float(
                sampling.get("corner_dwell_duration_s", SamplingConfig.corner_dwell_duration_s)
            ),
        ),
        thresholds=ThresholdConfig(**_merge_dataclass_defaults(ThresholdConfig, thresholds)),
        adaptive_policy=AdaptivePolicyConfig(
            **_merge_dataclass_defaults(AdaptivePolicyConfig, adaptive)
        ),
        executor=_load_executor_config(executor),
        force_control=ForceControlConfig(**_merge_dataclass_defaults(ForceControlConfig, force_control)),
    )


def load_isaac_scene_config(path: str | Path, project_root: str | Path | None = None) -> IsaacSceneConfig:
    data = load_yaml_subset(path)
    isaac = data.get("isaac", {})
    backend = data.get("backend", {})
    usd_path = str(isaac.get("usd_path", "usd/franka_drawing_scene_clean.usda"))
    if project_root is not None and not Path(usd_path).is_absolute():
        usd_path = str(Path(project_root) / usd_path)
    joint_names = tuple(str(name) for name in backend.get("joint_names", FRANKA_ARM_JOINT_NAMES))
    if joint_names != FRANKA_ARM_JOINT_NAMES:
        raise ValueError(
            "Isaac backend must command only panda_joint1 through panda_joint7; "
            f"got {joint_names!r}."
        )
    return IsaacSceneConfig(
        usd_path=usd_path,
        robot_prim_path=str(isaac.get("robot_prim_path", IsaacSceneConfig.robot_prim_path)),
        board_prim_path=str(isaac.get("board_prim_path", IsaacSceneConfig.board_prim_path)),
        board_frame_prim_path=str(
            isaac.get("board_frame_prim_path", IsaacSceneConfig.board_frame_prim_path)
        ),
        ee_frame_name=str(backend.get("ee_frame_name", IsaacSceneConfig.ee_frame_name)),
        pen_tip_prim_path=str(
            isaac.get("pen_tip_prim_path", IsaacSceneConfig.pen_tip_prim_path)
        ),
        contact_sensor_prim_path=str(
            isaac.get("contact_sensor_prim_path", IsaacSceneConfig.contact_sensor_prim_path)
        ),
        contact_filter_prim_paths=tuple(
            str(path) for path in isaac.get("contact_filter_prim_paths", IsaacSceneConfig.contact_filter_prim_paths)
        ),
        physics_dt=float(isaac.get("physics_dt", IsaacSceneConfig.physics_dt)),
        control_dt=float(isaac.get("control_dt", IsaacSceneConfig.control_dt)),
        use_gpu=bool(isaac.get("use_gpu", IsaacSceneConfig.use_gpu)),
        command_mode=str(backend.get("command_mode", IsaacSceneConfig.command_mode)),
        joint_names=joint_names,
        initial_joint_positions_rad=_optional_joint_tuple(
            backend.get("initial_joint_positions_rad", None)
        ),
    )


def load_yaml_subset(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return _parse_yaml_subset(path_obj.read_text(encoding="utf-8"))
    loaded = yaml.safe_load(path_obj.read_text(encoding="utf-8"))
    return {} if loaded is None else dict(loaded)


def _merge_dataclass_defaults(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    defaults = cls()
    return {key: type(getattr(defaults, key))(values.get(key, getattr(defaults, key))) for key in defaults.__dict__}


def _load_executor_config(values: dict[str, Any]) -> ExecutorConfig:
    defaults = ExecutorConfig()
    posture_target = values.get("posture_target_rad", defaults.posture_target_rad)
    if posture_target is not None:
        posture_target = tuple(float(item) for item in posture_target)
        if len(posture_target) != 7:
            raise ValueError("executor.posture_target_rad must contain exactly 7 values.")
    posture_weights = values.get("posture_weights", defaults.posture_weights)
    if posture_weights is not None:
        posture_weights = tuple(float(item) for item in posture_weights)
        if len(posture_weights) != 7:
            raise ValueError("executor.posture_weights must contain exactly 7 values.")
    position_axis_weights = tuple(
        float(item) for item in values.get("position_axis_weights", defaults.position_axis_weights)
    )
    if len(position_axis_weights) != 3:
        raise ValueError("executor.position_axis_weights must contain exactly 3 values.")
    command_position_offset_m = tuple(
        float(item) for item in values.get("command_position_offset_m", defaults.command_position_offset_m)
    )
    if len(command_position_offset_m) != 3:
        raise ValueError("executor.command_position_offset_m must contain exactly 3 values.")
    locked_joint_indices = tuple(int(item) for item in values.get("locked_joint_indices", defaults.locked_joint_indices))
    for index in locked_joint_indices:
        if index < 0 or index >= 7:
            raise ValueError("executor.locked_joint_indices must contain zero-based indices in [0, 6].")
    return ExecutorConfig(
        max_qdot_rad_s=float(values.get("max_qdot_rad_s", defaults.max_qdot_rad_s)),
        max_delta_q_rad=float(values.get("max_delta_q_rad", defaults.max_delta_q_rad)),
        tracking_mode=str(values.get("tracking_mode", defaults.tracking_mode)),
        iterative_servo_iterations=int(
            values.get("iterative_servo_iterations", defaults.iterative_servo_iterations)
        ),
        iterative_servo_drawing_only=bool(
            values.get("iterative_servo_drawing_only", defaults.iterative_servo_drawing_only)
        ),
        command_integration_enabled=bool(
            values.get("command_integration_enabled", defaults.command_integration_enabled)
        ),
        max_command_tracking_error_rad=float(
            values.get("max_command_tracking_error_rad", defaults.max_command_tracking_error_rad)
        ),
        startup_lift_height_m=float(values.get("startup_lift_height_m", defaults.startup_lift_height_m)),
        startup_lift_steps=int(values.get("startup_lift_steps", defaults.startup_lift_steps)),
        startup_lateral_steps=int(values.get("startup_lateral_steps", defaults.startup_lateral_steps)),
        startup_descent_steps=int(values.get("startup_descent_steps", defaults.startup_descent_steps)),
        lookahead_time_s=float(values.get("lookahead_time_s", defaults.lookahead_time_s)),
        lookahead_drawing_only=bool(values.get("lookahead_drawing_only", defaults.lookahead_drawing_only)),
        lookahead_same_stroke_only=bool(
            values.get("lookahead_same_stroke_only", defaults.lookahead_same_stroke_only)
        ),
        tangent_integral_enabled=bool(values.get("tangent_integral_enabled", defaults.tangent_integral_enabled)),
        tangent_integral_gain=float(values.get("tangent_integral_gain", defaults.tangent_integral_gain)),
        tangent_integral_leak_per_s=float(
            values.get("tangent_integral_leak_per_s", defaults.tangent_integral_leak_per_s)
        ),
        max_tangent_integral_offset_m=float(
            values.get("max_tangent_integral_offset_m", defaults.max_tangent_integral_offset_m)
        ),
        position_gain=float(values.get("position_gain", defaults.position_gain)),
        position_axis_weights=position_axis_weights,
        orientation_gain=float(values.get("orientation_gain", defaults.orientation_gain)),
        position_task_weight=float(values.get("position_task_weight", defaults.position_task_weight)),
        orientation_task_weight=float(values.get("orientation_task_weight", defaults.orientation_task_weight)),
        feedforward_gain=float(values.get("feedforward_gain", defaults.feedforward_gain)),
        command_position_offset_m=command_position_offset_m,
        orientation_mode=str(values.get("orientation_mode", defaults.orientation_mode)),
        posture_gain=float(values.get("posture_gain", defaults.posture_gain)),
        posture_target_rad=posture_target,
        posture_weights=posture_weights,
        locked_joint_indices=locked_joint_indices,
    )


def _optional_joint_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    out = tuple(float(item) for item in value)
    if len(out) != 7:
        raise ValueError("initial_joint_positions_rad must contain exactly 7 values.")
    return out


def _optional_float_value(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _parse_yaml_subset(text: str) -> dict[str, Any]:
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append((len(raw) - len(raw.lstrip(" ")), stripped))
    if not lines:
        return {}
    value, index = _parse_block(lines, 0, lines[0][0])
    if index != len(lines):
        raise ValueError("Could not parse complete YAML config.")
    if not isinstance(value, dict):
        raise ValueError("Top-level YAML value must be a mapping.")
    return value


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if lines[index][1].startswith("- "):
        return _parse_list(lines, index, indent)
    return _parse_dict(lines, index, indent)


def _parse_dict(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[dict[str, Any], int]:
    out: dict[str, Any] = {}
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"Unexpected indentation before {content!r}.")
        if content.startswith("- "):
            break
        if ":" not in content:
            raise ValueError(f"Expected key: value line, got {content!r}.")
        key, raw_value = content.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        index += 1
        if raw_value:
            out[key] = _parse_scalar(raw_value)
        else:
            if index >= len(lines) or lines[index][0] <= line_indent:
                out[key] = {}
            else:
                out[key], index = _parse_block(lines, index, lines[index][0])
    return out, index


def _parse_list(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[list[Any], int]:
    out: list[Any] = []
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent != indent or not content.startswith("- "):
            break
        out.append(_parse_scalar(content[2:].strip()))
        index += 1
    return out, index


def _parse_scalar(raw: str) -> Any:
    if raw in {"true", "True"}:
        return True
    if raw in {"false", "False"}:
        return False
    if raw in {"null", "None", "~"}:
        return None
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw.strip("\"'")
