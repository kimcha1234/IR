"""Convert LLM DrawingPlan actions into board-frame pose samples."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from franka_llm_drawing.llm_bridge.plan_loader import drawing_plan_from_obj
from franka_llm_drawing.llm_bridge.plan_schema import DrawingPlan, PrimitiveAction
from franka_llm_drawing.llm_bridge.primitive_parser import (
    get_float,
    get_point,
    get_str,
    optional_point,
)
from franka_llm_drawing.trajectory.path_primitives import PoseSample, sample_arc, sample_line
from franka_llm_drawing.trajectory.retiming import TrajectoryLimits


def sample_drawing_plan(
    plan: DrawingPlan | Mapping[str, Any],
    dt: float,
    default_speed_m_s: float,
    hover_height_m: float,
    draw_height_m: float,
    default_normal_force_n: float | None = None,
    time_scaling: str = "cubic",
    max_linear_speed_m_s: float | None = None,
    max_linear_accel_m_s2: float | None = None,
    max_linear_jerk_m_s3: float | None = None,
    contact_force_ramp_duration_s: float = 0.0,
    pen_down_settle_duration_s: float = 0.0,
) -> list[PoseSample]:
    """Sample a DrawingPlan into board-frame desired pen-tip poses.

    Valid action order is enforced. The first motion must establish a current
    position with ``move_to_start`` before pen contact or drawing actions.
    """

    drawing_plan = drawing_plan_from_obj(plan)
    if not drawing_plan.actions:
        raise ValueError("DrawingPlan contains no actions.")

    samples: list[PoseSample] = []
    current_position: np.ndarray | None = None
    pen_state = "up"
    rotation = np.eye(3)
    limits = TrajectoryLimits(
        max_linear_speed_m_s=max_linear_speed_m_s,
        max_linear_accel_m_s2=max_linear_accel_m_s2,
        max_linear_jerk_m_s3=max_linear_jerk_m_s3,
    )

    for action in drawing_plan.actions:
        if action.frame != "board":
            raise ValueError(f"Unsupported action frame {action.frame!r}; expected 'board'.")

        start_time = samples[-1].t if samples else 0.0
        new_samples: list[PoseSample]

        if action.name == "move_to_start":
            _require_pen_state(pen_state, "up", action)
            target = get_point(action.params, "target", default_z=hover_height_m)
            target[2] = float(action.params.get("hover_height_m", target[2]))
            if current_position is None:
                new_samples = [
                    PoseSample(
                        t=start_time,
                        position=target,
                        rotation=rotation.copy(),
                        linear_velocity=np.zeros(3),
                        linear_acceleration=np.zeros(3),
                        pen_contact_desired=False,
                        source_action=action.name,
                        stroke_id=action.stroke_id,
                    )
                ]
            else:
                new_samples = sample_line(
                    current_position,
                    target,
                    get_float(action.params, "speed_m_s", default_speed_m_s),
                    dt,
                    start_time=start_time,
                    rotation=rotation,
                    pen_contact_desired=False,
                    source_action=action.name,
                    stroke_id=action.stroke_id,
                    time_scaling=time_scaling,
                    limits=limits,
                )
            current_position = target.copy()

        elif action.name == "align_pen_orientation":
            # The final end-effector convention belongs to the simulator backend.
            # Offline samples carry a fixed board-frame tip rotation placeholder.
            rotation = np.eye(3)
            continue

        elif action.name == "pen_down":
            _require_current(current_position, action)
            _require_pen_state(pen_state, "up", action)
            target = optional_point(action.params, "target", default_z=draw_height_m)
            if target is None:
                target = np.array([current_position[0], current_position[1], draw_height_m])
            if "target_z_m" in action.params:
                target[2] = float(action.params["target_z_m"])
            target[2] = float(target[2])
            new_samples = sample_line(
                current_position,
                target,
                get_float(action.params, "speed_m_s", default_speed_m_s),
                dt,
                start_time=start_time,
                rotation=rotation,
                pen_contact_desired=False,
                source_action=action.name,
                stroke_id=action.stroke_id,
                time_scaling=time_scaling,
                limits=limits,
            )
            if new_samples:
                new_samples = _apply_contact_force_ramp(
                    new_samples,
                    mode="up",
                    normal_force_n=default_normal_force_n,
                    ramp_duration_s=contact_force_ramp_duration_s,
                )
                new_samples.extend(
                    _sample_contact_settle(
                        target,
                        rotation,
                        duration_s=pen_down_settle_duration_s,
                        dt=dt,
                        start_time=new_samples[-1].t,
                        normal_force_n=default_normal_force_n,
                        source_action="pen_down_settle",
                        stroke_id=action.stroke_id,
                    )
                )
            current_position = target.copy()
            pen_state = "down"

        elif action.name in {"draw_line", "draw_line_to"}:
            _require_current(current_position, action)
            _require_pen_state(pen_state, "down", action)
            if action.name == "draw_line":
                explicit_start = optional_point(action.params, "start", default_z=draw_height_m)
                if explicit_start is not None and not np.allclose(
                    explicit_start,
                    current_position,
                    atol=1e-6,
                ):
                    raise ValueError(
                        "draw_line start does not match current sampler position."
                    )
                target = get_point(action.params, "end", default_z=draw_height_m)
            else:
                target = get_point(action.params, "target", default_z=draw_height_m)
            target[2] = float(target[2])
            new_samples = sample_line(
                current_position,
                target,
                get_float(action.params, "speed_m_s", default_speed_m_s),
                dt,
                start_time=start_time,
                rotation=rotation,
                pen_contact_desired=True,
                desired_normal_force_n=default_normal_force_n,
                source_action=action.name,
                stroke_id=action.stroke_id,
                time_scaling=time_scaling,
                limits=limits,
            )
            current_position = target.copy()

        elif action.name == "draw_arc":
            _require_current(current_position, action)
            _require_pen_state(pen_state, "down", action)
            center = get_point(action.params, "center", default_z=draw_height_m)
            radius = get_float(action.params, "radius_m")
            theta0 = get_float(action.params, "start_angle_rad")
            theta1 = get_float(action.params, "end_angle_rad")
            direction = get_str(action.params, "direction")
            expected_start = center + np.array(
                [radius * np.cos(theta0), radius * np.sin(theta0), 0.0]
            )
            if not np.allclose(expected_start, current_position, atol=1e-5):
                raise ValueError("draw_arc start point does not match current sampler position.")
            new_samples = sample_arc(
                center,
                radius,
                theta0,
                theta1,
                direction,
                get_float(action.params, "speed_m_s", default_speed_m_s),
                dt,
                start_time=start_time,
                rotation=rotation,
                pen_contact_desired=True,
                desired_normal_force_n=default_normal_force_n,
                source_action=action.name,
                stroke_id=action.stroke_id,
                time_scaling=time_scaling,
                limits=limits,
            )
            current_position = new_samples[-1].position.copy()

        elif action.name == "pen_up":
            _require_current(current_position, action)
            _require_pen_state(pen_state, "down", action)
            lift_height = get_float(action.params, "lift_height_m", hover_height_m)
            if "lift_distance_m" in action.params:
                lift_height = float(current_position[2]) + float(action.params["lift_distance_m"])
            target = np.array([current_position[0], current_position[1], lift_height])
            new_samples = sample_line(
                current_position,
                target,
                get_float(action.params, "speed_m_s", default_speed_m_s),
                dt,
                start_time=start_time,
                rotation=rotation,
                pen_contact_desired=False,
                source_action=action.name,
                stroke_id=action.stroke_id,
                time_scaling=time_scaling,
                limits=limits,
            )
            if new_samples:
                new_samples = _apply_contact_force_ramp(
                    new_samples,
                    mode="down",
                    normal_force_n=default_normal_force_n,
                    ramp_duration_s=contact_force_ramp_duration_s,
                )
            current_position = target
            pen_state = "up"

        else:
            raise ValueError(f"Unsupported action name: {action.name!r}.")

        _append_without_duplicate_time(samples, new_samples)

    return samples


def _append_without_duplicate_time(
    samples: list[PoseSample],
    new_samples: list[PoseSample],
) -> None:
    # Preserve segment-boundary samples because their source action and contact
    # flags are useful diagnostics even when the timestamp equals the previous
    # segment endpoint.
    samples.extend(new_samples)


def _sample_contact_settle(
    position: np.ndarray,
    rotation: np.ndarray,
    *,
    duration_s: float,
    dt: float,
    start_time: float,
    normal_force_n: float | None,
    source_action: str,
    stroke_id: str | None,
) -> list[PoseSample]:
    duration = max(float(duration_s), 0.0)
    dt_s = max(float(dt), 1e-12)
    if duration <= 1e-12:
        return []
    count = max(1, int(np.ceil(duration / dt_s)))
    desired_force = None if normal_force_n is None else max(float(normal_force_n), 0.0)
    contact = desired_force is None or desired_force > 0.0
    return [
        PoseSample(
            t=float(start_time + tau),
            position=np.asarray(position, dtype=float).copy(),
            rotation=np.asarray(rotation, dtype=float).copy(),
            linear_velocity=np.zeros(3),
            linear_acceleration=np.zeros(3),
            linear_jerk=np.zeros(3),
            pen_contact_desired=bool(contact),
            desired_normal_force_n=desired_force if contact else None,
            source_action=source_action,
            stroke_id=stroke_id,
        )
        for tau in np.linspace(dt_s, duration, count)
    ]


def _apply_contact_force_ramp(
    samples: list[PoseSample],
    *,
    mode: str,
    normal_force_n: float | None,
    ramp_duration_s: float,
) -> list[PoseSample]:
    if not samples:
        return samples
    if normal_force_n is None:
        out = [replace(sample, desired_normal_force_n=None) for sample in samples]
        if mode == "up":
            out[-1] = replace(out[-1], pen_contact_desired=True, desired_normal_force_n=None)
            return out
        if mode == "down":
            out[0] = replace(out[0], pen_contact_desired=True, desired_normal_force_n=None)
            out[-1] = replace(out[-1], pen_contact_desired=False, desired_normal_force_n=None)
            return out
        raise ValueError("mode must be 'up' or 'down'.")

    force = float(normal_force_n)
    if force <= 0.0:
        return [replace(sample, pen_contact_desired=False, desired_normal_force_n=None) for sample in samples]
    ramp = max(float(ramp_duration_s), 0.0)
    if ramp <= 1e-12 or len(samples) == 1:
        if mode == "up":
            out = list(samples)
            out[-1] = replace(out[-1], pen_contact_desired=True, desired_normal_force_n=force)
            return out
        if mode == "down":
            return [
                replace(sample, pen_contact_desired=False, desired_normal_force_n=None)
                for sample in samples
            ]
        raise ValueError("mode must be 'up' or 'down'.")

    start_t = float(samples[0].t)
    end_t = float(samples[-1].t)
    duration = max(end_t - start_t, 1e-12)
    active_ramp = min(ramp, duration)
    output: list[PoseSample] = []
    for sample in samples:
        tau = float(sample.t) - start_t
        if mode == "up":
            ratio = np.clip((tau - (duration - active_ramp)) / active_ramp, 0.0, 1.0)
            desired_force = force * ratio
        elif mode == "down":
            ratio = np.clip(tau / active_ramp, 0.0, 1.0)
            desired_force = force * (1.0 - ratio)
        else:
            raise ValueError("mode must be 'up' or 'down'.")
        contact = desired_force > 1e-6
        output.append(
            replace(
                sample,
                pen_contact_desired=bool(contact),
                desired_normal_force_n=float(desired_force) if contact else None,
            )
        )
    if mode == "up":
        output[-1] = replace(output[-1], pen_contact_desired=True, desired_normal_force_n=force)
    else:
        output[-1] = replace(output[-1], pen_contact_desired=False, desired_normal_force_n=None)
    return output


def _require_current(current_position: np.ndarray | None, action: PrimitiveAction) -> None:
    if current_position is None:
        raise ValueError(f"{action.name} requires a current position; call move_to_start first.")


def _require_pen_state(actual: str, expected: str, action: PrimitiveAction) -> None:
    if actual != expected:
        raise ValueError(f"{action.name} requires pen_state == {expected!r}.")
