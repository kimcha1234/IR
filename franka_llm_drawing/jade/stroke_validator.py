"""Stroke-level JADE feasibility validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from franka_llm_drawing.config import AdaptivePolicyConfig, ExecutorConfig, ThresholdConfig
from franka_llm_drawing.jade.adaptive_policy import classify_metrics, recommend_motion_policy
from franka_llm_drawing.jade.jacobian_metrics import JacobianMetrics
from franka_llm_drawing.jade.reports import FeasibilityReport


def validate_stroke(
    metrics: Iterable[JacobianMetrics],
    thresholds: ThresholdConfig | None = None,
    adaptive: AdaptivePolicyConfig | None = None,
    executor: ExecutorConfig | None = None,
) -> FeasibilityReport:
    """Aggregate waypoint metrics into one feasibility report."""

    metric_list = list(metrics)
    if not metric_list:
        raise ValueError("metrics must not be empty.")
    th = thresholds or ThresholdConfig()
    ad = adaptive or AdaptivePolicyConfig()
    ex = executor or ExecutorConfig()
    warning_waypoints: list[int] = []
    fail_waypoints: list[int] = []
    reasons: list[str] = []
    worst_policy = recommend_motion_policy(metric_list[0], th, ad, ex)

    for item in metric_list:
        status, reason = classify_metrics(item, th)
        if status == "FAIL":
            fail_waypoints.append(item.waypoint_index)
            reasons.append(f"wp {item.waypoint_index}: {reason}")
        elif status == "WARNING":
            warning_waypoints.append(item.waypoint_index)
            reasons.append(f"wp {item.waypoint_index}: {reason}")
        policy = recommend_motion_policy(item, th, ad, ex)
        if policy.lambda_dls > worst_policy.lambda_dls:
            worst_policy = policy

    status = "FAIL" if fail_waypoints else "WARNING" if warning_waypoints else "OK"
    return FeasibilityReport(
        stroke_id=metric_list[0].stroke_id,
        status=status,
        min_manipulability=float(min(item.manipulability for item in metric_list)),
        max_condition_number=float(max(item.condition_number for item in metric_list)),
        min_sigma=float(min(item.sigma_min for item in metric_list)),
        min_joint_limit_margin=float(min(item.joint_limit_margin for item in metric_list)),
        recommended_speed_scale=worst_policy.speed_scale,
        recommended_damping=worst_policy.lambda_dls,
        warning_waypoints=warning_waypoints,
        fail_waypoints=fail_waypoints,
        reasons=reasons or ["all sampled waypoints are within configured thresholds"],
    )


def validate_metrics_by_stroke(
    metrics: Iterable[JacobianMetrics],
    thresholds: ThresholdConfig | None = None,
    adaptive: AdaptivePolicyConfig | None = None,
    executor: ExecutorConfig | None = None,
) -> list[FeasibilityReport]:
    grouped: dict[str, list[JacobianMetrics]] = defaultdict(list)
    for item in metrics:
        grouped[item.stroke_id].append(item)
    return [
        validate_stroke(grouped[stroke_id], thresholds, adaptive, executor)
        for stroke_id in sorted(grouped)
    ]
