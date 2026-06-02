"""Adaptive speed and DLS damping policy for JADE."""

from __future__ import annotations

from franka_llm_drawing.config import AdaptivePolicyConfig, ExecutorConfig, ThresholdConfig
from franka_llm_drawing.jade.jacobian_metrics import JacobianMetrics
from franka_llm_drawing.jade.reports import MotionPolicy, Status


def recommend_motion_policy(
    metrics: JacobianMetrics,
    thresholds: ThresholdConfig | None = None,
    adaptive: AdaptivePolicyConfig | None = None,
    executor: ExecutorConfig | None = None,
) -> MotionPolicy:
    """Return a conservative damping/speed policy for one waypoint."""

    th = thresholds or ThresholdConfig()
    ad = adaptive or AdaptivePolicyConfig()
    ex = executor or ExecutorConfig()
    status, reason = classify_metrics(metrics, th)
    risk = _risk_score(metrics, th)
    lambda_dls = ad.lambda_min + risk * (ad.lambda_max - ad.lambda_min)
    if status == "OK":
        speed_scale = 1.0
    elif status == "WARNING":
        speed_scale = max(ad.speed_scale_warning, ad.speed_scale_min)
    else:
        speed_scale = ad.speed_scale_min
    return MotionPolicy(
        speed_scale=float(speed_scale),
        lambda_dls=float(lambda_dls),
        max_qdot=float(ex.max_qdot_rad_s),
        max_delta_q=float(ex.max_delta_q_rad),
        status=status,
        reason=reason,
    )


def classify_metrics(metrics: JacobianMetrics, thresholds: ThresholdConfig | None = None) -> tuple[Status, str]:
    th = thresholds or ThresholdConfig()
    fail_reasons: list[str] = []
    warning_reasons: list[str] = []
    if metrics.sigma_min < th.sigma_min_fail:
        fail_reasons.append("sigma_min below fail threshold")
    elif metrics.sigma_min < th.sigma_min_warning:
        warning_reasons.append("sigma_min below warning threshold")
    if metrics.condition_number > th.condition_fail:
        fail_reasons.append("condition number above fail threshold")
    elif metrics.condition_number > th.condition_warning:
        warning_reasons.append("condition number above warning threshold")
    if metrics.joint_limit_margin < th.joint_margin_fail_rad:
        fail_reasons.append("joint limit margin below fail threshold")
    elif metrics.joint_limit_margin < th.joint_margin_warning_rad:
        warning_reasons.append("joint limit margin below warning threshold")
    if metrics.manipulability < th.manipulability_warning:
        warning_reasons.append("manipulability below warning threshold")

    if fail_reasons:
        return "FAIL", "; ".join(fail_reasons)
    if warning_reasons:
        return "WARNING", "; ".join(warning_reasons)
    return "OK", "metrics within configured thresholds"


def _risk_score(metrics: JacobianMetrics, th: ThresholdConfig) -> float:
    risks = [
        _low_value_risk(metrics.sigma_min, th.sigma_min_warning, th.sigma_min_fail),
        _high_value_risk(metrics.condition_number, th.condition_warning, th.condition_fail),
        _low_value_risk(metrics.joint_limit_margin, th.joint_margin_warning_rad, th.joint_margin_fail_rad),
    ]
    return max(risks)


def _low_value_risk(value: float, warning: float, fail: float) -> float:
    if value >= warning:
        return 0.0
    if value <= fail:
        return 1.0
    return float((warning - value) / max(warning - fail, 1e-12))


def _high_value_risk(value: float, warning: float, fail: float) -> float:
    if value <= warning:
        return 0.0
    if value >= fail:
        return 1.0
    return float((value - warning) / max(fail - warning, 1e-12))
