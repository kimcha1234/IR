"""Helpers for extracting typed primitive parameters from JSON actions."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from franka_llm_drawing.llm_bridge.plan_schema import point3d_from_mapping


def get_point(
    params: Mapping[str, Any],
    key: str,
    *,
    default_z: float = 0.0,
) -> np.ndarray:
    """Return a point parameter as a shape ``(3,)`` meter array."""

    value = params.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Action parameter {key!r} must be a point mapping.")
    return point3d_from_mapping(value, default_z=default_z).to_array()


def optional_point(
    params: Mapping[str, Any],
    key: str,
    *,
    default_z: float = 0.0,
) -> np.ndarray | None:
    """Return a point parameter if present."""

    value = params.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"Action parameter {key!r} must be a point mapping.")
    return point3d_from_mapping(value, default_z=default_z).to_array()


def get_float(params: Mapping[str, Any], key: str, default: float | None = None) -> float:
    """Return a float parameter, optionally using a default."""

    value = params.get(key, default)
    if value is None:
        raise ValueError(f"Action parameter {key!r} is required.")
    return float(value)


def get_str(params: Mapping[str, Any], key: str, default: str | None = None) -> str:
    """Return a string parameter, optionally using a default."""

    value = params.get(key, default)
    if value is None:
        raise ValueError(f"Action parameter {key!r} is required.")
    return str(value)
