"""Load LLM DrawingPlan JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from franka_llm_drawing.llm_bridge.plan_schema import DrawingPlan


def load_plan_json(path: str | Path) -> DrawingPlan:
    """Load a ``DrawingPlan`` from a JSON file."""

    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return drawing_plan_from_obj(payload)


def drawing_plan_from_obj(obj: DrawingPlan | Mapping[str, Any]) -> DrawingPlan:
    """Normalize either an existing dataclass or a JSON mapping."""

    if isinstance(obj, DrawingPlan):
        return obj
    if not isinstance(obj, Mapping):
        raise TypeError("Expected DrawingPlan or JSON mapping.")
    return DrawingPlan.from_mapping(obj)
