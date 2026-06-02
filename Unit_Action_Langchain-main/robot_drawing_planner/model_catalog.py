"""OpenAI model discovery helpers for planner UI choices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robot_drawing_planner.config import DEFAULT_MODEL, require_openai_api_key

DEFAULT_MODEL_CHOICES = [
    DEFAULT_MODEL,
    "gpt-5-mini",
    "gpt-5",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4o",
]

EXCLUDED_MODEL_ID_PARTS = (
    "audio",
    "dall",
    "embedding",
    "image",
    "moderation",
    "realtime",
    "sora",
    "speech",
    "tts",
    "transcribe",
    "vision",
    "whisper",
)


@dataclass(frozen=True)
class ModelInfo:
    """Small JSON-safe model metadata object for UI display."""

    id: str
    owned_by: str | None = None
    created: int | None = None
    object: str | None = None


def list_openai_models(client: Any | None = None) -> list[ModelInfo]:
    """List models visible to OPENAI_API_KEY.

    The OpenAI models endpoint returns basic metadata such as model id, object,
    created timestamp, and owner. It does not fully describe planner capability.
    """

    require_openai_api_key()
    if client is None:
        from openai import OpenAI

        client = OpenAI()
    response = client.models.list()
    data = getattr(response, "data", response)
    return [_model_info_from_any(item) for item in data]


def planner_model_choices(models: list[ModelInfo]) -> list[str]:
    """Return dropdown-friendly planner model ids with safe fallbacks."""

    discovered = sorted(
        {model.id for model in models if is_planner_model_id(model.id)},
        key=_model_sort_key,
    )
    choices = []
    for model_id in [*DEFAULT_MODEL_CHOICES, *discovered]:
        if model_id not in choices:
            choices.append(model_id)
    return choices


def get_planner_model_choices(client: Any | None = None) -> list[str]:
    """Fetch planner model choices from the API, falling back to defaults on error."""

    try:
        return planner_model_choices(list_openai_models(client=client))
    except Exception:
        return list(DEFAULT_MODEL_CHOICES)


def is_planner_model_id(model_id: str) -> bool:
    """Heuristic filter for text/reasoning models useful for this planner."""

    normalized = model_id.lower()
    if any(part in normalized for part in EXCLUDED_MODEL_ID_PARTS):
        return False
    return normalized.startswith(("gpt-", "o1", "o3", "o4"))


def _model_info_from_any(item: Any) -> ModelInfo:
    if isinstance(item, dict):
        return ModelInfo(
            id=str(item["id"]),
            owned_by=item.get("owned_by"),
            created=item.get("created"),
            object=item.get("object"),
        )
    return ModelInfo(
        id=str(getattr(item, "id")),
        owned_by=getattr(item, "owned_by", None),
        created=getattr(item, "created", None),
        object=getattr(item, "object", None),
    )


def _model_sort_key(model_id: str) -> tuple[int, str]:
    normalized = model_id.lower()
    if normalized == DEFAULT_MODEL:
        return (0, normalized)
    if normalized.startswith("gpt-5"):
        return (1, normalized)
    if normalized.startswith("gpt-4"):
        return (2, normalized)
    if normalized.startswith(("o1", "o3", "o4")):
        return (3, normalized)
    return (4, normalized)
