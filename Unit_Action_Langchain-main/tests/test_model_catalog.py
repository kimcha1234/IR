from types import SimpleNamespace

from robot_drawing_planner.model_catalog import (
    DEFAULT_MODEL_CHOICES,
    ModelInfo,
    get_planner_model_choices,
    is_planner_model_id,
    list_openai_models,
    planner_model_choices,
)


class FakeModelsClient:
    def __init__(self, data):
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=data))


def test_is_planner_model_id_filters_non_text_models():
    assert is_planner_model_id("gpt-5-nano")
    assert is_planner_model_id("gpt-4.1-mini")
    assert is_planner_model_id("o4-mini")
    assert not is_planner_model_id("text-embedding-3-small")
    assert not is_planner_model_id("gpt-image-1")
    assert not is_planner_model_id("gpt-realtime")


def test_planner_model_choices_includes_defaults_and_discovered_models():
    choices = planner_model_choices(
        [
            ModelInfo(id="gpt-5.2"),
            ModelInfo(id="text-embedding-3-small"),
            ModelInfo(id="gpt-image-1"),
        ]
    )

    assert DEFAULT_MODEL_CHOICES[0] in choices
    assert "gpt-5.2" in choices
    assert "text-embedding-3-small" not in choices


def test_list_openai_models_with_fake_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = FakeModelsClient(
        [
            {"id": "gpt-5-nano", "object": "model", "created": 1, "owned_by": "openai"},
            SimpleNamespace(id="gpt-4o-mini", object="model", created=2, owned_by="openai"),
        ]
    )

    models = list_openai_models(client=client)

    assert [model.id for model in models] == ["gpt-5-nano", "gpt-4o-mini"]
    assert models[0].owned_by == "openai"


def test_get_planner_model_choices_falls_back_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert get_planner_model_choices() == DEFAULT_MODEL_CHOICES
