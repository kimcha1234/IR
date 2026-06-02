"""LangChain ChatOpenAI client construction and structured parsing."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Protocol

from robot_drawing_planner.config import (
    DEFAULT_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    require_openai_api_key,
)
from robot_drawing_planner.prompts import SYSTEM_PROMPT
from robot_drawing_planner.schemas import ParsedGoal

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


class Invokable(Protocol):
    """Small protocol used by real LangChain runnables and tests."""

    def invoke(self, input: Any) -> Any:
        ...


def get_llm(
    model_name: str | None = None,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
) -> "ChatOpenAI":
    """Build a deterministic ChatOpenAI client.

    Model priority:
    1. explicit model_name argument
    2. OPENAI_MODEL environment variable
    3. gpt-5-nano
    """

    require_openai_api_key()
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model_name or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL,
        temperature=0,
        timeout=timeout_seconds if timeout_seconds is not None else DEFAULT_TIMEOUT_SECONDS,
        max_retries=max_retries if max_retries is not None else DEFAULT_MAX_RETRIES,
    )


def parse_command_with_llm(command: str, llm: Any | None = None) -> ParsedGoal:
    """Parse a natural-language drawing command into a ParsedGoal."""

    if not command.strip():
        raise ValueError("Drawing command must not be empty.")

    model_or_fake = llm if llm is not None else get_llm()
    if hasattr(model_or_fake, "with_structured_output"):
        parser = model_or_fake.with_structured_output(ParsedGoal, method="json_schema")
    else:
        parser = model_or_fake

    result = parser.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", command),
        ]
    )
    if isinstance(result, ParsedGoal):
        return result
    if isinstance(result, dict) and "raw_command" not in result:
        result = {**result, "raw_command": command}
    return ParsedGoal.model_validate(result)


def build_chat_model() -> "ChatOpenAI":
    """Backward-compatible alias for get_llm()."""

    return get_llm()


def build_structured_parser(llm: Any | None = None) -> Invokable:
    """Return a LangChain runnable that emits ParsedGoal instances."""

    model_or_fake = llm if llm is not None else get_llm()
    return model_or_fake.with_structured_output(ParsedGoal, method="json_schema")


def parse_goal(command: str, parser: Invokable | None = None) -> ParsedGoal:
    """Backward-compatible wrapper around parse_command_with_llm()."""

    return parse_command_with_llm(command, llm=parser)
