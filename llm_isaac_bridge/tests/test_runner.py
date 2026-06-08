from __future__ import annotations

import pytest

from llm_isaac_bridge.runner import _isaac_extra_args, build_parser


def test_recording_options_are_forwarded_to_isaac_runner() -> None:
    args = build_parser().parse_args(
        [
            "--command",
            "중앙에 8cm 별을 그려줘",
            "--debug-draw",
            "--start-delay-s",
            "5",
            "--keep-open",
        ]
    )

    assert _isaac_extra_args(args) == ["--debug-draw", "--start-delay-s", "5.0", "--keep-open"]


def test_default_recording_options_do_not_add_isaac_args() -> None:
    args = build_parser().parse_args(["--command", "중앙에 8cm 별을 그려줘"])

    assert _isaac_extra_args(args) == []


def test_negative_start_delay_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--command", "draw a star", "--start-delay-s", "-1"])
