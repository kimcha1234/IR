import json
import subprocess
import sys

import pytest

from robot_drawing_planner import cli
from robot_drawing_planner.schemas import DrawingPlan, NormalizedGoal, Point2D


def _minimal_plan(command: str) -> DrawingPlan:
    return DrawingPlan(
        source_command=command,
        goal=NormalizedGoal(
            shape_type="custom",
            center=Point2D(x=0.0, y=0.0),
            radius_m=None,
            side_length_m=None,
            size_m=0.1,
            orientation_rad=0.0,
            letter=None,
            assumptions=[],
            warnings=[],
        ),
        strokes=[],
        actions=[],
        diagnostics={"validation_ok": True},
    )


def _run_no_api_payload(command: str, capsys):
    result = cli.main([command, "--no-api"])
    captured = capsys.readouterr()

    assert result == 0
    assert captured.err == ""
    return json.loads(captured.out)


def test_cli_no_api_square_exits_zero_and_prints_valid_json(capsys):
    result = cli.main(["중앙에 한 변 10cm짜리 네모를 그려줘", "--no-api"])
    captured = capsys.readouterr()

    assert result == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["goal"]["shape_type"] == "square"
    assert payload["goal"]["side_length_m"] == 0.1
    assert payload["actions"][0]["name"] == "move_to_start"


@pytest.mark.parametrize(
    ("command", "goal_key", "expected"),
    [
        ("중앙에 한 변 20cm짜리 네모를 그려줘", "side_length_m", 0.20),
        ("중앙에 반지름 8cm짜리 원을 그려줘", "radius_m", 0.08),
        ("중앙에 크기 15cm인 글자 A를 써줘", "size_m", 0.15),
        ("중앙에 한 변 0.12m짜리 세모를 그려줘", "side_length_m", 0.12),
        ("중앙에 반지름 5mm짜리 원을 그려줘", "radius_m", 0.005),
        ("중앙에 크기 20cm인 원을 그려줘", "radius_m", 0.10),
        ("중앙에 지름 20cm인 원을 그려줘", "radius_m", 0.10),
        ("Draw a circle with diameter 20 cm", "radius_m", 0.10),
        ("중앙에 크기 20cm인 네모를 그려줘", "side_length_m", 0.20),
        ("중앙에 크기 20cm인 세모를 그려줘", "side_length_m", 0.20),
    ],
)
def test_cli_no_api_measurement_suffixes_and_size_semantics(
    command,
    goal_key,
    expected,
    capsys,
):
    payload = _run_no_api_payload(command, capsys)

    assert payload["goal"][goal_key] == pytest.approx(expected)


def test_cli_no_api_out_writes_file_and_prints_stdout(tmp_path, capsys):
    output = tmp_path / "outputs" / "circle.json"
    result = cli.main(
        [
            "Draw a circle with radius 5 cm",
            "--no-api",
            "--out",
            str(output),
            "--pretty",
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert output.exists()
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    stdout_payload = json.loads(captured.out)
    assert file_payload == stdout_payload
    assert file_payload["goal"]["shape_type"] == "circle"
    assert file_payload["actions"][3]["name"] == "draw_arc"


def test_cli_no_api_out_only_suppresses_stdout(tmp_path, capsys):
    output = tmp_path / "plan.json"
    result = cli.main(["세모 변 길이 10cm", "--no-api", "--out", str(output), "--out-only"])
    captured = capsys.readouterr()

    assert result == 0
    assert captured.out == ""
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["goal"]["shape_type"] == "triangle"


def test_cli_no_api_korean_letter_a(tmp_path, capsys):
    output = tmp_path / "letter_A.json"
    result = cli.main(
        ["중앙에 크기 10cm인 글자 A를 써줘", "--no-api", "--out", str(output), "--out-only"]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert captured.out == ""
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["goal"]["shape_type"] == "letter"
    assert payload["goal"]["letter"] == "A"
    assert payload["actions"]


def test_cli_no_api_plot_out_writes_plan_and_png(tmp_path):
    plan_path = tmp_path / "plan.json"
    plot_path = tmp_path / "plan.png"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "robot_drawing_planner.cli",
            "중앙에 한 변 10cm짜리 네모를 그려줘",
            "--no-api",
            "--out",
            str(plan_path),
            "--plot-out",
            str(plot_path),
            "--out-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert plan_path.exists()
    assert plot_path.exists()
    assert plot_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_cli_mode_template_with_no_api_still_uses_demo_fallback(capsys):
    result = cli.main(["Draw a circle with radius 5 cm", "--mode", "template", "--no-api"])
    captured = capsys.readouterr()

    assert result == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["goal"]["shape_type"] == "circle"
    assert payload["actions"][3]["name"] == "draw_arc"


def test_cli_default_mode_missing_live_key_is_nonzero_and_no_secret(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = cli.main(["Draw a circle with radius 5 cm"])
    captured = capsys.readouterr()

    assert result != 0
    assert "OPENAI_API_KEY is not set" in captured.err
    assert "sk-" not in captured.err
    assert captured.out == ""


def test_cli_help_mentions_agentic_unit_action_tool_planner(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    captured = capsys.readouterr()

    assert exc.value.code == 0
    help_text = " ".join(captured.out.split())
    assert "Default mode uses LLM agentic planning over Unit Action tools" in help_text
    assert "template" in help_text
    assert "ParsedGoal/template compiler baseline" in help_text
    assert "development/testing only" in help_text
    assert "--chatgpt-version" in help_text
    assert "--max-steps" in help_text
    assert "--max-llm-steps" in help_text
    assert "--max-tool-calls" in help_text
    assert "--stream-events" in help_text
    assert "--request-timeout" in help_text
    assert "default: 80" in help_text
    assert "default: 200" in help_text
    assert "default: 120" in help_text


def test_cli_rejects_agentic_max_steps_above_1000(capsys):
    result = cli.main(["draw a square", "--max-steps", "1001"])
    captured = capsys.readouterr()

    assert result == 1
    assert "--max-steps/--max-llm-steps must be between 1 and 1000" in captured.err
    assert captured.out == ""


def test_cli_rejects_agentic_max_tool_calls_above_1000(capsys):
    result = cli.main(["draw a square", "--max-tool-calls", "1001"])
    captured = capsys.readouterr()

    assert result == 1
    assert "--max-tool-calls must be between 1 and 1000" in captured.err
    assert captured.out == ""


def test_cli_no_api_ignores_agentic_budget_options(capsys):
    result = cli.main(
        [
            "중앙에 한 변 10cm짜리 네모를 그려줘",
            "--no-api",
            "--max-llm-steps",
            "1001",
            "--max-tool-calls",
            "1001",
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["goal"]["shape_type"] == "square"


def test_cli_passes_agentic_budget_values_to_planner(monkeypatch, capsys):
    captured_values = {}

    def fake_get_llm(model_name, timeout_seconds=None):
        captured_values["model_name"] = model_name
        captured_values["timeout_seconds"] = timeout_seconds
        return object()

    def fake_plan_drawing_agentic(
        command,
        *,
        config,
        llm,
        max_llm_steps,
        max_tool_calls,
        event_callback=None,
    ):
        captured_values["command"] = command
        captured_values["config"] = config
        captured_values["llm"] = llm
        captured_values["max_llm_steps"] = max_llm_steps
        captured_values["max_tool_calls"] = max_tool_calls
        captured_values["event_callback"] = event_callback
        return _minimal_plan(command)

    monkeypatch.setattr(cli, "get_llm", fake_get_llm)
    monkeypatch.setattr(cli, "plan_drawing_agentic", fake_plan_drawing_agentic)

    result = cli.main(
        [
            "draw a custom shape",
            "--max-llm-steps",
            "77",
            "--max-tool-calls",
            "123",
            "--request-timeout",
            "45",
            "--pretty",
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert captured.err == ""
    assert json.loads(captured.out)["source_command"] == "draw a custom shape"
    assert captured_values["command"] == "draw a custom shape"
    assert captured_values["max_llm_steps"] == 77
    assert captured_values["max_tool_calls"] == 123
    assert captured_values["timeout_seconds"] == 45
    assert captured_values["event_callback"] is None


def test_cli_rejects_nonpositive_request_timeout(capsys):
    result = cli.main(["draw a square", "--request-timeout", "0"])
    captured = capsys.readouterr()

    assert result == 1
    assert "--request-timeout must be positive" in captured.err
    assert captured.out == ""


def test_cli_terminal_event_line_is_single_line():
    event = cli.AgentRunEvent(
        event_index=0,
        step_index=1,
        event_type="tool_result",
        tool_name="draw_line_to",
        tool_args=None,
        tool_result=None,
        message="draw_line appended.\nnext",
        ok=True,
        timestamp="2026-01-01T00:00:00+00:00",
        metadata={},
    )

    line = cli._event_to_terminal_line(event)

    assert "\n" not in line
    assert "draw_line_to" in line
    assert "ok=True" in line
