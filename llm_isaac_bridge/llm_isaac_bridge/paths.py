"""Path defaults for the standalone LLM-to-Isaac bridge."""

from __future__ import annotations

from pathlib import Path

BRIDGE_ROOT = Path(__file__).resolve().parents[1]
IR_ROOT = BRIDGE_ROOT.parent
ISAACLAB_ROOT = IR_ROOT.parent

DEFAULT_LLM_ROOT = IR_ROOT / "Unit_Action_Langchain-main"
DEFAULT_CONTROL_ROOT = IR_ROOT
DEFAULT_PLANNER_CONFIG = BRIDGE_ROOT / "configs" / "isaac_planner_config.json"
DEFAULT_TEXT_TO_ISAAC_OUTPUT_ROOT = IR_ROOT / "outputs" / "text_to_isaac"
DEFAULT_ISAAC_RUNNER = IR_ROOT / "examples" / "run_jade_isaac.py"
DEFAULT_PLAN_DIR = DEFAULT_TEXT_TO_ISAAC_OUTPUT_ROOT / "plans"
DEFAULT_REPORT_DIR = DEFAULT_TEXT_TO_ISAAC_OUTPUT_ROOT / "reports"
DEFAULT_ISAAC_OUTPUT_DIR = DEFAULT_TEXT_TO_ISAAC_OUTPUT_ROOT / "isaac_runs"
