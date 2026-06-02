import re
from pathlib import Path

import numpy as np

from franka_llm_drawing.config import (
    FRANKA_ARM_JOINT_NAMES,
    load_frames_config,
    load_isaac_scene_config,
    load_jade_config,
)


ROOT = Path(__file__).resolve().parents[1]
USD_PATH = ROOT / "usd" / "franka_drawing_scene_clean.usda"


def test_project_config_does_not_use_shape_specific_command_offset() -> None:
    jade = load_jade_config(ROOT / "configs" / "jade.yaml")

    assert np.allclose(jade.executor.command_position_offset_m, np.zeros(3))


def test_scene_backend_commands_only_franka_arm_joints() -> None:
    scene = load_isaac_scene_config(ROOT / "configs" / "isaac_scene.yaml", project_root=ROOT)

    assert scene.joint_names == FRANKA_ARM_JOINT_NAMES
    assert scene.ee_frame_name == "panda_link7"


def test_board_frame_matches_paper_and_frame_config() -> None:
    frames = load_frames_config(ROOT / "configs" / "frames.yaml")
    usda = USD_PATH.read_text(encoding="utf-8")

    board_origin = _translate(usda, 'def Xform "BoardFrame"')
    paper_center = _translate(usda, 'def Cube "Paper"')
    paper_scale = _scale(usda, 'def Cube "Paper"')
    table_center = _translate(usda, 'def Cube "Table"')
    table_scale = _scale(usda, 'def Cube "Table"')

    np.testing.assert_allclose(board_origin, frames.T_base_board[:3, 3], atol=1e-9)
    np.testing.assert_allclose(board_origin, np.array([0.5, 0.0, 0.202]), atol=1e-9)
    assert abs((paper_center[2] + paper_scale[2]) - board_origin[2]) < 1e-9
    assert abs((paper_center[2] - paper_scale[2]) - (table_center[2] + table_scale[2])) < 1e-9


def test_pen_tip_frame_matches_pen_contact_geometry() -> None:
    usda = USD_PATH.read_text(encoding="utf-8")

    tip_center = _translate(usda, 'def Sphere "PenTip"')
    tip_radius = _float_attr(usda, 'def Sphere "PenTip"', "radius")
    tip_frame = _translate(usda, 'def Xform "PenTipFrame"')

    assert abs((tip_center[2] + tip_radius) - tip_frame[2]) < 1e-9
    np.testing.assert_allclose(tip_frame, np.array([0.0, 0.0, 0.1]), atol=1e-9)


def test_tip_orientation_points_against_board_normal() -> None:
    frames = load_frames_config(ROOT / "configs" / "frames.yaml")

    np.testing.assert_allclose(frames.R_base_tip.T @ frames.R_base_tip, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(frames.T_ee_tip[:3, :3].T @ frames.T_ee_tip[:3, :3], np.eye(3), atol=1e-6)
    assert float(np.dot(frames.R_base_tip[:, 2], frames.board_normal_base)) < -0.999


def _block(text: str, header: str) -> str:
    match = re.search(re.escape(header) + r".*?(?=\n\s*(?:def|over|}$))", text, re.DOTALL)
    if not match:
        raise AssertionError(f"Could not find USDA block header: {header}")
    return match.group(0)


def _translate(text: str, header: str) -> np.ndarray:
    block = _block(text, header)
    match = re.search(r"double3 xformOp:translate = \(([^)]*)\)", block)
    if not match:
        raise AssertionError(f"Could not find xformOp:translate in {header}")
    return _vector(match.group(1))


def _scale(text: str, header: str) -> np.ndarray:
    block = _block(text, header)
    match = re.search(r"double3 xformOp:scale = \(([^)]*)\)", block)
    if not match:
        raise AssertionError(f"Could not find xformOp:scale in {header}")
    return _vector(match.group(1))


def _float_attr(text: str, header: str, attr: str) -> float:
    block = _block(text, header)
    match = re.search(rf"double {re.escape(attr)} = ([^\n]+)", block)
    if not match:
        raise AssertionError(f"Could not find {attr} in {header}")
    return float(match.group(1).strip())


def _vector(value: str) -> np.ndarray:
    return np.asarray([float(item.strip()) for item in value.split(",")], dtype=float)
