"""Validate the cleaned Franka drawing USD scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError as exc:  # pragma: no cover - depends on local USD install
    raise SystemExit(
        "This validator requires OpenUSD Python bindings. "
        "Run with Isaac Sim Python or install usd-core."
    ) from exc


EXPECTED_PRIMS = [
    "/World",
    "/World/Franka",
    "/World/Franka/panda_link7/panda_link8/panda_hand",
    "/World/Franka/panda_link7/panda_link8/panda_hand/PenTool",
    "/World/Franka/panda_link7/panda_link8/panda_hand/PenTool/PenBody",
    "/World/Franka/panda_link7/panda_link8/panda_hand/PenTool/PenTip",
    "/World/Franka/panda_link7/panda_link8/panda_hand/PenTool/PenTipFrame",
    "/World/Table",
    "/World/Paper",
    "/World/BoardFrame",
    "/World/PhysicsMaterials/PaperMaterial",
    "/World/PhysicsMaterials/PenTipMaterial",
    "/World/PhysicsMaterials/TableMaterial",
]

EXPECTED_MATERIALS = {
    "/World/PhysicsMaterials/PaperMaterial": (0.18, 0.12, 0.0),
    "/World/PhysicsMaterials/PenTipMaterial": (0.08, 0.05, 0.0),
    "/World/PhysicsMaterials/TableMaterial": (0.6, 0.45, 0.0),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene",
        nargs="?",
        default=str(Path(__file__).with_name("franka_drawing_scene_clean.usda")),
    )
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.scene)
    if stage is None:
        print(f"ERROR: failed to open scene: {args.scene}", file=sys.stderr)
        return 1

    errors: list[str] = []
    warnings: list[str] = []

    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        errors.append("Stage upAxis must be Z.")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-9:
        errors.append("Stage metersPerUnit must be 1.0.")

    for path in EXPECTED_PRIMS:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            errors.append(f"Missing prim: {path}")

    _check_bounds(stage, "/World/Table", (0.6, 0.6, 0.1), errors)
    _check_bounds(stage, "/World/Paper", (0.3, 0.3, 0.002), errors)
    _check_translation(stage, "/World/BoardFrame", (0.5, 0.0, 0.202), errors)
    _check_float_attr(
        stage,
        "/World/Franka/panda_link7/panda_link8/panda_hand/PenTool/PenBody",
        "height",
        0.094,
        errors,
    )
    _check_local_translation(
        stage,
        "/World/Franka/panda_link7/panda_link8/panda_hand/PenTool/PenBody",
        (0.0, 0.0, 0.047),
        errors,
    )
    _check_local_translation(
        stage,
        "/World/Franka/panda_link7/panda_link8/panda_hand/PenTool/PenTip",
        (0.0, 0.0, 0.097),
        errors,
    )
    _check_local_translation(
        stage,
        "/World/Franka/panda_link7/panda_link8/panda_hand/PenTool/PenTipFrame",
        (0.0, 0.0, 0.1),
        errors,
    )
    _check_translation(
        stage,
        "/World/Franka/panda_link7/panda_link8/panda_hand/PenTool/PenTipFrame",
        None,
        errors,
        require_finite=True,
    )

    for path, expected in EXPECTED_MATERIALS.items():
        _check_material(stage, path, expected, errors)

    bindings = {
        "/World/Paper": "/World/PhysicsMaterials/PaperMaterial",
        "/World/Table": "/World/PhysicsMaterials/TableMaterial",
        "/World/Franka/panda_link7/panda_link8/panda_hand/PenTool/PenBody": (
            "/World/PhysicsMaterials/PenTipMaterial"
        ),
        "/World/Franka/panda_link7/panda_link8/panda_hand/PenTool/PenTip": (
            "/World/PhysicsMaterials/PenTipMaterial"
        ),
    }
    for prim_path, material_path in bindings.items():
        _check_binding(stage, prim_path, material_path, errors)

    for path in [
        "/World/Franka/panda_leftfinger/collisions",
        "/World/Franka/panda_rightfinger/collisions",
    ]:
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsActive():
            errors.append(f"Finger collision prim should be inactive: {path}")

    arm_joints = []
    finger_joints = []
    for prim in stage.Traverse():
        if prim.GetTypeName() in ["PhysicsRevoluteJoint", "PhysicsPrismaticJoint"]:
            path = str(prim.GetPath())
            if "/panda_joint" in path:
                arm_joints.append(path)
            if "finger" in path:
                finger_joints.append(path)
    if len(arm_joints) != 7:
        errors.append(f"Expected 7 arm joints, found {len(arm_joints)}.")
    if finger_joints:
        warnings.append(
            "Finger joints still exist; Isaac backend should command only panda_joint1..7."
        )

    print(f"Scene: {args.scene}")
    print(f"Arm joints: {len(arm_joints)}")
    print(f"Finger joints: {len(finger_joints)}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Validation OK")
    return 0


def _check_bounds(stage: Usd.Stage, path: str, expected_size: tuple[float, ...], errors: list[str]) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        return
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    size = bbox.GetMax() - bbox.GetMin()
    if any(abs(float(size[i]) - expected_size[i]) > 1e-6 for i in range(3)):
        errors.append(f"{path} size mismatch: got {tuple(size)}, expected {expected_size}")


def _check_translation(
    stage: Usd.Stage,
    path: str,
    expected: tuple[float, ...] | None,
    errors: list[str],
    require_finite: bool = False,
) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        return
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = tuple(float(matrix[3][i]) for i in range(3))
    if require_finite and any(not (abs(value) < 1e6) for value in translation):
        errors.append(f"{path} translation is not finite: {translation}")
    if expected and any(abs(translation[i] - expected[i]) > 1e-6 for i in range(3)):
        errors.append(f"{path} translation mismatch: got {translation}, expected {expected}")


def _check_local_translation(
    stage: Usd.Stage,
    path: str,
    expected: tuple[float, float, float],
    errors: list[str],
) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        return
    value = prim.GetAttribute("xformOp:translate").Get()
    if value is None:
        errors.append(f"{path} missing xformOp:translate")
        return
    actual = tuple(float(value[i]) for i in range(3))
    if any(abs(actual[i] - expected[i]) > 1e-6 for i in range(3)):
        errors.append(f"{path} local translation mismatch: got {actual}, expected {expected}")


def _check_float_attr(
    stage: Usd.Stage,
    path: str,
    attr_name: str,
    expected: float,
    errors: list[str],
) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        return
    value = prim.GetAttribute(attr_name).Get()
    if value is None:
        errors.append(f"{path} missing {attr_name}")
        return
    if abs(float(value) - expected) > 1e-6:
        errors.append(f"{path} {attr_name} mismatch: got {value}, expected {expected}")


def _check_material(
    stage: Usd.Stage,
    path: str,
    expected: tuple[float, float, float],
    errors: list[str],
) -> None:
    prim = stage.GetPrimAtPath(path)
    if not prim:
        return
    api = UsdPhysics.MaterialAPI(prim)
    values = (
        api.GetStaticFrictionAttr().Get(),
        api.GetDynamicFrictionAttr().Get(),
        api.GetRestitutionAttr().Get(),
    )
    if any(abs(float(values[i]) - expected[i]) > 1e-6 for i in range(3)):
        errors.append(f"{path} material mismatch: got {values}, expected {expected}")


def _check_binding(stage: Usd.Stage, prim_path: str, material_path: str, errors: list[str]) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        return
    material = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterial()
    if not material or str(material.GetPath()) != material_path:
        errors.append(f"{prim_path} material binding mismatch: expected {material_path}")


if __name__ == "__main__":
    raise SystemExit(main())
