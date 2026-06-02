# Franka Drawing Scene Clean Draft

Scene file:

```text
usd/franka_drawing_scene_clean.usda
```

This scene is a cleaned draft intended to replace the two earlier files:

```text
usd/franka_scene.usd
usd/franka_scene(1).usd
```

## Main Decisions

- The Franka asset is referenced from the local project folder:
  `usd/assets/franka/franka_panda.usd`
- The external Omniverse URL reference was removed.
- The gripper fingers are not deleted from the articulation. Instead, their
  visuals are hidden and their collision geometry is disabled.
- A pen tool is attached under:
  `/World/Franka/panda_link7/panda_link8/panda_hand/PenTool`
- The controller target frame is:
  `/World/Franka/panda_link7/panda_link8/panda_hand/PenTool/PenTipFrame`

## Geometry

Table:

```text
Path: /World/Table
Size: 0.60 m x 0.60 m x 0.10 m
Center: (0.5, 0.0, 0.15)
Top surface z: 0.20 m
```

Paper:

```text
Path: /World/Paper
Size: 0.30 m x 0.30 m x 0.002 m
Center: (0.5, 0.0, 0.201)
Drawing surface z: 0.202 m
Drawing x range: 0.35 to 0.65
Drawing y range: -0.15 to 0.15
```

Board frame:

```text
Path: /World/BoardFrame
Position: (0.5, 0.0, 0.202)
Convention: x/y are paper plane axes, +z is board normal
```

Pen:

```text
Path: /World/Franka/panda_link7/panda_link8/panda_hand/PenTool/PenBody
Radius: 0.005 m
Body length: 0.094 m
Axis: local +Z from panda_hand
```

The pen-tip collision sphere has radius `0.003 m`. Its center is at local
translation `(0.0, 0.0, 0.097)`, so the lowest contact point coincides with
`PenTipFrame` at local translation `(0.0, 0.0, 0.10)`.

Pen tip frame:

```text
Path: /World/Franka/panda_link7/panda_link8/panda_hand/PenTool/PenTipFrame
Local pose relative to panda_hand:
  translation: (0.0, 0.0, 0.10)
  rotation: identity
```

Isaac Lab exposes `panda_link7`, not `panda_hand`, as the rigid body used for
the Jacobian in this USD. Therefore the controller config uses `panda_link7` as
the end-effector body and maps from that body to the pen tip:

```text
ee_frame_name: panda_link7
T_ee_tip translation: [0.0, 0.0, 0.207000001]
T_ee_tip rotation:
  [[ 0.707106677, 0.707106886, 0.0],
   [-0.707106886, 0.707106677, 0.0],
   [ 0.0,         0.0,         1.0]]
```

## Contact Materials

Physics materials are authored in `/World/PhysicsMaterials` and bound to the
colliders.

```text
PaperMaterial:
  staticFriction: 0.55
  dynamicFriction: 0.40
  restitution: 0.0

PenTipMaterial:
  staticFriction: 0.55
  dynamicFriction: 0.40
  restitution: 0.0

TableMaterial:
  staticFriction: 0.60
  dynamicFriction: 0.45
  restitution: 0.0
```

These are moderate initial simulation values chosen to make early contact
control debugging easier. Stable drawing will still require contact sensor
validation, normal-force controller tuning, and solver timestep checks.

## Lighting

The scene includes a small lighting rig under `/World/Lights`:

```text
/World/Lights/AmbientDome
/World/Lights/OverheadSoftbox
/World/Lights/KeyLight
/World/Lights/FrontFill
```

The overhead softbox is the main light for seeing the table, paper, pen, and
robot motion. The dome and fill lights reduce harsh shadows during debugging.

## Control Caveat

The referenced Franka asset still contains two finger joints:

```text
panda_finger_joint1
panda_finger_joint2
```

The finger visuals are hidden and finger collisions are disabled, but the
articulation may still expose 9 joints instead of only the 7 arm joints. The
Isaac backend should explicitly select and command only:

```text
panda_joint1 ... panda_joint7
```

Finger joints should be held fixed or ignored by the arm controller.

## Notes

The gripper was not fully removed because deleting finger links and finger
joints directly from a referenced Franka articulation can break the robot model.
For the first Isaac integration, it is safer to keep the articulation intact,
hide finger visuals, disable finger collisions, and attach the pen to
`panda_hand`.

If a final report or production demo requires a Franka model with no gripper at
all, create a separate custom Franka USD asset where the finger links and finger
joints are removed consistently from the articulation.
