# Isaac Scene Notes

Fill this backend after the final USD scene is ready.

Required values:

- USD path
- Franka articulation prim path
- end-effector frame name or body index
- board prim path
- pen-tip frame name or measured `T_ee_tip`
- contact sensor path and force sign convention
- command API for joint position targets
- command API for torque control

The core package must stay free of `omni.*`, `isaacsim.*`, and `isaaclab.*`
imports. Put those imports inside the future Isaac backend only.
