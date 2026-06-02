### Scene Info — franka_scene.usd

**Robot Franka Panda**

- Path: `/World/Franka`
- Position: `(0.0, 0.0, 0.0)` — origin
- Initial joint angles (home): `[0, -π/4, 0, -3π/4, 0, π/2, π/4]`

**Pen**

- Path: `/World/Franka/panda_hand/Pen`
- Position relative to panda_hand: `(0.0, 0.0, 0.05)`
- Tip position (approx): `(0.3, 0.0, 0.4)` in world frame
- Shape: Cylinder — radius 5mm, height 5cm

**Paper**

- Path: `/World/Paper`
- Position: `(0.5, 0.0, 0.21)` in world frame
- Size: `30cm x 30cm`
- Drawing surface Z: `0.211` ← pen tip must reach this height to draw

**Table**

- Path: `/World/Table`
- Position: `(0.5, 0.0, 0.1)`
- Size: `60cm x 60cm x 20cm`

---

### Key info for controller:

- To draw, pen tip must reach **Z = 0.211**
- Paper center is at **X=0.5, Y=0.0**
- Drawing area: **X: 0.35-0.65, Y: -0.15 to 0.15**