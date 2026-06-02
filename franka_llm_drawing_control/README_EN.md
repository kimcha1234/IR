# Franka LLM Drawing Control

Korean version: [README.md](README.md)

This repository contains the downstream robot-control side of the LLM drawing
project. The existing `Unit_Action_Langchain-main` folder remains the LLM
planner layer. It converts a natural-language drawing command into board-frame
primitive action JSON.

This package starts from that JSON and provides both an offline robot-control
core and a minimal JADE execution layer that can run hover tracing in Isaac Lab
with the cleaned local USD scene.

- Parse a `DrawingPlan` JSON produced by the LLM planner
- Sample line, arc, pen-down, and pen-up primitives into Cartesian pen-tip poses
- Transform board-frame poses into robot base-frame poses
- Compensate the fixed offset between the end-effector frame and the real pen tip
- Run a pure Python damped least-squares Differential IK fallback
- Run JADE: Jacobian-aware stroke validation, adaptive damping/speed policy,
  and planned-vs-actual logging
- Run pure math hybrid position-force controller calculations
- Use a mock Franka backend for offline tests and examples
- Provide a minimal Isaac Lab runner and Franka backend

The important point is that the LLM output is not a robot trajectory. It is a
symbolic drawing primitive plan. This package converts that plan into desired
Cartesian poses and controller commands.

## Current Scope

Implemented now:

- Offline core library under `franka_llm_drawing/`
- JADE validation/policy/execution layer under `franka_llm_drawing/jade/`
- Circle, square, and letter A plans under `examples/sample_plans/`
- `configs/frames.yaml`, `configs/jade.yaml`, and `configs/isaac_scene.yaml`
- pytest tests for transforms, samplers, controllers, and JADE metrics
- Mock robot backend for controller math and command wiring
- `examples/run_jade_offline.py` for Isaac-free JADE validation
- `examples/run_jade_isaac.py` for Isaac Lab hover tracing

Intentionally not implemented yet:

- Contact sensor paths
- Real force/contact drawing execution
- ROS2 or real Franka robot integration

The first Isaac demo is hover tracing, not contact drawing. The pen follows the
path about 2 cm above the paper. Contact and force control should be added only
after this path tracking is stable.

## If You Are New To Isaac Sim

This project first validates the offline core, then runs the same trajectory in
Isaac Lab. You can run the offline tests without Isaac Sim. If Isaac Sim is
installed correctly, `run_jade_isaac.py` opens the Franka USD and checks that
the pen follows the trajectory.

The key terms are:

- USD scene:
  The 3D scene file used by Isaac Sim. It will contain the robot, drawing board,
  pen, lights, cameras, and other scene objects.
- prim path:
  A string address for an object inside the USD scene, such as `/World/Franka`.
  This project uses `/World/Franka`, `/World/Paper`, and `/World/BoardFrame`.
- articulation:
  A jointed robot model such as Franka. The Isaac backend will read joint state,
  Jacobians, and command interfaces from this object.
- backend:
  The adapter between the core controller and the simulator. For now,
  `MockFrankaBackend` imitates the interface. Later, an Isaac backend will
  implement the same interface using real Isaac APIs.
- frame:
  A coordinate system. The LLM plan is expressed in the board frame, while robot
  control is usually computed in the robot base frame.

The intended development order is `LLM JSON -> offline JADE validation -> Isaac
hover tracing -> contact drawing`. Even if you are new to Isaac Sim, you can
understand the pipeline first through the offline tests and `run_jade_offline.py`.

## Layout

The existing team GitHub repository already has `assets/`, `examples/`,
`robot_drawing_planner/`, `scripts/`, `tests/`, `README.md`, and
`pyproject.toml`. This control code is meant to be added as one new top-level
folder named `franka_llm_drawing_control/` without overwriting those files.

```text
Unit_Action_Langchain/                  Existing GitHub repo root
├── assets/                             Existing team folder
├── examples/                           Existing LLM examples
├── robot_drawing_planner/              Existing LLM planner
├── scripts/                            Existing scripts
├── tests/                              Existing LLM tests
├── README.md                           Existing README
├── pyproject.toml                      Existing pyproject
└── franka_llm_drawing_control/          This folder
    ├── README.md
    ├── README_EN.md
    ├── pyproject.toml
    ├── configs/
    ├── examples/
    ├── franka_llm_drawing/
    └── offline_tests/
```

## Code Structure

The project separates the LLM planner and the robot-control code by role.

```text
Existing robot_drawing_planner/ in the GitHub repo:
    LLM planner that converts natural-language commands into DrawingPlan JSON.
    The existing planner does not handle IK, Jacobians, joint commands, or Isaac
    execution.

franka_llm_drawing/llm_bridge/
    Loads DrawingPlan JSON and converts the action list into internal dataclasses.
    It does not call OpenAI or LangChain.

franka_llm_drawing/trajectory/
    Converts move_to_start, pen_down, draw_line, draw_arc, and pen_up actions
    into time-sampled board-frame pen-tip poses. Lines and arcs use cubic or
    quintic time scaling.

franka_llm_drawing/frames/
    Converts board-frame poses to robot base-frame poses and computes the
    desired end-effector pose from the desired pen-tip pose.

franka_llm_drawing/controllers/
    Provides Differential IK fallback, pose error calculation, nullspace posture
    control, and hybrid position-force controller math. These modules are
    NumPy-based and do not depend on Isaac.

franka_llm_drawing/jade/
    Computes Jacobian singular values, condition number, manipulability, and
    joint-limit margin. It produces OK/WARNING/FAIL reports per stroke and
    increases DLS damping or reduces speed scale in risky regions.

franka_llm_drawing/robot/
    Defines the backend interface and provides MockFrankaBackend. The mock is
    not a real Franka model. It is a shape-correct mock for tests and controller
    wiring.

franka_llm_drawing/sim/
    Contains the minimal Isaac Lab Franka backend. The backend commands only
    the 7 arm joints, `panda_joint1` through `panda_joint7`. Finger joints may
    exist in the USD articulation, but they are not JADE command targets.

franka_llm_drawing/evaluation/
    Provides metrics and logging for desired path, actual path, force, singular
    values, and IK success rate.
```

Short code flow:

```text
DrawingPlan JSON
-> llm_bridge
-> trajectory sampler
-> frame transform + pen-tip offset compensation
-> controller
-> robot backend
-> evaluation/logging
```

## Run Tests

Run this from the repository root.

```bash
pytest -q
```

The pytest configuration in this upload folder limits discovery to
`offline_tests/`. The folder is intentionally not named `tests/` so it does not
collide with the existing GitHub repository's LLM planner tests.

## Run Offline Plan Sampling

```bash
python examples/run_offline_plan_sampling.py --plan examples/sample_plan_circle.json
```

This script loads a plan, samples board-frame pose targets, applies placeholder
`T_base_board` and `T_ee_tip` transforms, and prints a trajectory summary.

## Run The Mock Control Loop

```bash
python examples/run_mock_control_loop.py --plan examples/sample_plan_circle.json
```

This script uses the mock backend and the pure Python Differential IK fallback.
It checks software wiring and controller math only. It does not simulate real
Franka kinematics or contact physics.

## Run JADE Offline

This checks parsing, sampling, frame transforms, Jacobian-aware validation,
adaptive policy selection, and planned-vs-actual mock logging without Isaac Sim.

```bash
python examples/run_jade_offline.py --plan examples/sample_plans/circle.json --mode hover --out outputs/jade_offline_circle
```

Outputs:

```text
outputs/jade_offline_circle/
├── sampled_trajectory.csv
├── jacobian_metrics.csv
├── feasibility_report.json
├── execution_log.csv
├── summary.json
├── planned_vs_actual_xy.svg
├── tracking_error.svg
├── condition_number.svg
└── adaptive_policy.svg
```

## Run Isaac Hover Tracing

Run this only in a working Isaac Sim / Isaac Lab environment.

```bash
cd /home/kimchangyeol/IsaacLab
./isaaclab.sh -p /home/kimchangyeol/IsaacLab/IR/franka_llm_drawing_control/examples/run_jade_isaac.py \
  --mode hover \
  --plan /home/kimchangyeol/IsaacLab/IR/franka_llm_drawing_control/examples/sample_plans/circle.json \
  --path-speed-scale 0.35 \
  --out /home/kimchangyeol/IsaacLab/IR/franka_llm_drawing_control/outputs/jade_isaac_circle
```

`--path-speed-scale` scales every speed in the plan by the same ratio. The
current MVP uses closed-loop Differential IK without feed-forward, so `0.3~0.5`
is a better range for the first trajectory-tracking checks.

Quick headless smoke test:

```bash
cd /home/kimchangyeol/IsaacLab
./isaaclab.sh -p /home/kimchangyeol/IsaacLab/IR/franka_llm_drawing_control/examples/run_jade_isaac.py \
  --headless \
  --path-speed-scale 0.35 \
  --max-samples 20 \
  --settle-steps 5 \
  --hold-steps 2
```

The backend validates `configs/isaac_scene.yaml`. The command target list must
contain only:

```text
panda_joint1, panda_joint2, panda_joint3, panda_joint4,
panda_joint5, panda_joint6, panda_joint7
```

## Initial Posture Calibration

Use this command when you want the robot to start with the pen already hovering
above the paper and roughly aligned with the paper normal. This script also
commands only `panda_joint1` through `panda_joint7`.

```bash
cd /home/kimchangyeol/IsaacLab
./isaaclab.sh -p /home/kimchangyeol/IsaacLab/IR/franka_llm_drawing_control/examples/calibrate_initial_pose.py \
  --headless \
  --plan /home/kimchangyeol/IsaacLab/IR/franka_llm_drawing_control/examples/sample_plans/circle.json \
  --clearance-m 0.03 \
  --out /home/kimchangyeol/IsaacLab/IR/franka_llm_drawing_control/outputs/initial_pose_calibration
```

The result is written to `best_initial_pose.json` and
`initial_pose_candidates.csv`. Add `--write-scene-config` to the same command if
you want to update `configs/isaac_scene.yaml` with the best verified posture.

## USD Scene Draft

The cleaned scene draft is:

```text
usd/franka_drawing_scene_clean.usda
```

This file is intended to replace the earlier draft files
`usd/franka_scene.usd` and `usd/franka_scene(1).usd`.

- Franka is referenced from `usd/assets/franka/franka_panda.usd`, not from an
  external URL.
- The table is `0.60 m x 0.60 m x 0.10 m`, with top surface at `z = 0.20 m`.
- The paper is `0.30 m x 0.30 m x 0.002 m`, with drawing surface at
  `z = 0.202 m`.
- `/World/BoardFrame` is placed at the paper surface center `(0.5, 0.0, 0.202)`.
- The pen is attached under
  `/World/Franka/panda_link7/panda_link8/panda_hand/PenTool`.
- The controller should track `PenTool/PenTipFrame` as the pen-tip frame.
- A lighting rig under `/World/Lights` adds dome, overhead softbox, key, and
  fill lights so the robot, table, paper, and pen motion are visible.
- Initial friction values are moderate for easier contact-control debugging:
  Paper/PenTip use `static=0.55`, `dynamic=0.40`; Table uses `static=0.60`,
  `dynamic=0.45`.

The gripper finger links are not fully deleted from the asset. Removing them
directly can break the Franka articulation because the finger joints still
reference those links. For this first version, the finger visuals are hidden,
finger collisions are disabled, and the pen is attached to `panda_hand`. If a
true no-gripper Franka asset is required, create a separate custom Franka USD
asset where the finger links and finger joints are removed together.

## How Robot Execution Will Work

There is no real robot execution yet because the USD scene and Isaac backend are
not ready. The interface and mock backend are prepared so the same flow can be
connected later.

Once a real robot backend or Isaac Sim backend exists, the runtime flow will be:

```text
1. Run the LLM planner
   Convert a natural-language command into DrawingPlan JSON.

2. Load the DrawingPlan
   llm_bridge reads the JSON action list.

3. Generate a Cartesian trajectory
   The trajectory sampler converts move_to_start, pen_down, draw_line, draw_arc,
   and pen_up into board-frame pen-tip poses at dt intervals.

4. Apply frame transforms
   T_base_board converts board-frame pen-tip poses into base-frame pen-tip poses.

5. Compensate the pen-tip offset
   T_ee_tip converts the desired pen-tip pose into a desired end-effector pose.

6. Read robot state
   The backend reads q, q_dot, end-effector pose, Jacobian, gravity torque, and
   contact force.

7. Compute controller output
   The default path uses Differential IK to compute q_target for the desired
   end-effector pose. The contact-aware path uses the hybrid position-force
   controller to compute task wrench and joint torque command.

8. Send commands
   The backend sends q_target or tau_cmd to Isaac Sim or the real robot API.

9. Step simulation or robot control
   The backend advances one timestep and repeats the same process for the next
   desired pose.

10. Log and evaluate
    Desired path, actual path, force, torque, singular values, and condition
    numbers are logged and metrics such as RMSE are computed.
```

In the current mock run, `MockFrankaBackend` imitates steps 6 through 9. This
does not mean a real robot moved. It only verifies that controller inputs,
controller outputs, and the software pipeline have compatible shapes.

When the Isaac backend is finished, most changes should stay inside the backend.
The `trajectory`, `frames`, `controllers`, and `evaluation` modules should be
reused. The methods defined in `robot/interfaces.py` should be connected to the
real Isaac articulation, Jacobian, contact sensor, and command APIs.

In Isaac Sim, the mapping will roughly be:

```text
DrawingPlan JSON             Input loaded by Python
T_base_board                 Board pose from the USD scene or config file
T_ee_tip                     Fixed offset from Franka wrist frame to pen tip
RobotBackend.get_state()     Read q, q_dot, and end-effector pose from articulation
get_end_effector_jacobian()  Read Jacobian from Isaac / Isaac Lab
send_joint_position_target() Write q_target to the articulation controller
send_joint_torque_command()  Write tau_cmd when torque control is used
step()                       Advance Isaac simulation by one timestep
```

For a first Isaac Sim integration, start with Differential IK and joint position
targets. After contact sensor values are stable, connect the hybrid
position-force controller.

## Course Concept Mapping

- Homogeneous transform:
  `T_base_tip = T_base_board @ T_board_tip`,
  `T_base_ee = T_base_tip @ inv(T_ee_tip)`.
- Trajectory planning:
  Line and arc primitives are time-sampled with cubic time scaling.
- Jacobian-based Differential IK:
  `delta_q = J.T @ inv(J @ J.T + lambda^2 I) @ delta_x`.
- Singularity handling:
  Controller diagnostics return Jacobian singular values and condition numbers.
- Force / torque relation:
  The hybrid controller uses `tau = J.T @ wrench`.
- Dynamics hook:
  Gravity torque can be added when the real backend provides it.

## Later Isaac Integration

When the final USD is ready, fill in the Isaac backend with:

- `robot_prim_path`
- `ee_frame_name`
- `board_prim_path`
- `pen_tip_frame_name` or measured `T_ee_tip`
- contact sensor path
- APIs to read `q`, `q_dot`, end-effector pose, Jacobian, and force
- APIs to write joint position targets or torque commands
- simulation stepping

Core modules should remain free of Isaac dependencies.
