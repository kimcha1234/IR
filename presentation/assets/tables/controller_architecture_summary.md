# Controller and Architecture Summary

| Slide | Component | Implemented method | Message for presentation |
|---|---|---|---|
| 3 | System pipeline | Text -> DrawingPlan -> trajectory -> DLS IK -> Isaac Sim | LLM/planning and robotics execution are separated by a schema. |
| 4 | Unit actions | move_to_start, pen_down, draw_line, draw_arc, pen_up | Complex drawings are decomposed into simple primitives. |
| 7 | Trajectory generation | Cartesian primitive sampling with quintic timing | The robot tracks smooth time-parameterized targets. |
| 8 | Kinematics | FK for pen-tip pose, Jacobian for local motion | End-effector motion is computed from robot joint states. |
| 9 | IK/control | Damped Least Squares differential IK | Handles near-singular configurations more robustly than plain inverse. |
| 9 | JADE diagnostics | condition number, damping lambda, joint-limit margin | Robotics quality is monitored during execution. |
| 10 | Hybrid control | XY position tracking + normal-force admittance | Pen follows the drawing while maintaining paper contact. |
| 11 | Evaluation | tracking error, contact force, orientation error | Performance is shown with quantitative plots and summary metrics. |
