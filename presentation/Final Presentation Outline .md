# Final Presentation Outline

---

# LLM-Assisted Robot Shape Drawing

## Final Presentation (12 Slides)

---

# PART 1 — LLM & PLANNING (최준열)

---

## Slide 1 – Title Slide

**LLM-Assisted Robot Shape Drawing**

Final Project Presentation

Team Members:

- 최준열
- 김찬결
- Angel Chavez

Sungkyunkwan University

Intelligent Robotics

---

## Slide 2 – Motivation & Problem Statement

### Problem

Traditional robot programming requires technical expertise.

### Goal

Enable a user to simply type:

> "Draw a circle"
> 

and allow the robot to automatically execute the drawing task.

### Challenge

Bridge the gap between:

Natural Language → Robot Motion

---

## Slide 3 – System Architecture

Overall pipeline:

```
User Command
      ↓
Large Language Model
      ↓
Primitive Action Sequence
      ↓
Trajectory Generator
      ↓
Robot Controller
      ↓
Isaac Sim Execution
```

Explain how the LLM acts as a planner while robotics algorithms handle execution.

---

## Slide 4 – LLM Planner & Unit Actions

Example:

User Input:

> Draw a circle
> 

Generated Plan:

```
move_to_start()
align_pen_orientation()
pen_down()
draw_arc()
pen_up()
```

Introduce the primitive action framework.

---

## Slide 5 – Repository Structure & Software Framework

Main Components:

- Unit_Action_Langchain
- LLM Planner
- Prompt Templates
- Action Parser
- LLM–Isaac Bridge
- Test Framework

Show GitHub structure.

This is a good slide for screenshots from the repository.

---

# PART 2 — ROBOTICS & CONTROL (Angel)

---

## Slide 6 – Robot Modeling and Simulation Environment

### Platform

NVIDIA Isaac Sim

### Robot

Franka Emika Panda

### Environment

- Drawing board
- Pen end-effector
- Workspace definition

Include screenshot of Isaac Sim.

---

## Slide 7 – Primitive Actions to Robot Motion

Once the action sequence is generated:

```
draw_line
draw_arc
pen_down
pen_up
```

the robot must convert them into executable trajectories.

### Implemented Components

- Trajectory Generation
- Motion Planning
- Workspace Constraints

---

## Slide 8 – Kinematics

### Forward Kinematics

Compute end-effector position:

[

x=f(q)

]

### Inverse Kinematics

Compute joint angles:

[

q=f^{-1}(x)

]

Purpose:

Transform Cartesian drawing points into robot joint configurations.

This is one of your major contributions according to the proposal.

---

## Slide 9 – Jacobian-Based Motion Control

Jacobian relationship:

[

\dot{x}=J(q)\dot{q}

]

### Why Use the Jacobian?

- Smooth motion generation
- Differential inverse kinematics
- Continuous trajectory tracking

### Additional Analysis

- Jacobian Condition Number
- Singularities
- IK Feasibility

---

## Slide 10 – Hybrid Position–Force Control

### Position Control

Track the desired drawing trajectory in:

- X direction
- Y direction

### Force Control

Maintain stable contact force in:

- Z direction

### Objective

Ensure the pen remains in contact with the board while drawing.

Include controller diagram here.

---

## Slide 11 – Results and Performance Evaluation

Metrics from the proposal:

### Position Tracking

- XY RMSE
- Maximum XY Error

### Contact Performance

- Contact Maintenance Ratio
- Force RMSE

### Kinematic Performance

- Jacobian Condition Number
- IK Failure Rate

### Demonstrations

Representative Isaac Sim videos:

- Circle
- Square
- Triangle
- Star
- Full-scene star demo

Quantitative result graphs cover all six demo shapes:

- Circle
- Square
- Triangle
- Letter A
- House
- Star

Top-view drawing behavior is shown using planned-vs-actual XY trajectory graphs.

---

## Slide 12 – Conclusions & Future Work

### Achievements

✔ Natural Language → Robot Motion

✔ Primitive Action Framework

✔ Trajectory Generation

✔ Inverse Kinematics

✔ Jacobian-Based Control

✔ Isaac Sim Validation

### Future Work

- More complex drawings
- Real robot implementation
- Improved force control
- Vision-based feedback
- Adaptive planning using LLMs

---
