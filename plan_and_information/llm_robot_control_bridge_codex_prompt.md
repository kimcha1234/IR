# LLM Drawing Planner와 로봇 제어기 연결 정리

이 문서는 `Unit_Action_Langchain-main` 프로젝트의 LLM drawing planner 코드를 로봇 제어 코드와 어떻게 연결해야 하는지 정리한 문서이다. 나중에 Codex 또는 다른 코드 생성 도구에 그대로 프롬프트로 넣어 사용할 수 있도록, 설명과 구현 요구사항을 함께 정리했다.

---

## 1. 가장 중요한 결론

현재 LLM 코드는 **trajectory를 직접 만들어서 로봇에 주는 코드가 아니다.**

이 코드는 자연어 명령을 받아서 다음과 같은 **symbolic drawing primitive action JSON**으로 바꾸는 상위 작업 계획기이다.

```text
사용자 명령: "중앙에 반지름 5cm짜리 원을 그려줘"
        ↓
LLM / LangChain tool calling
        ↓
DrawingPlan JSON
        ↓
move_to_start, pen_down, draw_arc, pen_up 등의 primitive action
```

즉, 이 코드가 출력하는 것은 로봇 제어기에 바로 넣는 다음 값들이 아니다.

```text
q_d(t)
qdot_d(t)
qddot_d(t)
torque command
joint position command
Isaac Sim command
```

대신 이 코드가 출력하는 것은 다음에 가깝다.

```text
어디서 시작할지
펜을 언제 내릴지
어떤 직선 또는 원호를 그릴지
펜을 언제 올릴지
```

따라서 이 LLM 코드는 **trajectory generator 자체**라기보다, **trajectory generator의 입력을 만들어주는 상위 planner**라고 보는 것이 정확하다.

로봇 제어 쪽에서는 이 JSON을 받아서 직접 다음 값을 생성해야 한다.

```text
펜 끝의 Cartesian desired trajectory:
    x_d(t), y_d(t), z_d(t)

또는 end-effector desired pose:
    T_base_ee,d(t)

그리고 IK 또는 Jacobian 기반 제어를 통해:
    q_d(t), qdot_d(t), torque command
```

---

## 2. 현재 LLM 코드의 역할

ZIP 파일 내부의 주요 구조는 다음과 같다.

```text
Unit_Action_Langchain-main/
├── README.md
├── examples/
├── robot_drawing_planner/
│   ├── action_tools.py
│   ├── agent_planner.py
│   ├── config.py
│   ├── geometry.py
│   ├── plan_state.py
│   ├── planner.py
│   ├── schemas.py
│   ├── validators.py
│   └── visualization.py
└── tests/
```

핵심 흐름은 다음과 같다.

```text
natural language command
→ ChatOpenAI bound to Unit Action tools
→ multi-step tool calls
→ PlanBuilder state
→ primitive action JSON
```

이 프로젝트는 명시적으로 다음을 하지 않는다.

```text
IK 계산
FK 계산
Jacobian 계산
joint angle 계산
joint command 생성
torque command 생성
Isaac Sim 실행
trajectory sample 생성
실제 로봇 실행 결과 생성
```

따라서 현재 코드는 로봇 제어 전체 중에서 가장 위에 있는 **LLM-assisted task planner**이다.

---

## 3. 주요 파일별 역할

| 파일 | 역할 |
|---|---|
| `README.md` | 프로젝트 범위 설명. 이 코드는 planner layer만 담당하고, IK/Jacobian/trajectory/Isaac Sim 실행은 downstream module의 책임이라고 명시한다. |
| `schemas.py` | 출력 JSON 구조 정의. `Point2D`, `Point3D`, `LineStroke`, `ArcStroke`, `PrimitiveAction`, `DrawingPlan` 등이 있다. |
| `agent_planner.py` | 실제 agentic LLM planner. LLM이 tool call을 순서대로 호출하게 한다. |
| `action_tools.py` | LLM에게 제공되는 tool 정의. `move_to_start`, `pen_down`, `draw_line_to`, `draw_arc`, `pen_up`, `check_plan`, `finish_plan` 등이 있다. |
| `plan_state.py` | tool call이 누적되는 상태 관리자. 현재 위치, pen 상태, stroke/action 목록을 관리한다. |
| `planner.py` | template baseline. 원, 네모, 세모, 글자 등을 deterministic하게 plan으로 변환한다. |
| `geometry.py` | 사각형 꼭짓점, 삼각형 꼭짓점, 원호 등 기하 계산을 담당한다. |
| `validators.py` | board boundary 검사, low-level robot field 금지 검사 등을 담당한다. |
| `visualization.py` | 계획된 board-frame path를 matplotlib으로 시각화한다. |
| `cli.py` | 명령줄 실행 인터페이스이다. |
| `demo_app.py` | Streamlit GUI 데모이다. |

---

## 4. DrawingPlan JSON의 의미

LLM 코드의 핵심 출력은 `DrawingPlan`이다.

간단한 원 명령은 대략 다음과 같은 구조로 표현된다.

```json
{
  "schema_version": "1.0",
  "source_command": "중앙에 반지름 5cm짜리 원을 그려줘",
  "goal": {
    "shape_type": "circle",
    "center": {"x": 0.0, "y": 0.0, "unit": "m"},
    "radius_m": 0.05,
    "orientation_rad": 0.0,
    "frame": "board"
  },
  "strokes": [
    {
      "type": "arc",
      "stroke_id": "stroke_001",
      "center": {"x": 0.0, "y": 0.0, "unit": "m"},
      "radius_m": 0.05,
      "start_angle_rad": 0.0,
      "end_angle_rad": 6.283185307179586,
      "direction": "ccw"
    }
  ],
  "actions": [
    {
      "name": "move_to_start",
      "frame": "board",
      "params": {
        "target": {"x": 0.05, "y": 0.0, "z": 0.03, "unit": "m"},
        "hover_height_m": 0.03
      }
    },
    {
      "name": "align_pen_orientation",
      "frame": "board",
      "params": {
        "mode": "normal_to_board",
        "board_normal_axis": "+z",
        "pen_axis_target": "-z"
      }
    },
    {
      "name": "pen_down",
      "frame": "board",
      "params": {
        "target": {"x": 0.05, "y": 0.0, "z": 0.0, "unit": "m"},
        "approach_axis": "-z",
        "speed_m_s": 0.01
      }
    },
    {
      "name": "draw_arc",
      "frame": "board",
      "stroke_id": "stroke_001",
      "params": {
        "center": {"x": 0.0, "y": 0.0, "z": 0.0, "unit": "m"},
        "radius_m": 0.05,
        "start_angle_rad": 0.0,
        "end_angle_rad": 6.283185307179586,
        "direction": "ccw",
        "speed_m_s": 0.03
      }
    },
    {
      "name": "pen_up",
      "frame": "board",
      "params": {
        "lift_height_m": 0.03,
        "speed_m_s": 0.02
      }
    }
  ]
}
```

중요한 점은 좌표가 전부 **board frame 기준 meter 단위**라는 것이다.

기본 board frame convention은 다음과 같이 보면 된다.

```text
board frame 원점: 보드 중심
x-y 평면: 그림 그리는 보드 평면
z = 0: 실제 drawing surface
z = 0.03 정도: pen hover height
```

---

## 5. Path와 trajectory의 차이

이 프로젝트를 이해할 때 가장 헷갈리기 쉬운 부분은 path와 trajectory의 차이이다.

### 5.1 Path

Path는 시간 정보가 없는 기하학적 경로이다.

```text
이 점에서 저 점까지 직선을 그려라.
중심이 여기고 반지름이 이만큼인 원호를 그려라.
```

예를 들어 line path는 다음처럼 표현된다.

```text
start = [x0, y0, z0]
end   = [x1, y1, z1]
```

arc path는 다음처럼 표현된다.

```text
center = [cx, cy, z]
radius = r
start_angle = theta0
end_angle = theta1
direction = ccw 또는 cw
```

### 5.2 Trajectory

Trajectory는 시간 정보까지 포함한 경로이다.

```text
t = 0.00 s → p_d = [x0, y0, z0]
t = 0.01 s → p_d = [x1, y1, z1]
t = 0.02 s → p_d = [x2, y2, z2]
...
```

그리고 제어에 따라 속도와 가속도도 필요할 수 있다.

```text
p_d(t)
pdot_d(t)
pddot_d(t)
```

또는 joint space에서는 다음이 필요할 수 있다.

```text
q_d(t)
qdot_d(t)
qddot_d(t)
```

현재 LLM 코드가 주는 것은 대부분 **path specification**이다. `speed_m_s` 같은 값은 들어 있지만, 실제로 `dt`마다 샘플링된 desired position을 만들어주지는 않는다.

따라서 downstream robot-control module에서 다음 변환이 필요하다.

```text
LLM action JSON
→ line/arc를 시간에 따라 sample
→ pen tip desired pose 생성
→ board frame에서 robot base frame으로 변환
→ pen tip offset 보정
→ IK 또는 Jacobian으로 joint command 생성
→ controller로 추종
```

---

## 6. 전체 연결 구조

팀원 코드와 로봇 제어 코드는 다음 구조로 연결하면 된다.

```text
[LLM planner]
자연어 명령
→ DrawingPlan JSON
        ↓
[Trajectory generator]
draw_line / draw_arc / pen_down / pen_up
→ p_tip,d(t), R_tip,d(t)
        ↓
[Frame transform]
board frame → robot base frame
        ↓
[Tool offset compensation]
pen tip pose → robot end-effector pose
        ↓
[IK or differential IK]
T_base_ee,d(t)
→ q_d(t), qdot_d(t)
        ↓
[Robot controller]
joint position / joint velocity / torque command
        ↓
[Simulation or robot]
Isaac Sim / PyBullet / Gazebo / 실제 로봇
```

한 문장으로 정리하면 다음과 같다.

```text
팀원 코드: 무엇을 그릴지 결정한다.
내 코드: 그걸 로봇이 실제로 어떻게 움직여서 그릴지 계산한다.
```

---

## 7. 각 primitive action을 로봇 제어 쪽에서 해석하는 방법

| LLM action | 로봇 제어 쪽 해석 |
|---|---|
| `move_to_start` | 펜을 든 상태로 목표점 위 hover 위치까지 이동한다. 보통 `z = hover_height_m`이다. |
| `align_pen_orientation` | 펜 축이 보드에 수직이 되도록 end-effector orientation을 설정한다. |
| `pen_down` | hover 위치에서 drawing surface `z = 0`까지 천천히 내려간다. |
| `draw_line` | start부터 end까지 직선 Cartesian trajectory를 생성한다. |
| `draw_arc` | center, radius, start/end angle, direction을 이용해 원호 Cartesian trajectory를 생성한다. |
| `pen_up` | drawing surface에서 다시 hover height까지 들어 올린다. |

---

## 8. Line primitive를 trajectory로 바꾸는 방법

`draw_line` action은 보통 다음 정보를 가진다.

```text
start = [x0, y0, z0]
end   = [x1, y1, z1]
speed_m_s = v
```

시간 없는 path는 다음처럼 쓸 수 있다.

```text
p(s) = start + s(end - start),  0 ≤ s ≤ 1
```

제어기에 넣으려면 시간 parameter가 필요하므로 `s = s(t)`를 정의한다.

가장 단순한 constant-speed 방식:

```text
length = ||end - start||
T = length / speed_m_s
s(t) = t / T
```

더 부드러운 움직임을 원하면 cubic time scaling을 사용할 수 있다.

```text
s(t) = 3(t/T)^2 - 2(t/T)^3
```

따라서 desired position은 다음과 같다.

```text
p_d(t) = start + s(t)(end - start)
```

속도와 가속도가 필요하면 다음을 계산한다.

```text
pdot_d(t)  = sdot(t)(end - start)
pddot_d(t) = sddot(t)(end - start)
```

---

## 9. Arc primitive를 trajectory로 바꾸는 방법

`draw_arc` action은 보통 다음 정보를 가진다.

```text
center = [cx, cy, z]
radius_m = r
start_angle_rad = theta0
end_angle_rad = theta1
direction = "ccw" 또는 "cw"
speed_m_s = v
```

원호의 길이는 다음과 같다.

```text
arc_length = radius * abs(delta_theta)
T = arc_length / speed_m_s
```

각도는 시간에 따라 다음처럼 만든다.

```text
theta(t) = theta0 + s(t) * delta_theta
```

여기서 direction이 `ccw`면 양의 방향, `cw`면 음의 방향으로 해석해야 한다.

```text
ccw: delta_theta = positive angular travel
cw : delta_theta = negative angular travel
```

Cartesian desired position은 다음과 같다.

```text
p_d(t) = [
    cx + r cos(theta(t)),
    cy + r sin(theta(t)),
    z
]
```

속도는 다음과 같다.

```text
pdot_d(t) = theta_dot(t) * [
    -r sin(theta(t)),
     r cos(theta(t)),
     0
]
```

가속도까지 필요하면 다음을 사용한다.

```text
pddot_d(t) = theta_ddot(t) * [
    -r sin(theta(t)),
     r cos(theta(t)),
     0
] + theta_dot(t)^2 * [
    -r cos(theta(t)),
    -r sin(theta(t)),
     0
]
```

---

## 10. Pen down과 pen up을 trajectory로 바꾸는 방법

`pen_down`은 현재 x-y 위치에서 z 방향으로 내려가는 motion이다.

```text
start = [x, y, hover_height]
end   = [x, y, drawing_z]
```

`pen_up`은 반대이다.

```text
start = [x, y, drawing_z]
end   = [x, y, hover_height]
```

이것도 line trajectory와 같은 방식으로 처리하면 된다.

```text
p_d(t) = start + s(t)(end - start)
```

다만 pen_down은 너무 빠르면 보드와 충돌하거나 펜이 눌릴 수 있으므로 낮은 속도를 사용해야 한다.

---

## 11. Desired position의 기준

로봇 제어에서 desired position은 두 종류로 나눌 수 있다.

```text
1. Cartesian desired position
   x_d(t), y_d(t), z_d(t)
   즉, 펜 끝 또는 end-effector가 가야 할 위치

2. Joint desired position
   q_d(t)
   즉, 각 관절이 가져야 할 각도
```

LLM output은 둘 중 어느 것도 완성된 형태로 주지 않는다. 하지만 가장 가까운 것은 **Cartesian desired path의 재료**이다.

따라서 로봇 제어 코드는 다음 흐름으로 만드는 것이 좋다.

```text
DrawingPlan JSON
→ pen tip Cartesian desired pose T_base_tip,d(t)
→ IK
→ q_d(t)
→ joint controller
```

---

## 12. Board frame에서 robot base frame으로 변환

LLM output은 `board` frame 기준이다. 하지만 로봇 제어는 보통 `robot base` frame 기준으로 계산한다.

따라서 반드시 다음 변환이 필요하다.

```text
p_base = T_base_board · p_board
```

동차좌표로 쓰면 다음과 같다.

```text
[ p_base ]   [              ] [ p_board ]
[   1    ] = [ T_base_board ] [   1     ]
```

예를 들어 보드가 로봇 base 기준으로 x 방향 0.50m, y 방향 0.00m, z 방향 0.20m 위치에 있고 기울어져 있지 않다면:

```python
T_base_board = [
    [1, 0, 0, 0.50],
    [0, 1, 0, 0.00],
    [0, 0, 1, 0.20],
    [0, 0, 0, 1.00],
]
```

실제 값은 시뮬레이션 또는 실제 로봇 환경에서 보드가 어디에 놓였는지에 따라 정해야 한다.

---

## 13. Pen tip offset 보정

로봇의 end-effector frame과 실제 펜 끝 frame은 보통 일치하지 않는다.

예를 들어 그리퍼에 펜을 잡고 있다면 다음 변환이 존재한다.

```text
T_ee_tip
```

이는 end-effector frame에서 pen tip frame까지의 고정 변환이다.

우리가 진짜 원하는 것은 펜 끝이 보드 위의 desired pose를 따라가는 것이다.

```text
T_base_tip,d(t)
```

하지만 로봇에 명령해야 하는 것은 end-effector pose이다.

```text
T_base_ee,d(t)
```

따라서 다음 관계를 사용한다.

```text
T_base_tip,d = T_base_ee,d · T_ee_tip
```

따라서 end-effector desired pose는 다음과 같다.

```text
T_base_ee,d = T_base_tip,d · inverse(T_ee_tip)
```

이 보정을 하지 않으면 로봇 손목 위치는 맞아도 실제 펜 끝은 엉뚱한 위치를 따라갈 수 있다.

---

## 14. Pen orientation 설정

그림 그리기 작업에서는 펜 끝 위치뿐 아니라 펜의 방향도 중요하다.

일반적으로 펜 축이 보드에 수직이 되도록 설정한다.

LLM action의 `align_pen_orientation`에는 다음 의미가 들어 있다.

```text
board_normal_axis: +z
pen_axis_target: -z
```

즉, 보드의 +z 방향이 위쪽이고, 펜의 끝 방향이 보드를 향해야 하므로 펜 축은 대략 board frame 기준 `-z` 방향을 향하게 한다.

구현에서는 다음 중 하나를 선택할 수 있다.

```text
1. orientation을 항상 고정한다.
2. 보드 frame의 normal을 기준으로 R_base_tip을 계산한다.
3. 로봇 모델의 end-effector convention에 맞춰 R_base_ee를 정한다.
```

처음 구현에서는 orientation을 고정하는 방식이 가장 쉽다.

---

## 15. IK와 제어기 연결

가장 단순한 구조는 다음과 같다.

```text
p_tip,d(t)
→ T_base_tip,d(t)
→ T_base_ee,d(t)
→ IK
→ q_d(t)
→ joint position controller
```

매 timestep마다 IK를 풀어서 원하는 관절각 `q_d`를 만들고, 로봇에 position command를 보내면 된다.

조금 더 제어 이론적으로 구현하려면 differential IK 또는 Jacobian 기반 제어를 쓸 수 있다.

속도 관계는 다음과 같다.

```text
x_dot = J(q) q_dot
```

이를 이용하면 다음과 같이 joint velocity command를 만들 수 있다.

```text
q_dot_cmd = J⁺(q) [ x_dot_d + K(x_d - x) ]
```

여기서 `J⁺`는 pseudo-inverse이다.

특이점 근처에서는 damped least squares를 사용하는 것이 좋다.

```text
J_dls⁺ = Jᵀ (J Jᵀ + λ² I)^-1
```

그리고 다음처럼 적분한다.

```text
q_cmd(t + dt) = q_cmd(t) + q_dot_cmd * dt
```

---

## 16. 제어기 선택

처음부터 torque control까지 구현할 필요는 없다. 단계적으로 구현하는 것이 좋다.

### 16.1 1단계: Joint position control

가장 쉬운 방식이다.

```text
controller input = q_d[k]
```

Isaac Sim이나 PyBullet의 기본 position controller를 사용할 수 있다면 이 방식으로 시작한다.

### 16.2 2단계: Joint velocity control

Differential IK로 `q_dot_cmd`를 만든 뒤 velocity command로 보낸다.

```text
controller input = q_dot_cmd[k]
```

### 16.3 3단계: Joint PD + gravity compensation

수업 내용과 연결하려면 다음 형태를 사용할 수 있다.

```text
τ = Kp(q_d - q) + Kd(qdot_d - qdot) + G(q)
```

### 16.4 4단계: Computed torque control

동역학까지 사용하면 다음 형태가 된다.

```text
τ = M(q)(qddot_d + Kd e_dot + Kp e) + C(q, qdot)qdot + G(q)
```

여기서:

```text
e     = q_d - q
e_dot = qdot_d - qdot
```

프로젝트 초기 구현에서는 1단계 또는 2단계로 충분하다.

---

## 17. 추천 구현 순서

처음부터 모든 것을 구현하지 말고 다음 순서로 진행한다.

### Step 1. DrawingPlan JSON parser 작성

입력 JSON에서 `actions`를 순서대로 읽는다.

```text
move_to_start
align_pen_orientation
pen_down
draw_line
draw_arc
pen_up
```

### Step 2. Board-frame Cartesian waypoint 생성

각 action을 작은 점들로 쪼갠다.

```text
p_board[0], p_board[1], ..., p_board[N]
```

예:

```text
line: 50개 점으로 샘플링
arc: 100개 점으로 샘플링
pen_down: 20개 점으로 샘플링
pen_up: 20개 점으로 샘플링
```

### Step 3. Board frame → base frame 변환

```text
p_base[k] = T_base_board · p_board[k]
```

### Step 4. Orientation 붙이기

펜이 보드에 수직이 되도록 고정 orientation을 붙인다.

```text
T_base_tip,d[k] = [R_base_tip,d, p_base[k]]
```

### Step 5. Pen tip offset 보정

```text
T_base_ee,d[k] = T_base_tip,d[k] · inverse(T_ee_tip)
```

### Step 6. IK 계산

각 desired pose에 대해 IK를 푼다.

```text
q_d[k] = IK(T_base_ee,d[k])
```

처음에는 현재 관절각과 가장 가까운 IK 해를 고르면 된다.

### Step 7. Controller로 추종

처음에는 joint position controller로 시작한다.

```text
command = q_d[k]
```

### Step 8. 결과 확인

다음 그래프를 그리면 프로젝트 결과로 좋다.

```text
desired x-y path vs actual x-y path
q_d(t) vs q(t)
position error over time
```

---

## 18. Codex에게 맡길 수 있는 구현 작업

아래 내용은 Codex에 그대로 넣을 수 있는 작업 지시 형태이다.

```text
You are working with a Python repository named Unit_Action_Langchain-main.
The existing package robot_drawing_planner converts natural-language drawing commands into a DrawingPlan JSON containing board-frame symbolic primitive actions.

Important: The existing LLM planner does not compute IK, FK, Jacobians, joint commands, torque commands, Isaac Sim commands, or sampled trajectories. Do not modify that scope unless absolutely necessary.

Your task is to implement a downstream robot-control bridge module that converts DrawingPlan actions into robot-executable Cartesian and joint-space trajectories.

Create a new module, for example:

robot_drawing_planner/robot_bridge/
    __init__.py
    trajectory_sampler.py
    frames.py
    tool_offset.py
    ik_interface.py
    controller_interface.py
    plan_to_trajectory.py

Requirements:

1. Read a DrawingPlan JSON or Python dict.
2. Iterate through plan.actions in order.
3. Support these action names:
   - move_to_start
   - align_pen_orientation
   - pen_down
   - draw_line
   - draw_arc
   - pen_up
4. Convert action parameters from board-frame geometry into sampled Cartesian pen-tip waypoints.
5. For line motions, implement p(s) = start + s(end - start).
6. For arc motions, implement p(theta) = center + radius * [cos(theta), sin(theta), 0].
7. Use speed_m_s and a control timestep dt to compute the number of samples.
8. Use smooth cubic time scaling by default: s(t) = 3(t/T)^2 - 2(t/T)^3.
9. Convert board-frame points to robot-base-frame points using T_base_board.
10. Attach a fixed pen orientation that makes the pen normal to the board.
11. Convert desired pen-tip pose to desired end-effector pose using:
    T_base_ee = T_base_tip @ inv(T_ee_tip)
12. Provide an IK interface but allow it to be a placeholder at first.
13. If no IK solver is available, return Cartesian trajectory only and clearly mark joint trajectory as unavailable.
14. Include unit tests for line sampling, arc sampling, frame transform, and tool offset transform.
15. Do not call OpenAI or LangChain from this bridge module.
16. Do not fabricate actual robot execution results.
17. Keep the output explicit and machine-readable.

Desired output data structures:

CartesianTrajectoryPoint:
    t: float
    p_base_tip: np.ndarray shape (3,)
    R_base_tip: np.ndarray shape (3, 3)
    T_base_tip: np.ndarray shape (4, 4)
    T_base_ee: np.ndarray shape (4, 4)
    pen_state: "up" or "down"
    source_action_name: str
    stroke_id: Optional[str]

JointTrajectoryPoint:
    t: float
    q: np.ndarray
    qdot: Optional[np.ndarray]
    source_cartesian_index: int

Main public function:

    convert_plan_to_cartesian_trajectory(
        plan: dict | DrawingPlan,
        T_base_board: np.ndarray,
        T_ee_tip: np.ndarray,
        dt: float = 0.01,
        default_speed_m_s: float = 0.03,
    ) -> list[CartesianTrajectoryPoint]

Optional public function:

    convert_cartesian_to_joint_trajectory(
        cart_traj: list[CartesianTrajectoryPoint],
        ik_solver: Callable,
        q_seed: np.ndarray,
    ) -> list[JointTrajectoryPoint]

Make the implementation robust to missing speed_m_s by using default_speed_m_s.
Make sure all coordinates are in meters.
```

---

## 19. Python 의사코드

아래는 실제 구현의 기본 구조이다.

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class CartesianTrajectoryPoint:
    t: float
    p_base_tip: np.ndarray
    R_base_tip: np.ndarray
    T_base_tip: np.ndarray
    T_base_ee: np.ndarray
    pen_state: Literal["up", "down"]
    source_action_name: str
    stroke_id: Optional[str] = None


def make_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    ph = np.array([p[0], p[1], p[2], 1.0])
    return (T @ ph)[:3]


def cubic_s(u: float) -> float:
    return 3.0 * u**2 - 2.0 * u**3


def sample_line(start, end, speed, dt):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    length = np.linalg.norm(end - start)
    if length < 1e-12:
        return [start]

    T = max(length / speed, dt)
    n = max(2, int(np.ceil(T / dt)) + 1)

    points = []
    for i in range(n):
        u = i / (n - 1)
        s = cubic_s(u)
        p = start + s * (end - start)
        points.append(p)
    return points


def sample_arc(center, radius, theta0, theta1, direction, speed, dt):
    center = np.asarray(center, dtype=float)

    raw_delta = theta1 - theta0
    if direction == "ccw":
        delta = raw_delta
        if delta < 0:
            delta += 2.0 * np.pi
    elif direction == "cw":
        delta = raw_delta
        if delta > 0:
            delta -= 2.0 * np.pi
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    arc_length = abs(radius * delta)
    T = max(arc_length / speed, dt)
    n = max(2, int(np.ceil(T / dt)) + 1)

    points = []
    for i in range(n):
        u = i / (n - 1)
        s = cubic_s(u)
        theta = theta0 + s * delta
        p = np.array([
            center[0] + radius * np.cos(theta),
            center[1] + radius * np.sin(theta),
            center[2],
        ])
        points.append(p)
    return points
```

---

## 20. 현재 프로젝트에서 성공적으로 보여줄 수 있는 결과물

프로젝트 결과물은 다음처럼 구성하면 좋다.

```text
입력:
    "중앙에 반지름 5cm 원을 그려줘"

LLM 출력:
    DrawingPlan JSON
    draw_arc primitive 포함

로봇 제어 모듈 출력:
    sampled Cartesian trajectory
    T_base_tip,d(t)
    T_base_ee,d(t)
    q_d(t) if IK is implemented

시뮬레이션 결과:
    로봇이 원을 그림

그래프:
    desired x-y path vs actual x-y path
    q_d(t) vs q(t)
    tracking error over time
```

처음에는 actual robot execution이 없어도 다음까지 구현하면 의미 있는 중간 결과가 된다.

```text
DrawingPlan JSON
→ board-frame sampled path
→ base-frame sampled path
→ pen-tip/end-effector desired pose sequence
→ matplotlib visualization
```

그 다음 단계로 IK와 simulator를 붙이면 된다.

---

## 21. 발표 또는 보고서에 쓸 수 있는 표현

다음 표현이 가장 정확하다.

```text
본 프로젝트에서 LLM planner는 자연어 drawing command를 board-frame primitive action plan으로 변환한다.
이 primitive plan은 직접적인 joint trajectory나 torque command가 아니므로, downstream robot-control module에서 line/arc primitive를 Cartesian desired trajectory로 샘플링한다.
이후 board-to-base frame transform, pen-tip offset compensation, inverse kinematics 또는 Jacobian-based control을 통해 로봇이 해당 drawing path를 추종하도록 한다.
```

더 짧게 쓰면 다음과 같다.

```text
LLM planner는 무엇을 그릴지 결정하고, robot-control module은 그것을 실제 로봇 움직임으로 변환한다.
```

---

## 22. 주의할 점

1. LLM output을 바로 로봇 joint command로 착각하면 안 된다.
2. `DrawingPlan`의 좌표는 board frame 기준이다.
3. 로봇 제어에는 반드시 `T_base_board`가 필요하다.
4. 펜 끝과 end-effector가 일치하지 않으므로 `T_ee_tip` 보정이 필요하다.
5. line/arc primitive는 시간 샘플링을 해야 trajectory가 된다.
6. IK가 없으면 joint command를 만들 수 없다.
7. 특이점 근처에서는 Jacobian inverse 대신 damped least squares를 쓰는 것이 안전하다.
8. 처음에는 joint position control로 시작하는 것이 가장 현실적이다.
9. 실제 실행 전에는 board boundary뿐 아니라 robot workspace와 joint limit도 검사해야 한다.
10. LLM planner는 IK feasibility를 검사하지 않으므로, downstream module에서 reachability validation을 해야 한다.

---

## 23. 최종 요약

이 LLM 코드는 trajectory code를 출력하는 것이 아니다.

정확히는 다음을 출력한다.

```text
board-frame symbolic primitive action JSON
```

이 JSON은 다음과 같은 의미를 가진다.

```text
move_to_start
pen_down
draw_line
draw_arc
pen_up
```

네가 만들어야 할 핵심 연결 부분은 다음이다.

```text
DrawingPlan JSON
→ Cartesian trajectory sampler
→ board-to-base transform
→ pen-tip offset compensation
→ IK / Jacobian control
→ robot controller
```

따라서 이 프로젝트의 역할 분담은 다음처럼 정리된다.

```text
팀원 LLM 코드:
    자연어 명령을 그림 primitive plan으로 변환한다.

내 로봇 제어 코드:
    primitive plan을 실제 로봇이 따라갈 desired trajectory와 control command로 변환한다.
```
