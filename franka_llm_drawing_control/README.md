# Franka LLM Drawing Control

English version: [README_EN.md](README_EN.md)

이 저장소는 LLM drawing planner 뒤에 붙는 로봇 제어 쪽 코드이다. 기존
`Unit_Action_Langchain-main` 폴더는 그대로 LLM planner 계층으로 유지한다.
그 코드는 자연어 drawing command를 board frame 기준 primitive action JSON으로
바꾼다.

이 패키지는 그 JSON을 입력으로 받아, USD scene이 준비되지 않아도 테스트할 수
있는 offline robot-control core와, 현재 정리된 USD scene을 Isaac Lab에서 바로
hover tracing으로 실행해 볼 수 있는 최소 JADE 실행 계층을 제공한다.

- LLM planner가 만든 `DrawingPlan` JSON 파싱
- line, arc, pen-down, pen-up primitive를 Cartesian pen-tip pose로 샘플링
- board-frame pose를 robot base-frame pose로 변환
- end-effector frame과 실제 pen tip frame 사이의 고정 offset 보정
- pure Python damped least-squares Differential IK fallback 실행
- JADE: Jacobian-aware stroke validation, adaptive damping/speed policy,
  planned-vs-actual logging
- pure math hybrid position-force controller 계산
- offline test와 예제를 위한 mock Franka backend 제공
- Isaac Lab 실행 스크립트와 최소 Franka backend 제공

중요한 점은 LLM 출력이 로봇 trajectory가 아니라는 것이다. LLM 출력은 symbolic
drawing primitive plan이고, 이 패키지가 그 plan을 desired Cartesian pose와
controller command로 변환한다.

## 현재 범위

현재 구현된 내용:

- `franka_llm_drawing/` 아래 offline core library
- `franka_llm_drawing/jade/` 아래 JADE 검증/정책/실행 계층
- `examples/sample_plans/` 아래 circle, square, letter A JSON plan
- `configs/frames.yaml`, `configs/jade.yaml`, `configs/isaac_scene.yaml`
- transform, sampler, controller, JADE metric에 대한 pytest 테스트
- controller math와 command 배선 확인용 mock robot backend
- `examples/run_jade_offline.py`: Isaac 없이 JADE pipeline 검증
- `examples/run_jade_isaac.py`: Isaac Lab에서 local USD로 hover tracing 실행

아직 의도적으로 구현하지 않은 내용:

- contact sensor path
- force/contact drawing controller의 실제 실행
- ROS2 또는 실제 Franka robot 연동

첫 Isaac demo는 contact drawing이 아니라, 펜이 종이 위 약 2 cm에서 경로를
따라가는 hover tracing이다. contact/force 제어는 이 경로 추종이 안정적으로
확인된 뒤 추가한다.

## Isaac Sim을 처음 사용하는 경우

이 프로젝트는 먼저 offline core를 검증하고, 그 다음 같은 trajectory를 Isaac
Lab에서 실행하는 구조이다. Isaac Sim을 설치하지 않아도 offline 테스트는 돌릴 수
있고, Isaac Sim이 준비된 환경에서는 `run_jade_isaac.py`로 실제 Franka USD를
열어 펜이 움직이는지 확인한다.

처음 볼 때 중요한 용어는 다음 정도만 이해하면 된다.

- USD scene:
  Isaac Sim에서 로봇, 보드, 펜, 조명, 카메라 같은 물체가 들어 있는 3D 장면 파일.
- prim path:
  USD scene 안에서 특정 물체를 가리키는 주소이다. 이 프로젝트에서는
  `/World/Franka`, `/World/Paper`, `/World/BoardFrame`을 쓴다.
- articulation:
  Franka처럼 여러 joint로 연결된 로봇 모델이다. 실제 joint state, Jacobian,
  command API는 Isaac backend가 여기서 읽고 쓴다.
- backend:
  core controller와 실제 simulator 사이의 어댑터이다. 지금은
  `MockFrankaBackend`가 이 역할을 흉내 내고, 나중에는 Isaac backend가 같은
  interface를 구현한다.
- frame:
  좌표계이다. LLM plan은 board frame 기준이고, 로봇 제어는 robot base frame
  기준으로 계산해야 한다.

따라서 개발 순서는 `LLM JSON -> offline JADE 검증 -> Isaac hover tracing ->
contact drawing`이 된다. Isaac Sim 사용법을 아직 잘 몰라도, 먼저 offline test와
`run_jade_offline.py`로 전체 흐름을 이해할 수 있다.

## 폴더 구조

기존 팀원 GitHub repo에는 이미 `assets/`, `examples/`, `robot_drawing_planner/`,
`scripts/`, `tests/`, `README.md`, `pyproject.toml`이 있다. 이 코드들은 그 파일을
덮어쓰지 않고, repo root에 `franka_llm_drawing_control/` 폴더 하나로 추가하는
것을 기준으로 한다.

```text
Unit_Action_Langchain/                  기존 GitHub repo root
├── assets/                             기존 팀원 폴더
├── examples/                           기존 LLM 예제
├── robot_drawing_planner/              기존 LLM planner
├── scripts/                            기존 스크립트
├── tests/                              기존 LLM 테스트
├── README.md                           기존 README
├── pyproject.toml                      기존 pyproject
└── franka_llm_drawing_control/          이 폴더
    ├── README.md
    ├── README_EN.md
    ├── pyproject.toml
    ├── configs/
    ├── examples/
    ├── franka_llm_drawing/
    └── offline_tests/
```

## 코드 구성

현재 프로젝트는 LLM planner와 로봇 제어 코드를 역할별로 분리한다.

```text
기존 GitHub repo의 robot_drawing_planner/
    자연어 명령을 DrawingPlan JSON으로 바꾸는 LLM planner.
    이 기존 코드는 IK, Jacobian, joint command, Isaac 실행을 담당하지 않는다.

franka_llm_drawing/llm_bridge/
    LLM planner가 저장한 DrawingPlan JSON을 읽고, action 목록을 내부 dataclass로
    변환한다. OpenAI나 LangChain을 호출하지 않는다.

franka_llm_drawing/trajectory/
    move_to_start, pen_down, draw_line, draw_arc, pen_up action을 시간 샘플된
    board-frame pen-tip pose로 바꾼다. line과 arc는 cubic 또는 quintic time
    scaling을 사용한다.

franka_llm_drawing/frames/
    board frame에서 robot base frame으로 변환하고, pen tip pose에서
    end-effector pose를 계산한다.

franka_llm_drawing/controllers/
    Differential IK fallback, pose error 계산, nullspace posture control,
    hybrid position-force controller 수식을 제공한다. 모두 NumPy 기반이며 Isaac에
    의존하지 않는다.

franka_llm_drawing/jade/
    Jacobian singular value, condition number, manipulability, joint-limit
    margin을 계산하고 stroke별 OK/WARNING/FAIL report를 만든다. 위험 구간에서는
    DLS damping과 speed scale을 보수적으로 조정한다.

franka_llm_drawing/robot/
    로봇 backend가 반드시 제공해야 하는 interface와 MockFrankaBackend가 있다.
    MockFrankaBackend는 실제 Franka 모델이 아니라, controller 배선과 테스트를 위한
    shape-correct mock이다.

franka_llm_drawing/sim/
    Isaac Lab Franka articulation을 core interface로 감싸는 최소 backend가 있다.
    단, command 대상은 반드시 panda_joint1부터 panda_joint7까지 7개 arm joint만
    사용한다. finger joint는 USD에 있어도 command 대상이 아니다.

franka_llm_drawing/evaluation/
    desired path와 actual path, force, singular value, IK 성공률 등을 평가하는
    metric과 logger를 제공한다.
```

코드 흐름을 짧게 쓰면 다음과 같다.

```text
DrawingPlan JSON
-> llm_bridge
-> trajectory sampler
-> frame transform + pen-tip offset compensation
-> controller
-> robot backend
-> evaluation/logging
```

## 테스트 실행

이 디렉터리에서 실행한다.

```bash
pytest -q
```

루트의 pytest 설정은 이 패키지의 `tests/` 폴더만 찾도록 제한되어 있다.
이 업로드용 폴더에서는 기존 GitHub repo의 `tests/`와 충돌하지 않도록 테스트
폴더 이름을 `offline_tests/`로 둔다. 기존 LLM planner는 GitHub repo root에
별도 패키지와 테스트를 가지고 있다.

## Offline Plan Sampling 실행

```bash
python examples/run_offline_plan_sampling.py --plan examples/sample_plan_circle.json
```

이 스크립트는 plan을 로드하고, board-frame pose target을 샘플링한 뒤,
placeholder `T_base_board`와 `T_ee_tip` 변환을 적용해서 trajectory 요약을
출력한다.

## Mock Control Loop 실행

```bash
python examples/run_mock_control_loop.py --plan examples/sample_plan_circle.json
```

이 스크립트는 mock backend와 pure Python Differential IK fallback을 사용한다.
목적은 코드 배선과 controller math 확인이다. 실제 Franka kinematics나 contact
physics를 시뮬레이션하지 않는다.

## JADE Offline 실행

Isaac Sim 없이 parser, sampler, frame transform, Jacobian-aware validator,
adaptive policy, planned-vs-actual mock log까지 확인한다.

```bash
python examples/run_jade_offline.py --plan examples/sample_plans/circle.json --mode hover --out outputs/jade_offline_circle
```

결과 파일:

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

## Isaac Hover Tracing 실행

Isaac Sim / Isaac Lab이 정상 설치된 환경에서 실행한다.

```bash
cd /home/kimchangyeol/IsaacLab
./isaaclab.sh -p /home/kimchangyeol/IsaacLab/IR/franka_llm_drawing_control/examples/run_jade_isaac.py \
  --mode hover \
  --plan /home/kimchangyeol/IsaacLab/IR/franka_llm_drawing_control/examples/sample_plans/circle.json \
  --path-speed-scale 0.35 \
  --out /home/kimchangyeol/IsaacLab/IR/franka_llm_drawing_control/outputs/jade_isaac_circle
```

`--path-speed-scale`은 plan 안의 모든 speed를 같은 비율로 낮춘다. 현재 MVP는
feed-forward가 없는 폐루프 Differential IK라서, 처음 궤적 검증에서는 `0.3~0.5`
범위로 천천히 돌리는 편이 추종 확인에 적합하다.

빠른 headless smoke test:

```bash
cd /home/kimchangyeol/IsaacLab
./isaaclab.sh -p /home/kimchangyeol/IsaacLab/IR/franka_llm_drawing_control/examples/run_jade_isaac.py \
  --headless \
  --path-speed-scale 0.35 \
  --max-samples 20 \
  --settle-steps 5 \
  --hold-steps 2
```

현재 backend는 `configs/isaac_scene.yaml`의 joint list를 검사한다. 여기에는 반드시
아래 7개만 있어야 한다.

```text
panda_joint1, panda_joint2, panda_joint3, panda_joint4,
panda_joint5, panda_joint6, panda_joint7
```

## 초기 자세 보정

펜이 처음부터 종이에 닿지 않은 상태로, 종이 법선 방향에 최대한 정렬된 자세를
찾고 싶으면 아래 명령을 사용한다. 이 스크립트도 command 대상은
`panda_joint1`부터 `panda_joint7`까지만 사용한다.

```bash
cd /home/kimchangyeol/IsaacLab
./isaaclab.sh -p /home/kimchangyeol/IsaacLab/IR/franka_llm_drawing_control/examples/calibrate_initial_pose.py \
  --headless \
  --plan /home/kimchangyeol/IsaacLab/IR/franka_llm_drawing_control/examples/sample_plans/circle.json \
  --clearance-m 0.03 \
  --out /home/kimchangyeol/IsaacLab/IR/franka_llm_drawing_control/outputs/initial_pose_calibration
```

결과는 `best_initial_pose.json`과 `initial_pose_candidates.csv`에 저장된다.
검증한 값을 바로 `configs/isaac_scene.yaml`에 반영하려면 같은 명령에
`--write-scene-config`를 추가한다.

## USD Scene 초안

새로 정리한 scene 초안은 다음 파일이다.

```text
usd/franka_drawing_scene_clean.usda
```

이 파일은 기존 `usd/franka_scene.usd`, `usd/franka_scene(1).usd`를 바로 쓰지 않고
다시 정리한 버전이다.

- Franka는 외부 URL이 아니라 `usd/assets/franka/franka_panda.usd`에서 불러온다.
- Table은 `0.60 m x 0.60 m x 0.10 m`이고 top surface는 `z = 0.20 m`이다.
- Paper는 `0.30 m x 0.30 m x 0.002 m`이고 drawing surface는 `z = 0.202 m`이다.
- `/World/BoardFrame`은 paper surface 중심 `(0.5, 0.0, 0.202)`에 둔다.
- 펜은 `/World/Franka/panda_link7/panda_link8/panda_hand/PenTool` 아래에 붙였다.
- controller가 추종할 펜 끝 frame은 `PenTool/PenTipFrame`이다.
- `/World/Lights` 아래에 dome, overhead softbox, key, fill light를 넣어 로봇,
  테이블, 종이, 펜 움직임이 잘 보이도록 했다.
- 초기 마찰값은 접촉 제어 디버깅이 쉽도록 중간 정도로 설정했다.
  Paper/PenTip은 `static=0.55`, `dynamic=0.40`, Table은 `static=0.60`,
  `dynamic=0.45`이다.

그리퍼 finger link를 asset에서 완전히 삭제하지는 않았다. Franka articulation의
joint 관계가 깨질 수 있으므로, 첫 버전에서는 finger visual을 숨기고 collision을
비활성화한 뒤 `panda_hand`에 펜을 붙이는 방식을 사용한다. 완전한 no-gripper
Franka가 필요하면 별도의 custom Franka USD asset을 만들어 finger link와 finger
joint를 함께 제거해야 한다.

## 로봇을 실행하면 돌아가는 방식

현재는 USD scene과 Isaac backend가 없기 때문에 실제 로봇 실행은 하지 않는다.
대신 실행 흐름이 나중에 그대로 연결될 수 있도록 interface와 mock backend를 먼저
준비해 둔 상태이다.

실제 로봇 또는 Isaac Sim backend가 붙으면 전체 흐름은 다음 순서로 동작한다.

```text
1. LLM planner 실행
   자연어 명령을 받아 DrawingPlan JSON을 만든다.

2. DrawingPlan 로드
   llm_bridge가 JSON의 action 목록을 읽는다.

3. Cartesian trajectory 생성
   trajectory sampler가 move_to_start, pen_down, draw_line, draw_arc, pen_up을
   dt 간격의 board-frame pen-tip pose로 샘플링한다.

4. Frame transform 적용
   T_base_board를 사용해 board-frame pen-tip pose를 base-frame pen-tip pose로
   바꾼다.

5. Pen-tip offset 보정
   T_ee_tip을 사용해 원하는 pen tip pose에서 원하는 end-effector pose를 계산한다.

6. Robot state 읽기
   backend가 현재 q, q_dot, end-effector pose, Jacobian, gravity torque,
   contact force를 읽어온다.

7. Controller 계산
   기본 실행에서는 Differential IK가 desired end-effector pose를 따라갈
   q_target을 만든다. contact-aware 실행에서는 hybrid position-force controller가
   task wrench와 joint torque command를 계산한다.

8. Command 전송
   backend가 q_target 또는 tau_cmd를 Isaac Sim / 실제 robot API로 보낸다.

9. Simulation 또는 robot step
   backend가 한 timestep을 진행하고, 다음 desired pose에 대해 같은 과정을 반복한다.

10. Logging과 평가
    desired path, actual path, force, torque, singular value, condition number를
    기록하고 RMSE 같은 metric을 계산한다.
```

현재 mock 실행에서는 6번부터 9번까지를 `MockFrankaBackend`가 대신한다. 따라서
로봇이 실제로 움직였다는 의미가 아니라, controller 입력과 출력의 형태가 맞고
전체 software pipeline이 끊기지 않는지 확인하는 용도이다.

나중에 Isaac backend가 완성되면 바뀌는 부분은 주로 backend 내부이다.
`trajectory`, `frames`, `controllers`, `evaluation` 모듈은 그대로 재사용하고,
`robot/interfaces.py`가 정의한 method를 실제 Isaac articulation, Jacobian,
contact sensor, command API에 연결하면 된다.

Isaac Sim 안에서는 대략 다음처럼 대응된다.

```text
DrawingPlan JSON             Python 파일에서 로드하는 입력
T_base_board                 USD scene 안의 board pose 또는 설정 파일 값
T_ee_tip                     Franka 손목 frame에서 pen tip까지의 고정 offset
RobotBackend.get_state()     Isaac articulation에서 q, q_dot, end-effector pose 읽기
get_end_effector_jacobian()  Isaac/Isaac Lab이 제공하는 Jacobian 읽기
send_joint_position_target() Isaac articulation controller에 q_target 쓰기
send_joint_torque_command()  torque control을 사용할 때 tau_cmd 쓰기
step()                       Isaac simulation을 한 timestep 진행
```

처음 Isaac Sim에 연결할 때는 전체 hybrid force controller부터 시작하지 말고,
joint position target 기반 Differential IK 실행부터 붙이는 것이 가장 안전하다.
그 다음 contact sensor 값이 안정적으로 읽히는 것을 확인한 뒤 hybrid
position-force controller를 연결한다.

## 수업 개념과의 연결

- Homogeneous transform:
  `T_base_tip = T_base_board @ T_board_tip`,
  `T_base_ee = T_base_tip @ inv(T_ee_tip)`.
- Trajectory planning:
  line/arc primitive를 cubic time scaling으로 시간 샘플링한다.
- Jacobian-based Differential IK:
  `delta_q = J.T @ inv(J @ J.T + lambda^2 I) @ delta_x`.
- Singularity handling:
  controller diagnostics로 Jacobian singular value와 condition number를 반환한다.
- Force / torque relation:
  hybrid controller에서 `tau = J.T @ wrench` 관계를 사용한다.
- Dynamics hook:
  실제 backend가 gravity torque를 제공하면 torque command에 더할 수 있다.

## 이후 Isaac 연동

최종 USD가 준비되면 Isaac backend에 다음 값을 채운다.

- `robot_prim_path`
- `ee_frame_name`
- `board_prim_path`
- `pen_tip_frame_name` 또는 측정된 `T_ee_tip`
- contact sensor path
- `q`, `q_dot`, end-effector pose, Jacobian, force를 읽는 API
- joint position target 또는 torque command를 쓰는 API
- simulation stepping

core module에는 계속 Isaac 의존성을 넣지 않는다.
