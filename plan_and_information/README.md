# Franka LLM Drawing Control

English version: [README_EN.md](README_EN.md)

이 저장소는 LLM drawing planner 뒤에 붙는 로봇 제어 쪽 코드이다. 기존
`Unit_Action_Langchain-main` 폴더는 그대로 LLM planner 계층으로 유지한다.
그 코드는 자연어 drawing command를 board frame 기준 primitive action JSON으로
바꾼다.

이 패키지는 그 JSON을 입력으로 받아, 최종 USD scene이 준비되기 전에도 실행할
수 있는 offline robot-control core를 제공한다.

- LLM planner가 만든 `DrawingPlan` JSON 파싱
- line, arc, pen-down, pen-up primitive를 Cartesian pen-tip pose로 샘플링
- board-frame pose를 robot base-frame pose로 변환
- end-effector frame과 실제 pen tip frame 사이의 고정 offset 보정
- pure Python damped least-squares Differential IK fallback 실행
- pure math hybrid position-force controller 계산
- offline test와 예제를 위한 mock Franka backend 제공
- Isaac Sim / Isaac Lab 연동 코드는 stub 뒤로 격리

중요한 점은 LLM 출력이 로봇 trajectory가 아니라는 것이다. LLM 출력은 symbolic
drawing primitive plan이고, 이 패키지가 그 plan을 desired Cartesian pose와
controller command로 변환한다.

## 현재 범위

현재 구현된 내용:

- `franka_llm_drawing/` 아래 offline core library
- `examples/` 아래 sample JSON plan
- `configs/` 아래 placeholder 설정 파일
- transform, sampler, controller, metric에 대한 pytest 테스트
- controller math와 command 배선 확인용 mock robot backend

아직 의도적으로 구현하지 않은 내용:

- 최종 USD 로딩
- 실제 Isaac prim path
- 실제 Franka articulation handle
- 실제 end-effector frame 이름
- contact sensor path
- 실제 simulator dynamics

이 항목들은 USD scene이 준비된 뒤
`franka_llm_drawing/sim/isaac_backend_stub.py`를 채우면서 구현한다.

## Isaac Sim을 처음 사용하는 경우

이 프로젝트는 Isaac Sim을 바로 실행하는 프로젝트가 아니라, Isaac Sim에 나중에
연결될 로봇 제어 코어를 먼저 준비한 프로젝트이다. 그래서 지금은 Isaac Sim이나
USD 파일이 없어도 테스트와 예제가 실행된다.

처음 볼 때 중요한 용어는 다음 정도만 이해하면 된다.

- USD scene:
  Isaac Sim에서 로봇, 보드, 펜, 조명, 카메라 같은 물체가 들어 있는 3D 장면 파일.
- prim path:
  USD scene 안에서 특정 물체를 가리키는 주소이다. 예를 들면 `/World/Franka`
  같은 문자열이 될 수 있다. 최종 scene이 없으므로 아직 확정하지 않는다.
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

따라서 초반 개발 순서는 `LLM JSON -> offline trajectory/controller test ->
Isaac backend 연결`이 된다. Isaac Sim 사용법을 아직 잘 몰라도, 먼저 이 패키지의
offline test와 mock control loop로 전체 흐름을 이해할 수 있다.

## 폴더 구조

```text
Unit_Action_Langchain-main/     기존 LLM planner. 별도 폴더로 유지한다.
franka_llm_drawing/             Offline robot-control core.
configs/                        Placeholder frame/controller/task 설정.
examples/                       Sample plan과 offline 실행 스크립트.
tests/                          Isaac 없이 실행되는 unit test.
```

## 코드 구성

현재 프로젝트는 LLM planner와 로봇 제어 코드를 역할별로 분리한다.

```text
Unit_Action_Langchain-main/
    자연어 명령을 DrawingPlan JSON으로 바꾸는 LLM planner.
    이 폴더는 IK, Jacobian, joint command, Isaac 실행을 담당하지 않는다.

franka_llm_drawing/llm_bridge/
    LLM planner가 저장한 DrawingPlan JSON을 읽고, action 목록을 내부 dataclass로
    변환한다. OpenAI나 LangChain을 호출하지 않는다.

franka_llm_drawing/trajectory/
    move_to_start, pen_down, draw_line, draw_arc, pen_up action을 시간 샘플된
    board-frame pen-tip pose로 바꾼다. line과 arc는 cubic time scaling을 사용한다.

franka_llm_drawing/frames/
    board frame에서 robot base frame으로 변환하고, pen tip pose에서
    end-effector pose를 계산한다.

franka_llm_drawing/controllers/
    Differential IK fallback, pose error 계산, nullspace posture control,
    hybrid position-force controller 수식을 제공한다. 모두 NumPy 기반이며 Isaac에
    의존하지 않는다.

franka_llm_drawing/robot/
    로봇 backend가 반드시 제공해야 하는 interface와 MockFrankaBackend가 있다.
    MockFrankaBackend는 실제 Franka 모델이 아니라, controller 배선과 테스트를 위한
    shape-correct mock이다.

franka_llm_drawing/sim/
    나중에 Isaac Sim / Isaac Lab backend를 채울 위치이다. 지금은 stub만 있고,
    core module이 Isaac 설치 여부에 영향받지 않도록 분리되어 있다.

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
기존 LLM planner는 `Unit_Action_Langchain-main` 안에 별도 패키지와 테스트를
가지고 있다.

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

## USD Scene 초안

새로 정리한 scene 초안은 다음 파일이다.

```text
usd/franka_drawing_scene_clean.usda
```

이 파일은 기존 `usd/franka_scene.usd`, `usd/franka_scene(1).usd`를 바로 쓰지 않고
다시 정리한 버전이다.

- Franka는 외부 URL이 아니라 `usd/assets/franka/franka_panda.usd`에서 불러온다.
- Table은 `0.60 m x 0.60 m x 0.20 m`이고 top surface는 `z = 0.20 m`이다.
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
