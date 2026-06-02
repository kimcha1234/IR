# Franka Drawing 제어 전략 정리

## 현재 결론

지금 단계에서는 기본 제어기를 `full torque hybrid position-force controller`로 바꾸지 않는다.
대신 현재 구조를 유지한다.

1. 기본 제어: DLS 기반 task-space position control
2. 접촉 제어: 종이면 법선 방향만 position-level admittance로 보정
3. 성능 검토: JADE 기반 Jacobian/특이점/관절한계/추종오차 진단
4. 향후 확장: 필요할 때 torque hybrid controller를 별도 고급 백엔드로 추가

이 방향이 프로젝트 목적에 더 맞다. 수업에서 배운 위치 제어와 힘 제어의 구조가 코드에서 잘 보이고, Isaac USD/접촉 센서/로봇 프레임이 아직 계속 조정되는 단계에서도 안정적으로 디버깅할 수 있다.

## 왜 지금 full torque hybrid로 바꾸지 않는가

토크 기반 hybrid force/position control 자체는 잘못된 방향이 아니다. Franka 같은 7자유도 로봇에서 종이면 접촉력을 직접 제어하려면 이론적으로는 자연스러운 방법이다.

다만 지금 프로젝트의 기본 제어기로 바로 쓰기에는 부담이 크다.

- Isaac backend에서 관절 position target 대신 raw torque command를 안정적으로 써야 한다.
- 중력보상, 감쇠, null-space posture, 토크 제한, 접촉 안정성 처리가 같이 필요하다.
- USD의 펜 길이, tip frame, contact sensor, 종이 높이가 조금만 틀려도 토크 제어는 진동하거나 강하게 눌릴 수 있다.
- 현재 목표는 우선 “펜 tip이 원하는 궤적을 따라가고, 접촉력 로그로 정상 여부를 확인하는 것”이다.

그래서 기본 구조는 단순하게 두고, 힘 제어는 종이면 법선 방향 command를 조금씩 올리거나 내리는 admittance 방식으로 구현하는 것이 더 안전하다.

## 현재 기본 제어 구조

현재 실행 흐름은 다음과 같다.

1. LLM 또는 JSON plan이 `move_to_start`, `pen_down`, `draw_arc`, `pen_up` 같은 동작을 만든다.
2. trajectory sampler가 각 동작을 Cartesian tip target으로 샘플링한다.
3. frame 변환 코드가 USD에 맞는 pen tip offset을 반영해서 end-effector target을 만든다.
4. Differential IK/DLS executor가 `panda_joint1`부터 `panda_joint7`까지만 관절 명령으로 변환한다.
5. Isaac backend가 Franka articulation에 joint position target을 보낸다.
6. contact sensor가 종이면 접촉력을 읽고, normal-force admittance가 다음 target의 z 방향을 보정한다.

중요한 점은 finger joint는 command 대상으로 쓰지 않는다는 것이다. Isaac backend에서는 arm joint인 `panda_joint1`부터 `panda_joint7`까지만 제어한다.

## Unit action trajectory planning

LLM이 만든 `move_to_start`, `pen_down`, `draw_line`, `draw_arc`, `pen_up`은 바로
관절 명령으로 들어가지 않는다. 먼저 board frame 기준의 geometric primitive로
해석되고, 그 다음 trajectory planner가 시간 정보를 붙인다.

현재 trajectory layer는 수업에서 다룬 기본 개념을 코드에 직접 반영한다.

- 각 unit action은 line 또는 circular arc segment로 샘플링된다.
- segment에는 cubic 또는 quintic time scaling이 적용된다.
- 기본 설정은 quintic time scaling이다.
- 각 sample은 position, velocity, acceleration, jerk 정보를 가진다.
- segment duration은 peak speed, acceleration, jerk limit을 만족하도록 늘어난다.
- `pen_down`은 필요하면 접촉력 ramp를 쓸 수 있지만, 현재 기본값은 off이다.
- `pen_down` 직후에는 같은 위치에서 짧게 정지해서 접촉 힘이 안정될 시간을 준다.
- `pen_up`은 목표 힘을 0 N으로 두고, 접촉이 풀릴 때까지 release controller가
  lift offset을 갑자기 reset하지 않는다.

이 구조는 원 하나에만 맞춘 것이 아니라, LLM이 어떤 도형을 여러 unit action으로
만들어도 같은 trajectory planning 계층을 통과하게 한다.

현재 설정은 `configs/jade.yaml`의 `sampling` 섹션에 있다.

```yaml
time_scaling: quintic
max_linear_speed_m_s: 0.025
max_linear_accel_m_s2: 0.08
max_linear_jerk_m_s3: 0.40
contact_force_ramp_duration_s: 0.0
pen_down_settle_duration_s: 0.0
```

The smooth contact-force ramp is kept available in the sampler, but the
default run currently disables it. In Isaac contact tests, the 0.60 s ramp made
the contact-force standard deviation and peak force worse, so the next stable
comparison should keep the trajectory retiming layer and remove only the force
ramp.

The Cartesian speed limit is intentionally conservative. In contact drawing,
tracking accuracy and stable normal force matter more than drawing quickly, so
the default cap is 0.025 m/s.

The sampler can insert a `pen_down_settle` hold segment after each `pen_down`,
but the default is currently disabled. In the latest Isaac contact tests, the
hold segment kept the pen in a high-force contact state before drawing and made
force RMSE worse. The safer baseline is to keep guarded contact entry and start
drawing without an extra hold.

The normal-force admittance gains are intentionally responsive enough to release
excess force quickly while still using a simple position-level admittance loop:

```yaml
kp_offset_m_per_n: 0.0015
ki_offset_m_per_n_s: 0.00035
max_offset_step_m: 0.0015
force_filter_alpha: 0.35
```

During `pen_up`, the force target is 0 N. The admittance controller remains
active during the release action so the previous lift offset decays smoothly
instead of being reset in one control step while the pen is still touching the
paper.

The same continuity rule is used for paper-tangent XY correction. During
`pen_up`, the last drawing tangent-integral offset is kept and decayed smoothly
so the command does not jump sideways while the pen is still in contact. Once a
non-contact move such as `move_to_start` begins, that tangent offset is reset so
the next stroke or shape starts without stale correction from the previous one.

이 계층은 trajectory를 더 부드럽게 만드는 역할이다. 단, 현재 DLS controller가
`xdot_d`, `xddot_d`를 완전한 computed-torque 방식으로 쓰는 것은 아니다. 지금
단계에서는 부드러운 `x_d(t)`를 만드는 것에 집중하고, velocity/acceleration
feedforward는 이후 고급 기능으로 확장한다.

## Command-integrated DLS를 쓰는 이유

이 제어기는 새로운 임의의 방식이 아니라 resolved-rate inverse kinematics를
joint-position backend에 맞게 이산시간으로 구현한 것이다.

수업에서 자주 보는 형태는 다음과 같다.

```text
qdot_cmd = J# Kx (x_des - x)
q_cmd[k+1] = q_cmd[k] + qdot_cmd * dt
```

코드의 DLS solver는 `qdot_cmd * dt`에 해당하는 `delta_q`를 계산한다.
초기 구현은 매 step `q_target = q_measured + delta_q`를 보냈다. 이 방식도
differential IK 예제에서는 흔하지만, 접촉 때문에 실제 관절이 목표를 완전히
따라가지 못하면 joint-position PD 입장에서 항상 작은 target error만 보게 된다.
그래서 정지 목표에서도 x 방향 steady-state error가 남을 수 있었다.

이론적으로는 내부 `q_command`를 유지해서 다음처럼 보낼 수 있다.

```text
q_command[k+1] = q_command[k] + delta_q
q_target = q_command[k+1]
```

즉 외부 loop는 task-space 오차로부터 관절 속도/증분을 만들고, 내부 Isaac
joint-position actuator가 그 누적된 관절 목표를 따라간다. 이는 위치 제어 구조로
설명할 수 있다.

하지만 contact drawing에서 full joint-command integration은 z 방향 접촉 오차와
orientation 오차까지 함께 누적할 수 있다. 실제 테스트에서 이 방식은 종이면을
과하게 누르는 현상을 만들었기 때문에 기본값은 `command_integration_enabled: false`
로 둔다.

대신 현재 기본 controller는 더 제한적인 누적항을 쓴다.

```text
e_tangent = (I - nn^T) (x_des - x)
i_tangent[k+1] = leak * i_tangent[k] + e_tangent * dt
x_cmd = x_des + clip(Ki * i_tangent)
```

여기서 `n`은 종이면 법선이다. 이 누적항은 종이면 접선 방향, 즉 그림의 XY 방향에만
작용한다. Z 방향 접촉력은 기존 normal-force admittance가 담당한다. 그래서 상수
Cartesian offset보다 일반화 가능성이 높고, full joint-command integration보다
접촉 안정성이 높다.

## Guarded pen_down

기존 `pen_down`은 계획된 높이까지 내려가는 동안 실제 접촉이 발생해도 마지막 샘플 전까지는 “아직 접촉 목표가 아님”으로 처리될 수 있었다.
이 경우 접촉을 0N으로 풀려고 하거나, 접촉과 위치 추종이 서로 충돌할 수 있다.

수정 방향은 다음과 같다.

- `pen_down` 중 contact sensor가 접촉을 감지하면 그 순간부터 guarded contact 상태로 본다.
- guarded contact 상태에서는 목표 접촉력을 0N이 아니라 drawing target force, 기본 1N으로 전환한다.
- `move_to_start`나 `pen_up` 중 원하지 않는 접촉이 감지되면 목표 접촉력은 0N으로 두고 위로 빠져나오게 한다.

즉, `pen_down`은 “정해진 z까지 무조건 내려가기”가 아니라 “접촉이 확인되면 힘 제어로 부드럽게 넘기기”가 된다.

## Contact-mode actuator gain

현재 Isaac backend는 torque command가 아니라 joint-position target을 사용한다.
따라서 contact-mode에서 관절 PD gain은 force tracking과 pen orientation 사이의 절충점이 된다.

펜 축 수직 유지가 최우선 요구사항이면 arm joint는 너무 부드럽게 만들면 안 된다.
관절 PD를 낮추면 종이면 접촉력은 조금 부드러워질 수 있지만, 펜이 눌릴 때 자세가 쉽게 무너진다.
그래서 현재 기본값은 HIGH_PD와 같은 강한 gain을 유지한다.

- 기본 HIGH_PD gain: stiffness 400, damping 80
- contact-mode 기본 gain: stiffness 400, damping 80

이 변경은 토크 제어로 바꾸는 것이 아니다. 여전히 joint-position backend를 쓰되, 자세가 무너지지 않도록 arm PD는 단단하게 유지하고, 접촉력은 normal-direction command offset으로 조절한다.
만약 나중에 접촉력 overshoot가 다시 커지면 gain을 낮추기 전에 pen_down 높이, draw height, admittance limit, 접촉 센서 위치를 먼저 확인하는 것이 좋다.

## JADE의 역할

JADE는 기본 제어기를 대체하는 것이 아니라, 실행 중 위험도를 판단하고 결과를 해석하는 고급 제어/검토 계층이다.

현재 JADE가 보는 항목은 다음과 같다.

- Jacobian conditioning
- 특이점 근접 여부
- 관절 한계 margin
- Cartesian tracking error
- tip orientation error
- contact force tracking error

따라서 보고서나 발표에서는 JADE를 단순한 “평가 코드”보다는 `Jacobian-Aware Drawing Execution` 또는 `Jacobian-aware execution monitor`로 설명하는 것이 자연스럽다.

## 향후 torque hybrid를 추가할 때

나중에 torque 기반 hybrid controller를 넣는다면 현재 기본 제어기를 지우지 말고 별도 backend/controller로 추가하는 것이 좋다.

- 기본 모드: joint position + DLS + normal-force admittance
- 고급 모드: torque command + operational-space hybrid force/position control

이렇게 두면 수업에서 배운 기본 제어 구조와 고급 제어 확장을 모두 보여줄 수 있고, 문제가 생겼을 때 안정적인 baseline으로 바로 돌아갈 수 있다.
