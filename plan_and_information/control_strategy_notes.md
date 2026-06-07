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

## Iterative DLS servo 실험 모드

현재 기본 실행은 한 Cartesian target마다 한 번의 resolved-rate DLS command를 보낸다.

```text
q_target = q_current + delta_q
```

이 방식은 단순하고 안정적이지만, moving trajectory에서는 실제 joint-position
actuator가 `q_target`을 즉시 따라가지 못하므로 planned path보다 뒤처지는 offset이
남을 수 있다. 특히 종이면 접촉과 펜 자세 유지가 동시에 걸리면, 위치 추종 lag가 더
잘 보인다.

이를 확인하기 위해 기본 controller를 지우지 않고 별도 실험 모드를 추가했다.

```yaml
tracking_mode: iterative_servo
iterative_servo_iterations: 3
iterative_servo_drawing_only: true
```

이 모드는 JADE를 제거하지 않는다. 같은 FK, Jacobian, DLS, condition number,
joint-limit check를 그대로 쓰되, drawing target마다 DLS step과 Isaac physics step을
여러 번 반복해서 해당 target에 더 수렴한 뒤 다음 target으로 넘어간다.

```text
for each drawing target:
  repeat N times:
    FK로 현재 tip pose 확인
    Jacobian 계산
    DLS로 delta_q 계산
    q_target = q_current + delta_q
    Isaac joint-position backend 실행
```

따라서 이 모드는 수업 내용과의 연결이 더 약해지는 것이 아니라, 오히려
`FK -> Jacobian -> Damped Least Squares IK -> joint position control` 흐름을 더
명확하게 보여준다. 단점은 drawing 시간이 길어진다는 것이다. 즉 속도보다 궤적 추종
정확도를 우선하는 실험 모드로 봐야 한다.

이 모드는 튜닝 과정에서 비교했지만, 최종 기본 실행에서는 사용하지 않는다. 기본
`configs/jade.yaml`의 `tracking_mode`는 `differential`이므로 최종 실행은 기존 방식
그대로 유지된다.

## 260607 tangent integral tuning

13번 iterative servo 실험은 draw XY RMSE를 조금 줄였지만, x 방향 평균 offset을
근본적으로 없애지는 못했다.

핵심 로그는 다음과 같았다.

```text
13번 iterative servo:
  desired - actual x 평균      약 3.20 mm
  commanded - desired x 평균   약 3.02 mm
  commanded - actual x 평균    약 6.22 mm
```

즉 controller가 이미 planned path보다 x 방향으로 3 mm 정도 앞선 command를 보내고
있는데도, 실제 tip은 그 command보다 약 6 mm 뒤에 있었다. 이 결과는 단순한
resolved-rate IK 반복 횟수 문제가 아니라, 종이면 접촉, 마찰, joint-position actuator
tracking 때문에 생기는 steady tracking offset 문제에 가깝다는 뜻이다.

14번 실험에서는 paper-tangent integral correction을 더 강하게 했다.

```yaml
tangent_integral_gain: 0.45
tangent_integral_leak_per_s: 0.05
max_tangent_integral_offset_m: 0.015
```

이 보정은 z 방향 접촉력 제어를 건드리지 않고, 종이면 접선 방향의 위치 오차만
적분해서 Cartesian command를 조금 앞쪽으로 보낸다.

```text
08번 기존 추천:
  draw_xy_rmse      3.448 mm
  mean_x_offset     3.379 mm
  force_mean        3.119 N
  force_max        11.468 N

14번 tangent_i_strong:
  draw_xy_rmse      1.402 mm
  mean_x_offset     1.016 mm
  force_mean        3.134 N
  force_max        10.970 N
```

따라서 14번은 평균 offset을 크게 줄인 좋은 후보이다. 다만 offset-compensated RMSE는
08번보다 약간 커졌으므로, command를 너무 강하게 앞세워서 선 내부의 작은 흔들림이
늘어났을 가능성이 있다.

다음 튜닝에서는 14번의 장점은 유지하되 흔들림을 줄이는 것이 목표였다. 비교 후보는
다음과 같은 gain 조합이었다.

```text
15번 tangent_i_mid:
    gain 0.35, leak 0.08, max 12 mm

16번 tangent_i_040:
    gain 0.40, leak 0.06, max 13 mm

17번 tangent_i_strong_limited:
    gain 0.45, leak 0.05, max 12 mm
```

선택 기준은 단순하다.

- `mean_x_offset_mm`는 14번처럼 1 mm 근처로 낮을수록 좋다.
- `offset_comp_xy_mm`는 08번처럼 0.7 mm 근처로 낮을수록 선 내부 흔들림이 적다.
- `force_mean_n`, `force_p95_n`, `force_max_n`가 14번보다 크게 악화되면 제외한다.

## Final default contact drawing setting

최종 기본 설정은 20번 실험 결과를 따른다.

```text
20_star_phase_lift_014_xff_0034
```

20번은 강한 tangent integral이나 iterative servo를 쓰지 않고, 08번 안정 baseline에
작은 workspace calibration offset만 더한 설정이다.

```yaml
tracking_mode: differential
tangent_integral_gain: 0.20
tangent_integral_leak_per_s: 0.20
max_tangent_integral_offset_m: 0.008
phase_gating_enabled: true
max_drawing_lift_offset_m: 0.018
command_position_offset_m: [0.0034, 0.0, 0.0]
```

결과는 다음과 같다.

```text
08번 기존 안정 baseline:
  draw_xy_rmse      3.448 mm
  mean_x_offset     3.379 mm
  offset_comp_xy    0.684 mm
  draw force max   11.468 N

20번 최종 기본 후보:
  draw_xy_rmse      0.683 mm
  mean_x_offset    -0.009 mm
  offset_comp_xy    0.682 mm
  draw force max   11.285 N
```

즉 평균 x offset은 거의 0으로 줄었고, 선 내부 흔들림 지표와 접촉력은 기존 안정
baseline 수준을 유지했다. 최종 정리 단계에서는 drawing 중 normal-force loop가 lift
offset 상한 근처에서 동작하는 것을 확인했기 때문에, XY/DLS 제어는 그대로 두고
`max_drawing_lift_offset_m`만 14 mm에서 18 mm로 완화했다. 이 변경은 종이면 법선
방향 command offset 한계만 바꾸는 것이므로 그림의 XY 추종 구조에는 직접 개입하지
않는다.

이 offset은 제어기 성능을 속이기 위한 값이 아니라, 현재 USD의 table/paper/tool 접촉
환경에서 반복적으로 관측된 base-X systematic bias를 보정하는 workspace calibration으로
보는 것이 맞다. 발표에서는 이것을 핵심 기여로 길게 설명할 필요는 없지만, 질문을 받으면
“시뮬레이션 환경의 반복적인 tool/contact bias를 보정하기 위한 고정 workspace
calibration을 적용했다”고 설명하면 된다.

단, 이 값이 모든 도형, 모든 크기, 모든 위치에서 항상 최적이라고 단정하면 안 된다.
같은 종이 위치, 같은 접촉력, 같은 drawing speed, 같은 펜 자세 조건에서는 비슷한 보정
효과를 기대할 수 있지만, 도형 크기와 위치가 크게 바뀌거나 속도/접촉력이 달라지면 다시
검증해야 한다.

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

## 260607 contact tuning 결과

최근 별 궤적 실험에서는 controller를 새로 바꾼 것이 아니라, 기존 DLS position
controller와 normal-force admittance controller는 유지한 상태에서 force loop의
phase별 limit만 실험했다.

이 실험에서 가장 좋았던 설정은 baseline에 다음을 추가한 phase-gated normal-force
admittance였다.

```yaml
lookahead_time_s: 0.0
phase_gating_enabled: true
max_drawing_lift_offset_m: 0.014
max_drawing_offset_step_m: 0.0005
```

의미는 단순하다.

- `phase_gating_enabled: true`는 drawing, pen_down 접촉 진입, pen_up release,
  원하지 않는 접촉 release를 구분해서 normal-force admittance limit을 다르게 쓴다.
- drawing 중에는 접촉력이 너무 커졌을 때 위로 빠져나갈 수 있는 lift offset을
  8 mm에서 14 mm로 늘렸다.
- offset step은 0.5 mm/step으로 제한해서 접촉력 보정이 갑자기 변하지 않게 했다.
- lookahead는 꺼두었다. 별 궤적 테스트에서는 lookahead가 sharp corner에서
  위치 추종을 개선하지 못하고 오히려 일부 지표를 악화시켰다.

baseline과 비교하면 다음과 같다.

```text
baseline:
  draw_xy_rmse      3.418 mm
  draw max force   17.626 N
  draw force p95    6.539 N
  contact ratio    98.3 %

phase_lift_014:
  draw_xy_rmse      3.448 mm
  draw max force   11.468 N
  draw force p95    6.082 N
  contact ratio   100.0 %
```

즉 08번 설정은 위치 추종 성능을 거의 잃지 않으면서 drawing 중 접촉 유지와
force spike를 개선한 설정이다. 반대로 `phase_only`처럼 drawing lift limit을
8 mm로 너무 작게 묶으면 위치 오차는 약간 줄어도 평균 접촉력이 14 N 이상으로
올라갔기 때문에 추천하지 않는다.

다만 모든 로그에서 약 47 N 수준의 큰 force spike가 `move_to_start` 초반에
나타났다. 이 spike는 drawing 중 hybrid force loop의 문제가 아니라, 시작 자세에서
첫 hover target으로 정렬되는 동안 펜이 종이를 순간적으로 누르는 startup 접근 문제로
보는 것이 맞다. 그래서 다음 실험은 force gain을 더 키우는 것이 아니라, 안전한
startup 접근 target을 앞에 추가하는 방식으로 진행한다.

처음 시도한 방식은 첫 target의 5 cm 위로 바로 보내는 것이었는데, 실제 로그에서는
초반에 tip z가 목표와 반대로 종이면까지 내려가며 약 30 N의 접촉이 발생했다. 이유는
시작 자세와 첫 target 사이의 XY/orientation 오차가 큰 상태에서 하나의 높은 target을
바로 추종하려고 했기 때문이다. 따라서 안전 접근은 다음처럼 더 명확히 나누는 편이
맞다.

```text
현재 tip 위치에서 위로 lift
-> 높은 z를 유지한 채 첫 target XY로 lateral move
-> 첫 hover target으로 descent
-> 기존 trajectory 실행
```

이후 실험한 startup-safe 후보는 08번 설정에 다음을 추가하는 방식이었다.

```yaml
startup_lift_height_m: 0.050
startup_lift_steps: 80
startup_lateral_steps: 100
startup_descent_steps: 80
```

의도는 현재 위치에서 먼저 종이면과 거리를 확보한 다음, 종이 위에서 움직이지 않고
공중에서 XY와 pen-axis 자세를 정렬하는 것이다. 이렇게 하면 drawing controller
자체는 그대로 두면서 `move_to_start` 초반 접촉 스파이크만 분리해서 줄일 수 있다.

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
