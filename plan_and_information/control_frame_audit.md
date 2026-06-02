# Franka Drawing 좌표계 검수 메모

이 문서는 2026-06-01 기준 `franka_drawing_scene_clean.usda`, `frames.yaml`,
`isaac_scene.yaml`, JADE Isaac runner의 좌표계 연결을 검수한 결과이다.

## 기본 원칙

- 기본 제어에서 그림별 상수 위치 보정값은 사용하지 않는다.
  `configs/jade.yaml`의 `executor.command_position_offset_m` 기본값은
  `[0.0, 0.0, 0.0]`이다.
- Isaac backend는 항상 7개 arm joint만 command 대상으로 사용한다:
  `panda_joint1`부터 `panda_joint7`까지이다.
- finger joint는 USD articulation 안에 남아 있을 수 있지만, 제어기 command
  대상에는 포함하지 않는다.

## 기준 프레임

- `BoardFrame`
  - USD path: `/World/BoardFrame`
  - base 기준 위치: `[0.5, 0.0, 0.202]`
  - 종이 윗면과 같은 z 높이에 둔다.
- `Paper`
  - USD cube scale 기준 실제 크기: `0.30 m x 0.30 m x 0.002 m`
  - 종이 아랫면은 테이블 윗면과 맞고, 종이 윗면은 `BoardFrame` z와 맞는다.
- `PenTipFrame`
  - USD path:
    `/World/Franka/panda_link7/panda_link8/panda_hand/PenTool/PenTipFrame`
  - local translation: `[0.0, 0.0, 0.1]`
  - `PenTip` sphere center local z `0.097` + radius `0.003` = `0.100`이므로
    실제 접촉점은 `PenTipFrame`과 일치한다.

## 제어기에서 쓰는 변환

- `frames.T_base_board`
  - board-frame drawing sample을 robot base frame으로 옮긴다.
  - 현재 USD의 `/World/BoardFrame` 위치와 일치한다.
- `frames.T_ee_tip`
  - Isaac Lab에서 직접 제어/측정 가능한 rigid body는 `panda_link7`이다.
  - `T_ee_tip`은 `panda_link7`에서 실제 `PenTipFrame`까지의 pose이다.
  - 목표 tip pose에서 목표 `panda_link7` pose를 만들 때 사용한다.
- `frames.R_base_tip`
  - pen local +Z 축이 종이면 법선의 반대 방향을 향하도록 둔다.
  - 현재 `R_base_tip[:, 2] dot board_normal_base < -0.999`이다.

## Jacobian 보정값

`T_ee_tip.translation_m`과 `T_ee_tip.jacobian_translation_m`은 일부러 다르다.

- `translation_m`은 pose/FK/시각화/로그에서 실제 펜팁 위치를 계산할 때 쓴다.
- `jacobian_translation_m`은 Isaac PhysX가 제공하는 `panda_link7` Jacobian을
  펜팁 Jacobian으로 shift할 때 쓴다.
- 현재 USD/Isaac Lab 조합에서는 PhysX Jacobian의 기준점이 USD pose 계층의
  `PenTipFrame` pose와 완전히 같은 방식으로 노출되지 않아서, 유한차분 FK 검사로
  맞춘 별도 translation을 사용한다.
- `examples/check_isaac_jacobian_consistency.py` 결과:
  - max linear error norm: `0.000497 m/rad`
  - max angular error norm: `0.001672 rad/rad`
  - `panda_joint7`의 linear cosine은 작은 선속도에서 수치적으로 의미가 약하므로,
    cosine보다 error norm을 기준으로 판단한다.

## 남은 x 방향 오차 해석

최근 실행에서 보였던 x 방향 offset은 좌표축 반전, x/y swap, board origin mismatch로
보기 어렵다. Jacobian 유한차분 검사와 정적 프레임 검사가 통과했기 때문이다.

따라서 이 오차는 그림별 상수 offset으로 고정 보정하기보다, 다음 범주의 제어 문제로
다루는 것이 맞다.

- joint-position backend의 추종 지연
- DLS position/orientation task weight 경쟁
- contact force admittance가 z 방향 명령을 바꾸면서 생기는 접촉 동역학 영향
- path speed, acceleration, force target, actuator gain에 따른 폐루프 잔류 오차

상수 offset은 특정 원 궤적의 평균 오차를 줄일 수는 있지만, 다른 위치/크기/도형으로
일반화된다는 보장이 없으므로 기본 설정으로 두지 않는다.

## 자동 검수

아래 테스트가 이 문서의 핵심 가정을 확인한다.

```bash
pytest -q
```

주요 테스트 파일:

- `tests/test_scene_frame_consistency.py`
- `tests/test_run_jade_isaac.py`
- `examples/check_isaac_jacobian_consistency.py`
