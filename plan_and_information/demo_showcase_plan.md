# API-free demo showcase plan

현재 발표/영상용 데모는 OpenAI API를 사용하지 않는 `no-api` fallback으로 진행한다.
이 방식은 실제 LLM reasoning은 아니지만, 자연어처럼 보이는 입력을 고정 규칙으로
`DrawingPlan` JSON으로 변환하고, 같은 JADE/Isaac 실행 계층으로 넘긴다.

## 데모 목적

지금까지 구현한 전체 흐름을 안정적으로 보여주는 것이 목적이다.

```text
자연어 입력
-> no-api fallback planner
-> DrawingPlan JSON
-> bridge validator
-> JADE trajectory/control
-> Isaac Sim 실행
-> 결과 로그/그래프
```

발표에서는 다음처럼 설명한다.

```text
현재 데모에서는 API 비용 없이 재현 가능한 rule-based natural-language fallback을
사용했다. 실제 LLM planner는 같은 DrawingPlan schema로 교체 가능하도록 분리했다.
```

## 지원 데모 입력

현재 `llm_isaac_bridge/demo_commands.json`에 고정한 입력은 다음이다.

```text
중앙에 반지름 4cm짜리 원을 그려줘
중앙에 한 변 6cm짜리 사각형을 그려줘
중앙에 한 변 6cm짜리 삼각형을 그려줘
중앙에 알파벳 A를 8cm 크기로 그려줘
중앙에 8cm 집 모양을 그려줘
중앙에 8cm 별을 그려줘
```

각 입력은 다음 파일들을 만든다.

```text
outputs/demo_showcase/plans/<case_id>.json
outputs/demo_showcase/reports/<case_id>_validation.json
outputs/demo_showcase/isaac_runs/<case_id>/
outputs/demo_showcase/manifest.json
outputs/demo_showcase/isaac_commands.sh
```

## plan 생성과 검증

```bash
cd /home/kimchangyeol/IsaacLab/IR
/home/kimchangyeol/IsaacLab/IR/llm_isaac_bridge/scripts/prepare_demo_showcase.sh
```

특정 case만 준비하려면:

```bash
/home/kimchangyeol/IsaacLab/IR/llm_isaac_bridge/scripts/prepare_demo_showcase.sh \
  --case circle_r4cm
```

## Isaac 실행

한 개씩 안정적으로 영상 녹화하려면 생성된 명령을 하나씩 실행한다.

```bash
bash /home/kimchangyeol/IsaacLab/IR/outputs/demo_showcase/isaac_commands.sh
```

또는 특정 case만 바로 실행한다.

```bash
/home/kimchangyeol/IsaacLab/IR/llm_isaac_bridge/scripts/prepare_demo_showcase.sh \
  --case circle_r4cm \
  --execute-isaac
```

영상 녹화용으로는 한 번에 여러 case를 연속 실행하기보다, case 하나를 실행하고
화면을 확인한 뒤 다음 case를 실행하는 방식을 권장한다.

## 추천 영상 순서

1. terminal에 자연어 입력 command가 보이게 한다.
2. plan JSON과 validation OK 메시지를 보여준다.
3. Isaac Sim에서 Franka가 펜을 내려 도형을 그리는 장면을 녹화한다.
4. 실행 후 `planned_vs_actual_xy.svg`, `tracking_error.svg`,
   `normal_force.svg`를 짧게 보여준다.

## 주의사항

- 이 demo fallback은 API-free 재현용이다.
- 실제 LLM 자동 planner는 `Unit_Action_Langchain-main`의 `agentic/template`
  모드로 같은 schema에 연결할 수 있다.
- Isaac backend는 계속 `panda_joint1`부터 `panda_joint7`까지만 command 대상으로
  사용한다. finger joint는 command하지 않는다.
