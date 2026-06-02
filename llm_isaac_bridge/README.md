# LLM to Isaac Bridge

이 폴더는 기존 코드를 수정하지 않고, 이미 있는 두 계층을 연결하는 얇은 실행
계층이다.

```text
Unit_Action_Langchain-main/  자연어 -> DrawingPlan JSON
llm_isaac_bridge/           JSON 생성/검증/Isaac 실행 orchestration
franka_llm_drawing/          DrawingPlan JSON -> trajectory/control/Isaac backend
```

중요한 원칙:

- 이 폴더는 Isaac Sim, Isaac Lab, OpenAI, LangChain을 직접 import하지 않는다.
- LLM planner는 subprocess로 실행한다.
- Isaac 실행은 기존 `examples/run_jade_isaac.py`를 그대로 호출한다.
- 기존 LLM 코드와 기존 제어 코드는 수정하지 않는다.

## 1. Plan만 생성하고 검증하기

OpenAI 없이 deterministic fallback으로 확인한다. `--planner-mode no-api`는 기존
LLM planner를 import하지 않고 bridge 내부의 작은 개발용 fallback을 바로 사용한다.
지원 도형은 circle, square, triangle, letter A, house, star이다. 실제 LLM 실험은
`--planner-mode agentic` 또는 `--planner-mode template`을 사용한다.

```bash
cd /home/kimchangyeol/IsaacLab/IR
PYTHONPATH=/home/kimchangyeol/IsaacLab/IR/llm_isaac_bridge \
python -m llm_isaac_bridge \
  --command "중앙에 반지름 4cm짜리 원을 그려줘" \
  --planner-mode no-api \
  --print-isaac-command
```

이 명령은 Isaac을 실행하지 않는다. 대신 다음을 만든다.

```text
outputs/text_to_isaac/plans/*.json
outputs/text_to_isaac/reports/*_validation.json
```

마지막에 실제 Isaac 실행 명령을 출력한다.

## 2. 기존 plan JSON 검증하기

```bash
cd /home/kimchangyeol/IsaacLab/IR
PYTHONPATH=/home/kimchangyeol/IsaacLab/IR/llm_isaac_bridge \
python -m llm_isaac_bridge \
  --plan /home/kimchangyeol/IsaacLab/IR/Unit_Action_Langchain-main/examples/circle_plan.json \
  --print-isaac-command
```

## 3. 발표용 demo suite 준비하기

고정된 데모 입력 6개를 한 번에 plan 생성/검증하고, Isaac 실행 명령을 모은다.

```bash
cd /home/kimchangyeol/IsaacLab/IR
/home/kimchangyeol/IsaacLab/IR/llm_isaac_bridge/scripts/prepare_demo_showcase.sh
```

생성되는 주요 파일:

```text
outputs/demo_showcase/plans/*.json
outputs/demo_showcase/reports/*_validation.json
outputs/demo_showcase/manifest.json
outputs/demo_showcase/isaac_commands.sh
```

특정 case만 준비하려면:

```bash
/home/kimchangyeol/IsaacLab/IR/llm_isaac_bridge/scripts/prepare_demo_showcase.sh \
  --case circle_r4cm
```

## 4. Isaac까지 한 번에 실행하기

IsaacLab 루트가 `/home/kimchangyeol/IsaacLab`일 때:

```bash
cd /home/kimchangyeol/IsaacLab/IR
PYTHONPATH=/home/kimchangyeol/IsaacLab/IR/llm_isaac_bridge \
python -m llm_isaac_bridge \
  --command "중앙에 반지름 4cm짜리 원을 그려줘" \
  --planner-mode no-api \
  --mode contact \
  --execute-isaac
```

실제 LLM을 쓰려면 `OPENAI_API_KEY`가 설정된 환경에서 `--planner-mode agentic`
또는 `--planner-mode template`을 사용한다.

```bash
PYTHONPATH=/home/kimchangyeol/IsaacLab/IR/llm_isaac_bridge \
python -m llm_isaac_bridge \
  --command "중앙에 작은 사각형을 그려줘" \
  --planner-mode agentic \
  --model gpt-5-nano \
  --max-llm-steps 40 \
  --max-tool-calls 100 \
  --mode contact \
  --execute-isaac
```

## 5. Board 설정

`configs/isaac_planner_config.json`은 현재 Isaac USD의 종이 크기에 맞춘 LLM planner
설정이다.

```text
board_width_m: 0.30
board_height_m: 0.30
hover_height_m: 0.02
drawing_z_m: 0.0
default_circle_radius_m: 0.04
```

이 설정을 LLM planner에 넘기므로, LLM이 처음부터 현재 종이 크기 안에서 plan을
만들 가능성이 높아진다.

## 6. 검증 항목

bridge validator는 Isaac 실행 전에 다음을 확인한다.

- action list가 비어 있지 않은지
- 지원 action만 있는지
- 모든 action frame이 `board`인지
- `move_to_start -> pen_down -> draw -> pen_up` 순서가 맞는지
- 좌표가 현재 paper 범위 안에 있는지
- arc 시작점이 현재 pen 위치와 맞는지
- drawing 속도가 너무 빠르지 않은지
- 마지막에 pen이 올라간 상태인지

이 검증은 로봇 IK feasibility를 완전히 보장하지 않는다. 실제 Jacobian, joint limit,
contact force 평가는 기존 JADE Isaac runner가 실행 중 기록한다.
