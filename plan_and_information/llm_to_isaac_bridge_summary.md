# LLM to Isaac Bridge 정리

이 문서는 기존 `Unit_Action_Langchain-main` LLM planner와 현재
`franka_llm_drawing` JADE/Isaac 제어 코드를 어떻게 연결할지 정리한다.

## 현재 LLM 코드가 구현한 범위

`Unit_Action_Langchain-main`은 자연어를 로봇 primitive action plan으로 바꾸는
상위 planner이다.

```text
natural language command
-> LLM / LangChain unit-action tool calls
-> DrawingPlan JSON
```

이미 구현된 기능은 다음이다.

- agentic LLM planner
- template baseline planner
- OpenAI 없이 돌리는 `--no-api` deterministic fallback
- `move_to_start`, `align_pen_orientation`, `pen_down`, `draw_line_to`,
  `draw_arc`, `pen_up`, `check_plan`, `finish_plan` unit-action tools
- board-frame boundary validation
- plan JSON 저장
- planned path plot 생성

중요한 점은 이 LLM 코드는 IK, FK, Jacobian, joint command, torque command,
Isaac Sim 실행을 하지 않는다는 것이다. 출력은 항상 symbolic `DrawingPlan`이다.

## 현재 제어 코드가 구현한 범위

`franka_llm_drawing`은 `DrawingPlan` JSON을 받아 JADE 실행 계층으로 넘긴다.

```text
DrawingPlan JSON
-> action parser
-> trajectory sampler
-> board-to-base frame transform
-> pen-tip offset compensation
-> Adaptive DLS executor
-> normal-force admittance
-> Isaac backend
-> evaluation plots/logs
```

Isaac backend는 반드시 `panda_joint1`부터 `panda_joint7`까지만 command 대상으로
사용한다. finger joint는 USD articulation 안에 남아 있어도 제어 대상이 아니다.

## 새로 추가한 bridge의 역할

새 폴더는 다음이다.

```text
llm_isaac_bridge/
```

이 폴더는 기존 LLM 코드와 기존 제어 코드를 수정하지 않고 연결한다.

```text
1. 자연어 command를 받는다.
2. 기존 LLM planner를 subprocess로 실행한다.
3. DrawingPlan JSON을 저장한다.
4. bridge validator로 Isaac 실행 전 안전 검사를 한다.
5. 통과하면 기존 run_jade_isaac.py 명령을 출력하거나 실행한다.
```

LLM planner를 subprocess로 실행하는 이유는 Isaac Python 환경에 LangChain/OpenAI
의존성을 섞지 않기 위해서다. 이렇게 하면 LLM 단계에서 실패하더라도 Isaac Sim을
무겁게 띄우지 않는다.

현재 테스트 환경에는 `langchain_core`가 설치되어 있지 않아 기존 LLM CLI의
`--no-api` 모드도 import 단계에서 실패한다. 기존 코드는 수정하지 않기 위해,
bridge 안에만 개발용 no-api fallback을 넣었다. 이 fallback은 circle, square,
triangle 같은 단순 도형을 만드는 테스트 용도이고, 실제 LLM 연결은 기존 planner의
`agentic` 또는 `template` 모드를 subprocess로 실행하는 방식이다.

## 왜 기존 코드를 수정하지 않았는가

현재 문서들의 공통 원칙은 다음이다.

```text
LLM output은 trajectory가 아니다.
LLM output은 symbolic Unit Action이다.
Trajectory, validation, correction, execution, evaluation은 JADE가 담당한다.
```

따라서 기존 LLM planner는 planner layer로 그대로 두고, 기존 JADE runner는 robot
execution layer로 그대로 둔다. 새 bridge는 두 layer 사이의 orchestration만 맡는다.

## 추가된 주요 파일

```text
llm_isaac_bridge/README.md
llm_isaac_bridge/configs/isaac_planner_config.json
llm_isaac_bridge/llm_isaac_bridge/planner.py
llm_isaac_bridge/llm_isaac_bridge/validation.py
llm_isaac_bridge/llm_isaac_bridge/isaac_command.py
llm_isaac_bridge/llm_isaac_bridge/runner.py
llm_isaac_bridge/scripts/run_text_to_isaac.sh
llm_isaac_bridge/tests/test_validation.py
```

## 실행 방식

OpenAI 없이 plan 생성/검증/Isaac 명령 출력:

```bash
cd /home/kimchangyeol/IsaacLab/IR
PYTHONPATH=/home/kimchangyeol/IsaacLab/IR/llm_isaac_bridge \
python -m llm_isaac_bridge \
  --command "중앙에 반지름 4cm짜리 원을 그려줘" \
  --planner-mode no-api \
  --print-isaac-command
```

Isaac까지 실행:

```bash
cd /home/kimchangyeol/IsaacLab/IR
PYTHONPATH=/home/kimchangyeol/IsaacLab/IR/llm_isaac_bridge \
python -m llm_isaac_bridge \
  --command "중앙에 반지름 4cm짜리 원을 그려줘" \
  --planner-mode no-api \
  --mode contact \
  --execute-isaac
```

실제 LLM agentic planner를 쓰려면 `OPENAI_API_KEY`를 설정한 뒤
`--planner-mode agentic`을 사용한다.

## 검증 항목

bridge validator는 다음을 검사한다.

- action list가 비어 있지 않은지
- 지원 action만 포함하는지
- 모든 action frame이 `board`인지
- pen state 순서가 맞는지
- 좌표가 현재 `0.30 m x 0.30 m` paper 안에 있는지
- line/arc 시작점이 현재 pen 위치와 이어지는지
- drawing speed가 너무 빠르지 않은지
- 마지막에 pen이 올라간 상태인지

이 검증은 사전 형식/기하 안전 검사이다. Jacobian, joint limit, manipulability,
contact force, planned-vs-actual 평가는 기존 JADE Isaac 실행 결과에서 확인한다.

## 현재 전체 프로젝트 안에서의 위치

현재 프로젝트는 다음 단계까지 연결 가능하다.

```text
사람 텍스트
-> LLM planner
-> DrawingPlan JSON
-> bridge validator
-> JADE trajectory/control
-> Isaac Sim
-> plots/logs/evaluation
```

따라서 다음 개발 단계는 controller를 원 하나에만 더 튜닝하는 것보다, 여러 자연어
명령과 여러 도형 plan을 bridge로 통과시켜 전체 pipeline이 일반화되는지 확인하는
것이다.
