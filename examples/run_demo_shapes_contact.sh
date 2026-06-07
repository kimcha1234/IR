#!/usr/bin/env bash
set -euo pipefail

IR_ROOT="/home/kimchangyeol/IsaacLab/IR"
ISAACLAB_SH="/home/kimchangyeol/IsaacLab/isaaclab.sh"
RUN_NAME="${1:-demo_shapes_260607_default}"
MODE="${MODE:-contact}"
PATH_SPEED_SCALE="${PATH_SPEED_SCALE:-1.0}"
DEBUG_DRAW="${DEBUG_DRAW:-0}"
RERUN="${RERUN:-0}"

OUT_ROOT="${IR_ROOT}/outputs/${RUN_NAME}"
LOG_ROOT="${OUT_ROOT}/terminal_logs"
mkdir -p "${LOG_ROOT}"

CASES=(
  "circle_r4cm"
  "square_s6cm"
  "triangle_s6cm"
  "letter_a_8cm"
  "house_8cm"
  "star_8cm"
)

echo "[INFO] IR root: ${IR_ROOT}"
echo "[INFO] output root: ${OUT_ROOT}"
echo "[INFO] mode: ${MODE}"
echo "[INFO] path speed scale: ${PATH_SPEED_SCALE}"
echo "[INFO] rerun completed cases: ${RERUN}"

FAILED_CASES=()

for CASE_ID in "${CASES[@]}"; do
  PLAN_PATH="${IR_ROOT}/outputs/demo_showcase/plans/${CASE_ID}.json"
  CASE_OUT="${OUT_ROOT}/${CASE_ID}"
  CASE_LOG="${LOG_ROOT}/${CASE_ID}.log"
  CASE_SUMMARY="${CASE_OUT}/summary.json"

  if [[ ! -f "${PLAN_PATH}" ]]; then
    echo "[ERROR] missing plan: ${PLAN_PATH}" >&2
    exit 1
  fi

  if [[ "${RERUN}" != "1" && -f "${CASE_SUMMARY}" ]]; then
    echo
    echo "[INFO] skipping ${CASE_ID}; existing summary found at ${CASE_SUMMARY}"
    continue
  fi

  CMD=(
    "TERM=xterm"
    "${ISAACLAB_SH}"
    "-p" "${IR_ROOT}/examples/run_jade_isaac.py"
    "--mode" "${MODE}"
    "--plan" "${PLAN_PATH}"
    "--jade" "${IR_ROOT}/configs/jade.yaml"
    "--out" "${CASE_OUT}"
    "--path-speed-scale" "${PATH_SPEED_SCALE}"
  )

  if [[ "${DEBUG_DRAW}" == "1" ]]; then
    CMD+=("--debug-draw")
  fi

  echo
  echo "[INFO] running ${CASE_ID}"
  echo "[INFO] plan: ${PLAN_PATH}"
  echo "[INFO] out: ${CASE_OUT}"
  echo "[INFO] log: ${CASE_LOG}"

  set +e
  env "${CMD[@]}" 2>&1 | tee "${CASE_LOG}"
  STATUS="${PIPESTATUS[0]}"
  set -e

  if [[ "${STATUS}" -ne 0 && ! -f "${CASE_SUMMARY}" ]]; then
    echo "[ERROR] ${CASE_ID} failed with status ${STATUS}, and no summary was written." >&2
    FAILED_CASES+=("${CASE_ID}")
  elif [[ "${STATUS}" -ne 0 ]]; then
    echo "[WARN] ${CASE_ID} returned status ${STATUS}, but summary exists; treating as completed."
  fi
done

echo
echo "[INFO] outputs: ${OUT_ROOT}"
if [[ "${#FAILED_CASES[@]}" -ne 0 ]]; then
  echo "[ERROR] failed cases: ${FAILED_CASES[*]}" >&2
  exit 1
fi
echo "[INFO] completed all demo cases."
