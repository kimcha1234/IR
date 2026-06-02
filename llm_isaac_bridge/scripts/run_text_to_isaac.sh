#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 \"drawing command\" [extra bridge args...]" >&2
  exit 2
fi

COMMAND="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHONPATH="${BRIDGE_ROOT}:${PYTHONPATH:-}" python -m llm_isaac_bridge \
  --command "${COMMAND}" \
  --planner-mode no-api \
  --mode contact \
  --print-isaac-command \
  "$@"
