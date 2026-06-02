#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHONPATH="${BRIDGE_ROOT}:${PYTHONPATH:-}" python -m llm_isaac_bridge.demo_suite "$@"
