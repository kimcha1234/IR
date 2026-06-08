#!/usr/bin/env bash
set -euo pipefail

# circle_r4cm: 중앙에 반지름 4cm짜리 원을 그려줘
TERM=xterm /home/kimchangyeol/IsaacLab/isaaclab.sh -p /home/kimchangyeol/IsaacLab/IR/examples/run_jade_isaac.py --mode contact --plan /home/kimchangyeol/IsaacLab/IR/outputs/demo_showcase/plans/circle_r4cm.json --out /home/kimchangyeol/IsaacLab/IR/outputs/demo_showcase/isaac_runs/circle_r4cm

# square_s6cm: 중앙에 한 변 6cm짜리 사각형을 그려줘
TERM=xterm /home/kimchangyeol/IsaacLab/isaaclab.sh -p /home/kimchangyeol/IsaacLab/IR/examples/run_jade_isaac.py --mode contact --plan /home/kimchangyeol/IsaacLab/IR/outputs/demo_showcase/plans/square_s6cm.json --out /home/kimchangyeol/IsaacLab/IR/outputs/demo_showcase/isaac_runs/square_s6cm

# triangle_s6cm: 중앙에 한 변 6cm짜리 삼각형을 그려줘
TERM=xterm /home/kimchangyeol/IsaacLab/isaaclab.sh -p /home/kimchangyeol/IsaacLab/IR/examples/run_jade_isaac.py --mode contact --plan /home/kimchangyeol/IsaacLab/IR/outputs/demo_showcase/plans/triangle_s6cm.json --out /home/kimchangyeol/IsaacLab/IR/outputs/demo_showcase/isaac_runs/triangle_s6cm

# letter_a_8cm: 중앙에 알파벳 A를 8cm 크기로 그려줘
TERM=xterm /home/kimchangyeol/IsaacLab/isaaclab.sh -p /home/kimchangyeol/IsaacLab/IR/examples/run_jade_isaac.py --mode contact --plan /home/kimchangyeol/IsaacLab/IR/outputs/demo_showcase/plans/letter_a_8cm.json --out /home/kimchangyeol/IsaacLab/IR/outputs/demo_showcase/isaac_runs/letter_a_8cm

# house_8cm: 중앙에 8cm 집 모양을 그려줘
TERM=xterm /home/kimchangyeol/IsaacLab/isaaclab.sh -p /home/kimchangyeol/IsaacLab/IR/examples/run_jade_isaac.py --mode contact --plan /home/kimchangyeol/IsaacLab/IR/outputs/demo_showcase/plans/house_8cm.json --out /home/kimchangyeol/IsaacLab/IR/outputs/demo_showcase/isaac_runs/house_8cm

# star_8cm: 중앙에 8cm 별을 그려줘
TERM=xterm /home/kimchangyeol/IsaacLab/isaaclab.sh -p /home/kimchangyeol/IsaacLab/IR/examples/run_jade_isaac.py --mode contact --plan /home/kimchangyeol/IsaacLab/IR/outputs/demo_showcase/plans/star_8cm.json --out /home/kimchangyeol/IsaacLab/IR/outputs/demo_showcase/isaac_runs/star_8cm
