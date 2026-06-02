"""CLI for plotting planned DrawingPlan JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

from robot_drawing_planner.visualization import load_drawing_plan, save_plan_plot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot planned board-frame primitive drawing paths from DrawingPlan JSON. "
            "This visualizes planned paths only, not robot execution."
        )
    )
    parser.add_argument("plan_json", help="Path to a DrawingPlan JSON file.")
    parser.add_argument("--out", help="Output image path. Defaults to input path with .png suffix.")
    parser.add_argument("--title", help="Optional plot title.")
    parser.add_argument(
        "--no-board",
        action="store_true",
        help="Hide the board boundary rectangle.",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Hide stroke/action labels.",
    )
    parser.add_argument(
        "--show-pen-up",
        action="store_true",
        help="Show dashed free-space pen-up moves reconstructed from actions.",
    )
    parser.add_argument(
        "--arc-samples",
        type=int,
        default=96,
        help="Number of visualization samples for arcs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Output image DPI.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan_path = Path(args.plan_json)
        if not plan_path.exists():
            raise FileNotFoundError(f"Plan JSON file not found: {plan_path}")
        out_path = Path(args.out) if args.out else plan_path.with_suffix(".png")
        plan = load_drawing_plan(plan_path)
        result = save_plan_plot(
            plan,
            out_path,
            title=args.title,
            show_board=not args.no_board,
            show_action_labels=not args.no_labels,
            show_pen_up_moves=args.show_pen_up,
            arc_samples=args.arc_samples,
            dpi=args.dpi,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Saved plan plot to {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
